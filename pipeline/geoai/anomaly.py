"""Label-free detection of road surface that does not look like normal pavement.

Two signals:

1. DINOv2 patch novelty (in the spirit of PatchCore / AnomalyDINO, training-free). Every
   processed frame shows the same image window, so each patch row always sees the same strip of
   road at the same scale. A bank of "normal" patch features per row is built from frames spread
   over the session itself. A query patch is anomalous when its nearest bank neighbours (same
   rows, other places on the route) are far away. Things that recur along the route (lane
   markings, the usual asphalt texture, the vehicle's own shadow) end up in the bank and are
   normal; a pothole or a patch is not.
   - Bank patches that look like cracks (absolute detector) are kept out, so a heavily cracked
     road can't teach the bank that cracks are normal.
   - The most unusual few per cent of bank patches are trimmed as likely distress.
   - Bank patches from within `exclude_window_m` of the query are ignored, so a distress that
     happens to be in a bank frame can't match itself.
2. Absolute crack-likeness (geoai.cracks).

Scores are robust z-scores per patch row against the bank's own leave-one-place-out distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold

from .blur import gaussian_blur, set_device
from .config import AnomalyConfig, CrackConfig
from .cracks import CrackDetector
from .dinov2 import PATCH
from .probe import DistressProbe

_INF = 1e4


def _knn_mean(Q: torch.Tensor, B: torch.Tensor, k: int, exclude: torch.Tensor | None) -> torch.Tensor:
    """Mean of the k smallest cosine distances from each row of Q to B (rows L2-normalised).

    exclude: bool (len(Q), len(B)) or (1, len(B)); True entries are ignored.
    """
    out = []
    for i in range(0, Q.shape[0], 4096):
        q = Q[i:i + 4096]
        d = 1.0 - q @ B.T
        if exclude is not None:
            ex = exclude if exclude.shape[0] == 1 else exclude[i:i + 4096]
            d = d.masked_fill(ex, _INF)
        kk = min(k, d.shape[1])
        out.append(torch.topk(d, kk, dim=1, largest=False).values.mean(1))
    return torch.cat(out) if out else torch.zeros(0, device=Q.device)


def _kcenter(X: torch.Tensor, m: int, seed: int) -> torch.Tensor:
    """Greedy k-center coreset indices (PatchCore's selection), deterministic given the seed."""
    n = X.shape[0]
    if n <= m:
        return torch.arange(n, device=X.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=g))
    idx = torch.empty(m, dtype=torch.long, device=X.device)
    idx[0] = first
    mind = 1.0 - X @ X[first]
    for i in range(1, m):
        j = int(torch.argmax(mind))
        idx[i] = j
        mind = torch.minimum(mind, 1.0 - X @ X[j])
    return idx


@dataclass
class _RowBank:
    vecs: torch.Tensor      # (m, D)
    frame: torch.Tensor     # (m,) frame id
    s: torch.Tensor         # (m,) path position of the frame (m)
    med: float = 0.0        # median leave-place-out distance (normal road)
    hi: float = 1.0         # null_quantile of the same distances


class PatchBank:
    def __init__(self, grid_h: int, grid_w: int, cfg: AnomalyConfig, device: torch.device, seed: int = 0):
        self.gh, self.gw = grid_h, grid_w
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self._v: list[list[np.ndarray]] = [[] for _ in range(grid_h)]
        self._f: list[list[np.ndarray]] = [[] for _ in range(grid_h)]
        self._s: list[list[np.ndarray]] = [[] for _ in range(grid_h)]
        self.n_frames = 0
        self.rows: list[_RowBank] = []
        self.stats: dict = {}

    def add(self, feats: np.ndarray, s_m: float, keep: np.ndarray | None = None) -> None:
        assert feats.shape[:2] == (self.gh, self.gw), (feats.shape, self.gh, self.gw)
        fid = self.n_frames
        self.n_frames += 1
        for r in range(self.gh):
            sel = slice(None) if keep is None else keep[r]
            v = feats[r][sel]
            if len(v):
                self._v[r].append(v.astype(np.float32))
                self._f[r].append(np.full(len(v), fid, np.int64))
                self._s[r].append(np.full(len(v), s_m, np.float64))

    def _neighbours(self, r: int):
        rows = [i for i in (r - 1, r, r + 1) if 0 <= i < self.gh]
        return (torch.cat([self.rows[i].vecs for i in rows]), torch.cat([self.rows[i].frame for i in rows]),
                torch.cat([self.rows[i].s for i in rows]))

    def finalize(self, max_collect: int = 16384) -> None:
        rng = np.random.default_rng(self.seed)
        dev = self.device
        raw = []
        for r in range(self.gh):
            if not self._v[r]:
                raise RuntimeError(f"bank row {r} is empty")
            V = np.concatenate(self._v[r])
            F = np.concatenate(self._f[r])
            S = np.concatenate(self._s[r])
            if len(V) > max_collect:
                sel = np.sort(rng.choice(len(V), max_collect, replace=False))
                V, F, S = V[sel], F[sel], S[sel]
            raw.append((torch.from_numpy(V).to(dev), torch.from_numpy(F).to(dev), torch.from_numpy(S).to(dev)))
        self._v = self._f = self._s = None  # free

        # Trim likely distress: each patch's k-NN distance to other frames' patches (rows r-1..r+1).
        trimmed = 0
        rows_after_trim = []
        for r in range(self.gh):
            idx = [i for i in (r - 1, r, r + 1) if 0 <= i < self.gh]
            RV = torch.cat([raw[i][0] for i in idx])
            RF = torch.cat([raw[i][1] for i in idx])
            V, F, S = raw[r]
            d = _knn_mean(V, RV, self.cfg.knn_k, F[:, None] == RF[None, :])
            n_keep = max(1, int(round(len(V) * (1.0 - self.cfg.trim_frac))))
            order = torch.argsort(d)[:n_keep]
            trimmed += len(V) - n_keep
            rows_after_trim.append((V[order], F[order], S[order]))
        # Coreset per row.
        self.rows = []
        for r, (V, F, S) in enumerate(rows_after_trim):
            ci = _kcenter(V, self.cfg.bank_size_per_row, self.seed + r)
            self.rows.append(_RowBank(V[ci], F[ci], S[ci]))
        # Null distribution per row: leave-place-out distances of the bank patches (all of them,
        # before trimming, up to 10k), scored exactly like queries against the final bank.
        null = []
        for r, (V, F, S) in enumerate(raw):
            if len(V) > 10000:
                sel = torch.from_numpy(np.sort(rng.choice(len(V), 10000, replace=False))).to(dev)
                V, F, S = V[sel], F[sel], S[sel]
            BV, BF, BS = self._neighbours(r)
            ex = (F[:, None] == BF[None, :]) | ((S[:, None] - BS[None, :]).abs() < self.cfg.exclude_window_m)
            null.append(_knn_mean(V, BV, self.cfg.knn_k, ex).double().cpu().numpy())
        for r in range(self.gh):
            med, q90 = (float(v) for v in np.quantile(null[r], [0.5, 0.9]))
            hi = med + self.cfg.tail_factor * max(q90 - med, 1e-5)
            self.rows[r].med, self.rows[r].hi = med, hi
        self.stats = {"frames": self.n_frames, "trimmed_patches": trimmed,
                      "bank_patches": int(sum(len(b.vecs) for b in self.rows)),
                      "null_samples_per_row": int(np.median([len(n) for n in null])),
                      "row_median": [b.med for b in self.rows], "row_hi": [b.hi for b in self.rows]}

    def score(self, feats: np.ndarray, s_m: float) -> tuple[np.ndarray, np.ndarray]:
        """(distance, u) maps of shape (gh, gw). u = (d - median) / (null quantile - median)."""
        dist = np.zeros((self.gh, self.gw), np.float32)
        u = np.zeros((self.gh, self.gw), np.float32)
        q = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).to(self.device)
        for r in range(self.gh):
            BV, _, BS = self._neighbours(r)
            ex = ((BS - s_m).abs() < self.cfg.exclude_window_m)[None, :]
            d = _knn_mean(q[r], BV, self.cfg.knn_k, ex).cpu().numpy()
            dist[r] = d
            u[r] = (d - self.rows[r].med) / (self.rows[r].hi - self.rows[r].med)
        return dist, u


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]       # analysis-image pixels x0, y0, x1, y1 (exclusive)
    score: float                          # evidence used for ranking (see AnomalyDetector.detect)
    z_dino: float                         # peak patch score u
    z_crack: float                        # peak crack z
    kinds: list[str] = field(default_factory=list)
    patches: int = 0
    crack_length_mm: float = 0.0
    mass: float = 0.0                     # sum over patches of (u - grow level), DINOv2 evidence
    probe_max: float | None = None        # peak probe logit, if the probe flagged it


def _merge_boxes(cands: list[Candidate], gap: int) -> list[Candidate]:
    """Union candidates whose boxes overlap or lie within `gap` pixels of each other."""
    cands = list(cands)
    changed = True
    while changed:
        changed = False
        out: list[Candidate] = []
        for c in cands:
            for o in out:
                if (c.bbox[0] <= o.bbox[2] + gap and o.bbox[0] <= c.bbox[2] + gap and
                        c.bbox[1] <= o.bbox[3] + gap and o.bbox[1] <= c.bbox[3] + gap):
                    o.bbox = (min(o.bbox[0], c.bbox[0]), min(o.bbox[1], c.bbox[1]),
                              max(o.bbox[2], c.bbox[2]), max(o.bbox[3], c.bbox[3]))
                    o.score += c.score  # evidence accumulates (adjacent parts, or both signals)
                    o.mass += c.mass
                    o.z_dino = max(o.z_dino, c.z_dino)
                    o.z_crack = max(o.z_crack, c.z_crack)
                    o.kinds = sorted(set(o.kinds) | set(c.kinds))
                    o.patches += c.patches
                    o.crack_length_mm = max(o.crack_length_mm, c.crack_length_mm)
                    if c.probe_max is not None:
                        o.probe_max = c.probe_max if o.probe_max is None else max(o.probe_max, c.probe_max)
                    changed = True
                    break
            else:
                out.append(c)
        cands = out
    return cands


def normalise_illumination(rgb: np.ndarray, sigma: float | None) -> np.ndarray:
    """Divide by the local mean brightness (keeps colour ratios); removes most of a shadow."""
    if not sigma:
        return rgb
    f = rgb.astype(np.float32)
    lum = f.mean(-1)
    local = gaussian_blur(lum, sigma)
    target = float(np.median(lum))
    return np.clip(f * (target / np.maximum(local, 1.0))[..., None], 0, 255).astype(np.uint8)


def paint_fraction(rgb: np.ndarray, grid_h: int, grid_w: int, ratio: float, sigma: float = 40.0) -> np.ndarray:
    """Share of each patch covered by road paint: much brighter than the local road surface.

    Local road brightness is a median-like estimate (blur of a brightness image with paint
    clipped), so a marking doesn't raise its own reference.
    """
    lum = rgb.astype(np.float32).mean(-1)
    ref0 = gaussian_blur(lum, sigma)
    clipped = np.minimum(lum, ref0 * 1.15)
    ref = gaussian_blur(clipped, sigma)
    paint = (lum > ratio * ref).astype(np.uint8)
    # Grow by half a patch: patches that only touch a marking's edge see it too.
    paint = cv2.dilate(paint, np.ones((PATCH, PATCH), np.uint8)).astype(np.float32)
    h, w = grid_h * PATCH, grid_w * PATCH
    return paint[:h, :w].reshape(grid_h, PATCH, grid_w, PATCH).mean(axis=(1, 3))


class AnomalyDetector:
    """Bank building (pass 1) and per-frame detection (pass 2) on fixed-size analysis images."""

    CRACK_STATS_FRAMES = 20  # bank frames used for the crack detector's background statistics

    def __init__(self, features, image_h: int, image_w: int, cfg: AnomalyConfig, crack_cfg: CrackConfig, seed: int = 0):
        self.features = features
        self.cfg = cfg
        self.crack_cfg = crack_cfg
        self.gh, self.gw = image_h // PATCH, image_w // PATCH
        self.bank = PatchBank(self.gh, self.gw, cfg, features.device, seed)
        set_device(features.device)
        self.cracks = CrackDetector(crack_cfg.sigmas_px, crack_cfg.z_threshold, crack_cfg.min_length_mm) \
            if crack_cfg.enabled else None
        self._pending: list[tuple[np.ndarray, float, np.ndarray]] = []
        self._crack_ready = False
        self._crack_excluded = 0
        self._probe_excluded = 0
        self._rng = np.random.default_rng(seed)
        self.probe = None
        if cfg.probe and Path(cfg.probe).exists():
            self.probe = DistressProbe.load(Path(cfg.probe))
            if tuple(self.probe.layers) != tuple(cfg.layers) or self.probe.illum_sigma_px != cfg.illum_sigma_px:
                raise ValueError(f"probe {cfg.probe} was trained on layers {self.probe.layers}, illumination sigma "
                                 f"{self.probe.illum_sigma_px}; config has {cfg.layers}, {cfg.illum_sigma_px}")

    def _probe_keep(self, feats: np.ndarray) -> np.ndarray | None:
        if self.probe is None or not self.cfg.probe_bank_exclude:
            return None
        keep = self.probe.logits(feats) < self.probe.seed_logit
        self._probe_excluded += int((~keep).sum())
        return keep

    def _add_with_mask(self, feats: np.ndarray, s_m: float, rgb: np.ndarray, gsd_rows_mm: np.ndarray) -> None:
        keep = self._probe_keep(feats)
        if self.crack_cfg.bank_mask_frac > 0:
            mask, _ = self.cracks.detect(rgb, gsd_rows_mm)
            h, w = self.gh * PATCH, self.gw * PATCH
            frac = mask[:h, :w].reshape(self.gh, PATCH, self.gw, PATCH).mean(axis=(1, 3))
            ck = frac <= self.crack_cfg.bank_mask_frac
            self._crack_excluded += int((~ck).sum())
            keep = ck if keep is None else (keep & ck)
        self.bank.add(feats, s_m, keep)

    def _flush(self, gsd_rows_mm: np.ndarray) -> None:
        self.cracks.finalize()
        self._crack_ready = True
        for feats, s_m, rgb in self._pending:
            self._add_with_mask(feats, s_m, rgb, gsd_rows_mm)
        self._pending = []

    def _features(self, rgb: np.ndarray) -> np.ndarray:
        return self.features(normalise_illumination(rgb, self.cfg.illum_sigma_px))

    def add_bank_frame(self, rgb: np.ndarray, s_m: float, gsd_rows_mm: np.ndarray) -> None:
        feats = self._features(rgb)
        self._gsd = gsd_rows_mm
        if self.cracks is None:
            self.bank.add(feats, s_m, self._probe_keep(feats))
        elif not self._crack_ready:
            # Crack statistics come from the first bank frames; their masks are applied once known.
            self.cracks.add_background(rgb, self._rng)
            self._pending.append((feats, s_m, rgb))
            if len(self._pending) >= self.CRACK_STATS_FRAMES:
                self._flush(gsd_rows_mm)
        else:
            self._add_with_mask(feats, s_m, rgb, gsd_rows_mm)

    def finalize(self) -> None:
        if self.bank.n_frames + len(self._pending) < 2:
            raise RuntimeError("need at least 2 bank frames")
        if self.cracks is not None and not self._crack_ready:
            self._flush(self._gsd)
        self.bank.finalize()
        self.bank.stats["crack_excluded_patches"] = self._crack_excluded
        self.bank.stats["probe_excluded_patches"] = self._probe_excluded

    def detect(self, rgb: np.ndarray, s_m: float, gsd_rows_mm: np.ndarray) -> tuple[list[Candidate], dict]:
        feats = self._features(rgb)
        _, z = self.bank.score(feats, s_m)
        paint = None
        if self.cfg.mask_paint:
            paint = paint_fraction(rgb, self.gh, self.gw, self.cfg.paint_ratio) > self.cfg.paint_patch_frac
            z = np.where(paint, 0.0, z)
        probe_cands, pdiag = self._probe_candidates(feats, paint)
        thr = self.cfg.score_threshold
        seeds = apply_hysteresis_threshold(z, self.cfg.grow_fraction * thr, thr)
        labels, n = ndimage.label(seeds, structure=np.ones((3, 3), int))
        cands: list[Candidate] = []
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            comp = labels[sl] == i
            npatch = int(comp.sum())
            zmax = float(z[sl][comp].max())
            grow = self.cfg.grow_fraction * thr
            mass = float(np.clip(z[sl][comp] - grow, 0, None).sum())
            if mass < self.cfg.min_mass:
                continue
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            cands.append(Candidate((c0 * PATCH, r0 * PATCH, c1 * PATCH, r1 * PATCH), mass, zmax, 0.0,
                                   ["dino"], npatch, mass=mass))
        crack_mask = None
        if self.cracks is not None:
            crack_mask, comps = self.cracks.detect(rgb, gsd_rows_mm)
            if self.crack_cfg.as_candidates:
                for c in comps:
                    # Crack evidence on the same scale as DINOv2 mass: a crack just at the length and
                    # z thresholds scores min_mass.
                    ev = self.cfg.min_mass * (c.z_max / self.crack_cfg.z_threshold) * (c.length_mm / self.crack_cfg.min_length_mm)
                    cands.append(Candidate(c.bbox, ev, 0.0, c.z_max, ["crack"], 0, c.length_mm))
        cands = _merge_boxes(cands + probe_cands, gap=PATCH)
        diag = {"z_max": float(z.max()), "z_p99": float(np.percentile(z, 99)),
                "anomalous_patch_frac": float((z >= thr).mean()),
                "crack_pixel_frac": float(crack_mask.mean()) if crack_mask is not None else 0.0, **pdiag}
        return cands, diag

    def _probe_candidates(self, feats: np.ndarray, paint: np.ndarray | None) -> tuple[list[Candidate], dict]:
        """Regions the probe calls distress; evidence = sum of (logit - grow level) over the region."""
        if self.probe is None:
            return [], {}
        logit = self.probe.logits(feats)
        if paint is not None:
            logit = np.where(paint, -1e3, logit)
        seed = self.probe.seed_logit
        grow = seed - self.cfg.probe_grow_logit
        out: list[Candidate] = []
        if self.cfg.probe_candidates:
            labels, _ = ndimage.label(apply_hysteresis_threshold(logit, grow, seed), structure=np.ones((3, 3), int))
            for i, sl in enumerate(ndimage.find_objects(labels), start=1):
                if sl is None:
                    continue
                comp = labels[sl] == i
                mass = float(np.clip(logit[sl][comp] - grow, 0, None).sum())
                if mass < self.cfg.min_mass:
                    continue
                r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
                out.append(Candidate((c0 * PATCH, r0 * PATCH, c1 * PATCH, r1 * PATCH), mass, 0.0, 0.0, ["probe"],
                                     int(comp.sum()), mass=0.0, probe_max=float(logit[sl][comp].max())))
        return out, {"probe_max_logit": float(logit.max()), "probe_patch_frac": float((logit >= seed).mean())}
