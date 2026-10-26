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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

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
# Probe logit given to paint patches: far below any seed or grow level, so paint never starts or
# joins a probe anomaly (road markings are not pavement distress).
PAINT_LOGIT = -1e3


_CHUNK = 4096  # queries per distance block (4096 x 49k fp32 = 0.8 GB for a 16k-patch row's neighbours)


def _knn_mean(Q: torch.Tensor, B: torch.Tensor, k: int,
              exclude: torch.Tensor | Callable[[int, int], torch.Tensor] | None) -> torch.Tensor:
    """Mean of the k smallest cosine distances from each row of Q to B (rows L2-normalised).

    exclude: True entries are ignored. A (1, len(B)) bool tensor applies to every query; a function
    (i0, i1) -> bool (i1 - i0, len(B)) gives the mask of queries i0..i1-1, so it is built one block
    at a time (`_exclusion`): a whole (len(Q), len(B)) mask of a 16k-patch row against its three
    neighbour rows would be 0.8 GB of GPU memory.
    """
    out = []
    for i in range(0, Q.shape[0], _CHUNK):
        q = Q[i:i + _CHUNK]
        d = 1.0 - q @ B.T
        if exclude is not None:
            d.masked_fill_(exclude(i, i + q.shape[0]) if callable(exclude) else exclude, _INF)
        kk = min(k, d.shape[1])
        out.append(torch.topk(d, kk, dim=1, largest=False).values.mean(1))
    return torch.cat(out) if out else torch.zeros(0, device=Q.device)


def _exclusion(q_frame: torch.Tensor, b_frame: torch.Tensor, q_s: torch.Tensor | None = None,
               b_s: torch.Tensor | None = None, window_m: float = 0.0) -> Callable[[int, int], torch.Tensor]:
    """Block-wise leave-place-out mask for `_knn_mean`: bank patches of the query's own frame, and
    (with positions) those within window_m of the query's position on the route."""
    def mask(i0: int, i1: int) -> torch.Tensor:
        m = q_frame[i0:i1, None] == b_frame[None, :]
        if q_s is not None:
            m |= (q_s[i0:i1, None] - b_s[None, :]).abs() < window_m
        return m
    return mask


def _kcenter(X: torch.Tensor, m: int, seed: int) -> torch.Tensor:
    """Greedy k-center coreset indices (PatchCore's selection), deterministic given the seed.

    Each step is one matrix-vector product over all of X, so the loop is bound by memory bandwidth
    (0.25 ms per step for 15k x 768 fp32 on the RTX 4050 laptop GPU, 187 GB/s). The chosen index
    stays on the GPU: reading it on the host each step stalled the queue and made the loop 1.5x
    slower. The arithmetic is the same matrix-vector product as before, so the choice is identical.
    """
    n = X.shape[0]
    if n <= m:
        return torch.arange(n, device=X.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=g))
    idx = torch.empty(m, dtype=torch.long, device=X.device)
    idx[0] = first
    mind = 1.0 - X @ X[first]
    for i in range(1, m):
        j = torch.argmax(mind, 0, keepdim=True)
        idx[i:i + 1] = j
        mind = torch.minimum(mind, 1.0 - X @ X.index_select(0, j)[0])
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

    def finalize(self, max_collect: int = 16384, null_samples: int = 10000) -> None:
        """Trim, coreset and null distribution of every row.

        Memory: the collected patches stay in host memory. The GPU holds only the three rows the
        current row's neighbour search reads, one block of distances and the finished coresets.
        Holding every row there, as the first version did, peaked at 7.3 GB for the 51-row bank of
        session f428 on a 6 GB laptop GPU: the driver spilled to system memory and finalising took
        231 s. The random draws and the arithmetic of every step are unchanged, so the bank is the
        same (tests/test_gate.py compares it with the first version's).
        """
        rng = np.random.default_rng(self.seed)
        dev = self.device
        raw: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
        for r in range(self.gh):
            if not self._v[r]:
                raise RuntimeError(f"bank row {r} is empty")
            V = np.concatenate(self._v[r])
            F = np.concatenate(self._f[r])
            S = np.concatenate(self._s[r])
            self._v[r] = self._f[r] = self._s[r] = None  # per-frame pieces are copies: free them now
            if len(V) > max_collect:
                sel = np.sort(rng.choice(len(V), max_collect, replace=False))
                V, F, S = V[sel], F[sel], S[sel]
            raw.append((V, F, S))
        self._v = self._f = self._s = None
        # The null samples are drawn now, in the row order the first version drew them after the
        # coresets (trimming and coresets draw nothing from rng), so the same patches are sampled.
        null_sel = [np.sort(rng.choice(len(V), null_samples, replace=False)) if len(V) > null_samples else None
                    for V, _, _ in raw]

        # Per row: trim likely distress (each patch's k-NN distance to other frames' patches in rows
        # r-1..r+1; the most distant trim_frac go), then the k-center coreset of what is left.
        trimmed = 0
        self.rows = []
        window: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}  # rows r-1..r+1 on the GPU
        for r in range(self.gh):
            idx = [i for i in (r - 1, r, r + 1) if 0 <= i < self.gh]
            for i in [i for i in window if i < idx[0]]:
                del window[i]
            for i in idx:
                if i not in window:
                    window[i] = tuple(torch.from_numpy(a).to(dev) for a in raw[i])
            RV = torch.cat([window[i][0] for i in idx])
            RF = torch.cat([window[i][1] for i in idx])
            V, F, S = window[r]
            d = _knn_mean(V, RV, self.cfg.knn_k, _exclusion(F, RF))
            n_keep = max(1, int(round(len(V) * (1.0 - self.cfg.trim_frac))))
            order = torch.argsort(d)[:n_keep]
            trimmed += len(V) - n_keep
            V, F, S = V[order], F[order], S[order]
            ci = _kcenter(V, self.cfg.bank_size_per_row, self.seed + r)
            self.rows.append(_RowBank(V[ci], F[ci], S[ci]))
        del window, RV, RF  # free the GPU for the null step
        # Null distribution per row: leave-place-out distances of the bank patches (all of them,
        # before trimming, up to null_samples), scored exactly like queries against the final bank.
        null = []
        for r in range(self.gh):
            V, F, S = raw[r]
            raw[r] = None
            if null_sel[r] is not None:
                V, F, S = V[null_sel[r]], F[null_sel[r]], S[null_sel[r]]
            V, F, S = (torch.from_numpy(a).to(dev) for a in (V, F, S))
            BV, BF, BS = self._neighbours(r)
            ex = _exclusion(F, BF, S, BS, self.cfg.exclude_window_m)
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
        """Crop mode: the merged candidates of one analysis image and its statistics (see `maps`)."""
        m = self.maps(rgb, s_m, gsd_rows_mm)
        return m["candidates"], m["diag"]

    def maps(self, rgb: np.ndarray, s_m: float, gsd_rows_mm: np.ndarray) -> dict:
        """Per-patch maps of one analysis image, and the candidates `detect` reports for it.

        The crop mode ranks whole-image candidates; the segment gate (geoai.gate) scores only the
        patches and the detections of one road segment. Both read this one computation, so the two
        modes can't drift apart.

        Returns a dict:
          z            (gh, gw) float32 bank score u, 0 on paint patches (with mask_paint)
          probe_logit  (gh, gw) probe logits, PAINT_LOGIT on paint patches (exactly the map the probe
                       candidates are grown on); None without a probe
          paint        (gh, gw) bool, patches mostly covered by road paint; all False without mask_paint
          crack_mask   (H, W) bool analysis-pixel crack mask; None with the crack filter off
          components   {"dino": [...], "crack": [...], "probe": [...]}: each detector's candidates
                       before merging, as independent copies. The merged candidates sum the evidence
                       of all three, so a per-detector total needs these.
          candidates   merged list, exactly what `detect` returns
          diag         per-image statistics, exactly what `detect` returns
        """
        feats = self._features(rgb)
        _, z = self.bank.score(feats, s_m)
        paint = None
        if self.cfg.mask_paint:
            paint = paint_fraction(rgb, self.gh, self.gw, self.cfg.paint_ratio) > self.cfg.paint_patch_frac
            z = np.where(paint, 0.0, z)
        logit = None
        if self.probe is not None:
            logit = self.probe.logits(feats)
            if paint is not None:
                logit = np.where(paint, PAINT_LOGIT, logit)
        probe_cands, pdiag = self._probe_candidates(logit)
        thr = self.cfg.score_threshold
        seeds = apply_hysteresis_threshold(z, self.cfg.grow_fraction * thr, thr)
        labels, n = ndimage.label(seeds, structure=np.ones((3, 3), int))
        dino: list[Candidate] = []
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
            dino.append(Candidate((c0 * PATCH, r0 * PATCH, c1 * PATCH, r1 * PATCH), mass, zmax, 0.0,
                                  ["dino"], npatch, mass=mass))
        crack_mask = None
        crack: list[Candidate] = []
        if self.cracks is not None:
            crack_mask, comps = self.cracks.detect(rgb, gsd_rows_mm)
            if self.crack_cfg.as_candidates:
                for c in comps:
                    # Crack evidence on the same scale as DINOv2 mass: a crack just at the length and
                    # z thresholds scores min_mass.
                    ev = self.cfg.min_mass * (c.z_max / self.crack_cfg.z_threshold) * (c.length_mm / self.crack_cfg.min_length_mm)
                    crack.append(Candidate(c.bbox, ev, 0.0, c.z_max, ["crack"], 0, c.length_mm))
        # _merge_boxes grows the first box of each cluster in place: merge copies, keep the originals.
        cands = _merge_boxes([replace(c, kinds=list(c.kinds)) for c in dino + crack + probe_cands], gap=PATCH)
        diag = {"z_max": float(z.max()), "z_p99": float(np.percentile(z, 99)),
                "anomalous_patch_frac": float((z >= thr).mean()),
                "crack_pixel_frac": float(crack_mask.mean()) if crack_mask is not None else 0.0, **pdiag}
        return {"z": z, "probe_logit": logit,
                "paint": paint if paint is not None else np.zeros((self.gh, self.gw), bool),
                "crack_mask": crack_mask, "components": {"dino": dino, "crack": crack, "probe": probe_cands},
                "candidates": cands, "diag": diag}

    def _probe_candidates(self, logit: np.ndarray | None) -> tuple[list[Candidate], dict]:
        """Regions the probe calls distress; evidence = sum of (logit - grow level) over the region.

        logit: the paint-masked probe map from `maps`, None without a probe.
        """
        if logit is None:
            return [], {}
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
