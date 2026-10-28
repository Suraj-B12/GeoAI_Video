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

Speed and exactness of the bank (tests/test_gate.py). Building the bank is two discrete choices
(which patches the trim keeps, which ones the k-center coreset picks) made on fp32 distances. A
change in rounding, from another matrix-product kernel, block size or reduction order, can flip a
near tie and then the coreset diverges, which is not a float-tolerance change. So every distance is
computed by the same kernels on the same shapes as in the first version, and only what surrounds
them changed: GPU memory held per row instead of for the whole bank, no host round trip per
coreset step, no second distance matrix per block, the bank stored contiguously so a row's
neighbour rows are a view instead of a copy, one device-to-host copy per scored image. The test
compares bank, coreset and scores with a verbatim copy of the first version, bit for bit.
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

from .blur import blur_tensor, device_for, gaussian_blur, set_device
from .config import AnomalyConfig, CrackConfig
from .cracks import CrackDetector
from .dinov2 import PATCH
from .probe import DistressProbe

_INF = 1e4
# Probe logit given to paint patches: far below any seed or grow level, so paint never starts or
# joins a probe anomaly (road markings are not pavement distress).
PAINT_LOGIT = -1e3
# Queries per distance block. Also part of the arithmetic: the block height is a matrix-product shape,
# and a different shape may pick a different kernel and rounding (module docstring), so it stays 4096.
_CHUNK = 4096


def _knn_mean(Q: torch.Tensor, B: torch.Tensor, k: int,
              exclude: torch.Tensor | Callable[[int, int], torch.Tensor] | None) -> torch.Tensor:
    """Mean of the k smallest cosine distances from each row of Q to B (rows L2-normalised).

    exclude: True entries are ignored (distance _INF). A (1, len(B)) bool tensor applies to every
    query; a function (i0, i1) -> bool (i1 - i0, len(B)) gives the mask of queries i0..i1-1
    (`_exclusion`), so the mask of a 16k-patch bank row against its three neighbour rows (0.8 GB)
    is never held whole.

    The k smallest distances are taken as the k largest similarities s and converted after the
    selection: fl(1 - s) is monotone in s, so they are the same values in the same order as the
    smallest of the distance matrix 1 - QB', bit for bit, and the (block x len(B)) distance matrix
    is never written. An excluded entry gets similarity 1 - _INF, i.e. distance exactly _INF.
    """
    out = []
    for i in range(0, Q.shape[0], _CHUNK):
        q = Q[i:i + _CHUNK]
        sim = q @ B.T
        if exclude is not None:
            sim.masked_fill_(exclude(i, i + q.shape[0]) if callable(exclude) else exclude, 1.0 - _INF)
        kk = min(k, sim.shape[1])
        out.append((1.0 - torch.topk(sim, kk, dim=1, largest=True).values).mean(1))
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
    (X of a 15k-patch row is 47 MB, read 4096 times). The chosen index stays on the GPU: reading it
    on the host every step stalled the queue between steps. The product is the same call on the same
    tensors as before, so the choice is identical.

    On the GPU the steps are replayed as a CUDA graph of KCENTER_GRAPH_STEPS steps (`_kcenter_steps_graph`):
    the same kernels on the same tensors, launched without the per-kernel host overhead, which was half of
    each step (149 us per step for 16k x 768, i.e. 59 of the 79 s that finalising a 97-row bank took).
    """
    n = X.shape[0]
    if n <= m:
        return torch.arange(n, device=X.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=g))
    idx = torch.empty(m, dtype=torch.long, device=X.device)
    idx[0] = first
    mind = 1.0 - X @ X[first]
    if X.is_cuda and m - 1 >= 2 * KCENTER_GRAPH_STEPS:
        return _kcenter_steps_graph(X, idx, mind, m)
    for i in range(1, m):
        j = torch.argmax(mind, 0, keepdim=True)
        idx[i:i + 1] = j
        mind = torch.minimum(mind, 1.0 - X @ X.index_select(0, j)[0])
    return idx


# k-center steps per captured CUDA graph: long enough that launching the graph costs little per step, short
# enough to capture quickly (the remainder after the last whole graph runs step by step).
KCENTER_GRAPH_STEPS = 64


def _kcenter_steps_graph(X: torch.Tensor, idx: torch.Tensor, mind: torch.Tensor, m: int) -> torch.Tensor:
    """Steps 1..m-1 of _kcenter, as replays of a captured CUDA graph (results identical to the loop).

    A step is the loop's kernels in its order: argmax of mind, index_select of that row, the matrix-vector
    product, 1 - it, and the element-wise minimum (written into mind in place, which an element-wise
    kernel allows). The step number lives on the device (`pos`), so one graph serves every step. One step
    is first run eagerly on a side stream, as CUDA graph capture requires (it also initialises cuBLAS);
    then a graph of KCENTER_GRAPH_STEPS steps is captured (capturing runs nothing) and replayed; the steps
    left over run eagerly. tests/test_gate.py compares the result with the first version of the loop.
    """
    pos = torch.ones(1, dtype=torch.long, device=X.device)

    def step():
        j = torch.argmax(mind, 0, keepdim=True)
        idx.index_copy_(0, pos, j)
        pos.add_(1)
        torch.minimum(mind, 1.0 - X @ X.index_select(0, j)[0], out=mind)

    side = torch.cuda.Stream(device=X.device)
    side.wait_stream(torch.cuda.current_stream(X.device))
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream(X.device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(KCENTER_GRAPH_STEPS):
            step()
    remaining = m - 2
    for _ in range(remaining // KCENTER_GRAPH_STEPS):
        graph.replay()
    for _ in range(remaining % KCENTER_GRAPH_STEPS):
        step()
    torch.cuda.current_stream(X.device).synchronize()
    del graph
    return idx


@dataclass
class _RowBank:
    vecs: torch.Tensor      # (m, D), a view into PatchBank's contiguous bank
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
        # The finished bank, all rows back to back (row r at _off[r]:_off[r + 1]), so rows r-1..r+1,
        # which every query of row r is compared with, are one contiguous slice.
        self._bv: torch.Tensor | None = None
        self._bf: torch.Tensor | None = None
        self._bs: torch.Tensor | None = None
        self._off: list[int] = []

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

    def _span(self, r: int) -> tuple[int, int]:
        """Slice of the contiguous bank holding rows r-1..r+1 (those that exist)."""
        return self._off[max(r - 1, 0)], self._off[min(r + 2, self.gh)]

    def _neighbours(self, r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b = self._span(r)
        return self._bv[a:b], self._bf[a:b], self._bs[a:b]

    def finalize(self, max_collect: int = 16384, null_samples: int = 10000) -> None:
        """Trim, coreset and null distribution of every row.

        Memory: the collected patches stay in host memory; the GPU holds the finished bank, the
        three collected rows the current row's trim reads, and one block of similarities. Holding
        every collected row there, as the first version did, peaked at 7.3 GB for the 51-row bank of
        session f428 on the 6 GB laptop GPU: the driver spilled to system memory and finalising took
        231 s. The random draws and the arithmetic of every step are those of the first version, so
        the bank is the same bit for bit (tests/test_gate.py).
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
            self._v[r] = self._f[r] = self._s[r] = None  # the per-frame pieces are copies: free them now
            if len(V) > max_collect:
                sel = np.sort(rng.choice(len(V), max_collect, replace=False))
                V, F, S = V[sel], F[sel], S[sel]
            raw.append((V, F, S))
        self._v = self._f = self._s = None
        # The null samples are drawn here, in the row order in which the first version drew them after
        # the coresets (trimming and coresets draw nothing from rng), so the same patches are sampled.
        null_sel = [np.sort(rng.choice(len(V), null_samples, replace=False)) if len(V) > null_samples else None
                    for V, _, _ in raw]

        # Coreset sizes are known before the trim, so the contiguous bank is allocated once.
        sizes = []
        for V, _, _ in raw:
            n_keep = max(1, int(round(len(V) * (1.0 - self.cfg.trim_frac))))
            sizes.append((n_keep, min(n_keep, self.cfg.bank_size_per_row)))
        self._off = [0]
        for _, m in sizes:
            self._off.append(self._off[-1] + m)
        dim = raw[0][0].shape[1]
        self._bv = torch.empty((self._off[-1], dim), dtype=torch.float32, device=dev)
        self._bf = torch.empty(self._off[-1], dtype=torch.int64, device=dev)
        self._bs = torch.empty(self._off[-1], dtype=torch.float64, device=dev)

        # Per row: trim likely distress (each patch's k-NN distance to other frames' patches in rows
        # r-1..r+1; the most distant trim_frac go), then the k-center coreset of what is left.
        trimmed = 0
        self.rows = []
        window: dict[int, tuple[torch.Tensor, ...]] = {}  # collected rows r-1..r+1 on the GPU
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
            del RV, RF
            n_keep = sizes[r][0]
            order = torch.argsort(d)[:n_keep]
            trimmed += len(V) - n_keep
            V, F, S = V[order], F[order], S[order]
            ci = _kcenter(V, self.cfg.bank_size_per_row, self.seed + r)
            a, b = self._off[r], self._off[r + 1]
            self._bv[a:b] = V[ci]
            self._bf[a:b] = F[ci]
            self._bs[a:b] = S[ci]
            self.rows.append(_RowBank(self._bv[a:b], self._bf[a:b], self._bs[a:b]))
        del window
        # Null distribution per row: leave-place-out distances of the bank patches (all of them,
        # before trimming, up to null_samples), scored exactly like queries against the final bank.
        null = []
        for r in range(self.gh):
            V, F, S = raw[r]
            raw[r] = None
            if null_sel[r] is not None:
                V, F, S = V[null_sel[r]], F[null_sel[r]], S[null_sel[r]]
            V, F, S = (torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in (V, F, S))
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
        """(distance, u) maps of shape (gh, gw). u = (d - median) / (null quantile - median).

        Distances stay on the device until the last row (one copy to the host per image); u is
        computed on the host with the first version's float32 arithmetic.
        """
        q = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).to(self.device)
        near = (self._bs - s_m).abs() < self.cfg.exclude_window_m  # bank patches of this place
        d = torch.empty((self.gh, self.gw), dtype=torch.float32, device=self.device)
        for r in range(self.gh):
            a, b = self._span(r)
            d[r] = _knn_mean(q[r], self._bv[a:b], self.cfg.knn_k, near[None, a:b])
        dist = d.cpu().numpy()
        u = np.zeros((self.gh, self.gw), np.float32)
        for r in range(self.gh):
            u[r] = (dist[r] - self.rows[r].med) / (self.rows[r].hi - self.rows[r].med)
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
    """Divide by the local mean brightness (keeps colour ratios); removes most of a shadow.

    Where the blur runs on the GPU (blur.device_for), so does the rest, with the same result bit for bit
    (`_normalise_illumination_gpu`): the numpy version spent ~200 ms per 4 MP gate image on float32
    temporaries, a median and a mean over three channels, around an 8 ms blur.
    """
    if not sigma:
        return rgb
    dev = device_for(rgb.shape[:2], sigma)
    if dev is not None:
        return _normalise_illumination_gpu(rgb, sigma, dev)
    f = rgb.astype(np.float32)
    lum = f.mean(-1)
    local = gaussian_blur(lum, sigma)
    target = float(np.median(lum))
    return np.clip(f * (target / np.maximum(local, 1.0))[..., None], 0, 255).astype(np.uint8)


@torch.inference_mode()
def _normalise_illumination_gpu(rgb: np.ndarray, sigma: float, dev: torch.device) -> np.ndarray:
    """normalise_illumination's numpy arithmetic on the GPU, operation for operation, so the result is identical.

    The channel sum of uint8 values is an integer below 2^24, exact in float32 in any order. numpy's mean
    then divides it by the count, an int64 scalar, which promotes the division to float64; the quotient is
    rounded to float32 only when stored, so it is computed the same way here. The median is numpy's: the middle value,
    or for an even count the float32 mean of the two middle values. The scale is float32(target) /
    max(local, 1) (numpy 2 casts the Python float to the array's float32), times the channel value, clipped
    to [0, 255] and truncated to uint8. IEEE float32 +, -, *, / are correctly rounded on both devices and
    each is its own kernel here (nothing is fused), so every value is the numpy value
    (tests/test_gpu_filters.py compares them).
    """
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).float()
    lum = _channel_mean_gpu(x)
    flat = torch.sort(lum.reshape(-1)).values
    n = flat.numel()
    med = flat[n // 2] if n % 2 else (flat[n // 2 - 1] + flat[n // 2]) / 2
    scale = med / torch.clamp_min(blur_tensor(lum, sigma), 1.0)
    return torch.clamp(x * scale[..., None], 0, 255).to(torch.uint8).cpu().numpy()


def paint_fraction(rgb: np.ndarray, grid_h: int, grid_w: int, ratio: float, sigma: float = 40.0) -> np.ndarray:
    """Share of each patch covered by road paint: much brighter than the local road surface.

    Local road brightness is a median-like estimate (blur of a brightness image with paint
    clipped), so a marking doesn't raise its own reference. Where the blur runs on the GPU, the paint mask
    is computed there, bit for bit as in numpy (`_paint_mask_gpu`); dilation and the patch means stay here.
    """
    dev = device_for(rgb.shape[:2], sigma)
    if dev is not None:
        paint = _paint_mask_gpu(rgb, ratio, sigma, dev)
    else:
        lum = rgb.astype(np.float32).mean(-1)
        ref0 = gaussian_blur(lum, sigma)
        clipped = np.minimum(lum, ref0 * 1.15)
        ref = gaussian_blur(clipped, sigma)
        paint = (lum > ratio * ref).astype(np.uint8)
    # Grow by half a patch: patches that only touch a marking's edge see it too.
    paint = cv2.dilate(paint, np.ones((PATCH, PATCH), np.uint8)).astype(np.float32)
    h, w = grid_h * PATCH, grid_w * PATCH
    return paint[:h, :w].reshape(grid_h, PATCH, grid_w, PATCH).mean(axis=(1, 3))


def _channel_mean_gpu(x: torch.Tensor) -> torch.Tensor:
    """numpy's float32 mean over the last axis (3 channels) of uint8 values, exactly: the integer sum, then a
    float64 division by the count (numpy divides by an int64 scalar), rounded to float32 when stored."""
    return ((x[..., 0] + x[..., 1] + x[..., 2]).double() / 3).float()


@torch.inference_mode()
def _paint_mask_gpu(rgb: np.ndarray, ratio: float, sigma: float, dev: torch.device) -> np.ndarray:
    """paint_fraction's mask (uint8 0/1) on the GPU with numpy's float32 operations (the Python floats 1.15
    and ratio are rounded to float32 first, as numpy 2 does), so it is identical to the numpy mask."""
    lum = _channel_mean_gpu(torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).float())
    clipped = torch.minimum(lum, blur_tensor(lum, sigma) * 1.15)
    return (lum > blur_tensor(clipped, sigma) * ratio).to(torch.uint8).cpu().numpy()


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

        The crop mode ranks whole-image candidates; the segment gate (geoai.gate) scores the patches
        and detections of one road segment only. Both read this one computation, so the two modes
        can't drift apart.

        Returns a dict:
          z            (gh, gw) float32 bank score u, 0 on paint patches (with mask_paint)
          probe_logit  (gh, gw) probe logits, PAINT_LOGIT on paint patches (exactly the map the probe
                       candidates grow on); None without a probe
          paint        (gh, gw) bool, patches mostly covered by road paint; all False without mask_paint
          crack_mask   (H, W) bool analysis-pixel crack mask; None with the crack filter off
          components   {"dino": [...], "crack": [...], "probe": [...]}: each detector's own candidates
                       before merging, as separate objects (merging sums the evidence of all three,
                       so a per-detector total needs these). "crack" holds every long thin crack
                       component, also when CrackConfig.as_candidates keeps them out of `candidates`.
          candidates   merged list, exactly what `detect` returns
          diag         per-image statistics, exactly what `detect` returns
        """
        feats = self._features(rgb)
        _, z = self.bank.score(feats, s_m)
        paint = self._paint(rgb)
        if paint is not None:
            z = np.where(paint, 0.0, z)
        logit = self._probe_logits(feats, paint)
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
            for c in comps:
                # Crack evidence on the same scale as DINOv2 mass: a crack just at the length and
                # z thresholds scores min_mass.
                ev = self.cfg.min_mass * (c.z_max / self.crack_cfg.z_threshold) * (c.length_mm / self.crack_cfg.min_length_mm)
                crack.append(Candidate(c.bbox, ev, 0.0, c.z_max, ["crack"], 0, c.length_mm))
        merged = dino + (crack if self.crack_cfg.as_candidates else []) + probe_cands
        # _merge_boxes grows the first box of each cluster in place: merge copies, keep the components.
        cands = _merge_boxes([replace(c, kinds=list(c.kinds)) for c in merged], gap=PATCH)
        diag = {"z_max": float(z.max()), "z_p99": float(np.percentile(z, 99)),
                "anomalous_patch_frac": float((z >= thr).mean()),
                "crack_pixel_frac": float(crack_mask.mean()) if crack_mask is not None else 0.0, **pdiag}
        return {"z": z, "probe_logit": logit,
                "paint": paint if paint is not None else np.zeros((self.gh, self.gw), bool),
                "crack_mask": crack_mask, "components": {"dino": dino, "crack": crack, "probe": probe_cands},
                "candidates": cands, "diag": diag}

    def probe_maps(self, rgb: np.ndarray) -> dict:
        """The probe's part of `maps` alone: no bank, no crack filter (the segment gate when only the probe
        decides, docs/pipeline/13 §4). Needs no bank frames and no `finalize`.

        Returns {"probe_logit": (gh, gw) logits or None without a probe, "paint": (gh, gw) bool}, computed
        by the same functions as in `maps` (illumination normalisation, DINOv2, paint mask, probe), so
        both are identical to those of `maps` for the same image. Without a probe DINOv2 isn't run.
        """
        paint = self._paint(rgb)
        logit = None if self.probe is None else self._probe_logits(self._features(rgb), paint)
        return {"probe_logit": logit, "paint": paint if paint is not None else np.zeros((self.gh, self.gw), bool)}

    def _paint(self, rgb: np.ndarray) -> np.ndarray | None:
        """Patches mostly covered by road paint, or None with mask_paint off."""
        if not self.cfg.mask_paint:
            return None
        return paint_fraction(rgb, self.gh, self.gw, self.cfg.paint_ratio) > self.cfg.paint_patch_frac

    def _probe_logits(self, feats: np.ndarray, paint: np.ndarray | None) -> np.ndarray | None:
        """Probe logits, PAINT_LOGIT on paint patches; None without a probe."""
        if self.probe is None:
            return None
        logit = self.probe.logits(feats)
        return logit if paint is None else np.where(paint, PAINT_LOGIT, logit)

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
