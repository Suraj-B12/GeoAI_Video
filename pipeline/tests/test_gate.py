"""Segment-mode distress check (geoai.gate) and the bank it relies on (geoai.anomaly).

- The faster bank (finalize, score, the k-NN and k-center kernels) reproduces the first version bit
  for bit on random data; the first version is copied below verbatim as the reference.
- AnomalyDetector.maps() gives exactly the candidates and statistics of the first version's detect().
- The gate view, a segment's patch rows and its scores, against independent calculations (the
  analysis camera model, brute-force loops), and on rendered frames with a real DINOv2 bank.
- The decision rule, mode `all`, the deterministic audit draw and its rate.
- A probe-only gate (evidence and crack_mm off): no bank, no crack filter, the probe scores of `maps`.
- Calibration against exhaustive search and hand-computed answers, its edge cases, the JSON round trip.
"""
import hashlib
import itertools
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from geoai.anomaly import (AnomalyDetector, Candidate, PatchBank, _exclusion, _kcenter, _knn_mean, _merge_boxes,
                           normalise_illumination, paint_fraction)
from geoai.config import AnomalyConfig, Config, CrackConfig, GateConfig, load_config
from geoai.dinov2 import PATCH, PatchFeatures
from geoai.extract import ExtractionError
from geoai.gate import (CALIBRATION_SCHEMA, GATE_SIGNALS, SCORE_KEYS, SegmentGate, _need, audit_draw, calibrate,
                        calibration_thresholds, gate_far_m, plan_segment_view, read_calibration, write_calibration)
from geoai.geometry import CameraModel, vehicle_from_camera
from geoai.probe import DistressProbe
from geoai.simtexture import asphalt, draw_crack, draw_marking, draw_patch, draw_pothole, to_rgb

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = Path(AnomalyConfig().weights)
needs_dino = pytest.mark.skipif(not WEIGHTS.exists(), reason="DINOv2 weights not downloaded")

# The S21 of the 26 Oct sessions (field/runs/..._293d/report.json): 4K, pitch 22.7 deg, roll -4.4 deg, 1.35 m.
S21 = CameraModel(3840, 2160,
                  np.array([[2910.287388392857, 0.0, 1933.5457356770833], [0.0, 2908.4112258184523, 1096.880580357143],
                            [0.0, 0.0, 1.0]]), np.zeros(5),
                  np.array([[0.997043475893689, -0.07683948970308926, -3.469446951953614e-18],
                            [-0.029686348938060027, -0.3852001184048247, 0.9223554572222343],
                            [-0.07087332265781605, -0.9196284910783691, -0.38634234887255525]]),
                  np.array([0.0, 0.0, 1.35]))


# ==== first version of the bank and of detect (pipeline 0.1.0, commit 8186b89), verbatim ==========

_INF = 1e4


def _v1_knn_mean(Q, B, k, exclude):
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


def _v1_kcenter(X, m, seed):
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
class _V1Row:
    vecs: torch.Tensor
    frame: torch.Tensor
    s: torch.Tensor
    med: float = 0.0
    hi: float = 1.0


class _V1Bank:
    def __init__(self, grid_h, grid_w, cfg, device, seed=0):
        self.gh, self.gw = grid_h, grid_w
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self._v = [[] for _ in range(grid_h)]
        self._f = [[] for _ in range(grid_h)]
        self._s = [[] for _ in range(grid_h)]
        self.n_frames = 0
        self.rows = []
        self.stats = {}

    def add(self, feats, s_m, keep=None):
        fid = self.n_frames
        self.n_frames += 1
        for r in range(self.gh):
            sel = slice(None) if keep is None else keep[r]
            v = feats[r][sel]
            if len(v):
                self._v[r].append(v.astype(np.float32))
                self._f[r].append(np.full(len(v), fid, np.int64))
                self._s[r].append(np.full(len(v), s_m, np.float64))

    def _neighbours(self, r):
        rows = [i for i in (r - 1, r, r + 1) if 0 <= i < self.gh]
        return (torch.cat([self.rows[i].vecs for i in rows]), torch.cat([self.rows[i].frame for i in rows]),
                torch.cat([self.rows[i].s for i in rows]))

    def finalize(self, max_collect=16384):
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
        self._v = self._f = self._s = None
        trimmed = 0
        rows_after_trim = []
        for r in range(self.gh):
            idx = [i for i in (r - 1, r, r + 1) if 0 <= i < self.gh]
            RV = torch.cat([raw[i][0] for i in idx])
            RF = torch.cat([raw[i][1] for i in idx])
            V, F, S = raw[r]
            d = _v1_knn_mean(V, RV, self.cfg.knn_k, F[:, None] == RF[None, :])
            n_keep = max(1, int(round(len(V) * (1.0 - self.cfg.trim_frac))))
            order = torch.argsort(d)[:n_keep]
            trimmed += len(V) - n_keep
            rows_after_trim.append((V[order], F[order], S[order]))
        self.rows = []
        for r, (V, F, S) in enumerate(rows_after_trim):
            ci = _v1_kcenter(V, self.cfg.bank_size_per_row, self.seed + r)
            self.rows.append(_V1Row(V[ci], F[ci], S[ci]))
        null = []
        for r, (V, F, S) in enumerate(raw):
            if len(V) > 10000:
                sel = torch.from_numpy(np.sort(rng.choice(len(V), 10000, replace=False))).to(dev)
                V, F, S = V[sel], F[sel], S[sel]
            BV, BF, BS = self._neighbours(r)
            ex = (F[:, None] == BF[None, :]) | ((S[:, None] - BS[None, :]).abs() < self.cfg.exclude_window_m)
            null.append(_v1_knn_mean(V, BV, self.cfg.knn_k, ex).double().cpu().numpy())
        for r in range(self.gh):
            med, q90 = (float(v) for v in np.quantile(null[r], [0.5, 0.9]))
            hi = med + self.cfg.tail_factor * max(q90 - med, 1e-5)
            self.rows[r].med, self.rows[r].hi = med, hi
        self.stats = {"frames": self.n_frames, "trimmed_patches": trimmed,
                      "bank_patches": int(sum(len(b.vecs) for b in self.rows)),
                      "null_samples_per_row": int(np.median([len(n) for n in null])),
                      "row_median": [b.med for b in self.rows], "row_hi": [b.hi for b in self.rows]}

    def score(self, feats, s_m):
        dist = np.zeros((self.gh, self.gw), np.float32)
        u = np.zeros((self.gh, self.gw), np.float32)
        q = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).to(self.device)
        for r in range(self.gh):
            BV, _, BS = self._neighbours(r)
            ex = ((BS - s_m).abs() < self.cfg.exclude_window_m)[None, :]
            d = _v1_knn_mean(q[r], BV, self.cfg.knn_k, ex).cpu().numpy()
            dist[r] = d
            u[r] = (d - self.rows[r].med) / (self.rows[r].hi - self.rows[r].med)
        return dist, u


def _v1_detect(self, rgb, s_m, gsd_rows_mm):
    """AnomalyDetector.detect and _probe_candidates of the first version (self = the detector).

    Returns (candidates, diag) as the first version did, plus its z and probe-logit maps for comparison.
    """
    from scipy import ndimage
    from skimage.filters import apply_hysteresis_threshold
    feats = self._features(rgb)
    _, z = self.bank.score(feats, s_m)
    paint = None
    if self.cfg.mask_paint:
        paint = paint_fraction(rgb, self.gh, self.gw, self.cfg.paint_ratio) > self.cfg.paint_patch_frac
        z = np.where(paint, 0.0, z)
    # _probe_candidates(feats, paint)
    logit = None
    if self.probe is None:
        probe_cands, pdiag = [], {}
    else:
        logit = self.probe.logits(feats)
        if paint is not None:
            logit = np.where(paint, -1e3, logit)
        seed = self.probe.seed_logit
        grow = seed - self.cfg.probe_grow_logit
        probe_cands = []
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
                probe_cands.append(Candidate((c0 * PATCH, r0 * PATCH, c1 * PATCH, r1 * PATCH), mass, 0.0, 0.0,
                                             ["probe"], int(comp.sum()), mass=0.0,
                                             probe_max=float(logit[sl][comp].max())))
        pdiag = {"probe_max_logit": float(logit.max()), "probe_patch_frac": float((logit >= seed).mean())}
    thr = self.cfg.score_threshold
    seeds = apply_hysteresis_threshold(z, self.cfg.grow_fraction * thr, thr)
    labels, n = ndimage.label(seeds, structure=np.ones((3, 3), int))
    cands = []
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
                ev = self.cfg.min_mass * (c.z_max / self.crack_cfg.z_threshold) * (c.length_mm / self.crack_cfg.min_length_mm)
                cands.append(Candidate(c.bbox, ev, 0.0, c.z_max, ["crack"], 0, c.length_mm))
    cands = _merge_boxes(cands + probe_cands, gap=PATCH)
    diag = {"z_max": float(z.max()), "z_p99": float(np.percentile(z, 99)),
            "anomalous_patch_frac": float((z >= thr).mean()),
            "crack_pixel_frac": float(crack_mask.mean()) if crack_mask is not None else 0.0, **pdiag}
    return cands, diag, z, logit


# ==== bank speed-up: bit-identical to the first version ============================================

def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def test_knn_mean_identical_to_first_version():
    rng = np.random.default_rng(0)
    Q = torch.from_numpy(_unit(rng.standard_normal((5000, 64)))).to(DEV)       # two query blocks
    B = torch.from_numpy(_unit(rng.standard_normal((700, 64)))).to(DEV)
    qf = torch.from_numpy(rng.integers(0, 40, 5000)).to(DEV)
    bf = torch.from_numpy(rng.integers(0, 40, 700)).to(DEV)
    qs = torch.from_numpy(rng.uniform(0, 100, 5000)).to(DEV)
    bs = torch.from_numpy(rng.uniform(0, 100, 700)).to(DEV)
    full = (qf[:, None] == bf[None, :]) | ((qs[:, None] - bs[None, :]).abs() < 8.0)
    full[:50] = True
    full[:50, :2] = False            # fewer unmasked entries than k: _INF enters the mean
    full[50:60] = True               # nothing left: the mean is _INF
    row = torch.from_numpy(rng.random(700) < 0.3).to(DEV)[None, :]
    ex = _exclusion(qf, bf, qs, bs, 8.0)
    ex_patched = lambda i0, i1: full[i0:i1]  # noqa: E731
    for k in (1, 3, 5):
        assert torch.equal(_knn_mean(Q, B, k, None), _v1_knn_mean(Q, B, k, None))
        assert torch.equal(_knn_mean(Q, B, k, row), _v1_knn_mean(Q, B, k, row))
        assert torch.equal(_knn_mean(Q, B, k, ex_patched), _v1_knn_mean(Q, B, k, full))
    full2 = (qf[:, None] == bf[None, :]) | ((qs[:, None] - bs[None, :]).abs() < 8.0)
    assert torch.equal(_knn_mean(Q, B, 3, ex), _v1_knn_mean(Q, B, 3, full2))
    out = _knn_mean(Q, B, 3, ex_patched)
    assert float(out[55]) == _INF and float(out[0]) > 1.0          # masked rows really are masked
    assert torch.equal(_knn_mean(Q, B[:2], 3, None), _v1_knn_mean(Q, B[:2], 3, None))  # fewer bank rows than k


def test_kcenter_identical_to_first_version():
    rng = np.random.default_rng(1)
    X = torch.from_numpy(_unit(rng.standard_normal((3000, 96)) + 2.0 * rng.standard_normal((1, 96)))).to(DEV)
    for m, seed in ((300, 0), (1000, 7)):
        assert torch.equal(_kcenter(X, m, seed), _v1_kcenter(X, m, seed))
    assert torch.equal(_kcenter(X[:200], 300, 0), torch.arange(200, device=DEV))


def _bank_frames(gh=3, gw=310, frames=40, dim=192, seed=2):
    """Clustered unit features (texture-like k-NN structure), rows of different sizes:
    row 0 keeps everything (12,400 > max_collect 11,000), row 1 ~85 % (> 10,000 null samples), row 2 ~72 %."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((16, dim))
    out = []
    for f in range(frames):
        lab = rng.integers(0, 16, size=(gh, gw))
        feats = _unit(centres[lab] + 0.7 * rng.standard_normal((gh, gw, dim)))
        keep = rng.random((gh, gw)) < np.array([1.0, 0.85, 0.72])[:, None]
        out.append((feats, 2.5 * f, keep))
    return out


def test_bank_identical_to_first_version():
    """finalize (trim, coreset, null) and score, new vs first version: every tensor and number equal."""
    cfg = AnomalyConfig(bank_size_per_row=512, exclude_window_m=3.0, probe=None)
    frames = _bank_frames()
    new, old = PatchBank(3, 310, cfg, DEV, seed=5), _V1Bank(3, 310, cfg, DEV, seed=5)
    for feats, s, keep in frames:
        new.add(feats, s, keep)
        old.add(feats, s, keep)
    sizes = [sum(int(k[r].sum()) for _, _, k in frames) for r in range(3)]
    assert sizes[0] > 11000 and 10000 < sizes[1] <= 11000 and sizes[2] < 10000   # every branch is taken
    new.finalize(max_collect=11000)
    old.finalize(max_collect=11000)
    for a, b in zip(new.rows, old.rows):
        assert len(a.vecs) == 512
        assert torch.equal(a.vecs, b.vecs) and torch.equal(a.frame, b.frame) and torch.equal(a.s, b.s)
        assert a.med == b.med and a.hi == b.hi
    assert new.stats == old.stats
    rng = np.random.default_rng(9)
    for s_m in (0.0, 47.5, 200.0):
        q = _unit(rng.standard_normal((3, 310, 192)))
        dn, un = new.score(q, s_m)
        do, uo = old.score(q, s_m)
        assert np.array_equal(dn, do) and np.array_equal(un, uo)
        assert dn.dtype == np.float32 and un.dtype == np.float32


# ==== maps() = first version's detect() ===========================================================

RES = 0.0025
WIN_H, WIN_W = 224, 392


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(5)
    road = asphalt(1.0, 30.0, RES, seed=4, lateral0_m=-0.5)
    draw_marking(road, 0.35, 0.10, dash_m=1.0, gap_m=2.0)
    draw_pothole(road, (-0.1, 14.6), 0.12, rng)
    draw_crack(road, (-0.4, 22.4), 90.0, 0.6, 5.0, rng)
    draw_patch(road, (-0.05, 7.6), (0.35, 0.30), 10.0, tone=0.65)
    return road


def _window(road, along_m):
    r0 = int(round((along_m - road.along0_m) / RES))
    return to_rgb(road.img[r0:r0 + WIN_H, 0:WIN_W])


@pytest.fixture(scope="module")
def detector(scene, tmp_path_factory):
    """Bank on synthetic asphalt, with a probe made for this scene (direction plain asphalt -> pothole)."""
    feats = PatchFeatures(WEIGHTS, "cpu", tile_patches=32, overlap=4, layers=(4, 8))
    sigma = AnomalyConfig().illum_sigma_px
    f_pot = feats(normalise_illumination(_window(scene, 14.45), sigma))
    f_clean = feats(normalise_illumination(_window(scene, 3.0), sigma))
    gh, gw = f_pot.shape[:2]
    rows, cols = np.mgrid[0:gh, 0:gw]
    pot = (np.abs(rows * PATCH + 7 - 0.15 / RES) < 0.14 / RES) & (np.abs(cols * PATCH + 7 - 0.4 / RES) < 0.14 / RES)
    w = f_pot[pot].mean(0) - f_clean.reshape(-1, f_clean.shape[-1]).mean(0)
    w = (w / np.linalg.norm(w)).astype(np.float32)
    seed_logit = float(np.quantile(f_pot @ w, 0.97))
    path = tmp_path_factory.mktemp("probe") / "probe.npz"
    DistressProbe(w, 0.0, (4, 8), sigma, seed_logit, {"name": "test", "illum_sigma_px": sigma}).save(path)
    cfg = AnomalyConfig(exclude_window_m=3.0, probe=str(path))
    det = AnomalyDetector(feats, WIN_H, WIN_W, cfg, CrackConfig(min_length_mm=150.0))
    gsd = np.full(WIN_H, RES * 1000)
    rng = np.random.default_rng(0)
    along = 0.3
    while along < 29.0:
        det.add_bank_frame(_window(scene, along), float(along), gsd)
        along += 0.9 * (1 + rng.uniform(-cfg.bank_jitter, cfg.bank_jitter))
    det.finalize()
    return det, gsd


ALONGS = (14.45, 22.2, 7.4, 3.0, 18.3)   # pothole, crack, patch, clean, clean


@needs_dino
def test_maps_equals_first_version_detect(scene, detector):
    det, gsd = detector
    seen = {"dino": 0, "crack": 0, "probe": 0}
    for along in ALONGS:
        rgb = _window(scene, along)
        m = det.maps(rgb, along, gsd)
        ref_c, ref_d, ref_z, ref_logit = _v1_detect(det, rgb, along, gsd)
        cands, diag = det.detect(rgb, along, gsd)
        assert cands == ref_c and diag == ref_d
        assert m["candidates"] == ref_c and m["diag"] == ref_d
        assert np.array_equal(m["z"], ref_z) and np.array_equal(m["probe_logit"], ref_logit)
        assert m["paint"].dtype == bool and m["paint"].shape == (det.gh, det.gw)
        assert np.array_equal(m["crack_mask"], det.cracks.detect(rgb, gsd)[0])
        comps = m["components"]
        for k in seen:
            seen[k] += len(comps[k])
        # Merging only regroups the components: total evidence is conserved, originals are untouched.
        total = sum(c.score for c in comps["dino"] + comps["crack"] + comps["probe"])
        assert math.isclose(total, sum(c.score for c in m["candidates"]), rel_tol=1e-12, abs_tol=1e-12)
        assert all(c.kinds == ["dino"] for c in comps["dino"]) and all(c.kinds == ["crack"] for c in comps["crack"])
    assert all(n > 0 for n in seen.values()), seen        # all three detectors' paths were exercised


@needs_dino
def test_maps_equals_first_version_without_crack_candidates_or_probe(scene, detector):
    det, gsd = detector
    crack_cfg, probe = det.crack_cfg, det.probe
    try:
        det.crack_cfg = replace(crack_cfg, as_candidates=False)
        rgb = _window(scene, 22.2)
        m = det.maps(rgb, 22.2, gsd)
        ref_c, ref_d, _, _ = _v1_detect(det, rgb, 22.2, gsd)
        assert m["candidates"] == ref_c and m["diag"] == ref_d
        assert m["components"]["crack"] and not any("crack" in c.kinds for c in m["candidates"])
        det.crack_cfg, det.probe = crack_cfg, None
        m = det.maps(rgb, 22.2, gsd)
        ref_c, ref_d, _, ref_logit = _v1_detect(det, rgb, 22.2, gsd)
        assert m["candidates"] == ref_c and m["diag"] == ref_d
        assert m["probe_logit"] is None and ref_logit is None and m["components"]["probe"] == []
    finally:
        det.crack_cfg, det.probe = crack_cfg, probe


@needs_dino
def test_probe_maps_are_the_probe_part_of_maps_without_a_bank(scene, detector):
    """probe_maps gives maps' probe logits and paint mask bit for bit (maps is checked against the first
    version above), and works on a detector that has no bank frame and was never finalized."""
    det, gsd = detector
    fresh = AnomalyDetector(det.features, WIN_H, WIN_W, det.cfg, det.crack_cfg)
    assert fresh.bank.n_frames == 0
    for along in ALONGS:
        rgb = _window(scene, along)
        m = det.maps(rgb, along, gsd)
        for d in (det, fresh):
            pm = d.probe_maps(rgb)
            assert set(pm) == {"probe_logit", "paint"}
            assert np.array_equal(pm["probe_logit"], m["probe_logit"]) and np.array_equal(pm["paint"], m["paint"])
    fresh.probe = None
    calls = []
    fresh.features = lambda x: calls.append(x)              # without a probe DINOv2 is not run at all
    pm = fresh.probe_maps(_window(scene, 3.0))
    assert pm["probe_logit"] is None and pm["paint"].shape == (fresh.gh, fresh.gw) and calls == []


# ==== the gate view ===============================================================================

def _near(model: CameraModel, cfg: Config) -> float:
    """Closest road on the centre column at the bottom usable row (as segments.SegmentGeometry)."""
    v = math.floor(model.height * (1 - cfg.geometry.bottom_exclude_frac) - 1)
    _, y, _ = model.pixel_to_ground(np.array([(model.width - 1) / 2]), np.array([float(v)]))
    return float(y[0])


def test_gate_view_s21():
    cfg = Config()
    near = _near(S21, cfg)
    far = gate_far_m(near, cfg)
    assert far == pytest.approx(near + 5.0 + 1.0 + 0.5)
    v = plan_segment_view(S21, near, far, cfg, False)
    _, v_far, _ = S21.ground_to_pixel(np.array([0.0]), np.array([far]))
    assert v.v1 == 2160 and 0 <= v.v0 <= math.floor(v_far[0])          # far road and bottom row in view
    assert v.ana_w % PATCH == 0 and v.ana_h % PATCH == 0
    assert v.ana_w * v.ana_h <= cfg.gate.max_analysis_px and v.notes["area_capped"]
    assert v.ana_w * v.ana_h > 0.98 * cfg.gate.max_analysis_px           # the cap is used, not wasted
    assert abs(v.sx - v.sy) * (v.v1 - v.v0) < 1.0                        # isotropic within one frame row
    # The median along-track resolution (~2.3 mm/px) is coarser than the 2 mm target: no resolution downscale.
    assert v.notes["downscale_resolution"] == 1.0 and 2.0 < v.notes["median_gsd_mm"] < 2.6
    assert v.analysis_image(np.zeros((2160, 3840, 3), np.uint8)).shape == (v.ana_h, v.ana_w, 3)
    # The analysis camera model is the frame's, cropped and scaled: ground points land where View maps them.
    x = np.array([-1.5, 0.0, 0.8, 2.0, 0.0])
    y = np.array([1.6, 2.5, 4.0, 6.0, 7.9])
    u, vv, ok = S21.ground_to_pixel(x, y)
    ua, va, oka = v.model_ana.ground_to_pixel(x, y)
    assert ok.all() and oka.all()
    np.testing.assert_allclose(ua, (u + 0.5) * v.sx - 0.5, atol=1e-6)
    np.testing.assert_allclose(va, (vv - v.v0 + 0.5) * v.sy - 0.5, atol=1e-6)
    g = v.gsd_rows_mm
    assert g.shape == (v.ana_h,) and np.all(np.diff(g) < 0)              # finer towards the bottom


def test_gate_view_rows_land_on_whole_patch_rows():
    """A bright band of frame rows appears at the analysis rows View's scale maps it to (no bottom crop)."""
    cfg = Config()
    near = _near(S21, cfg)
    v = plan_segment_view(S21, near, gate_far_m(near, cfg), cfg, False)
    assert round((v.v1 - v.v0) * v.sy) == v.ana_h and round(v.width * v.sx) == v.ana_w
    img = np.zeros((2160, 3840, 3), np.uint8)
    for r0, r1 in ((v.v0, v.v0 + 60), (1200, 1290), (2100, 2160)):
        img[:] = 0
        img[r0:r1] = 255
        prof = v.analysis_image(img)[:, :, 0].mean(1) / 255.0
        lit = np.flatnonzero(prof > 0.5)
        lo, hi = (r0 - v.v0) * v.sy, (r1 - v.v0) * v.sy        # band edges in analysis pixel edges
        assert abs(lit[0] - lo) <= 1 and abs(lit[-1] + 1 - hi) <= 1, (r0, r1, lit[0], lit[-1], lo, hi)


def test_gate_view_uncapped_scale_and_clipping():
    cfg = Config()
    near = _near(S21, cfg)
    # A short window: the area is under the cap and the scale comes from the resolution target.
    far = near + 1.0
    v = plan_segment_view(S21, near, far, cfg, False)
    _, vf, _ = S21.ground_to_pixel(np.array([0.0]), np.array([far]))
    rows = np.arange(math.floor(vf[0]), 2160, 2, dtype=float)
    gsd, _ = S21.gsd_mm(np.full_like(rows, (3840 - 1) / 2), rows)
    ds = min(1.0, max(0.25, float(np.median(gsd)) / cfg.extract.target_gsd_mm))
    assert not v.notes["area_capped"] and v.notes["downscale_resolution"] == pytest.approx(ds, rel=1e-12)
    assert v.ana_w == PATCH * math.floor(3840 * ds / PATCH) and v.sx == v.ana_w / 3840
    # Far road just below the top row (30 m: row ~30): whole patch rows would reach above the frame,
    # so the window starts at row 0; beyond the top row (1 km) it is clipped there. The cap holds.
    v = plan_segment_view(S21, near, 30.0, cfg, True)
    assert 0 < v.notes["far_row"] < 60 and v.v0 == 0 and v.notes["far_row_clipped"] is False and v.rotate180
    assert v.ana_w * v.ana_h <= cfg.gate.max_analysis_px and round(v.v1 * v.sy) == v.ana_h
    v = plan_segment_view(S21, near, 1000.0, cfg, False)
    assert v.v0 == 0 and v.notes["far_row_clipped"] is True
    # Excluded bottom rows (vehicle body) end the window higher.
    cfg2 = load_config(overrides={"geometry": {"bottom_exclude_frac": 0.1}})
    near2 = _near(S21, cfg2)
    v = plan_segment_view(S21, near2, gate_far_m(near2, cfg2), cfg2, False)
    assert v.v1 == math.floor(2160 * 0.9 - 1) + 1


def test_gate_view_rejects_impossible_geometry():
    cfg = Config()
    with pytest.raises(ExtractionError):
        plan_segment_view(S21, 2.0, 2.0, cfg, False)
    with pytest.raises(ExtractionError):
        plan_segment_view(S21, 0.0, 5.0, cfg, False)
    up = CameraModel(3840, 2160, S21.K, np.zeros(5), vehicle_from_camera(np.array([0.0, -0.9, 0.44]), "forward"),
                     np.array([0.0, 0.0, 1.35]))                       # looking 26 deg above the horizon
    with pytest.raises(ExtractionError):
        plan_segment_view(up, 1.4, 8.0, cfg, False)


# ==== a segment's patches and scores ==============================================================

def _stub(view, seed_logit=0.3, as_candidates=True, probe=True):
    """What SegmentGate reads from a detector, without DINOv2."""
    return SimpleNamespace(gh=view.ana_h // PATCH, gw=view.ana_w // PATCH, cfg=AnomalyConfig(probe=None),
                           crack_cfg=CrackConfig(as_candidates=as_candidates),
                           probe=SimpleNamespace(seed_logit=seed_logit) if probe else None)


@pytest.fixture(scope="module")
def s21_view():
    cfg = Config()
    near = _near(S21, cfg)
    return cfg, near, plan_segment_view(S21, near, gate_far_m(near, cfg), cfg, False)


def _seg_rows(model, d0, d1):
    _, vt, _ = model.ground_to_pixel(np.array([0.0]), np.array([d1]))
    _, vb, _ = model.ground_to_pixel(np.array([0.0]), np.array([d0]))
    return float(vt[0]), float(vb[0])


def _cand(y0, y1, score, kind, length=0.0):
    return Candidate((100, int(y0), 300, int(y1)), score, 2.0, 7.0, [kind], 3, length)


def test_segment_patch_rows_match_ground_geometry(s21_view):
    """Patch rows of a segment = those whose centre lies between the analysis rows of its two edges,
    computed independently through the analysis camera model."""
    cfg, near, view = s21_view
    gate = SegmentGate(_stub(view), view, cfg)
    centres = PATCH * np.arange(view.ana_h // PATCH) + (PATCH - 1) / 2       # pixel-centre coordinates
    for d0 in (near + 1e-3, near + 0.37, near + 0.81, near + 1.0):
        d1 = d0 + cfg.segments.length_m
        _, ya, _ = view.model_ana.ground_to_pixel(np.array([0.0, 0.0]), np.array([d1, d0]))
        expected = np.flatnonzero((centres > ya[0]) & (centres <= ya[1]))
        got = gate.segment_patch_rows(_seg_rows(S21, d0, d1))
        assert np.array_equal(got, expected) and 40 < got.size < view.ana_h // PATCH
        if d0 < near + 0.01:
            assert got[-1] == view.ana_h // PATCH - 1        # a segment starting at the bottom row reaches it
    # Half-open convention: rows (v_top, v_bottom]; abutting segments share no patch and miss none.
    pv = gate.patch_rows_v
    a = gate.segment_patch_rows((pv[10], pv[30]))
    b = gate.segment_patch_rows((pv[30], pv[50]))
    assert np.array_equal(a, np.arange(11, 31)) and np.array_equal(b, np.arange(31, 51))


def test_segment_scores_against_brute_force(s21_view):
    cfg, near, view = s21_view
    gh, gw = view.ana_h // PATCH, view.ana_w // PATCH
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 0.6, (gh, gw)).astype(np.float32)
    logit = rng.normal(0.0, 0.3, (gh, gw)).astype(np.float32)
    d0 = near + 0.42
    d1 = d0 + 5.0
    _, ya, _ = view.model_ana.ground_to_pixel(np.array([0.0, 0.0]), np.array([d1, d0]))
    yt, yb = float(ya[0]), float(ya[1])
    # Candidates centred just inside / just outside both edges, and well inside.
    dino = [_cand(yt - 9, yt - 1, 11.0, "dino"), _cand(yt + 1, yt + 5, 13.0, "dino"),
            _cand(yb - 6, yb + 2, 17.0, "dino"), _cand(yb + 1, yb + 7, 19.0, "dino"),
            _cand((yt + yb) / 2 - 20, (yt + yb) / 2 + 20, 23.0, "dino")]
    crack = [_cand(yt + 30, yt + 34, 29.0, "crack", 410.0), _cand(yb + 2, yb + 8, 31.0, "crack", 220.0),
             _cand(yb - 40, yb - 2, 37.0, "crack", 180.0)]
    maps = {"z": z, "probe_logit": logit, "paint": np.zeros((gh, gw), bool), "crack_mask": None,
            "components": {"dino": dino, "crack": crack, "probe": [_cand(yt + 5, yt + 50, 99.0, "probe")]},
            "candidates": [], "diag": {}}

    def inside(c):
        yc = (c.bbox[1] + c.bbox[3]) / 2 - 0.5             # centre in analysis pixel-centre coordinates
        return yt < yc <= yb

    patch_rows = [p for p in range(gh) if yt < PATCH * p + (PATCH - 1) / 2 <= yb]
    n = len(patch_rows) * gw
    rows_full = _seg_rows(S21, d0, d1)
    for as_cand, probe in ((True, True), (False, True), (True, False)):
        gate = SegmentGate(_stub(view, 0.3, as_cand, probe), view, cfg)
        s = gate.restrict({**maps, "probe_logit": logit if probe else None}, rows_full)
        assert set(s) == set(SCORE_KEYS) and s["n_patches"] == n
        if probe:
            assert s["probe_frac"] == sum(int(logit[p, q] >= 0.3) for p in patch_rows for q in range(gw)) / n
            assert s["probe_max_logit"] == max(float(logit[p, q]) for p in patch_rows for q in range(gw))
        else:
            assert s["probe_frac"] is None and s["probe_max_logit"] is None
        ev = [c for c in dino + (crack if as_cand else []) if inside(c)]
        assert s["evidence"] == pytest.approx(sum(c.score for c in ev), abs=1e-12)
        assert s["crack_mm"] == pytest.approx(sum(c.crack_length_mm for c in crack if inside(c)), abs=1e-12)
        assert s["z_max"] == max(float(z[p, q]) for p in patch_rows for q in range(gw))
        assert s["anomalous_patch_frac"] == sum(int(z[p, q] >= 1.0) for p in patch_rows for q in range(gw)) / n
    # The construction really straddles both edges.
    assert [inside(c) for c in dino] == [False, True, True, False, True]
    assert [inside(c) for c in crack] == [True, False, True]


def test_segment_rows_errors(s21_view):
    cfg, near, view = s21_view
    gate = SegmentGate(_stub(view), view, cfg)
    with pytest.raises(ValueError):
        gate.segment_patch_rows((900.0, 900.0))                      # empty interval
    with pytest.raises(ValueError):
        gate.segment_patch_rows((float("nan"), 900.0))
    with pytest.raises(ValueError):
        gate.segment_patch_rows((view.v0 - 10.0, 1500.0))            # reaches above the view
    with pytest.raises(ValueError):
        gate.segment_patch_rows((2170.0, 2190.0))                    # below the last patch row
    with pytest.raises(ValueError):
        gate.score(np.zeros((1080, 1920, 3), np.uint8), 0.0, (1000.0, 1500.0))   # not the view's frame size
    bad = SimpleNamespace(**{**vars(_stub(view)), "gw": 3})
    with pytest.raises(ValueError):
        SegmentGate(bad, view, cfg)


def _probe_only_cal(path: Path, probe_frac: float | None) -> Path:
    """A calibration file that turns evidence and crack_mm off (the shape gate-calibrate writes)."""
    write_calibration(path, {"schema": CALIBRATION_SCHEMA, "target_recall": 0.9, "n_rows": 10, "n_positive": 3,
                             "any_of": {"status": "ok", "recall": 1.0,
                                        "thresholds": {"probe_frac": probe_frac, "evidence": None, "crack_mm": None}}})
    return path


def test_probe_only_gate_has_no_bank_and_scores_the_probe_alone(s21_view, tmp_path):
    """With evidence and crack_mm off the gate never touches the bank or the crack filter: score() takes the
    probe map from probe_maps(), the bank-dependent scores are None, and the decision is the probe's."""
    cfg, near, view = s21_view
    gh, gw = view.ana_h // PATCH, view.ana_w // PATCH
    logit = np.random.default_rng(5).normal(0.0, 0.3, (gh, gw)).astype(np.float32)
    calls = []

    def probe_maps(ana):
        calls.append(ana.shape)
        return {"probe_logit": logit, "paint": np.zeros((gh, gw), bool)}

    def maps(*a):
        raise AssertionError("the bank path must not run")

    det = SimpleNamespace(**vars(_stub(view)), probe_maps=probe_maps, maps=maps)
    gcfg = load_config(overrides={"gate": {"calibration": str(_probe_only_cal(tmp_path / "c.json", 0.02)),
                                           "audit_frac": 0.0}})
    gate = SegmentGate(det, view, gcfg)
    assert not gate.needs_bank
    assert gate.thresholds["evidence"] is None and gate.thresholds["crack_mm"] is None
    for fn in (lambda: gate.add_bank_frame(np.zeros((2160, 3840, 3), np.uint8), 0.0), gate.finalize):
        with pytest.raises(RuntimeError, match="probe only"):
            fn()
    d0 = near + 0.42
    rows_full = _seg_rows(S21, d0, d0 + 5.0)
    _, ya, _ = view.model_ana.ground_to_pixel(np.array([0.0, 0.0]), np.array([d0 + 5.0, d0]))
    patch_rows = [p for p in range(gh) if ya[0] < PATCH * p + (PATCH - 1) / 2 <= ya[1]]
    n = len(patch_rows) * gw
    s = gate.score(np.zeros((2160, 3840, 3), np.uint8), 0.0, rows_full)
    assert calls == [(view.ana_h, view.ana_w, 3)]
    frac = sum(int(logit[p, q] >= 0.3) for p in patch_rows for q in range(gw)) / n
    assert s == {"probe_frac": frac, "probe_max_logit": max(float(logit[p, q]) for p in patch_rows for q in range(gw)),
                 "evidence": None, "crack_mm": None, "z_max": None, "anomalous_patch_frac": None, "n_patches": n}
    assert 0.02 < frac < 0.5                                   # the threshold below really is crossed
    assert gate.decide(s, "r")["reasons"] == ["probe_frac"]
    assert gate.decide({**s, "probe_frac": 0.019}, "r")["decision"] == "skip"
    # Either signal on keeps the bank: the configured defaults, and evidence alone.
    assert SegmentGate(det, view, cfg).needs_bank
    assert SegmentGate(det, view, GateConfig(crack_mm=None)).needs_bank is True
    assert SegmentGate(det, view, GateConfig(evidence=None, crack_mm=None)).needs_bank is False


# ==== decision ====================================================================================

ZERO = {"probe_frac": 0.0, "probe_max_logit": -2.0, "evidence": 0.0, "crack_mm": 0.0, "z_max": 0.1,
        "anomalous_patch_frac": 0.0, "n_patches": 1000}


def _gate(view, cfg, probe=True):
    return SegmentGate(_stub(view, probe=probe), view, cfg)


def test_decision_rule(s21_view):
    cfg, _, view = s21_view
    gcfg = GateConfig(probe_frac=0.05, evidence=30.0, crack_mm=300.0, audit_frac=0.0)
    gate = _gate(view, gcfg)
    assert gate.decide(ZERO, "x")["decision"] == "skip"
    for k, at in (("probe_frac", 0.05), ("evidence", 30.0), ("crack_mm", 300.0)):
        d = gate.decide({**ZERO, k: at}, "x")                            # reaching the threshold passes (>=)
        assert (d["decision"], d["decision_if_gated"], d["reasons"]) == ("send", "send", [k])
        d = gate.decide({**ZERO, k: np.nextafter(at, -np.inf)}, "x")
        assert d["decision"] == "skip" and d["reasons"] == []
    d = gate.decide({**ZERO, "probe_frac": 0.2, "crack_mm": 500.0}, "x")
    assert d["reasons"] == ["probe_frac", "crack_mm"]
    d = gate.decide({**ZERO, "probe_frac": None, "probe_max_logit": None}, "x")   # no probe: never fires
    assert d["decision"] == "skip"
    assert set(d) == {"mode", "decision", "decision_if_gated", "reasons", "scores", "thresholds", "audit_draw"}
    assert d["thresholds"] == {"probe_frac": 0.05, "evidence": 30.0, "crack_mm": 300.0, "audit_frac": 0.0,
                               "source": "config", "probe_seed_logit": 0.3}
    assert d["scores"] == {**ZERO, "probe_frac": None, "probe_max_logit": None}
    with pytest.raises(KeyError):
        gate.decide({"probe_frac": 0.0}, "x")                           # incomplete scores are a bug
    with pytest.raises(ValueError):
        _gate(view, GateConfig(mode="some"))
    with pytest.raises(ValueError):
        _gate(view, GateConfig(audit_frac=1.5))


def test_mode_all_sends_everything_and_keeps_the_gate_decision(s21_view):
    cfg, _, view = s21_view
    gated = _gate(view, GateConfig(mode="gate", audit_frac=0.3))
    every = _gate(view, GateConfig(mode="all", audit_frac=0.3))
    seen = set()
    for k in range(300):
        rid = f"S_seg{k:05d}"
        sc = {**ZERO, "evidence": 40.0 if k % 7 == 0 else 0.0}
        a, b = gated.decide(sc, rid), every.decide(sc, rid)
        assert b["decision"] == "send" and b["mode"] == "all" and a["mode"] == "gate"
        assert b["decision_if_gated"] == a["decision_if_gated"] == a["decision"]
        assert b["reasons"] == a["reasons"] and b["audit_draw"] == a["audit_draw"]
        seen.add(a["decision"])
    assert seen == {"send", "audit", "skip"}


def test_audit_draw_is_sha256_of_the_id_and_hits_the_rate(s21_view):
    cfg, _, view = s21_view
    rid = "20261026T114907Z_SM-G991B-F0D1_293d_seg00042"
    h = hashlib.sha256(rid.encode("utf-8")).hexdigest()
    assert audit_draw(rid) == (int(h[:16], 16) >> 11) / 2**53 and 0.0 <= audit_draw(rid) < 1.0
    gate = _gate(view, GateConfig(audit_frac=0.1))
    assert gate.decide(ZERO, rid) == gate.decide(ZERO, rid)
    n = 20000
    dec = [gate.decide(ZERO, f"S_seg{k:05d}")["decision"] for k in range(n)]
    rate = dec.count("audit") / n
    assert abs(rate - 0.1) < 5 * math.sqrt(0.1 * 0.9 / n)                  # 5 sigma: 0.0106
    assert set(dec) == {"audit", "skip"}
    assert all(gate.decide({**ZERO, "evidence": 99.0}, f"S_seg{k:05d}")["decision"] == "send" for k in range(200))
    zero = _gate(view, GateConfig(audit_frac=0.0))
    assert all(zero.decide(ZERO, f"S_seg{k:05d}")["decision"] == "skip" for k in range(500))


# ==== calibration =================================================================================

def _row(y, probe_frac=0.0, evidence=0.0, crack_mm=0.0, **info):
    return {"scores": {"probe_frac": probe_frac, "evidence": evidence, "crack_mm": crack_mm,
                       "probe_max_logit": info.get("probe_max_logit", 0.0), "z_max": info.get("z_max", 0.0),
                       "anomalous_patch_frac": info.get("anomalous_patch_frac", 0.0), "n_patches": 100},
            "vlm_distress": y}


def test_need_is_exact_decimal():
    assert _need(0.9, 10) == 9 and _need(0.9, 11) == 10 and _need(1.0, 7) == 7 and _need(0.5, 3) == 2
    assert _need(0.28, 25) == 7 and math.ceil(0.28 * 25) == 8             # 0.28 x 25 = 7.000000000000001 in floats


def test_single_signal_by_hand():
    """Positives evidence 10..100, negatives 5..105 step 10: 90 % recall needs the 9 highest positives,
    threshold 20, which also passes the negatives 25..105 (9 of them)."""
    rows = [_row(True, evidence=10.0 * k) for k in range(1, 11)] + [_row(False, evidence=5.0 + 10 * k) for k in range(11)]
    cal = calibrate(rows, 0.9)
    s = cal["signals"]["evidence"]
    assert (s["threshold"], s["n_positive_pass"], s["n_pass"]) == (20.0, 9, 18)
    assert s["recall"] == 0.9 and s["pass_rate"] == 18 / 21 and s["status"] == "ok"
    curve = {round(p["threshold"], 6): p for p in s["curve"]}
    assert curve[20.0]["recall"] == 0.9 and curve[20.0]["pass_rate"] == 18 / 21
    assert curve[105.0]["n_pass"] == 1 and curve[105.0]["recall"] == 0.0 and len(s["curve"]) == 21
    assert cal["n_rows"] == 21 and cal["n_positive"] == 10 and cal["status"] == "ok"


def test_any_of_rule_by_hand():
    """5 positives found only by probe_frac (10..14), 5 only by evidence (10..14); a negative at
    probe_frac 10.5, one at evidence 9. Recall 0.9: probe_frac >= 11 (drops a positive and the
    negative), evidence >= 10, crack off: 9 rows pass, all positive. Recall 1.0: 11 rows."""
    rows = ([_row(True, probe_frac=10.0 + k) for k in range(5)] + [_row(True, evidence=10.0 + k) for k in range(5)]
            + [_row(False, probe_frac=10.5), _row(False, evidence=9.0)] + [_row(False) for _ in range(8)])
    a = calibrate(rows, 0.9)["any_of"]
    assert a["thresholds"] == {"probe_frac": 11.0, "evidence": 10.0, "crack_mm": None}
    assert (a["n_pass"], a["n_positive_pass"], a["recall"], a["pass_rate"]) == (9, 9, 0.9, 9 / 20)
    a = calibrate(rows, 1.0)["any_of"]
    assert a["thresholds"] == {"probe_frac": 10.0, "evidence": 10.0, "crack_mm": None}
    assert (a["n_pass"], a["recall"]) == (11, 1.0)
    curve = {c["target_recall"]: c for c in a["curve"]}
    assert curve[0.9]["n_pass"] == 9 and curve[1.0]["n_pass"] == 11 and curve[0.5]["n_pass"] == 5


def _brute(X, y, need):
    """Exhaustive: every threshold of every signal from ALL its distinct values, and off."""
    opts = [np.concatenate([[np.inf], np.unique(X[np.isfinite(X[:, j]), j])]) for j in range(X.shape[1])]
    best = None
    for t in itertools.product(*opts):
        p = (X >= np.array(t)).any(1)
        npos = int((p & y).sum())
        if npos >= need and (best is None or (int(p.sum()), -npos) < best):
            best = (int(p.sum()), -npos)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_any_of_rule_is_the_exhaustive_optimum(seed):
    rng = np.random.default_rng(seed)
    n = 26
    y = rng.random(n) < 0.4
    cols = [rng.integers(0, 6, n).astype(float),                  # heavy ties
            np.round(rng.gamma(2.0, 10.0, n) + 15 * y, 1),         # informative
            rng.uniform(0, 1, n)]
    cols[0][rng.random(n) < 0.2] = np.nan                         # probe missing on some rows
    rows = [{"scores": {"probe_frac": None if np.isnan(cols[0][i]) else float(cols[0][i]),
                        "evidence": float(cols[1][i]), "crack_mm": float(cols[2][i])}, "vlm_distress": bool(y[i])}
            for i in range(n)]
    X = np.column_stack([np.where(np.isnan(c), -np.inf, c) for c in cols])
    for target in (0.5, 0.75, 0.9, 1.0):
        cal = calibrate(rows, target)
        a = cal["any_of"]
        P = int(y.sum())
        assert (a["n_pass"], -a["n_positive_pass"]) == _brute(X, y, _need(target, P))
        th = np.array([np.inf if a["thresholds"][k] is None else a["thresholds"][k] for k in GATE_SIGNALS])
        passed = (X >= th).any(1)                                  # the thresholds give what is reported
        assert int(passed.sum()) == a["n_pass"] and int((passed & y).sum()) == a["n_positive_pass"]
        assert a["recall"] >= target and a["status"] == "ok"
        for j, k in enumerate(GATE_SIGNALS):                       # each signal alone, also exhaustive
            s = cal["signals"][k]
            reach = int((y & np.isfinite(X[:, j])).sum())          # positives this signal can pass at all
            assert (s["n_pass"], -s["n_positive_pass"]) == _brute(X[:, [j]], y, min(_need(target, P), reach))
            assert s["status"] == ("ok" if reach >= _need(target, P) else "target_not_reached")


def test_calibration_edge_cases():
    # No VLM positives: nothing to keep, no thresholds, and the gate refuses the file.
    cal = calibrate([_row(False, evidence=float(k)) for k in range(5)])
    assert cal["status"] == "no_positives" and cal["any_of"]["thresholds"] is None
    assert all(s["threshold"] is None for s in cal["signals"].values())
    assert cal["signals"]["evidence"]["curve"][0]["recall"] is None
    with pytest.raises(ValueError):
        calibration_thresholds(cal)
    # Every segment positive: the pass rate can't be below the recall.
    cal = calibrate([_row(True, evidence=float(k)) for k in range(1, 11)], 0.9)
    assert cal["status"] == "all_positive"
    assert (cal["any_of"]["n_pass"], cal["any_of"]["pass_rate"], cal["any_of"]["recall"]) == (9, 0.9, 0.9)
    # Ties at the threshold all pass: recall above the target.
    rows = [_row(True, evidence=5.0) for _ in range(9)] + [_row(True, evidence=1.0)] + \
        [_row(False, evidence=5.0), _row(False), _row(False)]
    s = calibrate(rows, 0.8)["signals"]["evidence"]
    assert (s["threshold"], s["n_positive_pass"], s["n_pass"], s["recall"]) == (5.0, 9, 10, 0.9)
    # A signal nobody computed (no probe anywhere): unavailable, off in the rule.
    rows = [_row(True, probe_frac=None, evidence=40.0), _row(True, probe_frac=None, crack_mm=500.0),
            _row(False, probe_frac=None, evidence=1.0)]
    cal = calibrate(rows, 1.0)
    assert cal["signals"]["probe_frac"]["available"] is False and cal["signals"]["probe_frac"]["threshold"] is None
    assert cal["any_of"]["available"] == ["evidence", "crack_mm"]
    assert cal["any_of"]["thresholds"] == {"probe_frac": None, "evidence": 40.0, "crack_mm": 500.0}
    # Positives no signal can pass (every score None): the best reachable recall, flagged.
    rows = [_row(True, evidence=float(10 + k)) for k in range(8)] + \
        [_row(True, probe_frac=None, evidence=None, crack_mm=None) for _ in range(2)] + [_row(False)] * 5
    a = calibrate(rows, 0.9)["any_of"]
    assert a["status"] == "target_not_reached" and a["recall"] == 0.8 and a["n_pass"] == 8
    # Bad input.
    with pytest.raises(ValueError):
        calibrate([])
    with pytest.raises(ValueError):
        calibrate([{"scores": {}, "vlm_distress": None}])
    with pytest.raises(ValueError):
        calibrate([_row(True, evidence=float("nan"))])
    with pytest.raises(ValueError):
        calibrate([_row(True)], 0.0)


def test_calibration_json_round_trip_and_use(tmp_path, s21_view):
    cfg, _, view = s21_view
    rng = np.random.default_rng(4)
    rows = [_row(bool(rng.random() < 0.3), probe_frac=float(rng.random() * 0.2), evidence=float(rng.gamma(2, 15)),
                 crack_mm=float(rng.gamma(2, 120)), z_max=float(rng.random() * 3)) for _ in range(60)]
    cal = calibrate(rows, 0.9)
    p, p2 = tmp_path / "gate_calibration.json", tmp_path / "again.json"
    write_calibration(p, cal)
    write_calibration(p2, calibrate(rows, 0.9))
    assert read_calibration(p) == cal and cal["schema"] == CALIBRATION_SCHEMA
    assert p.read_bytes() == p2.read_bytes()                               # deterministic file
    assert "VLM" in cal["caveat"] and cal["reference"] == "vlm"
    gate = _gate(view, GateConfig(calibration=str(p), audit_frac=0.0))
    th = gate.thresholds
    assert {k: th[k] for k in GATE_SIGNALS} == calibration_thresholds(cal) == cal["any_of"]["thresholds"]
    assert th["source"] == "calibration" and th["calibration_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    X = np.array([[r["scores"][k] for k in GATE_SIGNALS] for r in rows])
    sent = [gate.decide({**ZERO, **dict(zip(GATE_SIGNALS, x))}, "r")["decision"] == "send" for x in X]
    assert sum(sent) == cal["any_of"]["n_pass"]                           # the gate applies the calibrated rule
    with pytest.raises(ValueError):
        write_calibration(tmp_path / "x.json", {"schema": "other"})
    (tmp_path / "bad.json").write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_calibration(tmp_path / "bad.json")
    write_calibration(tmp_path / "none.json", calibrate([_row(False)] * 3))
    with pytest.raises(ValueError):
        _gate(view, GateConfig(calibration=str(tmp_path / "none.json")))


# ==== the whole check on rendered frames, with a real DINOv2 bank ==================================

def _camera_23deg(W=960, H=540, hfov_deg=67.0, pitch_deg=23.0, height_m=1.35) -> CameraModel:
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    th = math.radians(pitch_deg)
    R = vehicle_from_camera(np.array([0.0, -math.cos(th), -math.sin(th)]), "forward")
    return CameraModel(W, H, K, np.zeros(5), R, np.array([0.0, 0.0, height_m]), "sim")


def _render(road, cam: CameraModel, c: float) -> np.ndarray:
    """Frame of a camera whose nadir is at path position c (flat road, no rolling shutter)."""
    u, v = np.meshgrid(np.arange(cam.width, dtype=np.float64), np.arange(cam.height, dtype=np.float64))
    x, y, ok = cam.pixel_to_ground(u, v, 40.0)
    mx = np.where(ok, (x - road.lateral0_m) / road.res_m, -10).astype(np.float32)
    my = np.where(ok, (y + c - road.along0_m) / road.res_m, -10).astype(np.float32)
    g = cv2.remap(road.img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.75)
    g[~ok] = 0.8
    return to_rgb(g)


@needs_dino
def test_gate_scores_a_rendered_pothole_in_its_segment():
    rng = np.random.default_rng(8)
    road = asphalt(4.0, 46.0, 0.005, seed=6, lateral0_m=-2.0, along0_m=-2.0)
    draw_marking(road, 1.2, 0.12, dash_m=3.0, gap_m=4.5)
    draw_pothole(road, (0.2, 21.5), 0.2, rng)
    cam = _camera_23deg()
    cfg = load_config(overrides={"anomaly": {"probe": None}})
    near = _near(cam, cfg)
    view = plan_segment_view(cam, near, gate_far_m(near, cfg), cfg, False)
    # Coarse camera (~6 mm/px median): no downscale, whole patches (960 -> 952 columns).
    assert not view.notes["area_capped"] and view.notes["downscale_resolution"] == 1.0 and view.ana_w == 952
    feats = PatchFeatures(WEIGHTS, "cpu", tile_patches=32, overlap=4, layers=cfg.anomaly.layers)
    gate = SegmentGate(AnomalyDetector(feats, view.ana_h, view.ana_w, cfg.anomaly, cfg.cracks), view, cfg)
    for c in np.arange(0.0, 36.0, 1.5):
        gate.add_bank_frame(_render(road, cam, float(c)), float(c))
    gate.finalize()
    d0 = near + 0.5
    rows = _seg_rows(cam, d0, d0 + 5.0)
    c_pot = 21.5 - (d0 + 1.5)                    # the pothole 1.5 m into the segment
    pot = gate.score(_render(road, cam, c_pot), c_pot, rows)
    clean = gate.score(_render(road, cam, 6.0), 6.0, rows)
    assert pot["evidence"] >= cfg.anomaly.min_mass and pot["z_max"] >= cfg.anomaly.score_threshold, pot
    assert clean["evidence"] < pot["evidence"] / 3 and clean["z_max"] < pot["z_max"], (clean, pot)
    # The pothole in view but outside the segment, nearer than its near edge or beyond its far edge:
    # its detection is centred outside the segment's rows and is not counted.
    for ahead in (near + 0.1, d0 + 5.0 + 0.45):
        c = 21.5 - ahead
        out = gate.score(_render(road, cam, c), c, rows)
        assert out["evidence"] < pot["evidence"] / 3, (ahead, out, pot)
    assert pot["n_patches"] == clean["n_patches"] == gate.segment_patch_rows(rows).size * view.ana_w // PATCH
