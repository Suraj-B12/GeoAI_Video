"""Segment mode, part 2: the frame-level distress check on a segment (docs/pipeline/13 §4).

Every covered segment has one best full frame (geoai.segments). Before that frame goes to the VLM
(about 14 s each), the segment's part of it is checked on the GPU with the crop mode's three
detectors, all from one `AnomalyDetector.maps` call: the session bank (DINOv2 patch novelty u),
the Attain probe (absolute distress logit per patch) and the Hessian crack filter.

Analysis view (`plan_segment_view`). Full width, rows from the farthest road a candidate frame
needs (`gate_far_m`: near_m + length_m + window_m + FAR_MARGIN_M) down to the bottom usable row.
It is ONE view for the session: the bank compares patch rows across frames, which needs every bank
row to show the same strip of road in every frame; the segment's rows then move inside the view
from frame to frame. It is downscaled as in the crop mode (median along-track resolution to
`extract.target_gsd_mm`, never upscaled) and capped at `GateConfig.max_analysis_px`. On the S21 at
23 deg the window is 3840 x ~1750 px at ~2.3 mm/px median, so the cap sets the scale (~0.77), not
the resolution target. The window is fitted to whole patch rows by extending it upwards (instead
of letting `View.analysis_image` drop up to 13 rows at the bottom, which is the nearest, finest road).

Scores (`SegmentGate.score`, `restrict`). The segment [s0, s1) occupies display rows
(v_top, v_bottom] on the centre line (v_top = row of s1, the far edge; v_bottom = row of s0). A patch
belongs to the segment when its centre row, mapped to the full frame exactly as `View` maps analysis
pixels, lies in that interval; all columns count. Abutting segments therefore never share a patch.
A detection (bank or crack component) belongs to it when the centre row of its box does.

Decision (`SegmentGate.decide`). `send` when any of probe_frac, evidence, crack_mm reaches its
threshold; else `audit` for a deterministic `audit_frac` of segments (SHA-256 of the record id), so
the gate's misses can be measured without bias; else `skip`. Mode `all` sends everything and keeps
the gate's would-be decision.

Calibration (`calibrate`). With the VLM as the reference (no labels by us), it finds the thresholds
that keep `target_recall` of the VLM-positive segments at the lowest pass rate, per signal and for
the any-of rule, exactly (see `_solve_any_of`), and writes them with their curves to
gate_calibration.json, which `GateConfig.calibration` loads.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__
from .config import Config, GateConfig
from .dinov2 import PATCH
from .extract import ExtractionError, View
from .geometry import CameraModel
from .util import log, sha256_file, write_json_atomic

# Road beyond the farthest segment far edge a candidate frame can need, kept in the view: the
# camera position is that of the centre row, and a frame can lie a fraction of a frame of travel
# outside its window.
FAR_MARGIN_M = 0.5
GATE_SIGNALS = ("probe_frac", "evidence", "crack_mm")             # the any-of rule
INFO_SIGNALS = ("probe_max_logit", "z_max", "anomalous_patch_frac")  # calibrated alone, for information
SCORE_KEYS = ("probe_frac", "probe_max_logit", "evidence", "crack_mm", "z_max", "anomalous_patch_frac",
              "n_patches")
MODES = ("gate", "all")
DECISIONS = ("send", "audit", "skip")
CALIBRATION_FILE = "gate_calibration.json"
CALIBRATION_SCHEMA = "geoai.gate_calibration.v1"
# Recall levels at which the any-of rule is also solved, for the trade-off curve in the calibration file.
CURVE_RECALLS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
CALIBRATION_CAVEAT = ("The reference is the VLM (distress = any valid non-Normal label), not labels made by people: "
                      "recall and pass rate are measured against the VLM's answers, including its mistakes.")
_MAX_BLOCK = 1 << 22  # grid points x rows per vectorised block of the threshold search (~20 MB of int32)


# ---- analysis view ------------------------------------------------------------------------------

def gate_far_m(near_m: float, cfg: Config) -> float:
    """Farthest road ahead (centre line) the gate view must show: docs/pipeline/13 §4."""
    return near_m + cfg.segments.length_m + cfg.segments.window_m + FAR_MARGIN_M


def plan_segment_view(model: CameraModel, near_m: float, far_needed_m: float, cfg: Config,
                      rotate180: bool) -> View:
    """The gate's analysis view of a full frame (display orientation), see the module docstring.

    Rows [v0, v1): v1 - 1 is the bottom usable row (GeometryConfig.bottom_exclude_frac, as in
    segments.SegmentGeometry); v0 is at or above the row of `far_needed_m` on the centre line (the
    top row when that road is above the image). Scale: ds = min(1, max(0.25, median gsd / target))
    as in extract.plan_view, the median taken over the window's rows at the centre column; lowered
    to sqrt(max_analysis_px / window area) when the area would exceed the cap. The analysis image is
    then whole patches: gw = floor(W ds / 14) columns, gh = ceil(rows x ds / 14) rows, with the
    window extended upwards to exactly gh patch rows (or clipped at the top of the frame).
    """
    if not (math.isfinite(near_m) and near_m > 0):
        raise ExtractionError(f"near_m must be a positive distance ahead, got {near_m}")
    if not (math.isfinite(far_needed_m) and far_needed_m > near_m):
        raise ExtractionError(f"far_needed_m ({far_needed_m}) must lie beyond near_m ({near_m})")
    W, H = model.width, model.height
    cap = int(cfg.gate.max_analysis_px)
    if cap < 16 * PATCH * PATCH:
        raise ExtractionError(f"gate.max_analysis_px {cap} is below 4 x 4 patches")
    v_use = int(math.floor(H * (1.0 - cfg.geometry.bottom_exclude_frac) - 1))
    if v_use < 1:
        raise ExtractionError(f"bottom_exclude_frac {cfg.geometry.bottom_exclude_frac} leaves no usable rows")
    uc = (W - 1) / 2
    _, y_bottom, ok_bottom = model.pixel_to_ground(np.array([uc]), np.array([float(v_use)]))
    if not ok_bottom[0] or y_bottom[0] <= 0:
        raise ExtractionError("the gate needs a forward view of the road at the bottom image row")
    _, vf, okf = model.ground_to_pixel(np.array([0.0]), np.array([float(far_needed_m)]))
    v_far = float(vf[0]) if okf[0] else -math.inf
    if v_far >= v_use:
        raise ExtractionError(f"road {far_needed_m:.2f} m ahead is not above the bottom usable row")
    v1 = v_use + 1
    v_needed = max(0, int(math.floor(v_far)))            # the row containing the far point, or the top row
    rows_needed = v1 - v_needed

    vs = np.arange(v_needed, v1, 2, dtype=np.float64)
    along, _ = model.gsd_mm(np.full_like(vs, uc), vs)
    along = along[np.isfinite(along)]
    if along.size == 0:
        raise ExtractionError("no row of the gate window sees the road")
    med_gsd = float(np.median(along))
    ds_res = min(1.0, max(0.25, med_gsd / cfg.extract.target_gsd_mm))
    area = W * rows_needed
    capped = area * ds_res * ds_res > cap
    ds = math.sqrt(cap / area) if capped else ds_res
    gw = int(math.floor(W * ds / PATCH))
    gh = int(math.ceil(rows_needed * ds / PATCH))
    while gw * gh * PATCH * PATCH > cap:   # the rounded-up rows can add at most a few patch columns' worth
        gw -= 1
    if gw < 4 or gh < 4:
        raise ExtractionError(f"gate analysis view too small ({gw} x {gh} patches)")
    sx = gw * PATCH / W
    rows = max(rows_needed, int(round(gh * PATCH / sx)))  # frame rows of gh patch rows at the same scale
    if rows > v1:                                       # the frame's top: whole patch rows of the frame
        rows = v1
        gh = int(math.floor(rows * sx / PATCH))
    v0 = v1 - rows
    sy = gh * PATCH / rows
    ana_w, ana_h = gw * PATCH, gh * PATCH
    model_ana = model.cropped(0, v0, W, rows).scaled(sx, sy)
    model_ana = CameraModel(ana_w, ana_h, model_ana.K, model_ana.dist, model_ana.R_v_from_c, model_ana.C_v,
                            model_ana.source, model_ana.notes)
    rr = np.arange(ana_h, dtype=np.float64)
    gsd_rows, _ = model_ana.gsd_mm(np.full_like(rr, (ana_w - 1) / 2), rr)
    gsd_rows = np.where(np.isfinite(gsd_rows), gsd_rows, np.nanmax(gsd_rows))
    notes = {"forward_view": True, "near_m": near_m, "far_needed_m": far_needed_m,
             "far_row": v_far if math.isfinite(v_far) else None, "far_row_clipped": v_far < 0,
             "bottom_row": v_use, "median_gsd_mm": med_gsd, "downscale_resolution": ds_res, "downscale": sx,
             "area_capped": capped, "max_analysis_px": cap, "analysis_px": ana_w * ana_h}
    return View(W, H, model, v0, v1, sx, sy, ana_w, ana_h, model_ana, near_m, far_needed_m - near_m,
                gsd_rows, rotate180, notes)


# ---- decision -----------------------------------------------------------------------------------

def audit_draw(record_id: str) -> float:
    """Deterministic uniform draw in [0, 1) for a record: the first 53 bits of SHA-256(id) / 2^53.

    53 bits are exactly representable in a double, so the draw is never rounded up to 1.0.
    """
    h = hashlib.sha256(record_id.encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "big") >> 11) / float(1 << 53)


def _gate_cfg(cfg: Config | GateConfig) -> GateConfig:
    return cfg.gate if isinstance(cfg, Config) else cfg


def gate_thresholds(cfg: Config | GateConfig) -> dict:
    """The thresholds in force (None = signal off) and where they come from.

    With `GateConfig.calibration` set, the any-of thresholds of that file replace the configured
    ones; the file's SHA-256 is kept so every record says which calibration decided it.
    """
    g = _gate_cfg(cfg)
    if not 0.0 <= g.audit_frac <= 1.0:
        raise ValueError(f"gate.audit_frac must be in [0, 1], got {g.audit_frac}")
    if g.calibration:
        path = Path(g.calibration)
        cal = read_calibration(path)
        if cal["any_of"]["status"] == "target_not_reached":
            log.warning("gate calibration %s reaches recall %.3f only (target %.3f)", path,
                        cal["any_of"]["recall"], cal["target_recall"])
        return {**calibration_thresholds(cal), "audit_frac": float(g.audit_frac), "source": "calibration",
                "calibration_file": path.name, "calibration_sha256": sha256_file(path)}
    return {"probe_frac": float(g.probe_frac), "evidence": float(g.evidence), "crack_mm": float(g.crack_mm),
            "audit_frac": float(g.audit_frac), "source": "config"}


class SegmentGate:
    """The distress check of one recording size: a detector whose bank is built on `view`.

    detector: an AnomalyDetector created for the view's analysis size,
        AnomalyDetector(features, view.ana_h, view.ana_w, cfg.anomaly, cfg.cracks).
    Bank frames go through `add_bank_frame` (full frames; the view crops and scales them), then
    `finalize`. Per segment: `score` the best frame, then `decide`.
    """

    def __init__(self, detector, view: View, cfg: Config | GateConfig):
        g = _gate_cfg(cfg)
        if g.mode not in MODES:
            raise ValueError(f"gate.mode must be one of {MODES}, got {g.mode!r}")
        if (detector.gh, detector.gw) != (view.ana_h // PATCH, view.ana_w // PATCH):
            raise ValueError(f"detector grid {detector.gh} x {detector.gw} does not match the view's "
                             f"{view.ana_h // PATCH} x {view.ana_w // PATCH} patches")
        self.detector = detector
        self.view = view
        self.mode = g.mode
        probe = detector.probe
        self.thresholds = {**gate_thresholds(g), "probe_seed_logit": None if probe is None else float(probe.seed_logit)}
        if probe is None and self.thresholds["probe_frac"] is not None:
            log.warning("gate: no Attain probe loaded; the probe_frac threshold can never be reached")
        # Centre of each patch row in full-frame display rows (pixel centres at integers), by View's
        # own mapping: analysis rows [14p, 14p + 14) are frame rows v0 + [14p, 14p + 14) / sy.
        self.patch_rows_v = self.frame_row(PATCH * np.arange(detector.gh), PATCH * np.arange(1, detector.gh + 1))

    # -- bank ------------------------------------------------------------------------------------
    def add_bank_frame(self, rgb_full: np.ndarray, s_m: float) -> None:
        """One bank frame (full resolution, display orientation) at path position s_m."""
        self._check_frame(rgb_full)
        self.detector.add_bank_frame(self.view.analysis_image(rgb_full), s_m, self.view.gsd_rows_mm)

    def finalize(self) -> None:
        self.detector.finalize()

    # -- geometry --------------------------------------------------------------------------------
    def frame_row(self, y0, y1):
        """Full-frame display row of the centre of analysis rows [y0, y1) (pixel edges)."""
        return self.view.v0 + (np.asarray(y0, dtype=np.float64) + np.asarray(y1, dtype=np.float64)) / (2.0 * self.view.sy) - 0.5

    def _check_rows(self, seg_rows_full: Sequence[float]) -> tuple[float, float]:
        v_top, v_bottom = (float(v) for v in seg_rows_full)
        if not (math.isfinite(v_top) and math.isfinite(v_bottom) and v_top < v_bottom):
            raise ValueError(f"segment rows must be finite with v_top < v_bottom, got {seg_rows_full}")
        if v_top < self.view.v0 - 0.5:
            raise ValueError(f"segment rows {v_top:.1f}..{v_bottom:.1f} reach above the gate view (top row "
                             f"{self.view.v0}): plan the view with gate_far_m")
        return v_top, v_bottom

    def segment_patch_rows(self, seg_rows_full: Sequence[float]) -> np.ndarray:
        """Patch rows whose centre lies in the segment's rows (v_top, v_bottom], ascending."""
        return self._patch_rows(*self._check_rows(seg_rows_full))

    def _patch_rows(self, v_top: float, v_bottom: float) -> np.ndarray:
        rows = np.flatnonzero((self.patch_rows_v > v_top) & (self.patch_rows_v <= v_bottom))
        if rows.size == 0:
            raise ValueError(f"segment rows {v_top:.1f}..{v_bottom:.1f} contain no patch row of the gate view")
        return rows

    def _check_frame(self, rgb_full: np.ndarray) -> None:
        if rgb_full.shape[:2] != (self.view.height, self.view.width):
            raise ValueError(f"frame {rgb_full.shape[1]}x{rgb_full.shape[0]} is not the view's "
                             f"{self.view.width}x{self.view.height}")

    # -- scores and decision ---------------------------------------------------------------------
    def score(self, rgb_full: np.ndarray, s_m: float, seg_rows_full: Sequence[float]) -> dict:
        """Scores of one segment in its best frame (full resolution, display orientation).

        s_m: the frame's camera position on the route (bank patches within exclude_window_m of it
        are ignored); seg_rows_full: (v_top, v_bottom) of the segment, display rows.
        """
        self._check_frame(rgb_full)
        maps = self.detector.maps(self.view.analysis_image(rgb_full), s_m, self.view.gsd_rows_mm)
        return self.restrict(maps, seg_rows_full)

    def restrict(self, maps: dict, seg_rows_full: Sequence[float]) -> dict:
        """The segment's scores from the detector maps of its frame (`AnomalyDetector.maps`).

        probe_frac           share of the segment's patches with probe logit >= the probe's seed level
                             (paint patches never are); None without a probe
        probe_max_logit      highest probe logit among them; None without a probe
        evidence             summed evidence of bank and crack candidates centred in the segment (the
                             crop mode's ranking scale; crack candidates only with as_candidates)
        crack_mm             summed length of the crack filter's components centred in the segment
        z_max                highest bank score u among the segment's patches (0 on paint)
        anomalous_patch_frac share of them with u >= score_threshold
        n_patches            patches in the segment's rows (all columns)
        """
        v_top, v_bottom = self._check_rows(seg_rows_full)
        rows = self._patch_rows(v_top, v_bottom)
        det = self.detector
        z = np.asarray(maps["z"])[rows]
        n = int(z.size)
        logit = maps["probe_logit"]
        if logit is None:
            probe_frac = probe_max = None
        else:
            lg = np.asarray(logit)[rows]
            probe_frac = int(np.count_nonzero(lg >= det.probe.seed_logit)) / n
            probe_max = float(lg.max())

        def inside(c) -> bool:
            v = float(self.frame_row(c.bbox[1], c.bbox[3]))
            return v_top < v <= v_bottom

        comps = maps["components"]
        ev = list(comps["dino"]) + (list(comps["crack"]) if det.crack_cfg.as_candidates else [])
        return {"probe_frac": probe_frac, "probe_max_logit": probe_max,
                "evidence": math.fsum(float(c.score) for c in ev if inside(c)),
                "crack_mm": math.fsum(float(c.crack_length_mm) for c in comps["crack"] if inside(c)),
                "z_max": float(z.max()),
                "anomalous_patch_frac": int(np.count_nonzero(z >= det.cfg.score_threshold)) / n,
                "n_patches": n}

    def decide(self, scores: dict, record_id: str) -> dict:
        """The record's `gate` block: mode, decision, decision_if_gated, reasons, scores, thresholds, audit_draw.

        reasons: the signals that reached their thresholds (decision_if_gated = send), ["audit"]
        (audit), or [] (skip). A signal whose threshold or score is None never fires.
        """
        th = self.thresholds
        reasons = [k for k in GATE_SIGNALS if th[k] is not None and scores[k] is not None and scores[k] >= th[k]]
        draw = audit_draw(record_id)
        if reasons:
            gated = "send"
        elif draw < th["audit_frac"]:
            gated, reasons = "audit", ["audit"]
        else:
            gated = "skip"
        return {"mode": self.mode, "decision": "send" if self.mode == "all" else gated,
                "decision_if_gated": gated, "reasons": reasons, "scores": {k: scores[k] for k in SCORE_KEYS},
                "thresholds": dict(th), "audit_draw": draw}


# ---- calibration --------------------------------------------------------------------------------

def _need(target_recall: float, n_pos: int) -> int:
    """Smallest count k with k / n_pos >= target, in exact decimal arithmetic (0.9 x 10 -> 9, not 10)."""
    return math.ceil(Fraction(repr(float(target_recall))) * n_pos)


def _column(rows: list[dict], key: str) -> np.ndarray:
    """Values of one score over the rows; -inf where the score is None (it never passes a threshold)."""
    out = np.full(len(rows), -np.inf)
    for i, r in enumerate(rows):
        v = r["scores"].get(key)
        if v is None:
            continue
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"calibration row {i}: score {key} = {v} is not finite")
        out[i] = v
    return out


def _candidates(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Thresholds worth trying for one signal of the any-of rule: +inf (off), then each distinct value
    of a positive row, descending. Any other threshold can be raised to the next of these without
    losing a positive and without passing another row."""
    vals = np.unique(x[y & np.isfinite(x)])[::-1]
    return np.concatenate([[np.inf], vals])


def _solve_any_of(X: np.ndarray, y: np.ndarray, needs: Sequence[int]) -> list[tuple[np.ndarray, int, int] | None]:
    """Exact minimum-pass thresholds of the rule "pass if X[:, j] >= t_j for any j".

    X: (N, s) signal values, -inf where unavailable (never passes); y: (N,) VLM-positive.
    For each required number of positives K in `needs`: (t, n_pass, n_pos_pass) with t the (s,)
    thresholds (+inf = signal off) passing at least K positives with the fewest rows, ties broken
    by more positives, then by the first in the search order (strictest thresholds first); None
    when no thresholds pass K positives.

    Search: every combination of `_candidates` for all signals but one (c, the one with the most
    distinct positive values), and for each the exact best threshold of c: with the rows the others
    pass fixed, c must pass the K' = K - (positives already passed) highest-c positives among the
    rest, so t_c is the K'-th largest of their c values (all rows tied with it pass too). The grid
    of the other signals is evaluated a block at a time with cumulative counts in c order:
    O(prod |candidates| x N) operations, exact.
    """
    N, s = X.shape
    P = int(y.sum())
    n_distinct = [len(_candidates(X[:, j], y)) for j in range(s)]
    c = int(np.argmax(n_distinct))
    grid = [j for j in range(s) if j != c]
    order = np.argsort(-X[:, c], kind="stable")      # c descending; -inf (unavailable) last
    Cs, ys, Xo = X[order, c], y[order], X[order]
    tie_end = np.searchsorted(-Cs, -Cs, side="right") - 1    # last position of each value's tie group
    pf = np.flatnonzero(ys & np.isfinite(Cs))                  # positives c can pass, in c order
    pos_upto = np.searchsorted(pf, np.arange(N), side="right")  # such positives at positions <= i
    cands = [_candidates(X[:, j], y) for j in grid]
    inner = cands[-1] if grid else np.array([np.inf])
    block = max(1, _MAX_BLOCK // max(N, 1))
    best: list[tuple | None] = [None] * len(needs)
    serial = 0
    for outer in itertools.product(*cands[:-1]):
        base = np.ones(N, bool)
        for j, t in zip(grid[:-1], outer):
            base &= Xo[:, j] < t
        for b0 in range(0, len(inner), block):
            T = inner[b0:b0 + block]
            U = base[None, :] & (Xo[None, :, grid[-1]] < T[:, None]) if grid else base[None, :]
            G = U.shape[0]
            covered = N - U.sum(1)
            covered_pos = P - (U & ys[None, :]).sum(1)
            cum_all = np.cumsum(U, axis=1, dtype=np.int32)
            cum_pos = np.cumsum(U[:, pf], axis=1, dtype=np.int32) if pf.size else np.zeros((G, 0), np.int32)
            avail = cum_pos[:, -1] if pf.size else np.zeros(G, np.int32)
            gi = np.arange(G)
            for q, K in enumerate(needs):
                kp = K - covered_pos
                off = kp <= 0
                feasible = off | (avail >= kp)
                if not feasible.any():
                    continue
                if pf.size:
                    j = np.argmax(cum_pos >= np.maximum(kp, 1)[:, None], axis=1)
                    idx = pf[j]
                    e = tie_end[idx]
                    tc = np.where(off, np.inf, Cs[idx])
                    n_pass = np.where(off, covered, covered + cum_all[gi, e])
                    n_pos = np.where(off, covered_pos, covered_pos + cum_pos[gi, pos_upto[e] - 1])
                else:
                    tc = np.full(G, np.inf)
                    n_pass, n_pos = covered, covered_pos
                cand_idx = np.flatnonzero(feasible)
                k = cand_idx[np.lexsort((cand_idx, -n_pos[cand_idx], n_pass[cand_idx]))[0]]
                key = (int(n_pass[k]), -int(n_pos[k]), serial + int(k))
                if best[q] is None or key < best[q][0]:
                    t = np.empty(s)
                    t[c] = tc[k]
                    for jj, tv in zip(grid, (*outer, T[k]) if grid else ()):
                        t[jj] = tv
                    best[q] = (key, t, int(n_pass[k]), int(n_pos[k]))
            serial += G
    return [None if b is None else (b[1], b[2], b[3]) for b in best]


def _thr(v: float) -> float | None:
    return None if not math.isfinite(v) else float(v)


def _curve(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Recall and pass rate of "x >= t" at every distinct value t of x, strictest first."""
    N, P = len(x), int(y.sum())
    fin = np.isfinite(x)
    xs = np.sort(x[fin])
    xp = np.sort(x[fin & y])
    out = []
    for t in np.unique(xs)[::-1]:
        n_pass = int(xs.size - np.searchsorted(xs, t, side="left"))
        n_pos = int(xp.size - np.searchsorted(xp, t, side="left"))
        out.append({"threshold": float(t), "recall": n_pos / P if P else None, "pass_rate": n_pass / N,
                    "n_pass": n_pass, "n_positive_pass": n_pos})
    return out


def _result(sol: tuple[np.ndarray, int, int] | None, names: Sequence[str], N: int, P: int) -> dict:
    if sol is None:
        return {"thresholds": None, "n_pass": None, "pass_rate": None, "n_positive_pass": None, "recall": None}
    t, n_pass, n_pos = sol
    return {"thresholds": {n: _thr(v) for n, v in zip(names, t)}, "n_pass": n_pass, "pass_rate": n_pass / N,
            "n_positive_pass": n_pos, "recall": n_pos / P if P else None}


def calibrate(rows: list[dict], target_recall: float = 0.9) -> dict:
    """Gate thresholds from segments the VLM answered (docs/pipeline/13 §4); see module docstring.

    rows: [{"scores": {...SCORE_KEYS...}, "vlm_distress": bool}], vlm_distress = the VLM gave any
    valid non-Normal label. Returns the content of gate_calibration.json:
      signals[name]  each signal alone (GATE_SIGNALS and INFO_SIGNALS): the threshold keeping
                     target_recall of the positives at the lowest pass rate, and its full curve
      any_of         the gate's rule over GATE_SIGNALS, solved jointly and exactly; its curve is the
                     same optimum at each recall of CURVE_RECALLS (and the target)
    Cases: no positives -> status "no_positives", no thresholds (nothing to keep; such a file can't
    be used by the gate). Every row positive -> "all_positive": thresholds as usual, but the pass
    rate can't be below the recall. Positives no signal can pass (scores None) -> the rule's status
    "target_not_reached", with the thresholds of the highest reachable recall. Ties at a threshold
    all pass (>=), so the recall can exceed the target.
    """
    if not rows:
        raise ValueError("calibration needs at least one answered segment")
    if not 0.0 < float(target_recall) <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    y = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        v = r.get("vlm_distress")
        if not isinstance(v, (bool, np.bool_)):
            raise ValueError(f"calibration row {i}: vlm_distress must be true or false, got {v!r}")
        y[i] = bool(v)
    N, P = len(rows), int(y.sum())
    cols = {k: _column(rows, k) for k in GATE_SIGNALS + INFO_SIGNALS}
    status = "no_positives" if P == 0 else ("all_positive" if P == N else "ok")
    need = _need(target_recall, P)

    signals = {}
    for k, x in cols.items():
        available = bool(np.isfinite(x).any())
        entry = {"available": available, "curve": _curve(x, y)}
        if P and available:
            reach = int((y & np.isfinite(x)).sum())
            sol = _solve_any_of(x[:, None], y, [min(need, reach)])[0]
            res = _result(sol, ["threshold"], N, P)
            entry.update({"threshold": res["thresholds"]["threshold"],
                          **{f: res[f] for f in ("n_pass", "pass_rate", "n_positive_pass", "recall")},
                          "status": "ok" if reach >= need else "target_not_reached"})
        else:
            entry.update({"threshold": None, "n_pass": None, "pass_rate": None, "n_positive_pass": None,
                          "recall": None, "status": "no_positives" if not P else "unavailable"})
        signals[k] = entry

    avail = [k for k in GATE_SIGNALS if signals[k]["available"]]
    any_of = {"signals": list(GATE_SIGNALS), "available": avail}
    if P and avail:
        X = np.column_stack([cols[k] for k in avail])
        reach = int((y & np.isfinite(X).any(1)).sum())
        targets = sorted(set(CURVE_RECALLS) | {float(target_recall)})
        needs = [min(need, reach)] + [min(_need(t, P), reach) for t in targets]
        sols = _solve_any_of(X, y, needs)

        def full(sol):
            res = _result(sol, avail, N, P)
            res["thresholds"] = {k: (res["thresholds"] or {}).get(k) for k in GATE_SIGNALS}
            return res

        any_of.update(full(sols[0]))
        any_of["status"] = "ok" if reach >= need else "target_not_reached"
        any_of["curve"] = [{"target_recall": t, "reached": _need(t, P) <= reach, **full(sol)}
                           for t, sol in zip(targets, sols[1:])]
    else:
        any_of.update({"thresholds": None, "n_pass": None, "pass_rate": None, "n_positive_pass": None,
                       "recall": None, "status": "no_positives" if not P else "unavailable", "curve": []})
    return {"schema": CALIBRATION_SCHEMA, "pipeline_version": __version__, "reference": "vlm",
            "caveat": CALIBRATION_CAVEAT, "target_recall": float(target_recall), "status": status,
            "n_rows": N, "n_positive": P, "positive_rate": P / N, "rule": "send if any of "
            + ", ".join(f"{k} >= threshold" for k in GATE_SIGNALS), "signals": signals, "any_of": any_of}


def write_calibration(path: Path, cal: dict) -> None:
    """gate_calibration.json (atomic write; byte-identical for the same input)."""
    if cal.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"not a gate calibration (schema {cal.get('schema')!r})")
    write_json_atomic(Path(path), cal)


def read_calibration(path: Path) -> dict:
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    if cal.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"{path}: not a gate calibration (schema {cal.get('schema')!r})")
    return cal


def calibration_thresholds(cal: dict) -> dict:
    """The any-of thresholds of a calibration ({signal: value or None = off})."""
    th = cal["any_of"]["thresholds"]
    if th is None:
        raise ValueError(f"gate calibration has no thresholds (status {cal['any_of']['status']!r}: "
                         f"{cal['n_positive']} VLM-positive of {cal['n_rows']} segments)")
    return {k: (None if th[k] is None else float(th[k])) for k in GATE_SIGNALS}
