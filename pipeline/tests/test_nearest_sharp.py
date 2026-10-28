"""segments.choice = nearest_sharp (docs/pipeline/13 §3): FairSharpness and the streaming choice.

FairSharpness is checked against an independent implementation (closed-form flat-road projection of a
zero-roll pinhole camera, numpy bilinear sampling, block means and a hand-written Sobel), and against
physics: the same road seen from different distances scores the same while the raw score grows, and
blur lowers it. The choice rule is checked on hand-made windows with known fair values and against a
brute-force evaluation of the rule over a whole drive.
"""
import json
import math

import cv2
import numpy as np
import pytest

from geoai.config import Config, SegmentConfig
from geoai.segments import (NEAREST_REASONS, BestFrameSelector, FairSharpness, SegmentGeometry, plan_offset_m,
                            plan_segments, sharpness)
from geoai.simtexture import to_rgb, value_noise
from geoai.util import to_json

from test_segments import PITCH, SMALL, _cam, _cf_row, _drive, _info, _long_drive, _segs, small  # noqa: F401


def _road(m, shift_m: float) -> np.ndarray:
    """Asphalt-like value noise on the ground (world y = ahead + shift_m), seen by camera m."""
    u, v = np.meshgrid(np.arange(m.width, dtype=np.float64), np.arange(m.height, dtype=np.float64))
    x, y, ok = m.pixel_to_ground(u, v)
    X, Y = np.where(ok, x, 0.0), np.where(ok, y + shift_m, 0.0)
    lum = np.full(X.shape, 0.42, np.float32)
    for cell, amp, k in ((0.8, 0.05, 1), (0.2, 0.03, 2), (0.05, 0.04, 3), (0.025, 0.05, 4)):
        lum += amp * (value_noise(X, Y, cell, 93 + k) - 0.5) * 2
    lum[~ok] = 0.8
    return to_rgb(np.clip(lum, 0, 1))


def _fair_reference(rgb, m, near, d0, grid_mm, k, strip_m, hw_cap):
    """FairSharpness written out: closed-form projection, bilinear sampling, k x k means, Tenengrad."""
    th = math.radians(PITCH)
    fx, fy, cx, cy = m.K[0, 0], m.K[1, 1], m.K[0, 2], m.K[1, 2]
    v_bot = m.height - 1.0

    def x_at(u, d):
        return (u - cx) / fx * (d * math.cos(th) + m.height_m * math.sin(th))

    near_bottom = m.height_m / math.tan(th + math.atan((v_bot - cy) / fy))
    hw = min(hw_cap, -x_at(0.0, near_bottom), x_at(m.width - 1.0, near_bottom)) - grid_mm / 1000
    n_cells = math.floor(2 * hw / (grid_mm / 1000))
    step = grid_mm / 1000 / k
    n_x, n_y = n_cells * k, k * round(strip_m / (grid_mm / 1000))
    i0 = round((d0 - near) / step)
    xs = (np.arange(n_x) - (n_x - 1) / 2) * step
    ys = near + (i0 + np.arange(n_y) + 0.5) * step
    X, Y = np.meshgrid(xs, ys)
    V = np.array([[_cf_row(y, m) for y in ys]]).T * np.ones_like(X)
    U = cx + fx * X / (Y * math.cos(th) + m.height_m * math.sin(th))
    r, g, b = (rgb[..., i].astype(np.int64) for i in range(3))
    grey = ((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14).astype(np.float64)  # OpenCV's 8-bit BT.601
    u0, v0 = np.floor(U).astype(int), np.floor(V).astype(int)
    a, c = U - u0, V - v0
    s = (grey[v0, u0] * (1 - a) * (1 - c) + grey[v0, u0 + 1] * a * (1 - c) + grey[v0 + 1, u0] * (1 - a) * c
         + grey[v0 + 1, u0 + 1] * a * c)
    cells = s.reshape(n_y // k, k, n_x // k, k).mean(axis=(1, 3))
    gx = (cells[:-2, 2:] + 2 * cells[1:-1, 2:] + cells[2:, 2:]) - (cells[:-2, :-2] + 2 * cells[1:-1, :-2] + cells[2:, :-2])
    gy = (cells[2:, :-2] + 2 * cells[2:, 1:-1] + cells[2:, 2:]) - (cells[:-2, :-2] + 2 * cells[:-2, 1:-1] + cells[:-2, 2:])
    return float(np.mean(gx * gx + gy * gy) / cells[1:-1, 1:-1].mean() ** 2), cells


def test_fair_sharpness_matches_an_independent_implementation():
    m = _cam(scale=0.25)
    g = SegmentGeometry.build(m, Config())
    sc = SegmentConfig(fair_grid_mm=16.0)
    fs = FairSharpness(g, g.near_m, sc)
    rgb = _road(m, 3.0)
    for d0 in (g.near_m, g.near_m + 0.37, g.near_m + 1.0):
        want, cells = _fair_reference(rgb, m, g.near_m, d0, 16.0, 4, 1.0, 1.5)
        got_cells = fs.grid(rgb, d0)[::-1]
        assert got_cells.shape == cells.shape
        # cv2.remap quantises bilinear weights to 1/32 px: grey values agree to a few hundredths of a level.
        assert np.abs(got_cells - cells).max() < 0.1
        assert fs(rgb, d0) == pytest.approx(want, rel=2e-3)


@pytest.mark.slow
def test_fair_sharpness_is_free_of_scale_and_sees_blur_full_resolution():
    """S21 at full resolution: the same road from the near and far end of the window."""
    m = _cam()
    g = SegmentGeometry.build(m, Config())
    fs = FairSharpness(g, g.near_m, SegmentConfig())
    seg_s0 = 10.0
    near_view, far_view = _road(m, seg_s0 - g.near_m), _road(m, seg_s0 - (g.near_m + 1.0))
    f_near, f_far = fs(near_view, g.near_m), fs(far_view, g.near_m + 1.0)
    raw = [sharpness(img, g.row_slice(*g.rows_for(d, d + 5.0)), cols=(0.2, 0.8), downscale=0.5)
           for img, d in ((near_view, g.near_m), (far_view, g.near_m + 1.0))]
    assert f_far == pytest.approx(f_near, rel=0.02)   # same road quality, same score
    assert raw[1] > 1.4 * raw[0]                       # the raw score prefers the farther frame
    for img, d, f in ((near_view, g.near_m, f_near), (far_view, g.near_m + 1.0, f_far)):
        drops = [fs(cv2.GaussianBlur(img, (0, 0), s), d) / f for s in (1.0, 2.0, 3.0)]
        assert 1.0 > drops[0] > drops[1] > drops[2] and drops[2] < 0.9


def test_fair_sharpness_blur_and_exposure_at_quarter_size():
    m = _cam(scale=0.25)
    g = SegmentGeometry.build(m, Config())
    fs = FairSharpness(g, g.near_m, SegmentConfig(fair_grid_mm=16.0))
    rgb = _road(m, 2.0)
    f = fs(rgb, g.near_m + 0.5)
    blurred = [fs(cv2.GaussianBlur(rgb, (0, 0), s), g.near_m + 0.5) for s in (0.5, 1.0, 2.0)]
    assert f > blurred[0] > blurred[1] > blurred[2]
    darker = np.round(rgb.astype(np.float64) * 0.5).astype(np.uint8)
    assert fs(darker, g.near_m + 0.5) == pytest.approx(f, rel=0.02)
    with pytest.raises(ValueError):
        fs(rgb[:-2], g.near_m)
    with pytest.raises(ValueError):
        FairSharpness(g, g.near_m, SegmentConfig(fair_supersample=0))


# ---- the choice rule with known fair values ---------------------------------------------------------

def _coded_images(m):
    """Uniform frames whose first pixel names their fair value (raw sharpness 0 for all)."""
    return [np.full((m.height, m.width, 3), code, np.uint8) for code in range(16)]


FAIR = {1: 1.0, 2: 0.95, 3: 0.92, 4: 0.85, 5: 0.8, 6: 0.89, 7: 0.5, 8: 0.9, 9: 2.0}


@pytest.fixture()
def coded_fair(monkeypatch):
    monkeypatch.setattr(FairSharpness, "__call__", lambda self, rgb, d0: FAIR[int(rgb[0, 0, 0])])


def test_nearest_sharp_takes_the_nearest_frame_within_the_tolerance(small, coded_fair):
    m, g, _ = small
    imgs = _coded_images(m)
    cfg = SegmentConfig(choice="nearest_sharp", max_blur_px=1.0)
    slow, low = {"exposure_ns": 8_000_000}, {"low_light": True}   # 8 ms at 10 m/s: ~2 px, over the limit
    specs = [  # frames far -> near: (code, extra); expected chosen index, its fair / best of its tier, reason
        ([(1, {}), (2, {}), (3, {})], 2, 0.92, "nearest_sharp"),             # all within 10 %: the nearest
        ([(1, {}), (2, {}), (4, {})], 1, 0.95, "nearest_sharp"),             # nearest 0.85 < 0.9
        ([(5, {}), (1, {}), (6, {})], 1, 1.0, "nearest_sharp"),              # 0.89 just below 0.9 x 1.0
        ([(1, {}), (7, {}), (2, {})], 2, 0.95, "nearest_sharp"),             # a blurred frame in between
        ([(8, {}), (2, {}), (1, {})], 2, 1.0, "nearest_sharp"),              # sharper as it nears
        ([(1, {}), (9, slow), (2, {})], 2, 0.95, "nearest_sharp"),           # over-blur frames never compete
        ([(1, slow), (7, slow)], 0, 1.0, "nearest_sharp_over_blur_limit"),   # only smeared frames
        ([(7, low), (1, low)], 1, 1.0, "only_low_light"),
        ([(9, slow), (1, {}), (7, {})], 1, 1.0, "nearest_sharp"),            # a better tier starts afresh
    ]
    segs = _segs(len(specs))
    sel = BestFrameSelector(segs, g, cfg)
    infos, code_of = [], {}
    t = 0.0
    for k, (frames, *_rest) in enumerate(specs):
        lo, hi = (float(x) for x in sel.windows[k])
        for j, (code, extra) in enumerate(frames):
            t += 0.03
            code_of[(k, j)] = code
            infos.append(_info(t, lo + (j + 0.5) / len(frames) * (hi - lo), (k, j), **extra))
    out, _, max_held, offered = _drive(sel, infos, lambda f: imgs[code_of[f.ref]])
    for (res, _), (frames, idx, rel, reason) in zip(out, specs):
        assert res.best.info.ref == (res.segment.k, idx) and res.rgb is offered[(res.segment.k, idx)]
        assert res.reason == reason and res.choice == "nearest_sharp" and res.fair_rel == pytest.approx(rel)
        assert res.n_candidates == len(frames) and 1 <= res.sharpness_rank <= len(frames)
        q = res.quality()
        to_json(q)
        assert q["choice"] == "nearest_sharp" and q["fair_sharpness"] == FAIR[code_of[res.best.info.ref]]
        assert q["fair_sharpness_rel"] == pytest.approx(rel)
    assert max_held <= 4   # the 3 kept frames of one window, and the previous window's frame until it is emitted


def test_nearest_sharp_settings_are_checked(small):
    _, g, _ = small
    with pytest.raises(ValueError, match="choice"):
        BestFrameSelector(_segs(1), g, SegmentConfig(choice="nearest"))
    with pytest.raises(ValueError, match="fair_rel_min"):
        BestFrameSelector(_segs(1), g, SegmentConfig(choice="nearest_sharp", fair_rel_min=0.0))
    res = BestFrameSelector(_segs(1), g, SegmentConfig(choice="nearest_sharp")).flush()[0]
    assert res.quality()["choice"] == "nearest_sharp" and res.quality()["fair_sharpness"] is None


# ---- the rule over a whole drive ---------------------------------------------------------------------

def test_nearest_sharp_long_drive_matches_brute_force(small):
    m, g, imgs = small
    # 320 x 180 frames: along-track pixels up to ~40 mm at the strip's far end, so a 40 mm grid.
    cfg = SegmentConfig(choice="nearest_sharp", max_blur_px=1.0, fair_grid_mm=40.0, fair_supersample=2)
    traj, infos = _long_drive()
    segs = plan_segments(traj, cfg, t_span=(infos[0].t, infos[-1].t), offset_m=plan_offset_m(g, cfg))
    sel = BestFrameSelector(segs, g, cfg)
    fs = FairSharpness(g, g.near_m, cfg)
    out, _, max_held, _ = _drive(sel, infos, lambda f: imgs[f.ref[1]])
    by_k = {r.segment.k: r for r, _ in out}
    checked = 0
    for seg in segs:
        hi = seg.s0 - g.near_m
        lo = max(hi - cfg.window_m, seg.s1 - g.far_m)
        cands = []
        for f in infos:
            if not lo <= f.c <= hi or not f.valid or f.black or f.reversing or not f.speed_mps >= cfg.min_speed_mps:
                continue
            r0, r1 = g.row_slice(*g.rows_for(seg.s0 - f.c, seg.s1 - f.c))
            e = f.exposure_ns / 1e9
            motion = f.speed_mps * e / (g.gsd_along_at((r0 + r1 - 1) / 2) / 1000.0)
            rot = abs(f.omega_perp_rad_s) * e * g.f_px
            blur = math.sqrt(motion * motion + (0.0 if math.isnan(rot) else rot * rot))
            tier = 2 * f.low_light + (blur > cfg.max_blur_px)
            cands.append((tier, f.c, fs(imgs[f.ref[1]], seg.s0 - f.c), f.ref))
        res = by_k[seg.k]
        if not cands:
            assert not res.covered
            continue
        best_tier = min(c[0] for c in cands)
        tier = [c for c in cands if c[0] == best_tier]
        top = max(c[2] for c in tier)
        ok = [c for c in tier if c[2] >= cfg.fair_rel_min * top]
        want = max(ok, key=lambda c: (c[1], tier.index(c)))      # nearest; at one position the later frame
        assert res.best.info.ref == want[3], seg
        assert res.reason == NEAREST_REASONS[best_tier] and res.fair_rel == pytest.approx(want[2] / top)
        assert res.n_candidates == len(cands)
        checked += 1
    assert checked > 100
    nearest = sum(1 for r, _ in out if r.best is not None and r.fair_rel == 1.0)
    assert 0 < nearest < checked   # the rule does both: keeps the best and passes over it for a nearer frame
    assert max_held <= 6


# ---- the record's optional quality fields -------------------------------------------------------------

def test_record_quality_with_the_choice_fields():
    from geoai.segment_records import build_segment_record, validate_segment_record
    from test_segment_records import record_args

    args = record_args(k=3)
    args["quality"] = {**args["quality"], "choice": "nearest_sharp", "fair_sharpness": 0.034, "fair_sharpness_rel": 0.93}
    rec = build_segment_record(**args)
    assert rec["quality"]["choice"] == "nearest_sharp" and rec["quality"]["fair_sharpness_rel"] == 0.93
    rec["frame"]["sha256"] = "ab" * 32
    assert validate_segment_record(rec) == []
    for mutate, expect in [(lambda q: q.__setitem__("choice", "sharpest"), "quality.choice"),
                           (lambda q: q.pop("fair_sharpness"), "fair_sharpness must be given"),
                           (lambda q: q.pop("choice"), "come only with quality.choice"),
                           (lambda q: q.__setitem__("fair_sharpness_rel", -1.0), "fair_sharpness_rel")]:
        bad = json.loads(json.dumps(rec))
        mutate(bad["quality"])
        assert any(expect in p for p in validate_segment_record(bad)), expect
    plain = build_segment_record(**record_args(k=3))
    assert set(plain["quality"]) == {"sharpness", "sharpness_rank", "n_candidates", "blur_motion_px",
                                     "blur_rotation_px", "blur_px", "gyro_rad_s", "reason"}
