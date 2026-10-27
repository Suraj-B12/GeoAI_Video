"""Segment planning, view geometry, frame quality and the streaming best-frame selector (docs/pipeline/13 §2-3).

Expected values come from independent calculations: flat-road closed forms for a camera with pitch
theta, height h and intrinsics K (no roll, no distortion), hand-computed blur, a numpy Tenengrad, the
physical gyro axes, and a brute-force selection over all frames of a drive.
"""
import math
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon

from geoai.config import Config, GeometryConfig, SegmentConfig
from geoai.geometry import CameraModel, cam_from_device, nominal_model, vehicle_from_camera
from geoai.segments import (CHOSEN_REASONS, EXCLUSION_REASONS, FAR_CAP_M, BestFrameSelector, FrameInfo,
                            GyroImageRates, Segment, SegmentGeometry, candidate_window, plan_offset_m,
                            plan_segments, predicted_blur, sharpness)
from geoai.simtexture import to_rgb, value_noise
from geoai.trajectory import LatencyEstimate, Trajectory
from geoai.util import to_json

# S21 pilot camera (field/runs/.../report.json): 3840 x 2160, pitch 23 deg, height 1.35 m.
FX, FY, CX, CY = 2910.287, 2908.411, 1933.546, 1096.881
PITCH, HEIGHT = 23.0, 1.35
SMALL = 1 / 12                     # 320 x 180 frames for the selector tests
T0_NS = 5_000_000_000_000


def _cam(pitch=PITCH, roll=0.0, h=HEIGHT, scale=1.0, facing="forward") -> CameraModel:
    """Pinhole camera, S21 intrinsics scaled by `scale` (pixel-centre convention of CameraModel.scaled)."""
    W, H = int(round(3840 * scale)), int(round(2160 * scale))
    K = np.array([[FX * scale, 0, (CX + 0.5) * scale - 0.5], [0, FY * scale, (CY + 0.5) * scale - 0.5], [0, 0, 1.0]])
    th, ph = math.radians(pitch), math.radians(roll)
    up_cam = np.array([math.cos(th) * math.sin(ph), -math.cos(th) * math.cos(ph), -math.sin(th)])
    return CameraModel(W, H, K, np.zeros(5), vehicle_from_camera(up_cam, facing), np.array([0.0, 0.0, h]))


# ---- closed forms: flat road, zero roll, no distortion ----------------------------------------------
# Row v looks at phi = theta + atan((v - cy) / fy) below the horizontal, whatever the column, and meets
# the road at y = h / tan(phi). Column u at distance d is at x = (u - cx) / fx x (d cos theta + h sin theta).

def _cf_ahead(v, m, pitch=PITCH):
    phi = math.radians(pitch) + math.atan((v - m.K[1, 2]) / m.K[1, 1])
    return m.height_m / math.tan(phi) if phi > 0 else math.inf


def _cf_row(d, m, pitch=PITCH):
    return m.K[1, 2] + m.K[1, 1] * math.tan(math.atan2(m.height_m, d) - math.radians(pitch))


def _cf_gsd_mm(v, m, pitch=PITCH):
    yn = (v - m.K[1, 2]) / m.K[1, 1]
    phi = math.radians(pitch) + math.atan(yn)
    return 1000.0 * m.height_m / math.sin(phi) ** 2 / (m.K[1, 1] * (1 + yn * yn))


def _cf_x(u, d, m, pitch=PITCH):
    th = math.radians(pitch)
    return (u - m.K[0, 2]) / m.K[0, 0] * (d * math.cos(th) + m.height_m * math.sin(th))


def _cf_row_slice(d0, d1, m, v_usable):
    v_top, v_bottom = _cf_row(d1, m), _cf_row(d0, m)
    r0 = max(0, math.ceil(v_top))
    return r0, max(r0, min(v_usable + 1, math.floor(v_bottom) + 1))


def _traj(t, s, *, valid=None, speed=None) -> Trajectory:
    """A straight eastward trajectory with path distance s(t), built directly (no GNSS fitting)."""
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n = len(t)
    return Trajectory(origin=(12.9715987, 77.5945627, 900.0), t=t, e=s.copy(), n=np.zeros(n), u=np.zeros(n),
                      speed=np.gradient(s, t) if speed is None else np.asarray(speed, float),
                      heading=np.full(n, 90.0), heading_from_gyro=np.zeros(n, bool), s=s, h_sigma=np.ones(n),
                      valid=np.ones(n, bool) if valid is None else np.asarray(valid, bool),
                      reversing=np.zeros(n, bool), latency=LatencyEstimate(0.0, None, "none"), utc=None,
                      gaps=[], reversing_detection=False)


# ---- planning --------------------------------------------------------------------------------------

@pytest.mark.parametrize("total,n_full,partial", [(23.0, 4, 3.0), (22.4, 4, None), (20.0, 4, None),
                                                  (2.5, 0, 2.5), (2.4, 0, None), (5.0, 1, None)])
def test_plan_full_segments_and_partial_rule(total, n_full, partial):
    t = np.arange(0.0, 10.0 + 1e-9, 0.01)
    s0 = 100.0
    segs = plan_segments(_traj(t, s0 + total * t / t[-1]), SegmentConfig())
    assert [g.k for g in segs] == list(range(len(segs)))
    assert len(segs) == n_full + (partial is not None)
    for k, g in enumerate(segs[:n_full]):
        assert g.s0 == s0 + k * 5.0 and g.s1 == s0 + (k + 1) * 5.0 and not g.partial
    if partial is not None:
        last = segs[-1]
        assert last.partial and last.s0 == s0 + n_full * 5.0
        assert last.s1 == pytest.approx(s0 + total, abs=1e-9) and last.length_m == pytest.approx(partial, abs=1e-9)
    assert all(a.s1 == b.s0 for a, b in zip(segs, segs[1:]))  # contiguous, no overlap


def test_plan_starts_at_first_valid_position_and_ends_at_last():
    t = np.arange(0.0, 20.0 + 1e-9, 0.01)
    s = 3.0 * t
    valid = (t >= 2.0) & (t <= 17.5)
    segs = plan_segments(_traj(t, s, valid=valid), Config())
    first, last = float(s[valid][0]), float(s[valid][-1])
    assert segs[0].s0 == first
    # 46.5 m: 9 full segments; the 1.5 m remainder is under half a segment and dropped.
    assert len(segs) == 9 and segs[-1].s1 == pytest.approx(first + 45.0) and segs[-1].s1 <= last
    assert not any(g.partial for g in segs)


def test_plan_with_recording_span_and_view_offset_keeps_every_window_inside_the_recording():
    t = np.arange(0.0, 60.0 + 1e-9, 0.01)
    traj = _traj(t, 0.2 * t ** 2)  # accelerating: s is not linear in t
    geom = SegmentGeometry.build(_cam(), Config())
    cfg = Config()
    t_span = (5.123, 52.987)
    off = plan_offset_m(geom, cfg)
    assert off == pytest.approx(geom.near_m + cfg.segments.window_m)
    segs = plan_segments(traj, cfg, t_span=t_span, offset_m=off)
    c_first, c_last = 0.2 * t_span[0] ** 2, 0.2 * t_span[1] ** 2
    assert segs[0].s0 == pytest.approx(c_first + off, abs=1e-4)  # interpolated span start, not a grid point
    total = c_last - c_first                                       # 556.28 m: 111 segments + 1.28 m dropped
    assert len(segs) == 111 == int(total // 5.0)
    for g in segs:
        lo, hi = candidate_window(g, geom, cfg)
        assert c_first - 1e-4 <= lo and hi <= c_last + 1e-4
    assert candidate_window(segs[0], geom, cfg)[0] == pytest.approx(c_first, abs=1e-4)  # first window: first frame


def test_plan_rejects_bad_input_and_handles_no_valid_position():
    t = np.arange(0.0, 10.0, 0.01)
    traj = _traj(t, 5.0 * t)
    with pytest.raises(ValueError):
        plan_segments(traj, SegmentConfig(length_m=0.0))
    with pytest.raises(ValueError):
        plan_segments(traj, SegmentConfig(), t_span=(5.0, 4.0))
    with pytest.raises(ValueError):
        plan_segments(traj, SegmentConfig(), offset_m=math.nan)
    assert plan_segments(_traj(t, 5.0 * t, valid=np.zeros(len(t), bool)), SegmentConfig()) == []


# ---- geometry --------------------------------------------------------------------------------------

def test_geometry_matches_closed_form_flat_road_s21():
    m = _cam()
    g = SegmentGeometry.build(m, Config())
    assert g.v_usable == 2159
    assert g.near_m == pytest.approx(_cf_ahead(2159, m), rel=1e-9)
    assert g.near_m == pytest.approx(1.44, abs=0.01)    # pilot: road from 1.44 m ahead
    assert g.far_m == pytest.approx(_cf_ahead(0, m), rel=1e-9) and not g.far_capped
    assert 32.0 < g.far_m < 34.0                        # pilot: ~33 m
    for d0 in (g.near_m, 2.0, 3.7):
        v_top, v_bottom = g.rows_for(d0, d0 + 5.0)
        assert v_top == pytest.approx(_cf_row(d0 + 5.0, m), abs=1e-6)
        assert v_bottom == pytest.approx(_cf_row(d0, m), abs=1e-6)
    vt, vb = g.rows_for(np.array([1.5, 2.5]), np.array([6.5, 7.5]))
    assert np.allclose(vt, [_cf_row(6.5, m), _cf_row(7.5, m)], atol=1e-6)
    assert np.allclose(vb, [_cf_row(1.5, m), _cf_row(2.5, m)], atol=1e-6)
    for v in (2159.0, 1500.0, 800.0, 100.0):
        assert g.gsd_along_at(v) == pytest.approx(_cf_gsd_mm(v, m), rel=1e-3)
    assert np.allclose(g.gsd_along_at(np.array([2000.0, 600.0])), [_cf_gsd_mm(2000.0, m), _cf_gsd_mm(600.0, m)],
                       rtol=1e-3)
    assert g.f_px == pytest.approx((FX + FY) / 2)
    with pytest.raises(ValueError):
        g.rows_for(3.0, 3.0)


def test_geometry_far_cap_horizon_and_beyond_cap():
    assert math.isinf(_cf_ahead(0, _cam(pitch=15.0), 15.0))     # top row looks above the horizon
    above = SegmentGeometry.build(_cam(pitch=15.0), Config())
    assert above.far_m == FAR_CAP_M and above.far_capped
    assert above.near_m == pytest.approx(_cf_ahead(2159, _cam(pitch=15.0), 15.0), rel=1e-9)
    assert _cf_ahead(0, _cam(pitch=21.0), 21.0) > FAR_CAP_M      # top row meets the road ~230 m ahead
    beyond = SegmentGeometry.build(_cam(pitch=21.0), Config())
    assert beyond.far_m == FAR_CAP_M and beyond.far_capped


def test_geometry_bottom_exclusion_moves_near_edge():
    cfg = Config()
    cfg.geometry.bottom_exclude_frac = 0.1
    m = _cam()
    g = SegmentGeometry.build(m, cfg)
    assert g.v_usable == 1943  # the rule of extract.plan_view: v <= H x (1 - frac) - 1
    assert g.near_m == pytest.approx(_cf_ahead(1943, m), rel=1e-9)
    assert g.row_slice(1500.2, 2100.0) == (1501, 1944)


def test_geometry_rejects_unusable_views():
    with pytest.raises(ValueError, match="forward"):
        SegmentGeometry.build(_cam(facing="rearward"), Config())
    with pytest.raises(ValueError, match="no road"):
        SegmentGeometry.build(_cam(pitch=-30.0), Config())
    with pytest.raises(ValueError, match="never fits"):
        SegmentGeometry.build(_cam(pitch=60.0), Config())  # road visible only ~0.24..1.65 m ahead


def test_geometry_of_an_upside_down_mount_comes_from_the_rotated_model():
    meta = {"camera": {"sensor_orientation_deg": 90}, "mount": {"nominal_height_m": HEIGHT, "camera_facing": "forward"}}
    th = math.radians(PITCH)
    g_dev = 9.80665 * cam_from_device(90).T @ np.array([0.0, -math.cos(th), -math.sin(th)])
    g_flipped = np.diag([-1.0, -1.0, 1.0]) @ g_dev  # phone turned 180 deg about its screen normal
    upright = SegmentGeometry.build(nominal_model(meta, 3840, 2160, g_dev, GeometryConfig()), Config())
    rotated = SegmentGeometry.build(nominal_model(meta, 3840, 2160, g_flipped, GeometryConfig(), rotate180=True),
                                    Config())
    assert rotated.near_m == pytest.approx(upright.near_m, rel=1e-9)
    assert rotated.far_m == pytest.approx(upright.far_m, rel=1e-9)
    assert np.allclose(rotated.rows_for(2.0, 7.0), upright.rows_for(2.0, 7.0), atol=1e-6)
    with pytest.raises(ValueError):  # the sensor-orientation model sees the near road at the top
        SegmentGeometry.build(nominal_model(meta, 3840, 2160, g_flipped, GeometryConfig()), Config())


def test_row_slice_is_rows_whose_centres_lie_in_the_range():
    g = SegmentGeometry.build(_cam(), Config())
    assert g.row_slice(520.3, 2170.0) == (521, 2160)  # clipped to the usable bottom row
    assert g.row_slice(-3.0, 10.0) == (0, 11)
    assert g.row_slice(100.0, 100.0) == (100, 101)
    assert g.row_slice(100.2, 100.8) == (101, 101)    # no row centre inside: empty
    with pytest.raises(ValueError):
        g.row_slice(math.nan, 10.0)


def test_footprint_is_the_closed_form_trapezoid():
    m = _cam()
    g = SegmentGeometry.build(m, Config())
    for d0, d1 in ((2.0, 7.0), (g.near_m + 0.4, g.near_m + 5.4)):
        ring = g.footprint(d0, d1)
        assert np.array_equal(ring[0], ring[-1]) and len(ring) == 5
        assert Polygon(ring).exterior.is_ccw
        corners = [(_cf_x(u, d, m), d) for u in (0.0, m.width - 1.0) for d in (d0, d1)]
        for x, y in corners:
            assert np.min(np.hypot(ring[:-1, 0] - x, ring[:-1, 1] - y)) < 1e-6
        width_near = _cf_x(m.width - 1.0, d0, m) - _cf_x(0.0, d0, m)
        width_far = _cf_x(m.width - 1.0, d1, m) - _cf_x(0.0, d1, m)
        assert Polygon(ring).area == pytest.approx((width_near + width_far) / 2 * (d1 - d0), rel=1e-9)
    # Road below the bottom row is not in the frame: the polygon starts at near_m.
    ring = g.footprint(1.0, 6.0)
    assert ring[:, 1].min() == pytest.approx(g.near_m, abs=1e-9) and len(ring) == 5
    with pytest.raises(ValueError):
        g.footprint(5.0, 5.0)


def test_footprint_with_roll_is_exactly_what_the_frame_sees():
    m = _cam(roll=-4.4)
    assert m.roll_deg == pytest.approx(-4.4)
    g = SegmentGeometry.build(m, Config())
    d0, d1 = g.near_m, g.near_m + 5.0
    poly = Polygon(g.footprint(d0, d1))
    xs, ys = np.meshgrid(np.linspace(-5.0, 5.0, 241), np.linspace(d0 - 0.5, d1 + 0.5, 241))
    u, v, ok = m.ground_to_pixel(xs, ys)
    seen = ok & (u >= 0) & (u <= m.width - 1) & (v >= 0) & (v <= g.v_usable) & (ys >= d0) & (ys <= d1)
    inside = shapely.contains_xy(poly, xs, ys)
    near_edge = shapely.distance(poly.exterior, shapely.points(xs, ys)) < 0.005
    assert np.array_equal(inside[~near_edge], seen[~near_edge])
    assert seen.sum() > 5000
    assert len(poly.exterior.coords) == 6  # the rolled bottom row cuts one corner of the strip


def test_summary_is_json_ready():
    g = SegmentGeometry.build(_cam(), Config())
    out = g.summary(5.0)
    to_json(out)  # raises on NaN or unknown types
    assert out["near_m"] == g.near_m and out["far_m"] == g.far_m and out["usable_bottom_row"] == 2159
    assert out["nearest_segment"]["ahead_m"] == [g.near_m, g.near_m + 5.0]


# ---- candidate window -----------------------------------------------------------------------------

def test_candidate_window_closed_form():
    m = _cam()
    g = SegmentGeometry.build(m, Config())
    near, far = _cf_ahead(2159, m), _cf_ahead(0, m)
    seg = Segment(7, 135.0, 140.0)
    lo, hi = candidate_window(seg, g, Config())
    assert hi == pytest.approx(135.0 - near, abs=1e-9) and lo == pytest.approx(135.0 - near - 1.0, abs=1e-9)
    # From hi the near edge is on the bottom row; from lo it is window_m further ahead.
    assert g.rows_for(seg.s0 - hi, seg.s1 - hi)[1] == pytest.approx(_cf_row(near, m), abs=1e-6)
    assert g.rows_for(seg.s0 - lo, seg.s1 - lo)[1] == pytest.approx(_cf_row(near + 1.0, m), abs=1e-6)
    # Long segments: the far edge must be in view as well, so the window starts at s1 - far.
    lo, hi = candidate_window(Segment(0, 100.0, 131.0), g, SegmentConfig(length_m=31.0, window_m=1.0))
    assert hi == pytest.approx(100.0 - near) and lo == pytest.approx(131.0 - far) and hi - lo < 1.0
    with pytest.raises(ValueError):
        candidate_window(Segment(0, 100.0, 140.0), g, SegmentConfig(length_m=40.0))
    with pytest.raises(ValueError):
        candidate_window(seg, g, SegmentConfig(window_m=-0.1))


# ---- sharpness -------------------------------------------------------------------------------------

def _tenengrad_ref(rgb, r0, r1, c0, c1, half=False):
    """Independent numpy implementation: BT.601 grey, optional 2x2 mean, 3x3 Sobel, interior only."""
    reg = rgb[r0:r1, c0:c1].astype(np.float64)
    g = 0.299 * reg[..., 0] + 0.587 * reg[..., 1] + 0.114 * reg[..., 2]
    if half:
        g = (g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2]) / 4
    gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    return float(np.mean(gx ** 2 + gy ** 2) / g[1:-1, 1:-1].mean() ** 2)


def test_sharpness_is_the_specified_normalised_tenengrad():
    rgb = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 256, (180, 320, 3), dtype=np.uint8), (0, 0), 0.8)
    # cols (0.2, 0.8) of 320 px are columns 64..256.
    assert sharpness(rgb, (40, 160), cols=(0.2, 0.8), downscale=1.0) == pytest.approx(
        _tenengrad_ref(rgb, 40, 160, 64, 256), rel=1e-5)
    assert sharpness(rgb, (40, 160), cols=(0.2, 0.8), downscale=0.5) == pytest.approx(
        _tenengrad_ref(rgb, 40, 160, 64, 256, half=True), rel=1e-5)


def test_sharpness_uses_only_the_segment_rows_and_lane_columns():
    rgb = cv2.GaussianBlur(np.random.default_rng(2).integers(0, 256, (180, 320, 3), dtype=np.uint8), (0, 0), 1.0)
    ref = sharpness(rgb, (40, 160), cols=(0.2, 0.8), downscale=0.5)
    other = rgb.copy()
    other[:40] = 0
    other[160:] = 255
    other[:, :64] = np.random.default_rng(3).integers(0, 256, (180, 64, 3), dtype=np.uint8)
    other[:, 256:] = 17
    assert sharpness(other, (40, 160), cols=(0.2, 0.8), downscale=0.5) == ref
    assert sharpness(np.zeros((180, 320, 3), np.uint8), (40, 160), cols=(0.2, 0.8), downscale=0.5) == 0.0


def test_sharpness_rejects_bad_arguments():
    rgb = np.zeros((180, 320, 3), np.uint8)
    for kw in ({"cols": (0.8, 0.2), "downscale": 0.5}, {"cols": (0.2, 1.2), "downscale": 0.5},
               {"cols": (0.2, 0.8), "downscale": 0.0}, {"cols": (0.2, 0.8), "downscale": 1.5}):
        with pytest.raises(ValueError):
            sharpness(rgb, (40, 160), **kw)
    with pytest.raises(ValueError):
        sharpness(rgb[..., 0], (40, 160), cols=(0.2, 0.8), downscale=0.5)
    with pytest.raises(ValueError):
        sharpness(rgb, (100, 102), cols=(0.2, 0.8), downscale=1.0)   # 2 rows
    with pytest.raises(ValueError):
        sharpness(rgb, (100, 104), cols=(0.2, 0.8), downscale=0.5)   # 4 rows -> 2 after halving


@pytest.fixture(scope="module")
def road_view():
    """Asphalt (simtexture's octaves and stones) seen by the S21 camera at quarter size, and the rows
    of the segment whose near edge is at near_m."""
    m = _cam(scale=0.25)
    u, v = np.meshgrid(np.arange(m.width, dtype=np.float64), np.arange(m.height, dtype=np.float64))
    x, y, ok = m.pixel_to_ground(u, v)
    X, Y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    lum = np.full(X.shape, 0.42, np.float32)
    for cell, amp, k in ((0.8, 0.05, 1), (0.2, 0.03, 2), (0.05, 0.03, 3), (0.012, 0.05, 4)):
        lum += amp * (value_noise(X, Y, cell, 93 + k) - 0.5) * 2
    stones = value_noise(X, Y, 0.007, 100)
    lum += np.where(stones > 0.80, 0.10, 0.0) - np.where(stones < 0.12, 0.08, 0.0)
    lum[~ok] = 0.8
    g = SegmentGeometry.build(m, Config())
    return to_rgb(np.clip(lum, 0, 1)), g.row_slice(*g.rows_for(g.near_m, g.near_m + 5.0))


def test_sharpness_ranks_sharp_road_above_defocus_and_motion_blur_and_ignores_exposure(road_view):
    rgb, rows = road_view
    cfg = SegmentConfig()

    def score(img):
        return sharpness(img, rows, cols=cfg.sharp_cols, downscale=cfg.sharp_downscale)

    smear = np.zeros((5, 5), np.float32)
    smear[:, 2] = 1 / 5  # 5 px along-track motion: the road moves vertically through the image
    variants = {"sharp": rgb, "defocus1": cv2.GaussianBlur(rgb, (0, 0), 1.0),
                "defocus2": cv2.GaussianBlur(rgb, (0, 0), 2.0), "motion5": cv2.filter2D(rgb, -1, smear)}
    s = {k: score(img) for k, img in variants.items()}
    assert s["sharp"] > s["defocus1"] > s["defocus2"]
    assert s["sharp"] > 1.3 * s["motion5"] and s["sharp"] > 1.3 * s["defocus1"]
    for k, img in variants.items():
        half = np.round(img.astype(np.float64) * 0.5).astype(np.uint8)  # x0.5 exposure, 8-bit
        assert score(half) == pytest.approx(s[k], rel=5e-3)
    half_sharp = score(np.round(rgb.astype(np.float64) * 0.5).astype(np.uint8))
    assert half_sharp > s["defocus1"] and half_sharp > s["motion5"]  # a darker sharp frame still wins


# ---- predicted blur ---------------------------------------------------------------------------------

def test_predicted_blur_hand_calculation():
    b = predicted_blur(10.0, 2_000_000, 2.5, -0.3, 2909.0)
    assert b["blur_motion_px"] == pytest.approx(8.0, rel=1e-12)       # 10 m/s x 2 ms = 20 mm at 2.5 mm/px
    assert b["blur_rotation_px"] == pytest.approx(1.7454, rel=1e-12)  # 0.3 rad/s x 2 ms x 2909 px
    assert b["blur_px"] == pytest.approx(math.sqrt(8.0 ** 2 + 1.7454 ** 2), rel=1e-12)
    # S21 at 40 km/h, the sessions' 75th-percentile exposure 0.826 ms, nearest candidate's centre row.
    m = _cam()
    g = SegmentGeometry.build(m, Config())
    r0, r1 = _cf_row_slice(g.near_m, g.near_m + 5.0, m, 2159)
    gsd = _cf_gsd_mm((r0 + r1 - 1) / 2, m)
    b = predicted_blur(40 / 3.6, 826_000, gsd, 0.0, g.f_px)
    assert b["blur_motion_px"] == pytest.approx((40 / 3.6) * 0.826e-3 / (gsd / 1000.0), rel=1e-12)
    assert b["blur_rotation_px"] == 0.0 and b["blur_px"] == pytest.approx(b["blur_motion_px"], rel=1e-15)


def test_predicted_blur_unknowns():
    no_gyro = predicted_blur(10.0, 2_000_000, 2.5, math.nan, 2909.0)
    assert math.isnan(no_gyro["blur_rotation_px"]) and no_gyro["blur_px"] == pytest.approx(8.0)
    for exposure in (None, pd.NA, math.nan, "n/a"):
        assert all(math.isnan(v) for v in predicted_blur(10.0, exposure, 2.5, 0.3, 2909.0).values())


# ---- gyro rates about the image axes --------------------------------------------------------------

def _gyro_df(t_s, w_dev) -> pd.DataFrame:
    w_dev = np.asarray(w_dev, dtype=np.float64)
    return pd.DataFrame({"t_ns": pd.array(T0_NS + np.round(np.asarray(t_s) * 1e9).astype(np.int64), dtype="Int64"),
                         "sensor": "gyro", "x": w_dev[:, 0], "y": w_dev[:, 1], "z": w_dev[:, 2], "w": np.nan,
                         "accuracy": 3})


def test_gyro_rate_about_image_axes_for_every_sensor_orientation():
    t = np.arange(0.0, 2.0 + 1e-9, 0.005)
    w = np.column_stack([0.3 * np.sin(2 * np.pi * t), -0.2 * np.cos(2 * np.pi * t), np.full_like(t, 0.7)])
    gyro = _gyro_df(t, w)
    acc = gyro.assign(sensor="acc", x=50.0, y=-50.0, z=9.8)            # other sensors share the file
    imu = pd.concat([acc, gyro]).sample(frac=1.0, random_state=0)       # and arrive unsorted
    mid = (t[:-1] + t[1:]) / 2
    for so in (0, 90, 180, 270):
        rates = GyroImageRates(imu, cam_from_device(so), T0_NS)
        # The back camera's optical axis is device -z for every orientation, so the rate about the
        # two image axes is the device x/y rate; the 0.7 rad/s about z (image rotation) is not part.
        assert np.allclose(rates(t), np.hypot(w[:, 0], w[:, 1]), atol=1e-12)
        wx_mid, wy_mid = (w[:-1, 0] + w[1:, 0]) / 2, (w[:-1, 1] + w[1:, 1]) / 2  # linear in each component
        assert np.allclose(rates(mid), np.hypot(wx_mid, wy_mid), atol=1e-12)
    # Per axis, from the app's orientation rule (sensor up = device +x at 90 deg, +y at 0 deg).
    r90 = GyroImageRates(gyro, cam_from_device(90), T0_NS)
    assert np.allclose(r90.wx, -w[:, 1]) and np.allclose(r90.wy, -w[:, 0])  # image right = -y, down = -x
    r0 = GyroImageRates(gyro, cam_from_device(0), T0_NS)
    assert np.allclose(r0.wx, w[:, 0]) and np.allclose(r0.wy, -w[:, 1])     # image right = +x, down = -y
    roll_only = GyroImageRates(_gyro_df(t, np.column_stack([0 * t, 0 * t, 1.5 + 0 * t])), cam_from_device(90), T0_NS)
    assert np.allclose(roll_only(t), 0.0)


def test_gyro_rate_gaps_edges_and_missing_data():
    t = np.r_[np.arange(0.0, 1.0 + 1e-9, 0.005), np.arange(1.2, 2.0 + 1e-9, 0.005)]  # 200 ms dropout
    w = np.column_stack([np.full_like(t, 0.3), np.full_like(t, 0.4), np.zeros_like(t)])
    rates = GyroImageRates(_gyro_df(t, w), cam_from_device(90), T0_NS)
    got = rates([-0.1, 0.5025, 1.0, 1.1, 1.2, 2.0, 2.5])
    assert np.allclose(got[[1, 2, 4, 5]], 0.5)       # inside, and exactly on the samples at the gap edges
    assert np.isnan(got[[0, 3, 6]]).all()            # before, inside the gap, after
    assert np.isnan(GyroImageRates(_gyro_df(t, w), cam_from_device(90), T0_NS, max_gap_s=0.001)(0.5025))[0]
    for imu in (None, pd.DataFrame(), _gyro_df(t, w).assign(sensor="acc")):
        assert np.isnan(GyroImageRates(imu, cam_from_device(90), T0_NS)([0.5, 1.0])).all()
    single = GyroImageRates(_gyro_df([0.25], [[0.3, 0.4, 0.0]]), cam_from_device(90), T0_NS)
    assert single(0.25)[0] == pytest.approx(0.5) and np.isnan(single(0.26))[0]
    dup = _gyro_df([0.0, 0.0, 0.005], [[1.0, 0.0, 0.0], [5.0, 5.0, 5.0], [1.0, 0.0, 0.0]])
    assert GyroImageRates(dup, cam_from_device(90), T0_NS)(0.0)[0] == pytest.approx(1.0)  # first of duplicates


# ---- best-frame selector ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small():
    """320 x 180 S21-like camera, its geometry, and four textures from sharpest (0) to blurriest (3)."""
    m = _cam(scale=SMALL)
    g = SegmentGeometry.build(m, Config())
    base = cv2.GaussianBlur(np.random.default_rng(5).integers(0, 256, (m.height, m.width, 3)).astype(np.float32),
                            (0, 0), 0.7)
    imgs = [np.clip(cv2.GaussianBlur(base, (0, 0), s) if s else base, 0, 255).astype(np.uint8)
            for s in (0.0, 0.8, 1.6, 3.2)]
    for r0 in range(0, 100, 7):  # strictly decreasing sharpness on any segment's rows
        sh = [sharpness(i, (r0, 180), cols=(0.2, 0.8), downscale=0.5) for i in imgs]
        assert sh[0] > sh[1] > sh[2] > sh[3]
    return m, g, imgs


def _info(t, c, ref=None, *, speed=10.0, valid=True, reversing=False, black=False, low_light=False,
          exposure_ns=1_000_000, omega=0.0) -> FrameInfo:
    return FrameInfo(t=t, c=c, speed_mps=speed, valid=valid, reversing=reversing, black=black,
                     low_light=low_light, exposure_ns=exposure_ns, omega_perp_rad_s=omega, ref=ref)


def _drive(sel, infos, image_of):
    """Feed frames by the documented protocol. Returns [(result, c when emitted, None = flush)], the
    most open segments and held frames at any time, and the RGB array offered for each frame."""
    out, max_open, max_held, offered = [], 0, 0, {}
    for info in infos:
        want = sel.wants(info)
        rgb = image_of(info).copy() if want else None   # a fresh array per frame, as from the decoder
        n = sel.offer(info, rgb)
        assert (n > 0) == want
        if rgb is not None:
            offered[info.ref] = rgb
        max_open, max_held = max(max_open, sel.n_open), max(max_held, sel.held_frames)
        out += [(r, info.c) for r in sel.pop_ready(info.c)]
    out += [(r, None) for r in sel.flush()]
    return out, max_open, max_held, offered


def _scenario(sel, specs, gap_after=True):
    """Frames evenly inside each segment's window: specs[k] is a list of _info keyword dicts (+ 'img')."""
    infos, imgs = [], {}
    t = 0.0
    for k, frames in enumerate(specs):
        lo, hi = (float(x) for x in sel.windows[k])
        for j, f in enumerate(frames):
            f = dict(f)
            t += 0.03
            imgs[(k, j)] = f.pop("img", 0)
            infos.append(_info(t, lo + (j + 0.5) / len(frames) * (hi - lo), (k, j), **f))
    return infos, imgs


def _run(segs, g, cfg, specs, imgs_all):
    sel = BestFrameSelector(segs, g, cfg)
    infos, img_of = _scenario(sel, specs)
    out, _, _, offered = _drive(sel, infos, lambda f: imgs_all[img_of[f.ref]])
    assert [r.segment.k for r, _ in out] == list(range(len(segs)))
    return [r for r, _ in out], offered, sel


def _segs(n, start=20.0, gap=10.0):
    return [Segment(k, start + k * gap, start + k * gap + 5.0) for k in range(n)]


def test_selector_picks_the_sharpest_frame_with_hand_checked_geometry_and_blur(small):
    m, g, imgs = small
    cfg = SegmentConfig()
    seg = Segment(0, 20.0, 25.0)
    sel = BestFrameSelector([seg], g, cfg)
    near = _cf_ahead(g.v_usable, m)
    assert tuple(sel.windows[0]) == pytest.approx((20.0 - near - 1.0, 20.0 - near), abs=1e-9)
    lo, hi = sel.windows[0]
    cs = [lo - 0.3, lo + 0.1, lo + 0.35, lo + 0.6, lo + 0.85, hi + 0.2]
    img_idx = [0, 2, 0, 3, 1, 0]           # frames 0 and 5 are sharp but outside the window
    infos = [_info(0.03 * (i + 1), c, i) for i, c in enumerate(cs)]
    out, _, _, offered = _drive(sel, infos, lambda f: imgs[img_idx[f.ref]])
    assert len(out) == 1
    res, c_emit = out[0]
    assert c_emit == cs[5]                  # emitted as soon as the camera passed the window
    assert res.covered and res.best.info.ref == 2 and res.rgb is offered[2]
    assert res.n_candidates == 4 and res.sharpness_rank == 1 and res.reason == "sharpest" and res.excluded == {}
    b, c = res.best, cs[2]
    assert b.near_edge_ahead_m == pytest.approx(20.0 - c) and b.far_edge_ahead_m == pytest.approx(25.0 - c)
    assert b.rows == pytest.approx((_cf_row(25.0 - c, m), _cf_row(20.0 - c, m)), abs=1e-6)
    assert b.row_slice == _cf_row_slice(20.0 - c, 25.0 - c, m, g.v_usable)
    gsd = _cf_gsd_mm((b.row_slice[0] + b.row_slice[1] - 1) / 2, m)
    assert b.gsd_along_mm == pytest.approx(gsd, rel=1e-3)
    assert b.blur_motion_px == pytest.approx(10.0 * 1e-3 / (gsd / 1000.0), rel=1e-3)
    assert b.blur_rotation_px == 0.0 and not b.over_blur_limit and b.tier == 0
    assert b.sharpness == sharpness(imgs[0], b.row_slice, cols=cfg.sharp_cols, downscale=cfg.sharp_downscale)
    q = res.quality()
    to_json(q)
    assert q == {"sharpness": b.sharpness, "sharpness_rank": 1, "n_candidates": 4, "blur_motion_px": b.blur_motion_px,
                 "blur_rotation_px": 0.0, "blur_px": b.blur_px, "gyro_rad_s": 0.0, "reason": "sharpest"}


def test_selector_blur_limit(small):
    m, g, imgs = small
    cfg = SegmentConfig(max_blur_px=1.0)   # at 320 x 180 the segment's centre row is ~37 mm/px
    slow_shutter = {"exposure_ns": 8_000_000}           # 10 m/s x 8 ms = 80 mm: ~2 px
    specs = [
        [{"img": 0, **slow_shutter}, {"img": 1}, {"img": 2}, {"img": 3}],   # sharpest is smeared
        [{"img": 1, **slow_shutter}, {"img": 0, **slow_shutter}, {"img": 2, **slow_shutter}],  # all smeared
        [{"img": 0, "omega": 6.0}, {"img": 2}, {"img": 1}],                  # 6 rad/s x 1 ms x 242 px = 1.45 px
        [{"img": 1, "exposure_ns": None}, {"img": 0, "exposure_ns": None}],  # unknown exposure never excludes
    ]
    res, _, _ = _run(_segs(4), g, cfg, specs, imgs)
    assert (res[0].best.info.ref, res[0].sharpness_rank, res[0].reason) == ((0, 1), 2, "sharpest")
    assert (res[1].best.info.ref, res[1].sharpness_rank, res[1].reason) == ((1, 1), 1, "sharpest_over_blur_limit")
    b = res[1].best
    gsd = _cf_gsd_mm((b.row_slice[0] + b.row_slice[1] - 1) / 2, m)
    assert b.blur_motion_px == pytest.approx(10.0 * 8e-3 / (gsd / 1000.0), rel=1e-3) and b.over_blur_limit
    assert (res[2].best.info.ref, res[2].sharpness_rank, res[2].reason) == ((2, 2), 2, "sharpest")
    assert (res[3].best.info.ref, res[3].reason) == ((3, 1), "sharpest")
    assert res[3].quality()["blur_px"] is None and res[3].quality()["blur_motion_px"] is None
    assert [r.n_candidates for r in res] == [4, 3, 3, 2]


def test_selector_low_light_rules(small):
    _, g, imgs = small
    low, smeared = {"low_light": True}, {"exposure_ns": 8_000_000}
    specs = [
        [{"img": 0, **low}, {"img": 1, **low}, {"img": 2}],                 # a normal frame beats sharper dark ones
        [{"img": 2, **low}, {"img": 0, **low}, {"img": 1, **low}],          # all dark: the sharpest dark one
        [{"img": 1, **low, **smeared}, {"img": 0, **low, **smeared}],
        [{"img": 0, **low}, {"img": 3, **smeared}],                         # not dark but smeared beats dark
    ]
    res, _, _ = _run(_segs(4), g, SegmentConfig(max_blur_px=1.0), specs, imgs)
    assert [(r.best.info.ref, r.sharpness_rank, r.reason) for r in res] == [
        ((0, 2), 3, "sharpest"), ((1, 1), 1, "only_low_light"), ((2, 1), 1, "only_low_light_over_blur_limit"),
        ((3, 1), 2, "sharpest_over_blur_limit")]
    assert [r.best.tier for r in res] == [0, 2, 3, 1]
    # skip_low_light: dark frames are never candidates (and need no RGB).
    specs = [[{"img": 0, **low}, {"img": 1, **low}], [{"img": 0, **low}, {"img": 3}]]
    res, offered, _ = _run(_segs(2), g, SegmentConfig(skip_low_light=True), specs, imgs)
    assert not res[0].covered and res[0].reason == "low_light" and res[0].excluded == {"low_light": 2}
    assert res[0].quality() == {"sharpness": None, "sharpness_rank": None, "n_candidates": 0, "blur_motion_px": None,
                                "blur_rotation_px": None, "blur_px": None, "gyro_rad_s": None, "reason": "low_light"}
    assert (res[1].best.info.ref, res[1].n_candidates, res[1].excluded) == ((1, 1), 1, {"low_light": 1})
    assert set(offered) == {(1, 1)}


def test_selector_exclusions_and_their_reasons(small):
    _, g, imgs = small
    specs = [
        [{"valid": False}] * 3,
        [{"black": True}] * 2,
        [{"reversing": True}] * 2,
        [{"speed": 0.2}] * 3,
        [{"speed": 0.0}, {"black": True}, {"speed": 0.1}],          # most frequent: standing
        [{"valid": False}, {"reversing": True}],                    # tie: the earlier reason
        # One reason per frame, by priority: gnss_gap > black > reversing > standing > low_light.
        [{"valid": False, "black": True, "reversing": True, "speed": 0.0, "low_light": True}] * 2,
        [{"black": True, "reversing": True, "speed": 0.0, "low_light": True}] * 2,
        [{"reversing": True, "speed": 0.0, "low_light": True}] * 2,
        [{"speed": 0.0, "low_light": True}] * 2,
        [{"speed": math.nan}, {"speed": 0.5, "img": 2}],            # NaN speed: standing; 0.5 m/s: moving
    ]
    res, offered, _ = _run(_segs(len(specs)), g, SegmentConfig(skip_low_light=True), specs, imgs)
    assert [r.reason for r in res] == ["gnss_gap", "black", "reversing", "standing", "standing", "gnss_gap",
                                       "gnss_gap", "black", "reversing", "standing", "sharpest"]
    assert [r.excluded for r in res] == [{"gnss_gap": 3}, {"black": 2}, {"reversing": 2}, {"standing": 3},
                                         {"black": 1, "standing": 2}, {"gnss_gap": 1, "reversing": 1},
                                         {"gnss_gap": 2}, {"black": 2}, {"reversing": 2}, {"standing": 2},
                                         {"standing": 1}]
    assert all(r.best is None and r.rgb is None and r.n_candidates == 0 for r in res[:-1])
    assert set(offered) == {(10, 1)}  # excluded frames never needed RGB


def test_selector_tie_goes_to_the_nearest_frame(small):
    _, g, imgs = small
    seg = Segment(0, 20.0, 25.0)
    sel = BestFrameSelector([seg], g, SegmentConfig())
    hi = float(sel.windows[0][1])
    cs = [hi - 3e-4, hi - 2e-4, hi - 1e-4]  # the same rows, the same image: identical sharpness
    assert len({g.row_slice(*g.rows_for(20.0 - c, 25.0 - c)) for c in cs}) == 1
    out, _, _, _ = _drive(sel, [_info(0.03 * (i + 1), c, i) for i, c in enumerate(cs)], lambda f: imgs[1])
    res = out[0][0]
    assert res.best.info.ref == 2 and res.sharpness_rank == 1 and res.n_candidates == 3
    # A full tie (same position too) keeps the frame offered first, so the choice is deterministic.
    sel = BestFrameSelector([seg], g, SegmentConfig())
    out, _, _, _ = _drive(sel, [_info(0.03, cs[0], "a"), _info(0.06, cs[0], "b")], lambda f: imgs[1])
    assert out[0][0].best.info.ref == "a" and out[0][0].n_candidates == 2


def test_selector_reasons_for_windows_without_frames(small):
    _, g, imgs = small
    near = g.near_m
    # Frames every 2 m from c = 10 to 98: wider apart than the 1 m window.
    cs = np.arange(10.0, 98.0 + 1e-9, 2.0)
    segs = [Segment(0, 2.0, 7.0), Segment(1, 7.0, 12.0), Segment(2, 51.5 + near, 56.5 + near),
            Segment(3, 60.0, 65.0), Segment(4, 200.0, 205.0)]
    sel = BestFrameSelector(segs, g, SegmentConfig())
    out, _, _, _ = _drive(sel, [_info(0.03 * (i + 1), float(c), i) for i, c in enumerate(cs)], lambda f: imgs[0])
    res = {r.segment.k: (r, c) for r, c in out}
    assert res[0][0].reason == res[1][0].reason == "before_recording" and res[0][1] == res[1][1] == 10.0
    assert res[2][0].reason == "no_frame_in_window"   # window [50.5, 51.5] lies between frames at 50 and 52
    assert res[3][0].covered
    assert res[4][0].reason == "after_recording" and res[4][1] is None
    empty = BestFrameSelector(segs, g, SegmentConfig())
    assert [r.reason for r in empty.flush()] == ["no_frames"] * 5


def _long_drive(rng_seed=11):
    """90 s drive: accelerate, cruise at 10 m/s, brake to a 10 s stop, drive on at 6 m/s. A 2 s GNSS gap,
    a black interval, a reversing flag, a low-light stretch; frame intervals 25-45 ms, 3 % dropped."""
    rng = np.random.default_rng(rng_seed)
    tg = np.arange(0.0, 90.0 + 1e-9, 0.01)
    v = np.interp(tg, [0, 10, 40, 45.1, 55, 60, 90], [0, 10, 10, 0, 0, 6, 6])
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * 0.01)])
    traj = _traj(tg, s, valid=~((tg >= 20.0) & (tg < 22.0)), speed=v)
    tf = np.cumsum(rng.uniform(0.025, 0.045, 3000))
    tf = tf[(tf <= 89.9) & (rng.random(3000) > 0.03)]
    p = traj.sample(tf)
    infos = [FrameInfo(t=float(t), c=float(p["s"][i]), speed_mps=float(p["speed"][i]), valid=bool(p["valid"][i]),
                       reversing=bool(70.0 <= t < 71.0), black=bool(30.0 <= t < 30.5),
                       low_light=bool(75.0 <= t < 80.0), exposure_ns=int(rng.choice([400_000, 1_000_000, 8_000_000])),
                       omega_perp_rad_s=float(rng.uniform(0.0, 0.5)) if rng.random() > 0.1 else math.nan,
                       ref=(i, int(rng.integers(0, 4))))
             for i, t in enumerate(tf)]
    return traj, infos


def _brute_force(segs, infos, imgs, m, g, cfg):
    """The choice of §3 written out over all frames, with closed-form rows and hand-computed blur."""
    out = []
    for seg in segs:
        hi = seg.s0 - g.near_m
        lo = max(hi - cfg.window_m, seg.s1 - g.far_m)
        excluded, cands = Counter(), []
        for f in infos:
            if not lo <= f.c <= hi:
                continue
            why = ("gnss_gap" if not f.valid else "black" if f.black else "reversing" if f.reversing
                   else "standing" if not f.speed_mps >= cfg.min_speed_mps
                   else "low_light" if f.low_light and cfg.skip_low_light else None)
            if why:
                excluded[why] += 1
                continue
            r0, r1 = _cf_row_slice(seg.s0 - f.c, seg.s1 - f.c, m, g.v_usable)
            sharp = sharpness(imgs[f.ref[1]], (r0, r1), cols=cfg.sharp_cols, downscale=cfg.sharp_downscale)
            e = f.exposure_ns / 1e9
            motion = f.speed_mps * e / (g.gsd_along_at((r0 + r1 - 1) / 2) / 1000.0)
            rot = abs(f.omega_perp_rad_s) * e * g.f_px
            blur = math.sqrt(motion * motion + (0.0 if math.isnan(rot) else rot * rot))
            cands.append((2 * f.low_light + (blur > cfg.max_blur_px), -sharp, -f.c, f.ref, sharp))
        exc = dict(sorted(excluded.items()))
        if cands:
            best = min(cands)
            out.append({"ref": best[3], "n": len(cands), "rank": 1 + sum(c[4] > best[4] for c in cands),
                        "reason": CHOSEN_REASONS[best[0]], "excluded": exc})
        else:
            top = max(exc.values())
            out.append({"ref": None, "n": 0, "rank": None, "excluded": exc,
                        "reason": next(r for r in EXCLUSION_REASONS if exc.get(r) == top)})
    return out


@pytest.mark.parametrize("window_m,max_open", [(1.0, 1), (6.0, 3)])
def test_selector_long_drive_matches_brute_force_with_bounded_memory(small, window_m, max_open):
    m, g, imgs = small
    cfg = SegmentConfig(window_m=window_m, max_blur_px=1.0)
    traj, infos = _long_drive()
    segs = plan_segments(traj, cfg, t_span=(infos[0].t, infos[-1].t), offset_m=plan_offset_m(g, cfg))
    assert len(segs) > 100
    sel = BestFrameSelector(segs, g, cfg)
    out, n_open, n_held, offered = _drive(sel, infos, lambda f: imgs[f.ref[1]])
    # Each segment exactly once, in order; as soon as the camera passed its window.
    assert [r.segment.k for r, _ in out] == list(range(len(segs)))
    for r, c_emit in out:
        assert r.window[1] < c_emit if c_emit is not None else r.window[1] >= infos[-1].c
    # Bounded memory: one 4K frame with disjoint windows; ceil(window / length) + 1 with overlapping ones.
    assert n_open <= max_open and n_held <= max_open
    if window_m > 5.0:
        assert n_open >= 2  # the windows really overlapped
    expected = _brute_force(segs, infos, imgs, m, g, cfg)
    for (r, _), e in zip(out, expected):
        assert (r.best.info.ref if r.best else None) == e["ref"], r.segment
        assert (r.n_candidates, r.sharpness_rank, r.reason, r.excluded) == (e["n"], e["rank"], e["reason"],
                                                                             e["excluded"]), r.segment
        if r.best is not None:
            assert r.rgb is offered[r.best.info.ref]
    reasons = set(r.reason for r, _ in out)
    assert {"sharpest", "only_low_light", "gnss_gap"} <= reasons
    if window_m < 5.0:  # with ~3 candidates per window some windows hold only smeared frames
        assert "sharpest_over_blur_limit" in reasons
    # The 10 s stop (300 frames at one position) falls in a window: its standing frames are excluded,
    # the moving frames around them still give the segment a frame.
    stop = [r for r, _ in out if r.excluded.get("standing", 0) > 200]
    assert len(stop) >= 1 and all(r.covered for r in stop)


def test_selector_with_two_recording_sizes(small):
    _, g12, imgs = small
    g24 = SegmentGeometry.build(_cam(scale=SMALL / 2), Config())
    assert g24.near_m > g12.near_m and g24.far_m < g12.far_m
    seg = Segment(0, 20.0, 25.0)
    cfg = SegmentConfig()
    sel = BestFrameSelector([seg], [g12, g24], cfg)
    assert (sel.near_m, sel.far_m) == (g24.near_m, g24.far_m)
    assert plan_offset_m([g12, g24], cfg) == g24.near_m + 1.0
    assert sel.geometry_for(160, 90) is g24 and sel.geometry_for(320, 180) is g12
    with pytest.raises(KeyError):
        sel.geometry_for(640, 360)
    lo, hi = (float(x) for x in sel.windows[0])
    small_sharp = cv2.resize(imgs[0], (160, 90), interpolation=cv2.INTER_AREA)
    frames = {0: imgs[3], 1: small_sharp}
    infos = [_info(0.03, lo + 0.2, 0), _info(0.06, lo + 0.7, 1)]
    out, _, _, _ = _drive(sel, infos, lambda f: frames[f.ref])
    res = out[0][0]
    assert res.n_candidates == 2 and res.best.info.ref == 1 and (res.best.width, res.best.height) == (160, 90)
    c = lo + 0.7
    assert res.best.rows == pytest.approx(g24.rows_for(20.0 - c, 25.0 - c), abs=1e-9)
    assert res.best.sharpness == sharpness(small_sharp, g24.row_slice(*res.best.rows), cols=cfg.sharp_cols,
                                           downscale=cfg.sharp_downscale)
    with pytest.raises(ValueError):
        BestFrameSelector([seg], [g12, g12], cfg)
    with pytest.raises(ValueError):
        BestFrameSelector([seg], [], cfg)


def test_selector_rejects_misuse(small):
    _, g, imgs = small
    seg = Segment(0, 20.0, 25.0)
    lo, hi = (float(x) for x in BestFrameSelector([seg], g, SegmentConfig()).windows[0])
    sel = BestFrameSelector([seg], g, SegmentConfig())
    sel.offer(_info(1.0, lo - 1.0), None)
    with pytest.raises(ValueError, match="time order"):
        sel.offer(_info(1.0, lo - 0.9), None)            # the same frame time twice
    with pytest.raises(ValueError, match="time order"):
        sel.offer(_info(0.5, lo - 0.9), None)
    with pytest.raises(ValueError, match="went back"):
        sel.offer(_info(2.0, lo - 1.5), None)
    sel.offer(_info(2.1, lo - 1.0 - 1e-9), None)         # trajectory rounding is tolerated
    with pytest.raises(ValueError, match="finite"):
        sel.offer(_info(2.2, math.nan), None)
    with pytest.raises(ValueError, match="without its RGB"):
        sel.offer(_info(3.0, lo + 0.5), None)
    with pytest.raises(ValueError, match="no SegmentGeometry"):
        sel.offer(_info(3.1, lo + 0.6), np.zeros((90, 160, 3), np.uint8))
    with pytest.raises(ValueError, match="no SegmentGeometry"):
        sel.offer(_info(3.2, lo + 0.7), np.zeros((180, 320), np.uint8))
    assert sel.offer(_info(3.3, lo + 0.8), imgs[0].copy()) == 1
    assert sel.wants(lo + 0.9) and not sel.wants(hi + 0.1) and sel.n_emitted == 0
    assert [r.segment.k for r in sel.pop_ready(hi + 0.1)] == [0] and sel.pop_ready(hi + 0.2) == []
    assert sel.n_emitted == 1 and sel.n_open == 0 and sel.held_frames == 0
    assert sel.flush() == []
    with pytest.raises(RuntimeError):
        sel.offer(_info(4.0, hi + 1.0), None)
    assert not sel.wants(hi + 1.0) and sel.pop_ready(hi + 2.0) == []
    with pytest.raises(ValueError, match="numbered"):
        BestFrameSelector([Segment(1, 20.0, 25.0)], g, SegmentConfig())
    with pytest.raises(ValueError, match="path order"):
        BestFrameSelector([Segment(0, 30.0, 35.0), Segment(1, 20.0, 25.0)], g, SegmentConfig())


def test_wants_matches_the_closed_form_windows(small):
    m, g, _ = small
    segs = _segs(5, start=20.0, gap=5.0)
    sel = BestFrameSelector(segs, g, SegmentConfig())
    near = _cf_ahead(g.v_usable, m)
    for c in np.linspace(10.0, 50.0, 4001):
        inside = any(s.s0 - near - 1.0 - 1e-9 <= c <= s.s0 - near + 1e-9 for s in segs)
        edge = any(min(abs(c - (s.s0 - near)), abs(c - (s.s0 - near - 1.0))) < 1e-8 for s in segs)
        if not edge:
            assert sel.wants(float(c)) == inside
    excluded = _info(1.0, float(sel.windows[0][0]) + 0.5, speed=0.0)
    assert sel.wants(excluded.c) and not sel.wants(excluded)
