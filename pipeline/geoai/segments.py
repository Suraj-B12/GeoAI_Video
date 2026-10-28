"""Segment mode, part 1: road segments and the best full frame of each (docs/pipeline/13 §2-3).

The road is cut into fixed segments of path distance s (metres along the GNSS trajectory; s is
integrated speed, so it never decreases). Each segment [s0, s1) is classified once, from ONE full
frame chosen while the vehicle drives towards it.

Coordinates. Ground points are those of geometry.py: x right, y ahead of the camera nadir, on a flat
road. Image rows are display rows of the full-resolution frame (an UPSIDE_DOWN mount is decoded
rotated and its CameraModel is already the rotated one, so nothing here depends on the mount).

Which frames can show a segment (§2). `near_m` is the closest road the camera sees (bottom usable
row, centre column), `far_m` the farthest (top row, or the horizon), capped at FAR_CAP_M. From camera
position c the segment is fully in view when c + near_m <= s0 and c + far_m >= s1. The candidate
window keeps only the first `window_m` of that range, c in [s0 - near_m - window_m, s0 - near_m]:
the segment then starts at the bottom of the frame, and along-track resolution degrades roughly
with the square of the distance ahead, so these are the frames that resolve the segment best.

Which of them is best (§3). Every candidate is scored by the normalised Tenengrad of the segment's
rows (lane columns only). Defocus, vibration and compression differ from frame to frame, and the
sharpest frame is the one with the least of them. Motion smear cannot be chosen away (speed and
shutter are the same in neighbouring frames): it is predicted from speed, exposure and the gyro,
used as a filter, and recorded. That raw score also grows with distance (the same road in fewer,
denser rows), so `choice = nearest_sharp` instead takes the nearest candidate that is nearly as sharp
as the best one on a common ground grid (`FairSharpness`): the finest view of the segment, unless
it is visibly blurred.

Streaming (one decode pass). Frames are offered in time order, so c never decreases; a segment is
final once c has passed its window. Only the best RGB of each open segment is kept: with the default
5 m segments and 1 m window that is at most one 4K frame at a time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from .config import Config, SegmentConfig
from .geometry import CameraModel
from .trajectory import Trajectory

# Farthest road counted as "visible". At 60 m one pixel row spans about 0.8 m of road (S21, 23 deg,
# 1.35 m height), so a farther "far edge" would add nothing a classifier could use.
FAR_CAP_M = 60.0
# Gyro samples further apart than this are a gap (the stream is 200 Hz, 5 ms; the S21 sessions of
# 26 Oct never exceeded 7.6 ms): no rate is interpolated across it, so a dropout never reads as a
# calm camera.
GYRO_MAX_GAP_S = 0.05
# Camera positions come from np.interp on the 100 Hz trajectory grid; rounding at grid boundaries can
# make consecutive positions decrease by ~1e-15 m. Anything larger is an ordering error.
_C_TOL_M = 1e-6
# Footprint polygons are simplified to this tolerance (m); image edges are straight lines on the
# ground for a pinhole camera, so this only removes collinear boundary samples.
_FOOTPRINT_SIMPLIFY_M = 0.002
# Image-edge sampling step (px) for the footprint; fine enough to follow lens distortion.
_FOOTPRINT_STEP_PX = 8

# Reason codes. A chosen frame's reason is indexed by its tier (2 x low_light + over_blur_limit).
CHOSEN_REASONS = ("sharpest", "sharpest_over_blur_limit", "only_low_light", "only_low_light_over_blur_limit")
# The same tiers under segments.choice = nearest_sharp (the low-light names are shared: they say why the
# frame was used, not how it was picked; quality.choice says that).
NEAREST_REASONS = ("nearest_sharp", "nearest_sharp_over_blur_limit", "only_low_light", "only_low_light_over_blur_limit")
CHOICES = ("sharpest", "nearest_sharp")
# Why a frame inside a window is not a candidate, in priority order (one reason per frame).
EXCLUSION_REASONS = ("gnss_gap", "black", "reversing", "standing", "low_light")
# Why a window saw no frame at all. no_frame_in_window: consecutive frames straddled the window
# (speed x frame period > window_m, about 108 km/h at 30 fps with the 1 m default).
EMPTY_REASONS = ("no_frames", "before_recording", "after_recording", "no_frame_in_window")


def _seg_cfg(cfg: Config | SegmentConfig) -> SegmentConfig:
    """The `segments` section, from either the full Config or the section itself."""
    return cfg.segments if isinstance(cfg, Config) else cfg


def _finite(x) -> float:
    """float(x), or NaN for missing values (None, NaN, pandas NA, non-numeric)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _json_num(x: float | None) -> float | None:
    """A float for a JSON record: None where the value is unknown (NaN)."""
    return None if x is None or not math.isfinite(x) else float(x)


# ---- planning --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """Road segment [s0, s1) in path distance (m). `partial`: the shorter last segment of a run."""
    k: int
    s0: float
    s1: float
    partial: bool = False

    @property
    def length_m(self) -> float:
        return self.s1 - self.s0


def plan_segments(traj: Trajectory, cfg: Config | SegmentConfig, *, t_span: tuple[float, float] | None = None,
                  offset_m: float = 0.0) -> list[Segment]:
    """Consecutive segments of `length_m` from the first valid trajectory position to the last.

    A final remainder is kept as a `partial` segment when it is at least half a segment long;
    shorter remainders are dropped (a classification of under half a segment would be weighted like
    a whole one in "% of length" statistics). Segments are not split at GNSS gaps: a segment inside a
    gap simply gets no candidate frame and is reported as not covered. Positions are
    s0 = start + k x length_m (no accumulated rounding); the partial segment ends at the last position.

    With the defaults this is the spec's plan (§3). Two optional arguments make it the plan of the
    road the RECORDING can cover; the orchestrator should pass both:
    t_span: (first, last) frame time in trajectory seconds (extract._RowClock of the first and last
        frame's centre row). GNSS usually starts before the recording and stops after it; without
        t_span the segments of those stretches are planned and reported as not covered.
    offset_m: shift of the whole plan along s; use `plan_offset_m(geoms, cfg)` = near_m + window_m.
        The camera at c sees road from c + near_m on, so segment k's candidate window is
        [s0 - near_m - window_m, s0 - near_m]. Unshifted, the first segment's window lies before the
        first frame and it can never be covered; shifted by near_m + window_m, the first window starts
        at the first recorded position and every window of the plan lies inside the recording.
    """
    L = float(_seg_cfg(cfg).length_m)
    if not L > 0:
        raise ValueError(f"segment length must be positive, got {L}")
    if not math.isfinite(offset_m):
        raise ValueError(f"offset_m must be finite, got {offset_m}")
    t_lo, t_hi = (float(traj.t[0]), float(traj.t[-1])) if t_span is None else (float(t_span[0]), float(t_span[1]))
    if not t_lo <= t_hi:
        raise ValueError(f"t_span must be (first, last) with first <= last, got {t_span}")
    ok = traj.valid & (traj.t >= t_lo) & (traj.t <= t_hi)
    if not ok.any():
        return []
    # Exact positions at the span ends when they are valid (the trajectory grid is 10 ms apart).
    p = traj.sample(np.array([t_lo, t_hi]))
    s_first = float(p["s"][0]) if p["valid"][0] else float(traj.s[ok][0])
    s_last = float(p["s"][1]) if p["valid"][1] else float(traj.s[ok][-1])
    start = s_first + float(offset_m)
    total = s_last - s_first
    n_full = int(math.floor(total / L + 1e-9))
    segs = [Segment(k, start + k * L, start + (k + 1) * L, False) for k in range(n_full)]
    if total - n_full * L >= 0.5 * L - 1e-9:
        segs.append(Segment(n_full, start + n_full * L, start + total, True))
    return segs


def _view_range(geoms: Sequence["SegmentGeometry"]) -> tuple[float, float]:
    """(near_m, far_m) valid for every recording size: the largest near and the smallest far."""
    if not geoms:
        raise ValueError("at least one SegmentGeometry is needed")
    return max(g.near_m for g in geoms), min(g.far_m for g in geoms)


def plan_offset_m(geoms: "SegmentGeometry | Sequence[SegmentGeometry]", cfg: Config | SegmentConfig) -> float:
    """near_m + window_m: the `offset_m` that gives every planned segment a whole candidate window
    inside the recording (see plan_segments). With several recording sizes, the largest near_m."""
    g = [geoms] if isinstance(geoms, SegmentGeometry) else list(geoms)
    return _view_range(g)[0] + float(_seg_cfg(cfg).window_m)


# ---- geometry --------------------------------------------------------------------------------

@dataclass
class SegmentGeometry:
    """Where a segment appears in a full-resolution frame of one recording size (docs/pipeline/13 §2)."""
    model: CameraModel          # full resolution, display orientation
    near_m: float               # closest visible road: bottom usable row, centre column
    far_m: float                # farthest visible road: top row or horizon, at most FAR_CAP_M
    far_capped: bool            # far_m is the cap (horizon in view, or top row beyond the cap)
    v_usable: int               # lowest usable image row (GeometryConfig.bottom_exclude_frac)

    @classmethod
    def build(cls, model: CameraModel, cfg: Config) -> "SegmentGeometry":
        """Geometry of `model` (full resolution, display orientation, e.g. extract.nominal_model).

        The lowest usable row is the one plan_view uses: v <= H x (1 - bottom_exclude_frac) - 1.
        Raises ValueError when segment mode cannot work with this view: no road at the bottom row,
        a rearward camera (segments are defined ahead of the camera), or a view too short to hold a
        whole segment (`length_m` > far_m - near_m).
        """
        W, H = model.width, model.height
        v_use = int(math.floor(H * (1.0 - cfg.geometry.bottom_exclude_frac) - 1))
        if v_use < 1:
            raise ValueError(f"bottom_exclude_frac {cfg.geometry.bottom_exclude_frac} leaves no usable rows")
        uc = (W - 1) / 2
        _, y, ok = model.pixel_to_ground(np.array([uc, uc]), np.array([float(v_use), 0.0]))
        if not ok[0]:
            raise ValueError("the bottom image row sees no road: check the mount pitch")
        if y[0] <= 0:
            raise ValueError("segment mode needs a forward-looking camera (the bottom row sees road behind it)")
        near = float(y[0])
        far_top = float(y[1]) if ok[1] else math.inf  # top row above the horizon: road to the horizon in view
        far = min(far_top, FAR_CAP_M)
        L = float(cfg.segments.length_m)
        if far - near < L:
            raise ValueError(f"a {L:g} m segment never fits in view: road visible only {near:.2f}..{far:.2f} m ahead")
        return cls(model, near, far, far_top >= FAR_CAP_M, v_use)

    @property
    def width(self) -> int:
        return self.model.width

    @property
    def height(self) -> int:
        return self.model.height

    @property
    def u_centre(self) -> float:
        return (self.model.width - 1) / 2

    @property
    def f_px(self) -> float:
        """Focal length in pixels for rotation blur: the mean of fx and fy (they differ by <0.1 % on the S21)."""
        return float(self.model.K[0, 0] + self.model.K[1, 1]) / 2

    def rows_for(self, d0, d1):
        """(v_top, v_bottom): display rows of ground distances d1 and d0 ahead on the centre line (x = 0).

        Floats, not clipped to the image: at d0 = near_m the centre line and the centre column that
        defines near_m differ by a pixel or two (roll, principal point), so v_bottom can be just
        below the bottom row. Scalars give floats, arrays give arrays. NaN behind the camera.
        """
        d0a, d1a = np.asarray(d0, dtype=np.float64), np.asarray(d1, dtype=np.float64)
        if np.any(~(d1a > d0a)):
            raise ValueError("rows_for needs d0 < d1 (near edge first)")
        _, v_bottom, _ = self.model.ground_to_pixel(np.zeros_like(d0a), d0a)
        _, v_top, _ = self.model.ground_to_pixel(np.zeros_like(d1a), d1a)
        if np.ndim(v_top) == 0:
            return float(v_top), float(v_bottom)
        return v_top, v_bottom

    def row_slice(self, v_top: float, v_bottom: float) -> tuple[int, int]:
        """Half-open integer rows [r0, r1) whose centres lie in [v_top, v_bottom], within the usable image."""
        if not (math.isfinite(v_top) and math.isfinite(v_bottom)):
            raise ValueError("segment rows are not finite (segment behind the camera?)")
        r0 = max(0, math.ceil(v_top))
        r1 = min(self.v_usable + 1, math.floor(v_bottom) + 1)
        return r0, max(r0, r1)

    def gsd_along_at(self, row):
        """Along-track ground size of one pixel (mm) at display row(s) `row`, centre column."""
        v = np.asarray(row, dtype=np.float64)
        along, _ = self.model.gsd_mm(np.full_like(v, self.u_centre), v)
        return float(along) if np.ndim(along) == 0 else along

    @cached_property
    def _image_footprint(self) -> Polygon:
        """Ground polygon of the usable image (outermost pixel centres), cut where rows reach 2 x FAR_CAP_M.

        Image rows map to straight ground lines, and along a row the distance ahead is monotonic, so
        a row whose two end pixels see the ground within the bound lies wholly within it. Rows are
        good from the first such row down to the bottom (the camera looks down at the road).
        """
        m = self.model
        W = m.width
        rows = np.arange(self.v_usable + 1, dtype=np.float64)
        _, yl, okl = m.pixel_to_ground(np.zeros_like(rows), rows)
        _, yr, okr = m.pixel_to_ground(np.full_like(rows, W - 1.0), rows)
        good = okl & okr & (yl <= 2 * FAR_CAP_M) & (yr <= 2 * FAR_CAP_M)
        if not good[-1]:
            raise ValueError("the bottom usable row does not see the road across the whole width")
        v_a = float(rows[int(np.argmax(good))])
        st = _FOOTPRINT_STEP_PX
        us = np.unique(np.r_[np.arange(0, W - 1, st), W - 1]).astype(np.float64)
        vs = np.unique(np.r_[np.arange(v_a, self.v_usable, st), self.v_usable]).astype(np.float64)
        u = np.concatenate([us, np.full(len(vs), W - 1.0), us[::-1], np.zeros(len(vs))])
        v = np.concatenate([np.full(len(us), v_a), vs, np.full(len(us), float(self.v_usable)), vs[::-1]])
        x, y, ok = m.pixel_to_ground(u, v)
        poly = Polygon(np.column_stack([x[ok], y[ok]]))
        return poly if poly.is_valid else poly.buffer(0)

    def footprint(self, d0: float, d1: float) -> np.ndarray:
        """Ground polygon of the segment as seen: the strip d0 <= y <= d1 inside the usable image.

        Returns the closed ring (first vertex repeated last) of (x, y) ground points, metres from the
        camera nadir, counter-clockwise (x right, y ahead). Rotating into (east, north) by the camera
        heading is a proper rotation, so the ring stays counter-clockwise as RFC 7946 wants for an
        exterior ring. The width is what the frame shows at each distance; parts of the strip below
        the bottom row (d0 < near_m) are not in the frame and not in the polygon, and near the bottom
        row a rolled camera cuts one corner, which the polygon follows. For the record pass the chosen
        frame's (near_edge_ahead_m, far_edge_ahead_m).
        """
        if not d1 > d0:
            raise ValueError("footprint needs d0 < d1")
        fp = self._image_footprint
        x0, _, x1, _ = fp.bounds
        part = fp.intersection(box(x0 - 1.0, d0, x1 + 1.0, d1))
        if part.is_empty or part.geom_type != "Polygon":
            raise ValueError(f"segment {d0:.2f}..{d1:.2f} m ahead is not visible as one region")
        part = orient(part.simplify(_FOOTPRINT_SIMPLIFY_M, preserve_topology=True), sign=1.0)
        return np.asarray(part.exterior.coords, dtype=np.float64)

    def summary(self, length_m: float) -> dict:
        """For report.json: the view, and a segment whose near edge is at near_m (the nearest candidate)."""
        v_top, v_bottom = self.rows_for(self.near_m, self.near_m + length_m)
        r0, r1 = self.row_slice(v_top, v_bottom)
        return {"width": self.width, "height": self.height, "near_m": self.near_m, "far_m": self.far_m,
                "far_capped": self.far_capped, "usable_bottom_row": self.v_usable, "f_px": self.f_px,
                "nearest_segment": {"ahead_m": [self.near_m, self.near_m + length_m], "rows": [v_top, v_bottom],
                                    "row_slice": [r0, r1],
                                    "gsd_along_mm_centre_row": self.gsd_along_at((r0 + r1 - 1) / 2),
                                    "gsd_along_mm_range": [self.gsd_along_at(r1 - 1), self.gsd_along_at(r0)]},
                "camera": self.model.to_dict()}


def _window(seg: Segment, near_m: float, far_m: float, window_m: float) -> tuple[float, float]:
    if not window_m >= 0:
        raise ValueError(f"window_m must be >= 0, got {window_m}")
    hi = seg.s0 - near_m
    lo = max(hi - window_m, seg.s1 - far_m)
    if lo > hi:
        raise ValueError(f"segment {seg.k} ({seg.length_m:.2f} m) never fits in view ({near_m:.2f}..{far_m:.2f} m)")
    return lo, hi


def candidate_window(seg: Segment, geom: SegmentGeometry, cfg: Config | SegmentConfig) -> tuple[float, float]:
    """Camera positions (lo, hi), inclusive, of the frames that are candidates for `seg`.

    hi = s0 - near_m: the segment's near edge is at the bottom of the frame. lo = hi - window_m, or
    s1 - far_m if that is later (the far edge must be in view too; it only binds when far_m - near_m
    < length_m + window_m).
    """
    return _window(seg, geom.near_m, geom.far_m, _seg_cfg(cfg).window_m)


# ---- frame quality ---------------------------------------------------------------------------

def sharpness(rgb: np.ndarray, rows: tuple[int, int], *, cols: tuple[float, float], downscale: float) -> float:
    """Normalised Tenengrad of rows [r0, r1) and the column band `cols` (fractions of the width).

    mean(Gx² + Gy²) / mean(I)², with 3x3 Sobel gradients of the grey image (BT.601 weights, float, so
    no rounding before the gradients), after INTER_AREA downscaling by `downscale` (<= 1). Grey
    conversion and area averaging are both linear, so cutting the region first gives the same image
    as cutting it from a downscaled whole frame (up to the region's border pixels) at a fraction of
    the cost. Only pixels whose 3x3 neighbourhood lies inside the region count, so the region's
    borders add nothing. Dividing by the squared mean makes it independent of exposure and gain:
    I -> kI scales both by k². The column band keeps the lane and leaves out the roadside
    (vegetation, parked vehicles), whose texture would otherwise dominate. 0 for a black region.
    Float, not 8-bit grey: 8-bit BT.601 grey is ~3x faster but chose another frame in 2 of the 813
    segments of the S21 sessions of 26-27 Oct (gate/speed analysis of 28 Oct 2026), so it was not adopted.
    """
    c_lo, c_hi = cols
    if not 0.0 <= c_lo < c_hi <= 1.0:
        raise ValueError(f"sharpness columns must satisfy 0 <= lo < hi <= 1, got {cols}")
    if not 0.0 < downscale <= 1.0:
        raise ValueError(f"sharpness downscale must be in (0, 1], got {downscale}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"sharpness needs an H x W x 3 RGB image, got {rgb.shape}")
    H, W = rgb.shape[:2]
    r0, r1 = max(0, int(rows[0])), min(H, int(rows[1]))
    c0, c1 = int(round(c_lo * W)), int(round(c_hi * W))
    if r1 - r0 < 3 or c1 - c0 < 3:
        raise ValueError(f"sharpness region too small: rows {r0}..{r1}, columns {c0}..{c1}")
    grey = cv2.cvtColor(rgb[r0:r1, c0:c1].astype(np.float32), cv2.COLOR_RGB2GRAY)
    if downscale < 1.0:
        h, w = grey.shape
        size = (max(1, int(round(w * downscale))), max(1, int(round(h * downscale))))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        raise ValueError(f"sharpness region too small ({grey.shape[1]} x {grey.shape[0]} px after downscaling)")
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1].astype(np.float64)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1].astype(np.float64)
    mean_i = float(grey[1:-1, 1:-1].mean(dtype=np.float64))
    if mean_i <= 0.0:
        return 0.0
    return float(np.mean(gx * gx + gy * gy)) / (mean_i * mean_i)


def _tenengrad(grey: np.ndarray) -> float:
    """mean(Gx² + Gy²) / mean(I)² of a float32 grey image, 3x3 Sobel, border pixels excluded (as `sharpness`)."""
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1].astype(np.float64)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1].astype(np.float64)
    mean_i = float(grey[1:-1, 1:-1].mean(dtype=np.float64))
    if mean_i <= 0.0:
        return 0.0
    return float(np.mean(gx * gx + gy * gy)) / (mean_i * mean_i)


class FairSharpness:
    """Sharpness of the same piece of road at the same ground resolution in every candidate frame.

    The raw `sharpness` of the segment's rows depends on scale: a farther frame shows the road in fewer
    rows, and its per-pixel gradients are larger for the same image quality. Here every candidate is
    resampled onto one ground grid, so the score compares blur alone:
    - the strip: the segment's first `fair_strip_m` (ground y in [d0, d0 + strip], d0 = s0 - c, the near
      edge ahead of the camera), |x| <= half width;
    - the grid: `fair_grid_mm` cells. The image is sampled bilinearly at grid / `fair_supersample` and each
      cell is the mean of its supersample² samples (area averaging), so no frame aliases. The grid is
      coarser than the full-resolution pixel everywhere in the strip of every candidate (S21: along-track
      0.8-3.4 mm), so no frame is upsampled either;
    - the score: the normalised Tenengrad of that grid image (`sharpness`'s formula), exposure-invariant.
    The half width is capped by what the bottom row of the frame shows (the nearest candidate's strip
    starts there), so every grid point lies inside the image in every candidate: no image border ever
    enters the gradients (a constant border would add a strong edge that grows as the frame gets
    nearer).

    The sampling map is computed once per recording size, for strips starting anywhere in
    [near_m, near_m + window_m]; a candidate's strip is the block of map rows starting at the grid row
    nearest to d0 (within half a sample, 0.5 mm). Ground points are on the flat road, as everywhere in
    segment mode.
    """

    def __init__(self, geom: SegmentGeometry, near_m: float, cfg: Config | SegmentConfig):
        c = _seg_cfg(cfg)
        k = int(c.fair_supersample)
        if k < 1 or not c.fair_grid_mm > 0 or not c.fair_strip_m > 0 or not c.fair_half_width_m > 0:
            raise ValueError("fair sharpness needs fair_supersample >= 1 and positive grid, strip and half width")
        self.k = k
        self.step_m = c.fair_grid_mm / 1000.0 / k
        self.near_m = float(near_m)
        self.n_strip = k * max(3, int(round(c.fair_strip_m / (c.fair_grid_mm / 1000.0))))
        n_y = self.n_strip + int(math.ceil(c.window_m / self.step_m)) + 1
        m = geom.model
        # Half width: what both bottom corners of the usable image see (the narrowest view of the strip).
        xb, _, okb = m.pixel_to_ground(np.array([0.0, m.width - 1.0]), np.array([float(geom.v_usable)] * 2))
        if not okb.all():
            raise ValueError("the bottom row's corners see no road")
        hw = min(float(c.fair_half_width_m), float(-xb[0]), float(xb[1])) - c.fair_grid_mm / 1000.0
        n_cells = int(math.floor(2 * hw / (c.fair_grid_mm / 1000.0)))
        if n_cells < 3:
            raise ValueError(f"fair sharpness strip too narrow ({2 * hw:.2f} m)")
        n_x = n_cells * k
        xs = (np.arange(n_x) - (n_x - 1) / 2.0) * self.step_m
        ys = self.near_m + (np.arange(n_y) + 0.5) * self.step_m
        X, Y = np.meshgrid(xs, ys)
        u, v, ok = m.ground_to_pixel(X, Y)
        inside = ok & (u >= 0) & (u <= m.width - 1) & (v >= 0) & (v <= geom.v_usable)
        if not inside.all():
            raise ValueError("fair sharpness strip leaves the image (camera roll?)")
        # The block of the frame the map reads, so each call converts only that block to grey.
        self.u0, self.v0 = int(math.floor(u.min())), int(math.floor(v.min()))
        self.u1, self.v1 = min(m.width, int(math.ceil(u.max())) + 2), min(geom.v_usable + 1, int(math.ceil(v.max())) + 2)
        self.map_u = (u - self.u0).astype(np.float32)
        self.map_v = (v - self.v0).astype(np.float32)
        self.half_width_m = n_cells * c.fair_grid_mm / 2000.0
        self.size = (m.width, m.height)

    def start_row(self, d0: float) -> int:
        """First map row of the strip whose near edge is d0 metres ahead (clipped to the map)."""
        i = int(round((float(d0) - self.near_m) / self.step_m))
        return min(max(i, 0), self.map_u.shape[0] - self.n_strip)

    def grid(self, rgb: np.ndarray, d0: float) -> np.ndarray:
        """The strip on the common ground grid (float32 grey, far edge in row 0)."""
        if rgb.shape[1] != self.size[0] or rgb.shape[0] != self.size[1]:
            raise ValueError(f"fair sharpness for {self.size[0]}x{self.size[1]} frames, got {rgb.shape[1]}x{rgb.shape[0]}")
        i0 = self.start_row(d0)
        block = cv2.cvtColor(np.ascontiguousarray(rgb[self.v0:self.v1, self.u0:self.u1]), cv2.COLOR_RGB2GRAY)
        fine = cv2.remap(block.astype(np.float32), self.map_u[i0:i0 + self.n_strip], self.map_v[i0:i0 + self.n_strip],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        h, w = fine.shape
        cells = fine.reshape(h // self.k, self.k, w // self.k, self.k).mean(axis=(1, 3), dtype=np.float64)
        return cells[::-1].astype(np.float32)  # map rows run near -> far; image rows far -> near

    def __call__(self, rgb: np.ndarray, d0: float) -> float:
        return _tenengrad(self.grid(rgb, d0))


def predicted_blur(speed_mps: float, exposure_ns, gsd_along_mm: float, omega_perp_rad_s: float,
                   f_px: float) -> dict[str, float]:
    """Predicted blur (px) during one exposure (docs/pipeline/13 §3).

    motion = speed x exposure / along-track ground resolution (the road moves through the image
    along the track); rotation = |ω⊥| x exposure x focal length (a rotation of δθ shifts the image
    centre by f δθ); blur = sqrt(motion² + rotation²). Keys match the record's `quality` fields.
    Unknowns are NaN: no exposure makes every value NaN; an unknown rotation (no gyro) leaves
    blur_rotation_px NaN and the total is the motion part alone, so the motion limit still applies.
    """
    e_s = _finite(exposure_ns) / 1e9
    motion = _finite(speed_mps) * e_s / (_finite(gsd_along_mm) / 1000.0)
    rotation = abs(_finite(omega_perp_rad_s)) * e_s * _finite(f_px)
    known = [v for v in (motion, rotation) if math.isfinite(v)]
    total = math.sqrt(sum(v * v for v in known)) if known else math.nan
    return {"blur_motion_px": motion, "blur_rotation_px": rotation, "blur_px": total}


class GyroImageRates:
    """|ω⊥|: the camera's angular rate about its two image axes (rad/s), from imu.csv gyro samples.

    Axes. imu.csv `gyro` is rad/s about the DEVICE axes (Android sensor frame: x right, y up, z out of
    the screen, phone in its natural orientation), counter-clockwise positive, i.e. right-handed.
    `cam_from_device` (geometry.cam_from_device(sensor_orientation)) has the camera axes as rows, in
    device coordinates: x = image right, y = image down, z = optical axis (device -z for the back
    camera); it is a proper rotation (det +1). Angular velocity is an axial vector, and a proper
    rotation transforms it like any vector: ω_cam = R ω_dev. A rotation about camera x moves the
    image vertically (pitch), about camera y horizontally (pan); a rotation about z turns the image
    around the principal point and is not part of ω⊥ (spec §3). Because the optical axis is device
    ±z for every sensor orientation, |ω⊥| = hypot(ω_dev_x, ω_dev_y); the rotation is applied anyway so
    the per-axis rates (wx = pitch, wy = pan) are right too. The 180° display rotation of an
    UPSIDE_DOWN mount negates both image axes, so |ω⊥| does not depend on it.

    Times are trajectory seconds (t_ns - t0_ns) / 1e9, the axis of extract._RowClock and
    Trajectory.sample. Rates are interpolated linearly between samples (both components, then the
    norm); NaN outside the samples and inside gaps longer than `max_gap_s` (exactly at a sample: its
    value).
    """

    def __init__(self, imu: pd.DataFrame | None, cam_from_device: np.ndarray, t0_ns: int,
                 max_gap_s: float = GYRO_MAX_GAP_S):
        self.max_gap_s = float(max_gap_s)
        if imu is None or imu.empty or "sensor" not in imu.columns:
            g = pd.DataFrame(columns=["t_ns", "x", "y", "z"])
        else:
            g = imu[imu["sensor"] == "gyro"]
        g = g.dropna(subset=["t_ns", "x", "y", "z"]).sort_values("t_ns", kind="stable")
        t = (g["t_ns"].to_numpy(dtype=np.int64) - np.int64(t0_ns)) / 1e9
        w_dev = g[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1, 3)
        keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.zeros(0, bool)
        w_cam = w_dev[keep] @ np.asarray(cam_from_device, dtype=np.float64).T
        self.t = t[keep]
        self.wx = w_cam[:, 0]          # about image x (right): vertical image motion
        self.wy = w_cam[:, 1]          # about image y (down): horizontal image motion

    def __call__(self, t) -> np.ndarray:
        """|ω⊥| (rad/s) at trajectory time(s) `t`; always an array."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.full(t.shape, np.nan)
        n = len(self.t)
        if n == 0:
            return out
        if n == 1:
            hit = t == self.t[0]
            out[hit] = math.hypot(self.wx[0], self.wy[0])
            return out
        i = np.clip(np.searchsorted(self.t, t, side="right"), 1, n - 1)
        exact = (self.t[i - 1] == t) | (self.t[i] == t)
        ok = (t >= self.t[0]) & (t <= self.t[-1]) & ((self.t[i] - self.t[i - 1] <= self.max_gap_s) | exact)
        wx = np.interp(t, self.t, self.wx)
        wy = np.interp(t, self.t, self.wy)
        out[ok] = np.hypot(wx, wy)[ok]
        return out


# ---- best-frame selection --------------------------------------------------------------------

@dataclass
class FrameInfo:
    """What the selector needs to know about one decoded frame; the caller fills it per frame.

    t, c: the exposure-middle time of the image's centre row (extract._RowClock at display row
    (H - 1) / 2), and the trajectory s there. speed_mps, valid, reversing: Trajectory.sample at t.
    black: inside Session.black_intervals_ns(); low_light: Session.low_light_at(t_ns).
    exposure_ns: frames.csv exposure_ns (None/NA if unknown); omega_perp_rad_s: GyroImageRates at t.
    ref: the caller's handle for the frame (e.g. segment, index, frames.csv row), returned untouched.
    """
    t: float
    c: float
    speed_mps: float
    valid: bool
    reversing: bool
    black: bool
    low_light: bool
    exposure_ns: Any = None
    omega_perp_rad_s: float = math.nan
    ref: Any = None


@dataclass
class FrameScore:
    """A candidate frame scored for one segment."""
    info: FrameInfo
    width: int
    height: int
    rows: tuple[float, float]        # (v_top, v_bottom) of the segment, display rows, unclipped
    row_slice: tuple[int, int]       # [r0, r1): the visible rows sharpness was measured on
    near_edge_ahead_m: float         # s0 - c
    far_edge_ahead_m: float          # s1 - c
    sharpness: float
    gsd_along_mm: float              # at the centre of the visible segment rows
    blur_motion_px: float
    blur_rotation_px: float
    blur_px: float
    over_blur_limit: bool
    tier: int                        # 2 x low_light + over_blur_limit; lower is preferred
    fair_sharpness: float = math.nan  # FairSharpness of the segment's near strip (choice nearest_sharp only)

    def key(self) -> tuple[int, float, float]:
        """Sort key, best first: tier, then sharpest, then nearest (largest c: the segment is closest).
        On a full tie (only possible at an identical position) the frame offered first is kept."""
        return self.tier, -self.sharpness, -self.info.c


@dataclass
class SegmentResult:
    """A finished segment: its best frame (None when no frame qualified) and why."""
    segment: Segment
    window: tuple[float, float]
    best: FrameScore | None
    rgb: np.ndarray | None           # the best frame, full resolution (None without one)
    n_candidates: int
    sharpness_rank: int | None       # 1 = the sharpest candidate; >1 when a sharper one was filtered
    reason: str                      # CHOSEN_REASONS / NEAREST_REASONS, EXCLUSION_REASONS or EMPTY_REASONS
    excluded: dict[str, int] = field(default_factory=dict)  # frames in the window that were not candidates
    choice: str = "sharpest"         # SegmentConfig.choice that picked the frame
    fair_rel: float | None = None    # nearest_sharp: the chosen frame's fair sharpness / the best in its tier

    @property
    def covered(self) -> bool:
        return self.best is not None

    def quality(self) -> dict:
        """The record's `quality` section (schema geoai.segment.v1), JSON-ready (unknown -> None).

        With choice nearest_sharp it also has the optional fields `choice`, `fair_sharpness` and
        `fair_sharpness_rel` (docs/pipeline/13 §5); with the default choice it is exactly the spec's eight.
        """
        b = self.best
        q = {
            "sharpness": _json_num(b.sharpness) if b else None,
            "sharpness_rank": self.sharpness_rank,
            "n_candidates": self.n_candidates,
            "blur_motion_px": _json_num(b.blur_motion_px) if b else None,
            "blur_rotation_px": _json_num(b.blur_rotation_px) if b else None,
            "blur_px": _json_num(b.blur_px) if b else None,
            "gyro_rad_s": _json_num(_finite(b.info.omega_perp_rad_s)) if b else None,
            "reason": self.reason,
        }
        if self.choice != "sharpest":
            q.update({"choice": self.choice, "fair_sharpness": _json_num(b.fair_sharpness) if b else None,
                      "fair_sharpness_rel": _json_num(self.fair_rel) if b and self.fair_rel is not None else None})
        return q


@dataclass
class _Open:
    best: FrameScore | None = None
    rgb: np.ndarray | None = None
    sharp: list[float] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)
    # nearest_sharp: the best tier seen, its highest fair sharpness, and the frames of that tier that can
    # still be chosen, in arrival (= distance) order, each with a higher fair sharpness than every later one.
    tier: int | None = None
    fair_max: float = -math.inf
    kept: list[tuple[FrameScore, np.ndarray]] = field(default_factory=list)


class BestFrameSelector:
    """Streaming choice of the best frame per segment, in one decode pass (docs/pipeline/13 §3).

    Protocol, for every decoded frame in time order:
        need_rgb = sel.wants(info)          # in a window and not excluded: convert this frame to RGB
        sel.offer(info, rgb or None)        # every frame; RGB only where wants() said so
        for res in sel.pop_ready(info.c): ...   # segments whose window c has passed, in order
    and `flush()` at the end, which emits every remaining segment. Each segment is emitted exactly
    once, in k order, with or without a frame. Frame times must increase strictly (a frame offered
    twice is an error) and positions must not decrease (path distance s is monotonic in time).

    Choice within a window: tier first (not low-light before low-light; within each, predicted blur
    <= max_blur_px before above it), then, with choice `sharpest` (default), the highest sharpness, then
    the nearest frame (largest c). With choice `nearest_sharp`, within the best tier present: the
    nearest frame whose FairSharpness is >= fair_rel_min x the highest FairSharpness of that tier (ties
    at one position: the frame offered last). Low-light frames are excluded outright with skip_low_light. Frames in GNSS gaps, black
    intervals, while reversing or below min_speed_mps (standing still: no progress along s) are never
    candidates; they are counted, and a window with only such frames is reported with the most
    frequent of those reasons.

    Memory: per open segment the best frame's RGB (by reference: callers must not write into an
    offered array afterwards) and the candidates' sharpness values (for the rank). A segment is open
    from its first frame until c passes its window, so with window_m < length_m at most one frame is
    held; with overlapping windows ceil(window_m / length_m) + 1. With nearest_sharp a segment keeps
    the frames that can still be chosen: those within fair_rel_min of the best so far that no nearer
    frame matches. That is usually one or two; at worst every candidate of the window.

    Several recording sizes (e.g. a thermal step-down from 4K to 1080p between video segments) are
    handled by passing one SegmentGeometry per size; each frame is measured with the geometry of its
    own size (`geometry_for`). Windows then use the largest near_m and the smallest far_m, so a
    candidate shows the whole segment at every size.
    """

    def __init__(self, segments: Sequence[Segment], geoms: SegmentGeometry | Sequence[SegmentGeometry],
                 cfg: Config | SegmentConfig):
        self.cfg = _seg_cfg(cfg)
        if self.cfg.choice not in CHOICES:
            raise ValueError(f"segments.choice must be one of {CHOICES}, got {self.cfg.choice!r}")
        if self.cfg.choice == "nearest_sharp" and not 0.0 < self.cfg.fair_rel_min <= 1.0:
            raise ValueError(f"segments.fair_rel_min must be in (0, 1], got {self.cfg.fair_rel_min}")
        geom_list = [geoms] if isinstance(geoms, SegmentGeometry) else list(geoms)
        self.near_m, self.far_m = _view_range(geom_list)
        self._geoms: dict[tuple[int, int], SegmentGeometry] = {(g.width, g.height): g for g in geom_list}
        if len(self._geoms) != len(geom_list):
            raise ValueError("one SegmentGeometry per recording size")
        self._fair: dict[tuple[int, int], FairSharpness] = {}
        if self.cfg.choice == "nearest_sharp":
            self._fair = {size: FairSharpness(g, self.near_m, self.cfg) for size, g in self._geoms.items()}
        self.segments = list(segments)
        if [s.k for s in self.segments] != list(range(len(self.segments))):
            raise ValueError("segments must be numbered 0..n-1 in order")
        w = np.array([_window(s, self.near_m, self.far_m, self.cfg.window_m) for s in self.segments],
                     dtype=np.float64).reshape(-1, 2)
        if np.any(np.diff(w, axis=0) < 0):
            raise ValueError("segments must be in path order (windows must not go backwards)")
        self.windows = w
        self._lo, self._hi = w[:, 0].copy(), w[:, 1].copy()
        self._next = 0
        self._open: dict[int, _Open] = {}
        self._t_last = -math.inf
        self._c_last = -math.inf
        self._c_first: float | None = None
        self._flushed = False

    # ---- queries -------------------------------------------------------------------------------
    @property
    def n_open(self) -> int:
        """Segments that have seen a frame and are not emitted yet."""
        return len(self._open)

    @property
    def held_frames(self) -> int:
        """Distinct RGB frames currently held (the memory the selector keeps alive)."""
        held = {id(o.rgb) for o in self._open.values() if o.rgb is not None}
        held |= {id(rgb) for o in self._open.values() for _, rgb in o.kept}
        return len(held)

    @property
    def n_emitted(self) -> int:
        """Segments emitted so far (they are emitted in k order, so these are 0..n_emitted-1)."""
        return self._next

    def geometry_for(self, width: int, height: int) -> SegmentGeometry:
        """The SegmentGeometry of a recording size (e.g. FrameScore.width/height of a chosen frame)."""
        g = self._geoms.get((int(width), int(height)))
        if g is None:
            raise KeyError(f"no SegmentGeometry for {width}x{height} (have {sorted(self._geoms)})")
        return g

    def _in_windows(self, c: float) -> range:
        """Indices of not-yet-emitted segments whose window contains c (lo and hi are non-decreasing)."""
        a = max(self._next, int(np.searchsorted(self._hi, c, side="left")))
        b = int(np.searchsorted(self._lo, c, side="right"))
        return range(a, max(a, b))

    def exclusion(self, info: FrameInfo) -> str | None:
        """Why this frame cannot be a candidate (EXCLUSION_REASONS), or None."""
        if not info.valid:
            return "gnss_gap"
        if info.black:
            return "black"
        if info.reversing:
            return "reversing"
        if not _finite(info.speed_mps) >= self.cfg.min_speed_mps:
            return "standing"
        if info.low_light and self.cfg.skip_low_light:
            return "low_light"
        return None

    def wants(self, x: float | FrameInfo) -> bool:
        """Whether a frame needs its RGB image: its camera position is in the window of a segment not
        emitted yet and, given a FrameInfo, it is not excluded. With a bare position c only the
        window is checked."""
        if self._flushed:
            return False
        if isinstance(x, FrameInfo):
            return self.exclusion(x) is None and len(self._in_windows(float(x.c))) > 0
        return len(self._in_windows(float(x))) > 0

    # ---- streaming -----------------------------------------------------------------------------
    def _advance(self, c: float) -> float:
        if not math.isfinite(c):
            raise ValueError(f"camera position must be finite, got {c}")
        if c < self._c_last - _C_TOL_M:
            raise ValueError(f"camera position went back from {self._c_last:.6f} to {c:.6f} m: "
                             "frames must be offered in time order")
        c = max(c, self._c_last)
        self._c_last = c
        if self._c_first is None:
            self._c_first = c
        return c

    def offer(self, info: FrameInfo, rgb: np.ndarray | None) -> int:
        """Consider one frame; returns the number of segments it is a candidate for (0 if none).

        Every decoded frame may be offered (frames outside all windows only advance the position).
        A candidate must come with its RGB image; excluded frames need none.
        """
        if self._flushed:
            raise RuntimeError("offer() after flush(): the selector is finished")
        t = _finite(info.t)
        if not t > self._t_last:
            raise ValueError(f"frame time {info.t} is not after the previous frame's {self._t_last}: "
                             "frames must be offered once each, in time order")
        c = self._advance(_finite(info.c))
        self._t_last = t
        ks = self._in_windows(c)
        if not ks:
            return 0
        reason = self.exclusion(info)
        if reason is not None:
            for k in ks:
                o = self._open.setdefault(k, _Open())
                o.excluded[reason] = o.excluded.get(reason, 0) + 1
            return 0
        if rgb is None:
            raise ValueError("a candidate frame was offered without its RGB image (wants() returned True)")
        geom = self._geoms.get((rgb.shape[1], rgb.shape[0])) if rgb.ndim == 3 else None
        if geom is None:
            raise ValueError(f"no SegmentGeometry for a {rgb.shape} frame (have {sorted(self._geoms)})")
        for k in ks:
            score = self._score(self.segments[k], info, c, geom, rgb)
            o = self._open.setdefault(k, _Open())
            o.sharp.append(score.sharpness)
            if self._fair:
                self._keep_nearest_sharp(o, score, rgb)
            elif o.best is None or score.key() < o.best.key():
                o.best, o.rgb = score, rgb
        return len(ks)

    def _keep_nearest_sharp(self, o: _Open, score: FrameScore, rgb: np.ndarray) -> None:
        """nearest_sharp bookkeeping for one candidate (frames arrive far to near).

        A frame of a worse tier than the best seen is never chosen; a better tier starts afresh. Within the
        tier, a kept frame is dropped when a nearer frame is at least as sharp (whenever the farther one
        qualifies, so does the nearer one) or when it falls below fair_rel_min x the tier's best, which only
        grows. The last kept frame is therefore the nearest one that qualifies so far.
        """
        if o.tier is not None and score.tier > o.tier:
            return
        if o.tier is None or score.tier < o.tier:
            o.tier, o.fair_max, o.kept = score.tier, -math.inf, []
        score.fair_sharpness = self._fair[(score.width, score.height)](rgb, score.near_edge_ahead_m)
        o.fair_max = max(o.fair_max, score.fair_sharpness)
        bar = self.cfg.fair_rel_min * o.fair_max
        o.kept = [(s, im) for s, im in o.kept if s.fair_sharpness > score.fair_sharpness and s.fair_sharpness >= bar]
        if score.fair_sharpness >= bar:
            o.kept.append((score, rgb))
        o.best, o.rgb = o.kept[-1]

    def _score(self, seg: Segment, info: FrameInfo, c: float, geom: SegmentGeometry, rgb: np.ndarray) -> FrameScore:
        cfg = self.cfg
        d0, d1 = seg.s0 - c, seg.s1 - c
        v_top, v_bottom = geom.rows_for(d0, d1)
        r0, r1 = geom.row_slice(v_top, v_bottom)
        sharp = sharpness(rgb, (r0, r1), cols=cfg.sharp_cols, downscale=cfg.sharp_downscale)
        gsd = float(geom.gsd_along_at((r0 + r1 - 1) / 2))
        blur = predicted_blur(info.speed_mps, info.exposure_ns, gsd, info.omega_perp_rad_s, geom.f_px)
        over = bool(blur["blur_px"] > cfg.max_blur_px)  # NaN (unknown exposure) never excludes
        return FrameScore(info=info, width=geom.width, height=geom.height, rows=(v_top, v_bottom),
                          row_slice=(r0, r1), near_edge_ahead_m=d0, far_edge_ahead_m=d1, sharpness=sharp,
                          gsd_along_mm=gsd, over_blur_limit=over, tier=2 * int(bool(info.low_light)) + int(over),
                          **blur)

    def pop_ready(self, c_now: float) -> list[SegmentResult]:
        """Segments whose window lies wholly behind camera position c_now, in order (each once)."""
        if self._flushed:
            return []
        c = self._advance(_finite(c_now))
        out = []
        while self._next < len(self.segments) and self._hi[self._next] < c:
            out.append(self._emit(self._next))
            self._next += 1
        return out

    def flush(self) -> list[SegmentResult]:
        """End of the recording: every segment not emitted yet, in order. The selector is then finished."""
        out = [self._emit(k) for k in range(self._next, len(self.segments))]
        self._next = len(self.segments)
        self._flushed = True
        return out

    def _emit(self, k: int) -> SegmentResult:
        seg = self.segments[k]
        window = (float(self._lo[k]), float(self._hi[k]))
        o = self._open.pop(k, None)
        if o is not None and o.best is not None:
            rank = 1 + sum(1 for s in o.sharp if s > o.best.sharpness)
            if self._fair:
                rel = o.best.fair_sharpness / o.fair_max if o.fair_max > 0 else 1.0
                return SegmentResult(seg, window, o.best, o.rgb, len(o.sharp), rank, NEAREST_REASONS[o.best.tier],
                                     dict(sorted(o.excluded.items())), self.cfg.choice, rel)
            return SegmentResult(seg, window, o.best, o.rgb, len(o.sharp), rank, CHOSEN_REASONS[o.best.tier],
                                 dict(sorted(o.excluded.items())))
        excluded = dict(sorted(o.excluded.items())) if o is not None else {}
        if excluded:
            # Most frequent exclusion; ties go to the earlier (more fundamental) reason.
            reason = max(EXCLUSION_REASONS, key=lambda r: (excluded.get(r, 0), -EXCLUSION_REASONS.index(r)))
        elif self._c_first is None:
            reason = "no_frames"
        elif window[1] < self._c_first:
            reason = "before_recording"
        elif window[0] > self._c_last:
            reason = "after_recording"
        else:
            reason = "no_frame_in_window"
        return SegmentResult(seg, window, None, None, 0, None, reason, excluded, self.cfg.choice)
