"""Procedural asphalt and distress drawing, for tests and the session simulator.

The road is a grey raster in a straight "road frame": column = lateral position (m, positive to
the right when driving along the route), row = distance along the route, `res_m` metres per
pixel. Distresses and look-alikes are drawn into it, each with its ground-truth footprint.
Every drawing operation works on a local window, so long rasters stay cheap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.uint64) * np.uint64(374761393) + iy.astype(np.uint64) * np.uint64(668265263)
         + np.uint64(seed) * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFFFF)).astype(np.float32) / np.float32(0x1000000)


def value_noise(x: np.ndarray, y: np.ndarray, cell: float, seed: int) -> np.ndarray:
    gx, gy = x / cell, y / cell
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = (gx - ix).astype(np.float32), (gy - iy).astype(np.float32)
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = ix + (1 << 20), iy + (1 << 20)  # keep lattice indices non-negative
    v00, v10 = _hash(ix, iy, seed), _hash(ix + 1, iy, seed)
    v01, v11 = _hash(ix, iy + 1, seed), _hash(ix + 1, iy + 1, seed)
    a = v00 + (v10 - v00) * sx
    b = v01 + (v11 - v01) * sx
    return a + (b - a) * sy


@dataclass
class Feature:
    kind: str                     # crack | alligator | pothole | patch | marking | shadow
    polygon_m: list               # [(lateral_m, along_m), ...] ground footprint
    is_distress: bool
    params: dict = field(default_factory=dict)


@dataclass
class RoadRaster:
    img: np.ndarray               # float32 grey, 0..1
    res_m: float
    lateral0_m: float             # lateral coordinate of column 0
    along0_m: float               # along-route coordinate of row 0
    features: list[Feature] = field(default_factory=list)

    def px(self, lateral_m, along_m):
        return ((np.asarray(lateral_m, float) - self.lateral0_m) / self.res_m,
                (np.asarray(along_m, float) - self.along0_m) / self.res_m)

    def window(self, pts_m: np.ndarray, pad_m: float):
        """Pixel window (r0, r1, c0, c1) around points, clipped to the raster."""
        c, r = self.px(pts_m[:, 0], pts_m[:, 1])
        pad = pad_m / self.res_m
        h, w = self.img.shape
        c0, c1 = int(max(0, math.floor(c.min() - pad))), int(min(w, math.ceil(c.max() + pad) + 1))
        r0, r1 = int(max(0, math.floor(r.min() - pad))), int(min(h, math.ceil(r.max() + pad) + 1))
        return r0, r1, c0, c1

    def local_px(self, pts_m: np.ndarray, win) -> np.ndarray:
        c, r = self.px(pts_m[:, 0], pts_m[:, 1])
        return np.stack([c - win[2], r - win[0]], -1)


def asphalt(width_m: float, length_m: float, res_m: float = 0.003, seed: int = 0,
            lateral0_m: float | None = None, along0_m: float = 0.0, base: float = 0.42) -> RoadRaster:
    w = int(round(width_m / res_m))
    h = int(round(length_m / res_m))
    lat0 = -width_m / 2 if lateral0_m is None else lateral0_m
    xs = lat0 + np.arange(w, dtype=np.float64) * res_m
    img = np.empty((h, w), np.float32)
    for r0 in range(0, h, 1024):
        r1 = min(h, r0 + 1024)
        ys = along0_m + np.arange(r0, r1, dtype=np.float64) * res_m
        X, Y = np.meshgrid(xs, ys)
        L = np.full(X.shape, base, np.float32)
        for cell, amp, k in ((0.8, 0.05, 1), (0.2, 0.03, 2), (0.05, 0.03, 3), (0.012, 0.05, 4)):
            L += amp * (value_noise(X, Y, cell, seed * 31 + k) - 0.5) * 2
        stones = value_noise(X, Y, 0.007, seed * 31 + 7)
        L += np.where(stones > 0.80, 0.10, 0.0) - np.where(stones < 0.12, 0.08, 0.0)
        img[r0:r1] = L
    return RoadRaster(np.clip(img, 0, 1), res_m, lat0, along0_m)


def _poly_mask(r: RoadRaster, pts_m: np.ndarray, win) -> np.ndarray:
    h, w = win[1] - win[0], win[3] - win[2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(r.local_px(pts_m, win) * 16).astype(np.int32)], 255,
                 lineType=cv2.LINE_AA, shift=4)
    return mask.astype(np.float32) / 255.0


def _buffer_polyline(pts: np.ndarray, half: float) -> list:
    from shapely.geometry import LineString
    return [tuple(p) for p in LineString(pts).buffer(half, cap_style="flat").exterior.coords]


def draw_crack(r: RoadRaster, start_m, heading_deg: float, length_m: float, width_mm: float,
               rng: np.random.Generator, darkness: float = 0.45) -> Feature:
    """A wandering dark line. heading_deg: 0 = along the route, 90 = across it."""
    pts = [np.array(start_m, float)]
    h = math.radians(heading_deg)
    h0 = h
    step = 0.02
    for _ in range(max(2, int(length_m / step))):
        h += rng.normal(0, 0.12) + 0.08 * (h0 - h)   # wander around the main direction
        pts.append(pts[-1] + step * np.array([math.sin(h), math.cos(h)]))
    pts = np.array(pts)
    win = r.window(pts, 0.05)
    sub = r.img[win[0]:win[1], win[2]:win[3]]
    mask = np.zeros(sub.shape, np.uint8)
    thick = max(1, int(round(width_mm / 1000 / r.res_m)))
    cv2.polylines(mask, [np.round(r.local_px(pts, win) * 16).astype(np.int32)], False, 255,
                  thickness=thick, lineType=cv2.LINE_AA, shift=4)
    a = mask.astype(np.float32) / 255.0
    sub *= 1 - a * (1 - darkness)
    f = Feature("crack", _buffer_polyline(pts, max(width_mm / 2000, 0.01)), True,
                {"width_mm": width_mm, "length_m": length_m})
    r.features.append(f)
    return f


def draw_alligator(r: RoadRaster, centre_m, size_m: tuple[float, float], cell_m: float, width_mm: float,
                   rng: np.random.Generator) -> Feature:
    cx, cy = centre_m
    sw, sl = size_m
    n = max(4, int(sw * sl / (cell_m**2)))
    seeds = np.column_stack([rng.uniform(cx - sw / 2, cx + sw / 2, n), rng.uniform(cy - sl / 2, cy + sl / 2, n)])
    box = np.array([[cx - sw / 2, cy - sl / 2], [cx + sw / 2, cy + sl / 2]])
    win = r.window(box, 0.0)
    xs = r.lateral0_m + np.arange(win[2], win[3]) * r.res_m
    ys = r.along0_m + np.arange(win[0], win[1]) * r.res_m
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], -1)
    d = np.sqrt(((P[:, None, :] - seeds[None, :, :]) ** 2).sum(-1))
    d.sort(axis=1)
    edge = ((d[:, 1] - d[:, 0]) / 2).reshape(X.shape)
    ell = ((X - cx) / (sw / 2)) ** 2 + ((Y - cy) / (sl / 2)) ** 2
    a = np.clip((width_mm / 2000 - edge) / r.res_m + 0.5, 0, 1) * (ell < 1)
    r.img[win[0]:win[1], win[2]:win[3]] *= (1 - a * 0.55).astype(np.float32)
    poly = [(cx + sw / 2 * math.cos(t), cy + sl / 2 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 33)]
    f = Feature("alligator", poly, True, {"cell_m": cell_m, "width_mm": width_mm})
    r.features.append(f)
    return f


def draw_pothole(r: RoadRaster, centre_m, radius_m: float, rng: np.random.Generator) -> Feature:
    t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    rr = radius_m * (1 + 0.25 * np.sin(3 * t + rng.uniform(0, 6)) + 0.1 * np.sin(7 * t + rng.uniform(0, 6)))
    pts = np.column_stack([centre_m[0] + rr * np.cos(t), centre_m[1] + rr * np.sin(t)])
    win = r.window(pts, 0.1)
    sub = r.img[win[0]:win[1], win[2]:win[3]]
    m = _poly_mask(r, pts, win)
    soft = cv2.GaussianBlur(m, (0, 0), 1.5)
    rim = np.clip(cv2.GaussianBlur(m, (0, 0), 6) - soft, 0, 1)
    noise = cv2.resize(rng.random((max(2, sub.shape[0] // 6), max(2, sub.shape[1] // 6))).astype(np.float32),
                       (sub.shape[1], sub.shape[0]))
    r.img[win[0]:win[1], win[2]:win[3]] = sub * (1 - soft) + soft * (0.20 + 0.12 * noise) + 0.08 * rim
    f = Feature("pothole", [tuple(p) for p in pts], True, {"radius_m": radius_m})
    r.features.append(f)
    return f


def draw_patch(r: RoadRaster, centre_m, size_m: tuple[float, float], angle_deg: float, tone: float = 0.7) -> Feature:
    cx, cy = centre_m
    a = math.radians(angle_deg)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(size_m) / 2
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = corners @ rot.T + np.array([cx, cy])
    win = r.window(pts, 0.05)
    sub = r.img[win[0]:win[1], win[2]:win[3]]
    m = _poly_mask(r, pts, win)
    smooth = cv2.GaussianBlur(sub, (0, 0), 2.0)
    r.img[win[0]:win[1], win[2]:win[3]] = sub * (1 - m) + m * np.clip(smooth * tone + 0.02, 0, 1)
    f = Feature("patch", [tuple(p) for p in pts], True, {"tone": tone})
    r.features.append(f)
    return f


def draw_marking(r: RoadRaster, lateral_m: float, width_m: float, dash_m: float | None = None,
                 gap_m: float = 0.0) -> Feature:
    h, _ = r.img.shape
    c0, _ = r.px(lateral_m - width_m / 2, 0)
    c1, _ = r.px(lateral_m + width_m / 2, 0)
    c0, c1 = int(round(float(c0))), int(round(float(c1)))
    along = r.along0_m + np.arange(h) * r.res_m
    on = np.ones(h, bool)
    if dash_m:
        on &= ((along - r.along0_m) % (dash_m + gap_m)) < dash_m
    r.img[on, max(c0, 0):max(c1, 0)] = 0.85
    a0, a1 = r.along0_m, r.along0_m + h * r.res_m
    poly = [(lateral_m - width_m / 2, a0), (lateral_m + width_m / 2, a0), (lateral_m + width_m / 2, a1),
            (lateral_m - width_m / 2, a1)]
    f = Feature("marking", poly, False, {"dash_m": dash_m, "gap_m": gap_m})
    r.features.append(f)
    return f


def draw_shadow(r: RoadRaster, centre_m, size_m: tuple[float, float], rng: np.random.Generator,
                depth: float = 0.5) -> Feature:
    cx, cy = centre_m
    t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    rr = 1 + 0.3 * np.sin(4 * t + rng.uniform(0, 6))
    pts = np.column_stack([cx + size_m[0] / 2 * rr * np.cos(t), cy + size_m[1] / 2 * rr * np.sin(t)])
    win = r.window(pts, 0.3)
    m = _poly_mask(r, pts, win)
    m = cv2.GaussianBlur(m, (0, 0), 0.08 / r.res_m)
    r.img[win[0]:win[1], win[2]:win[3]] *= 1 - m * (1 - depth)
    f = Feature("shadow", [tuple(p) for p in pts], False, {})
    r.features.append(f)
    return f


def to_rgb(gray: np.ndarray, tint=(1.0, 0.98, 0.95)) -> np.ndarray:
    return np.clip(gray[..., None] * np.array(tint, np.float32) * 255.0, 0, 255).astype(np.uint8)
