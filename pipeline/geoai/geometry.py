"""Camera model: pixel <-> road plane (docs/pipeline/04 §3).

Vehicle frame V: X right, Y forward, Z up; the origin is on the ground directly below the camera,
so ground points (x, y) are metres right of / ahead of the camera nadir.

Two sources of the model:
- `calibrated`: lens + mount calibration sessions (geoai.calibration).
- `nominal`: computed automatically from the session itself. Intrinsics come from the camera
  characteristics in session.json, pitch and roll from the gravity sensor, height from the
  survey form. No manual step is needed; positions are a little less accurate (yaw is assumed 0
  or 180 degrees, and the height is the operator's nominal value).

All pixel coordinates are in display orientation (an UPSIDE_DOWN mount is rotated 180 degrees on
decoding; see `rotated_180`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import GeometryConfig

# Camera axes of a BACK camera in device coordinates, by sensor orientation (rows: x_cam, y_cam,
# z_cam). Derived from the app's orientation rule: "sensor up" is device +y, +x, -y, -x for
# 0, 90, 180, 270 degrees; image right is (up_y, -up_x); the back camera looks along device -z.
_SENSOR_UP = {0: (0.0, 1.0), 90: (1.0, 0.0), 180: (0.0, -1.0), 270: (-1.0, 0.0)}


def cam_from_device(sensor_orientation_deg: int) -> np.ndarray:
    ux, uy = _SENSOR_UP[int(sensor_orientation_deg) % 360]
    return np.array([[uy, -ux, 0.0],
                     [-ux, -uy, 0.0],
                     [0.0, 0.0, -1.0]])


def vehicle_from_camera(up_cam: np.ndarray, facing: str = "forward") -> np.ndarray:
    """Rotation V <- camera from the world-up direction expressed in camera coordinates.

    Yaw is not observable from gravity: the vehicle's forward axis is taken as the horizontal
    projection of the optical axis (forward camera) or its opposite (rearward camera).
    """
    z_v = up_cam / np.linalg.norm(up_cam)
    axis = np.array([0.0, 0.0, 1.0])
    f = axis - (axis @ z_v) * z_v
    if np.linalg.norm(f) < 1e-6:
        raise ValueError("camera points straight down/up: forward direction undefined")
    f /= np.linalg.norm(f)
    y_v = -f if facing == "rearward" else f
    x_v = np.cross(y_v, z_v)
    return np.vstack([x_v, y_v, z_v])


@dataclass
class CameraModel:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray                 # OpenCV order k1, k2, p1, p2, k3
    R_v_from_c: np.ndarray           # rotation camera -> vehicle frame
    C_v: np.ndarray                  # camera centre in V (x, y = 0 by construction for nominal)
    source: str = "nominal"
    notes: dict = field(default_factory=dict)

    # ---- derived ------------------------------------------------------------------------------
    @property
    def height_m(self) -> float:
        return float(self.C_v[2])

    @property
    def pitch_deg(self) -> float:
        """Optical axis below the horizontal (positive = looking down)."""
        z_cam_in_v = self.R_v_from_c[:, 2]
        return float(math.degrees(math.asin(-z_cam_in_v[2])))

    @property
    def roll_deg(self) -> float:
        up_c = self.R_v_from_c[2, :]
        return float(math.degrees(math.atan2(up_c[0], -up_c[1])))

    def scaled(self, sx: float, sy: float | None = None) -> "CameraModel":
        """Same camera for an image resized by (sx, sy), e.g. analysis downscaling."""
        sy = sx if sy is None else sy
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return CameraModel(int(round(self.width * sx)), int(round(self.height * sy)), K, self.dist.copy(),
                           self.R_v_from_c.copy(), self.C_v.copy(), self.source, dict(self.notes))

    def cropped(self, u0: int, v0: int, width: int, height: int) -> "CameraModel":
        """Same camera for the sub-image [v0:v0+height, u0:u0+width]."""
        K = self.K.copy()
        K[0, 2] -= u0
        K[1, 2] -= v0
        return CameraModel(width, height, K, self.dist.copy(), self.R_v_from_c.copy(), self.C_v.copy(),
                           self.source, dict(self.notes))

    def rotated_180(self) -> "CameraModel":
        """Model for the image rotated by 180 degrees (u -> W-1-u, v -> H-1-v)."""
        K = self.K.copy()
        K[0, 2] = self.width - 1 - K[0, 2]
        K[1, 2] = self.height - 1 - K[1, 2]
        d = self.dist.copy()
        d[2:4] *= -1  # tangential terms change sign under a 180 degree rotation
        R = self.R_v_from_c @ np.diag([-1.0, -1.0, 1.0])
        return CameraModel(self.width, self.height, K, d, R, self.C_v.copy(), self.source, dict(self.notes))

    # ---- projection ---------------------------------------------------------------------------
    def _undistort(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        if np.any(self.dist != 0):
            n = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        else:
            n = (pts.reshape(-1, 2) - self.K[:2, 2]) / np.array([self.K[0, 0], self.K[1, 1]])
        return n[:, 0].reshape(u.shape), n[:, 1].reshape(u.shape)

    def _distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]

    def pixel_to_ground(self, u, v, max_range_m: float | None = None):
        """Ground point (x right, y ahead, metres from the camera nadir) seen at pixel (u, v)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        xn, yn = self._undistort(u, v)
        d_c = np.stack([xn, yn, np.ones_like(xn)], -1)
        d_v = d_c @ self.R_v_from_c.T
        dz = d_v[..., 2]
        valid = dz < -1e-9
        lam = np.where(valid, -self.C_v[2] / np.where(valid, dz, -1.0), np.nan)
        x = lam * d_v[..., 0]  # relative to the nadir (C_v.x, C_v.y)
        y = lam * d_v[..., 1]
        if max_range_m is not None:
            valid &= np.hypot(x, y) <= max_range_m
        return np.where(valid, x, np.nan), np.where(valid, y, np.nan), valid

    def ground_to_pixel(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = np.stack([x, y, np.zeros_like(x)], -1) + np.array([self.C_v[0], self.C_v[1], 0.0]) - self.C_v
        d_c = p @ self.R_v_from_c  # (R_v_from_c^T p) row-wise
        valid = d_c[..., 2] > 1e-9
        zc = np.where(valid, d_c[..., 2], 1.0)
        u, v = self._distort(d_c[..., 0] / zc, d_c[..., 1] / zc)
        return np.where(valid, u, np.nan), np.where(valid, v, np.nan), valid

    def gsd_mm(self, u, v):
        """(along-track, across-track) ground size of one pixel in mm."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        _, y0, _ = self.pixel_to_ground(u, v - 0.5)
        _, y1, _ = self.pixel_to_ground(u, v + 0.5)
        x2, _, _ = self.pixel_to_ground(u - 0.5, v)
        x3, _, _ = self.pixel_to_ground(u + 0.5, v)
        return np.abs(y1 - y0) * 1000.0, np.abs(x3 - x2) * 1000.0

    def ground_map(self, step: int = 4, max_range_m: float | None = None):
        """Ground y (ahead) and x of a pixel grid with the given step; NaN above the horizon."""
        us = np.arange(0, self.width, step) + 0.5 * (step - 1)
        vs = np.arange(0, self.height, step) + 0.5 * (step - 1)
        uu, vv = np.meshgrid(us, vs)
        x, y, valid = self.pixel_to_ground(uu, vv, max_range_m)
        return us, vs, x, y, valid

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "K": self.K.tolist(), "dist": self.dist.tolist(),
                "R_v_from_c": self.R_v_from_c.tolist(), "C_v": self.C_v.tolist(), "source": self.source,
                "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg, "height_m": self.height_m,
                "notes": self.notes}


# ---- nominal model from session metadata ---------------------------------------------------------

def _num(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def intrinsics_from_camera_meta(cam: dict, width: int, height: int, hfov_deg: float):
    """K and distortion for the recording resolution from Camera2 characteristics.

    The recording stream is the largest centred crop of the active array with the stream's
    aspect ratio, scaled to the stream size (Camera2 default crop behaviour at zoom 1.0).
    """
    act = cam.get("active_array_size")
    pix = cam.get("pixel_array_size")
    cal = cam.get("lens_intrinsic_calibration")
    if act and len(act) == 4:
        aw, ah = float(act[2] - act[0]), float(act[3] - act[1])
    elif pix and len(pix) == 2:
        aw, ah = float(pix[0]), float(pix[1])
    else:
        aw = ah = None
    dist = np.zeros(5)
    if aw and ah:
        out_aspect = width / height
        if aw / ah > out_aspect:
            crop_w, crop_h = ah * out_aspect, ah
        else:
            crop_w, crop_h = aw, aw / out_aspect
        ox, oy = (aw - crop_w) / 2, (ah - crop_h) / 2
        sc = width / crop_w
        if cal and len(cal) >= 4 and _num(cal[0], 0) > 0:
            fx, fy, cx, cy = (float(c) for c in cal[:4])
            K = np.array([[fx * sc, 0, (cx - ox) * sc], [0, fy * sc, (cy - oy) * sc], [0, 0, 1.0]])
            ld = cam.get("lens_distortion")
            if ld and len(ld) == 5 and any(_num(x, 0) for x in ld):
                k1, k2, k3, p1, p2 = (float(x) for x in ld)  # Android order: radial 1-3, tangential 1-2
                dist = np.array([k1, k2, p1, p2, k3])
            return K, dist, "camera_characteristics"
        focal = (cam.get("focal_lengths_mm") or [None])[0]
        phys = cam.get("sensor_physical_size_mm")
        if _num(focal) and phys and _num(phys[0]) and pix:
            f_px = float(focal) / float(phys[0]) * float(pix[0])  # pixel-array pixels
            K = np.array([[f_px * sc, 0, (width - 1) / 2], [0, f_px * sc, (height - 1) / 2], [0, 0, 1.0]])
            return K, dist, "focal_length"
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
    return K, dist, "default_hfov"


def nominal_model(meta: dict, width: int, height: int, gravity_device: np.ndarray, cfg: GeometryConfig,
                  rotate180: bool = False) -> CameraModel:
    cam = meta.get("camera", {}) or {}
    mount = meta.get("mount", {}) or {}
    K, dist, k_src = intrinsics_from_camera_meta(cam, width, height, cfg.default_hfov_deg)
    so = cam.get("sensor_orientation_deg")
    if so is None:
        so = (meta.get("orientation") or {}).get("sensor_orientation_deg")
    so = 90 if so is None else int(so)
    up_cam = cam_from_device(so) @ (np.asarray(gravity_device, float) / np.linalg.norm(gravity_device))
    facing = str(mount.get("camera_facing") or "forward")
    R = vehicle_from_camera(up_cam, facing)
    h = _num(mount.get("nominal_height_m")) or cfg.default_height_m
    model = CameraModel(width, height, K, dist, R, np.array([0.0, 0.0, h]), "nominal",
                        {"intrinsics": k_src, "height_source": "survey_form" if _num(mount.get("nominal_height_m")) else "default",
                         "attitude": "gravity", "facing": facing, "sensor_orientation_deg": so})
    # In sensor orientation the model above is exact; an upside-down mount is decoded rotated.
    return model.rotated_180() if rotate180 else model
