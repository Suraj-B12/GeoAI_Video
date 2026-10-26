"""Synthetic GNSS/IMU streams with known truth, shared by the trajectory tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymap3d

ORIGIN = (12.9715987, 77.5945627, 900.0)  # Bengaluru
T0_NS = 5_000_000_000_000                  # arbitrary session-clock origin


def fixes_frame(tau, e, n, speed, bearing, *, latency_s=0.0, h_acc=3.0, utc0_ms=1_790_000_000_000,
                utc_jitter_ms=0.0, rng=None) -> pd.DataFrame:
    """GNSS fixes measured at true times `tau` (s), stamped `latency_s` late."""
    rng = rng or np.random.default_rng(0)
    lat, lon, h = pymap3d.enu2geodetic(e, n, np.zeros_like(e), *ORIGIN)
    t_ns = T0_NS + np.round((np.asarray(tau) + latency_s) * 1e9).astype(np.int64)
    utc = utc0_ms + np.asarray(tau) * 1000.0 + rng.normal(0, utc_jitter_ms, len(tau)) if utc_jitter_ms else \
        utc0_ms + np.asarray(tau) * 1000.0
    return pd.DataFrame({
        "t_ns": pd.array(t_ns, dtype="Int64"),
        "lat_deg": lat, "lon_deg": lon, "alt_ellipsoid_m": h,
        "h_acc_m": np.full(len(tau), h_acc),
        "speed_mps": speed, "bearing_deg": bearing,
        "is_mock": np.zeros(len(tau), int),
        "utc_ms": pd.array(np.round(utc).astype(np.int64), dtype="Int64"),
    })


def imu_frame(t, acc_xyz, grav_xyz, gyro_xyz) -> pd.DataFrame:
    rows = []
    for name, v in (("acc", acc_xyz), ("gravity", grav_xyz), ("gyro", gyro_xyz)):
        rows.append(pd.DataFrame({
            "t_ns": pd.array(T0_NS + np.round(np.asarray(t) * 1e9).astype(np.int64), dtype="Int64"),
            "sensor": name, "x": v[:, 0], "y": v[:, 1], "z": v[:, 2], "w": np.nan, "accuracy": 3,
        }))
    return pd.concat(rows, ignore_index=True).sort_values("t_ns", kind="stable").reset_index(drop=True)


def truth_enu(traj, lat, lon):
    """Truth points expressed in the trajectory's own ENU frame."""
    e, n, _ = pymap3d.geodetic2enu(lat, lon, np.zeros_like(lat), *traj.origin)
    return e, n
