"""Time model and vehicle trajectory (docs/pipeline/02-time-and-trajectory.md).

Internal time axis: seconds since `t0_ns` on the session clock (CLOCK_BOOTTIME), on which camera
frames and IMU samples are stamped. GNSS fixes are moved onto it by removing the latency L
(t_true = t_stamped - L, L > 0 means fixes are stamped late).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymap3d
from scipy import signal, stats
from scipy.interpolate import CubicHermiteSpline, UnivariateSpline

from .config import GnssConfig
from .util import log

GRID_HZ = 100.0


def wrap180(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


@dataclass
class CleanFixes:
    t: np.ndarray        # stamped time (s since t0), before latency correction
    e: np.ndarray
    n: np.ndarray
    u: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    speed: np.ndarray    # Doppler speed, NaN when not reported
    h_acc: np.ndarray
    utc_ms: np.ndarray   # NaN when not reported
    origin: tuple[float, float, float]
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.t)


def _hermite_point(ta, pa, va, tb, pb, vb, t):
    h = tb - ta
    tau = (t - ta) / h
    h00 = 2 * tau**3 - 3 * tau**2 + 1
    h10 = tau**3 - 2 * tau**2 + tau
    h01 = -2 * tau**3 + 3 * tau**2
    h11 = tau**3 - tau**2
    return h00 * pa + h10 * h * va + h01 * pb + h11 * h * vb


def _velocities(t, e, n, speed, bearing):
    ve = np.full_like(e, np.nan)
    vn = np.full_like(n, np.nan)
    dop = np.isfinite(speed) & np.isfinite(bearing) & (speed >= 1.0)
    b = np.radians(bearing)
    ve[dop] = speed[dop] * np.sin(b[dop])
    vn[dop] = speed[dop] * np.cos(b[dop])
    if (~dop).any() and len(t) >= 2:
        ge = np.gradient(e, t)
        gn = np.gradient(n, t)
        if len(t) >= 3:
            k = np.ones(3) / 3
            ge = np.convolve(np.pad(ge, 1, mode="edge"), k, mode="valid")
            gn = np.convolve(np.pad(gn, 1, mode="edge"), k, mode="valid")
        ve[~dop] = ge[~dop]
        vn[~dop] = gn[~dop]
    ve[~np.isfinite(ve)] = 0.0
    vn[~np.isfinite(vn)] = 0.0
    return ve, vn


def clean_fixes(gnss: pd.DataFrame, t0_ns: int, cfg: GnssConfig) -> CleanFixes:
    """docs/pipeline/02 §2: drop bad fixes, convert to ENU, attach Doppler velocities."""
    dropped = []
    if gnss.empty:
        raise ValueError("no GNSS fixes")
    df = gnss.copy()
    for c in ["lat_deg", "lon_deg", "alt_ellipsoid_m", "h_acc_m", "speed_mps", "bearing_deg", "is_mock"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    bad = df["t_ns"].isna() | df["lat_deg"].isna() | df["lon_deg"].isna()
    dropped.append(df[bad].assign(reason="missing"))
    df = df[~bad].sort_values("t_ns", kind="stable")
    dup = df["t_ns"].duplicated(keep="first")
    dropped.append(df[dup].assign(reason="duplicate"))
    df = df[~dup]
    mock = df["is_mock"].fillna(0) > 0
    dropped.append(df[mock].assign(reason="mock"))
    df = df[~mock]
    acc = df["h_acc_m"].fillna(np.inf) > cfg.max_h_acc_m
    dropped.append(df[acc].assign(reason="h_acc"))
    df = df[~acc]
    if len(df) < 2:
        raise ValueError("fewer than 2 usable GNSS fixes")

    head = df.head(10)
    lat0 = float(np.median(head["lat_deg"]))
    lon0 = float(np.median(head["lon_deg"]))
    h0 = float(np.nanmedian(head["alt_ellipsoid_m"])) if head["alt_ellipsoid_m"].notna().any() else 0.0
    h = df["alt_ellipsoid_m"].fillna(h0).to_numpy(float)
    e, n, u = pymap3d.geodetic2enu(df["lat_deg"].to_numpy(float), df["lon_deg"].to_numpy(float), h, lat0, lon0, h0)
    t = (df["t_ns"].to_numpy(dtype=np.int64) - np.int64(t0_ns)) / 1e9
    speed = df["speed_mps"].to_numpy(float)
    bearing = df["bearing_deg"].to_numpy(float)
    hacc = df["h_acc_m"].fillna(cfg.max_h_acc_m).to_numpy(float)
    utc = pd.to_numeric(df.get("utc_ms"), errors="coerce").to_numpy(float) if "utc_ms" in df else np.full(len(df), np.nan)
    keep = np.ones(len(df), bool)

    # Outlier test (docs/pipeline/02 §2.6). A jump also spoils its neighbours' predictions, so each
    # pass drops only fixes whose excess residual is a local maximum, then re-tests.
    for _ in range(10):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        ve, vn = _velocities(t[idx], e[idx], n[idx], speed[idx], bearing[idx])
        excess = np.zeros(len(idx))
        for k in range(1, len(idx) - 1):
            a, b, c = k - 1, k, k + 1
            pe = _hermite_point(t[idx[a]], e[idx[a]], ve[a], t[idx[c]], e[idx[c]], ve[c], t[idx[b]])
            pn = _hermite_point(t[idx[a]], n[idx[a]], vn[a], t[idx[c]], n[idx[c]], vn[c], t[idx[b]])
            res = math.hypot(pe - e[idx[b]], pn - n[idx[b]])
            limit = max(cfg.outlier_sigma * hacc[idx[b]], cfg.outlier_min_m)
            excess[b] = max(0.0, res / limit - 1.0)
            dt1 = t[idx[b]] - t[idx[a]]
            dt2 = t[idx[c]] - t[idx[b]]
            a1 = math.hypot(ve[b] - ve[a], vn[b] - vn[a]) / max(dt1, 1e-3)
            a2 = math.hypot(ve[c] - ve[b], vn[c] - vn[b]) / max(dt2, 1e-3)
            if a1 > cfg.max_accel_mps2 and a2 > cfg.max_accel_mps2:
                excess[b] = max(excess[b], min(a1, a2) / cfg.max_accel_mps2 - 1.0 + 1e-6)
        left = np.concatenate([[0.0], excess[:-1]])
        right = np.concatenate([excess[1:], [0.0]])
        bad_local = (excess > 0) & (excess >= left) & (excess >= right)
        if not bad_local.any():
            break
        dropped.append(df.iloc[idx[bad_local]].assign(reason="outlier"))
        keep[idx[bad_local]] = False

    sel = keep
    ve, vn = _velocities(t[sel], e[sel], n[sel], speed[sel], bearing[sel])
    dropped_df = pd.concat([d for d in dropped if len(d)], ignore_index=True) if any(len(d) for d in dropped) else pd.DataFrame()
    return CleanFixes(t=t[sel], e=e[sel], n=n[sel], u=u[sel], ve=ve, vn=vn, speed=speed[sel], h_acc=hacc[sel],
                      utc_ms=utc[sel], origin=(lat0, lon0, h0), dropped=dropped_df)


# ---- latency (docs/pipeline/02 §3) -----------------------------------------------------------

@dataclass
class LatencyEstimate:
    latency_s: float
    corr: float | None
    method: str          # session | device_default | none
    note: str = ""


def _imu_axes(imu: pd.DataFrame, sensor: str, t0_ns: int):
    d = imu[imu["sensor"] == sensor]
    if d.empty:
        return None, None
    d = d.dropna(subset=["t_ns", "x", "y", "z"]).sort_values("t_ns")
    if d.empty:
        return None, None
    t = (d["t_ns"].to_numpy(dtype=np.int64) - np.int64(t0_ns)) / 1e9
    v = d[["x", "y", "z"]].to_numpy(float)
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], v[keep]


def estimate_latency(fixes: CleanFixes, imu: pd.DataFrame, t0_ns: int, cfg: GnssConfig) -> LatencyEstimate:
    default = LatencyEstimate(cfg.default_latency_s, None,
                              "device_default" if cfg.default_latency_measured else "none")
    if imu is None or imu.empty or len(fixes) < 30 or not np.isfinite(fixes.speed).sum() >= 30:
        return LatencyEstimate(default.latency_s, None, default.method, "not enough data")
    ta, acc = _imu_axes(imu, "acc", t0_ns)
    tg_, grav = _imu_axes(imu, "gravity", t0_ns)
    if ta is None or tg_ is None or len(ta) < 100:
        return LatencyEstimate(default.latency_s, None, default.method, "no accelerometer/gravity")
    ok = np.isfinite(fixes.speed)
    tf, vf = fixes.t[ok], fixes.speed[ok]
    try:
        spl = UnivariateSpline(tf, vf, k=3, s=len(tf) * 0.05**2)
    except Exception as e:  # noqa: BLE001
        return LatencyEstimate(default.latency_s, None, default.method, f"speed spline failed: {e}")
    grid = np.arange(tf[0] + 1.0, tf[-1] - 1.0, 0.1)
    if len(grid) < 100:
        return LatencyEstimate(default.latency_s, None, default.method, "session too short")
    a_g = spl.derivative()(grid)

    gi = np.column_stack([np.interp(ta, tg_, grav[:, k]) for k in range(3)])
    lin = acc - gi
    fs = 1.0 / np.median(np.diff(ta))
    if fs > 5:
        b, a = signal.butter(4, min(2.0 / (fs / 2), 0.99))
        lin = signal.filtfilt(b, a, lin, axis=0)
    ghat = gi / np.linalg.norm(gi, axis=1, keepdims=True)
    lin_h = lin - (lin * ghat).sum(1, keepdims=True) * ghat

    def imu_at(times):
        return np.column_stack([np.interp(times, ta, lin_h[:, k]) for k in range(3)])

    strong = np.abs(a_g) > 0.3
    if strong.sum() < 20:
        return LatencyEstimate(default.latency_s, None, default.method, "no strong accelerations")
    X = imu_at(grid[strong])
    x, *_ = np.linalg.lstsq(X, a_g[strong], rcond=None)
    gain = float(np.linalg.norm(x))
    if not 0.7 <= gain <= 1.3:
        return LatencyEstimate(default.latency_s, None, default.method, f"forward-axis gain {gain:.2f} out of range")
    f = x / gain

    # Driving time with real acceleration content (10 s windows with std > 0.3 m/s²).
    win = 100
    active = sum(10.0 for i in range(0, len(a_g) - win, win) if np.std(a_g[i:i + win]) > 0.3)
    lags = np.arange(-cfg.latency_search_s, cfg.latency_search_s + 1e-9, 0.01)
    corrs = np.array([np.corrcoef(a_g, imu_at(grid - L) @ f)[0, 1] for L in lags])
    k = int(np.nanargmax(corrs))
    peak = float(corrs[k])
    L = float(lags[k])
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = corrs[k - 1], corrs[k], corrs[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            L += 0.01 * 0.5 * (y0 - y2) / den
    inside = 0 < k < len(lags) - 1
    if peak < cfg.latency_min_corr or active < cfg.latency_min_driving_s or not inside:
        return LatencyEstimate(default.latency_s, peak, default.method,
                               f"session estimate rejected (corr {peak:.2f}, active {active:.0f} s, edge {not inside})")
    if cfg.default_latency_measured and abs(L - cfg.default_latency_s) > cfg.latency_max_disagreement_s:
        log.warning("session latency %.3f s disagrees with the device default %.3f s; using the default",
                    L, cfg.default_latency_s)
        return LatencyEstimate(cfg.default_latency_s, peak, "device_default", f"session estimate {L:.3f} s disagreed")
    return LatencyEstimate(L, peak, "session")


# ---- UTC (docs/pipeline/02 §4) -----------------------------------------------------------------

@dataclass
class UtcFit:
    a_ms: float      # utc_ms = a_ms + b_ms_per_s * t   (t: true session seconds)
    b_ms_per_s: float
    resid_rms_ms: float
    n: int

    def utc_ms(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.a_ms + self.b_ms_per_s * np.asarray(t)


def fit_utc(t_true: np.ndarray, utc_ms: np.ndarray) -> UtcFit | None:
    ok = np.isfinite(utc_ms) & np.isfinite(t_true)
    if ok.sum() < 2:
        return None
    t, u = t_true[ok], utc_ms[ok]
    if len(t) > 1500:
        sel = np.linspace(0, len(t) - 1, 1500).round().astype(int)
        t, u = t[sel], u[sel]
    ref = float(np.median(u - 1000.0 * t))
    y = u - ref - 1000.0 * t  # residual around the nominal slope keeps numbers small
    slope, intercept, *_ = stats.theilslopes(y, t)
    b = 1000.0 + slope
    a = ref + intercept
    res = u - (a + b * t)
    med = np.median(res)
    rms = float(np.sqrt(np.mean(np.clip(res - med, -50, 50) ** 2)))
    if abs(slope) > 0.1:  # 100 ppm
        log.warning("UTC fit slope deviates by %.1f ppm from nominal", slope * 1000)
    return UtcFit(a_ms=float(a), b_ms_per_s=float(b), resid_rms_ms=rms, n=int(ok.sum()))


# ---- trajectory (docs/pipeline/02 §5) ---------------------------------------------------------

@dataclass
class Trajectory:
    origin: tuple[float, float, float]
    t: np.ndarray            # grid times (s, true clock)
    e: np.ndarray
    n: np.ndarray
    u: np.ndarray
    speed: np.ndarray
    heading: np.ndarray      # deg clockwise from north, direction the vehicle body faces
    heading_from_gyro: np.ndarray
    s: np.ndarray            # path distance (m), monotonic
    h_sigma: np.ndarray
    valid: np.ndarray        # False inside GNSS gaps
    reversing: np.ndarray
    latency: LatencyEstimate
    utc: UtcFit | None
    gaps: list[tuple[float, float]]
    reversing_detection: bool

    def sample(self, t: np.ndarray) -> dict[str, np.ndarray]:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        inside = (t >= self.t[0]) & (t <= self.t[-1])
        out = {k: np.interp(t, self.t, getattr(self, k)) for k in ("e", "n", "u", "speed", "s", "h_sigma")}
        if getattr(self, "_heading_unwrapped", None) is None:
            self._heading_unwrapped = np.degrees(np.unwrap(np.radians(self.heading)))
        out["heading"] = np.interp(t, self.t, self._heading_unwrapped) % 360.0
        idx = np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1)
        out["valid"] = inside & self.valid[idx]
        out["reversing"] = self.reversing[idx] & inside
        out["heading_from_gyro"] = self.heading_from_gyro[idx] & inside
        return out

    def enu_to_geodetic(self, e, n, u=None):
        lat0, lon0, h0 = self.origin
        if u is None:
            u = np.zeros_like(np.asarray(e, dtype=float))
        return pymap3d.enu2geodetic(e, n, u, lat0, lon0, h0)

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "t_range_s": [float(self.t[0]), float(self.t[-1])],
            "latency_s": self.latency.latency_s,
            "latency_method": self.latency.method,
            "latency_corr": self.latency.corr,
            "latency_note": self.latency.note,
            "utc": None if self.utc is None else vars(self.utc),
            "gaps_s": self.gaps,
            "distance_m": float(self.s[-1] - self.s[0]),
            "reversing_detection": self.reversing_detection,
            "reversing_s": float(self.reversing.sum() / GRID_HZ),
        }


def _yaw_rate(imu: pd.DataFrame | None, t0_ns: int):
    if imu is None or imu.empty:
        return None
    tg, w = _imu_axes(imu, "gyro", t0_ns)
    tr, g = _imu_axes(imu, "gravity", t0_ns)
    if tg is None or tr is None or len(tg) < 10 or len(tr) < 2:
        return None
    gi = np.column_stack([np.interp(tg, tr, g[:, k]) for k in range(3)])
    uhat = gi / np.linalg.norm(gi, axis=1, keepdims=True)
    # Gravity sensor points up; gyro is counter-clockwise positive; heading is clockwise from north.
    rate = -np.degrees((w * uhat).sum(1))
    return tg, rate


def build_trajectory(fixes: CleanFixes, imu: pd.DataFrame | None, t0_ns: int, cfg: GnssConfig,
                     latency: LatencyEstimate | None = None) -> Trajectory:
    if latency is None:
        latency = estimate_latency(fixes, imu, t0_ns, cfg) if imu is not None else LatencyEstimate(
            cfg.default_latency_s, None, "device_default" if cfg.default_latency_measured else "none")
    tt = fixes.t - latency.latency_s
    utc = fit_utc(tt, fixes.utc_ms)

    # Split into runs at gaps.
    breaks = np.flatnonzero(np.diff(tt) > cfg.max_gap_s)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(tt) - 1]])
    gaps = [(float(tt[b]), float(tt[b + 1])) for b in breaks]
    grid = np.arange(tt[0], tt[-1] + 0.5 / GRID_HZ, 1.0 / GRID_HZ)
    e = np.interp(grid, tt, fixes.e)
    n = np.interp(grid, tt, fixes.n)
    u = np.interp(grid, tt, fixes.u)
    ve = np.zeros_like(grid)
    vn = np.zeros_like(grid)
    ae = np.zeros_like(grid)
    an = np.zeros_like(grid)
    valid = np.zeros(len(grid), bool)
    for a, b in zip(starts, ends):
        if b - a < 1:
            continue
        sl = (grid >= tt[a]) & (grid <= tt[b])
        se = CubicHermiteSpline(tt[a:b + 1], fixes.e[a:b + 1], fixes.ve[a:b + 1])
        sn = CubicHermiteSpline(tt[a:b + 1], fixes.n[a:b + 1], fixes.vn[a:b + 1])
        e[sl], n[sl] = se(grid[sl]), sn(grid[sl])
        ve[sl], vn[sl] = se(grid[sl], 1), sn(grid[sl], 1)
        ae[sl], an[sl] = se(grid[sl], 2), sn(grid[sl], 2)
        valid[sl] = True
    speed = np.hypot(ve, vn)

    # Path distance: integrated speed; straight-line distance across gaps keeps it monotonic.
    ds = np.concatenate([[0.0], 0.5 * (speed[1:] + speed[:-1]) / GRID_HZ])
    for g0, g1 in gaps:
        i0 = int(np.searchsorted(grid, g0))
        i1 = int(np.searchsorted(grid, g1))
        span = max(i1 - i0, 1)
        jump = math.hypot(e[min(i1, len(e) - 1)] - e[i0], n[min(i1, len(n) - 1)] - n[i0])
        ds[i0 + 1:i1 + 1] = jump / span
    s = np.cumsum(ds)

    # Heading: velocity when moving; gyro-propagated at low speed; reversing detection.
    heading_vel = np.degrees(np.arctan2(ve, vn)) % 360.0
    slow = speed < cfg.min_speed_for_heading_mps
    yr = _yaw_rate(imu, t0_ns)
    if yr is not None:
        gyro_heading = np.concatenate([[0.0], np.cumsum(0.5 * (np.interp(grid[1:], *yr) + np.interp(grid[:-1], *yr)) / GRID_HZ)])
    else:
        gyro_heading = None
    heading = heading_vel.copy()
    from_gyro = np.zeros(len(grid), bool)
    reversing = np.zeros(len(grid), bool)
    state_rev = False
    flips: list[int] = []
    i = 0
    N = len(grid)
    while i < N:
        if not slow[i]:
            if state_rev:
                reversing[i] = True
                heading[i] = (heading_vel[i] + 180.0) % 360.0
            i += 1
            continue
        j = i
        while j < N and slow[j]:
            j += 1
        # Low-speed interval [i, j)
        before = i - 1
        after = j if j < N else None
        base = heading[before] if before >= 0 else heading_vel[i]
        for k in range(i, j):
            if gyro_heading is not None and before >= 0:
                heading[k] = (base + gyro_heading[k] - gyro_heading[before]) % 360.0
            else:
                heading[k] = base
            from_gyro[k] = True
        if after is not None and before >= 0 and gyro_heading is not None:
            d_vel = wrap180(heading_vel[after] - heading_vel[before])
            d_gyro = gyro_heading[after] - gyro_heading[before]
            if abs(wrap180(d_vel - d_gyro)) > cfg.reversing_flip_deg:
                state_rev = not state_rev
                flips.append(after)
        if state_rev:
            reversing[i:j] = True
        i = j
    # The low-speed interval that ends a reversing leg belongs to it as well.
    for k in range(1, len(flips), 2):
        a = flips[k - 1]
        b = flips[k]
        lo = a
        while lo > 0 and slow[lo - 1]:
            lo -= 1
        reversing[lo:b] = True

    # Horizontal 1-sigma (docs/pipeline/02 §5 table).
    h_acc = np.interp(grid, tt, fixes.h_acc)
    sigma_L = 0.03 if latency.method == "session" else 0.05
    spacing = np.interp(grid, tt[1:], np.diff(tt)) if len(tt) > 1 else np.ones_like(grid)
    interp_term = np.minimum(2.0, 0.5 * np.hypot(ae, an) * (spacing / 2) ** 2)
    h_sigma = np.sqrt(h_acc**2 + (sigma_L * speed) ** 2 + interp_term**2)

    return Trajectory(origin=fixes.origin, t=grid, e=e, n=n, u=u, speed=speed, heading=heading,
                      heading_from_gyro=from_gyro, s=s, h_sigma=h_sigma, valid=valid, reversing=reversing,
                      latency=latency, utc=utc, gaps=gaps, reversing_detection=gyro_heading is not None)
