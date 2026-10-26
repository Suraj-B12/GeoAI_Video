import numpy as np
import pymap3d
import pytest

from geoai.config import GnssConfig
from geoai.trajectory import build_trajectory, clean_fixes, estimate_latency, fit_utc, wrap180

from synth_gnss import ORIGIN, T0_NS, fixes_frame, imu_frame, truth_enu


def _circle(duration=60.0, R=50.0, v=40 / 3.6):
    w = v / R
    tau = np.arange(0.0, duration + 1e-9, 1.0)
    e = R * np.sin(w * tau)
    n = R - R * np.cos(w * tau)
    ve, vn = v * np.cos(w * tau), v * np.sin(w * tau)
    bearing = np.degrees(np.arctan2(ve, vn)) % 360
    return tau, e, n, np.full_like(tau, v), bearing, (R, w, v)


def test_circle_interpolation_accuracy():
    tau, e, n, speed, bearing, (R, w, v) = _circle()
    df = fixes_frame(tau, e, n, speed, bearing)
    cfg = GnssConfig()
    fx = clean_fixes(df, T0_NS, cfg)
    traj = build_trajectory(fx, None, T0_NS, cfg)
    tq = np.linspace(2.0, 58.0, 500)
    s = traj.sample(tq)
    lat, lon, _ = pymap3d.enu2geodetic(R * np.sin(w * tq), R - R * np.cos(w * tq), 0 * tq, *ORIGIN)
    te, tn = truth_enu(traj, lat, lon)
    err = np.hypot(s["e"] - te, s["n"] - tn)
    assert err.max() < 0.10
    true_heading = np.degrees(np.arctan2(np.cos(w * tq), np.sin(w * tq))) % 360
    assert np.abs(wrap180(s["heading"] - true_heading)).max() < 1.0
    assert s["valid"].all()
    # Path distance follows the arc length.
    assert abs((s["s"][-1] - s["s"][0]) - v * (tq[-1] - tq[0])) < 0.2


def test_gap_frames_are_invalid():
    tau, e, n, speed, bearing, _ = _circle()
    keep = (tau < 20) | (tau > 25)          # 6 s without fixes
    df = fixes_frame(tau[keep], e[keep], n[keep], speed[keep], bearing[keep])
    cfg = GnssConfig()
    traj = build_trajectory(clean_fixes(df, T0_NS, cfg), None, T0_NS, cfg)
    s = traj.sample(np.array([10.0, 22.5, 30.0]))
    assert list(s["valid"]) == [True, False, True]
    assert len(traj.gaps) == 1
    assert np.all(np.diff(traj.s) >= 0)


def test_mock_and_inaccurate_fixes_dropped():
    tau, e, n, speed, bearing, _ = _circle(20)
    df = fixes_frame(tau, e, n, speed, bearing)
    df.loc[3, "is_mock"] = 1
    df.loc[5, "h_acc_m"] = 80.0
    fx = clean_fixes(df, T0_NS, GnssConfig())
    assert len(fx) == len(df) - 2
    assert set(fx.dropped["reason"]) == {"mock", "h_acc"}


def test_position_jump_rejected_as_outlier():
    tau, e, n, speed, bearing, _ = _circle(30)
    e = e.copy()
    e[12] += 40.0  # a 40 m multipath jump
    fx = clean_fixes(fixes_frame(tau, e, n, speed, bearing), T0_NS, GnssConfig())
    assert "outlier" in set(fx.dropped["reason"])
    assert len(fx) == len(tau) - 1


def test_utc_fit_with_jitter_and_outliers():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 3600, 3600))
    utc = 1_790_000_000_000 + 1000.0 * (1 + 20e-6) * t + rng.normal(0, 1.0, t.size)
    bad = rng.random(t.size) < 0.05
    utc[bad] += rng.choice([-1, 1], bad.sum()) * rng.uniform(100, 1000, bad.sum())
    fit = fit_utc(t, utc)
    truth = 1_790_000_000_000 + 1000.0 * (1 + 20e-6) * t
    assert np.abs(fit.utc_ms(t) - truth).max() < 2.0


def _straight_drive(duration=240.0, latency=0.15, seed=1):
    """Straight drive east with a varying speed profile; IMU on the true clock."""
    rng = np.random.default_rng(seed)
    fs = 200.0
    t = np.arange(0.0, duration, 1 / fs)
    v = 10 + 3 * np.sin(2 * np.pi * t / 20) + 1.5 * np.sin(2 * np.pi * t / 7 + 1.0)
    a = np.gradient(v, t)
    pos = np.cumsum(v) / fs
    up = np.array([0.0, 0.7071, 0.7071])
    fwd = np.array([0.0, 0.7071, -0.7071])
    grav = np.tile(up * 9.81, (t.size, 1))
    acc = grav + a[:, None] * fwd + rng.normal(0, 0.05, (t.size, 3))
    gyro = rng.normal(0, 0.002, (t.size, 3))
    tau = np.arange(1.0, duration - 1, 1.0)
    e = np.interp(tau, t, pos)
    speed = np.interp(tau, t, v)
    df = fixes_frame(tau, e, 0 * e, speed, np.full_like(tau, 90.0), latency_s=latency)
    imu = imu_frame(t, acc, grav, gyro)
    return df, imu


def test_latency_estimated_from_imu():
    df, imu = _straight_drive(latency=0.15)
    cfg = GnssConfig()
    fx = clean_fixes(df, T0_NS, cfg)
    est = estimate_latency(fx, imu, T0_NS, cfg)
    assert est.method == "session", est.note
    assert abs(est.latency_s - 0.15) < 0.02
    assert est.corr > 0.9


def test_latency_falls_back_to_default_on_short_session():
    df, imu = _straight_drive(duration=40.0)
    cfg = GnssConfig(default_latency_s=0.08, default_latency_measured=True)
    est = estimate_latency(clean_fixes(df, T0_NS, cfg), imu, T0_NS, cfg)
    assert est.method == "device_default" and est.latency_s == 0.08


@pytest.mark.parametrize("sensor", ["acc", "gravity", "gyro"])
def test_missing_sensor_values_fall_back_without_crashing(sensor):
    df, imu = _straight_drive(duration=40.0)
    imu.loc[imu["sensor"] == sensor, ["x", "y", "z"]] = np.nan
    cfg = GnssConfig(default_latency_s=0.08, default_latency_measured=True)
    trajectory = build_trajectory(clean_fixes(df, T0_NS, cfg), imu, T0_NS, cfg)
    assert trajectory.latency.method == "device_default"
    assert trajectory.latency.latency_s == 0.08
    assert np.isfinite(trajectory.heading).all()


def _three_point_turn():
    """Forward north 20 s, stop, reverse south 8 s (body still facing north), stop, forward again."""
    fs = 200.0
    t = np.arange(0.0, 60.0, 1 / fs)
    v = np.zeros_like(t)                         # signed speed along north (body faces north throughout)
    v[t < 20] = 5.0
    ramp = (t >= 20) & (t < 23)
    v[ramp] = 5.0 * (23 - t[ramp]) / 3
    rev = (t >= 25) & (t < 35)
    v[rev] = -2.5
    fwd2 = t >= 38
    v[fwd2] = 5.0
    pos = np.cumsum(v) / fs
    tau = np.arange(0.0, 59.0, 1.0)
    n = np.interp(tau, t, pos)
    sv = np.interp(tau, t, v)
    bearing = np.where(sv >= 0, 0.0, 180.0)
    df = fixes_frame(tau, 0 * n, n, np.abs(sv), bearing)
    up = np.array([0.0, 0.0, 9.81])
    grav = np.tile(up, (t.size, 1))
    imu = imu_frame(t, grav.copy(), grav, np.zeros((t.size, 3)))
    return df, imu


def test_reversing_leg_is_flagged():
    df, imu = _three_point_turn()
    cfg = GnssConfig()
    traj = build_trajectory(clean_fixes(df, T0_NS, cfg), imu, T0_NS, cfg)
    s = traj.sample(np.array([10.0, 30.0, 50.0]))
    assert traj.reversing_detection
    assert list(s["reversing"]) == [False, True, False]
    # The vehicle body faced north throughout, also while reversing.
    assert abs(wrap180(s["heading"][1])) < 5.0
