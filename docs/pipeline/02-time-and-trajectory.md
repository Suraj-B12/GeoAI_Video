# 02 · Time and Trajectory

Goal: for every video frame (and every image row), know **when** it was exposed, in UTC, and **where the phone was and which way it was heading**, with an honest uncertainty.

Inputs: `frames.csv`, `gnss_fixes.csv`, `imu.csv`, `clock_sync.csv`, `session.json` ([spec](../session-package-spec.md)).
Outputs: `work/<id>/time_model.json`, `gnss_clean.parquet`, `frame_times.parquet`, `trajectory.parquet`.

## 1. One clock

Everything stays on the session clock (`t_ns`, CLOCK_BOOTTIME nanoseconds) until stage 08.

- Frames: `frames.t_ns` is the start of exposure of the first sensor row, already on the session clock (the app applied `sensor_to_session_offset_ns`).
- GNSS: `gnss_fixes.t_ns`. IMU: `imu.t_ns`. Both on the session clock by Android contract.
- If `camera.timestamp_source != REALTIME`, the frame clock is only approximately aligned (± tens of ms). Mark the session `sync = approximate` and widen position uncertainty by `0.05 s × speed`.

**Never compute a frame time from its index or from container PTS** ([03](03-video-decoding.md)).

## 2. Clean the GNSS fixes

1. Parse; drop rows without `lat_deg`, `lon_deg` or `t_ns`; sort by `t_ns`; drop exact duplicates.
2. Any `is_mock = 1`: mark the session untrusted ([01](01-ingest.md)).
3. Drop fixes with `h_acc_m > 25` (config `gnss.max_h_acc_m`).
4. Convert to a local east-north-up (ENU) frame with `pymap3d.geodetic2enu`. Origin = median of the first 10 accepted fixes (ellipsoidal height `alt_ellipsoid_m`; use 0 if missing).
5. Velocity per fix:
   - if `speed_mps ≥ 1.0` and a bearing is present: `vE = speed·sin(bearing)`, `vN = speed·cos(bearing)` (Doppler-based, preferred);
   - otherwise: centred difference of neighbouring positions, smoothed over 3 fixes.
6. Outlier test (repeat once): predict each fix from a Hermite spline through its two neighbours; drop it if the residual exceeds `max(3·h_acc_m, 5 m)` or the implied acceleration exceeds 6 m/s².

Save to `gnss_clean.parquet` with a `reason` column for every dropped fix.

## 3. GNSS latency

The platform may stamp a fix tens to hundreds of milliseconds after the satellites measured it. At 40 km/h, 100 ms is 1.1 m of along-track error, so this must be measured and removed.

**Definition:** `L > 0` means GNSS `t_ns` is **later** than the true measurement time. The corrected time is `t_true = t_ns − L`.

**Per-session estimate (automatic):**
1. GNSS acceleration `a_g(t)`: fit a smoothing spline to `speed_mps` (Doppler) and differentiate. Resample at 10 Hz.
2. IMU linear acceleration: `lin = acc − gravity` (interpolate `gravity` samples to `acc` times), low-pass at 2 Hz (Butterworth, zero-phase), resample at 10 Hz. Remove the vertical part: `lin_h = lin − (lin·ĝ)ĝ`.
3. Forward axis `f` in device coordinates: least squares `a_g ≈ lin_h · x` over samples with `|a_g| > 0.3 m/s²`, then `f = x/|x|`. Sanity: `|x|` within 0.7–1.3.
4. For `L` from −0.5 s to +0.5 s in 10 ms steps, compute the Pearson correlation of `a_g(t)` with `(lin_h·f)(t − L)`. Take the peak and refine it by parabolic interpolation.
5. Accept if the peak correlation is ≥ 0.6, at least 60 s of driving has `std(a_g) > 0.3 m/s²`, and the peak lies inside the search window. Otherwise use the device default.

**Device default:** the value measured by the drive-past marker test ([11](11-testing-and-acceptance.md) §4), stored in `config/devices/<device_id>.yaml`. If the session estimate and the device default differ by more than 100 ms, use the device default and warn.

Record `latency_s`, `latency_method` (`session | device_default | none`), and the correlation in `time_model.json`.

## 4. UTC

Fit `utc_ms = a + b · t_true_ns` over all accepted fixes with a robust line fit (Theil–Sen or Huber).

- Expect `b ≈ 1e-6` ms per ns (clock drift well under 100 ppm). Warn outside that range.
- Report the residual RMS. Some chipsets round `getTime()` to the measurement epoch; the fit absorbs that.
- Frame UTC: `utc_ms(t) = a + b·t`. IST = UTC + 05:30 (India has no daylight saving).
- Never use `wall_ms` for anything but display.

## 5. Trajectory

**MVP:** piecewise cubic Hermite interpolation (`scipy.interpolate.CubicHermiteSpline`), separately for E, N and U, through accepted fixes at `t_true`, with the velocities from §2 as derivatives.

| Output | How |
|---|---|
| Position E, N, U and lat, lon, alt | Spline value; inverse `pymap3d.enu2geodetic` |
| Speed | `|dE/dt, dN/dt|` from the spline derivative |
| Heading ψ (deg, clockwise from north) | `atan2(vE, vN)` when speed ≥ 1.5 m/s. Below that, GNSS heading is noise: propagate the last reliable heading with the gyro yaw rate (formula under **Reversing** below) and flag `heading_from_gyro` |
| Path distance `s` (m) | Cumulative integral of speed (trapezoid at 100 Hz). Used by [06](06-association.md); immune to GNSS position jumps |
| Horizontal 1σ `h_sigma_m` | `sqrt(h_acc_interp² + (σ_L·speed)² + interp_term²)`, where `h_acc_interp` is linear between fixes (already a 68 % value), `σ_L` = 0.03 s for a session estimate or 0.05 s for a device default, and `interp_term = |a|·(gap/2)²/2` capped at 2 m |

**Gaps:** where consecutive accepted fixes are more than 3 s apart (`gnss.max_gap_s`), the trajectory is undefined. Frames inside get `pos_quality = gap` and are excluded from geolocated outputs; they are reported in QC. **Do not dead-reckon in the MVP.**

**Reversing:** the camera turns with the vehicle body, but while the vehicle reverses the velocity heading points backwards. A reversing frame would be projected 180° the wrong way and its distresses placed behind the vehicle instead of in front of it.

How to detect it, at every low-speed interval (speed < 1.5 m/s):
1. Compare the change in velocity heading `Δψ_vel` (last reliable heading before the interval → first one after) with the gyro yaw change `Δψ_gyro` over the same interval.
2. Compute the gyro yaw rate as `dψ/dt = −ω·û`, where `ω` is the `gyro` sample (rad/s, counter-clockwise positive), `û = gravity/|gravity|` points **up** in device coordinates (Android's gravity sensor reads +9.81 on z when the phone lies face-up), and ψ is clockwise from north.
3. If `|wrap(Δψ_vel − Δψ_gyro)| > 120°`, the direction of travel flipped.

Sessions are assumed to start moving forward. Frames between a flip and the next flip get `pos_quality = reversing` and are excluded like gap frames.

A reverse that stays below 1.5 m/s the whole time is not caught, because both ends of its interval face forward. Its frames are low-speed frames, thinned as in §8, and [06](06-association.md) §6 flags records on either side of a low-speed interval as possible duplicates.

**v2:** an error-state Kalman filter with a Rauch–Tung–Striebel smoother over GNSS + IMU (gyro yaw rate, forward acceleration, non-holonomic constraint). This bridges short gaps and smooths heading. Optional PPK input from `gnss_raw.txt` against Survey of India CORS data.

## 6. Per-frame and per-row time

```
t_row(r) = frames.t_ns + exposure_ns / 2 + rs_skew_ns · r / H     (r: row in sensor orientation, H: image height)
```

- `frame_times.parquet` stores `t_mid_ns = t_row(H/2)` for each frame.
- Stage 06 evaluates the trajectory at `t_row(r_detection)`, the detection's own centre row. At 40 km/h with a 20 ms readout, top and bottom rows are 22 cm apart in vehicle position.

## 7. Camera attitude

Use the **static mount calibration** ([04](04-calibration-and-geometry.md)) for geometry in the MVP. Do **not** correct the camera pose from the raw gravity vector. Gravity tilts relative to the road on grades and banked curves: a 5 % grade is 2.9°, which would move projected points by about 15 cm. That correction would add error, not remove it.

Gravity is still useful for QC. Compute the high-pass (> 0.3 Hz) part of the tilt between `gravity` and the calibration's reference gravity. It captures suspension pitch/roll and bumps, not road grade. Store `tilt_hp_deg` per frame. Frames with `tilt_hp_deg > 2°` are flagged `bump` and scored lower in [07](07-best-frame-and-crop.md). Using the high-pass component in geometry is a v2 option, and only after the field test shows it helps.

## 8. Outputs

`frame_times.parquet`: `segment, frame_in_segment, frame_global, t_ns, t_mid_ns, exposure_ns, rs_skew_ns, utc_ms, black, stationary, valid`.

`trajectory.parquet` (per frame, at `t_mid_ns`): `frame_global, e_m, n_m, u_m, lat_deg, lon_deg, alt_m, speed_mps, heading_deg, heading_from_gyro, s_m, h_sigma_m, pos_quality (ok | gap | low_accuracy | stationary | reversing), tilt_hp_deg, bump`.

The trajectory object must also be **callable** at arbitrary `t` (for per-detection rows); persist the spline knots in `time_model.json`.

`stationary` = speed < 0.5 m/s for longer than 2 s. Stationary frames are thinned to one per 2 s before detection. Other low-speed frames (< 1.5 m/s) are thinned to one per second: at 1.5 m/s that is 1.5 m of travel, less than the ROI's along-track extent, so no road is skipped.

## 9. Tests

| Test | Pass condition |
|---|---|
| Synthetic drive with braking events, IMU and GNSS generated with a known latency of 150 ms | Estimated L within ±20 ms |
| Constant-speed circle (r = 50 m, 40 km/h), 1 Hz fixes with Doppler velocity | Interpolated position error < 0.10 m; heading error < 1° |
| Fix gap of 5 s | Frames in the gap get `pos_quality = gap`; none are geolocated |
| UTC fit on synthetic fixes with ±1 ms `getTime()` jitter and 5 % outliers | Frame UTC error < 2 ms against ground truth; outliers rejected |
| Mock fix in the input | Session marked untrusted |
| Synthetic three-point turn (forward, stop, reverse with steering, stop, forward) | Only the reversing leg is flagged `reversing`; none of its frames are geolocated |
