# 01 · Ingest

Goal: get sessions off the phone **without loss or silent corruption**, verify them, and record what they contain before any processing.

## 1. Transfer (USB, offline)

The phone is a Galaxy S21 5G with USB debugging enabled and the PC's key authorised. The pull uses ADB (Android platform-tools), not MTP: MTP is unreliable for multi-GB files on Windows, and ADB lets the tool push a receipt back.

Reference implementation: [`tools/pull_sessions.py`](../../tools/pull_sessions.py). It is already tested against the app. `geoai pull` should call it or reimplement the same steps:

1. `adb shell ls -1 <root>` for each root (`/sdcard/GeoAI/sessions`, then the app-specific fallback).
2. Skip sessions without `manifest.json`. They are still recording or were interrupted; the operator finalises them in the app's **Sessions** screen first.
3. `adb pull <root>/<id> data/sessions/.incoming/` → verify (§2) → atomic rename to `data/sessions/<id>`.
4. Write `pc_receipt.json` and `adb push` it to the phone. The app then shows the session as **VERIFIED_ON_PC**.

**Never delete anything on the phone from the PC.** Deletion happens in the app. Operators delete only VERIFIED_ON_PC sessions; for any other session, the app shows a separate "Delete UNVERIFIED session?" warning.

## 2. Verification (blocking)

Implemented by [`tools/validate_session.py`](../../tools/validate_session.py) (spec §14). Blocking failures:

- a missing or modified file (size or SHA-256 mismatch with `manifest.json`);
- an invalid signature (when `cryptography` is installed: install it);
- unknown `spec_version` MAJOR;
- `boot_count_start != boot_count_end` (phone rebooted mid-session: the clock reset);
- a segment whose container PTS are not strictly increasing, or whose first sample is not a keyframe.

A session that fails stays in `.incoming/` with a `verification_failed.json` and is never processed.

## 3. QC report (non-blocking)

`geoai ingest` writes `work/<id>/qc_report.json` and a one-page HTML summary. It records each item as `ok | warn | bad` with the number behind it:

| Item | Source | Warn when |
|---|---|---|
| Session type, duration, distance, segments, resolutions | session.json, segments.csv | resolution changed (thermal step-down) |
| Exposure mode, streams, encoder config level | `capture_settings`, `encoders[]` | `AUTO_FALLBACK`, `encoder_only`, `config_level = 3` |
| Mount orientation | `orientation.verdict` | not `UPRIGHT` (`UPSIDE_DOWN` is fixable, see [03](03-video-decoding.md)) |
| Camera geometry constant | frames.csv `focal_length_mm`, `zoom_ratio`, `focus_diopters`, `eis_mode`, `ois_mode` | any change within a segment: **bad** for the affected frames, because a lens switch, zoom or focus change invalidates the calibration. Frames with `lens_state = 1` (lens moving) are flagged; more than 1 % of them is a warn |
| Mount stability | imu.csv `gravity` | a step in the gravity direction (1 Hz low-pass) of > 0.7° within 0.5 s that persists ≥ 10 s: possible mount shift after a bump. Grades change gravity gradually, a slip changes it at once |
| Frames: count, unmatched metadata, dropped (gaps > 1.5× median), black spans | frames.csv, events.csv | drops > 0.1 %, any unmatched |
| Samples vs rows per segment | video + frames.csv | rows > samples (crash): rows trimmed |
| Shutter distribution, low-light share | frames.csv, exposure_control.csv | median smear > blur budget, low-light > 5 % |
| GNSS: fix rate, max gap, h_acc distribution, mock, restarts, bracketing | gnss_fixes.csv, events.csv | rate < 0.9 Hz, gap > 2 s, median h_acc > 8 m, any mock (**bad**) |
| GNSS latency estimate | [02](02-time-and-trajectory.md) §3 | not estimable, or outside ±300 ms |
| Thermal max, battery temperature | thermal.csv | max status ≥ 2 |
| Camera interruptions, encoder errors | events.csv | any |
| Operator markers and km stones | events.csv | none for a survey (chainage anchoring impossible) |
| Calibration available for every resolution and the mount | calibration registry | missing (**blocks** `process`) |

## 4. Session registry (SQLite `sessions` table)

One row per session: `session_id, session_type, device_id, mount_id, start_utc, end_utc, duration_s, distance_m, road_id, direction, lane, start_chainage_km, spec_version, app_version, trusted (0/1), qc_status, intrinsics_ids, mount_calibration_id, ingested_at`. `trusted = 0` (mock location, invalid signature override) excludes a session from all exports unless explicitly included.

## 5. Calibration association

- **Intrinsics:** by `device.device_id` and the segment resolution (`width×height`). Require the calibration's `focus_diopters` within ±0.05 dpt of the survey's `capture_settings.focus_distance_diopters`; otherwise warn, because focus changes the focal length slightly. Every resolution used in the session (step-down adds 1080p) needs its own calibration.
- **Mount:** by `mount.mount_id`, the latest `calibration_mount` session before the survey (same day preferred). Check that the survey's gravity direction at start (`orientation.gravity_device_xyz`) is within 1.5° of the calibration's reference gravity. More than that means the mount moved: flag **bad**, because ground projection would be wrong.
- **End-of-day check:** a second `calibration_mount` at the end of the day brackets the day's surveys. If its pose differs from the morning's by more than 0.5° in pitch or roll, or 2 cm in height, the mount moved during the day. Flag every session in between, and use the QC "mount stability" events to find when it happened.

## 6. Tests

- A copy of a valid session with one byte flipped in `frames.csv` must be rejected.
- A crash-recovered session (more rows than samples) must ingest with a warning and trimmed rows.
- A session with `is_mock = 1` in any fix must be registered `trusted = 0`.
- Pulling the same phone twice must not duplicate or overwrite anything.
