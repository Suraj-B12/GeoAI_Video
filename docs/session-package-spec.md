# Session Package Specification v1.1

This is the contract between the **GeoAI Capture** Android app (writer, `android/`) and the extraction pipeline (reader, designed in [pipeline/](pipeline/README.md)). It describes exactly what app v0.1.0 writes; the column lists come from the source code (`SessionContext.kt`, `SegmentWriter.kt`, `GnssRecorder.kt`, `ImuRecorder.kt`, `ThermalAndClock.kt`).

Executable reference: [`tools/validate_session.py`](../tools/validate_session.py) implements the checks in §14.

**Versioning.** `spec_version` is `MAJOR.MINOR`. A reader must refuse an unknown MAJOR. A MINOR bump only adds fields or files; readers must ignore anything they don't know.

---

## 1. Conventions

| Rule | Detail |
|---|---|
| Session clock | Every field ending in `_ns` (except `mono_ns`, `pts_*`, `receipt_ns` where noted) is an int64 on Android `SystemClock.elapsedRealtimeNanos()` (CLOCK_BOOTTIME). Camera frames (REALTIME-source cameras), GNSS fixes and IMU samples all use it. It resets on reboot, so a session never spans a reboot (`boot_count_start` must equal `boot_count_end`). |
| UTC | `utc_ms` fields are Unix epoch milliseconds from the GNSS receiver (satellite time). `wall_ms` fields come from the phone's clock, which can be wrong: display only. |
| Units | metres, m/s, degrees (bearing clockwise from true north), nanoseconds, bytes. Latitude/longitude are WGS84 decimal degrees with 9 decimals. |
| Number format | `.` decimal separator, never a locale comma. Values may use E-notation (`1.5E-4`). An empty field means "not available". Booleans are `0`/`1`. |
| CSV | UTF-8, comma-separated, one header row, RFC 4180 quoting for fields that contain commas or quotes (`events.csv` JSON, `nmea.csv` sentences). Files are append-only and flushed to the OS every 250 ms and fsync'ed every 5 s. **After a crash the last line may be incomplete: ignore an incomplete last line.** |
| Image orientation | Video frames are stored in the **sensor's native orientation** with no rotation flag. Rolling-shutter timing always uses the stored (sensor) row. See `orientation` in session.json. |
| Folder name | `<UTC yyyyMMdd'T'HHmmss'Z' from the phone clock>_<device_id>_<4 hex>`, e.g. `20261022T062452Z_SM-G991B-7F3A_391e`. The time part is informational only. |
| Location on the phone | `/storage/emulated/0/GeoAI/sessions/<session_id>/` (normal, "All files access" granted). Fallback if not granted: `/storage/emulated/0/Android/data/com.geoai.capture/files/GeoAI/sessions/` (deleted by Android on uninstall). Device-check reports: `…/GeoAI/device_reports/`. |

## 2. Folder layout

```
<session_id>/
  session.json            session, device, camera, capture settings, encoders, survey metadata
  video/seg_0001.mp4      fragmented MP4, one video track, no audio (seg_0002.mp4, … every 5 min)
  frames.csv              one row per encoded video frame, in file order
  segments.csv            one row per closed video segment
  gnss_fixes.csv          one row per GPS_PROVIDER fix
  gnss_status.csv         one row per satellite per GNSS status callback (~1 Hz)
  nmea.csv                raw NMEA sentences
  gnss_raw.txt            OPTIONAL raw GNSS measurements (GnssLogger "Raw" format)
  imu.csv                 accelerometer, gyroscope, gravity, rotation vector, magnetometer
  analysis.csv            road-region brightness/sharpness of the preview stream (~6.7 Hz)
  exposure_control.csv    every shutter/ISO decision
  clock_sync.csv          clock pairs every 10 s
  thermal.csv             thermal status, headroom, battery every 5 s
  events.csv              operator markers and system events
  manifest.json           SHA-256 of every file + ECDSA signature (written last)
  pc_receipt.json         written by the PC pull tool after verification (not part of the manifest)
```

Sensor logging starts when the operator opens the camera (session creation) and stops 3 s after video stops (**post-roll**), so GNSS fixes normally bracket all frames.

## 3. Session types

| `session_type` | Purpose | Pipeline use |
|---|---|---|
| `survey` | Road survey. Recording requires a GNSS fix ≤ 10 m unless the operator forces it (`recording_forced_without_gnss` event). | Full extraction |
| `test` | Bench or emulator tests. No GNSS gate. | Ingest checks only |
| `calibration_intrinsics` | ChArUco board moved through the frame with the survey focus lock. | Lens calibration ([pipeline/04](pipeline/04-calibration-and-geometry.md)) |
| `calibration_mount` | Vehicle parked, ground target in view. | Mount pose / road homography |

## 4. session.json

Written three times, atomically (temp file + fsync + rename): at session creation, when recording starts (`recording_in_progress: true`), and at the end. A session that ended in a crash keeps the second version and is finalised by the app's **Finalise** action (`recovered: true`, `end.reason = recovered_after_interruption`).

```jsonc
{
  "spec_version": "1.1",
  "session_id": "20261022T062452Z_SM-G991B-7F3A_391e",
  "session_type": "survey",
  "boot_count_start": 412, "boot_count_end": 412,          // end: final version only
  "app": { "package": "com.geoai.capture", "version_name": "0.1.0", "version_code": 1, "build_type": "release",
           "settings": { /* every capture setting, see §4.1 */ } },
  "device": { "device_id": "SM-G991B-7F3A", "manufacturer": "samsung", "model": "SM-G991B", "soc_model": "…",
              "build_fingerprint": "…", "android_release": "15", "sdk_int": 35, "security_patch": "…" },
  "survey": { "road_id": "", "road_name": "", "direction": "increasing_chainage", "lane": "L1",
              "start_chainage_km_text": "12.4", "start_chainage_km": 12.4, "operator": "", "vehicle": "",
              "weather": "dry_sunny", "notes": "" },
  "mount": { "mount_id": "M1", "camera_facing": "forward", "nominal_height_m": 1.6, "nominal_pitch_deg": 45 },
  "calibration_target": { "spec": "…", "placement": "…" },  // calibration sessions only
  "camera": { /* §4.2 */ },
  "gnss": { "provider": "gps", "raw_measurements_logged": false, "full_tracking": false,
            "gnss_hardware_model_name": "…", "gnss_year_of_hardware": "…", "gnss_capabilities": "…" },
  "imu": { "sensors": [ { "name": "acc", "sensor_name": "…", "vendor": "…", "requested_period_us": 5000, "min_delay_us": 2500 } ] },
  "storage": { "root": "/storage/emulated/0/GeoAI", "shared_storage": true },
  "clock": {
    "session_clock": "elapsedRealtimeNanos (CLOCK_BOOTTIME)",
    "camera_timestamp_source": "REALTIME",            // REALTIME | UNKNOWN | UNREPORTED
    "elapsed_minus_mono_ns_at_start": 7190000000000,  // boot time minus monotonic = sleep since boot
    "encoder_timestamp_base": "SENSOR",               // from recording start: SENSOR (API 33+) | DEFAULT
    "sensor_to_session_offset_ns": 0,                 // add to SENSOR_TIMESTAMP to get the session clock
    "encoder_pts_offset_ns": 0,                       // final: pts_us*1000 − sensor_ts_ns, as determined by the app
    "encoder_pts_offset_method": "candidate",         // final: candidate | histogram
    "elapsed_minus_mono_ns_at_end": 7190000000000     // final
  },
  "start": { "t_ns": 0, "mono_ns": 0, "wall_ms": 0 },
  "capture_settings": { /* §4.3, from recording start */ },
  "encoders": [ { /* §4.4, one entry per encoder started (step-down/recovery add entries) */ } ],
  "orientation": { "sensor_orientation_deg": 90, "gravity_device_xyz": [6.9, 0.1, 6.9], "verdict": "UPRIGHT",
                   "rule": "UPRIGHT means sensor row 0 (top of the stored image) faces up in the world" },
  "recording": { "first_frame_t_ns": 0, "last_frame_t_ns": 0, "segments": 1,          // final only
                 "thermal_step_down": false, "camera_recoveries": 0 },
  "end": { "t_ns": 0, "mono_ns": 0, "wall_ms": 0, "reason": "user_stop" },           // final only
  "stats": { "frames_encoded": 0, "frames_unmatched": 0, "frames_matched": 0, "camera_frames_not_encoded": 0,
             "samples_dropped_before_first_keyframe": 0, "gnss_fixes": 0, "gnss_restarts": 0,
             "mock_location_seen": false, "max_thermal_status": 0, "distance_m": 0, "warnings": [] }
}
```

`end.reason`: `user_stop`, `thermal_critical`, `thermal_step_down_failed`, `low_storage`, `camera_error`, `service_destroyed`, `recovered_after_interruption`.

### 4.1 app.settings

`device_id, resolution (UHD_2160|FHD_1080), codec (HEVC|AVC), bitrate_mbps (at 4K; scaled by area for smaller sizes, floor 12 Mbps), fps (30), blur_budget_mm, max_exposure_denominator, low_light_exposure_denominator, max_iso, target_luma, unknown_speed_kmh, ois_enabled, segment_minutes, thermal_step_down, raw_gnss, gnss_full_tracking, keep_screen_on, alerts`.

### 4.2 camera (static characteristics of the camera used)

`camera_id, lens_facing, physical_camera_ids, hardware_level, capabilities[], timestamp_source, sensor_orientation_deg, focal_lengths_mm, equivalent_focal_mm, apertures, sensor_physical_size_mm [w,h], pixel_array_size [w,h], active_array_size [l,t,r,b], lens_intrinsic_calibration [fx,fy,cx,cy,s] (pixels of the sensor's pre-correction active array, may be null), lens_distortion (may be null), lens_pose_rotation, lens_pose_translation, exposure_time_range_ns, sensitivity_range, max_frame_duration_ns, focus_distance_calibration, min_focus_distance_diopters, video_stabilization_modes, ois_modes, zoom_ratio_range, record_sizes, ae_fps_ranges`.

> `lens_intrinsic_calibration` is a factory value for the full active array. It is a useful prior but **not** a substitute for the ChArUco calibration at the recording resolution.

### 4.3 capture_settings (what was actually configured)

`exposure_mode` (`MANUAL_SHUTTER_AUTO_ISO` normal; `AUTO_FALLBACK` = camera has no manual sensor control, motion blur uncontrolled), `blur_budget_mm, max_exposure_ns, low_light_max_exposure_ns, max_iso, target_luma, unknown_speed_mps, focus_mode` (`LOCKED_MANUAL | LOCKED_AF | FIXED | CONTINUOUS_VIDEO`), `focus_distance_diopters, eis` (always `OFF`), `ois, zoom_ratio` (1.0), `awb` (`AUTO`), `template` (`TEMPLATE_RECORD`), `fps, frame_duration_ns, ae_fps_range, analysis_stream {width,height,format}, streams_while_recording` (`encoder+analysis` normal; `encoder_only` = exposure frozen while recording), `encoder_timestamp_base, record_size`.

### 4.4 encoders[]

`mime, codec_name, hardware_accelerated, width, height, fps, bitrate_bps_requested, bitrate_bps_applied, bitrate_mode, profile, i_frame_interval_s (1), config_level` (0 = full format; 1 = without priority/parameter-set/profile keys; 2 = minimal; 3 = minimal **without** `KEY_MAX_B_FRAMES`, treat with suspicion), `config_failures[], max_b_frames, prepend_parameter_sets_to_keyframes, color {standard BT709, range LIMITED, transfer SDR} | null`.

## 5. video/seg_NNNN.mp4

- Fragmented MP4 (Media3 `FragmentedMp4Muxer`), fragments of about 2 s, keyframe every 1 s, every segment starts on a keyframe, no B-frames, no audio. Typical timescale 90 kHz.
- A new segment starts at the first keyframe after `segment_minutes`, or when the encoder restarts (thermal step-down to 1080p, camera recovery). **Segments of one session can have different resolutions.**
- If the app or phone died, the last segment is readable up to its last complete fragment.
- **Container PTS are sample keys, not times.** The muxer estimates the last sample duration of each fragment, so file PTS drift from the true capture times by milliseconds per fragment (measured on the emulator: 5–14 ms after 20 s). Frame times come **only** from `frames.csv`.

## 6. frames.csv: one row per encoded frame, in the same order as the samples in the files

| column | type | meaning |
|---|---|---|
| `segment` | int | 1-based; file `video/seg_%04d.mp4` |
| `frame_in_segment` | int | 0-based sample index inside that file (**the join key**) |
| `frame_global` | int | 0-based index across the session |
| `pts_us` | int64 | presentation timestamp handed to the muxer (encoder clock, µs) |
| `sensor_ts_ns` | int64 | `CaptureResult.SENSOR_TIMESTAMP`: start of exposure of the first sensor row. Empty = no camera metadata matched (see `t_ns`) |
| `t_ns` | int64 | start of exposure on the **session clock** = `sensor_ts_ns + clock.sensor_to_session_offset_ns`; for unmatched frames estimated from `pts_us` |
| `offset_ns` | int64 | encoder-to-camera offset used: `pts_us*1000 − sensor_ts_ns` (0 with `encoder_timestamp_base = SENSOR`) |
| `match_residual_ns` | int64 | `pts_us*1000 − offset_ns − sensor_ts_ns` (signed, normally < 1 µs) |
| `capture_frame_number` | int64 | camera request number (gaps = camera-side drops) |
| `exposure_ns` | int64 | actual exposure time |
| `frame_duration_ns` | int64 | actual frame duration |
| `iso` | int | actual sensitivity |
| `rs_skew_ns` | int64 | rolling-shutter skew: first row to one-past-last row exposure start |
| `focus_diopters` | float | `LENS_FOCUS_DISTANCE` |
| `focal_length_mm` | float | `LENS_FOCAL_LENGTH` |
| `lens_state`, `ois_mode`, `eis_mode`, `ae_state`, `af_state` | int | Camera2 enum values |
| `awb_gains` | string | `r;g_even;g_odd;b` |
| `zoom_ratio` | float | `CONTROL_ZOOM_RATIO` |
| `is_keyframe` | 0/1 | sync sample |
| `encoded_bytes` | int | sample size |

Invariants: `t_ns` strictly increasing; per segment, rows ⇔ samples 1:1 in order (after a crash `frames.csv` may hold **more** rows than the last file has samples: drop the excess rows). Time of a pixel row `r` (sensor orientation, image height `H`): `t_ns + exposure_ns/2 + rs_skew_ns × r / H`.

## 7. segments.csv: one row per closed segment

`segment, file, start_t_ns, end_t_ns, first_pts_us, last_pts_us, frames, bytes, width, height, mime, bitrate_bps, close_reason`

`start_t_ns`/`end_t_ns` are when the file was opened/closed (not frame times). `close_reason`: `rotation`, `format_change`, or the stop reason (`user_stop`, `thermal_step_down`, `thermal_critical`, `error`, …). A segment open at a crash has no row: derive it from `frames.csv`.

## 8. GNSS

**gnss_fixes.csv**: `t_ns, t_unc_ns, receipt_ns, utc_ms, lat_deg, lon_deg, alt_ellipsoid_m, alt_msl_m, h_acc_m, v_acc_m, speed_mps, speed_acc_mps, bearing_deg, bearing_acc_deg, is_mock, provider, n_sats_used`

- `t_ns` = `Location.getElapsedRealtimeNanos()` (time of the fix as reported by the platform); `receipt_ns` = when the app received it. `utc_ms` = `Location.getTime()` (GNSS time for `gps`).
- `h_acc_m`, `v_acc_m`, `speed_acc_mps`, `bearing_acc_deg` are 68 % values. `speed_mps`/`bearing_deg` are normally Doppler-derived. `bearing_deg` is meaningless below ~1 m/s.
- `is_mock = 1` makes the session untrusted.

**gnss_status.csv**: `receipt_ns, svid, constellation, cn0_dbhz, carrier_hz, used_in_fix, elevation_deg, azimuth_deg, has_ephemeris, has_almanac, baseband_cn0_dbhz`. Constellation codes are `android.location.GnssStatus.CONSTELLATION_*` (1 GPS, 3 GLONASS, 4 QZSS, 5 BeiDou, 6 Galileo, 7 IRNSS/NavIC). `carrier_hz` ≈ 1.176 GHz means an L5/E5a/B2a signal.

**nmea.csv**: `receipt_ns, timestamp_ms, sentence` (sentence is a quoted CSV field).

**gnss_raw.txt** (only if `gnss.raw_measurements_logged`): lines starting `#` are comments; data lines are `Raw,…` records with the columns of the `# Raw,` header (Google GnssLogger layout; `ChipsetElapsedRealtimeNanos` is on the session clock).

## 9. imu.csv

`t_ns, sensor, x, y, z, w, accuracy`. `sensor`: `acc` (m/s², 200 Hz), `gyro` (rad/s, 200 Hz), `gravity` (m/s², 50 Hz), `game_rv` (unit quaternion x,y,z,**w**, 50 Hz), `mag` (µT, 50 Hz). Device axes in the phone's natural orientation (x right, y up, z out of the screen). `acc`/`gravity` follow the Android convention: a phone lying face-up reads +9.81 on z.

## 10. analysis.csv and exposure_control.csv

- **analysis.csv** `t_ns, mean_luma, highlight_frac, dark_frac, sharpness, exposure_ns, iso, speed_mps`: statistics of the small YUV preview stream over the road region (lower 60 %, central 80 % of the frame). `t_ns` = that frame's start of exposure. `sharpness` is a relative Laplacian-variance index. Absent while `streams_while_recording = encoder_only`.
- **exposure_control.csv** `t_ns, reason, exposure_ns, iso, low_light, speed_mps, mean_luma`: each new shutter/ISO request (`reason`: `metered | luma | speed`). `low_light = 1` means ISO hit its cap and the shutter was lengthened: blur risk.

## 11. clock_sync.csv and thermal.csv

- **clock_sync.csv** `t_ns_before, mono_ns, t_ns_after, wall_ms`: `elapsedRealtimeNanos`, `System.nanoTime`, `elapsedRealtimeNanos` read back to back (best of 7). Every 10 s, at recording start and at the end.
- **thermal.csv** `t_ns, thermal_status, headroom_10s, battery_temp_c, battery_pct, charging`. `thermal_status`: 0 none, 1 light, 2 moderate, 3 severe, 4 critical, 5 emergency, 6 shutdown. `headroom_10s` is Android's 10-second thermal forecast: 0.0 = no throttling, 1.0 = severe throttling (empty if unsupported).

## 12. events.csv

`t_ns, type, value_json`. Error events carry `{error, stack[], causes[]}`.

| Group | Types |
|---|---|
| Session | `session_start, stop_requested, session_end, prepare_failed` |
| Recording | `recording_started, recording_start, recording_start_failed, recording_forced_without_gnss, segment_start, segment_end` |
| Camera / exposure | `lock, lock_restored, analysis_stream_unavailable, repeating_request_error, capture_failed, buffer_lost, camera_lost, camera_recovery_attempt, camera_recovered, camera_recovery_failed, image_black, image_ok` |
| Encoder / muxer | `encoder_output_format, encoder_fallback, encoder_error, encoder_eos_timeout, muxer_write_error, muxer_close_error, non_monotonic_pts` |
| GNSS | `gnss_lost, gnss_recovered, gnss_restart` |
| Thermal | `thermal_status, thermal_step_down, thermal_step_down_failed` |
| Operator | `operator_marker` (`marker_type`: `km_stone` with the chainage in `note`, `distress`, `event`), `warning` (each message at most once a minute) |

Frames between `image_black` and `image_ok` are black (lens covered, camera muted): exclude them.

## 13. manifest.json and pc_receipt.json

```json
{
  "spec_version": "1.1",
  "session_id": "…",
  "created_wall_ms": 1792649179701,
  "files": [ { "path": "frames.csv", "bytes": 89238, "sha256": "…" } ],
  "signature": { "alg": "ECDSA_P256_SHA256",
                 "payload": "canonical JSON of {spec_version, session_id, created_wall_ms, files}",
                 "public_key_x509_b64": "…", "value_b64": "…" },
  "attestation_chain_b64": [ "…" ]
}
```

- `files` lists every file except `manifest.json` and `pc_receipt.json`, sorted by path, `/` separators. Hashes are computed by re-reading the files from storage.
- Canonical payload = `json.dumps({spec_version, session_id, created_wall_ms, files}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`. The signature is DER ECDSA P-256/SHA-256 by a per-install Android Keystore key. If signing failed: `"signature": null, "signature_error": "…"`.
- **pc_receipt.json** (written by `tools/pull_sessions.py`, pushed back to the phone): `{status: "verified", session_id, manifest_sha256, files_verified, verified_utc, pc_host, tool, note}`. The app shows a session as VERIFIED_ON_PC (safe to delete) only when `manifest_sha256` equals the SHA-256 of its own `manifest.json`. Deleting any other session needs a separate "Delete UNVERIFIED session?" confirmation.

## 14. Checks the PC runs on ingest

| # | Check | On failure |
|---|---|---|
| 1 | `manifest.json` present; every listed file exists with matching size and SHA-256 | reject |
| 2 | Signature valid (if `cryptography` is available) | reject |
| 3 | `spec_version` MAJOR supported; `boot_count_start == boot_count_end` | reject / warn if end missing (recovered) |
| 4 | Per segment: container PTS strictly increasing and unique; first sample is a keyframe | reject segment |
| 5 | Per segment: samples == `frames.csv` rows; fewer samples than rows only in the last segment of a recovered session | reject segment / trim rows |
| 6 | `t_ns` strictly increasing; unmatched-metadata frames counted | warn |
| 7 | Dropped frames (gaps > 1.5 × median) | warn above 0.1 % |
| 8 | GNSS fixes before the first and after the last frame; max gap; `is_mock` | warn / untrusted |
| 9 | `exposure_mode`, `streams_while_recording`, `orientation.verdict`, encoder `config_level` | warn |
| 10 | Each segment's resolution has a matching lens calibration | block extraction until calibrated |
