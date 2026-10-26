# 13 · Segment mode: one best full frame per 5 m of road

Status: specification of 27 Oct 2026, the contract the implementation follows. The crop mode (documents 05–07) stays available as `--mode crops`.

## 1. Why

- **Fewer VLM calls.** The crop mode produced 1,500–2,200 crops per km on the first S21 sessions, about 6–8 h of VLM time per km. Segment mode sends at most one image per 5 m: 200 per km before gating, under 47 min per km at 14 s each.
- **The VLM's measured regime.** The VLM team evaluated Qwen2.5-VL on whole road frames (Attain, up to 1920×1080, resized by its processor to at most 1,048,576 px). A whole S21 frame resized the same way (3840×2160 → 1344×756) is that kind of input. Crops were never evaluated.
- **Countable output.** Each segment is classified once. Segments don't overlap, so there is no double counting and no duplicate merging. Results translate directly into "% of surveyed length with distress X", the quantity IRC:82 section ratings use.
- **Frame quality.** The sharpest frame of each segment is chosen, which removes focus and vibration blur. It cannot remove motion smear, which is the same in every frame at a given speed and shutter.

Known limit, stated in every report: after the VLM's resize, along-track ground resolution is about 2.5 mm/px at 1.5 m ahead, 4 mm at 2 m, 13 mm at 4 m and 25–30 mm at 6–7 m (S21, 23° tilt). Potholes, patches, ravelling, alligator and wide cracks stay visible; hairline and fine cracks beyond about 2 m do not.

## 2. Geometry (fixed mount, 23° tilt)

The mount is bolted and cannot be tilted further down, so the design works with what the frame shows.

- `near_m`: the closest road visible, at the bottom image row of the centre column (S21 pilot: 1.44 m ahead of the camera nadir).
- `far_m`: the farthest road visible, at the top image row or the horizon, capped at 60 m (pilot: 33 m).
- A segment `[s0, s1)` (path distance along the GNSS trajectory, `s1 = s0 + length_m`) is fully in view from camera position `c` when `c + near_m ≤ s0` and `c + far_m ≥ s1`.
- **Candidate window:** `c ∈ [s0 − near_m − window_m, s0 − near_m]`, i.e. frames where the segment's near edge is between the closest visible road and `window_m` (1.0 m) beyond it. The nearest views give the finest resolution.
- **Image rows of a segment** in a frame: rows showing ground distances `s0 − c` to `s1 − c` ahead on the centre line (flat road), from `CameraModel.ground_to_pixel`.
- Camera position `c` of a frame = trajectory `s` at the capture time of the image's centre row (`_RowClock`). The rolling shutter shifts other rows by at most ±8 ms, about 9 cm at 40 km/h; that is negligible against 5 m segments.

## 3. Best frame per segment (`geoai/segments.py`)

- Segments start at the first valid trajectory position and run to the last. A final partial segment is kept if it is at least half the length, flagged `partial`.
- **Candidates:** decoded frames whose camera position is in the window. Excluded: frames in black intervals, GNSS gaps, while reversing, or below `min_speed_mps`.
- **Sharpness:** the normalised Tenengrad of the segment's rows, on a grey image downscaled by `sharp_downscale`, columns `sharp_cols` (the lane, not the roadside). It is `mean(Gx² + Gy²) / mean(I)²` with 3×3 Sobel, so it doesn't depend on exposure.
- **Predicted blur** (metadata; also a filter):
  - motion = speed × exposure / along-track ground resolution at the segment's centre row;
  - rotation = |ω⊥| × exposure × focal length in px, with ω⊥ the gyro rate about the two image axes, from `imu.csv` at the frame time;
  - blur = √(motion² + rotation²).
- **Choice:** the candidate with the highest sharpness among those with blur ≤ `max_blur_px` and not low-light (unless `skip_low_light` is false and nothing else exists); ties go to the nearest. The record keeps the number of candidates, the chosen frame's rank and its reason.
- **No candidate** (GNSS gap, all frames black): the segment gets status `not_covered` with the reason. `report.json` totals the uncovered length.
- **Streaming, one decode pass:** `BestFrameSelector.wants(c)` decides whether a frame needs RGB conversion; `offer()` scores it; `pop_ready(c_now)` returns segments whose window has passed, each with its best frame's RGB (at most about two open segments at a time, so memory stays bounded).

## 4. Distress check on the segment (`geoai/gate.py`)

- **Analysis view:** full width, rows from the far edge of the window (`near_m + length_m + window_m + 0.5 m`) down to the bottom usable row. It is downscaled as in the crop mode (target 2 mm/px median, never upscaled), and the analysis area is capped at 4 MP.
- **Detectors:** the same three as the crop mode (DINOv2 blocks 4+8), from `AnomalyDetector.maps()`:
  - the session bank (patch score u);
  - the Attain probe (patch logit vs its seed threshold);
  - the Hessian crack filter.
- **Scores restricted to the segment's patches** (rows of `[s0, s1)` in the analysis image, all columns): `probe_frac`, `probe_max_logit`, `evidence` (bank and crack candidates whose centroid lies in the segment), `crack_mm`, `z_max`, `anomalous_patch_frac`, `n_patches`.
- **Decision:** `send` when any of `probe_frac`, `evidence`, `crack_mm` reaches its threshold (`GateConfig`). Otherwise `audit` for a deterministic `audit_frac` of segments (SHA-256 of the record id), or `skip`. With `mode = all`, every covered segment is `send` and the gate's would-be decision is still recorded (`gate.decision_if_gated`).
- **Calibration** (`geoai gate-calibrate BATCH...`, `gate.calibrate()`): from segments the VLM answered, with VLM distress = any valid non-Normal label, it finds, for each signal and for the any-of rule, the thresholds that keep `target_recall` (0.9) of the VLM-positive segments at the lowest pass rate. It writes `gate_calibration.json`, including the curves; `GateConfig.calibration` loads it. No labels by us: the VLM is the reference, and that caveat is written into every output.

## 5. Records and files

Run folder `RUNS/<session_id>/` (`report.json → mode = "segments"`):

| Path | Content |
|---|---|
| `frames/<id>.png` | The best frame, full resolution (3840×2160), lossless PNG, for `send` and `audit` segments |
| `sidecars/<id>.json` | One record per segment, all statuses |
| `segments.csv`, `segments.geojson` | All records; GeoJSON features are segment footprints with properties |
| `segment_views/<id>.png` | Only with `export_segment_view`: the segment rows at native resolution (research comparison) |
| `contact_sheet.jpg` | Thumbnails of the first 48 sent frames, in route order |
| `report.json` | Mode, geometry, coverage (covered / not covered metres), gate pass rates, timings, VLM-time estimate |

**Record schema `geoai.segment.v1`** (sidecar JSON, and the PNG's `geoai` text chunk):

```
id                "<session_id>_seg<k:05d>"; kind "segment"; schema; status send | audit | skip | not_covered
session_id, pipeline_version, config_sha256
segment   k, length_m, s0_m, s1_m, partial, chainage_start_km, chainage_end_km (from the survey form's start
          chainage and direction; null if not given), start{lat,lon}, end{lat,lon}, centre{lat,lon},
          footprint_lonlat [[lon,lat]...] (segment length x visible width at the segment, closed ring)
frame     file, sha256, width, height, segment (video segment), frame_in_segment, frame_global, t_ns, utc_ms,
          utc_iso, local_iso, rows_of_segment [v_top, v_bottom], near_edge_ahead_m, far_edge_ahead_m
camera    lat, lon, alt_ellipsoid_m, alt_msl_m, heading_deg, view_direction_deg, speed_mps, h_sigma_m, utc_ms
target    lat, lon (segment centre), bearing_deg, distance_m (camera -> centre), sigma_m
capture   exposure_ns, iso, frame_duration_ns, rs_skew_ns, focus_diopters, focus_mode, ois_mode, low_light
quality   sharpness, sharpness_rank, n_candidates, blur_motion_px, blur_rotation_px, blur_px, gyro_rad_s, reason
gate      mode, decision, decision_if_gated, reasons[], scores{probe_frac, probe_max_logit, evidence, crack_mm,
          z_max, anomalous_patch_frac, n_patches}, thresholds{...}, audit_draw
survey    road_id, road_name, direction, lane, start_chainage_km, operator, vehicle, weather
device    manufacturer, model, device_id;  geometry  source, pitch_deg, roll_deg, height_m
flags     low_light, geometry_nominal, latency_uncorrected, heading_from_gyro, partial, ...
vlm       (added by vlm-import) labels[], stage1_label, stage1_confidence, severity, description, valid, problem
```

**EXIF** (`records.build_exif`): GPS = camera position, speed, heading, horizontal error, UTC date and time; GPSDest = segment centre (bearing, distance); DateTimeOriginal with sub-seconds and offset (IST); Make, Model, Software; ImageUniqueID. `build_exif` reads `rec["target"]` (segment mode) or `rec["anomaly"]` (crop mode).

## 6. VLM hand-off and analysis

- `geoai vlm-export RUN... --batch DIR` (segment runs): one request per `send` and `audit` segment, `kind = "segment"`. The image is the full-resolution PNG; the VLM processor resizes it exactly as in its evaluation. The batch also carries `records/<id>.json`, so analysis needs only the batch folder.
- On the VLM machine: `python run_vlm_batch.py DIR` (standard library; resumable; respects 10 requests/min), then `python -m geoai vlm-import DIR` and `python -m geoai analyze DIR`, which needs only numpy and pandas, no GPU.
- **`analyze` writes `DIR/analysis/`:**
  - `segments_results.csv` and `.geojson`: per segment, gate and VLM labels;
  - `label_summary.csv`: per IRC label, segments and % of surveyed length, overall and per session;
  - `per_km.csv`;
  - `gate_eval.json`: on `audit` and send-all segments, the gate's recall and precision against the VLM with Wilson 95 % intervals;
  - `vlm_timing.json`;
  - `REPORT.md`: methods text with versions, model id, prompt version, hashes, counts, the tables, and the limits of section 1.

## 7. Tests (must pass before release)

- **Unit:**
  - segment planning;
  - candidate windows and row geometry, against independent calculations;
  - sharpness ranks sharp over blurred and defocused versions of the same image, and is exposure-invariant;
  - blur prediction against a hand calculation;
  - the streaming selector picks the sharpest frame and uses bounded memory;
  - gate scores restricted to segment rows, the decision rule, a deterministic audit draw, and calibration on synthetic data;
  - record schema, EXIF round trip, CSV and GeoJSON;
  - export/import/analyze round trip on a fake answered batch, including partially answered and interrupted runs.
- **End to end on the simulator:** every segment of the route is covered (or `not_covered` with a reason in GNSS gaps); every simulated distress lies in a `send` segment with the probe off; reruns are byte-identical.
- **Real sessions:** both 26 Oct sessions process without errors; every sidecar is valid against the schema; every PNG's embedded record and EXIF match its sidecar.
