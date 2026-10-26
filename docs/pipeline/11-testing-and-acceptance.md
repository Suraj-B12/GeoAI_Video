# 11 · Testing and Acceptance

> **Implemented (pipeline v0.1):**
> - **Simulator** (§2): `geoai/sim.py` writes complete spec-v1.1 sessions with ground truth.
> - **Tests:** about 60 automated tests in `pipeline/tests`, including the end-to-end run, DINOv2 checked against Meta's own output, Qwen sizing against the VLM team's measurements, and the trajectory against synthetic drives.
> - **Field items** (§4–6: marker test, surveyed marks, annotated route) involve manual work. They are optional and only needed if absolute accuracy has to be proven. The automatic first-session checks are in [pipeline/README](../../pipeline/README.md).

Goal: prove each stage works before the next one is built on it, then prove the whole system meets its targets on real roads. There are three layers of tests:
- **synthetic**: exact ground truth;
- **recorded**: emulator and phone sessions;
- **field**: a surveyed and annotated route.

## 1. Test data (`testdata/`, versioned, with checksums)

| Set | Source | Used for |
|---|---|---|
| T1 Emulator sessions | The app on the Android emulator, driven by the debug intents ([../../android/README.md](../../android/README.md)): a normal session, a crash-recovered session, a thermal-stop session | Ingest, join, decode, recovery paths |
| T2 Synthetic sessions | The simulator (§2) | Timing, geometry, association, crops, geotags, against exact truth |
| T3 S21 bench sessions | Phone on a tripod: ChArUco board, ArUco mat, colour chart | Calibration, colour (§3) |
| T4 S21 field sessions | Pilot drives | End-to-end, throughput |
| T5 Golden frames | 20–50 frames from T4, boxes drawn by engineers | Detector regression ([05](05-detection.md) §7) |
| T6 Annotated test route | 5–10 km route, recorded at least 3 times (different days and light) | Recall, duplicates, VLM accuracy (§6) |

## 2. The simulator (T2)

The simulator writes a **complete session package in the exact spec format** ([../session-package-spec.md](../session-package-spec.md)), plus a ground-truth file. That lets every stage after ingest be tested against known answers.

- **World:** a flat ground plane with a tiled asphalt texture and synthetic distresses at known ENU positions. Cracks are polylines with a width, potholes are dark ellipses with rims, patches are rectangles of different texture, and markings are white stripes (look-alikes).
- **Drive:** a scripted trajectory with straights, curves (R = 50 m), braking, a stop, a three-point turn, and a GNSS gap.
- **Rendering:** for each pixel, ray-cast to the ground with the intrinsics and mount pose ([04](04-calibration-and-geometry.md) §3) and sample the texture. Model motion blur by averaging sub-frames over the exposure, rolling shutter by a row-dependent pose, and add sensor noise. Encode to fragmented MP4 with FFmpeg (HEVC, no B-frames).
- **Logs:**
  - GNSS fixes at 1 Hz with configurable latency (e.g. 150 ms), noise, outliers and gaps; Doppler speed and bearing;
  - IMU accelerometer, gyro and gravity consistent with the trajectory;
  - `frames.csv`, `clock_sync.csv`, `session.json`, and a manifest signed with a test key.
- **Oracle detector:** projects the true distress polygons into each frame, with a configurable miss rate, box jitter and false positives. It replaces stage 05, so geometry and association are tested without detector error.

| Stage | Synthetic pass condition |
|---|---|
| 02 | Latency within ±20 ms; position error < 0.10 m between fixes; reversing leg flagged |
| 04 | Projected ground points within 1 mm of truth (no noise) |
| 06 | Duplicate rate 1.00 with a 0 % miss rate; ≤ 1.05 with a 20 % miss rate and jitter |
| 07 | Every record's chosen view is complete; streaming choice equals batch choice |
| 08 | `GPSDest*` within 0.05 m of the true distress centroid (no GNSS noise) |
| 10 | Section percentages equal the hand-computed values |

## 3. Colour test (T3)

1. Record a printed colour chart (e.g. a 24-patch ColorChecker) in daylight with the app.
2. Decode one frame with the pipeline ([03](03-video-decoding.md) §2), and extract the same frame with FFmpeg, giving the conversion explicitly: `-vf "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"`.
3. Pass: mean absolute difference ≤ 1 level per channel. On the six grey patches, |R−G| and |B−G| ≤ 4 levels.
4. Also run the decode deliberately with BT.601 and check that the test **fails**. A test that can't detect the mistake proves nothing.

## 4. GNSS latency (field, marker test)

Two painted or taped white crosses (50 cm), about 200 m apart on a straight, open-sky road:

1. Drive past both crosses in both directions at 20, 40 and 60 km/h, 3 times each, back to back. Process these sessions **without** latency correction (`L = 0`).
2. For each pass, project the cross's image position to the map ([04](04-calibration-and-geometry.md)). Take its coordinate along the road axis `a` (unit vector from cross 1 to cross 2), minus that cross's mean over all passes.
3. Fit `offset = −L · v · dir + b` by least squares, where `dir = +1` when driving along `a` and −1 against it. With `L > 0` (fixes stamped late), the trajectory lags the vehicle, so the cross appears shifted **backwards** along the direction of travel by `v · L`.

**No surveyed coordinates are needed.** GNSS absolute error is nearly the same on back-to-back passes in both directions, so it ends up in `b` and cancels out of `L`. Store `L` as the device default in `config/devices/<device_id>.yaml` ([02](02-time-and-trajectory.md) §3).

**Pass:**
- The standard error of the fitted `L` is ≤ 20 ms. With 18 passes and about 1 m of position noise per pass, this is achievable.
- Fitted separately per speed, `L` agrees across the three speeds within its uncertainty. An offset that doesn't grow with speed is not a timing error.
- On the same sessions, the automatic per-session estimate agrees with the marker `L` within 30 ms.

## 5. Position accuracy (field)

- About 30 painted marks along the test route, coordinates surveyed with RTK or DGPS (Survey of India CORS, or the agency's surveyors).
- The pipeline geolocates each mark through the full chain.
- Report horizontal error at 68 % and 95 %, split into along-track and across-track.
- **Relative accuracy:** compare the pipeline distance between pairs of marks 10–50 m apart with the surveyed distance.

## 6. Annotated test route (T6)

**Ground truth must not come from the pipeline**, or recall can't be measured: whatever the detector missed would also be missing from the truth.

- Engineers annotate the route **video** frame by frame in CVAT (track mode): each distress once, in its clearest frame, with its IRC term and extent. Its position comes from the annotated frame through the same geometry, so position error doesn't confuse the matching.
- Match a pipeline record to a truth distress when they share a family and lie within 1.5 m in path coordinates.

| Metric | Definition | Target |
|---|---|---|
| Distress-level recall | Matched truth distresses ÷ all truth distresses, per family | Potholes, alligator cracking, patches ≥ 90 %; linear cracks wider than ~3 mm ≥ 80 % |
| Distress-level precision | Matched records ÷ all records, after the VLM and after review (reported separately) | Reported; no target before the VLM is known |
| Duplicate rate | Records matched ÷ truth distresses matched | ≤ 1.05 |
| Merge errors | Records matched to 2+ truth distresses ÷ records | ≤ 2 % |
| Long-crack coverage | Matched piece length ÷ truth length | ≥ 90 % |
| VLM accuracy on pipeline crops | Per-class precision and recall against **blind** engineer labels ([10](10-review-and-export.md) §1) | Within 5 points of the VLM's own benchmark, per class |

Measure detection on held-out roads and sessions only (never on footage used for training, [../detection-model-plan.md](../detection-model-plan.md)).

## 7. Acceptance summary

| Metric | Target | Evidence |
|---|---|---|
| Frame join | 100 % of samples 1:1; residual < 1 ms | Ingest QC on T1, T3, T4 |
| Dropped frames | < 0.1 %, no gaps between segments | Ingest QC on T4 |
| GNSS latency residual | ≤ 20 ms | §4 |
| Distress position, phone GNSS, open sky | ≤ 3–5 m (68 %) | §5 |
| Relative position (marks 10–50 m apart) | ≤ 0.5 m | §5 |
| Detection recall, duplicates, merges, long cracks | As in §6 | §6 |
| VLM accuracy on pipeline crops | Within 5 points per class | §6 |
| Throughput | 1 h of video in ≤ 1 h | T4 on the pipeline PC |
| Determinism | Re-processing a session gives identical records and crops | T4 |
| Offline | The whole test suite passes with networking disabled (`pytest --disable-socket` via `pytest-socket`) | CI |

## 8. Regression discipline

- Every stage's unit tests (listed in each document) run on every change.
- T1 and T2 have **golden outputs** (hashes of the stage outputs). A change to a golden output needs a written reason in the commit.
- A new detector or VLM version is accepted only after T5/T6 metrics are re-measured and recorded in its model card.

## 9. Field protocol (every pilot day)

1. **Before driving:** clean the lens; check the mount; record a 10 s `calibration_mount` session on the ArUco mat. Record a `calibration_intrinsics` session after any phone change, drop or repair. Make sure the phone is cool and storage is ≥ 2× the planned recording.
2. **While driving:** press `km_stone` at every stone; keep to the speed band (the app's speed turns amber above 40 km/h and red above 50).
3. **End of the day:** record a second 10 s `calibration_mount` session on the mat before touching the mount ([01](01-ingest.md) §5, end-of-day check).
4. **Same day:** pull the sessions over USB ([01](01-ingest.md)) and read the QC report. Re-drive any section that failed QC while the team is still on site.

The phone-side checklist for the first days with the S21 (device check, soak test, kill-and-recover test) is in [../../android/README.md](../../android/README.md).
