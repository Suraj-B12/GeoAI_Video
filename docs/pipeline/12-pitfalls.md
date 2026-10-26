# 12 · Pitfalls: Common Mistakes and Where the Design Prevents Them

> **Added by testing (pipeline v0.1):**
> - **Last-block ViT features.** Using DINOv2's last block for anomaly maps made one pothole flag 62–90 % of its window. Blocks 4+8 are used instead ([05](05-detection.md) §2).
> - **Upper-quantile thresholds.** An upper quantile is either cut off by trimming or set by the bank's own distress. Bulk-based scaling is used instead ([05](05-detection.md) §3).
> - **Evenly spaced bank frames** phase-lock with periodic lane markings. Spacing is jittered.
> - **Inverted analysis downscaling** halved the resolution of an already-coarse image. Fixed and covered by the end-to-end test.
> - **Token minimum.** A frame-capped crop height could leave a crop under 324 tokens. Fixed by growing the other side; a regression test reproduces the case.
> - **JPEG re-encoding** flips VLM labels (VLM team: 9 of 49 at q95). Crops are PNG.
> - **Object-level hit rates without a chance baseline.** "At least one patch above threshold" flatters large objects. Block cracks reached 100 % against 90 % by chance. The probe card reports both numbers.
> - **Testing a real-data model on synthetic texture.** The Attain probe flagged 16 % of procedural asphalt, against 0–1.8 % of real clean road. Each component is tested on the data that is valid for it.
> - **Resizing before the VLM.** Sending sizes that the VLM's processor resizes again breaks the pixel-identity guarantee. Crops are fixed points of `smart_resize`.

Systems like this usually fail quietly: the output looks plausible but is wrong by metres, double-counts distresses, or the VLM sees images unlike its training data. Each row names a mistake seen often in vehicle video surveys, what it breaks, and the part of the design that prevents it.

Use this as a **review checklist** for the pipeline code. Every row has a test somewhere in 01–11.

## Capture (phone app)

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| C1 | Letting auto-exposure pick the shutter speed | 1/500–1/1000 s smears 11–22 mm at 40 km/h; cracks vanish | App: speed-adaptive shutter, 2.5 mm blur budget, `low_light` flag ([PLAN §4.3](../PLAN.md)) |
| C2 | Video stabilisation (EIS) on | It warps and crops each frame differently; the calibration no longer applies | App: EIS off, logged per frame; ingest checks ([01](01-ingest.md) §3) |
| C3 | Autofocus hunting during recording | Blurred frames; focus changes the focal length | App: focus locked before recording, `focus_diopters` per frame; calibration matched by focus ([01](01-ingest.md) §5) |
| C4 | Lens switching or zoom on a multi-camera phone | Part of the session comes from a different camera | App: main camera, zoom fixed at 1.0; ingest flags any `focal_length_mm` or `zoom_ratio` change ([01](01-ingest.md) §3) |
| C5 | Phone mounted upside down or in portrait | Rotated images, wrong geometry | App: orientation verdict; `UPSIDE_DOWN` rotated with correct row timing, `PORTRAIT` rejected ([03](03-video-decoding.md) §3) |
| C6 | Recording with MediaRecorder or CameraX | Gaps at file splits, no per-frame timing | App: Camera2 → MediaCodec → fragmented MP4 ([PLAN §4.5](../PLAN.md)) |
| C7 | A crash loses the whole file (moov atom written at the end) | Hours of driving unreadable | App: fragmented MP4 (2 s fragments), recovery on next launch |
| C8 | Overheating in Indian summer sun | Shutdown mid-survey, throttled frame rate | App: thermal ladder (warn, 1080p, clean stop) plus shade and cooling ([../../android/README.md](../../android/README.md)) |
| C9 | Samsung battery management kills the service | Recording stops silently with the screen off | Foreground service, wake lock, battery exemption, and the Samsung settings in [../../android/README.md](../../android/README.md) |
| C10 | Recording before GNSS has a fix | Video with no usable position | App: survey sessions wait for a fix ≤ 10 m no older than 2 s; "Record anyway" is logged (`recording_forced_without_gnss`) |
| C11 | Location switched off mid-session from quick settings | Silent loss of positions | App: banner and alert, watchdog restarts GNSS when location returns; ingest reports gaps |
| C12 | Mock-location apps | Fabricated positions | `isMock` per fix; the session is marked untrusted ([01](01-ingest.md), [02](02-time-and-trajectory.md) §2) |
| C13 | Phone locale writes decimal commas (`12,97`) | Corrupt CSV numbers | App: locale-independent formatting (unit-tested) |
| C14 | Timing from the wall clock | NTP jumps, manual changes. **Offline phones often have unsynchronised clocks** | App: one session clock (`elapsedRealtimeNanos`); `wall_ms` for display only |

## Transfer and ingest

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| T1 | Copying multi-GB files over MTP | Truncated or failed copies on Windows | ADB pull plus SHA-256 verification ([01](01-ingest.md) §1) |
| T2 | Deleting from the phone before verifying | Data loss | The PC never deletes; the app marks verified sessions (matching `pc_receipt.json`) and shows a separate warning before deleting an unverified one ([01](01-ingest.md) §1) |
| T3 | Processing a session that spans a reboot | The clock reset mid-session | Boot-count check, blocking ([01](01-ingest.md) §2) |
| T4 | Modifying raw data in place | Results can't be reproduced | Raw sessions read-only; everything derived goes under `work/` ([README](README.md) §5) |
| T5 | Something quietly needs the internet | Fails in the field; data leaves the site | App has no INTERNET permission; local wheelhouse, `HF_HUB_OFFLINE`, tests run with sockets disabled ([README](README.md) §2, [11](11-testing-and-acceptance.md) §7) |

## Time

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| M1 | Frame time = index ÷ fps | Drift with variable frame rate and drops | Per-frame `t_ns` from `frames.csv` ([02](02-time-and-trajectory.md) §1) |
| M2 | Container PTS used as time | Fragmented-MP4 PTS drift (measured 5–14 ms after 20 s, growing) | PTS used only as a key ([03](03-video-decoding.md) §1) |
| M3 | Counting decoded frames to find a frame's row | One failed decode shifts every later frame | PTS → sample-index map ([03](03-video-decoding.md) §1) |
| M4 | Mixing clocks (encoder MONOTONIC, camera, GNSS) | Every position off by speed × offset | `TIMESTAMP_BASE_SENSOR`, the app's offset matching, one session clock (spec; [02](02-time-and-trajectory.md) §1) |
| M5 | Ignoring GNSS latency | 100 ms = 1.1 m along-track at 40 km/h | Per-session IMU cross-correlation, plus a device default from the marker test ([02](02-time-and-trajectory.md) §3, [11](11-testing-and-acceptance.md) §4) |
| M6 | Ignoring rolling shutter, or using exposure start | Up to 22 cm between top and bottom rows at 40 km/h | Per-row, mid-exposure time ([02](02-time-and-trajectory.md) §6) |
| M7 | UTC from the phone's wall clock | Off by seconds or minutes when offline | UTC fitted from GNSS fix times ([02](02-time-and-trajectory.md) §4) |
| M8 | Assuming daylight saving, or storing local time | Ambiguous or shifted times | UTC in the database; IST (+05:30, no DST) only in EXIF and reports ([08](08-geotag-and-records.md)) |

## Position and geometry

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| G1 | Phone position used as the distress position | 0.5–16 m error, depending on the mount | Road-plane projection ([04](04-calibration-and-geometry.md) §3); distress written to `GPSDest*` ([08](08-geotag-and-records.md)) |
| G2 | Linear interpolation of 1 Hz fixes | Cuts corners on curves | Hermite spline with Doppler velocities ([02](02-time-and-trajectory.md) §5) |
| G3 | Interpolating across GNSS gaps | Invented positions | Gaps > 3 s excluded and reported ([02](02-time-and-trajectory.md) §5) |
| G4 | GNSS heading at low speed or when reversing | Random or 180°-wrong projection | Gyro-propagated heading below 1.5 m/s; reversing detection ([02](02-time-and-trajectory.md) §5) |
| G5 | Correcting camera attitude with raw gravity | A 5 % grade tilts it 2.9°, about 15 cm of error | Static mount calibration; gravity used only for QC ([02](02-time-and-trajectory.md) §7) |
| G6 | Taking `tvec` from `solvePnP` as the camera position | Wrong mount height and offset | `c = −Rᵀ·tvec` ([04](04-calibration-and-geometry.md) §3) |
| G7 | One calibration for every resolution | 1080p step-down frames projected with 4K intrinsics | Intrinsics per `W×H` ([03](03-video-decoding.md) §1, [04](04-calibration-and-geometry.md) §1) |
| G8 | Focus changed after lens calibration | Focal length shifts slightly | Focus match ±0.05 dpt ([01](01-ingest.md) §5) |
| G9 | Mount bumped after calibration | Every projection biased | Daily calibration, gravity check at start, end-of-day check, step detection ([01](01-ingest.md) §3, §5) |
| G10 | Rearward camera treated as forward | Distresses placed ahead of the vehicle instead of behind | Mount yaw ≈ 180° from calibration; dedicated test ([04](04-calibration-and-geometry.md) §5) |
| G11 | Ellipsoidal height written as MSL altitude | Tens of metres wrong in India (geoid up to ~100 m below the ellipsoid) | Separate fields; geoid grid ([08](08-geotag-and-records.md) §1–2) |
| G12 | Latitude/longitude swapped in GeoJSON or KML | Points land in the wrong country | `[lon, lat]` order with tests ([10](10-review-and-export.md) §3, §6) |
| G13 | Areas computed in degrees | Meaningless areas | Metric ENU polygons ([10](10-review-and-export.md) §3) |
| G14 | Expecting sub-metre absolute positions from phone GNSS | False confidence | Error budget and stated 1σ per record ([04](04-calibration-and-geometry.md) §4); RTK/PPK as the upgrade |

## Images

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| I1 | Wrong colour conversion (BT.601 or full range) | Colour and contrast shift between sessions | Explicit BT.709 limited-range conversion; colour test that must fail on BT.601 ([03](03-video-decoding.md) §2, [11](11-testing-and-acceptance.md) §3) |
| I2 | Shrinking 4K frames to 640 px for detection | A 3 mm crack becomes under 1 px | Native-resolution tiles ([05](05-detection.md) §3) |
| I3 | 4:2:0 JPEG, low quality, or repeated re-encoding | Colour bleeding at crack edges | One JPEG encode, q95, 4:4:4 set explicitly ([07](07-best-frame-and-crop.md) §4) |
| I4 | Enlarging small crops | Interpolation blur the VLM never saw in training | Never enlarge; add context instead ([07](07-best-frame-and-crop.md) §3) |
| I5 | Burning GPS or time text into the image | Covers pavement; confuses the VLM | Clean crops always; a separate strip below only if the VLM was trained that way ([07](07-best-frame-and-crop.md) §4) |
| I6 | Crops cut off at the image edge | Partial distress, wrong class | Completeness rule; shift inward ([07](07-best-frame-and-crop.md) §2–3) |

## Detection and training

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| D1 | Random train/test split by frame, or the same road in both splits (on different dates) | Neighbouring frames are near-duplicates and the same pothole reappears, so scores are inflated | The split unit is the road section ([../detection-model-plan.md](../detection-model-plan.md) §3.4) |
| D2 | Box-level metrics only | Says little about the inventory | Distress-level recall, duplicates and merges ([11](11-testing-and-acceptance.md) §6) |
| D3 | Ground truth built from the pipeline's own detections | Recall can't be measured | Independent frame-by-frame annotation ([11](11-testing-and-acceptance.md) §6) |
| D4 | Training only on public dashcam or handheld images | Domain gap with oblique, tilted-down 4K | Public data only for pre-training; own labelled footage is the core ([../detection-model-plan.md](../detection-model-plan.md)) |
| D5 | Geometry-breaking augmentations (vertical flips, 90° rotations of oblique views, heavy scale changes) | Teaches impossible views | Augmentation rules ([../detection-model-plan.md](../detection-model-plan.md)) |
| D6 | Look-alikes reported as distress | Manholes, markings, speed breakers and shadows in the inventory | Look-alike classes, VLM `no_distress`/`not_pavement`, review ([05](05-detection.md) §2, [09](09-vlm-interface.md) §6) |
| D7 | Detector tuned for precision | Missed distresses that no later stage can recover | High-recall operating point; the VLM and review handle precision ([05](05-detection.md) §3) |
| D8 | Licences checked too late | AGPL or share-alike obligations discovered at delivery | Apache-2.0 default (RF-DETR); licence decisions in Phase 0 ([PLAN §4.7](../PLAN.md)) |
| D9 | Training tiles that include unlabelled regions (far field, vehicle body) | Unlabelled distresses are taught as background, so recall drops | ROI-only training tiles ([../detection-model-plan.md](../detection-model-plan.md) §4) |
| D10 | Labelling a class only sometimes (e.g. some markings, not all) | The model learns the class is sometimes background | Exhaustive labelling rule ([../detection-model-plan.md](../detection-model-plan.md) §3.3) |
| D11 | Test labels made by correcting model pre-labels | Annotators miss what the model misses; recall is over-stated | Test labels made from scratch ([../detection-model-plan.md](../detection-model-plan.md) §3.3) |
| D12 | v1 box areas reported as true areas | Areas overstated (a round pothole fills ~79 % of its box) | v1 areas are upper bounds; masks in v2 ([../detection-model-plan.md](../detection-model-plan.md) §9) |

## Association and crops

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| A1 | Classifying every frame | One pothole becomes 8 records and 8 VLM calls | One record per physical distress ([06](06-association.md)) |
| A2 | IoU trackers at 160–400 px per frame | Tracks break, so duplicates | Road-plane path coordinates ([06](06-association.md) §1–2) |
| A3 | Auto-merging nearby records on a revisit | Two real potholes 2 m apart become one | Flag `possible_duplicate_of` for review, never merge ([06](06-association.md) §6) |
| A4 | Normalising crop scores within a track while streaming | The best view is discarded before it is known to be best | Absolute scores ([07](07-best-frame-and-crop.md) §1) |
| A5 | A 30 m crack as one record with one crop | No crop can show it; lengths unreliable | Grid pieces that fit one frame ([06](06-association.md) §5) |

## Metadata and records

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| R1 | Metadata only in EXIF | Lost when any tool strips EXIF | Sidecar JSON and database ([08](08-geotag-and-records.md) §3–4) |
| R2 | GPS tags written through a text form | Rounding to degrees-minutes | Numeric writes (`#` suffix) and read-back check ([08](08-geotag-and-records.md) §2) |
| R3 | Overwriting results with a new model's output | History and comparisons lost | Append-only classifications and reviews ([08](08-geotag-and-records.md) §4) |
| R4 | Non-idempotent stages | A re-run doubles the records | `stage_manifest.json`, deterministic IDs ([README](README.md) §5, [09](09-vlm-interface.md) §2) |

## VLM

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| V1 | Prompt, chat template or preprocessing differ from training | Silent accuracy loss | Parity table ([09](09-vlm-interface.md) §4) |
| V2 | Forcing a new output format (e.g. JSON) on a fine-tuned model | Changed answers | Use the trained format ([09](09-vlm-interface.md) §4) |
| V3 | Trusting self-reported confidence | Wrong review routing | Log-probabilities plus calibration ([09](09-vlm-interface.md) §7) |
| V4 | Judging the VLM by its own test set | Domain gap discovered only after deployment | Week-1 joint test; accuracy on pipeline crops ([../vlm-information-request.md](../vlm-information-request.md) §7, [11](11-testing-and-acceptance.md) §6) |
| V5 | Results from different model versions mixed | Unexplainable changes in reports | `model_id` and `prompt_version` on every result; `vlm.active` ([09](09-vlm-interface.md) §10) |
| V6 | Results lost or duplicated in the USB hand-off | Missing or double classifications | Deterministic `request_id`, checksums, resumable runner ([09](09-vlm-interface.md) §2–5) |

## Review and reporting

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| P1 | Reviewers see the model's answer first | They anchor on it; precision is over-stated | Blind audit sample ([10](10-review-and-export.md) §1) |
| P2 | Reviewing only the doubtful cases | Precision of the confident cases unknown | Random 5 % audit ([09](09-vlm-interface.md) §8) |
| P3 | % area relative to the nominal lane area | Understated distress where the camera didn't see the full lane | Observed area; coverage threshold ([10](10-review-and-export.md) §4) |
| P4 | Rut depth taken as zero, or a full IRC rating from video | Ratings better than the road | "Not measured"; partial ratings, clearly labelled ([10](10-review-and-export.md) §4) |
| P5 | Inventing a linear-crack-to-area rule | Unsupported cracking % | Required config; the report refuses without it ([10](10-review-and-export.md) §4) |
| P6 | Rating tables from the wrong IRC edition | Wrong ratings | Per-edition config, verified by an engineer ([10](10-review-and-export.md) §4) |

## Engineering

| # | Mistake | What goes wrong | Prevented by |
|---|---|---|---|
| E1 | In Kotlin, calling a method named `cancel()` inside `launch {}` | It resolves to `CoroutineScope.cancel()` and silently cancels the coroutine. This was found and fixed in this app | Method renamed `cancelSession()`, with a comment at the call site |
| E2 | Unpinned dependencies | Behaviour changes between machines | Wheelhouse with pinned versions; `pip install --require-hashes` |
| E3 | Phone OS updates during the project | Camera behaviour changes | Auto-update off; re-run the device check after any update ([PLAN §5.2](../PLAN.md)) |
