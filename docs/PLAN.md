# GeoAI Pavement Survey: Research Findings and Build Plan

Version 2, 22 Oct 2026 (v1: 18 Oct 2026).

**What changed in v2:**
- The pilot phone is fixed: Samsung Galaxy S21 5G.
- The pilot is **fully offline**: transfer by USB cable, no cloud upload, and no machine needs the internet.
- The VLM runs on a separate machine.
- Lens and mount calibration are recorded by the app and computed on the PC.
- The capture app is built ([../android/](../android/README.md)).
- The extraction pipeline has a full build specification ([pipeline/](pipeline/README.md)). It is built on another machine.

| Document | Purpose |
|---|---|
| [session-package-spec.md](session-package-spec.md) | The data contract between the phone and the PC (authoritative) |
| [pipeline/README.md](pipeline/README.md) | Build specification of the extraction pipeline, stages 01–12 |
| [detection-model-plan.md](detection-model-plan.md) | Detector data, labelling, training, evaluation |
| [vlm-information-request.md](vlm-information-request.md) | What we need to know about the VLM |
| [../android/README.md](../android/README.md) | Building, installing and operating the capture app |
| [../tools/](../tools/) | `mount_geometry.py` (mount calculator used for the numbers below), `validate_session.py`, `pull_sessions.py` |

Where this plan and those documents differ, **the documents win**: they were written later, against the built app.

**Roles.** The capture app and the extraction pipeline are both built in this repository: [`../pipeline/`](../pipeline/README.md), runnable offline on the pipeline PC. The VLM is owned and run by a separate team on its own machine.

**Update, 22 Oct 2026 (after the VLM team's answers in [`../vlm_handoff/`](../vlm_handoff/vlm-information-response.md)).**
- The VLM is stock Qwen2.5-VL-7B (no fine-tuning). It classifies; the pipeline only extracts "not healthy pavement".
- There is **no manual work** in the loop: no labelling and no review.
- So detection is label-free anomaly detection against the session's own normal pavement, instead of a trained detector.
- Coverage uses fixed road bands instead of tracking.
- Crops are PNG and sized exactly for Qwen2.5-VL.
- Results come back through an automatic folder batch.

Sections 4.7, 6.6–6.8 and 6.10–6.11 below describe the earlier design. Where they differ from [pipeline/README.md](pipeline/README.md), the implementation wins.

**Assumptions this plan is built on.** Correct any that are wrong; Section 11 lists the questions that change the design.

- Roads are in India, surveyed in daylight on dry pavement, mostly bituminous (flexible) pavements.
- One lane is surveyed per pass, at roughly 20 to 50 km/h.
- Pilot phone: Samsung Galaxy S21 5G (Android 13 or later on the unit).
- Fully offline: the phone, the pipeline PC and the VLM machine are never required to reach the internet.
- The VLM's exact input contract is not yet known ([vlm-information-request.md](vlm-information-request.md)).
- Android only. There is no requirement for iOS.

---

## 1. What we are building

Two layers joined by a strict data contract:

1. **Capture app (Android phone on the vehicle).** It records pavement video and, for every encoded frame, logs an exact timestamp on the same clock as the GNSS fixes and the motion sensors. It does no AI.
2. **Processing pipeline (PC).** It rebuilds the vehicle's path from the logs, finds distress in the frames, merges repeated sightings of the same distress into one record, cuts out the best view with a margin, attaches UTC time and coordinates, sends the crop to the VLM, and exports a geo-referenced distress inventory. IRC:82-style section ratings come later.

```
 PHONE (Android, Kotlin)                                PC (Python, NVIDIA GPU)
 ┌──────────────────────────────────┐   session        ┌──────────────────────────────────────────────┐
 │ Camera2 ─► MediaCodec (HEVC) ─►  │   package        │ 1 Ingest + integrity/QC report               │
 │   fragmented MP4 segments        │  (USB cable,     │ 2 Time model (one clock → UTC)               │
 │ CaptureResult per frame ─► CSV   │   ADB pull)      │ 3 Trajectory (GNSS + IMU → pose per frame)   │
 │ GNSS fixes/status/NMEA ─► CSV    │ ───────────────► │ 4 Decode (join by sample) + frame quality    │
 │ IMU 200 Hz ─► CSV                │                  │ 5 Detect (fine-tuned detector, tiled)        │
 │ Calibration, events, thermal     │                  │ 6 Project to road plane (calibration)        │
 │ manifest.json (SHA-256, signed)  │                  │ 7 Associate across frames → 1 record/distress│
 └──────────────────────────────────┘                  │ 8 Best frame → crop + margin                 │
                                                       │ 9 Geotag (EXIF/XMP + DB)                     │
                                                       │10 VLM classify (separate machine, USB)       │
                                                       │11 Review UI → corrections → retraining       │
                                                       │12 Export: GeoPackage/GeoJSON/CSV, IRC report │
                                                       └──────────────────────────────────────────────┘
```

The PC steps above are numbered as in this plan. The build specification organises them into documents 01–12 ([pipeline/README.md](pipeline/README.md)).

## 2. Decisions at a glance

| Area | Recommendation | Why (details in sections 4 to 6) |
|---|---|---|
| Phone software | Native Android (Kotlin) using Camera2 + MediaCodec + Media3 muxer | Needs per-frame timestamps, manual shutter, gap-free file splitting. CameraX, Flutter and React Native don't expose these. |
| Phone hardware | Pilot: **Samsung Galaxy S21 5G**, documented with the app's Device check (Camera2 level, MANUAL_SENSOR, timestamp source, 4K30 HEVC, GNSS). Later fleets: one standard model, qualified the same way | Camera timing and calibration differ by model, so one model means one calibration and one behaviour. |
| Video | 3840×2160 at 30 fps, HEVC 8-bit SDR, 60 Mbps, keyframe every 1 s, no audio, EIS off, focus locked, speed-adaptive shutter (about 1/4400 s at 40 km/h) | Motion blur, not pixel count, is what limits crack visibility (4.3). |
| Mount | Exterior rigid, vibration-damped bracket, 1.6 to 2.0 m high, pitched 45° to 55° down. Windshield mount is the fallback. | Even resolution across the frame, lane-wide coverage, no glass reflections or hood blocking the view (5.9). |
| Clock | Everything in Android `elapsedRealtimeNanos`; UTC derived from GNSS fix times | Four subsystems use different clocks by default (4.1). |
| Position per frame | Interpolated from roughly 1 Hz GNSS using Doppler velocity (Hermite spline); IMU smoothing in v2; RTK/PPK as an upgrade | 1 Hz at 40 km/h means 11 m between fixes (4.2). |
| Distress location | Project the detection onto the road plane with the camera calibration, then offset from the phone's position | The phone is 0.5 to 16 m away from what it sees (4.3). |
| Detector | Fine-tuned RF-DETR (Apache-2.0) by default, or Ultralytics YOLO26 if you buy its Enterprise licence. Bootstrap from RDD2022, then train on your own footage. | Licence and accuracy (4.7). |
| De-duplication | Associate detections across frames in road-plane coordinates, compensating for vehicle motion. Result: one record and one VLM call per physical distress. | Each distress appears in about 3 to 12 frames (4.3). |
| Crop | Best frame per distress; box plus per-class context margin; minimum size; cut from the full-resolution frame; JPEG q95 | Matches the VLM's input and avoids crops that are too small or cut off. |
| Geotag | EXIF GPS = camera position; EXIF GPSDest = distress position; full metadata in XMP + sidecar JSON + database | Standard-compliant, and readable by QGIS and photo tools. |
| VLM hand-off | The VLM runs on its owners' machine. Pilot: offline folder batch on a USB drive. Option: an OpenAI-compatible API on an isolated LAN ([pipeline/09](pipeline/09-vlm-interface.md)) | Offline by design; the same records either way. |
| Transfer | USB cable: ADB pull, checksum and signature verification, then a receipt pushed back to the phone. The app has no INTERNET permission | Offline pilot; MTP is unreliable for multi-GB files. |
| Calibration | The app records lens and mount calibration sessions; the PC computes them ([pipeline/04](pipeline/04-calibration-and-geometry.md)) | OpenCV on the PC is easier to verify and re-run than code on the phone. |
| Storage | SQLite + Parquet + GeoPackage on the single offline pipeline PC; PostgreSQL + PostGIS later if needed | Nothing to administer; GIS-native exports. |

## 3. The ten mistakes this plan is designed to avoid

1. **Treating per-frame GPS as measured.** Phones typically deliver about one fix per second. Per-frame coordinates must be interpolated, so log every raw fix and interpolate on the PC (4.2, 6.4).
2. **Mixing clocks.** Camera frames, encoder output, GNSS fixes and IMU samples use different time bases by default. If they aren't unified, every coordinate is wrong by speed × offset (4.1).
3. **Using the phone's position as the distress position.** With a forward windshield mount, the road the camera sees runs from 1.3 m to 16 m ahead (4.3).
4. **Letting auto-exposure choose the shutter speed.** At 40 km/h, 1/1000 s smears the road by 11 mm, which wipes out cracks a few millimetres wide (4.3, 4.4).
5. **Classifying every frame.** The same pothole in 8 frames becomes 8 records, 8 VLM calls and inflated statistics (6.7).
6. **Assuming the VLM works on video crops.** If it was trained on handheld close-ups, crops from moving-vehicle video (oblique view, blur, compression) are out of distribution. Measure this in week 1 (Phase 0).
7. **Assuming per-image labels give an IRC rating.** IRC:82 rates sections by percentage of area affected plus rut depth in mm. That needs area measurement and bookkeeping of the surveyed area, and rut depth can't be measured with one camera (4.8).
8. **Splitting train/test data by frame.** Adjacent frames are near-duplicates, so a random split leaks test data into training and inflates accuracy. Split by road and session.
9. **Ignoring licences until late.** Ultralytics is AGPL-3.0 or paid Enterprise; RDD2022 is CC BY-SA 4.0; SAM 3 has its own SAM License (4.7).
10. **Ignoring heat and vibration.** Hours of 4K recording in Indian summer sun will throttle or shut down a phone, and rigid mounts can damage phone camera stabilisers (4.6).

The full checklist, about 80 mistakes each mapped to the part of the design that prevents it, is [pipeline/12-pitfalls.md](pipeline/12-pitfalls.md).

---

## 4. Research findings

### 4.1 Android timekeeping: what the documentation guarantees

| Data | Field | Time base | Source |
|---|---|---|---|
| Camera frame | `CaptureResult.SENSOR_TIMESTAMP` | Start of exposure of the first sensor row. Same base as `SystemClock.elapsedRealtimeNanos()` **only if** `SENSOR_INFO_TIMESTAMP_SOURCE == REALTIME`. If `UNKNOWN`, it is monotonic but can't be accurately compared with other subsystems. | [CaptureResult](https://developer.android.com/reference/android/hardware/camera2/CaptureResult), [CameraMetadata](https://developer.android.com/reference/android/hardware/camera2/CameraMetadata) |
| Rolling shutter | `SENSOR_ROLLING_SHUTTER_SKEW` | Time from the first row's exposure start to the last row's. For phone sensors this is roughly the readout time. | CaptureResult |
| Encoder input | Frames sent to MediaCodec/MediaRecorder | With `TIMESTAMP_BASE_DEFAULT`, encoder surfaces get **MONOTONIC** time (≈ `uptimeMillis`), not REALTIME. API 33+ can call `OutputConfiguration.setTimestampBase(TIMESTAMP_BASE_SENSOR)` so encoder timestamps match `CaptureResult` timestamps. | [OutputConfiguration](https://developer.android.com/reference/android/hardware/camera2/params/OutputConfiguration) |
| GNSS fix | `Location.getElapsedRealtimeNanos()` (API 17) plus `getElapsedRealtimeUncertaintyNanos()` (API 29) | elapsedRealtime. `getTime()` from `GPS_PROVIDER` comes from the satellite constellation's clock, so it gives UTC independent of the phone's wall clock. `getAccuracy()` is a 68% radius. `isMock()` (API 31) flags fake locations. | [Location](https://developer.android.com/reference/android/location/Location) |
| IMU | `SensorEvent.timestamp` | Same base as `elapsedRealtimeNanos()` | [SensorEvent](https://developer.android.com/reference/android/hardware/SensorEvent) |

**Consequences.**
- Use `elapsedRealtimeNanos` as the single session clock. Require (or strongly prefer) phones whose camera reports `REALTIME`. The Skoltech OpenCamera Sensors project hit the same limitation: synchronised camera and IMU timestamps aren't available on every Camera2 device ([OpenCamera Sensors](https://github.com/MobileRoboticsSkoltech/OpenCamera-Sensors)).
- Don't assume the timestamps inside the MP4 are on the GNSS clock. Log, for every encoded frame, its presentation timestamp (PTS) *and* the matching `SENSOR_TIMESTAMP`, then join them (5.5).
- `getElapsedRealtimeNanos()` is meant to be the fix time, but phone chipsets may stamp it on delivery, adding tens to hundreds of milliseconds of lag. At 40 km/h, 100 ms is 1.1 m of along-track error, so this latency must be measured (8.1).

### 4.2 GNSS on phones

- **Rate.** Expect about 1 Hz on most phones; requesting a shorter interval usually doesn't help ([developer reports](https://b4x.com/android/forum/threads/high-resolution-gps-android-devices.144277)). At 40 km/h that is 11.1 m between fixes, so per-frame positions must be interpolated. GNSS speed and bearing are usually derived from Doppler measurements and can be more accurate than differencing positions (`Location.getSpeed()` docs), which makes a Hermite spline through positions plus velocities the right default interpolator.
- **Accuracy.** Phone GNSS typically gives metre-level positions: a few metres in open sky, worse under trees, beside buildings and near flyovers. Dual-frequency (L1+L5) helps mainly with multipath, but results vary by chipset ([Sensors 2022 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9788430/); [TU Wien 2022](https://repositum.tuwien.at/bitstream/20.500.12708/139226/3/Slateff%20Andreas%20-%202022%20-%20Data%20Quality%20and%20Outdoor%20Positioning%20Accuracy%20of%20Recent...pdf)). Plan on about 3 to 5 m (68%) for distress points, and measure it (8.3).
- **Upgrade path.** Survey of India runs a national CORS network offering RTK/NRTK corrections (about ±3 cm, RTCM 2.4/3.2) after registration ([SoI CORS](https://surveyofindia.gov.in/pages/continuously-operating-reference-stations-cors-), [note](https://Surveyofindia.gov.in/webroot/UserFiles/files/Note%20on%20RTK%20and%20CORS.pdf)). Two ways to use it: (a) an external multi-band RTK receiver read directly by our app; (b) log the phone's raw GNSS measurements and post-process against CORS data (PPK). Both are v2. The capture format reserves space for them now.
- **NavIC.** Several Qualcomm and MediaTek chipsets support India's NavIC system ([The Wire](https://science.thewire.in/aerospace/isro-navic-smartphone-chipsets/)). Treat it as a bonus during device qualification, not a requirement.
- **Integrity.** Log `isMock()` and reject mock locations. This matters if survey data is ever used to check contractor work.

### 4.3 Camera geometry, resolution and blur (computed)

All figures below come from [tools/mount_geometry.py](../tools/mount_geometry.py): 3840×2160, horizontal field of view 67° (a typical 26 mm-equivalent main camera in 16:9 with stabilisation off), flat road, 30 fps, 40 km/h. Re-run the tool with your phone's measured field of view before you finalise the mount.

| Mount | Height / pitch | Road seen (centre column) | Swath width bottom → centre → top | Resolution at image centre (across × along) | Image shift per frame | Frames a road point stays in lower 60% of image | Centre blur at 1/1000, 1/2000, 1/4000, 1/8000 s |
|---|---|---|---|---|---|---|---|
| A: windshield, forward | 1.3 m / 25° | 1.3 → 16.2 m (hood hides the nearest rows) | 2.3 → 4.1 → 20.2 m | 1.06 × 2.51 mm/px | ~166 px | ~5.8 | 4.4 / 2.2 / 1.1 / 0.6 px |
| B: exterior bracket | 1.6 m / 45° | 0.73 → 3.50 m | 2.2 → 3.0 → 4.8 m | 0.78 × 1.10 mm/px | ~380 px | ~3.0 | 10.1 / 5.0 / 2.5 / 1.3 px |
| C: roof/rear bracket | 2.0 m / 55° | 0.52 → 2.90 m | 2.6 → 3.2 → 4.4 m | 0.84 × 1.03 mm/px | ~395 px | ~3.0 | 10.8 / 5.4 / 2.7 / 1.4 px |

On the ground, the vehicle moves this far during the exposure, whatever the mount:

| Speed | 1/1000 s | 1/2000 s | 1/4000 s | 1/8000 s |
|---|---|---|---|---|
| 20 km/h | 5.6 mm | 2.8 mm | 1.4 mm | 0.7 mm |
| 40 km/h | 11.1 mm | 5.6 mm | 2.8 mm | 1.4 mm |
| 60 km/h | 16.7 mm | 8.3 mm | 4.2 mm | 2.1 mm |

**What this means.**
- **Resolution is not the bottleneck; blur is.** At 4K, B and C resolve about 1 mm per pixel, but an auto-exposed 1/500 to 1/1000 s frame smears 11 to 22 mm at 40 km/h. Blur runs along the direction of travel, so it hurts **transverse** cracks most; longitudinal cracks are smeared along their own length and survive better.
- **Target:** at most about 2.5 to 3 mm of smear. That means 1/4000 s at 40 km/h, 1/2000 s at 20 km/h and 1/6000 s or faster at 60 km/h. The app should set exposure time from live GNSS speed (5.3).
- **A distress is visible in only about 3 to 12 frames, and jumps 166 to 395 px between frames.** Simple box-overlap trackers fail at that displacement, so association must predict where each distress will move using the vehicle's motion (6.7).
- **Hairline cracks will not be measurable.** Measuring a crack's width needs about 1 mm/px resolution with blur under 1 px. Expect reliable detection and type classification of visible cracks, but only approximate width classes.
- Rolling shutter: during a ~20 ms sensor readout at 40 km/h the vehicle moves about 22 cm. The PC timestamps each detection at its own image row (6.3), which corrects this exactly.

### 4.4 Can a phone reach these shutter speeds in daylight?

Rough exposure arithmetic for an f/1.8 phone lens, metering on asphalt (which is darker than average): at 1/4000 s, expect roughly ISO 100 to 250 in sun, ISO 300 to 650 under overcast sky, and ISO 600 to 1300 in heavy shade. Under trees, 1/8000 s gets noisy. This is feasible in daylight, but the auto-exposure logic will never choose it on its own.

How to force the shutter speed:
- **Android 16 (API 36)** adds `CONTROL_AE_PRIORITY_MODE = SENSOR_EXPOSURE_TIME_PRIORITY`: the app fixes the exposure time and auto-exposure adjusts ISO. Support is optional per device ([CaptureRequest](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest)).
- **Otherwise:** turn auto-exposure off (needs the MANUAL_SENSOR capability), set `SENSOR_EXPOSURE_TIME` and `SENSOR_FRAME_DURATION` yourself, and run our own ISO loop driven by the brightness of a low-resolution analysis stream. **This is what the app does on the S21**, which will not receive Android 16.
- **Also from the docs:** video stabilisation (EIS) can change the crop region frame by frame, and enabling EIS together with OIS is discouraged. Turn EIS off so the camera geometry stays fixed.

### 4.5 Recording: why not CameraX or MediaRecorder

- CameraX's `Recorder` has no way to switch output files without a gap. Stopping and restarting a recording loses frames, roughly a few metres of road at speed. It also doesn't expose each frame's PTS mapping.
- `MediaRecorder.setNextOutputFile()` (API 26) switches to a new file when a size limit is reached, without stopping the recording, but it hides per-frame timing.
- **Chosen design:** Camera2 → MediaCodec (encoder input surface) → `androidx.media3.muxer.FragmentedMp4Muxer` ([docs](https://developer.android.com/reference/androidx/media3/muxer/FragmentedMp4Muxer)). Segments rotate on keyframes every 5 minutes. A fragmented MP4 stays readable up to its last completed fragment even if the app crashes. We also see every encoded frame's PTS, which is what lets the PC join frames to camera metadata.

### 4.6 Heat, power and vibration

- Android exposes `PowerManager.getCurrentThermalStatus()` and status listeners (API 29), and `getThermalHeadroom()` (API 30). Headroom runs from 0.0 (no throttling) to 1.0 (SEVERE); a status stuck at NONE may simply mean the device doesn't support the API ([ADPF Thermal](https://developer.android.com/games/optimize/adpf/thermal)). The app should degrade gracefully as the phone heats up, and physical shading and cooling still come first (5.11).
- Apple warns that high-amplitude vibration, especially from rigid mounts on high-power engines, can degrade phone OIS and autofocus hardware ([PetaPixel summary of Apple's note](https://petapixel.com/2021/09/11/motorcycle-vibrations-can-damage-your-iphone-camera-apple-warns)). The same physics applies to Android phones, so use vibration-damped mounts.

### 4.7 Datasets, models and licences

| Item | What it is | Licence / status | Use in this project |
|---|---|---|---|
| RDD2020 | 26,336 smartphone images (India, Japan, Czech Republic), >31k instances; classes D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole ([Mendeley](https://data.mendeley.com/datasets/5ty2wb6gvg)) | CC BY-SA 4.0 per the RDD repo; confirm on the Mendeley record | Pre-training; India subset |
| RDD2022 | 47,420 images from 6 countries incl. India, >55k instances, same 4 classes ([paper](https://arxiv.org/abs/2209.08538), [repo](https://github.com/sekilab/RoadDamageDetector)) | CC BY-SA 4.0 (images; faces and plates blurred) | Pre-training. Domain gap: dashboard view, not tilted down |
| Attain (2025) | 2,293 images, 19,761 instances, 10 classes with low/medium/high severity; phones on front and rear windshields at 20 to 70 km/h, frames taken from video ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12167439/)) | See Mendeley record | Extra classes (patching, raveling, block crack, manholes) |
| Ultralytics YOLO26 | Released Jan 2026; detect/segment/OBB/pose; optional NMS-free head; small-object label assignment ([docs](https://docs.ultralytics.com/models/yolo26/)) | AGPL-3.0, or paid Enterprise for closed or internal business use ([licence](https://www.ultralytics.com/license)) | Candidate detector; best tooling |
| RF-DETR | DETR-family real-time detector + segmentation ([repo](https://github.com/roboflow/rf-detr)) | Apache-2.0 for N/S/M/L detection and all segmentation sizes; XL/2XL detection under PML 1.0 | **Default detector** (licence-clean) |
| SAM 3 | Meta, Nov 2025; segments all instances of a text-named concept in images and video ([repo](https://github.com/facebookresearch/sam3)) | "SAM License"; gated checkpoints | Label assistance only (pseudo-masks), pending a licence review |

Notes:
- No public dataset follows IRC:82 classes or our tilted-down viewpoint. **Your own labelled footage is the critical asset**; public sets only speed up the start.
- CC BY-SA "share-alike" terms on training data, and whether they reach the trained model weights, is a legal question. Decide it before shipping a commercial product.

### 4.8 What IRC actually rates, and what a camera can measure

IRC:82-2015 rates bituminous highways from seven distress quantities measured per section. The table below is as reproduced in [Sinha & Hagawane, IRJET 2020](https://www.irjet.net/archives/V7/i5/IRJET-V7I5480.pdf). The IRC catalogue now lists IRC:82-2023, a second revision ([IRC catalogue](https://www.irc.nic.in/WriteReadData/LINKS/Updated%20Catalogue,%202026(1)ed1ad28e-67a2-4bfe-82c1-4c3b75fa9107.pdf)), so verify these values against the edition your VLM follows.

| Distress (unit) | Poor (rating ≤ 1.0) | Fair (1.1–2.0) | Good (2.1–3.0) | Weight | Measurable from our video? |
|---|---|---|---|---|---|
| Cracking (% area) | >10 | 5–10 | <5 | 1.00 | Yes |
| Ravelling (% area) | >10 | 1–10 | <1 | 0.75 | Yes (texture, lighting dependent) |
| Potholes (% area) | >1 | 0.1–1 | <0.1 | 0.50 | Yes (area; depth only roughly) |
| Shoving (% area) | >1 | 0.1–1 | <0.1 | 1.00 | Weakly (3D deformation) |
| Patching (% area) | >10 | 1–10 | <1 | 0.75 | Yes |
| Settlement & depression (% area) | >5 | 1–5 | <1 | 0.75 | Weakly (needs 3D) |
| Rut depth (mm) | >10 | 5–10 | <5 | 1.00 | **No** (needs a laser or profiler) |

**Consequences.** If section ratings are a goal: (1) the detector must produce a **ground area** for each distress, which needs segmentation masks for area-type classes, projected onto the road plane; (2) the pipeline must record the **surveyed pavement area** for every section (swath × length); (3) rut depth must come from another sensor or a manual survey. For context, NHAI makes Network Survey Vehicle (NSV) surveys with laser profilers mandatory at project completion and every six months after ([India Infra Hub](https://indiainfrahub.com/featured/nhai-mandates-compulsory-deployment-of-network-survey-vehicles-to-improve-road-quality/)). This system is a low-cost visual complement to an NSV, not a replacement for one.

### 4.9 Running the VLM on the PC

- vLLM has no official native Windows support; the supported routes are WSL2 or Docker Desktop (which added vLLM via WSL2 in Dec 2025, [Docker blog](https://www.docker.com/blog/docker-model-runner-vllm-windows)). A Linux workstation avoids the issue entirely.
- vLLM structured output: `response_format: {"type": "json_schema", ...}` or `structured_outputs`. The old `guided_json` parameters were removed in v0.12 ([docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)).
- GPU video decoding is optional. TorchCodec decodes on CUDA ([repo](https://github.com/meta-pytorch/torchcodec)), but its Windows CUDA support is experimental. CPU decoding with FFmpeg/PyAV is fast enough for the MVP (7.4).

### 4.10 Existing tools worth reusing for the pilot

- **OpenCamera Sensors** (GPL-3.0): records video, IMU and per-frame timestamps on one clock. *v2 note: no longer needed, because the GeoAI Capture app now covers this; kept as a cross-check.* Use it for Phase-0 pilot capture together with Google's **GnssLogger**. Before relying on it, check that your GnssLogger version writes elapsed-realtime timestamps for fixes. Don't fork it into the product, because GPL would then apply.
- **GoPro GPMF** telemetry (GPS in the same file as the video) is a useful secondary input. Support varies by model: the HERO13 has GPS, and several models dropped it ([TrekView](https://www.trekview.org/blog/evolution-of-gopro-camera-sensors-gpmf/)). Make the PC ingest step source-agnostic so a GoPro or dashcam adapter can be added later.

---

## 5. Layer 1: capture app design

### 5.1 Technology

- Kotlin, single activity + one foreground `SurveyService` that owns the camera, encoder, muxer, GNSS, IMU and file writers. The UI only binds to the service to show status.
- `minSdk 31` (Android 12): gives `isMock()`, `LocationRequest.Builder` and GNSS full tracking. Use API 33/34/36 features when the device has them. Target whatever SDK level Google Play currently requires, if you distribute through Play.
- No on-device AI in v1. It adds heat and battery drain and isn't needed.
- **Built:** [../android/](../android/README.md) (Kotlin, Jetpack Compose UI, compileSdk 36, targetSdk 35, minSdk 31).

### 5.2 Device qualification (built into the app, run before buying a fleet)

A "Device check" screen records and exports:

| Check | Pass condition |
|---|---|
| Camera2 hardware level | FULL or LEVEL_3 |
| Capabilities | MANUAL_SENSOR (manual exposure and focus) |
| `SENSOR_INFO_TIMESTAMP_SOURCE` | REALTIME (fail = expect only rough sync) |
| `SENSOR_INFO_EXPOSURE_TIME_RANGE` | Minimum ≤ 1/8000 s |
| Encoder | HEVC (or AVC) 3840×2160 at 30 fps at the target bitrate (`MediaCodecInfo.VideoCapabilities`) |
| Stabilisation | EIS can be turned off; OIS present or absent noted |
| AE priority mode | API 36 `CONTROL_AE_PRIORITY_MODE` available (nice to have) |
| GNSS | Fix rate (measured), constellations incl. NavIC, L5 signals seen (`GnssStatus` carrier frequency), raw measurements supported |
| Thermal API | Thermal status and headroom actually report values |
| 60-min soak test | 4K recording in sun, phone mounted: time to first thermal step-down, dropped-frame count |

Shortlist 2 or 3 current mid-to-high-end phones with dual-frequency GNSS, run this check plus a road test, then standardise on one model. Freeze its OS version during the project, because OS updates can change camera behaviour. Re-run the check after any update.

**For the pilot the phone is already chosen (Galaxy S21 5G).** Run the Device check on it first anyway: it documents what this particular unit supports, and the report goes with the pilot data. S21 variants differ by region (Exynos 2100 or Snapdragon 888 chipsets), so record the exact model number. The first-day checklist is in [../android/README.md](../android/README.md).

### 5.3 Camera configuration

| Setting | Value |
|---|---|
| Lens | Main (wide) back camera, zoom fixed at 1.0. The app logs focal length and zoom for every frame, so the PC detects any lens switch ([pipeline/01](pipeline/01-ingest.md) §3). |
| Orientation | Phone in landscape; long image side across the road; no rotation flag in the file |
| Streams | (1) encoder surface 3840×2160; (2) small preview, optional, screen may be off; (3) low-res YUV analysis at ~2 to 5 fps for brightness, sharpness and lens-dirt checks |
| Frame rate | Fixed 30 fps (`CONTROL_AE_TARGET_FPS_RANGE [30,30]`, or `SENSOR_FRAME_DURATION` 33.3 ms in manual mode) |
| Exposure | Shutter priority: `t = clamp(blur_budget / speed, device minimum, 1/1000 s)`, with `blur_budget` = 2.5 mm and speed from GNSS (50 km/h assumed without a fix). Our own ISO loop (AE off) holds the road brightness up to ISO 1600. Beyond that the shutter may lengthen to 1/500 s, and those frames are flagged `low_light`. The exposure and ISO actually used are logged for every frame. |
| Focus | Locked before recording: an autofocus trigger on the road, then held (`LOCKED_MANUAL` or `LOCKED_AF`). Never lets focus hunt; logged per frame. |
| Stabilisation | EIS OFF (`CONTROL_VIDEO_STABILIZATION_MODE_OFF`). OIS off by default (a setting; it doesn't help at 1/4000 s); state logged per frame. |
| HDR / scene modes / beauty | Off. 8-bit SDR only. |
| White balance | Auto, logged per frame. Optionally lock after warm-up. Keep identical settings for training data and production data. |
| Audio | None (privacy, file size) |

### 5.4 Encoding and files

- MediaCodec HEVC Main, 8-bit (AVC as fallback), 60 Mbps by default (a setting; check the encoder's supported range), BT.709 limited-range colour tags. Keyframe every 1 s. `KEY_MAX_B_FRAMES = 0` (API 29), so presentation order equals encode order.
- Muxer: Media3 `FragmentedMp4Muxer` with 2 s fragments. Segments rotate every 5 min on a keyframe: the app requests a sync frame (`PARAMETER_KEY_REQUEST_SYNC_FRAME`) and starts the new file on it, so no frame is lost between segments.
- Data volume: about 27 GB per hour at 60 Mbps, roughly 1 GB per km at survey speeds. The S21 has no microSD slot: a 128 GB unit holds about 3 hours and a 256 GB unit about 7, depending on free space. The app refuses to start below 5 GB free and stops cleanly at 2 GB.

### 5.5 Joining frames to camera metadata (the key algorithm)

1. On API 33+, configure the encoder `OutputConfiguration` with `TIMESTAMP_BASE_SENSOR`, so encoder timestamps equal `CaptureResult.SENSOR_TIMESTAMP`.
2. Keep a ring buffer of recent `CaptureResult`s keyed by `SENSOR_TIMESTAMP`.
3. For each encoded output buffer, take `pts_ns = presentationTimeUs × 1000` and find the result with `|pts_ns − Δ − sensor_ts| < 2 ms`. Here Δ = 0 if step 1 applied. Otherwise the app tries the expected offset first: on a REALTIME-camera phone with the default MONOTONIC encoder clock, Δ = `System.nanoTime() − elapsedRealtimeNanos()`, a negative number whose size is the phone's total sleep time since boot, constant while the device is awake. If that doesn't match, it falls back to the most common difference (a 1 ms histogram).
4. Write one `frames.csv` row per encoded frame: segment, index, PTS, sensor timestamp, exposure, ISO, frame duration, rolling-shutter skew, focus, stabilisation states, keyframe flag, and the match residual.
5. Write `clock_sync.csv` every 10 s: (`elapsedRealtimeNanos`, `System.nanoTime`, `currentTimeMillis`) sampled back-to-back. Also log every `onCaptureFailed` and buffer-lost event.

### 5.6 GNSS logging

- `LocationManager` with `GPS_PROVIDER`, not the fused provider (which blends in Wi-Fi and cell positions). Request the fastest interval (0) with high accuracy.
- Per fix: elapsed time + uncertainty, `getTime()` UTC, latitude, longitude, ellipsoidal altitude (+ MSL altitude on API 34+), horizontal/vertical/speed/bearing accuracies, speed, bearing, mock flag.
- Also log the `GnssStatus` callback (satellites, C/N0, carrier frequency, used-in-fix) and raw NMEA (`OnNmeaMessageListener`) with receipt time.
- Optional, off by default: raw `GnssMeasurementsEvent` with `GnssMeasurementRequest.setFullTracking(true)` (API 31) for later PPK. It costs battery.
- Gate on fix quality: a survey session can't start recording until a fix with horizontal accuracy ≤ 10 m arrives that is no older than 2 s. "Record anyway" starts without it and logs `recording_forced_without_gnss`. (10 m rather than 5 m: under tree cover, a stricter gate stops surveys that the PC can still use, with every affected record flagged.)
- While recording, a watchdog restarts the GNSS request after 15 s without a fix while location is on, and warns the operator if location is switched off.

### 5.7 IMU logging

Log the accelerometer and gyroscope at 200 Hz, and gravity, the game rotation vector and the magnetometer at 50 Hz. Android 12+ caps apps at 200 Hz unless they declare `HIGH_SAMPLING_RATE_SENSORS` (the app declares it). Uses:
- Estimating GNSS latency by cross-correlation (8.1).
- Heading at low speed (gyro) and detecting reversing ([pipeline/02](pipeline/02-time-and-trajectory.md) §5).
- Flagging bumps, suspension pitch and mount shifts (QC). **Not** for correcting the projection: on a grade, raw gravity tilts by the grade angle and would add error ([pipeline/02](pipeline/02-time-and-trajectory.md) §7).
- Smoothing the trajectory and bridging GNSS gaps (v2); a roughness proxy later (not an IRC:SP:16 IRI value).

### 5.8 Calibration

The app **records** calibration sessions; the PC **computes** the calibrations ([pipeline/04](pipeline/04-calibration-and-geometry.md)).
- **Intrinsics** (focal length, principal point, distortion): a `calibration_intrinsics` session per phone unit and resolution, in the exact recording mode (zoom 1.0, EIS off, focus locked as for surveying), moving a ChArUco board through the whole frame.
- **Mount extrinsics** (height, pitch, roll, yaw relative to the vehicle): a 10 s `calibration_mount` session, parked on flat ground, with six or more ArUco markers at measured positions relative to the vehicle frame. The PC solves the camera pose (solvePnP), checks it against held-out markers, and stores the road-plane homography and the ROI that masks the vehicle body and the unusable far field. The gravity vector at the same moment becomes the reference.
- **Daily:** a mount session at the start **and the end** of each survey day. The PC checks each survey's gravity direction against the calibration (1.5°) and looks for sudden mount shifts ([pipeline/01](pipeline/01-ingest.md) §3, §5).
- For rear-facing mounts, the calibration gives yaw ≈ 180°, so "ahead in the image" isn't confused with "ahead of the vehicle".

### 5.9 Mounting

| | Exterior bracket (B/C), recommended | Windshield (A), fallback or pilot |
|---|---|---|
| Image quality | Even, ~1 mm/px resolution, near top-down view, no glass | Glass reflections, dirt and tint; hood hides the near field; coarse far field |
| Lane coverage | 3.0 to 4.8 m swath (one 3.5 m lane) | Wide but low-resolution beyond ~5 m |
| GNSS | Clear sky view | Windshield coatings can weaken GNSS signal |
| Heat | Direct sun; needs a shade and airflow | Greenhouse effect behind glass; worst for heat |
| Risks | Weather, dust, vibration, theft, secure fixing | Easiest to fit |

Bracket requirements: aluminium arm on a roof rack or tow mast at 1.6 to 2.0 m, pitched 45° to 55°; rubber vibration isolators; safety tether; ventilated housing with sun shade and lens hood; nothing metal over the phone's GNSS antenna; USB-C PD power with strain relief. Check local rules on external fittings.

### 5.10 Operator workflow

- **Before the run:** checklist (lens clean, mount check passed, GNSS accuracy OK, storage ≥ 2× expected, temperature OK) and a survey form (road ID/name, direction, lane, start chainage, operator, vehicle, weather).
- **While driving:** large Record/Stop. The status panel shows GNSS accuracy, satellites and fix age; speed (amber over 40 km/h, red over 50); shutter, ISO and low light; focus; thermal status and headroom; battery temperature; storage left; frames and drops. A Marker button stamps km-stone passages (with the chainage), notable distresses and other events. Stationary periods are detected on the PC; recording continues.
- **After the run:** the app finalises the session (manifest and signature). Transfer it over USB the same day. The PC verifies it and writes a receipt; only then does the app allow deletion.

### 5.11 Background execution, power and heat

- Foreground service types `camera|location` (+ `connectedDevice` if an external GNSS receiver is added later), with `FOREGROUND_SERVICE_CAMERA` and `FOREGROUND_SERVICE_LOCATION`. These services can only start while the app is in the foreground ([FGS types](https://developer.android.com/develop/background-work/services/fgs/service-types)). Hold a partial wake lock and request a battery-optimisation exemption.
- Thermal ladder: MODERATE → warn the operator; SEVERE → step down to 1080p or a lower bitrate (logged as an event); CRITICAL → finalise files and stop cleanly.
- Physical measures matter more than software: shade, airflow, a clip-on phone cooler, a white housing, no case.

### 5.12 Integrity and security

- SHA-256 for every file in `manifest.json`, signed with a per-device Android Keystore key (ECDSA P-256). No Play Integrity token: it needs the internet.
- Record the app version/commit, settings hash, device model and build fingerprint, and calibration ID.
- Mock-location fixes are flagged, and the session is marked "untrusted".

### 5.13 Storage and transfer

- Sessions are written to `/sdcard/GeoAI/sessions` (with "All files access", so they survive an app uninstall and ADB can reach them), with an app-specific folder as the fallback.
- **Transfer is by USB cable only** ([pipeline/01](pipeline/01-ingest.md) §1): [`tools/pull_sessions.py`](../tools/pull_sessions.py) pulls with ADB, verifies checksums and the signature, renames atomically and pushes a `pc_receipt.json` back to the phone. MTP is avoided: it is unreliable for multi-GB files on Windows.
- Delete from the phone only after the receipt matches. The app marks such sessions VERIFIED_ON_PC. Deleting any other session needs a separate "Delete UNVERIFIED session?" confirmation that warns the data will be lost for good.

---

## 6. Layer 2: PC pipeline design

This section summarises the design. The build specification is [pipeline/](pipeline/README.md), documents 01–12; where they differ, the pipeline documents win.

### 6.1 Runtime

- Python 3.11 or 3.12. Libraries, directory layout and CLI are in [pipeline/README.md](pipeline/README.md). Everything installs from a local wheelhouse and runs offline.
- Every stage is idempotent and keyed by (input hash, stage version, config hash), so a new detector or VLM version re-runs only the stages downstream of it.

### 6.2 Ingest and QC report (per session)

Verify checksums and signature. Check that the MP4 frame count equals the `frames.csv` row count per segment, that PTS are monotonic, and that segments join without gaps. Count dropped frames (gaps over 1.5× the median frame duration). Report GNSS fix-rate gaps and accuracy distribution, mock flags, thermal events, and clock-offset stability. The output is an HTML/JSON QC report. Red flags block automatic processing.

### 6.3 Time model

- Session clock = `elapsedRealtimeNanos`.
- UTC = robust linear fit of (`elapsed_ns`, `getTime()`) over all GPS fixes. Its residuals measure clock quality.
- Each detection's time is computed at the detection's own image row: `t = sensor_ts + exposure/2 + skew × (row / image_height)`. Use the row in the sensor's native orientation. Frames are stored unrotated, so if the PC rotates an upside-down mount for display, it must map rows back before computing this.
- Correct GNSS latency (8.1) before interpolating positions.
- Store UTC; display IST (UTC+05:30).

### 6.4 Trajectory

- **MVP:** convert fixes to a local east-north-up frame. Fit a piecewise cubic Hermite spline through positions with Doppler velocities (speed × bearing) as derivatives. Heading comes from velocity; below 1.5 m/s it is propagated with the gyroscope. Reversing is detected and excluded, because the velocity heading then points backwards ([pipeline/02](pipeline/02-time-and-trajectory.md) §5). Use accuracy-weighted outlier rejection (drop jumps that are physically implausible).
- **v2:** a Kalman filter with backward smoothing over GNSS + IMU (better through tunnels, flyovers and urban canyons); optional PPK or RTK input; map-matching to a road centreline for chainage.
- **Output** per frame: position, heading, speed, pitch/roll offsets from the IMU, and a 1-sigma horizontal uncertainty.

### 6.5 Decoding and frame quality

- Decode with PyAV. Join each decoded sample to its `frames.csv` row by **sample index**, using the container PTS only as a lookup key. Fragmented-MP4 PTS drift slightly from the true capture times, so they are never used as times ([pipeline/03](pipeline/03-video-decoding.md) §1). Never use frame index ÷ fps.
- Per-frame quality: sharpness (variance of the Laplacian inside the ROI), exposure clipping, speed, lens-obstruction flag (blobs that stay fixed while the road moves). Skip frames while stationary.

### 6.6 Detection

- **Model:** RF-DETR (detection first, segmentation in Phase 3) or YOLO26, chosen on licence plus a benchmark on our own validation set.
- **Resolution:** don't shrink 4K to 640 px; a 3 mm crack would become under one pixel wide. Run on tiles at native resolution (about 1280 px tiles, ROI only), or on 2× downscaled frames for large-area classes plus native tiles for cracks.
- **Detector classes** (coarse, so the VLM does the IRC-level work): `crack_linear`, `crack_area` (alligator/map/block), `pothole`, `patch`, `surface_loss` (ravelling/stripping/aggregate loss), `bleeding`, `edge_failure`, `deformation_visible`, plus look-alikes as explicit classes, dropped later: `manhole_utility`, `speed_breaker`, `marking`. Shadows, oil stains and wet patches go in as unlabelled hard negatives.
- **Operating point:** high recall (low confidence threshold). The VLM's "no distress" answer handles precision.
- **Training data:** RDD2022 (which already contains RDD2020) + Attain for the first model; then pseudo-label our footage, have engineers correct it in CVAT or Label Studio (SAM-assisted masks), retrain, and repeat with active learning on low-confidence and disagreement samples. **Split by road section.** Details: [detection-model-plan.md](detection-model-plan.md).

### 6.7 Road-plane projection and association (one record per distress)

1. Project each detection's box or mask onto the road plane with the static mount calibration. This gives a footprint in metres relative to the camera, then in world coordinates via the trajectory pose. It is not adjusted with raw IMU gravity: on a grade, gravity tilts by the grade angle and the adjustment would add error ([pipeline/02](pipeline/02-time-and-trajectory.md) §7).
2. Predict where each active track's footprint should appear in the current frame from the vehicle's motion between the two frame times. Match with the Hungarian algorithm on road-plane overlap and centre distance, and require compatible classes.
3. Close a track after it has been missing for N frames. Distresses longer than one frame can show (longitudinal and edge cracks, long patches) are cut into pieces on a fixed 2 m grid along the path, each fully visible in one frame and each with its own crop ([pipeline/06](pipeline/06-association.md) §5).
4. Cross-check: run Ultralytics BoT-SORT with global motion compensation (`gmc_method: sparseOptFlow`) on the same data as a baseline ([Ultralytics tracking docs](https://docs.ultralytics.com/modes/track/)).
5. After the session, records within 2.5 m of each other that share a family and come from a revisit (U-turn, second pass) are flagged as possible duplicates for review. They are never merged automatically: two real potholes can be 2 m apart.

Quality metric: **duplicate rate** = records per physical distress, target ≤ 1.05 on the test route.

### 6.8 Best frame and cropping

- **Score each sighting** with absolute terms, so a single streaming pass can keep the best so far. The view must be complete (inside the ROI, not cut by the image edge). Then rank by effective ground resolution including motion smear, frame sharpness, detector confidence, exposure and centrality ([pipeline/07](pipeline/07-best-frame-and-crop.md) §2).
- **Crop** from the full-resolution decoded frame:
  ```
  m  = max(alpha[class] * max(w, h), m_min)      # alpha 0.15–0.5 by class; tune to the VLM
  box = bbox expanded by m on every side
  if VLM expects square: grow the short side
  if a side < S_min (448 px by default): grow context to S_min; never enlarge the crop
  shift the box inward at image edges instead of truncating; truncate only if larger than the frame (flag it)
  if the long side > S_max (1344 px): downscale
  ```
  Per-class margin: edge distresses need the pavement edge and shoulder in view, and potholes need their rim.
- **Save** as JPEG q95 with 4:4:4 chroma, plus an optional downscaled context image (full frame with the box drawn) for reviewers. An optional "rectified" (top-down) crop is a v2 feature; it is only useful if the VLM was trained on top-down images.

### 6.9 Geotagging

- **Database (the source of truth):** frame UTC time (ms), camera position + uncertainty, heading, speed, distress centroid and footprint polygon (WGS84), area in m², length in m, image-to-ground resolution, crop box, original box, track ID, all model versions.
- **EXIF:**

  | Tag | Value |
  |---|---|
  | `DateTimeOriginal`, `SubSecTimeOriginal`, `OffsetTimeOriginal` | Local IST, milliseconds, `+05:30` |
  | `GPSDateStamp`, `GPSTimeStamp` | UTC |
  | `GPSLatitude/Ref`, `GPSLongitude/Ref`, `GPSAltitude/Ref`, `GPSMapDatum=WGS-84` | Camera position at the frame. `GPSAltitude` is mean-sea-level altitude, never the ellipsoidal height, which differs by tens of metres in India |
  | `GPSDestLatitude/Ref`, `GPSDestLongitude/Ref`, `GPSDestBearing`, `GPSDestDistance` | **Distress position** relative to the camera |
  | `GPSHPositioningError` | 68% horizontal error (m) |
  | `GPSSpeed/Ref`, `GPSTrack/Ref`, `GPSImgDirection/Ref` | Speed (km/h), course, camera direction |
  | `ImageUniqueID` | Distress record UUID |
  | XMP (custom namespace) | session, segment, frame PTS, track, boxes, resolution, area, detector class/confidence, versions |

- **Never burn text overlays into the pixels.** They hide pavement and confuse the VLM. The only exception: if the VLM was trained on overlaid images, add a separate strip below the crop, not on top of it.

### 6.10 VLM classification

- **Where and how:** the VLM runs on its owners' machine. Pilot transport: an offline folder batch on a USB drive; option: an OpenAI-compatible API on an isolated LAN. Both use the same request and response records ([pipeline/09](pipeline/09-vlm-interface.md)).
- **Prompt and output format, exactly as trained:**
  - a byte-identical, versioned prompt, with the same chat template and image preprocessing as in training;
  - the output format the model was trained to produce. Constrain decoding only to that format: imposing a new one (e.g. JSON on a free-text model) can change its answers;
  - a label set that includes `no_distress`, `not_pavement` and `uncertain`, or mapped equivalents;
  - physical context (crop size in metres, estimated dimensions) **only if** the VLM was trained to use it;
  - temperature 0.
- **Confidence:** don't trust self-reported confidence. Use log-probabilities of the constrained label tokens and calibrate them on validation data.
- **Review routing:** low score, a detector-class vs VLM-label conflict, high severity, or a crop that was cut off → human review queue.
- **Acceptance:** before production, measure per-class precision and recall on **pipeline crops** labelled by IRC-trained engineers. If the gap to the VLM's original benchmark is large, fine-tune it (LoRA) on pipeline crops.

### 6.11 Review UI

- For the pilot, start with QGIS and a review GeoPackage. The MVP is a local web app (FastAPI + MapLibre) showing the map, a queue, the crop, the context frame and the VLM output, with actions accept / relabel / reject / duplicate. A random audit sample is reviewed **blind**, with the VLM's answer hidden ([pipeline/10](pipeline/10-review-and-export.md) §1).
- Every action is stored and goes into the next training set.
- FiftyOne is optional, for dataset curation and error analysis.

### 6.12 Outputs

- **Distress inventory:** GeoPackage (opens directly in QGIS), GeoJSON, KML, CSV.
- **Section report (Phase 3):** per lane per 100 m / 1 km, the % area of each IRC quantity, with ratings and weights exactly as defined in the IRC:82 edition in force. Percentages are of the road area the camera actually observed. The rule for converting linear cracks to area must be set by an engineer before cracking % is computed. Rut depth is marked "not measured". Every rating requires an engineer's sign-off ([pipeline/10](pipeline/10-review-and-export.md) §4).
- **Chainage:** linear referencing against the agency's road centreline (shapely `line.project`), anchored to km stones marked with the Marker button. Without a centreline, path distance from the start chainage is used.

### 6.13 Data model (core tables)

SQLite tables `sessions`, `distresses` (centroid, footprint, crop, provenance), `observations` (distress ↔ frame), `classifications` (model ID, prompt version, label, severity, raw output; append-only), `reviews` (append-only) and `models` ([pipeline/08](pipeline/08-geotag-and-records.md) §4). High-rate data (frames, GNSS, IMU, trajectory, detections) live in Parquet files per session.

---

## 7. Hardware and capacity

### 7.1 Field kit (per vehicle)

Qualified phone (S21 5G for the pilot; 256 GB preferred), exterior bracket with isolators, tether, ventilated housing, sun shade, clip-on cooler, USB-C PD car charger and cable, a data-capable USB-C cable for transfer, a USB drive for VLM batches, ChArUco board, ArUco ground markers with a steel tape, lens cloth. Optional later: a multi-band RTK GNSS receiver + antenna.

### 7.2 PC

| Part | Recommendation |
|---|---|
| GPU | NVIDIA with ≥ 12 GB VRAM for the detector. The VLM runs on its own machine, sized by its owners |
| CPU / RAM | 8 to 16 cores / 32 to 64 GB |
| Storage | 2 to 4 TB NVMe for working data + NAS or HDD archive (~30 GB per survey hour at 60 Mbps) |
| OS | Windows 11 or Ubuntu 24.04 LTS. The pipeline needs neither WSL2 nor vLLM; those matter only on the VLM machine |

### 7.3 Processing budget per hour of driving (estimate; benchmark in Phase 2)

- Decoding 108k frames of 4K HEVC: CPU about 60 to 150 fps, so roughly 12 to 30 min (runs in parallel with detection).
- Detection at native-resolution tiles inside the ROI: of the same order as the decode time on one modern GPU. Halve it by detecting large classes on downscaled frames.
- VLM (on the VLM machine): on the order of 1 to 10 crops per second batched, so even a badly damaged road (hundreds of distresses per km) finishes within the hour. The USB round trip adds handling time.
- Target: processing at or faster than real time.

## 8. Calibration and validation protocol

1. **Time-lag test (camera vs GNSS).** Paint or tape two markers about 200 m apart and survey them with RTK or a long static average. Drive past both directions at 20, 40 and 60 km/h, 3 times each. An along-track offset that flips sign with direction and grows with speed is a timing lag: lag = offset ÷ speed. Surveyed coordinates aren't needed for the lag itself, because the GNSS bias is common to back-to-back passes in opposite directions and cancels ([pipeline/11](pipeline/11-testing-and-acceptance.md) §4). Also estimate lag automatically every session by cross-correlating IMU forward acceleration with the derivative of GNSS speed (braking events give strong signals).
2. **Geometry test.** Measured distances on the ground mat vs the projected ones (target: under 2% scale error inside the ROI).
3. **Position accuracy.** About 30 surveyed marks along a 5 to 10 km test route; report the distribution of distress-position error (68% and 95%).
4. **Model evaluation.** Held-out roads and sessions only. Per-class precision/recall for the detector and the VLM, and the duplicate rate.
5. **End-to-end acceptance (proposed targets, adjust as needed):**

| Metric | Target |
|---|---|
| Camera ↔ GNSS residual lag after correction | ≤ 20 ms (≈ 0.2 m at 40 km/h) |
| Dropped frames | < 0.1%, no gaps between segments |
| Distress position error (phone GNSS, open sky) | ≤ 3 to 5 m (68%); ≤ 0.5 m with RTK upgrade |
| Detection recall: potholes, alligator, patches | ≥ 90% |
| Detection recall: linear cracks wider than ~3 mm | ≥ 80% |
| Duplicate rate | ≤ 1.05 records per distress |
| VLM accuracy on pipeline crops | Within 5 points of its original benchmark, per class |
| Throughput | 1 h of video in ≤ 1 h of processing |

The full test plan, including the simulator, the annotated test route and the daily field protocol, is [pipeline/11-testing-and-acceptance.md](pipeline/11-testing-and-acceptance.md).

## 9. Roadmap

Assumes 1 Android engineer, 1 to 2 Python/ML engineers and part-time pavement-engineer labelling. Durations are estimates.

| Phase | Weeks | Deliverables | Exit criteria |
|---|---|---|---|
| **0: De-risk** | 0–2 | VLM input contract written down ([vlm-information-request.md](vlm-information-request.md)); S21 device check and first-day checklist; pilot capture with the GeoAI Capture app on mounts A and B; **100 hand-cropped pilot distresses run through the VLM and compared with engineer labels**; licence decisions | The VLM's accuracy on video crops is known; mount chosen |
| **1: Capture MVP + ingest** | 2–8 | App: **built** ([../android/](../android/README.md)); remaining: S21 field validation. PC, on the pipeline machine: ingest, QC report, time model, trajectory, decoding ([pipeline/01–03](pipeline/README.md)) | Time-lag test passed; zero gaps between segments; 60-minute heat soak test passed |
| **2: Pipeline MVP** | 6–12 | Detector v1 (public data + first labels), projection, association, best frame, crop, EXIF/XMP, VLM adapter, database, exports, basic review UI | End-to-end run on the 10 km test route; metrics reported |
| **3: Accuracy + reporting** | 12–20 | Labelling loop, detector v2 (with segmentation for area classes), VLM fine-tune if needed, IRC section report, map dashboard, operator manual and training, hardening | Acceptance targets met on held-out roads; pilot deployment |
| **4: Upgrades** | later | RTK/PPK, Kalman smoothing, chainage map-matching, change detection across surveys, roughness proxy, GoPro/dashcam ingest, NHAI/PMS export formats, iOS if needed | Driven by pilot feedback |

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| VLM performs poorly on video crops | High | High | Phase-0 test; fine-tune on pipeline crops; match preprocessing |
| Camera/GNSS timing lag → systematic position bias | Medium | High | REALTIME-timestamp phones; lag test; automatic IMU-GNSS lag estimate |
| Motion blur hides cracks | High at auto-exposure | High | Speed-adaptive shutter; speed limits in the operator procedure; daylight only |
| Phone overheats or shuts down | High in summer | Medium | Shade, cooling, thermal ladder (warn → 1080p → clean stop), crash-safe fMP4 segments |
| S21 camera timestamps not on the session clock (timestamp source `UNKNOWN`) | Unknown until the device check | High | The app measures the encoder–camera offset; such sessions are marked `sync = approximate`; the marker test measures any remaining lag |
| Phone wall clock wrong (offline, no network time) | High | Low | UTC comes only from GNSS time ([pipeline/02](pipeline/02-time-and-trajectory.md) §4) |
| Detector misses IRC types absent from public data | High | Medium | Own labelled data, coarse taxonomy, active learning |
| Duplicate counts | Medium | Medium | Road-plane association; duplicate-rate metric |
| GNSS degraded (trees, flyovers, urban canyons) | Medium | Medium | Accuracy flags, IMU smoothing, map matching, RTK option |
| Licence conflicts (AGPL, CC BY-SA, SAM) | Medium | High | Decide in Phase 0; Apache-2.0 defaults |
| OS update changes camera behaviour | Medium | Medium | Freeze OS version; re-qualify after updates |
| Mount failure or vibration damage | Low | High | Isolators, tether, daily inspection |
| Expecting full IRC ratings from video | Medium | High | Report only what is measurable; rut depth explicitly out of scope |
| Personal data in video (faces, plates) | Low with a downward camera | Medium | No audio; blur in context images if needed; retention policy (DPDP Act, 2023) |

## 11. Questions that change the design (status 22 Oct 2026)

1. **The VLM** (open). Which model and size? What input size and preprocessing? What were its training images (handheld close-ups, dashcam frames, top-down)? Its exact label list and IRC edition (IRC:82-2015 or -2023; does it cover concrete pavements under IRC:SP:83)? Does it output severity? Does it have a "no distress" class? Does it expect metadata in the prompt, in EXIF, or overlaid on the image? The full list is in [vlm-information-request.md](vlm-information-request.md).
2. **Output** (open). Is the goal a distress inventory (points on a map), IRC section ratings, or both? Is chainage or lane referencing required?
3. **Accuracy** (open). Is ±3 to 5 m acceptable for distress positions, or do you need sub-metre (RTK)?
4. **Roads** (open). NH, SH, urban or rural? Typical survey speed? Multi-lane coverage?
5. **Budget** (partly answered). The pilot phone is the S21 5G. Still open: external mount fabrication, and the pipeline PC's GPU model.
6. **Distribution and use** (open). Internal tool or a product? This decides the AGPL question and whether the app ever ships through Google Play.
7. **Field support** (open). Who prints and measures the calibration targets, paints and surveys the test marks, and provides RTK or DGPS for the position test ([pipeline/11](pipeline/11-testing-and-acceptance.md) §5)?

---

## Sources

Android: [CaptureResult](https://developer.android.com/reference/android/hardware/camera2/CaptureResult) · [CameraMetadata](https://developer.android.com/reference/android/hardware/camera2/CameraMetadata) · [CaptureRequest](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest) · [OutputConfiguration](https://developer.android.com/reference/android/hardware/camera2/params/OutputConfiguration) · [Location](https://developer.android.com/reference/android/location/Location) · [SensorEvent](https://developer.android.com/reference/android/hardware/SensorEvent) · [MediaRecorder](https://developer.android.com/reference/android/media/MediaRecorder) · [Media3 FragmentedMp4Muxer](https://developer.android.com/reference/androidx/media3/muxer/FragmentedMp4Muxer) · [Foreground service types](https://developer.android.com/develop/background-work/services/fgs/service-types) · [Thermal API](https://developer.android.com/games/optimize/adpf/thermal) · [Camera: improve timestamping docs (AOSP)](https://android.googlesource.com/platform/frameworks/base/+/1ce3c167c5a3%5E%21/)

GNSS and India: [Sensors 2022: dual-frequency smartphone GNSS](https://pmc.ncbi.nlm.nih.gov/articles/PMC9788430/) · [TU Wien 2022 smartphone GNSS](https://repositum.tuwien.at/bitstream/20.500.12708/139226/3/Slateff%20Andreas%20-%202022%20-%20Data%20Quality%20and%20Outdoor%20Positioning%20Accuracy%20of%20Recent...pdf) · [Android 1 Hz reports](https://b4x.com/android/forum/threads/high-resolution-gps-android-devices.144277) · [Survey of India CORS](https://surveyofindia.gov.in/pages/continuously-operating-reference-stations-cors-) · [SoI note on RTK/CORS](https://Surveyofindia.gov.in/webroot/UserFiles/files/Note%20on%20RTK%20and%20CORS.pdf) · [NavIC chipsets](https://science.thewire.in/aerospace/isro-navic-smartphone-chipsets/) · [NHAI NSV mandate](https://indiainfrahub.com/featured/nhai-mandates-compulsory-deployment-of-network-survey-vehicles-to-improve-road-quality/)

IRC: [IRC:82-2015 rating table and weights, as reproduced in IRJET 2020](https://www.irjet.net/archives/V7/i5/IRJET-V7I5480.pdf) · [IRC catalogue 2026 (IRC:82-2023 listed)](https://www.irc.nic.in/WriteReadData/LINKS/Updated%20Catalogue,%202026(1)ed1ad28e-67a2-4bfe-82c1-4c3b75fa9107.pdf)

Models and data: [RDD2022 paper](https://arxiv.org/abs/2209.08538) · [RoadDamageDetector repo (licence)](https://github.com/sekilab/RoadDamageDetector) · [RDD2020](https://data.mendeley.com/datasets/5ty2wb6gvg) · [Attain dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12167439/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/) · [Ultralytics licence](https://www.ultralytics.com/license) · [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) · [RF-DETR](https://github.com/roboflow/rf-detr) · [SAM 3](https://github.com/facebookresearch/sam3) · [RF-DETR + ByteTrack pothole example](https://roboflow.com/blog/pothole-detection) · [Oblique phone image crack measurement (Remote Sensing 2023)](https://doaj.org/article/b0a11e715b204a448aee08d84187d7fb)

Serving and tooling: [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) · [Docker Model Runner + vLLM on Windows](https://www.docker.com/blog/docker-model-runner-vllm-windows) · [TorchCodec](https://github.com/meta-pytorch/torchcodec) · [OpenCamera Sensors](https://github.com/MobileRoboticsSkoltech/OpenCamera-Sensors) · [GoPro GPMF evolution](https://www.trekview.org/blog/evolution-of-gopro-camera-sensors-gpmf/) · [Phone camera vibration damage](https://petapixel.com/2021/09/11/motorcycle-vibrations-can-damage-your-iphone-camera-apple-warns)
