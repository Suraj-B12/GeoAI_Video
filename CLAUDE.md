# CLAUDE.md: rules for any agent working on this repository

You are continuing an existing, carefully designed project. **Do not redesign it.** Read this file, then `docs/HANDOFF_PLAN.md` (what to do, in order), then the contract for the part you touch. When something here is unclear or seems wrong, ask the owner (Suraj) before changing the design.

## What this project is

An offline, low-cost road-distress survey for Indian bituminous roads:

1. **Capture app** (`android/`, Kotlin): a Samsung Galaxy S21 5G on a vehicle records 4K30 HEVC video. Every frame is logged with its sensor timestamp on the same clock as GNSS (1 Hz) and the IMU (200 Hz), and the result is a signed "session package".
2. **Extraction pipeline** (`pipeline/`, Python, PyTorch on a GPU) turns a session into images for the VLM, with full metadata. It has two modes:
   - **segment mode** (the chosen design, being finished): the sharpest full frame of every 5 m of road, a frame-level distress check, and the full frame sent to the VLM;
   - **crop mode** (finished; kept for comparison): crops of anomalies.
3. **VLM classification**, on a separate PC owned by the VLM team: stock Qwen2.5-VL-7B (NF4, RTX A5000) behind `POST /classify`. It returns IRC:82-2015 distress labels (18 names plus `Normal`).

Everything runs offline. Data moves by USB.

## Non-negotiable rules

- **Contracts win over your ideas:**
  - session package: `docs/session-package-spec.md` (v1.1);
  - segment mode: `docs/pipeline/13-segment-frames.md`, including the record schema `geoai.segment.v1`;
  - pipeline stages: `docs/pipeline/01-12`.

  Don't change a field name, unit or clock convention. If a contract must change, update the doc and every reader in the same change, and tell the owner.
- **Time and position:** all times are Android `elapsedRealtimeNanos` (`t_ns`) and UTC is derived from GNSS. Never use container PTS or the phone's wall clock as time. Frame positions come from the Hermite trajectory (`geoai/trajectory.py`).
- **Images for the VLM are lossless PNG. Never JPEG**: re-encoding at q95 changed the VLM's labels on 9 of 49 photos.
- **Send the full-resolution frame and let the VLM's own processor resize it** (the VLM PC caps images at 1,048,576 px). Don't pre-resize.
- **Byte-exact files:**
  - `vlm_handoff/prompts/**` (hashed by the VLM team);
  - `pipeline/models/**`: DINOv2 weights (SHA-256 checked on load) and the Attain probe.

  Don't edit or re-save them. `.gitattributes` protects them.
- **No images, video or field data in git** (`.gitignore`): survey videos stay on disk as proof of work.
- **Commits:** no `Co-Authored-By` or Claude attribution lines in commit messages (owner's instruction). Commit and push at every milestone ("autosave"). Keep `PROGRESS.md` current.
- **Tests stay green.**
  - Pipeline: from `pipeline/`, run `python -m pytest -m "slow or not slow"`. Fast suite: `python -m pytest -m "not slow"`.
  - Video-join tests on a real recording: set `GEOAI_EMULATOR_SESSION=<session dir>`.
  - App: `(cd android && env -u ANDROID_HOME ./gradlew :app:assembleDebug :app:testDebugUnitTest --no-daemon --console=plain -q)`.
  - Every new behaviour gets tests whose expected values come from an independent calculation, not from the code under test.
- **No manual labelling or manual review steps.** The owner rules these out. Calibration is label-free, with the VLM as reference (`geoai gate-calibrate`).
- **Config:** all tunables live in `pipeline/geoai/config.py`, with comments giving the reason. Unknown keys in an override are an error, deliberately.
- **GPU sharing on the VLM PC:** the VLM team asked that nothing else run on their GPU while the VLM is classifying. Run extraction (DINOv2) and classification one after the other, never at the same time.
- **Downloads** (packages, models, datasets) only with the owner's approval. State the file, source and size first.
- **Style:** match the surrounding code. Dense docstrings explain why; dataclasses; type hints; vectorised numpy; deterministic outputs; Windows-safe paths. No TODOs and no dead code.

## Hardware facts (measured; don't re-derive)

**Galaxy S21 5G (SM-G991B, Android 15):**
- Camera2 offers 3840×2160 at **30 fps at most**: minimum frame duration 33.3 ms, and the auto-exposure frame-rate ranges go no higher than 30. 4K60 exists only in Samsung's own camera app.
- Timestamp source REALTIME. Lens focus distance CALIBRATED.
- Intrinsics at 4K: fx 2910, fy 2908, cx 1934, cy 1097. f = 5.4 mm, f/1.8. Rolling-shutter skew 16.6 ms.

**Mount:** bolted, **about 23° below horizontal, 1.35 m high**. It cannot be tilted further down, so work with it.
- The frame shows road from 1.44 m to about 33 m ahead.
- After the VLM's resize, along-track resolution is 2.5 mm/px at 1.5 m, 4 mm at 2 m, 13 mm at 4 m, and 25–30 mm at 6–7 m. Fine cracks beyond about 2 m are not resolvable; state this limit in every report.

**Focus:** set from geometry, at the dioptre midpoint of 1.44–7.45 m ahead, about 0.337 dpt. Expected defocus is 1.7 px at both ends. The old autofocus chose 0.80 dpt in session 2, which blurred the far road.

**Pipeline PC (owner's laptop):** RTX 4050 Laptop (6 GB), torch 2.9.1+cu126. Video decode runs at about 85 fps (PyAV, threaded). DINOv2 takes about 0.07 s per analysis window.

**VLM PC:** RTX A5000 24 GB, Windows 11. About 11–18 s per image; rate limit 10 requests/min. Stage 1 recall on Attain whole frames is 51.8 %. Details are in `vlm_handoff/vlm-information-response.md`.

## Where things are

| Path | What |
|---|---|
| `android/` | Capture app (see `android/README.md`). Key: `camera/FocusPlanner.kt`, `camera/CameraRecorder.kt`, `service/SurveyService.kt`, `coverage/` (non-repeating road-km) |
| `pipeline/geoai/` | Pipeline package. Crop mode: `extract.py`. Segment mode: `segments.py`, `gate.py`, `segment_records.py`, `analyze.py`; orchestrator `frames.py` to be written. Shared: `trajectory.py`, `geometry.py`, `video.py`, `dinov2.py`, `anomaly.py`, `cracks.py`, `probe.py`, `records.py`, `vlm.py`, `config.py`, `cli.py` |
| `pipeline/tools/run_vlm_batch.py` | Standard-library runner on the VLM PC: folder batch to `POST /classify`, resumable |
| `tools/` | `pull_sessions.py` (phone to PC, checksum-verified), `validate_session.py` (26 checks), `mount_geometry.py` |
| `docs/` | `PLAN.md` (research and design history), `session-package-spec.md`, `pipeline/01-13`, `HANDOFF_PLAN.md` (your plan) |
| `vlm_handoff/` | The VLM team's answers and byte-exact prompts. Sample images are not in git |
| `field/` | Field notes in git (validation outputs, focus diagnosis). Sessions and runs are local only |
| `PROGRESS.md` | Current state; update it as you go |

## Research record

The owner is writing a research paper and thesis. The living research record is a Claude Doc: https://claude.ai/artifact/5GroudyCievFkTPWUjgwCA (doc id `229b682c-1ec3-4c9c-8637-044502dab4f7`). If you can't edit it, write `docs/research/METRICS.md` and tell the owner.

- It covers only what this project builds.
- **After every test campaign, update it with the measured metrics.** The required list is in `docs/HANDOFF_PLAN.md` §6.
- Write only information with insight: numbers with units, how they were measured, what changed and why. No filler.
