# Progress log (resume point)

Updated as work lands. If a session ends mid-task, a new session can continue from here.

## State on 27 Oct 2026

**Done and tested:**
- **Capture app** (`android/`), installed on the S21:
  - one-tap start, auto-record and recommended defaults;
  - non-repeating road-km counter;
  - layout fixes;
  - road-only focus from geometry plus a sweep (`camera/FocusPlanner.kt`).
  - 33 JVM tests pass.
- **Extraction pipeline, crop mode** (`pipeline/`): DINOv2 session bank, Attain probe, crack filter.
  - No crop cap; ground-plane duplicate removal; distresses within 0.5 m in a frame share one crop.
  - Runs on the GPU (torch 2.9.1+cu126).
  - 80 tests pass.
- **Field test, 26 Oct:** two S21 sessions validated (`field/validate_*.txt`); focus diagnosis in `field/focus_diagnosis.md`.
- **Research record:** Claude Doc https://claude.ai/artifact/5GroudyCievFkTPWUjgwCA

**In progress: segment mode** (spec: `docs/pipeline/13-segment-frames.md`). One best full frame per 5 m of road; a frame-level distress check on the segment; the full frame goes to the VLM. Config sections `segments` and `gate` are already in `pipeline/geoai/config.py`.

| Part | Files | Status |
|---|---|---|
| Segment planning + best-frame selection | `pipeline/geoai/segments.py`, `tests/test_segments.py` | to build |
| Distress check + calibration | `pipeline/geoai/gate.py`, `AnomalyDetector.maps()` in `anomaly.py`, `tests/test_gate.py` | to build |
| Records, EXIF, VLM export/import, analysis | `segment_records.py`, `vlm.py`, `analyze.py`, `records.py`, `tools/run_vlm_batch.py`, tests | to build |
| Orchestrator + CLI | `pipeline/geoai/frames.py`, `cli.py` (`process --mode segments`, `gate-calibrate`, `analyze`), end-to-end simulator test | to build |
| App: focus range for 5 m segments, live tilt and road-range display | `android/.../FocusPlanner.kt`, `Screens.kt` | to build |
| VLM PC instructions | `docs/VLM_PC_SETUP.md` | to write |
| Run on both 26 Oct sessions; review; push | - | after the above |

## How to resume

1. Read `docs/pipeline/13-segment-frames.md`, which is the contract.
2. Run `python -m pytest -m "slow or not slow"` in `pipeline/`; it must stay green.
3. Build each missing part in the table, with its tests.
4. Commit and push to https://github.com/Suraj-B12/GeoAI_Video. No images or videos (see `.gitignore`); no co-author trailer in commit messages.
