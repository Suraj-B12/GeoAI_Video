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

**Handed off on 27 Oct 2026.** Segment mode is in progress. Follow `docs/HANDOFF_PLAN.md`; the rules are in `CLAUDE.md`.

| Part | Status |
|---|---|
| App focus range for 5 m segments, live tilt and road-range display | done; 44 app tests pass |
| `segments.py`, `gate.py` (+ `anomaly.maps`), `segment_records.py`, `vlm.py`/`run_vlm_batch.py`, `analyze.py` | drafted with unit tests; builders were cut off by usage limits; review per HANDOFF_PLAN step 1 |
| Fast suite | 185 passed, 1 failed (`test_analyze.py::test_segment_export_contents`), 2 skipped |
| `frames.py` + CLI + simulator end-to-end + real runs | not started (HANDOFF_PLAN step 2) |
| `docs/VLM_PC_SETUP.md` | not started (step 4) |
| Local Postgres, local image store, operator_ui `video_pipeline` toggle, cleanup | not started (steps 5–6) |
