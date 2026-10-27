# Progress log (resume point)

Updated as work lands. If a session ends mid-task, a new session can continue from here.

## State on 27 Oct 2026, evening (HANDOFF_PLAN steps 1–6 built on the VLM PC)

| Part | Status |
|---|---|
| Step 1: review of the drafted modules | **Done.** No deviations from spec 13. Two bugs against the real VLM server fixed (`cb1cdce`): a Stage 1 Normal answer has an empty `distress_types` and was read as "no valid label"; `/health` reports `prompts_version: "v2"` without the prompt id. The failing analyze test expected the wrong size (server `min_pixels` = 7168 → 112×63 becomes 140×84) |
| Step 2: `frames.py`, CLI, simulator end to end | **Done** (`ff7ee9b`). `process --mode segments [--send-all]`, `gate-calibrate`, `analyze`; JSON-lines run logs; bank and gate 1.7–2× faster with byte-identical outputs |
| Tests | Pipeline full suite **221 passed, 2 skipped** (emulator tests; pass with `GEOAI_EMULATOR_SESSION`). App **44/44** JVM tests (built on the VLM PC with the original signing key) |
| Step 3: real sessions | **Done.** Three sessions pulled and verified (`field/validate_*.txt`), processed in gate mode and send-all; send-all batch `field/batches/batch_001` (813 segments, 4.89 GB) exported |
| Step 4: classify, import, analyze, calibrate | **Running:** `batch_001` on the A5000 (~16 s per image, ~3.5 h). Then `vlm-import`, `analyze`, `gate-calibrate`. `docs/VLM_PC_SETUP.md` written |
| Step 5: local PostgreSQL + image store | **Done** in the VLM system (its repo, branch `stage2-probe-shadow`, `0ada133`): `DB_BACKEND=postgres`, migrations 000 (local shim) and 005 (video tables), `app/video_store.py`. Mapping: [docs/DB_MAPPING.md](docs/DB_MAPPING.md). **Owner:** apply migration 005 to Supabase |
| Step 6: operator UI `video_pipeline` toggle | **Built and tested on fakes** (134 checks). End-to-end run on a real session follows the batch (GPU is busy) |
| App on the S21 | APK with the 5 m segment focus range built and signed with the phone's key; **install pending: phone not connected** (`adb install -r`, never uninstall; [docs/UPDATING_THE_APP.md](docs/UPDATING_THE_APP.md)) |

### Real sessions (A5000, segment mode, uncalibrated gate)

| Session | Route | Segments (send / audit / skip / not covered) | Gate pass | Extraction | Bank |
|---|---|---|---|---|---|
| f428 (26 Oct, focus 0.46 dpt, autofocus) | 1,149.8 m, 100 % covered | 230 (220 / 0 / 10 / 0) | 95.7 % | 285 s | 78 frames, 99.5 s |
| 293d (26 Oct, focus 0.80 dpt, autofocus) | 273.0 m, 100 % | 55 (52 / 1 / 2 / 0) | 94.5 % | 87 s | 19 frames, 19 s |
| d47c (27 Oct, focus 0.374 dpt, geometry + sweep, old 4 m range) | 2,637.5 m, 100 % | 528 (467 / 3 / 58 / 0) | 88.4 % | 566 s | 175 frames, 156 s |

- Chosen-frame sharpness, median: 0.029 / 0.033 / **0.112**. d47c, with geometric focus, is ~3.5× sharper, though speed and light also differ.
- Predicted blur, median: 1.29 / 1.53 / 0.97 px.
- Low-light share of chosen frames: 39.6 % / 5.5 % / 0 %.
- d47c exposure median is 1/2703 s; it has 0.19 % dropped frames (30 gaps, over the 0.1 % target).
- **Main finding:** the gate's default thresholds don't discriminate. The crack filter reports 8–27 m of "cracks" per 5 m segment on real 4K frames, and the bank evidence median is 786–3,262 against a threshold of 30. Calibrating on `batch_001` is required before gate mode saves VLM time.

## Earlier (handed off 27 Oct, morning)

- Capture app, crop mode (80 tests), field test of 26 Oct (`field/focus_diagnosis.md`).
- Research record: Claude Doc https://claude.ai/artifact/5GroudyCievFkTPWUjgwCA
