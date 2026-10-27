# Handoff plan: finish segment mode and connect it to the VLM system

Written 27 Oct 2026 for the agent taking over. Read `CLAUDE.md` first. Work through the steps **in order**; each has acceptance criteria. Commit and push after each step (branch `main`; no co-author lines) and update `PROGRESS.md`.

## 1. System architecture

```
 PHONE (S21, android/)            LAPTOP or VLM PC (pipeline/)                         VLM PC (VLM team's system)
 ┌──────────────────────┐  USB   ┌────────────────────────────────────────────┐        ┌─────────────────────────────┐
 │ 4K30 HEVC + per-frame │ ─────► │ ingest + verify (tools/pull_sessions.py)   │        │ Qwen2.5-VL-7B (NF4, A5000)   │
 │ t_ns, GNSS 1 Hz,      │ session│ trajectory (GNSS+IMU, UTC from GNSS)       │        │ POST /classify  GET /health  │
 │ IMU 200 Hz, focus/exp │ package│ geometry (intrinsics + gravity + height)   │        └──────────────▲──────────────┘
 │ log, signed manifest  │        │ SEGMENT MODE:                               │                       │ one PNG
 └──────────────────────┘        │  5 m segments -> best (sharpest) frame      │                       │ per request
                                 │  -> gate (DINOv2 bank + probe + cracks)     │ ── batch folder ──────┘
                                 │  -> full-res PNG + EXIF + geoai.segment.v1  │ ◄─ responses.jsonl
                                 │ vlm-import -> analyze (thesis tables)       │
                                 └────────────────────────────────────────────┘
                                         │ (new, step 5-7) results + image paths
                                         ▼
                                 Local PostgreSQL (same schema as the existing Supabase project)
                                 + local image store (replaces Cloudinary) + operator_ui toggle "video_pipeline"
```

**Data units.**
- A *session* is one recording (`field/sessions/<id>/`).
- A *segment* is 5 m of road along the GNSS path, with exactly one record (`<session>_seg<k:05d>`).
- Segment statuses:
  - `send`: passed the distress check, so it goes to the VLM;
  - `audit`: a random 10 % of the segments that failed the check, also sent, to measure how many distressed segments the check misses;
  - `skip`: failed the check, not sent;
  - `not_covered`: no usable frame (GNSS gap or black video), with the reason.

**Contract:** `docs/pipeline/13-segment-frames.md` covers geometry, best-frame rule, distress check, record schema, files and analysis outputs. Do not deviate from it.

## 2. Status (27 Oct 2026)

| Part | State | Notes |
|---|---|---|
| Capture app | **Done, on the S21** | 44 JVM tests pass. Road-only focus from geometry (0.337 dpt for 5 m segments). Live tilt, road range and "5 m segment fits" display. Auto-record, recommended defaults, non-repeating road-km counter. **Not yet checked on a drive:** focus near 0.34 dpt, landscape layout |
| Crop mode (`extract.py`) | **Done** | 80 tests; results on both sessions are in the research record §9 |
| `segments.py`: segment planning, geometry, candidate window, sharpness, blur prediction, `BestFrameSelector` | **Drafted + unit tests** (`tests/test_segments.py`) | Builder interrupted by usage limits before its final report. Review against spec §2–3 |
| `gate.py` + `AnomalyDetector.maps()`: segment view, `SegmentGate.score/decide`, `calibrate` | **Drafted + unit tests** (`tests/test_gate.py`) | Same. Also check the bank-speed work (session f428's bank took 296 s for 78 frames) |
| `segment_records.py`: record builder, schema validator, PNG check, CSV/GeoJSON, contact sheet | **Drafted + unit tests** | Same |
| `vlm.py` segment export/import; `run_vlm_batch.py` hardening | **Drafted** | Crop export must stay identical |
| `analyze.py`: thesis outputs | **Drafted + tests** | 1 failing test: `test_analyze.py::test_segment_export_contents` expects the Qwen size of a 112×63 test image to be 112×56 and gets 140×84. Find out which is right with `geoai/qwen.smart_resize` and the VLM team's `min_pixels`, and fix the code or the test |
| `frames.py` orchestrator + CLI (`process --mode segments`, `gate-calibrate`, `analyze`) | **Not started** | Step 3 |
| End-to-end simulator test and real-session runs | **Not started** | Step 3 |
| `docs/VLM_PC_SETUP.md` | **Not started** | Step 4 |
| Local Postgres, local image store, operator_ui toggle, frame cleanup | **Not started** | Steps 5–7 |

Fast suite at handoff: 185 passed, 1 failed (above), 2 skipped (need `GEOAI_EMULATOR_SESSION`). Run it first to confirm your starting point.

## 3. Step 1: review and finish the drafted modules

1. Read spec §2–6, then each drafted module and its tests. Check against the spec:
   - names and statuses;
   - the record fields (`validate_segment_record` must enforce the full schema);
   - the candidate window;
   - sharpness = normalised Tenengrad on the segment rows and lane columns;
   - blur = motion and rotation from the gyro;
   - the gate restricted to segment rows; audit = SHA-256 of the record id;
   - calibration edge cases.
2. Fix the failing analyze test (above).
3. Run the full suite (`-m "slow or not slow"`): all tests pass.

**Acceptance:** the full suite is green, and no module deviates from the spec.

## 4. Step 2: orchestrator and CLI (`pipeline/geoai/frames.py`, `cli.py`)

`process_session_segments(session_dir, out_root, cfg, *, allow_test=False, allow_untrusted=False, features=None) -> report`. Mirror `extract.process_session` for loading, checks, trajectory, `nominal_model`, the `_median_gravity` view and the `PngPool`. Then:

1. `SegmentGeometry` from the model; `plan_segments(traj, cfg)`.
2. **Bank pass:** keyframes spread over the route, as in `extract.bank_pass`, on the gate view (`gate.plan_segment_view`, with far edge from `gate.gate_far_m`).
3. **One decode pass** with `video.iter_segment(want=selector.wants ...)`. For each frame, compute the camera's path position `c` at the centre-row time (`extract._RowClock`), its exposure, ISO and `low_light`, and its gyro rate (`segments.GyroImageRates`). Offer it to `BestFrameSelector`.
4. For each segment `pop_ready()` returns: gate score and decide; build the record (`segment_records.build_segment_record`). For `send`/`audit`, write the full-resolution PNG through `PngPool` (zlib level 1, record embedded, EXIF). Use `export_segment_view` only if configured.
5. Write the sidecars, `segments.csv`, `segments.geojson`, contact sheet and `report.json` (`mode: "segments"`; coverage metres, pass rates, timings, VLM-hour estimate).
6. CLI:
   - `process --mode segments|crops` (default `segments`; `--send-all` sets `gate.mode=all`);
   - `gate-calibrate BATCH... [--target-recall 0.9] [--out gate_calibration.json]`;
   - `analyze BATCH`;
   - `vlm-export` and `vlm-import` detect the mode from `report.json`.
7. **End-to-end test on the simulator** (`geoai.sim`, probe off as in `test_e2e_sim.py`):
   - every segment is covered, or `not_covered` with a reason in a simulated GNSS gap;
   - every simulated distress lies in a `send` segment;
   - every sidecar validates; PNG EXIF and embedded record match the sidecar;
   - reruns are byte-identical;
   - export, then a fake runner, then import, then analyze.

**Acceptance:**
- the simulator end-to-end test passes;
- both real sessions process without errors (`python -m geoai process ../field/sessions/<id> --out ../field/runs_segments --mode segments`);
- all records validate;
- record wall time per session, segments, pass rate and coverage.

## 5. Step 3: real-session check and calibration batch

- Run both 26 Oct sessions, plus the owner's new recordings, with `--send-all` (calibration run, about 290 segments for 1.46 km, about 70 min of VLM time).
- Export a batch: `geoai vlm-export ../field/runs_segments/<id>... --batch <usb>/batch_001`.

## 6. Step 4: on the VLM PC: classify, import, analyze, calibrate; write `docs/VLM_PC_SETUP.md`

On the VLM PC:
1. Clone the repo.
2. `pip install numpy pandas pillow` (analysis needs no torch).
3. `GET /health`.
4. `python run_vlm_batch.py <batch>`: resumable, 10 requests/min.
5. `python -m geoai vlm-import <batch>`.
6. `python -m geoai analyze <batch>`.
7. `python -m geoai gate-calibrate <batch>`, then copy `gate_calibration.json` back and set `gate.calibration`.

Document every command, the expected durations, how to resume after an interruption, and troubleshooting, in `docs/VLM_PC_SETUP.md`.

## 7. Step 5: local PostgreSQL and local image storage (the owner's requirement)

The owner's existing VLM system (its own repository, on the VLM PC, **not in this repo**) uses **Supabase** (Postgres) for records and **Cloudinary** for images, and has an **operator_ui**. Requirements:
- Store the outputs **locally**: install PostgreSQL locally on the VLM PC.
- Use **the same schema** as the existing Supabase project.
- Store images **on the local disk instead of Cloudinary**, with the image **file path** kept in the database in place of the Cloudinary URL.

How:
1. **Inspect the existing VLM project first.** Find its Supabase schema (migrations, SQL files, or `pg_dump --schema-only` from Supabase) and every place it uploads to Cloudinary or reads Cloudinary URLs. Write down the tables, columns and the code paths. Don't guess the schema.
2. **Install PostgreSQL locally** (get the owner's approval for the download; say the version and size). Create the database from the dumped schema unchanged. Point the VLM system's DB connection at it through configuration (environment variables), keeping Supabase as an option, not by hard-coding.
3. **Image store.** A configurable root (e.g. `D:\GeoAI\images\<session_id>\<segment_id>.png`), written atomically. Store the absolute (or root-relative, consistently) path in the column that held the Cloudinary URL. If the schema needs a new column, add it by migration and tell the owner. Keep a checksum column if the schema has one.
4. **Map `geoai.segment.v1` records and VLM results onto the existing tables:**
   - keep a mapping document `docs/DB_MAPPING.md` (table.column → record field);
   - store the full record JSON if there is a JSON column;
   - keep segment id, session id, coordinates, chainage, UTC time, labels, severity, confidences, model id and prompt version.
5. **Tests:** insert, read back, check the path on disk, and that re-running doesn't duplicate rows (idempotent: upsert keyed by segment id).

## 8. Step 6: operator_ui "video_pipeline" toggle

Add a button or toggle **"video_pipeline"** to the existing operator_ui. When enabled, it runs the whole chain for new sessions, with no manual step:

1. **Watch** an input folder for verified session packages (copied from the phone with `tools/pull_sessions.py`, or dropped by USB). Validate with `tools/validate_session.py`; refuse failed sessions and show why.
2. **Extract:** `process --mode segments` (best frame per 5 m, distress check). Never run this while the VLM is classifying; queue it (GPU sharing rule).
3. **Classify:** send each `send`/`audit` frame to the local `/classify` directly. Same machine, so no batch copy is needed, but reuse `run_vlm_batch.py`'s logic: rate limit, retries, resume.
4. **Store:** the image in the local store, and the record plus VLM result in local Postgres (step 5).
5. **Clean up:** after a session is stored and verified (row count = records, every image path exists and its checksum matches):
   - delete all intermediate frames and files that were not used: `skip` frames if any were written, segment views, temporary batch copies, crop-mode outputs and caches;
   - **keep:** the session package (video + CSVs + manifest) as proof of work, the stored best frames, and the records and results.
6. **Show progress and state per session** in the operator_ui: queued, extracting, classifying, stored, cleaned, failed (with the error). Disabling the toggle stops after the current item, cleanly and resumably.

Log every stage (section 9). Test the toggle on one real session end to end before handing back.

## 9. Logging and metrics (required)

- **Structured log per run** (JSON lines, `logs/<date>/<session>.jsonl`): stage, start/end time, counts, errors with stack traces. Keep the existing human log lines too.
- **Per-session metrics** (`report.json` plus a DB row if the schema allows):
  - route m, segments, covered and not-covered m;
  - candidates per segment, sharpness and predicted blur distribution, low-light share;
  - distress-check pass, audit and skip counts;
  - DINOv2 and decode and total timings;
  - VLM seconds per image (median, p95), errors and retries;
  - storage before and after cleanup (GB).
- **Research record updates** (Claude Doc, see `CLAUDE.md`) after each test campaign, as measured numbers:
  1. coverage (% of route with a best frame);
  2. best-frame quality (sharpness, blur in px) vs speed;
  3. how many distressed segments the check misses and how many of its passes the VLM confirms, with 95 % intervals (`analyze → gate_eval.json`), before and after calibration;
  4. VLM label distribution, % of surveyed length per IRC label, per km;
  5. VLM time per km and per image;
  6. processing time per km (laptop vs VLM PC);
  7. storage per km before and after cleanup;
  8. focus: the logged `capture_settings.focus` (target dpt, method) on the new drives, and road-band sharpness compared with session 1/2.

  Write only numbers that inform a decision or a thesis claim, with how each was measured.

## 10. Definition of done

1. All tests pass (pipeline full suite, app unit tests, new DB/UI tests).
2. With the toggle on, a new S21 session dropped into the input folder ends up in local Postgres with local image paths, without manual steps, and unused intermediates are deleted.
3. `docs/VLM_PC_SETUP.md`, `docs/DB_MAPPING.md`, `PROGRESS.md` and the research record are updated with measured metrics.
4. Everything is committed and pushed (no images or video in git; no co-author lines).
