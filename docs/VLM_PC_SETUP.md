# VLM PC setup and operation

The VLM PC is the Windows 11 machine with the RTX A5000 (24 GB) that runs the VLM system: stock Qwen2.5-VL-7B-Instruct, 4-bit NF4, images capped at 1,048,576 px, IRC:82-2015 "v2" prompts, served by FastAPI on port 8000 (`POST /classify`, `GET /health`). This document covers everything this repository does on that machine: classifying a segment batch that arrived on a USB drive, importing and analysing the answers, calibrating the distress check, and the hands-off `video_pipeline` toggle.

Two routes, same results:

| Route | When | Steps |
|---|---|---|
| **Batch (USB)** | Extraction ran on another PC (the laptop) | §3: runner → import → analyze → gate-calibrate |
| **Toggle (same machine)** | Sessions are copied straight onto the VLM PC | §5: drop the session in the inbox; everything else is automatic |

**GPU sharing rule.** Nothing else may run on the A5000 while the VLM classifies. Extraction (DINOv2) and classification therefore run one after the other, never together. The toggle enforces this; by hand, don't start `geoai process` while a runner or the VLM system's worker is busy.

## 1. One-time setup

1. Clone this repository, e.g. to `C:\Users\<you>\Documents\GeoAI_Video`.
2. Pick the Python environment:
   - **Analysis only** (batch route, extraction done elsewhere): Python 3.10+ with `pip install numpy pandas pillow`. The runner needs only the standard library.
   - **Full pipeline** (toggle route, or extraction on this PC): a venv with `pipeline/requirements.txt`, torch 2.9.1+cu126 from `https://download.pytorch.org/whl/cu126`, then `pip install --no-deps -e pipeline`. Check `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
3. Session validation needs `ffprobe` (FFmpeg) on PATH for its video checks; without it they are skipped with a warning.
4. The VLM system must be installed (its own repository) and its server started:

   ```powershell
   .\start.ps1 -AutoStart -Watchdog
   ```

## 2. Check the server

`GET http://127.0.0.1:8000/health` must report `model_name = Qwen/Qwen2.5-VL-7B-Instruct`, `quantization_bits = 4`, `max_image_pixels = 1048576`, `prompts_version = v2`, `taxonomy = IRC:82-2015`, `stage2_confidence_mode = field`. The batch expects model `Qwen/Qwen2.5-VL-7B-Instruct@cc594898…` and prompt `v2:9da1601cfec54f76`; the server reports both by name only (no revision, no prompt id), and the runner and the import match them by name. Any other prompt version is reported as a mismatch in `import_summary.json` and `REPORT.md`.

## 3. Batch route

The batch folder (`batch_001/`) holds `requests.jsonl`, `images/`, `records/`, `runs/<session>/report.json`, `batch.json`, `run_vlm_batch.py`, `RUNNING.txt` and `SHA256SUMS`. Work inside a copy on the VLM PC's disk, not on the USB drive.

| Step | Command (from the batch folder's parent, `python` = the env of §1) | Duration |
|---|---|---|
| 1. Classify | `python batch_001/run_vlm_batch.py batch_001` | 11–18 s per image; a 1.46 km send-all batch (~290 segments) ≈ 70 min |
| 2. Import | `python -m geoai vlm-import batch_001` | seconds |
| 3. Analyse | `python -m geoai analyze batch_001` | seconds |
| 4. Calibrate the check | `python -m geoai gate-calibrate batch_001 --target-recall 0.9 --out gate_calibration.json` | seconds |

Run steps 2–4 from `pipeline/` (or with the package installed). Outputs:
- `batch_001/responses.jsonl` (one line per answer), `vlm_run.json` (`/health` of every runner session);
- `batch_001/vlm_results.csv` / `.geojson`, `import_summary.json`;
- `batch_001/analysis/`: `segments_results.csv` / `.geojson`, `label_summary.csv`, `per_km.csv`, `gate_eval.json`, `vlm_timing.json`, `REPORT.md`;
- `gate_calibration.json`: copy it back to the extraction PC and set `"gate": {"calibration": "<path>"}` in the `--config` JSON. Calibration is meaningful only on a **send-all** run (`geoai process --send-all`) or with enough audit segments.

**Resuming.** Every step can be stopped and rerun. The runner skips images already answered, removes a half-written last line after a power cut, and re-sends images whose last attempt was an error. Ctrl-C stops cleanly after the current image.

Runner exit codes: `0` all answered; `1` some still to do (run again); `2` stopped before sending (checksums or `/health`) or on 401/403; `130` Ctrl-C.

Options: `--url http://127.0.0.1:8000`, `--api-key KEY` (when the server sets `API_KEYS`), `--timeout 300`, `--limit N`, `--min-interval 6.5` (10 requests/min), `--retries 4`, `--max-failures 5`.

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `STOP: … files differ from SHA256SUMS` | The batch changed in transit. Copy it again from the extraction PC |
| `STOP: cannot reach …/health` | Start the server (§1.4); check the port |
| `HTTP 401/403` | Pass `--api-key` with a key from the server's `API_KEYS` |
| `HTTP 429` | Another client shares the rate limit; the runner waits for `Retry-After` |
| `HTTP 500` with "CUDA out of memory" | Something else is on the GPU (§ GPU sharing). Stop it and rerun |
| Several images fail in a row, `STOP: … is the server up?` | The server crashed or restarted; the watchdog relaunches it. Rerun the runner |
| WARNING: `/health does not mention the expected prompt version` | The server runs another prompt set. Don't mix: finish or restart the batch under `v2` |
| `analyze` reports many "No valid label" segments | Answers with labels outside the IRC list (e.g. `N/A`). Normal answers (Stage 1 Normal, empty `distress_types`) count as `Normal` |
| `vlm-import` refuses the batch | A file changed after export; rerun the export |

## 5. Toggle route (`video_pipeline`)

The VLM system runs the whole chain itself (`app/video_pipeline.py`, `app/video_routes.py` in its repository). Operator UI (`http://localhost:8000/operator`) → **Video Pipeline** → switch on.

**Setup** (VLM system `.env`; defaults in brackets):

| Variable | Meaning |
|---|---|
| `VIDEO_INBOX` [`C:\GeoAI\inbox`] | Where verified session packages are dropped (`python tools/pull_sessions.py C:\GeoAI\inbox` pulls straight into it) |
| `GEOAI_VIDEO_REPO` [`…\Documents\GeoAI_Video`] | This repository; its `pipeline\.venv` (full pipeline, §1.2) runs validation and extraction |
| `VIDEO_RUNS_ROOT` [`C:\GeoAI\runs`] | Run folders (`geoai process --out`), kept after cleanup with the records and results |
| `VIDEO_EXTRACT_ARGS` [none] | Extra arguments for `geoai process`, e.g. `--config C:\GeoAI\gate_config.json` once the gate is calibrated (§3 step 4) |
| `VIDEO_BATCH_ROOT` [`C:\GeoAI\batches`], `VIDEO_LOG_ROOT` [`C:\GeoAI\logs`] | Temporary batches; JSON-lines logs `<date>\<session>.jsonl` plus each command's output |
| `VIDEO_POLL_S` [15], `VIDEO_SETTLE_S` [30] | Inbox polling interval; how long a package must stay unchanged before it is taken |
| `VIDEO_CLASSIFY_URL` [`http://127.0.0.1:<PORT>`], `VIDEO_CLASSIFY_CHUNK` [10] | The server's own `/classify`; images per runner call (the toggle can stop between chunks) |
| `FFPROBE_DIR` [newest `C:\GeoAI\ffmpeg\*\bin`] | ffprobe for the validator's video checks |
| `DATABASE_URL`, `LOCAL_IMAGE_ROOT` [`C:\GeoAI\images`] | Local PostgreSQL and the image store (VLM system `SETUP.md`, "Local PostgreSQL"; column mapping: [DB_MAPPING.md](DB_MAPPING.md)) |
| `VIDEO_PIPELINE_ENABLED` [false], `VIDEO_KEEP_RUN_FRAMES` [false] | Toggle default before it is first switched (the UI's choice is then kept in `<inbox>\.video_pipeline\state.json`); keep the run folder's frame copies after storing |

**What happens per session** (one at a time, oldest first; interrupted sessions first):
1. **Wait** until the package stops changing (30 s), then **validate** with `tools/validate_session.py`. A failure is refused with the validator's FAIL lines; it is re-validated only when the package changes, or with **Retry**.
2. **GPU section.** The citizen-upload worker is put on hold and drained (it finishes its current image), then: **extract** (`geoai process --mode segments`, skipped if the run folder is already complete) and **classify** (`vlm-export` → `run_vlm_batch.py` against this server, in chunks). The worker is restarted on every path, failures included, if it was running.
3. `vlm-import`, `analyze`, then **store** (images into `LOCAL_IMAGE_ROOT`, records and answers into PostgreSQL in one transaction) and **verify** (row count = records, every image present with its SHA-256).
4. **Clean up** only after verification: delete skip frames, segment views, crop-mode outputs, caches and the temporary batch; keep the session package (proof of work), the stored frames, the run's records and results (sidecars, `segments.csv/.geojson`, `report.json`, contact sheet, `vlm/` with the answers and `analysis/`). Storage before and after is recorded.

**States** shown per session: queued, extracting, classifying, storing, stored, cleaned, failed (with the error, a log tail and **Retry**). Switching off stops after the current stage (or chunk) and resumes there when switched on; a server restart resumes the same way.

**Expected durations** (A5000, measured 27 Oct 2026): extraction 87 s for 273 m, 285 s for 1.15 km, 566 s for 2.64 km; classification 11–23 s per sent segment, i.e. ~1 h per km while the gate sends ~90 % of segments with its uncalibrated defaults.
