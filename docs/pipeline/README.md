# Extraction Pipeline: Design

The pipeline runs **offline on a PC**. It pulls sessions from the GeoAI Capture phone over USB, finds every stretch of road surface that doesn't look like healthy pavement, cuts it out as a geotagged, time-stamped crop sized for the VLM, and hands the crops to the VLM on another machine.

> **Status (22 Oct 2026): implemented and tested in [`/pipeline`](../../pipeline/README.md).**
> The scope changed after the VLM team's answers ([`/vlm_handoff`](../../vlm_handoff/vlm-information-response.md)). The pipeline only extracts; the VLM classifies. The VLM is not fine-tuned, and there is **no manual work**: no labelling, no review.
>
> | Design document | Status |
> |---|---|
> | 01 Ingest, 02 Time and trajectory, 03 Video decoding | Implemented as designed |
> | 04 Calibration and geometry | Projection implemented; geometry is computed automatically from the session (gravity + camera characteristics + form height); calibration sessions not yet used |
> | 05 Detection | **Replaced** by label-free anomaly detection |
> | 06 Association | **Replaced** by band coverage: no tracking |
> | 07 Best frame and crop | **Updated** for the Qwen2.5-VL input rules, PNG crops, full frames |
> | 08 Geotag and records | Implemented with PNG EXIF + sidecar JSON + CSV/GeoJSON (no SQLite yet) |
> | 09 VLM interface | **Updated** to the VLM team's real API, with a ready runner script |
> | 10 Review and export | Manual review dropped; automatic feedback from VLM results; IRC section ratings not built |
> | 11 Testing and acceptance | Automated: simulator, fixtures, end-to-end tests. Field items are optional |
> | 12 Pitfalls | Extended with what testing found |

Implementation, usage, tests and validation results: [`pipeline/README.md`](../../pipeline/README.md). The input format is defined in [the session package spec](../session-package-spec.md).

| # | Document | Covers |
|---|---|---|
| 01 | [Ingest](01-ingest.md) | USB pull, verification, QC report, session registry |
| 02 | [Time and trajectory](02-time-and-trajectory.md) | One clock, UTC, GNSS latency, per-frame pose |
| 03 | [Video decoding](03-video-decoding.md) | Joining video samples to `frames.csv`, colour, frame quality |
| 04 | [Calibration and geometry](04-calibration-and-geometry.md) | Lens and mount calibration, road-plane projection |
| 05 | [Detection](05-detection.md) | Label-free detection of non-healthy pavement |
| 06 | [Association](06-association.md) | Covering each metre of road exactly once |
| 07 | [Best frame and crop](07-best-frame-and-crop.md) | Crops that reach Qwen2.5-VL unchanged |
| 08 | [Geotag and records](08-geotag-and-records.md) | EXIF, sidecars, records |
| 09 | [VLM interface](09-vlm-interface.md) | Offline folder batch over USB |
| 10 | [Review and export](10-review-and-export.md) | Exports, automatic feedback, IRC summary (later) |
| 11 | [Testing and acceptance](11-testing-and-acceptance.md) | Simulator, metrics, field protocol |
| 12 | [Pitfalls](12-pitfalls.md) | Common mistakes and where the design prevents each |

A trained detector is no longer needed; [../detection-model-plan.md](../detection-model-plan.md) is kept as an optional later upgrade. The VLM questions ([../vlm-information-request.md](../vlm-information-request.md)) are answered in [`/vlm_handoff`](../../vlm_handoff/vlm-information-response.md).

---|---|---|
| 01 | [Ingest](01-ingest.md) | USB pull, verification, QC report, session registry |
| 02 | [Time and trajectory](02-time-and-trajectory.md) | One clock, UTC, GNSS latency, per-frame pose |
| 03 | [Video decoding](03-video-decoding.md) | Joining video samples to `frames.csv`, colour, frame quality |
| 04 | [Calibration and geometry](04-calibration-and-geometry.md) | Lens and mount calibration, road-plane projection |
| 05 | [Detection](05-detection.md) | Running the distress detector on 4K frames |
| 06 | [Association](06-association.md) | One record per physical distress |
| 07 | [Best frame and crop](07-best-frame-and-crop.md) | Choosing the view, cutting the crop |
| 08 | [Geotag and records](08-geotag-and-records.md) | EXIF/XMP, sidecars, database schema |
| 09 | [VLM interface](09-vlm-interface.md) | Offline batch hand-off and LAN API |
| 10 | [Review and export](10-review-and-export.md) | Human review, GIS exports, IRC section summary |
| 11 | [Testing and acceptance](11-testing-and-acceptance.md) | Test data, simulator, metrics, field protocol |
| 12 | [Pitfalls](12-pitfalls.md) | Common mistakes and where the design prevents each |

Model training (detector data, labelling and evaluation) is in [../detection-model-plan.md](../detection-model-plan.md). Questions for the VLM owners are in [../vlm-information-request.md](../vlm-information-request.md).

---

## 1. Data flow

```
 phone (USB) ──pull──► data/sessions/<id>/  (read-only after verification)
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
   02 time & trajectory          04 calibration (from calibration_* sessions)
   frame_times.parquet           intrinsics.json, mount.json
   trajectory.parquet                     │
             │                            │
             ▼                            │
   03 decode + frame QC ◄─────────────────┘
             │  RGB frames (streamed, never stored in full)
             ▼
   05 detection ──► detections.parquet
             │
             ▼
   06 association (road-plane, path coordinates) ──► distresses.parquet + observations.parquet
             │
             ▼
   07 best frame + crop ──► crops/<distress_id>.jpg + context/<distress_id>.jpg
             │
             ▼
   08 geotag ──► EXIF/XMP in crops, sidecar JSON, SQLite records
             │
             ▼
   09 VLM batch out ──► (VLM machine) ──► responses in ──► classifications
             │
             ▼
   10 review ──► exports (GeoPackage, GeoJSON, CSV, KML) + IRC section summary
```

Detection needs decoded frames, and cropping needs the same frames at full resolution. The design keeps both in **one decoding pass**. Stages 05–07 run as a streaming pipeline over the video. They keep only the current best full-resolution candidate per open track (and per piece of a long distress) in memory, and spool finished candidates to disk ([07](07-best-frame-and-crop.md) §1).

## 2. Runtime

| Item | Choice | Why |
|---|---|---|
| Language | Python 3.11 or 3.12 | ML ecosystem |
| Core libraries | numpy, scipy, pandas, pyarrow, opencv-python (≥ 4.8; ArUco/ChArUco in `cv2.aruco`), av (PyAV ≥ 12), pymap3d, shapely ≥ 2, pyproj | All permissive licences; all work offline |
| Detector runtime | PyTorch (CUDA) + the chosen detector package; optional ONNX Runtime / TensorRT | See [05](05-detection.md) |
| Metadata writing | ExifTool (Phil Harvey, Perl, run via `pyexiftool`) | The most reliable EXIF + XMP writer |
| Storage | SQLite (WAL mode) for records; Parquet for per-session tables; files for images | Single PC, offline, no server to administer. PostgreSQL + PostGIS is a later option |
| GIS output | GeoPackage via geopandas/pyogrio | Opens natively in QGIS, offline |
| Geoid | EGM2008 grid `us_nga_egm08_25.tif` (from PROJ-data) copied into the PROJ data directory; `PROJ_NETWORK=OFF` | MSL altitude when the phone gives none ([08](08-geotag-and-records.md) §1) |
| UUIDs | `uuid6` package (UUIDv7) on Python < 3.14 | Time-ordered record IDs |
| Packaging | `pip install` from a local wheelhouse (`pip download` on a connected machine, carry it over) | The pipeline machine must never need the internet |

Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and the detector package's telemetry/settings flags to off. Store every model file under `models/` with its SHA-256.

## 3. Directory layout (pipeline machine)

```
geoai/
  config/pipeline.yaml          all thresholds and paths (versioned with the code)
  models/<name>/<version>/      detector weights + model card + sha256
  data/
    sessions/<session_id>/      verified raw sessions (read-only)
    calibration/
      intrinsics/<device_id>/<WxH>/<calibration_session_id>.json
      mount/<mount_id>/<calibration_session_id>.json
    work/<session_id>/          derived data; safe to delete and regenerate
      qc_report.json
      time_model.json  gnss_clean.parquet
      frame_times.parquet
      trajectory.parquet
      detections.parquet
      observations.parquet
      distresses.parquet
      spool/                    candidate crops during processing (deleted at the end of stage 07)
      crops/  context/  sidecars/
      stage_manifest.json
    vlm_batches/<batch_id>/     requests.jsonl, images/, responses.jsonl
  db/geoai.sqlite
  exports/<date>_<name>/
```

## 4. Command line

| Command | Stage | Notes |
|---|---|---|
| `geoai pull <phone>` | 01 | Wraps `tools/pull_sessions.py` |
| `geoai ingest <session>` | 01 | Validation + QC report + registry entry |
| `geoai calibrate-intrinsics <session>` | 04 | From a `calibration_intrinsics` session |
| `geoai calibrate-mount <session>` | 04 | From a `calibration_mount` session |
| `geoai process <session>` | 02–08 | Time, trajectory, decode, detect, associate, crop, geotag |
| `geoai vlm export <batch>` / `geoai vlm import <batch>` | 09 | Offline hand-off |
| `geoai vlm run` | 09 | Only if a LAN endpoint exists |
| `geoai review` | 10 | Local review UI |
| `geoai export <query>` / `geoai report <road>` | 10 | GIS files and section summaries |
| `geoai export-training` | 10 | Reviewed data for detector and VLM training, split by road section |

## 5. Rules every stage follows

1. **Raw data is read-only.** Stages write only under `work/`, `db/`, `vlm_batches/`, `exports/`.
2. **Idempotent and resumable.** Each stage writes `stage_manifest.json` entries `{stage, code_version, config_hash, input_hashes, output_hashes, started, finished}`. Re-running with the same inputs and config is a no-op. A changed model or config re-runs that stage and everything after it.
3. **One clock.** All times are session-clock nanoseconds (`t_ns`) until the final UTC/IST conversion in stage 08.
4. **Never invent data.** A frame with no valid position is excluded or flagged, never given an estimated position silently.
5. **Every record carries provenance:** session, segment, frame, detector version, calibration IDs, VLM model and prompt version.
6. **Fail loudly at ingest, degrade gracefully later.** A session that fails integrity checks is not processed at all. A session with weak GNSS is processed with every affected record flagged.

## 6. Hardware

The pipeline PC needs an NVIDIA GPU with ≥ 12 GB VRAM for the detector (the VLM runs elsewhere), 8+ CPU cores, 32–64 GB RAM, and fast NVMe storage. One survey hour at 4K is about 27–45 GB of video. Throughput target: one hour of video processed in one hour or less ([11](11-testing-and-acceptance.md)).
