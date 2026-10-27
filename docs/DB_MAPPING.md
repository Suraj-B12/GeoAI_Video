# DB mapping: segment records and VLM answers in the VLM system's database

Status: 27 Oct 2026, HANDOFF_PLAN step 5. Implemented in the VLM system repository (Capstone):
`app/video_store.py` (writer), `migrations/005_video_pipeline.sql` (new tables),
`scripts/setup_local_db.py` (local database), `scripts/tests/test_local_db.py` (tests).

## 1. Where things are stored

- **Database:** a local PostgreSQL 17 on the VLM PC (`DB_BACKEND=postgres`, `DATABASE_URL`), with the
  **same schema as the VLM system's Supabase project**. The schema was not dumped (no DB password exists);
  it is rebuilt from the VLM repo's own SQL, unchanged (`db_schema.sql`, migrations 001-004), after a local
  shim (`000_local_postgres_shim.sql`) for what Supabase provides: the `auth` schema and functions, the API
  roles, and the RoadSide app's `photos` table. A test compares every table and view, column by column
  (name, type, default), with the live Supabase PostgREST OpenAPI of 27 Oct 2026. Migration 005 adds two
  tables; the owner applies the same file to Supabase to keep the two identical.
- **Images:** the local disk replaces Cloudinary. One lossless PNG per `send`/`audit` segment, byte-identical
  to the run's `frames/<id>.png`, at `LOCAL_IMAGE_ROOT\<session_id>\<segment_id>.png`
  (default `C:\GeoAI\images`). Written atomically (temporary file, fsync, rename) and only after its
  SHA-256 equals the record's `frame.sha256`.
- **Path instead of URL:** `photos.image_url` and `assessments.image_url` (the Cloudinary URL for app
  uploads) hold the image's **absolute path**. The VLM server tells the two apart (`app/local_images.py`): a
  value with `<scheme>://` is a URL, an absolute path is a stored file, read and deleted only under
  `LOCAL_IMAGE_ROOT`. `video_segments.image_path` holds the same path and `video_segments.image_sha256` its
  checksum (the existing tables have no checksum column). Browsers can't load a local path, so the VLM
  server serves stored images at `GET /local-images?path=<image_url>` (only `.png`/`.jpg` under
  `LOCAL_IMAGE_ROOT`) and the operator pages map local paths to it.

## 2. One segment, four places

| Segment | `video_segments` | image file | `photos` + `assessments` |
|---|---|---|---|
| `send` / `audit`, answered by the VLM | row, `vlm_response` set | yes | one pair, filled with the answer |
| `send` / `audit`, not answered (yet) | row | yes | none (created when the answer arrives) |
| `skip` | row | no | none |
| `not_covered` | row | no | none |

A classified segment is an ordinary `photos` + `assessments` pair, so the operator dashboard and the metrics
views show it with no change. The expert review UI reads Supabase straight from the browser, so it sees
these rows only when the database is Supabase, not the local PostgreSQL. The `photos` insert fires the existing trigger
`trg_sync_photo_to_assessment`, which creates a `pending` assessment; the writer fills it with the VLM
result **in the same transaction**, so it is committed as `classified` / `expert_review` and the
PipelineWorker (which claims only committed `pending` rows) never classifies it a second time.

**Idempotency key:** the segment id `<session_id>_seg<k:05d>`. It is the primary key of
`video_segments` and is stored in `assessments.road_segment_id`, unique since migration 005. A re-run
updates the same rows and reuses images with a matching checksum; it never duplicates and never deletes.

## 3. `video_sessions` (new, one row per recording)

| Column | Source |
|---|---|
| `session_id` (pk) | record `session_id` (the session folder name) |
| `state` | pipeline state: `queued`, `extracting`, `classifying`, `storing`, `stored`, `cleaned`, `failed` |
| `error` | last failure with its stage; cleared when the session moves on |
| `metrics` | per-session metrics (HANDOFF_PLAN section 9) |
| `report` | the run's `report.json` |
| `queued_at` ... `failed_at` | when the session last entered each state |
| `created_at`, `updated_at` | row times (`updated_at` maintained by trigger) |

## 4. `video_segments` (new, one row per record, every status)

| Column | Record field |
|---|---|
| `segment_id` (pk) | `id` (checked: `= session_id || '_seg' || k` zero-padded to 5) |
| `session_id` (fk `video_sessions`) | `session_id` |
| `k` | `segment.k` (unique with `session_id`) |
| `status` | `status` (`send`, `audit`, `skip`, `not_covered`) |
| `length_m`, `s0_m`, `s1_m` | `segment.length_m`, `segment.s0_m`, `segment.s1_m` |
| `centre_lat`, `centre_lon` | `segment.centre.lat`, `segment.centre.lon` |
| `chainage_start_km`, `chainage_end_km` | `segment.chainage_start_km`, `segment.chainage_end_km` (null without a survey chainage) |
| `utc` | `frame.utc_ms` as timestamptz (null when not covered or `no_utc`) |
| `image_path` | absolute path of the stored PNG (send/audit) |
| `image_sha256` | `frame.sha256` (verified against the file) |
| `photo_id`, `assessment_id` | the segment's `photos` / `assessments` rows once answered (`ON DELETE SET NULL`) |
| `record` | the **full** `geoai.segment.v1` record (sidecar JSON), every field |
| `vlm_response` | the raw `/classify` JSON for this segment |

The columns are copies for queries; `record` stays the source of truth.

## 5. `photos` (existing; RoadSide app's table)

| Column | Value |
|---|---|
| `id` | generated |
| `image_url` | absolute path of the stored PNG (replaces the Cloudinary URL) |
| `address` | `"<survey.road_name> km <segment.chainage_start_km, 3 decimals>"`; the road name alone without chainage; else `session_id` |
| `latitude`, `longitude` | `segment.centre.lat`, `segment.centre.lon` (the segment centre, not the camera) |
| `captured_at` | `frame.utc_ms` (GNSS-derived UTC of the frame) |
| `created_at` | time of storing (upload time, as for app uploads) |

## 6. `assessments` (existing)

| Column | Value |
|---|---|
| `photo_id`, `image_url`, `latitude`, `longitude`, `address` | as the `photos` row (the trigger copies them; re-runs keep them equal) |
| `road_segment_id` | segment id (`record.id`) |
| `stage1_label`, `stage1_confidence` | VLM `stage1_label`, `stage1_confidence` |
| `is_distressed` | `stage1_label == "Distress"` |
| `distress_types` | VLM `distress_types` in the model's order; `[]` when Stage 1 says Normal |
| `severity` | VLM `severity` normalised exactly as the PipelineWorker does (`scripts/utils.normalize_severity`, e.g. `"High."` to `High`); `None` when Stage 1 says Normal |
| `description` | VLM `description` |
| `stage2_confidence` | VLM `stage2_confidence` (field confidence, the gate's value) |
| `needs_expert_review` | VLM `needs_expert_review` |
| `status` | `expert_review` if `needs_expert_review`, else `classified` |
| `image_width`, `image_height` | `frame.width`, `frame.height` (full resolution; the VLM resizes) |
| `processing_time_ms` | VLM `processing_time_ms` |
| `model_version` | the answer's model id if it has one; otherwise the VLM server's `MODEL_PATH` and `PROMPTS_VERSION` (e.g. `Qwen/Qwen2.5-VL-7B-Instruct prompts=v2`). `/classify` answers carry no model field; the exact model revision and prompt hash of a batch are in its `vlm_run.json` (`GET /health`) |
| `processed_at` | time of storing |
| `raw_response` | `{"source": "video_pipeline", "segment_record": <record>, "vlm_response": <answer>, "image_sha256": ..., "stage1_time_ms", "stage2_time_ms", "condition_indicators"}`; the last three are copies of answer fields the dashboard reads from `raw_response` |
| `error_message`, `claimed_at`, `claimed_by` | null |

Kept as required by HANDOFF_PLAN section 7: segment id and session id (`road_segment_id`,
`video_segments`), coordinates (segment centre in `photos`/`assessments`/`video_segments`; camera position,
target and footprint in `record`), chainage, UTC time, labels, severity, confidences, model id
(`model_version`) and prompt version (`model_version`, and the record's batch metadata).

## 7. Not mapped, and why

- **`assessments.user_id`, `reviewed_by`, `expert_*`, `reviewer_name`, `included_in_retrain`:** belong to app
  users and expert review; a video segment has no app user and starts unreviewed. Experts review it like any
  other row.
- **`assessments.pavement_filter_*`:** Stage 0 (pavement pre-filter) runs in the PipelineWorker, not in
  `/classify`; video frames are road by construction.
- **Camera pose, footprint, capture settings, quality, gate scores, survey, device, geometry, flags:** no
  matching column in the existing schema; they stay in `video_segments.record` and
  `assessments.raw_response.segment_record`, complete. The existing tables were not widened, so the schema
  stays identical to Supabase apart from migration 005.
- **The record's `vlm` block** (added by `vlm-import` to the batch's copy): the database stores the raw answer
  (`vlm_response`) it is derived from.
- **`skip` and `not_covered` segments** have no image and no VLM result, so no `photos`/`assessments` row: they
  would otherwise appear in the dashboard as unclassifiable uploads. They are in `video_segments`, which is
  the denominator for "% of surveyed length".
- **Cloudinary fields:** none; there is no Cloudinary public id for a local file. Deleting a video row from the
  operator dashboard deletes the local file first, then the rows (same order and refusal rules as for
  Cloudinary).

## 8. Checks

`verify_session(session_id, run_dir)` (run before cleanup): one `video_segments` row per sidecar record
with the same ids, every sent segment has an image path, every stored image exists and its SHA-256 matches
`image_sha256`, and every answered row's assessment exists.
