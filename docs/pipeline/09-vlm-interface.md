# 09 · VLM Interface: Offline Folder Batch

Goal: crops go to the Qwen2.5-VL classifier on the VLM machine and results come back, with no network, nothing lost or duplicated, and every answer validated.

Implementation: [`geoai/vlm.py`](../../pipeline/geoai/vlm.py), [`tools/run_vlm_batch.py`](../../pipeline/tools/run_vlm_batch.py). VLM facts: [`vlm_handoff/vlm-information-response.md`](../../vlm_handoff/vlm-information-response.md).

## 1. What the VLM is

- Qwen2.5-VL-7B-Instruct, base weights with **no fine-tuning**, 4-bit NF4, the VLM team's "v2" IRC:82-2015 prompts. It runs on one RTX A5000 under Windows 11.
- One image at a time, about 11–18 s each (200–330 images per hour).
- Served by FastAPI: `POST /classify` (multipart field `file`) returns JSON (`stage1_label`, `stage1_confidence`, `distress_types`, `severity`, `description`, `stage2_confidence`, `needs_expert_review`, raw outputs, timings). `GET /health` reports the model, prompt version and settings. The rate limit is 10 requests per minute per client.
- Metadata is ignored: no EXIF, nothing in the prompt. It reads pixels only.
- Valid labels are the 18 IRC:82-2015 names plus `Normal`. Anything else (`N/A`, `Unknown`) means "no valid label".

## 2. Hand-off

```
pipeline PC                         USB drive                      VLM machine
geoai vlm-export RUN --batch B  -->  B/  ---------------------->   python run_vlm_batch.py B
                                     requests.jsonl                  (calls 127.0.0.1:8000/classify)
                                     images/*.png                    appends B/responses.jsonl
                                     SHA256SUMS                      records B/vlm_run.json (/health)
geoai vlm-import B RUN          <--  B/  <----------------------
```

- **Requests.** One per sent crop, plus one per full frame (de-duplicated per frame). The request ID is the record ID, deterministic and human-readable. Images are copied byte-identical, and `SHA256SUMS` covers every file.
- **Runner** (standard library only; nothing to install on the VLM machine):
  - stops if any checksum differs;
  - records `/health`;
  - sends one image at a time at no more than the rate limit, retrying on HTTP 429;
  - appends one line per image with flush and fsync;
  - can be stopped and restarted: images already answered are skipped, and a half-written last line is ignored.

  **The VLM team doesn't need to build anything for week 1:** the runner uses their existing endpoint.
- **Import:**
  - refuses a batch whose files changed;
  - skips error lines, so those images are re-sent next time;
  - flags labels outside the list, and `Normal` mixed with distress labels;
  - compares each crop's labels with its full frame's (`crop_frame_agree`), for their week-1 crop-vs-frame question;
  - writes `vlm_results_<batch>.csv` and `.geojson` beside the run.

## 3. Their recommendations, and how they are handled

| VLM team's advice | Handling |
|---|---|
| Send the full frame of every crop | `crop.full_frame = true` (default) |
| Run Stage 2 on every crop; Stage 1 misses about half of all distress | A server switch on their side; the runner records whatever the server returns |
| Sides multiples of 28, ≤ 1,048,576 px, ≥ about 324 tokens; widen with real context | [07](07-best-frame-and-crop.md) |
| Keep the road direction visible; widen for the edge classes | Upright frames; 25 % margin; band crops include the shoulder when visible |
| Don't share the A5000 with the detector | The pipeline runs on its own PC |
| Hold back concrete pavement | Not automated yet (see [pipeline/README](../../pipeline/README.md), limitations) |

## 4. Tests

[`tests/test_vlm.py`](../../pipeline/tests/test_vlm.py) runs a fake server that mimics their API. It covers export, an interrupted runner and its resume (every image answered exactly once), and import with label validation and crop-vs-frame agreement. A tampered batch is refused by both the runner and the importer.
