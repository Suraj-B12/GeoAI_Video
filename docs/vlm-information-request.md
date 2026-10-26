# VLM Information Request

The extraction pipeline (designed in [pipeline/](pipeline/README.md)) cuts each pavement distress out of the survey video and sends that crop to your VLM. The VLM runs on a different machine, and the pilot is fully offline. To design the hand-off correctly, I need the information below.

Each question says **why** it matters. The most common failure in systems like this is a VLM that scores well on its own test set but is fed images that look different from what it was trained on. Most of these questions exist to prevent that.

Please answer in the template at the end (§9) and send **sample images** (§3.1). Approximate answers are fine; mark them as approximate.

---

## 0. If you can only answer six things

1. **3–5 training images per class**, exactly as the model saw them (§3.1).
2. The **exact label list** and which **IRC edition** it follows (§4.1).
3. What the model is given as **input**: image size, preprocessing and prompt (§2).
4. How it uses **GPS and timestamp**: EXIF, prompt text, or text printed on the image (§2.4).
5. **How to call it**: API, script, or folder batch (§1.3).
6. Its **accuracy per class** and the **test set** that measured it (§5).

---

## 1. Identity, access and runtime

| # | Question | Why it matters |
|---|---|---|
| 1.1 | Model name and base architecture (e.g. Qwen2.5-VL-7B, InternVL, LLaVA, PaliGemma, Florence-2, a cloud API). Parameter count. | Sets the image preprocessing rules, the context limits and the hardware it needs. |
| 1.2 | How was it adapted: prompt only, LoRA, or full fine-tune? Checkpoint ID/hash and version. Licence. | We record the exact model version beside every result, so results can be reproduced. |
| 1.3 | **How is it run?** (a) HTTP server (vLLM / TGI / Ollama / FastAPI) with its endpoint and request format; (b) a Python function; (c) a CLI or batch script; (d) a notebook. | Decides the adapter we build. Everything must stay offline. |
| 1.4 | Can the extraction PC reach the VLM machine on a **local network (no internet)**? Or must data move by **USB drive**? | If it can't, we use a folder-based batch hand-off (§8) instead of an API. |
| 1.5 | VLM machine hardware: GPU model, VRAM, OS. | Throughput planning. |
| 1.6 | Throughput: images per second, latency per image, maximum batch size, maximum concurrent requests. | A 50 km survey can produce thousands of crops. We size the batches and queue to suit. |
| 1.7 | Sampling settings: temperature, top-p, max tokens, seed. Does the same image always get the same answer? | Results must be repeatable. We run at temperature 0 unless you say otherwise. |
| 1.8 | Does anything need the internet at run time (model downloads, API keys, telemetry)? | The pilot must run with no internet. |
| 1.9 | Does the VLM machine keep copies or logs of the images it receives? | Data handling. Crops are road images, but the context frames could show people or vehicles. |

## 2. Input contract

| # | Question | Why it matters |
|---|---|---|
| 2.1 | Accepted image formats (JPEG / PNG / WebP), colour space (sRGB?), bit depth. | We'll write exactly that format. |
| 2.2 | Preferred and allowed **image sizes**. How the model resizes: fixed square resize, letterbox/pad, centre-crop, or dynamic resolution (min/max pixels)? | A crop that the model squashes or pads differently from training is a domain shift. The crop shape and size will be set to match. |
| 2.3 | One image per request, or several (e.g. tight crop **plus** a wider context view)? | We can send both if the model was trained that way. Otherwise we send one. |
| 2.4 | **GPS and timestamp: how does the model receive them?** (a) EXIF only; does your loader actually read EXIF? (b) In the prompt text, in what format? (c) As text **printed on the image**, as "GPS Map Camera"-style apps do? If so, send an exact example of the layout. (d) Not at all; the metadata just travels with the result. | We **never print text over the pavement** by default, because it hides the distress. If the model was trained on images with printed text, we would add a separate strip *below* the crop instead. This changes both the crop renderer and the expected accuracy. |
| 2.5 | The **exact prompt** (system plus user template) used in training and evaluation. Language. Can we change it? | The prompt is part of the model. We will send it byte-for-byte, versioned. |
| 2.6 | Can it use **extra context** if given: pavement type (bituminous / concrete), road class, physical scale (mm per pixel, crop width in metres), estimated distress size? Was it trained with any of these? | We know the physical size of every crop from the camera calibration. If the model can use that (e.g. for severity by crack width or pothole area), it may improve severity grading. If it wasn't trained with it, we won't send it. |
| 2.7 | Does it expect the **distress centred** with a margin? How large a margin? Or the whole road surface with the distress anywhere in it? | Sets our crop margin rule. |
| 2.8 | Does it ever need the road edge or shoulder visible (edge breaking, edge cracking)? | Some classes need extra context, so we widen the crop for them. |

## 3. Training data (the domain-gap check)

| # | Question | Why it matters |
|---|---|---|
| 3.1 | **Please send 3–5 training images per class**, exactly as fed to the model (after any cropping or resizing). | This is the single most useful item. It shows the viewpoint, scale, framing and quality we need to match. |
| 3.2 | Where did the training images come from: handheld phone photos, dashcam or vehicle-mounted video frames, top-down survey images (NSV / line-scan), drone, public datasets (RDD2020/2022, others)? | Our images are oblique 4K video frames from a vehicle at 20–50 km/h. They are slightly motion-blurred and compressed. If training used close-up handheld photos, we expect a gap and should test early. |
| 3.3 | Camera angle and distance to the pavement. Typical image size. Roughly how many mm of road one pixel covers (if known). | Tells us whether to send perspective crops or top-down rectified crops. |
| 3.4 | Images and instances per class. Regions and states, road types (NH/SH/MDR/urban), pavement types. | Class balance and coverage. Classes with little data need extra checking. |
| 3.5 | Who labelled them (IRC-trained engineers?), the labelling guideline, and the agreement between labellers if measured. | Sets how much the labels can be trusted, and lets our review team label the same way. |
| 3.6 | Lighting and weather mix: sun, overcast, tree shade, wet roads, night. | Indian roads have strong tree shadows; shadows are a common false-positive source. |
| 3.7 | Augmentations used in training (blur, JPEG compression, rotation, flips, colour jitter). | Tells us how tolerant the model is of our blur and compression. |
| 3.8 | Were "**no distress**" images (clean road, shadows, oil stains, markings, manholes, patches, joints) included? | Our detector is tuned for high recall, so the VLM will see look-alikes and must be able to reject them. |

## 4. Output contract

| # | Question | Why it matters |
|---|---|---|
| 4.1 | **The exact label list** (codes and names), each mapped to an IRC clause. Which edition: IRC:82-2015 or IRC:82-2023? Concrete pavements too (IRC:SP:83)? | The detector's coarse classes must cover every label the VLM can output (§6 of the model plan). |
| 4.2 | One label per image, or several? A crop can show alligator cracking around a pothole. | Sets the output schema and the aggregation rules. |
| 4.3 | **Severity**: does it output low/medium/high (or another scale)? By which criteria (crack width, pothole depth, % area)? | IRC severity uses physical measurements; we can provide physical scale (§2.6). |
| 4.4 | Is there a **"no distress" / "not pavement" / "uncertain"** output? | Without it, every false detection becomes a false distress. |
| 4.5 | **Confidence**: is there a score? Is it calibrated? Can the serving stack return token log-probabilities? | We route low-confidence results to human review. Self-reported confidence from a VLM is usually unreliable. |
| 4.6 | Does it also return measurements (crack width, pothole area or depth), a free-text explanation, or boxes inside the image? | Decides what we store and show reviewers. |
| 4.7 | **Output format**, with 3 real example outputs (including a wrong one, if you have one). Free text or JSON? Can it be constrained to a JSON schema? | We parse it automatically. Unparseable output must be impossible or detected. |

## 5. Accuracy and failure modes

| # | Question | Why it matters |
|---|---|---|
| 5.1 | Per-class precision, recall and F1, and a confusion matrix, on its test set. | Our baseline for spotting a domain gap. |
| 5.2 | How the test set was split from training: by image, by road, by location? | If split by image, near-duplicates leak and the scores are optimistic. |
| 5.3 | Known failure modes: shadows, wet patches, patching vs pothole, sealed vs open cracks, markings, manholes, debris. | Tells us which look-alikes to add as negatives and which classes need human review. |
| 5.4 | Any tests on blurred, compressed, low-light or small images? | Tells us how hard the capture settings must be. |

## 6. Change and ownership

| # | Question | Why it matters |
|---|---|---|
| 6.1 | Can the model be **fine-tuned on our pipeline crops** later? Is training code available, and do you have the data rights? | If the domain gap is large, a small LoRA fine-tune on pipeline crops is the standard fix. |
| 6.2 | How are new versions released and validated? Who approves a version for production use? | Every stored result carries a model version; results from different versions are never mixed silently. |
| 6.3 | Who is the technical contact for the VLM? | For the week-1 joint test. |

## 7. The week-1 joint test (proposed)

1. We record a 15–30 minute pilot drive with the S21 app.
2. We cut **about 100 distress crops** by hand from the video, using the crop rules in [pipeline/07-best-frame-and-crop.md](pipeline/07-best-frame-and-crop.md). An IRC-trained engineer labels each one.
3. You run the VLM on those crops and return its outputs.
4. We compare per class. A large gap means we change the capture (viewpoint, resolution, shutter), the crop rules, or fine-tune the VLM. Better to learn this in week 1 than after the pipeline is built.

## 8. What we propose to send (please confirm or correct)

- **Crop image:** JPEG, quality 95, 4:4:4 chroma, sRGB. Long side 336–1344 px (adjusted to your answer to 2.2). Distress centred with a class-dependent margin of about 15–25 % per side. No text on the image.
- **EXIF:** capture time (IST with `+05:30` offset, plus UTC GPS time), camera position (`GPSLatitude/Longitude`), distress position (`GPSDestLatitude/Longitude`), heading, speed, horizontal error.
- **Sidecar JSON** per crop (same fields as the EXIF plus physical scale and detector class), so nothing depends on an EXIF reader.
- **Hand-off,** one of:
  - **(a) LAN API:** one request per crop or batch; JSON response.
  - **(b) Offline folder batch:** we write `vlm_batch_<id>/requests.jsonl` plus `images/`. You run the VLM and write `responses.jsonl` with one line per request ID. Format in [pipeline/09-vlm-interface.md](pipeline/09-vlm-interface.md).
- **Expected response per crop:** `{request_id, model_id, prompt_version, raw_output, parsed: {label, severity, rationale}, label_logprobs, error}`. `raw_output` is the model's exact text, in whatever format it was trained to produce. We don't ask you to change the format ([pipeline/09-vlm-interface.md](pipeline/09-vlm-interface.md) §3–4).

## 9. Answer template

Copy, fill in, and return with the sample images.

```yaml
model:
  name: ""                    # e.g. Qwen2.5-VL-7B-Instruct + LoRA
  version_or_hash: ""
  base_architecture: ""
  parameters: ""
  adaptation: ""              # prompt-only | LoRA | full fine-tune
  licence: ""
runtime:
  how_run: ""                 # http_server | python_function | cli_batch | notebook
  endpoint_or_command: ""
  reachable_over_lan: null    # true | false
  needs_internet: null        # true | false
  gpu: ""
  images_per_second: null
  max_batch: null
  sampling: {temperature: null, top_p: null, max_tokens: null, seed: null}
  logs_inputs: null
input:
  formats: []                 # jpeg, png, webp
  preferred_size_px: ""       # e.g. "448x448" or "min_pixels=..., max_pixels=..."
  resize_method: ""           # square_resize | letterbox | center_crop | dynamic
  images_per_request: 1
  metadata_channel: ""        # exif | prompt_text | burned_in_overlay | not_used
  overlay_example_attached: false
  prompt_template: |
    (paste exactly)
  accepts_extra_context: []   # pavement_type, mm_per_px, crop_size_m, ...
  expects_centered_distress: null
  margin_fraction: null
training_data:
  sources: []                 # handheld_phone, vehicle_video, topdown_survey, drone, RDD2022, ...
  viewpoint: ""               # oblique | top-down | mixed
  typical_distance_m: null
  images_per_class: {}
  regions: []
  labellers: ""
  includes_no_distress_images: null
  augmentations: []
  sample_images_attached: false
output:
  labels:                     # code: description (IRC clause)
    - {code: "", name: "", irc_clause: ""}
  irc_edition: ""             # IRC:82-2015 | IRC:82-2023 | other
  covers_concrete_irc_sp_83: null
  multi_label: null
  severity: {provided: null, scale: [], criteria: ""}
  has_no_distress_class: null
  has_uncertain_class: null
  confidence: {provided: null, calibrated: null, logprobs_available: null}
  measurements: []
  format: ""                  # json | free_text
  example_outputs: []
evaluation:
  per_class_metrics_attached: false
  test_split_method: ""       # by_image | by_road | by_location
  known_failure_modes: []
  tested_on_blur_or_compression: null
ownership:
  can_fine_tune: null
  training_code_available: null
  versioning_process: ""
  technical_contact: ""
```
