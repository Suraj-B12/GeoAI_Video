# VLM Information Request: Response

**System:** GeoAI pavement distress pipeline (RTX A5000 lab machine)
**Answered:** 2026-10-22, from code commit `a59eec1` and the stored evaluation results
**Technical contact:** [CONFIRM: name, phone/email]

All answers come from the code and the stored evaluation files. Approximate values are marked "≈". Lines marked **[CONFIRM]** need a decision from our side.

---

## Read this first

1. **The model is not trained.** Production runs the off-the-shelf **Qwen2.5-VL-7B-Instruct** with prompts only. There is no LoRA and no training set. The sample images are **evaluation** frames, saved exactly as the model receives them.
2. **It has never been evaluated on crops.** Every labelled result comes from whole vehicle-mounted video frames (the Attain dataset, Tehran). That is the closest public match to your capture. Production itself sees handheld phone photos from Bengaluru. The only crop result is for an auxiliary probe, scored together with the full frame. **We suggest the week-1 test send each crop's full frame too.**
3. **Stage 1 (Normal/Distress) misses half the distress on Attain.** Its recall is 51.8%, and Stage 2 never runs on an image Stage 1 calls Normal. For crops your detector has already flagged, we suggest running Stage 2 on every crop and recording Stage 1's answer instead of gating on it. This needs a small change on our side (§7, item 2).
4. **Type accuracy is modest.** The free-form type list scores macro MCC 0.111 on the Attain test split, where a constant predictor scores 0. "Longitudinal Cracking, Transverse Cracking" is the whole answer on 61% of test frames. Alligator recall is 8%. Ravelling, weathering and patches are never named on frames that have them.
5. **Severity and confidence are not validated.** The model gives one severity per image, estimated by eye against IRC:82 mm thresholds without any physical scale. It never answered "High" on 847 Attain frames, though 48% of the distressed frames carry a High annotation. The Stage 2 confidence measures how many labels the model commits to, not whether they are right.
6. **It runs offline, one image at a time, at ≈16–18 s per image** as configured on one RTX A5000. That drops to ≈11 s if per-tile probe recording is switched off. A LAN API exists. A folder-batch runner would be a small new script.

---

## §0 The six answers

| # | Question | Answer |
|---|---|---|
| 1 | Training images per class | None exist (prompt-only model). Instead: 27 Attain evaluation frames in `sample_images/` (3 for each of the 7 classes, 3 with no distress, 3 worked examples), each saved pixel-identical to what the processor feeds the model. |
| 2 | Label list and IRC edition | 18 bituminous distress types from **IRC:82-2015**, plus `Patching` as an IRC condition indicator (probe only). No concrete (IRC:SP:83) types. Full list in the template below. |
| 3 | Input | One RGB image. Aspect ratio is kept, each side is rounded to a multiple of 28, and the image is downscaled only above 1,048,576 pixels. Fixed English prompts, byte-exact in `prompts/`. |
| 4 | GPS and timestamp | Not used in any channel. No EXIF is read, not even Orientation. |
| 5 | How to call it | HTTP `POST /classify` (multipart field `file`) on port 8000 returns JSON. A folder batch is not built yet but is easy to add. |
| 6 | Accuracy per class | See §5. Measured on the Attain WS_V2.0 test split (547 frames, split by blocks of 20 consecutive frames). |

---

## Answer template (§9)

The same YAML is attached as `vlm-information-response.yaml`.

```yaml
model:
  name: "Qwen2.5-VL-7B-Instruct, base weights, no adapter, with our 'v2' IRC:82 prompts"
  version_or_hash: "Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5 | prompts v2 (Stage 2 prompt id 9da1601cfec54f76; per-file SHA-256 in prompts/SHA256SUMS.txt) | probe config sha256 c3797e02ff17 | code commit a59eec189f1b"
  base_architecture: "Qwen2.5-VL: ViT vision encoder with dynamic resolution plus Qwen2.5 7B language model (transformers Qwen2_5_VLForConditionalGeneration)"
  parameters: "8,292,166,656 (marketed as 7B; includes the vision encoder). Served as 4-bit NF4 weights with bf16 compute."
  adaptation: "prompt-only"   # an RDD2022 QLoRA adapter exists but is switched off: it stopped naming potholes on Attain (see notes 6.1)
  licence: "Apache-2.0 (model weights). Our prompts and code: no licence declared."
runtime:
  how_run: "http_server"      # FastAPI + uvicorn, one worker. The same classifier is also callable as a Python function.
  endpoint_or_command: "POST http://<vlm-host>:8000/classify (multipart/form-data, field 'file'); POST /classify/base64 {\"image\": \"<base64>\"}; GET /health for model, prompt and config"
  reachable_over_lan: null    # [CONFIRM] the server listens on 0.0.0.0:8000; reachability depends on the lab network and Windows Firewall
  needs_internet: false       # weights are in the local Hugging Face cache; run with HF_HUB_OFFLINE=1; not yet tested with the network unplugged
  gpu: "1x NVIDIA RTX A5000 24 GB (driver 596.51), Windows 11 Pro, 128 GB RAM, torch 2.5.1+cu121, transformers 4.57.6, bitsandbytes 0.44.1, SDPA attention"
  images_per_second: 0.06     # approx.: ≈16-18 s per image as configured, ≈11 s with per-tile probe recording off (≈0.09/s); summed from per-stage medians, not timed end to end
  max_batch: 1                # one image at a time; concurrent requests wait on a lock
  sampling: {temperature: 0, top_p: null, max_tokens: 200, seed: null}   # greedy decoding (do_sample=False); repetition_penalty 1.05 from the checkpoint; Stage 1 max 12 tokens, Stage 2 max 200
  logs_inputs: false          # /classify keeps no copy of the image; request lines and error tracebacks go to the server log
input:
  formats: [jpeg, png, webp, bmp, tiff]   # max 10 MB, max 4096x4096, min 32x32; converted to 8-bit RGB, no colour management
  preferred_size_px: "min_pixels=7168, max_pixels=1048576. Send at most 1,048,576 px with each side a multiple of 28, e.g. 1344x756, 1176x868, 1008x1008."
  resize_method: "dynamic"    # aspect ratio kept; sides rounded to multiples of 28; bicubic downscale only above max_pixels; no padding, no cropping
  images_per_request: 1
  metadata_channel: "not_used"
  overlay_example_attached: false
  prompt_template: |
    Not pasted inline: the Stage 2 system prompt alone is 14,892 characters.
    The byte-exact UTF-8 files, with SHA-256 sums, are in prompts/:
      stage1_system.txt + stage1_user.txt       Stage 1: answers "Normal" or "Distress"
      stage2_system.txt + stage2_user.txt       Stage 2: DISTRESS_TYPES / SEVERITY / DESCRIPTION
      probe_system.txt + probe_questions.txt    per-type Yes/No probe (20 questions, recorded only)
      rendered_*_chat.txt                       full Qwen chat-template text of each call
    Every call: system turn = the system prompt; user turn = the image, then the user text.
    Language: English. Any byte change makes a new prompt version that must be re-evaluated.
  accepts_extra_context: []   # none was used in any evaluation; adding some is a prompt change (see notes 2.6)
  expects_centered_distress: false   # evaluated on whole frames and photos, with the distress anywhere
  margin_fraction: null       # never evaluated on crops
training_data:
  sources: []                 # no training. Evaluated on Attain WS_V2.0 (vehicle-mounted smartphone video frames) and Bengaluru citizen phone photos (unlabelled)
  viewpoint: "oblique"        # Attain: forward-facing through the windshield. Bengaluru: handheld close-ups.
  typical_distance_m: null    # not recorded in either source
  images_per_class: {}        # no training; evaluation counts per class are in notes 3.4
  regions: []                 # no training; evaluation: Tehran (Attain), Bengaluru (production, unlabelled)
  labellers: "No training labels. Attain labels are by the dataset's authors; their guideline and agreement were not reported to us. We mapped the Attain classes to IRC:82 names."
  includes_no_distress_images: null  # no training; the evaluation includes 78 Attain frames with no distress annotation
  augmentations: []
  sample_images_attached: true       # evaluation frames, NOT training images, saved exactly as the model receives them
output:
  labels:                     # what Stage 2 may put in distress_types
    - {code: "7.2.1", name: "Bleeding", irc_clause: "IRC:82-2015 §7.2.1", severity_scale: "Low/Medium/High"}
    - {code: "7.2.3", name: "Streaking", irc_clause: "IRC:82-2015 §7.2.3", severity_scale: "N/A"}
    - {code: "7.2.4", name: "Hungry Surface", irc_clause: "IRC:82-2015 §7.2.4", severity_scale: "N/A"}
    - {code: "7.3.2", name: "Hairline Cracks", irc_clause: "IRC:82-2015 §7.3.2", severity_scale: "N/A"}
    - {code: "7.3.3", name: "Alligator Cracking", irc_clause: "IRC:82-2015 §7.3.3", severity_scale: "Low/Medium/High"}
    - {code: "7.3.4", name: "Longitudinal Cracking", irc_clause: "IRC:82-2015 §7.3.4", severity_scale: "Low/Medium/High"}
    - {code: "7.3.5", name: "Transverse Cracking", irc_clause: "IRC:82-2015 §7.3.5 (includes block cracking, §7.3.5.1)", severity_scale: "Low/Medium/High"}
    - {code: "7.3.6", name: "Edge Cracking", irc_clause: "IRC:82-2015 §7.3.6", severity_scale: "Low/Medium/High"}
    - {code: "7.4.1", name: "Slippage", irc_clause: "IRC:82-2015 §7.4.1", severity_scale: "Low/High"}
    - {code: "7.4.2", name: "Rutting", irc_clause: "IRC:82-2015 §7.4.2", severity_scale: "Low/High"}
    - {code: "7.4.3", name: "Corrugation", irc_clause: "IRC:82-2015 §7.4.3", severity_scale: "N/A"}
    - {code: "7.4.4", name: "Shoving", irc_clause: "IRC:82-2015 §7.4.4", severity_scale: "N/A"}
    - {code: "7.4.5", name: "Shallow Depression", irc_clause: "IRC:82-2015 §7.4.5", severity_scale: "N/A"}
    - {code: "7.4.6", name: "Settlement", irc_clause: "IRC:82-2015 §7.4.6", severity_scale: "N/A"}
    - {code: "7.5.1", name: "Stripping", irc_clause: "IRC:82-2015 §7.5.1", severity_scale: "N/A"}
    - {code: "7.5.2", name: "Ravelling", irc_clause: "IRC:82-2015 §7.5.2", severity_scale: "Low/Medium/High"}
    - {code: "7.5.3", name: "Potholes", irc_clause: "IRC:82-2015 §7.5.3", severity_scale: "Small/Medium/Large"}
    - {code: "7.5.4", name: "Edge Breaking", irc_clause: "IRC:82-2015 §7.5.4", severity_scale: "N/A"}
    - {code: "CI-PATCH", name: "Patching", irc_clause: "IRC:82-2015 §5.1, Tables 5.1-5.3 (condition indicator, not a distress)", severity_scale: "N/A"}   # per-type probe only; not in the Stage 2 list
    - {code: "NORMAL", name: "Normal", irc_clause: "-", severity_scale: "N/A"}   # Stage 1 'Normal', or Stage 2 'Normal - No distress detected'
  irc_edition: "IRC:82-2015"  # First Revision, June 2015; not checked against IRC:82-2023
  covers_concrete_irc_sp_83: false
  multi_label: true           # comma-separated list in the model's own order; the first label is NOT the dominant one (see notes 4.2)
  severity: {provided: true, scale: ["Low/Medium/High", "Low/High (Slippage, Rutting)", "Small/Medium/Large (Potholes)", "N/A (9 types)"], criteria: "IRC:82-2015 thresholds (crack width 1-3 / 3-6 / >6 mm; pothole depth 25 / 50 mm and width 200 / 500 mm; rut depth 4-10 / >10 mm) estimated by eye from reference objects. One severity per image, not per label. The API maps it to None/Low/Medium/High/Unknown (Small->Low, Large->High, N/A->None). Not validated."}
  has_no_distress_class: true
  has_uncertain_class: false  # no label; a needs_expert_review flag is set below 0.80 confidence instead
  confidence: {provided: true, calibrated: false, logprobs_available: true}   # all scores come from logits; per-type probe P(yes) is Platt-calibrated on Attain only
  measurements: []            # no widths, areas or boxes; mm values in DESCRIPTION are visual guesses
  format: "free_text"         # three fixed lines parsed server-side into JSON; not grammar-constrained
  example_outputs:            # real stored outputs at production settings; full records in example_outputs.json
    - image: Attain_SMP_WS_v2_000536.jpg   # right labels: Attain has Linear + Alligator crack
      stage1: "Distress (0.8411)"
      stage2_raw: "DISTRESS_TYPES: Alligator Cracking, Longitudinal Cracking\nSEVERITY: Medium\nDESCRIPTION: Interconnected web-pattern cracks and longitudinal cracks along the wheel path indicate moderate structural damage (IRC:82 §7.3.3, §7.3.4)."
      stage2_confidence: 0.8648
      decision: "auto-accepted"
    - image: Attain_SMP_WS_v2_000020.jpg   # WRONG: weathering is 97% of the annotated area and a pothole is present
      stage1: "Distress (0.8833)"
      stage2_raw: "DISTRESS_TYPES: Longitudinal Cracking, Transverse Cracking\nSEVERITY: Medium\nDESCRIPTION: Multiple interconnected cracks, both longitudinal and transverse, with widths ranging from 3-6 mm, indicating moderate distress (IRC:82 §7.3.4, §7.3.5)."
      stage2_confidence: 0.8835
      decision: "auto-accepted (wrong, not sent to review)"
    - image: Attain_SMP_WS_v2_000490.jpg   # WRONG: Attain has Linear crack + a small pothole
      stage1: "Normal (0.9567)"
      stage2_raw: "(not run in production; forced for the record: DISTRESS_TYPES: Normal - No distress detected)"
      stage2_confidence: null
      decision: "auto-accepted as Normal; the safety-net probe stayed below its threshold (0.090 < 0.149)"
evaluation:
  per_class_metrics_attached: true   # tables in notes §5
  test_split_method: "by_block_of_20_consecutive_frames"   # closest to by_road; Attain has no road or location IDs. Also grouped 5-fold CV over the 43 blocks.
  known_failure_modes:
    - "Stage 1 calls 48% of Attain frames that carry annotated distress Normal (recall 51.8%), and Stage 2 then never runs. A probe safety net sends 313 of those 371 frames to review."
    - "The free-form list collapses to 'Longitudinal Cracking, Transverse Cracking' (61% of Attain test frames). Longitudinal Cracking is listed first on 90% of frames that get a type."
    - "Alligator cracking recall is 8%. Ravelling, weathering (Hungry Surface) and patches are never named on Attain frames that have them."
    - "Shadows, wet surfaces and repair patches get called 'Bleeding' on Bengaluru close-ups: 52 of 204 uploads were labelled Bleeding, and the 4 we inspected showed no bitumen film. This is an inspection, not ground truth."
    - "Severity is never 'High' on 847 Attain frames, although 368 of the 769 distressed frames carry a High annotation."
    - "Imperceptible input changes flip labels: re-encoding as JPEG quality 95 changed the Stage 2 label set on 9 of 49 phone photos."
    - "The Stage 2 confidence falls as more labels are named. It tracks over-prediction, not correctness."
    - "When Stage 2 runs on a frame Stage 1 called Normal, the DISTRESS_TYPES value can be 'N/A' or 'None'. The parser returns these as the labels 'N/A' and 'Unknown'."
    - "Concrete pavement has no labels; the taxonomy is bituminous only."
    - "Lane markings, manholes, oil stains and joints are excluded only by prompt text. There is no measured false-positive rate because there are no annotated look-alike negatives."
  tested_on_blur_or_compression: true   # JPEG q95/q75 and resolution caps down to 512x512: yes. Blur, motion blur, low light: no.
ownership:
  can_fine_tune: true          # QLoRA pipeline exists (LLaMA-Factory); a 43 h run on this A5000 has been done before
  training_code_available: true
  versioning_process: "Production config is pinned in the server .env (model, prompt version, quantization, pixel cap, probe config). Prompts are versioned by hash. Every model or prompt change must pass a smoke test and a per-class A/B on Attain before release. Probe changes follow pre-registered dev/test rules. Approver: [CONFIRM]."
  technical_contact: ""        # [CONFIRM]
```

---

## Notes by question

### 1. Identity, access and runtime

**1.1–1.2 Model and version.** We run the base model at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, which is the current `main` on Hugging Face. It has 8,292,166,656 parameters and is licensed Apache-2.0. It is served as 4-bit NF4 weights with double quantization and bf16 compute. The prompts are our "v2" set, and the Stage 2 prompt id `9da1601cfec54f76` matches the id recorded by the evaluation runs below. So the published numbers were measured with today's prompt. An RDD2022 QLoRA adapter exists but is off; see 6.1 for why.

**1.3 How it runs.** FastAPI under uvicorn, one worker:

```
POST /classify          multipart/form-data, field "file"          -> JSON
POST /classify/base64   {"image": "<base64>"}                     -> JSON
POST /classify/stream   same as /classify, as server-sent events
GET  /health            model, prompt version, quantization, pixel cap, probe status
```

API keys are optional (`X-API-Key`, off unless configured). Rate limit: 10 requests per minute per client on the classify endpoints. Per image, uploads are limited to 10 MB and to sizes between 32×32 and 4096×4096. The response contains `stage1_label`, `stage1_confidence`, `distress_types`, `severity`, `description`, `stage2_confidence`, `needs_expert_review`, both raw model outputs and timings.

**1.4 LAN or USB.** [CONFIRM] The server listens on `0.0.0.0:8000`, so any PC on the same LAN can call it if Windows Firewall allows the port. For a fully offline pilot we would rather use the folder batch (§8): no network and no rate limit.

**1.5 Hardware.** One RTX A5000 (24 GB, WDDM driver 596.51), Windows 11 Pro, 128 GB RAM. Peak reserved GPU memory during the full Attain run was 18.3 GB. Please do not run your detector on this GPU at the same time. When Windows over-commits GPU memory it pages to system RAM instead of failing, and inference silently runs about 25× slower.

**1.6 Throughput (approx.).** Per-stage medians at the production pixel cap:

| Step | Median |
|---|---|
| Stage 1 (Normal/Distress) | 1.3 s |
| Stage 2 (type list, generated) | 8.2 s, about the same for 644×644 and 1344×756 inputs |
| Per-type probe, full image | 1.1–1.6 s |
| Per-tile probe scores (recorded, not used) | 5.0–6.5 s |

That sums to ≈16–18 s per image as configured, or ≈11 s with tile recording off. It means roughly 200 images per hour, or 330 per hour without tiles: a 3,000-crop survey takes ≈9–15 h. Images run one at a time on a single GPU. Smaller inputs speed up Stage 1 (median 1.0 s at 644×644 against 1.5 s at 1344×756), but not Stage 2, which spends its ≈7–8 s generating text at every frame size we measured.

**1.7 Determinism.** Greedy decoding (`do_sample=False`, so effectively temperature 0, no seed needed). Stage 1 generates at most 12 tokens and Stage 2 at most 200. The checkpoint's `repetition_penalty` of 1.05 still applies. Identical bytes on the same machine and software should give identical output, but we have not formally re-tested this. **Imperceptible changes do flip answers:** re-encoding photos as JPEG quality 95 changed the Stage 2 label set on 9 of 49 Bengaluru photos (mean Jaccard 0.88), with Stage 1 unchanged on all 57. At quality 75, 16 of 49 changed and Stage 1 changed on 2 of 57.

**1.8 Internet.** Not needed for inference. The weights are in the local cache. Set `HF_HUB_OFFLINE=1`; the loader's only online call (a parameter-count lookup) is optional and skipped on failure. The citizen-app path (Supabase, Cloudinary, Cloudflare tunnel) is separate, and `/classify` does not need it. We have not yet tested with the network unplugged. That should be done before week 1.

**1.9 Copies and logs.** `/classify` keeps no copy of the image or the result. The server log records request lines (client IP, path, status), error tracebacks and any unparseable Stage 1 answer. A folder batch would keep only what is in the batch folder. [CONFIRM retention policy.] The attached Attain samples show vehicles with readable number plates; they come from a public CC BY 4.0 dataset.

### 2. Input contract

**2.1 Format.** JPEG, PNG, WebP, BMP and TIFF are accepted. The image is converted to 8-bit RGB with no colour management (ICC profiles are ignored), so sRGB is right.

**2.2 Size and resizing.** The processor keeps the aspect ratio and rounds each side to a multiple of 28. It downscales (bicubic) only when the image has more than 1,048,576 pixels. There is no padding and no cropping. Measured with the production processor:

| You send | Model receives | Visual tokens |
|---|---|---|
| 3840×2160 or 1920×1080 | 1344×756 | 1,296 |
| 1344×896 | 1232×812 | 1,276 |
| 1479×508 (Attain) | 1484×504 | 954 |
| 1024×683 | 1036×672 | 888 |
| 640×640 (Attain) | 644×644 | 529 |
| 672×448 | 672×448 | 384 |
| 336×224 | 336×224 | 96 |

Our evidence covers inputs from about 324 to 1,300 visual tokens. Lowering the cap from full resolution to 1024² made no significant accuracy difference on 200 labelled Attain frames, and neither did caps down to 512². Below that, and for anything near 96 tokens, we have no data.

**2.3 Images per request.** One. A crop-plus-context pair in one prompt was never tried and would need its own evaluation. Sending the context frame as a separate request works today.

**2.4 GPS and time.** Option (d): not used at all. The model receives only pixels and fixed prompt text; no EXIF is read. We agree with no text on the image. Please keep the metadata in your sidecar; we return results keyed by `request_id`.

**2.5 Prompt.** The prompt files are attached byte-exact, with hashes. Note that the prompts use a fictional senior-engineer persona and high-stakes framing. That is a deliberate prompt-engineering choice: it is part of the v2 set, which beat our plain prompts on Attain, though the persona was never tested on its own. The Stage 2 prompt asks for **all** distress types visible, so a crop centred on one distress can still get several labels.

**2.6 Extra context.** None was used in training or evaluation. The prompt tells the model to judge size from reference objects in the photo (a coin ≈ 25 mm, a tyre ≈ 250 mm, a lane ≈ 3.5 m). Tight crops usually contain none of these, so severity on crops will be even less grounded than on full frames. Your physical scale (mm per pixel, crop width in metres) is the obvious fix and worth a second arm in week 1. It is a prompt change, so it would get a new version and an evaluation before any use.

**2.7 Centring and margin.** We have no margin rule. The model was evaluated on whole frames and phone photos, with the distress anywhere in them. The only crop result is for the per-type probe (shadow mode, not the labels production reports). That probe was tried on crops around the *annotated* damage: each box was widened by 20% and to at least a quarter of the short side, then upscaled up to 2×. The best crop's score was averaged with the full-frame score. Against the full frame alone:

- On the test split, macro AUROC was +0.008 [−0.019, +0.039], so no clear gain.
- Cross-validated macro MCC was +0.078 [+0.039, +0.117].

Stage 1 and Stage 2 have never been run on crops. For the probe, only this combined crop-plus-frame score was carried to the test split; we have no tested result for a crop on its own.

**2.8 Context for some classes.** Yes. Several classes are defined relative to the pavement edge or the road direction:

- **Edge Cracking** (§7.3.6, within 0.3–0.5 m of the edge) and **Edge Breaking** (§7.5.4, within 30 cm) need the pavement edge in view.
- **Longitudinal vs Transverse** cracking is decided by the road direction.
- **Rutting** follows the wheel path; **Corrugation** runs across the lane.

So please keep the camera orientation (road running up the image) and enough lane context to show the direction. A tight crack crop with no edge or lane marking loses it.

### 3. Training data (domain-gap check)

**3.1 Samples.** There is no training set to sample, so `sample_images/` holds Attain evaluation frames instead:

- 3 per class, choosing frames where that class has the largest annotated area, from different road stretches, mixing the three frame sizes where the data allows.
- 3 frames with no distress annotation.
- The 3 worked examples.

Each PNG is the exact pixel array the production processor normalises; we checked every one against the processor. `sample_images/manifest.csv` gives each frame's Attain annotations, our Stage 1 answer, the production decision and the Stage 2 text.

**3.2 Sources.** No training. Evaluation used:

- **Attain WS_V2.0:** 847 frames from smartphones on a vehicle windshield, Amirkabir University of Technology, Tehran (CC BY 4.0, doi:10.17632/nykrzdm74f/1). Frame sizes: 1479×508 (403), 1920×1080 (275), 640×640 (169). This is the closest match to your capture, but these are full frames, not crops. Motion blur and compression levels are not documented.
- **Bengaluru RoadSide uploads:** about 230 handheld phone close-ups as of late September, typically 1512–1728 px square. This is production's real input, but there are no labels.
- Older Stage 1 results on GAPs (top-down survey patches) and Stage 2 results on RDD2022 (dashcam) used an earlier prompt set and do not describe this model.

**3.3 Viewpoint and scale.** Attain is forward-facing and oblique, from windshield height. Distance and mm per pixel are not recorded, and in an oblique frame scale varies strongly with depth. Bengaluru photos are handheld and oblique at close range. Neither source is top-down. We suggest sending perspective crops, not rectified ones, because that is what the model has seen.

**3.4 Per-class counts (Attain, 847 frames; frames containing the class).** Linear crack 687, Alligator crack 561, Weathering 167, Patch and utility cut 167, Pothole 163, Raveling 107, Block crack 32. 78 frames have no distress annotation. We map Attain classes to our labels as follows:

| Attain class | Our label |
|---|---|
| Linear crack | Longitudinal or Transverse Cracking |
| Block crack | Transverse Cracking (IRC §7.3.5.1) |
| Weathering | Hungry Surface |
| Raveling | Ravelling |
| Patch and utility cut | Patching (condition indicator) |

**3.5 Labellers.** Attain was labelled by the dataset's authors. We have no guideline or agreement figures, and we don't know whether they followed IRC or any other manual. The class mapping above is ours. No expert-labelled Bengaluru set exists yet. [CONFIRM who will label; we would like to label the same way as your engineer.]

**3.6 Lighting and weather.** Not catalogued for either source. Tree shadow and wet surfaces already show up in failure modes (§5.3).

**3.7 Augmentations.** None; nothing was trained.

**3.8 No-distress images.** The 78 Attain frames without distress annotation are our only negatives, and "not annotated" does not prove the road is clean. The prompts explicitly exclude shadows, lane markings, wet patches, oil stains, skid marks, manholes, drainage grates, intact joints and debris. A false-positive rate on such look-alikes has never been measured.

### 4. Output contract

**4.1 Labels.** See the template. They follow IRC:82-2015, First Revision (we have not checked IRC:82-2023), and cover bituminous surfaces only. The Stage 0 filter accepts concrete as "pavement", but no concrete distress label exists, so concrete crops would get bituminous labels. Please tag pavement type in your detector and hold concrete back. Two of IRC's §7 types are excluded on purpose because a photo alone cannot settle them: Smooth Surface (§7.2.2, needs a skid measurement) and Reflection Cracking (§7.3.7, needs knowledge of the layer below).

**4.2 Several labels per image.** Yes, as a comma-separated list. The order is the model's listing order, not dominance: on Attain, Longitudinal Cracking comes first on 90% of frames that get a type, which mirrors step 1 of the prompt. A prompt rule asking for "most prominent first" was tested and rejected (it added false positives, p = 0.021). Severity is one value for the whole image, not one per label.

**4.3 Severity.** The scale depends on the type:

- Low/Medium/High for most types
- Low/High only for Slippage and Rutting
- Small/Medium/Large for Potholes (IRC §7.5.3.4)
- N/A for 9 types

The criteria are IRC's mm thresholds, judged by eye. The prompt tells the model to pick the higher tier when unsure. The API maps the answer to None/Low/Medium/High/Unknown; the raw IRC tier stays in `stage2_raw`. **Severity has never been validated:** Attain severity is High/Low only, and the model never answered High there.

**4.4 No-distress, not-pavement and uncertain outputs.**

- **No distress:** yes. Stage 1 answers "Normal", and Stage 2 can answer "Normal - No distress detected".
- **Not pavement:** a Stage 0 filter (YES/NO/UNSURE; only a clear NO rejects) exists in the citizen-app worker but not on `/classify`. Your crops shouldn't need it.
- **Uncertain:** there is no uncertain label. `needs_expert_review` is set when Stage 1 or Stage 2 confidence is below 0.80, or when the safety net fires. The Stage 2 gate alone flags 26% of Attain test frames and ≈60% of the Bengaluru uploads that reach Stage 2.

**4.5 Confidence.** All scores come from model logits, not from self-reported confidence. None is calibrated except the probe, and that only on Attain.

| Score | How it is computed | What it is worth |
|---|---|---|
| Stage 1 | softmax over the first-token logits of "Normal" vs "Distress" | Ranks frames well (AUROC 0.957 on Attain test, but only 23 negatives), yet the 0.80 gate does not fix the Normal bias |
| Stage 2, used for the review gate | geometric mean of token probabilities over the DISTRESS_TYPES value | Falls as more labels are named. Attain test AUROC: 0.581 (no false positive), 0.568 (Jaccard ≥ 0.5), 0.346 (exact label set). Of frames it auto-accepts, 85.7% have no false positive and 7.6% have the exact label set. |
| Per-label joint probability | probability of each listed label | Recorded. Tested as a gate and rejected (AUROC 0.468). |
| Probe P(yes), 20 types | Yes vs No next-token probability, one question per type | Platt-calibrated on Attain (out-of-fold ECE 0.076). Its thresholds do not transfer to Bengaluru, so it is recorded, not used for labels. |

Log-probabilities are available. `/classify` currently returns the Stage 1 score, the Stage 2 gate score and the first-label score. Per-label and per-type probabilities are computed but not yet in the response schema. That is a small change, and the folder batch would include them.

**4.6 Measurements.** None: no crack widths, areas or boxes. DESCRIPTION is one sentence and may quote mm values; these are the model's visual guesses. Qwen can draw boxes, but in our test its boxes contained only 44% of annotated damage centres, and it gave no usable box on 18% of frames. It is not used.

**4.7 Format.** The model writes three lines (`DISTRESS_TYPES:`, `SEVERITY:`, `DESCRIPTION:`), which the server parses into JSON. Decoding is not constrained to a JSON schema. On 847 Attain frames, all three lines were always present. Three things to handle:

- When Stage 2 was run on the 449 frames Stage 1 had called Normal, its DISTRESS_TYPES value was "N/A" 119 times, "None" 14 times and "Normal - No distress detected" 68 times. The parser turns these into the labels `N/A`, `Unknown` and `Normal`.
- In production this has not happened yet, because Stage 2 only runs after Stage 1 says Distress (0 of 398 cases). It will happen if Stage 1 stops gating Stage 2.
- **Treat any label outside the 18 names and `Normal` as "no valid label, send to review".**

Three real outputs are in the template, and the full records are in `example_outputs.json`.

### 5. Accuracy and failure modes

**5.1 Per class.** These are the Attain WS_V2.0 test split results (547 frames, 23 without distress), with Stage 2 run on every frame. MCC is 0 for any constant predictor; a constant "Linear + Alligator" answer scores macro F1 0.346 here, so F1 alone flatters.

| Attain class → our label | Prevalence | Production list P / R | MCC | Probe (shadow) P / R | MCC [95% CI] |
|---|---:|---:|---:|---:|---:|
| Linear crack → Longitudinal / Transverse | 83.9% | 0.89 / 0.82 | 0.257 | 0.87 / 0.97 | 0.358 [0.147, 0.529] |
| Alligator crack → Alligator Cracking | 69.1% | 0.89 / 0.08 | 0.114 | 0.74 / 0.97 | 0.336 [0.196, 0.456] |
| Pothole → Potholes | 18.5% | 0.43 / 0.25 | 0.219 | 0.37 / 0.40 | 0.234 [0.032, 0.412] |
| Raveling → Ravelling | 12.8% | 0.00 / 0.00 | 0.000 | 0.09 / 0.14 | −0.063 [−0.164, 0.043] |
| Weathering → Hungry Surface | 26.3% | 0.00 / 0.00 | −0.036 | 0.40 / 0.58 | 0.249 [−0.040, 0.472] |
| Patch and utility cut → Patching | 19.4% | no such label | — | 0.40 / 0.32 | 0.224 [0.035, 0.442] |
| Block crack → Transverse (block form) | 3.3% | not separable | — | 0.07 / 0.72 | 0.138 [−0.052, 0.285] |
| **Macro MCC, first five rows** | | | **0.111** | | **0.223** |

Grouped 5-fold cross-validation over all 847 frames gives macro MCC 0.125 for the production list and 0.241 for the probe. Confidence intervals are cluster-bootstrapped over blocks of 20 frames.

Stage 1 on the same test split:

| | Value |
|---|---|
| Recall of distress | 51.7% |
| Specificity | 100% (only 23 negatives) |
| MCC | 0.208 |
| AUROC | 0.957 |

With Stage 1 in front, end-to-end macro MCC is 0.094. Because the task is multi-label, we report per-class P/R/MCC instead of a single confusion matrix. Per-class 2×2 counts are available on request.

**5.2 Test split.** The test split is built from blocks of 20 consecutive video frames, which keeps near-duplicate neighbouring frames on the same side except at block edges. It has 300 dev and 547 test frames, under pre-registered rules; the cross-validation is grouped by the same blocks. The base model was never trained, so there is no training leakage. The probe's thresholds were tuned on dev or out-of-fold. However, the same Attain frames have been used across several prompt and pipeline iterations, so they are not a pristine hold-out for the prompt wording.

**5.3 Known failure modes.** These are listed in the template; here is what lies behind them.

- **Shadows, wet surfaces and patches:** "Bleeding" on Bengaluru close-ups. The 4 uploads we inspected showed tree shadow, loose aggregate, repair patches and a wet night-time surface.
- **Patching versus potholes:** no measurement.
- **Sealed versus open cracks:** no measurement.
- **Markings, manholes and debris:** excluded only in the prompt text, with no measurement.
- **Small damage in wide frames:** the main Stage 1 miss. See example 3, where a small pothole and fine cracks sit in a 1920×1080 traffic scene.

**5.4 Blur, compression, low light and small images.**

- JPEG compression was tested (q95 and q75, numbers in 1.7).
- Resolution caps down to 512×512 made no significant accuracy difference on 200 Attain frames. Stage 1 agreement on Bengaluru photos fell to 89.5% at 512².
- Motion blur, low light and very small inputs have not been tested. Your 20–50 km/h motion blur is therefore an open question for week 1.

### 6. Change and ownership

**6.1 Fine-tuning.** Fine-tuning is possible. There is a QLoRA pipeline (LLaMA-Factory), the A5000 has run it (43 h for RDD2022), and the code is in our repository. The base weights are Apache-2.0, and the data rights to your crops are yours.

A caution from our own result: the adapter we trained on RDD2022's four classes stopped naming potholes on Attain (0 of 769 frames, 163 of which have potholes). It also lost every class it was not trained on, and it named more than one type on 16.9% of frames, against 52.5% for the prompt-only model. Any fine-tune on pipeline crops therefore has to cover every label and be compared per class with the prompt-only model before use.

**6.2 Versions.** The production configuration is pinned in the server's `.env` and reported by `/health`. Every model or prompt change must pass a load-time smoke test and a per-class A/B on Attain before release. Probe changes follow pre-registered dev/test rules.

What is missing today: results do not carry an explicit model-version stamp. For the pilot, every response would be stamped with the model revision, quantization, pixel cap and prompt hashes; that is a small change we have not made yet. [CONFIRM who approves a version for production.]

**6.3 Contact.** [CONFIRM]

---

## §7 Week-1 joint test: agreed, with additions

1. **Send the full source frame for every crop** as a separate request, for example `<id>-frame`. All our evidence is full-frame, so this measures the crop effect directly.
2. **Run Stage 2 on every crop and record Stage 1 instead of gating on it.** If Stage 1 behaves on crops as it does on Attain frames, about half of the real distress crops would otherwise never be typed. This needs a switch we have not built yet.
3. **Add about 20 look-alike negatives:** tree shadow, wet patch, oil stain, lane marking, manhole, repair patch and sealed joint. Our false-positive behaviour on these is unmeasured.
4. **Have the engineer record a measurement with severity where possible**, such as crack width in mm or pothole depth and width. Our severity has never been validated.
5. **Optional second arm:** the same crops with physical scale in the prompt. This is a new prompt version, run alongside the current one, not instead of it.
6. **What we return per request:** both raw outputs, the parsed labels, severity and description, every confidence score, the 20 probe probabilities, timings, model revision and prompt hashes.
7. **Scoring:** per-class MCC beside a constant baseline, plus precision and recall, with confidence intervals grouped by road stretch. 100 crops will give wide intervals. That is enough to see a large gap, not to set thresholds.

---

## §8 Your proposal: confirmed or corrected

| Item | Your proposal | Our answer |
|---|---|---|
| Crop image | JPEG q95, 4:4:4, sRGB | **OK.** Pixels are used as-is. |
| Crop size | long side 336–1344 px | **Correct to:** at most 1,048,576 px in total, each side a multiple of 28 (e.g. 1344×756, 1176×868, 1008×1008). Larger images are downscaled. Other sizes are resampled slightly. 336×224 gives the model 96 visual tokens, far below anything we have tested (≈324–1,300). If a crop would be small, prefer widening it with real context over upscaling it. |
| Framing | centred, 15–25% margin per side | **No evidence either way.** Keep your rule for week 1 and add the full frame. Widen for the edge classes and keep the road direction visible (notes 2.8). |
| Text on image | none | **Agreed.** |
| EXIF | time, GPS, heading, speed, error | **Fine, but ignored.** Our loader reads no EXIF at all, not even Orientation. Store pixels upright (Orientation = 1). |
| Sidecar JSON | per crop | **Fine.** The model does not read it, and responses echo `request_id`. |
| Hand-off (a) LAN API | one request per crop | **Exists** (notes 1.3). Limits: 10 requests per minute per client, one image at a time. |
| Hand-off (b) folder batch | `requests.jsonl` + `images/` → `responses.jsonl` | **Not built yet. We prefer it for the offline pilot.** It would be a small runner around the same classifier code, with no server, no network and no rate limit. [CONFIRM we build it before week 1.] |
| Response | `{request_id, model_id, prompt_version, raw_output, parsed:{label, severity, rationale}, label_logprobs, error}` | **Yes, with these differences.** `raw_output` holds two texts (Stage 1 and Stage 2). `parsed.label` is a list. `severity` is one per crop. `rationale` is DESCRIPTION. `label_logprobs` holds the Stage 1 probability, the per-label probabilities and the 20 probe P(yes). `model_id` = `Qwen/Qwen2.5-VL-7B-Instruct@cc594898…/nf4/max_pixels=1048576`. `prompt_version` = `v2:9da1601cfec54f76`. |

---

## Attachments

| Path | Contents |
|---|---|
| `vlm-information-response.yaml` | The template above, as a file |
| `prompts/` | Byte-exact prompt files, rendered chat templates, `SHA256SUMS.txt` |
| `sample_images/` | 27 PNGs as the model sees them, plus `manifest.csv` and `ATTRIBUTION.txt` |
| `example_outputs.json` | The three example records, with every stored score |

## Evidence (in our repository)

Accuracy comes from `eval_results/stage2_probe_report.md`, `stage2_probe_cv.md` and `stage2_views_report.json`, plus `stage2_probe_raw_attain.json` (all 847 raw outputs, production settings). Resolution and JPEG results are in `eval_results/resolution_ab.json`. The adapter comparison is `eval_results/attain_3way_comparison.md`. The label list is generated from `scripts/irc82_taxonomy.py`, and the prompts from `scripts/utils_v2_prompts.py`.
