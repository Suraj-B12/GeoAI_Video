# Detection Model Plan

> **Status (22 Oct 2026): not needed for the pilot.** Detection is label-free: anomalies relative to the session's own healthy pavement ([pipeline/05](pipeline/05-detection.md)), and the VLM classifies. This plan involves manual labelling and is kept only as an optional later upgrade. The automatic middle step is now done: a linear probe on DINOv2 features, trained on the public Attain dataset (CC BY 4.0). See [pipeline/models/probe/MODEL_CARD.md](../pipeline/models/probe/MODEL_CARD.md).

The detector finds distress in the S21's oblique 4K frames, so that the pipeline can crop each distress and send it to the VLM ([pipeline/05-detection.md](pipeline/05-detection.md) covers inference). The detector only has to *find* things. The VLM decides the IRC class, and review catches the rest. So the design goal is **high recall, licence-clean, reproducible**.

## 1. Requirements

| Requirement | Target |
|---|---|
| Distress-level recall (after association, on held-out roads) | Potholes, alligator cracking, patches ≥ 90 %; linear cracks wider than ~3 mm ≥ 80 % ([pipeline/11](pipeline/11-testing-and-acceptance.md) §6) |
| Classes | The coarse taxonomy plus look-alikes ([pipeline/05](pipeline/05-detection.md) §2) |
| Output | Boxes in v1; masks for area classes in v2, needed for IRC % area |
| Input | Native-resolution tiles from the ROI (about 1280 px), never downscaled 4K |
| Throughput | About 650,000 tile inferences per hour of video in ≤ 45 min on the pipeline GPU ([pipeline/05](pipeline/05-detection.md) §5) |
| Licence | Usable in an internal or commercial product without copyleft obligations, unless a licence is bought |

## 2. Model candidates and benchmark

| Candidate | Licence | Role |
|---|---|---|
| **RF-DETR** detection N/S/M/L, segmentation (all sizes) | Apache-2.0 (XL/2XL detection: PML 1.0) | **Default.** Real-time DETR; the segmentation variant covers v2 |
| Ultralytics YOLO26 detect/segment | AGPL-3.0, or the paid Enterprise licence | Only if the Enterprise licence is bought; best tooling |
| RT-DETRv2, D-FINE | Apache-2.0 (verify at time of use) | Fallbacks if RF-DETR underperforms |

**Benchmark protocol** (Phase 2): same training data, splits, tile size and training budget for every candidate. Compare at the operating point that reaches the recall target on validation (§6):
1. false positives per km after association, which is the VLM's load;
2. distress-level recall by class and by crack width;
3. throughput on the pipeline GPU (FP16).

Choose by recall first, then false-positive load, then speed.

**Tile size:** use a resolution the chosen model supports. RF-DETR restricts its input to multiples of a fixed block size, so check the installed version's documentation. Train and infer at the same tile size.

## 3. Data

### 3.1 Public data: for pre-training only

| Dataset | Content | Licence | Mapping to our classes |
|---|---|---|---|
| RDD2022 | 47,420 images from Japan, India, Czech Republic, Norway, USA and China. D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole | CC BY-SA 4.0 | D00, D10 → `crack_linear`; D20 → `crack_area`; D40 → `pothole`. Where marking-blur or manhole labels are present, map them to `marking` / `manhole_utility`. Check the label list of the version you download |
| Attain (2025) | 2,293 images, 19,761 instances, 10 classes with severity, from phones on vehicle windshields | See its Mendeley record | Adds patching, ravelling, block cracking, manholes |

- RDD2022 already contains the RDD2020 images (Japan, India, Czech Republic). Use one or the other, not both, or images are duplicated.
- **Domain gap:** public images are mostly forward-looking dashboard views, at low resolution and longer range, where a crack is a few pixels wide. Our frames look down at about 1 mm/px. Public data teaches *what* distress looks like, not *how it looks to our camera*. That is why our own data is the core.
- CC BY-SA "share-alike" and trained weights is a legal question. Decide it before any commercial release ([PLAN §4.7](PLAN.md)).
- Public data never enters our validation or test sets.

### 3.2 Our own data: the core asset

- **Source:** `survey` sessions from the S21 pilot drives, at the production mount, settings and app version.
- **Diversity:** NH, SH, MDR and urban roads; new and old surfaces; sun, overcast and **tree shade** (the most common false-positive source in India); morning, noon and afternoon; the full speed band.
- **Size for v1:** about 5,000–8,000 labelled frames, ≥ 500 instances for each main class (`crack_linear`, `crack_area`, `pothole`, `patch`, `surface_loss`), ≥ 200 for each rare class (`bleeding`, `edge_failure`, `deformation_visible`) where they exist, and about 20 % frames with no distress (clean road, shadows, wet patches, markings, oil stains, manholes, speed breakers).
- **Frame selection:**
  - Never label consecutive frames; they are near-duplicates.
  - Sample one frame per **1.5 m of travel**, using `s` from the trajectory. That is about half the ROI's along-track extent, so each distress appears in about 2 labelled frames at different ranges, which teaches scale.
  - Each labelling batch is at least 30 % **random** frames and up to 70 % chosen by active learning (§5). Random frames keep the data unbiased and expose misses the model doesn't know it makes.

### 3.3 Labelling specification

Use **CVAT** (MIT) or **Label Studio** (Apache-2.0), self-hosted on the offline network. Show the ROI outline and a mm scale bar (from the mount calibration) in the labelling view.

| Class | Geometry | Rules | Attributes |
|---|---|---|---|
| `crack_linear` | Box (v1); polyline with width (v2) | One instance per continuous crack inside the frame. Split at branches or a direction change > 30°. Label every crack visible at 100 % zoom | `orientation` (longitudinal / transverse / diagonal / edge); `width_class` (< 3, 3–6, > 6 mm) |
| `crack_area` | Polygon around the outer envelope of the connected cracks | One polygon per connected area | `pattern` (alligator / block / map) |
| `pothole` | Polygon along the outer rim of the broken area | Include the broken edge, not the surrounding cracks (label those separately) | `water_filled` |
| `patch` | Polygon along the patch boundary | A pothole inside a patch is labelled as both | `condition` (sound / deteriorated) |
| `surface_loss` | Polygon of the affected area | | `type` (ravelling / stripping / hungry surface) |
| `bleeding` | Polygon | | |
| `edge_failure` | Polygon along the broken edge | Only where the pavement edge is visible | `type` (breaking / drop) |
| `deformation_visible` | Polygon | Only when clearly visible in 2-D | |
| `manhole_utility`, `speed_breaker`, `marking` | Box (polygon for markings) | Look-alikes. **Label them exhaustively**, like distresses | `marking`: `worn` |
| Shadows, oil, wet patches, joints, debris | Not labelled | They are background; frames containing them are hard negatives | |

General rules:
- **Exhaustive labelling:** in a labelled frame, every instance of every class inside the ROI is labelled. A missing label teaches the model that the thing is background.
- **ROI only:** nothing outside the ROI is labelled (vehicle body, far field beyond the resolution limit), and training never sees those regions (§4).
- **Minimum size:** don't label distresses smaller than 0.005 m² on the ground (about 7 × 7 cm), matching the inference filter.
- **Truncation:** label the visible part and set `truncated` if it is cut by the frame or ROI edge.
- **Guide:** a visual guide with 3 clear and 3 tricky examples per class, agreed with an IRC-trained engineer before labelling starts, versioned with the data.
- **QA:** 10 % of frames are labelled twice, independently. Per class, matched-instance F1 at IoU 0.5 should reach ≥ 0.8. Below that, the guide is ambiguous: fix the guide, not the labellers. Hold a weekly review of disagreements.
- **Test-set labels are made from scratch**, without model pre-labels. Pre-labels speed up training-set labelling but bias annotators towards what the model already finds, which would hide misses from the test.

### 3.4 Splits

- **The unit is the road section,** e.g. a 1 km stretch of one road and direction. All frames of a section, from every session and every date, go to the same split. Otherwise the same pothole, seen on another day, leaks between training and test.
- Train / val / test = 70 / 15 / 15 % of road length, stratified by road type and lighting.
- The test set, including the annotated test route ([pipeline/11](pipeline/11-testing-and-acceptance.md) §6), is frozen once created. It is used only for release evaluation, never to tune thresholds; that is what val is for.

## 4. Training

- **Tiles:** cut training tiles from the ROI at native resolution, at the inference tile size, at random positions. Include 20–30 % tiles without objects. Regions outside the ROI never appear in training, because they are not labelled.
- **Starting point:** COCO-pretrained weights → fine-tune on public road data mapped to our classes → fine-tune on our data. Also try a single mixed run with our data up-weighted, and keep whichever does better on val.
- **Class imbalance:** repeat-factor sampling for rare classes, not extreme loss weights.
- **Augmentations that match the physics:**

| Use | Avoid |
|---|---|
| Horizontal flip (mirrors left and right, which is physically plausible) | **Vertical flip** and 90° rotations: in an oblique view the top of the image is far away, so these create impossible views |
| Brightness, contrast, gamma; white-balance shifts | Mosaic, or at least switch it off for the last epochs: it destroys the link between image row and object scale that a fixed mount gives |
| Synthetic shadow overlays (tree-shade patterns) | Perspective warps or scale changes beyond what the mount allows (±15 % scale jitter is enough) |
| Motion blur along the image's vertical axis (the direction of travel), up to 2× the blur budget | Colour changes so strong that bleeding and wet patches become indistinguishable |
| Sensor noise up to ISO 1600; HEVC/JPEG re-compression at the survey bitrate | Copy-paste of distresses at a different range than they were taken from |
| Small roll rotations (±5°) | |

- **Early stopping** on validation distress-level recall at the operating point (§6), not on loss.
- **Reproducibility:** fixed seeds. `training_manifest.json` records the frame list with label version hash, splits, config, code commit, library and CUDA versions, and the GPU.

## 5. Active learning loop

1. **v0:** train on public data only. Run it on the first pilot sessions.
2. **Select frames to label** from (a) detections scoring 0.15–0.5, (b) detector–VLM disagreements, (c) reviewer rejections (false positives) and "missed distress" notes ([pipeline/10](pipeline/10-review-and-export.md) §5), and (d) at least 30 % random frames.
3. **Correct pre-labels.** For masks, SAM 2 (Apache-2.0) can assist inside CVAT. SAM 3 has its own licence and needs review before use.
4. **Retrain and compare** on the frozen val set. Expect 3–4 rounds before the targets are met.

Rough effort: 1–2 min per frame for boxes with pre-labels, 3–5 min with polygons. That is about 100–250 h of labelling for v1. Measure the real rate in the first batch and re-plan.

## 6. Evaluation

| Metric | Why |
|---|---|
| **Distress-level recall** per class: run stages 05–06 with the model on validation sessions, then match records to annotated distresses | The number that matters for the inventory |
| Recall broken down by crack `width_class`, lighting, speed band and range within the ROI | Shows where the model fails and what data to add |
| False positives per km after association | The VLM's and reviewers' workload |
| Frame-level per-class recall and precision; COCO mAP50 and mAP50-95 | For comparison with literature and between runs, not for decisions |

- **Operating point:** set `detect.min_score` on val as the highest threshold that still meets the recall targets. Report precision and false positives per km there.
- **Release gate:** test-set evaluation once per release. No class may regress by more than 2 points against the previous model card. Golden frames (T5) must pass ([pipeline/05](pipeline/05-detection.md) §7).

## 7. Versioning and model cards

`models/<name>/<version>/` holds:
- `weights` and `sha256`;
- `model_card.md`: intended use, classes, training data and licences, metrics at the operating point, limitations (e.g. not tested at night or on wet roads), known failure modes;
- `training_manifest.json` and `eval_report.json`;
- exported engines, each with its own hash and the GPU, driver and TensorRT version it was built on.

Model ID: `<arch>-<size>@<date>+sha256:<first 12 hex>`, stored with every detection ([pipeline/05](pipeline/05-detection.md) §6).

## 8. Export and runtime

- MVP: PyTorch FP16.
- For speed: ONNX → TensorRT FP16. **A TensorRT engine only runs on the GPU model and TensorRT version it was built for:** rebuild it on the pipeline PC and record the build environment.
- **Parity check** after any export: on validation tiles, matched boxes have IoU > 0.95 and score difference < 0.02, and distress-level recall changes by less than 0.5 points. INT8 only with calibration on our data and the same parity check (v2).

## 9. Segmentation (v2, for IRC areas)

v1 areas come from projected boxes, which **overestimate** area for irregular shapes (a round pothole fills about 79 % of its box). Treat v1 areas as upper bounds. IRC % area reports (Phase 3) need masks: RF-DETR segmentation (Apache-2.0), trained on the polygon labels already collected in §3.3.

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| A rare class has < 200 instances after two rounds | Merge it into its nearest family for the detector (the VLM still names it), and keep collecting |
| Tree shadows dominate false positives | More shaded hard negatives; the shadow augmentation; the VLM's `no_distress` |
| Public pre-training doesn't help (domain gap too large) | Train from COCO weights on our data only; spend the effort on labelling |
| Labeller agreement stays low for a class | Coarsen that class's definition; let the VLM make the fine distinction |
