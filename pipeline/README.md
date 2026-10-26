# GeoAI extraction pipeline (offline)

Turns GeoAI Capture sessions into crops of road surface that doesn't look like healthy pavement. Each crop carries the exact time and GPS position and is sized for the Qwen2.5-VL classifier on the VLM machine. The pipeline only finds and crops; the VLM classifies.

**No labelled data and no manual steps.** "Healthy" is learned from the session itself, and everything is checked by automated tests.

```
phone --USB--> session --process--> crops + frames + records --vlm-export--> USB batch
                                                                               |
         results.csv / .geojson <--vlm-import-- responses.jsonl <--run_vlm_batch.py (VLM machine)
```

## Install (offline machine)

On a connected machine:
```bash
pip download -r requirements.txt -d wheelhouse
```
For the GPU build of torch, download the CUDA wheel matching the pipeline PC from download.pytorch.org into the same folder. Copy `pipeline/` and `wheelhouse/` to the pipeline PC, then:
```bash
pip install --no-index --find-links wheelhouse -r requirements.txt
```
```bash
pip install --no-index --no-deps -e .
```
- The DINOv2 weights are already in `models/dinov2/`: Meta's ViT-S/14 with registers, 88 MB, Apache-2.0. Their SHA-256 is checked on load.
- PyAV bundles FFmpeg; `ffprobe` is only needed by `../tools/validate_session.py`.
- Nothing contacts the internet at run time.

## Use

```bash
python ../tools/pull_sessions.py D:/geoai/sessions
```
```bash
python -m geoai process D:/geoai/sessions/<session_id> --out D:/geoai/runs
```
```bash
python -m geoai vlm-export D:/geoai/runs/<session_id> --batch E:/vlm_batch_001
```
Copy `E:/vlm_batch_001` to the VLM machine and, inside it, run `python run_vlm_batch.py .` (standard library only; it calls their existing `POST /classify`). Bring the folder back, then:
```bash
python -m geoai vlm-import E:/vlm_batch_001 D:/geoai/runs/<session_id>
```

Output of `process`, per session (`runs/<session_id>/`):

| File | Content |
|---|---|
| `crops/*.png` | Crops sent to the VLM. EXIF holds GPS = camera and GPSDest = anomaly; a `geoai` text chunk holds the full record |
| `frames/*.png` | The full frame of each crop, at 1344×756 (the VLM team asked for these in week 1) |
| `sidecars/*.json` | One record per anomaly: times (UTC + IST), positions, footprint, evidence, flags, crop box |
| `candidates.csv`, `anomalies.geojson` | All records, including `over_budget` and `duplicate` ones |
| `not_sent/` | Crops not sent (duplicates, or over an optional budget), kept for audit |
| `contact_sheet.jpg` | The 48 highest-evidence crops, for a quick look |
| `report.json` | Trajectory, latency, camera geometry, bank statistics, per-frame scores, timings, estimated VLM hours |

## How "not healthy" is found

1. **Coverage without tracking.** One frame is analysed per 1.5 m of travel. Each analysed frame owns the 1.5 m strip of road nearest the camera (the sharpest part of the image), so every metre of road is looked at once. Anomalies seen twice at a strip edge are de-duplicated by position.
2. **What normal looks like, per session.** About 400 frames spread over the route form a bank of DINOv2 patch features, kept separately for every image row.
   - Recurring things count as normal: the usual texture, lane markings, the vehicle's own shadow.
   - Bank patches within 10 m of the frame being scored are ignored, so a distress can't match itself.
   - The most unusual 5 % of bank patches are trimmed as likely distress.
3. **Score.** Each patch's distance to its nearest normal neighbours is compared with the bank's own distances: `u = (d − median) / (4 × (q90 − median))`. The median and 90th percentile come from the bulk, which a few per cent of distress in the bank can't move. Anomalies are connected regions with `u ≥ 1`, and their **evidence** is the sum of `u − 0.6` over their patches.
4. **Design choices measured in tests:**
   - DINOv2 blocks 4+8, not the last block: with the last block one pothole made 62–90 % of its neighbourhood look anomalous.
   - Brightness is normalised before DINOv2, so shadows mostly vanish.
   - Road paint is masked out.
5. **Absolute distress probe:** a linear probe on the same DINOv2 features, trained on the public Attain dataset (2,293 vehicle-mounted phone images, CC BY 4.0; no labelling by us). It gives every patch an absolute "looks like distress" score:
   - patches above its threshold stay out of the bank, so distress that is everywhere on a road is no longer learned as "normal";
   - it adds candidates of its own.

   Held-out metrics are in [models/probe/MODEL_CARD.md](models/probe/MODEL_CARD.md).
6. **Absolute crack detector:** a Hessian line filter with length and thinness checks in millimetres; a second absolute signal for thin cracks.
7. **Close distresses, one crop.** In one frame, distresses within 0.5 m of each other on the ground become one larger crop of that pavement segment (members listed in the record), unless that crop would be downscaled below half size.
8. **Duplicates merged, no cap.** The same stretch of road seen from several frames is merged on the ground: strongest evidence first, footprints overlapping by ≥ 50 % of the smaller one. Every remaining record is sent, however many there are per km. `extract.budget_per_km` (default none) can cap the count to limit VLM time; `report.json → estimated_vlm_hours` gives the cost at about 14 s per image.

**Geometry.** The camera model comes from the session itself, with no calibration step:
- intrinsics from the Camera2 characteristics;
- pitch and roll from the gravity sensor;
- height from the survey form.

This is marked `geometry_nominal`. The position uncertainty is dominated by phone GNSS (3–5 m), and each record carries a 1σ value.

## Crops for Qwen2.5-VL (from the VLM team's answers)

- Both sides are multiples of 28 and the area is at most 1,048,576 px, so the model receives the pixels unchanged. `qwen.smart_resize` reproduces every row of the VLM team's measured table.
- At least 324 visual tokens, the smallest size they have evaluated. Small anomalies get more real context and are never enlarged.
- Aspect ratio at most 3:1; the road runs up the image; 25 % margin.
- PNG, not JPEG. Their tests showed that re-encoding as JPEG q95 changed labels on 9 of 49 photos.

## Re-creating the probe (connected machine)

```bash
python tools/fetch_attain.py data/attain
```
```bash
python tools/train_probe.py data/attain models/probe
```
```bash
python tools/probe_card.py
```
- `fetch_attain.py` downloads 162.5 MB, checking every file's published SHA-256.
- `train_probe.py` takes about 30 minutes on a CPU.
- `probe_card.py` writes `MODEL_CARD.md` from the recorded metrics.
- Only `models/probe/attain_dinov2_probe_v1.npz` is needed on the pipeline PC; `data/attain/` can be deleted after training.

## Tests

```bash
python -m pytest -m "not slow"
```
```bash
python -m pytest -m slow
```
- The fast suite has 59 tests and runs in about 2 minutes.
- The slow suite (9 tests) is end to end on simulated sessions, including a GNSS outage and a run with the probe on, and takes about 5 minutes on a CPU.
- Set `GEOAI_EMULATOR_SESSION=<folder>` to also run the video-join tests on a real app recording.

The simulator (`python -m geoai simulate OUT`) writes a complete session package (spec v1.1) with known distresses, lane markings, a shadow, 1 Hz GNSS with noise and latency, a rolling shutter and H.264 fragmented MP4. `python -m geoai evaluate RUN truth.json` scores a run against it.

## Validation status

| Check | Result |
|---|---|
| Qwen input sizing | Identical to the VLM team's measured processor table (8 sizes) |
| DINOv2 implementation | Bit-identical (max difference 0.0) to Meta's official code on 4 input shapes |
| Trajectory (synthetic) | Interpolation < 0.10 m; GNSS latency within ±20 ms; UTC within 2 ms; three-point-turn reversing flagged |
| Video join (real app recording, emulator) | 496 of 496 samples joined to their rows |
| Simulated road, end to end | All 5 distresses found, centroids within 0.35 m, no false alarms on plain asphalt, look-alikes ranked below distress, byte-identical reruns |
| **Distress probe, held-out Attain test split** (real windshield frames; blocks of 20 consecutive frames held out) | Patch AUROC 0.863. At the threshold (1.8 % false positives on test road patches), it flags 85 % of linear cracks, 81 % of potholes and 94 % of alligator cracking, against 30 %, 24 % and 52 % expected by chance. Ravelling and lane-edge drop-off are not detected by the probe (below chance); faded markings are scored below chance. [Model card](models/probe/MODEL_CARD.md) |
| **Real asphalt, VLM team's 27 frames (held out)** | Bank alone: potholes and dark objects ranked highest, but fine cracks didn't stand out when the reference frames were cracked too, and the crack detector fired on kerbs. With the probe: the cracks, alligator zones, block-crack networks and the larger pothole are flagged, and the 3 no-distress frames show 0-0.7 % of patches flagged |

## First real sessions

These checks need no manual labelling:
1. `report.json`: `road_not_covered_m` near 0, and a bank of about 400 frames.
2. `contact_sheet.jpg`: a glance at the top crops.
3. After the VLM batch, `import_summary.json → feedback`: the share of crops the VLM called distress, by evidence band and by signal. If the lowest band is almost all `Normal`, raise `anomaly.min_mass`; if crack-only candidates are all `Normal`, set `cracks.as_candidates` to false. This is a recommendation only, because the VLM's own Stage 1 misses about half of all distress.

## Configuration

All parameters with defaults and reasons are in `geoai/config.py`; override them with `--config my.json`. Unknown keys are an error. The ones you are most likely to change:

| Key | Default | Meaning |
|---|---|---|
| `anomaly.device` | `auto` | `cuda` on the pipeline PC |
| `extract.budget_per_km` | none | Optional cap on crops sent per km (highest evidence first) |
| `extract.dedupe_overlap` | 0.5 | Footprint overlap (share of the smaller) that makes two records one |
| `crop.merge_gap_m` | 0.5 | Distresses this close on the ground (m) in one frame share one crop |
| `anomaly.min_mass` | 6.0 | Smallest evidence kept |
| `crop.full_frame` | true | Also send each crop's full frame |
| `gnss.default_latency_s` / `default_latency_measured` | 0 / false | Set from the marker test |
| `geometry.default_height_m` | 1.5 | Used only if the survey form has no mount height |

## Performance

- A GPU is needed for production. At 40 km/h about 7 frames per second of video are analysed: about 27,000 frames per hour of driving.
- **GPU:** DINOv2 ViT-S takes tens of milliseconds per 4K analysis window, under 30 minutes of analysis per hour of video.
- **CPU:** several seconds per frame, more than a day per hour of video.
- Decoding 4K HEVC on the CPU runs at 60–150 fps.
- The VLM (about 14 s per image) is the slowest step. Without a cap its time grows with the number of distresses; `extract.budget_per_km` can limit it.

## Limitations

- **Probe domain.** The probe was trained on windshield views. The S21's downward view is a different domain, and its real accuracy there is unknown until the first sessions. `vlm-import` reports how often the VLM confirms probe-only candidates (`feedback → by_signal`).
- **Distress the probe misses.** Ravelling and lane-edge drop-off rely on the session bank alone. Where they are everywhere on a road, they can still be learned as normal.
- **Synthetic texture.** The probe reads procedural texture as distressed: 16 % of synthetic plain asphalt against 0–1.8 % of real clean road. The simulator tests therefore run with the probe off; the probe is tested on real frames.
- Shadows and worn markings can still be sent; the VLM's prompt excludes them.
- Concrete roads: the VLM has no concrete labels, and the pipeline doesn't detect pavement type.
- Lens and mount calibration from calibration sessions (docs/pipeline/04) is not implemented; geometry is nominal.

## Code map

| Module | Role |
|---|---|
| `session.py` | Reads the session package (spec v1.1) |
| `trajectory.py` | GNSS cleaning, latency, UTC fit, Hermite trajectory, path distance, gyro heading, reversing |
| `video.py` | PyAV decoding; sample ↔ `frames.csv` join by PTS key; BT.709 conversion |
| `geometry.py` | Camera model, pixel ↔ road plane, nominal model from session metadata |
| `dinov2.py` | DINOv2 ViT-S/14 (registers), dense tiled patch features |
| `anomaly.py`, `cracks.py` | Session bank, patch scores, evidence; absolute crack detector |
| `qwen.py` | Qwen2.5-VL `smart_resize` and crop planning |
| `extract.py` | Frame selection, detection, geolocation, crops, ground-plane de-duplication, optional budget, outputs |
| `blur.py` | Large-sigma Gaussian blur on the GPU (OpenCV's kernel and border; checked against cv2) |
| `records.py` | PNG with EXIF/XMP-like metadata, sidecars, CSV, GeoJSON |
| `vlm.py`, `tools/run_vlm_batch.py` | Folder-batch export, VLM-side runner, validated import |
| `sim.py`, `simtexture.py`, `evaluate.py` | Session simulator and scoring |
