# 05 · Detection: Label-Free "Not Healthy Pavement"

Goal: find every stretch of road surface that doesn't look like normal, healthy pavement, with no labelled training data and no manual steps. The VLM decides what it is; this stage decides what is worth showing it.

Implementation: [`geoai/anomaly.py`](../../pipeline/geoai/anomaly.py), [`geoai/cracks.py`](../../pipeline/geoai/cracks.py), [`geoai/dinov2.py`](../../pipeline/geoai/dinov2.py). Every number below is a config default in [`geoai/config.py`](../../pipeline/geoai/config.py).

## 1. Analysis window

All analysed frames of a session show the road through the same fixed image window (the rows covering the band of [06](06-association.md) plus 0.5 m of context), downscaled to about 2 mm/px. So each patch row always sees the same strip of road at the same scale and the same perspective. That is what makes comparisons between frames meaningful.

## 2. Features

DINOv2 ViT-S/14 with registers, Meta's official weights (Apache-2.0), re-implemented in 100 lines and checked bit-identical against Meta's code.
- Patch features come from **blocks 4 and 8**, each L2-normalised and then concatenated.
- Tiles are 448 px with 4 patches of overlap; only tile centres are kept, so every patch has context around it.
- Brightness is divided by its local mean (σ = 24 px) first, so a shadow, which mostly multiplies brightness, looks like the road around it.

**Why not the last block.** Its features mix in context from the whole tile. In tests, one pothole made 62–90 % of the other patches in its window look anomalous. Blocks 4+8 flag 98–100 % of the distress and 3–8 % of its neighbourhood.

## 3. "Normal" from the session itself

- **Bank.** About 400 frames spread over the route, at jittered spacing, so periodic lane markings don't phase-lock with the sampling. Features are kept per patch row, and each row is compared with rows r−1..r+1.
  - Patches that the crack detector marks as cracked stay out of the bank.
  - The most unusual 5 % are trimmed, by k-NN distance to other frames.
  - A greedy coreset keeps up to 4096 vectors per row.
- **No self-matching.** Bank patches within 10 m of the frame being scored are ignored, so a distress that happens to be in a bank frame can't match itself.
- **Score.** `d` is the mean cosine distance to the 3 nearest bank neighbours. Then `u = (d − median) / (4 × (q90 − median))`, from the bank's own leave-place-out distances in that row.
  - The median and 90th percentile come from the bulk, which a few per cent of distress in the bank can't move.
  - An upper quantile was tried and rejected. Computed on the trimmed bank, its tail was cut off (13 false candidates in 4 clean windows). Computed on the untrimmed bank, it was set by the bank's own distress, 0.30–0.37 against at most 0.16 for clean road.
- **Anomaly.** A connected region with seed `u ≥ 1`, grown into `u ≥ 0.6`. Its **evidence** is the sum over its patches of `u − 0.6`; regions with evidence below 6 are dropped. In synthetic tests distress scored 31–136 and texture flukes 0.4–8.3.
- **Paint.** Patches mostly covered by road paint (brighter than 1.45 × the local road, mask grown by half a patch) add no evidence.

## 4. Absolute crack signal

A multi-scale Hessian line filter on illumination-normalised grey, with a robust z-score against the session's background (z ≥ 6). Components must be at least 150 mm long on the ground and thin. It adds candidates, and it keeps cracked patches out of the bank, so a road cracked almost everywhere can't teach the bank that cracks are normal.

## 4b. Absolute distress probe (trained on public data, no manual labels)

A logistic-regression probe on the same DINOv2 features, trained on the public **Attain** dataset (2,293 vehicle-mounted phone images, polygon and box annotations, CC BY 4.0) by `tools/train_probe.py`:
- **Positives:** patches inside distress annotations.
- **Negatives:** patches away from any distress; faded markings and manholes are hard negatives.
- **Split:** blocks of 20 consecutive frames. The VLM team's 27 sample frames are forced into the test split.
- **Threshold:** the seed level is set at 1 % false positives on validation road patches.

It does two jobs:
- Patches it calls distress **stay out of the session bank**, so a road cracked almost everywhere no longer teaches the bank that cracks are normal.
- It adds its own candidates, with evidence = the sum over the region of `logit − (seed − 1)`, on the same scale as the bank evidence.

A probe trained with different feature settings is refused. Held-out metrics are in [`pipeline/models/probe/MODEL_CARD.md`](../../pipeline/models/probe/MODEL_CARD.md). To re-create it on a connected machine: `tools/fetch_attain.py`, then `tools/train_probe.py`, then `tools/probe_card.py`.

## 5. What testing showed on real images

**Session bank alone.** 27 real Attain frames from the VLM team, a windshield view rather than our camera, with the bank built from other frames:
- Potholes and dark objects ranked highest.
- Fine cracks did not stand out when the bank frames were cracked too, which is the expected weakness of a session-relative "normal".
- The crack detector missed fine real cracks at that resolution and fired on kerbs and guardrails.

**With the probe** (§4b), on the same held-out frames: the cracks, alligator zones, block-crack networks and the larger pothole are flagged, and the 3 no-distress frames show 0–0.7 % of patches flagged.

On the Attain test split, the probe flags 85 % of linear cracks, 81 % of potholes and 94 % of alligator cracking. Those compare with 30 %, 24 % and 52 % expected by chance at its 1.8 % patch false-positive rate. Ravelling and lane-edge drop-off are not detected by the probe and rely on the bank.

The S21's downward 4K view is a different domain, so thresholds must be confirmed on the first real sessions. The automatic VLM feedback ([10](10-review-and-export.md)) reports per signal how often the VLM confirms a candidate.

## 6. Output

Per analysed frame: candidate boxes (analysis pixels → frame pixels) with evidence, peak `u`, peak crack z, signal kinds, and patch count. Per frame, `report.json → frame_stats` records the maximum `u`, the anomalous-patch fraction and the crack-pixel fraction, so thresholds can be checked on real data.

## 7. Tests

[`tests/test_anomaly.py`](../../pipeline/tests/test_anomaly.py) (synthetic asphalt windows) checks that:
- pothole, transverse crack and patch are flagged, with evidence above 3× the threshold;
- at most 1 false candidate appears in about 45 clean windows containing recurring markings;
- a distress present in a bank frame is still flagged.

[`tests/test_dinov2.py`](../../pipeline/tests/test_dinov2.py) checks the features against Meta's own output. End to end: [11](11-testing-and-acceptance.md).
