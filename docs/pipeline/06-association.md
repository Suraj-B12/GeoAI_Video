# 06 · Coverage: Each Metre of Road Analysed Once

Goal: one crop per physical anomaly, without tracking detections across frames.

Implementation: [`geoai/extract.py`](../../pipeline/geoai/extract.py).

## 1. Why not tracking

At survey speed a distress is visible in 3–12 frames and jumps 160–400 px between them. Box trackers fail at that displacement, and road-plane tracking needs accurate geometry and careful gating. The camera and vehicle motion give a simpler, exact alternative: choose which frame is responsible for which piece of road.

## 2. Bands

- For each recording resolution the stage computes, from the camera model, the nearest usable road ahead of (or behind) the camera, `band_y0`. It reports this in `report.json → views`.
- A frame is analysed every `band_length_m` (1.5 m) of path distance `s`.
- Analysed frame *i* is responsible for the road `s ∈ [s_i + band_y0, s_i + band_y0 + L)`: the nearest, sharpest strip in its image. Consecutive bands abut, so every metre of road is analysed once.
- If the usable road in view is shorter than the band plus context, the band is shortened automatically (with a warning).

## 3. Keeping each anomaly once

- An anomaly found in frame *i* is kept only if its ground centroid lies in frame *i*'s band, plus 0.3 m of tolerance. It also accepts up to 1.5 frames of travel beyond the band end, because the next analysed frame lands up to one frame past its target.
- **Duplicates are merged on the ground** by non-maximum suppression across all frames of the session:
  - records are taken strongest evidence first;
  - a record is a `duplicate` of an already kept one, from another frame, if their ground footprints overlap by at least `dedupe_overlap` (0.5) of the smaller footprint. A record without a footprint (part of its box above the horizon) falls back to centroids within 0.35 m;
  - the kept record lists its `duplicates`, and each duplicate names its `duplicate_of`.

  This catches an anomaly kept twice at a band edge and a long crack whose views from neighbouring frames overlap. It is not transitive: a record is compared only with kept records, never chained through other duplicates. So a road with hundreds of separate distresses per km keeps every one of them (tested: 300 potholes 0.33 m apart all kept; a carpet of 1 m boxes overlapping by 40 % all kept).
- Before 2026-10-26 the rule was centroids within 0.35 m from frames at most 2 bands apart. It missed re-sightings whose centroids moved further than that, and it merged separate small distresses with close centres.
- **Close distresses in one frame become one larger crop** of that pavement segment:
  - two anomalies merge when their ground footprints are within `crop.merge_gap_m` (0.5 m) of each other, or, without a footprint, their crops overlap by IoU ≥ 0.25;
  - merging repeats, so a cluster becomes one segment;
  - a merge is skipped if the merged crop would be downscaled below `crop.merge_min_scale` (0.5), where fine cracks would blur away;
  - the kept record has the highest evidence, the summed evidence, the union footprint area (`merged_area_m2`) and a `members` list with every member's id, position, evidence and area.
- Frames are skipped if they lie in GNSS gaps, while reversing (detected in [02](02-time-and-trajectory.md)), or while the image is black. While stationary, `s` doesn't advance, so no frame is chosen.
- `report.json → road_not_covered_m` reports road left uncovered by skipped frames.

## 4. No cap on the count

After de-duplication, every remaining record is `sent`, ranked by evidence (`rank`). There is no per-km cap by default, because some roads have hundreds of distresses per km.

`extract.budget_per_km` can still be set to limit VLM time. Then the top `budget_per_km × route km` records (at least 10) are `sent` and the rest are `over_budget`. All records are kept in `candidates.csv`; crops not sent (duplicates, over budget) go to `not_sent/` for audit.

## 5. Tests

[`tests/test_e2e_sim.py`](../../pipeline/tests/test_e2e_sim.py) runs on a simulated session with 1 Hz noisy GNSS, 100 ms latency and a rolling shutter. It checks that:
- every distress is found, with centroids within 0.35 m of the true footprint;
- with no cap, every non-duplicate is sent, each duplicate points to a sent record, and every crop's SHA-256 matches its file;
- reruns are byte-identical;
- road left uncovered is under 1 m.
