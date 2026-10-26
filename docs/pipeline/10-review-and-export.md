# 10 · Review and Export

> **Changed (22 Oct 2026):** no manual review. The VLM classifies and its results are imported automatically ([09](09-vlm-interface.md)). `vlm-import` also reports how often the VLM confirmed a distress by evidence band and by signal, so thresholds can be tuned from data. Exports today are CSV and GeoJSON per run. The review UI, GeoPackage, chainage and IRC section ratings described below are later work.

Goal: engineers confirm or correct what the pipeline found, and the results leave the system in formats that GIS tools and road agencies use. Every number in a report can be traced back to crops, model versions and human decisions.

## 1. Review tool

**Pilot start:** QGIS with the review GeoPackage (§3) and an attribute form, with value maps for `decision`, `label` and `severity`. It needs no extra software.

**MVP review app** (`geoai review`): a local web app (FastAPI + MapLibre GL JS), bound to `127.0.0.1` only.

| Panel | Shows |
|---|---|
| Queue | Items sorted by priority: invalid > safety-critical > family conflict > low confidence > truncated > possible duplicate > audit sample |
| Crop | The crop exactly as the VLM saw it, at 100 % and fitted |
| Context | The full frame with distress box and crop box, plus thumbnails of the other observations of the track |
| Map | Distress position with its 1σ circle, the trajectory, neighbouring records. Optional offline basemap: a PMTiles extract of the survey area, prepared once and carried over. |
| Facts | Detector class and score; VLM label, severity and rationale; area and length in m²/m; flags; versions |

**Actions:** Accept · Relabel (label and severity from `taxonomy.yaml`) · Reject (reason: look-alike, shadow, wet patch, marking, joint, debris, other) · Duplicate of (pick a record) · Merge error (two distresses in one record) · Note · Skip. Keyboard shortcuts make a decision take about 2 s.

**Rules:**
- **Every decision is appended** to `reviews` with reviewer, time, decision, label, severity and note. Nothing is edited in place; the latest decision wins.
- **Blind review for the audit sample** ([09](09-vlm-interface.md) §8). Hide the VLM's label until the reviewer has decided. Reviewers who see the model's answer tend to accept it, which inflates measured precision.
- **Double review of 10 % of the audit sample** by a second engineer, to measure agreement (Cohen's κ). Model accuracy can't be judged more finely than the reviewers agree with each other.

**Status of a record** (used by every export):

| Status | Meaning |
|---|---|
| `reviewed` | A reviewer decided; the reviewer's label is final |
| `auto` | Not routed to review; the latest valid VLM result for `vlm.active` is final |
| `pending_review` | Routed to review, not yet decided. Exported, but marked and excluded from ratings unless the report says otherwise |
| `rejected` | Reviewer or VLM says not a distress (`no_distress`, `not_pavement`, rejected). Kept, but excluded from the inventory |

## 2. Chainage (where on the road)

Pick the first method available and record it in `chainage_method`:

1. **Centreline** (`centreline`): the agency's road centreline with chainage (Shapefile or GeoPackage). Project each distress onto it in a metric CRS (shapely `line.project`), then correct piecewise-linearly between `km_stone` markers.
2. **Path distance** (`path_distance`): `start_chainage_km` from the survey form plus the session's path distance `s` ([02](02-time-and-trajectory.md)), corrected between `km_stone` markers. `direction = decreasing_chainage` subtracts. This method degrades with detours and lane changes.
3. **None:** chainage left empty. Never estimate it from wall-clock time.

A `km_stone` marker is pressed by the operator when passing the stone, so it is late by the reaction time, roughly 0.5–1 s, or 5–10 m at 40 km/h. The QC report shows marker spacing against the 1 km expected, so a missed or doubled press can be spotted.

## 3. Exports (`geoai export`)

Each export goes into its own folder `exports/<date>_<name>/`, with `export.json` holding the query, filters, pipeline and model versions, counts and file checksums.

| File | Content |
|---|---|
| `inventory.gpkg` | Layers: `distress_points` (centroids), `distress_footprints` (polygons), `long_groups` (one line per long distress), `trajectory` (one line per session), `coverage` (the road area actually observed), `excluded` (rejected records and look-alikes) |
| `inventory.geojson` | Points, RFC 7946: WGS-84, coordinates in **[longitude, latitude]** order |
| `inventory.csv` | One row per distress. UTF-8 with BOM (so Excel shows Hindi road names correctly), `.` as decimal separator, columns named `lat_deg` and `lon_deg` explicitly, footprint as WKT |
| `inventory.kml` | Points for Google Earth: `lon,lat[,alt]` order, with the crop as the description image |
| `crops/` | The crops of exported records, with their EXIF, named `<distress_id>.jpg` |

- **Attributes per distress:** `distress_id, session_id, road_id, road_name, direction, lane, chainage_km, chainage_method, utc, ist, lat_deg, lon_deg, sigma_m, status, label, irc_term, irc_clause, severity, area_m2, length_m, long_group_id, piece_index, possible_duplicate_of, vlm_model_id, vlm_label, vlm_confidence, detector_class, detector_model_id, reviewer, reviewed_utc, flags, crop_file`.
- **Areas and lengths come from metric coordinates.** The ground polygons in local ENU metres ([04](04-calibration-and-geometry.md)) are the source. Never compute area from lat/lon degrees.
- **Coordinates:** GeoPackage and GeoJSON use EPSG:4326. For metric analysis in QGIS, use the matching UTM zone: EPSG:32642–32646 (zones 42N–46N; most of India lies in 43N–45N).

## 4. Section summary in IRC terms (Phase 3)

`geoai report <road>` produces an HTML report (printable to PDF), a CSV of sections and a `sections` layer in the GeoPackage.

1. **Sections:** fixed lengths per lane along chainage (default 100 m; 1 km summaries roll up from these).
2. **Observed area per section** = the union of the ROI ground footprints of valid frames, not the nominal lane width × length. The camera sees about 2.2–4.8 m across ([PLAN §4.3](../PLAN.md)), which may not cover the full lane. If the observed area is below 80 % of the section's lane area, the section is marked `insufficient_coverage` and gets no rating.
3. **Quantity per IRC distress type** = Σ footprint area of `reviewed` and `auto` distresses of that type ÷ observed area × 100 %.
4. **Linear cracks.** IRC rates cracking as % area, but a crack is a line. The width to multiply its length by, or another counting convention, must come from the IRC edition and the agency's practice. Set `report.linear_crack_area_rule` in config; there is no default. **The report refuses to compute cracking % until an engineer sets it.**
5. **Ratings:** look up each quantity in the rating table in `config/irc82_<edition>.yaml`, and combine as that edition defines. The shipped table reproduces the IRC:82-2015 values in [PLAN §4.8](../PLAN.md). An engineer must verify it against the edition in force before first use.
6. **Not measurable from video:** rut depth is reported as `not measured`. Shoving and settlement are reported as `visual indication only`. If the combined rating needs a quantity that wasn't measured, the report shows a **partial** rating over the measured quantities, clearly labelled. It never assumes zero.
7. Every report shows the share of `auto` vs `reviewed` records, the model and prompt versions, the observed-area fraction, and a sign-off block for the responsible engineer.

## 5. Feedback into training

- **Detector:** rejected records become hard negatives; relabels correct the class; a "missed distress" note on a context frame becomes a new label (box drawing in v2). See [../detection-model-plan.md](../detection-model-plan.md).
- **VLM:** reviewed crops with final labels form the fine-tuning and validation set, if the VLM owners decide to fine-tune ([../vlm-information-request.md](../vlm-information-request.md) §6.1).
- `geoai export-training` writes COCO JSON for frames and boxes, and image–label pairs for crops. It always splits **by road and session**, never by frame.

## 6. Tests

- GeoPackage export and re-import: same records and attributes; coordinates within 1e-9°.
- A known point (lat 12.97, lon 77.59) appears in GeoJSON as `[77.59, 12.97]` and in KML as `77.59,12.97`.
- The CSV opened on a machine with a comma-decimal locale still reads `12.9715987`, not `12,9715987`.
- A synthetic section with known footprints and coverage reproduces hand-computed percentages and ratings.
- A section with 60 % observed area: `insufficient_coverage`, no rating.
- No `linear_crack_area_rule` set: the report stops with a clear message rather than guessing.
- Changing a review decision adds a row; the export uses the latest.
