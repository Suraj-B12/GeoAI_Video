# Drive-Video Survey WebGIS — Handoff Brief

**Paste this entire file as the first message to your coding agent (Claude Code or similar), and unzip
`gis_sample.zip` next to the folder you will work in.** Everything below is self-contained: you do not
need our database, our GPU, our server or any password. You build against a mock of our API that ships
in the zip; we then copy your folder onto our machine and it runs against the real data with no code
change.

Contract version: `geoai.gis.v1` (every response carries `"schema": "geoai.gis.v1"`). Fields are only
ever *added* under this version; nothing is renamed or removed. Date of this brief: 28 Oct 2026.

---

## 1. Context in one paragraph

GeoAI is a pavement-distress survey system built as a college capstone in Bengaluru. It has two data
sources. (a) **Citizen uploads**: people photograph damaged roads with the RoadSide app; each photo is
classified by a vision-language model (Qwen2.5-VL-7B) into IRC:82-2015 distress types with a severity.
A WebGIS for those points was briefed earlier ("WebGIS Layer — Handoff Brief"). (b) **Drive-video
survey** (new, this brief): a Samsung Galaxy S21 mounted on a vehicle's windshield records 4K video;
our pipeline cuts the road into **5 m segments**, picks the sharpest full frame of each segment, and the
same model classifies it. The audience is road-maintenance engineers and civic authorities (BBMP):
information-dense, professional, no marketing. **Your job: a WebGIS that shows both layers, with
everything the citizen WebGIS has plus what a continuous road survey needs (coverage, drive track,
% of road length per distress type, a frame-by-frame strip along the route).**

## 2. What the survey data is (read this before designing anything)

- **Capture.** Galaxy S21 (SM-G991B), 3840x2160 at 30 fps, fixed windshield mount tilted 23° down,
  GNSS at 1 Hz + IMU at 200 Hz on the same clock as the video frames.
- **Segments.** The GNSS track is cut into consecutive **5.0 m segments** along the driven path
  (`s0_m` to `s1_m` = path distance from the start of the drive; a final shorter segment is kept if it
  is at least 2.5 m and flagged `partial`). Segments do not overlap, so lengths add up: "% of surveyed
  length with distress X" is a real, countable number (it is what IRC:82 condition ratings use).
- **One frame per segment.** For each segment the sharpest candidate frame in which the whole segment
  is in view (its near edge ~2 m ahead of the camera) is chosen. That full 4K frame is what the model
  sees and what you show. The segment is the band of road roughly 2–7 m ahead of the camera in it.
- **Gate status** (`status`), decided by a cheap frame-level distress detector before the model runs:

  | status | meaning | has image? | has model answer? |
  |---|---|---|---|
  | `send` | detector found possible distress; frame sent to the model | yes | yes (once classified) |
  | `audit` | detector found none, but a random ~10 % sample is sent anyway to measure the detector | yes | yes |
  | `skip` | detector found none; not sent | no | no |
  | `not_covered` | no usable frame (GNSS gap, vehicle stopped, black frames) | no | no |

  **In the current data every segment is `send`**: these drives were run in "send everything" mode
  (`gate_mode = "all"`) to calibrate the detector. The detector's would-be decision is kept in
  `gate_decision_if_gated` (in the sample: 739 send, 70 skip, 4 audit). Future drives will contain all
  four statuses, so the GIS must handle them now (the mock's examples and our tests do).
- **Model labels.** The model answers with **IRC:82-2015 distress types** (Indian Roads Congress, Code of
  Practice for Maintenance of Bituminous Road Surfaces): 18 vision-only types in 4 categories, or
  `Normal`. A segment can have several labels. Severity: `None` (Normal), `Low`, `Medium`, `High`,
  `Unknown`. A confidence below 0.80 sends the result to expert review (`needs_expert_review`). The
  label list, categories, colours and marker sizes come from `GET /gis/meta`; never hard-code them.

  | Category | Labels |
  |---|---|
  | Surface Defects | Bleeding, Streaking, Hungry Surface |
  | Cracks | Hairline Cracks, Alligator Cracking, Longitudinal Cracking, Transverse Cracking, Edge Cracking |
  | Deformation | Slippage, Rutting, Corrugation, Shoving, Shallow Depression, Settlement |
  | Disintegration | Stripping, Ravelling, Potholes, Edge Breaking |
  | — | Normal |

- **Positions.** Phone GNSS: horizontal 1-sigma `position_sigma_m` is about **±6 m** (median 6.0,
  range 4–12 m in the sample). A marker is therefore "the 5 m segment around here", not a surveyed
  point; say so in the UI (e.g. an uncertainty circle on hover/selection, radius = `position_sigma_m`).
  The segment centre is projected onto the road ahead of the camera, so `centre_*` and `camera_*` differ
  by ~4–5 m. Heights: the current sample holds the ellipsoidal height only (`alt_msl_m` is null because
  these drives were processed before the MSL fix of 28 Oct; later drives carry the phone's MSL altitude
  in `camera.alt_msl_m`). **Never display the ellipsoidal height as "above mean sea level"** (in India
  the two differ by tens of metres); show an MSL height only when `alt_msl_m` is present.
- **Times.** `utc` is the frame's GNSS time (ISO 8601, `Z`); `local_time` is the same instant in IST
  (`+05:30`). Show local time to users.
- **Model caveat to show in an "About the data" panel:** the model is the reference here, not ground
  truth. On a labelled benchmark its Stage 1 found about half of distressed frames; every result below
  0.80 confidence is waiting for an expert (`needs_expert_review`), and `expert_reviewed` marks rows a
  human has confirmed or corrected.

**Data now (sample snapshot, 27 Oct 2026):** 3 drives, 813 segments, 4,060 m of road, all classified:
164 segments with distress (815 m, 20.1 % of surveyed length), 649 Normal; 389 awaiting expert review;
0 expert-reviewed yet. Per label: Potholes 131 segments (16.1 % of length), Longitudinal Cracking 73
(8.9 %), Bleeding 61 (7.5 %), Edge Breaking 18, Alligator Cracking 9, Hungry Surface 9, Streaking 7,
Transverse Cracking 7, Edge Cracking 4. Area: south Bengaluru, bbox `[77.5612, 12.9231, 77.5676,
12.9419]`. Citizen layer: 344 uploads (131 classified, 213 expert_review). Expect thousands of segments
per week once surveys run routinely.

## 3. Architecture you plug into

```
S21 drive video -> GeoAI_Video pipeline -> 5 m segments + best frames
   -> VLM (Qwen2.5-VL-7B, GPU) -> local PostgreSQL 17 (video_sessions, video_segments, photos, assessments)
                                  + image store C:\GeoAI\images\<session>\<segment_id>.png
RoadSide app -> Cloudinary + Supabase (citizen uploads) -> same VLM
                     |
        FastAPI server on the VLM PC (port 8000)
          GET /gis/*        read-only GIS API (both layers, images)     <- the ONLY thing you call
          /gis/app/         YOUR WebGIS (static files from Capstone/webgis/)
```

The browser never talks to a database. It calls `/gis/...` on the same origin it was loaded from.

## 4. Data model (for understanding; you read it only through the API)

`schema.sql` in the zip is the full DDL. The tables behind the API:

**`video_sessions`** — one row per drive.

| column | type | meaning |
|---|---|---|
| `session_id` | text PK | e.g. `20261026T114332Z_SM-G991B-F0D1_f428` (UTC start, device, hash) |
| `state` | text | `queued`/`extracting`/`classifying`/`storing`/`stored`/`cleaned`/`failed`; show only `stored`/`cleaned` drives as complete, others as "processing" |
| `stored_at`, `updated_at`, ... | timestamptz | when it entered each state |

**`video_segments`** — one row per 5 m segment, every status.

| column | type | meaning |
|---|---|---|
| `segment_id` | text PK | `<session_id>_seg<k:05d>` |
| `session_id` | text FK | the drive |
| `k` | int | segment index along the drive (0, 1, 2 ...) |
| `status` | text | `send` / `audit` / `skip` / `not_covered` |
| `length_m`, `s0_m`, `s1_m` | float | length and path distance from the drive start (m) |
| `centre_lat`, `centre_lon` | float | segment centre, WGS-84 |
| `chainage_start_km`, `chainage_end_km` | float | road chainage when the survey form gave one (null in current data) |
| `utc` | timestamptz | frame time (null for not_covered) |
| `image_path`, `image_sha256` | text | stored PNG (server-side path; you get a URL instead) |
| `photo_id`, `assessment_id` | FK | the model result rows, once answered |
| `record` | jsonb | the full `geoai.segment.v1` record (below) |
| `vlm_response` | jsonb | the raw model answer |

**`assessments`** (shared with the citizen uploads) holds each answered segment's result:
`stage1_label` (`Normal`/`Distress`), `stage1_confidence`, `distress_types` (jsonb array of IRC names in
the model's order), `severity`, `description` (one sentence), `stage2_confidence`, `needs_expert_review`,
`status` (`classified` or `expert_review`), `expert_reviewed`, `expert_corrected_types`,
`expert_corrected_severity`, `road_segment_id` (= segment id; null for citizen uploads).

**Record JSON (`video_segments.record`) — the fields that matter for a map**, all flattened into the API's
properties (section 6):

```
segment  k, length_m, s0_m, s1_m, partial, start{lat,lon}, end{lat,lon}, centre{lat,lon},
         footprint_lonlat [[lon,lat],...]   closed ring: segment length x visible road width
frame    width 3840, height 2160, utc_iso, local_iso
camera   lat, lon, heading_deg, speed_mps, h_sigma_m, alt_ellipsoid_m, alt_msl_m (null in the sample)
target   lat, lon (= segment centre), sigma_m (1-sigma position error), distance_m (camera -> centre)
quality  sharpness, sharpness_rank, n_candidates, blur_px, reason
gate     mode (gate|all), decision, decision_if_gated, reasons[], scores{probe_frac, evidence, crack_mm, ...}
survey   road_id, road_name, direction, lane, start_chainage_km, operator, vehicle, weather
device   manufacturer, model, device_id
```

**Footprint polygons.** `footprint_lonlat` is the patch of road the segment covers, in `[lon, lat]`
order: about 5 m along the road by the visible width (typically 4–8 m). The API returns it as a
GeoJSON Polygon (closed, counter-clockwise, RFC 7946). Footprints of consecutive segments tile the
road; drawn filled they make a continuous "condition ribbon" along the drive.

## 5. API contract

Base URL: `""` (same origin) in production and with the mock. Make it configurable in one place
(`config.js`: `window.GIS_CONFIG = { apiBase: "" }`). All endpoints are `GET`, need no key, return
`application/json` or `application/geo+json` (CSV and images excepted), and answer errors as
`{"detail": "<reason>"}` with 400 (bad parameter), 404 (no such image) or 503 (data store unavailable:
show a banner, retry later). Unknown query parameters are a 400 (a typo never silently returns
everything); parameters starting with `_` are ignored (use them as cache busters).

### 5.1 `GET /gis/meta`

Label list (name, category, IRC section, one-line description, colour, `marker_priority`), categories,
severities with marker radius, gate statuses and `vlm_classes` with meanings and colours, the 0.80
confidence threshold, allowed preview widths, limits, and the video data extent:

```json
{"schema":"geoai.gis.v1",
 "labels":[{"name":"Bleeding","is_distress":true,"category":"Surface Defects","irc_section":"7.2.1",
            "description":"A surface having a thin film of excess or free bituminous binder, creating a shiny, glass-like reflecting surface.",
            "color":"#6d28d9","marker_priority":16}, ... ,
           {"name":"Potholes",...,"color":"#7f1d1d","marker_priority":1}, ... ,
           {"name":"Normal","is_distress":false,"category":"Normal","color":"#16a34a","marker_priority":19}],
 "severities":[{"name":"None","radius_px":4},{"name":"Low","radius_px":6},{"name":"Medium","radius_px":9},
               {"name":"High","radius_px":13},{"name":"Unknown","radius_px":4}],
 "statuses":[{"name":"send","meaning":"..."}, ...],
 "vlm_classes":[{"name":"distress","color":null,"meaning":"..."},{"name":"normal","color":"#16a34a",...},
                {"name":"unknown","color":"#6b7280",...},{"name":"pending","color":"#9ca3af",...},
                {"name":"not_sent","color":"#d1d5db",...},{"name":"not_covered","color":"#111827",...}],
 "confidence_threshold":0.8, "preview_widths":[160,320,480,640,960,1280,1600,1920],
 "limits":{"default_limit":5000,"max_limit":20000},
 "video":{"sessions":3,"segments":813,"with_image":813,"answered":813,
          "segments_by_status":{"send":813,"audit":0,"skip":0,"not_covered":0},"route_m":4060.31,
          "bbox":[77.5612129,12.9230764,77.5675887,12.941901],
          "time_range":{"from":"2026-10-26T11:44:08.929Z","to":"2026-10-27T05:11:54.618Z"}},
 "endpoints":{...}}
```

**Marker rule (both layers):** colour = `marker_label`'s colour; radius = `final_severity`'s
`radius_px`; non-distress classes use their `vlm_classes` colour. `marker_label` is the first of the
segment's labels in `marker_priority` order (Potholes before Alligator Cracking before ... before
Normal); it is computed by the server, so you just look it up.

### 5.2 `GET /gis/video/segments.geojson`

A FeatureCollection, one Feature per segment, ordered by session then `k`. Query parameters (all
optional, combinable; comma = "any of"):

| parameter | example | matches |
|---|---|---|
| `session` | `session=A,B` | `session_id` |
| `date_from`, `date_to` | `2026-10-26`, `2026-10-27T05:05:00Z` | `utc`; ISO date or date-time, UTC when no offset; from inclusive, to exclusive; a plain date as `date_to` includes that whole day |
| `label` | `label=Potholes,Alligator Cracking` | any of `final_labels` (case-insensitive; must be one of the 19 names, else 400) |
| `severity` | `severity=Medium,High` | `final_severity` |
| `status` | `status=send,audit` | gate `status` |
| `vlm_class` | `vlm_class=distress` | `distress`, `normal`, `unknown`, `pending`, `not_sent`, `not_covered` |
| `answered` | `true`/`false` | has a model answer |
| `reviewed` | `true`/`false` | `expert_reviewed` |
| `needs_review` | `true`/`false` | `needs_expert_review` |
| `min_confidence` | `0.8` | `confidence` >= value (unanswered never match) |
| `bbox` | `77.56,12.92,77.57,12.94` | segment centre inside `minLon,minLat,maxLon,maxLat` (**longitude first**) |
| `geom` | `footprint` (default) / `point` / `line` | geometry: footprint Polygon, centre Point, or start->end LineString |
| `limit`, `offset` | default 5000, max 20000 | paging over the matches |

Response (real, abbreviated; full examples in the zip under `examples/`):

```json
{"type":"FeatureCollection","schema":"geoai.gis.v1","layer":"video",
 "generated_at":"2026-10-27T23:20:55.530Z","numberMatched":813,"numberReturned":813,
 "filters":{"limit":20000,"offset":0},"geom":"footprint",
 "bbox":[77.5612129,12.9230764,77.5675887,12.941901],
 "features":[{"type":"Feature","id":"20261026T114332Z_SM-G991B-F0D1_f428_seg00001",
   "geometry":{"type":"Polygon","coordinates":[[[77.5666613,12.9287039],[77.5666402,12.9286229],
              [77.5666916,12.9286376],[77.5666988,12.9286654],[77.5666613,12.9287039]]]},
   "properties":{"layer":"video","segment_id":"20261026T114332Z_SM-G991B-F0D1_f428_seg00001",
     "session_id":"20261026T114332Z_SM-G991B-F0D1_f428","k":1,"status":"send",
     "gate_mode":"all","gate_decision":"send","gate_decision_if_gated":"send",
     "gate_reasons":["probe_frac","evidence","crack_mm"],"gate_probe_frac":0.0602,"gate_probe_max_logit":6.668,
     "gate_evidence":1520.7,"gate_crack_mm":14305.7,
     "s0_m":7.45,"s1_m":12.45,"length_m":5.0,"partial":false,"chainage_start_km":null,"chainage_end_km":null,
     "utc":"2026-10-26T11:44:11.396Z","local_time":"2026-10-26T17:14:11.396+05:30",
     "centre_lat":12.9286534,"centre_lon":77.5666722,"position_sigma_m":6.0,
     "camera_lat":12.9286473,"camera_lon":77.5667135,"camera_heading_deg":284.4,"camera_speed_mps":2.6,
     "camera_speed_kmh":9.4,"camera_h_sigma_m":6.0,
     "road_id":"TST-1","road_name":"test 1","direction":"increasing_chainage","lane":"L1","device_model":"SM-G991B",
     "answered":true,"assessment_id":"be26297d-07ad-4079-8478-6bc9dc47b458","assessment_status":"expert_review",
     "vlm_class":"distress","stage1_label":"Distress","stage1_confidence":0.7216,"stage2_confidence":0.603,
     "confidence":0.603,"labels":["Bleeding","Edge Cracking"],"primary_label":"Bleeding","severity":"Low",
     "description":"Shiny patches and a ragged pavement edge indicate bleeding and edge cracking respectively (IRC:82 §7.2.1, §7.3.6).",
     "needs_expert_review":true,"expert_reviewed":false,"expert_corrected_types":null,"expert_corrected_severity":null,
     "final_labels":["Bleeding","Edge Cracking"],"final_primary_label":"Bleeding","marker_label":"Edge Cracking",
     "final_severity":"Low","condition_indicators":[],"reviewed_at":null,"processed_at":"2026-10-27T23:18:05.884Z",
     "quality_sharpness":0.05286,"quality_sharpness_rank":1,"quality_n_candidates":11,"quality_blur_px":0.81,
     "quality_reason":"sharpest","low_light":false,"image_width":3840,"image_height":2160,"has_image":true,
     "image_url":"/gis/image/20261026T114332Z_SM-G991B-F0D1_f428_seg00001",
     "preview_url":"/gis/image/20261026T114332Z_SM-G991B-F0D1_f428_seg00001?w=1280",
     "thumb_url":"/gis/image/20261026T114332Z_SM-G991B-F0D1_f428_seg00001?w=480"}}, ...]}
```

A geometry can be `null` (no position) or a Point when a footprint is missing; handle both.

### 5.3 `GET /gis/video/segments.csv`

Same filters (no `geom`), CSV with UTF-8 BOM (opens correctly in Excel), one row per segment, lists
joined with `"; "`, booleans `true`/`false`. Columns: `segment_id, session_id, k, status, vlm_class,
answered, assessment_status, final_labels, final_primary_label, marker_label, final_severity, labels,
primary_label, severity, confidence, stage1_label, stage1_confidence, stage2_confidence,
needs_expert_review, expert_reviewed, expert_corrected_types, expert_corrected_severity, description,
condition_indicators, utc, local_time, centre_lat, centre_lon, position_sigma_m, s0_m, s1_m, length_m,
partial, chainage_start_km, chainage_end_km, road_id, road_name, direction, lane, camera_lat, camera_lon,
camera_heading_deg, camera_speed_kmh, camera_h_sigma_m, gate_mode, gate_decision, gate_decision_if_gated,
gate_reasons, gate_probe_frac, gate_evidence, gate_crack_mm, quality_sharpness, quality_blur_px,
quality_reason, low_light, image_url`. Your "Export CSV" button simply opens this URL with the current
filters.

### 5.4 `GET /gis/video/sessions?session=&track=true&per_km=true`

Per drive (`track` default false, `per_km` default true):

```json
{"schema":"geoai.gis.v1","count":3,"sessions":[
 {"session_id":"20261027T050257Z_SM-G991B-F0D1_d47c","state":"stored","has_error":false,
  "road_id":"TST-2","road_name":"test 2","direction":"increasing_chainage","lane":"L1","device_model":"SM-G991B",
  "start_utc":"2026-10-27T05:03:19.384Z","end_utc":"2026-10-27T05:11:54.618Z","local_date":"2026-10-27",
  "segments":528,"segments_by_status":{"send":528,"audit":0,"skip":0,"not_covered":0},
  "segments_by_vlm_class":{"distress":...,"normal":...,"unknown":0,"pending":0,"not_sent":0,"not_covered":0},
  "answered":528,"needs_expert_review":...,"expert_reviewed":0,
  "route_m":2637.5,"surveyed_m":2637.5,"not_covered_m":0.0,"classified_m":2637.5,"distress_m":507.5,
  "normal_m":2130.0,"pct_distress_of_surveyed":19.242,"pct_distress_of_classified":19.242,
  "labels":[{"label":"Potholes","segments":...,"length_m":...,"pct_of_surveyed":...,"pct_of_classified":...}, ...19 rows],
  "bbox":[77.5651538,12.9231365,77.5675412,12.9418492],
  "per_km":[{"km":0,"from_m":0.0,"to_m":1000.0,"surveyed_m":...,"not_covered_m":...,"classified_m":...,
             "distress_m":...,"labels_m":{"Potholes":...,"Bleeding":...},"pct_distress_of_surveyed":...}, ...],
  "track":{"type":"LineString","coordinates":[[77.56...,12.92...], ...]}}, ...]}
```

The other two drives in the sample: `..._f428` (TST-1, 230 segments, 1,150 m, 26 Oct) and `..._293d`
(TST-2, 55 segments, 273 m, 26 Oct). `track` is a LineString through the segment centres, or a
MultiLineString when the drive has gaps.

**Definitions (use these words in the UI):** *surveyed* = covered segments (send, audit, skip);
*not covered* = never imaged; *classified* = answered by the model (distress or normal). A segment with
two labels counts once in each label row, so label rows do not add up to `distress_m`. Labels are the
**final** labels: the expert's correction when there is one, else the model's.

### 5.5 `GET /gis/video/summary`

Same filters as segments (limit/offset ignored). The stats bar numbers for the current filter:
`sessions, segments, segments_by_status, segments_by_vlm_class, answered,
answered_by_assessment_status, answered_by_final_severity, needs_expert_review, expert_reviewed,
with_image, route_m, surveyed_m, not_covered_m, sent_m, classified_m, distress_m, normal_m, unknown_m,
pending_m, not_sent_m, pct_distress_of_surveyed, pct_distress_of_classified, labels[] (as in sessions),
time_range{from,to}, bbox`. Sample (no filter): 813 segments, 4,060.31 m, distress 815.31 m = 20.08 %,
389 need review.

### 5.6 `GET /gis/image/{segment_id}?w=`

| URL | returns | use for |
|---|---|---|
| `/gis/image/<id>?w=480` (`thumb_url`) | JPEG 480x270, ~15–20 KB | popups, the segment strip, lists |
| `/gis/image/<id>?w=1280` (`preview_url`) | JPEG 1280x720, ~70–110 KB | the detail panel |
| `/gis/image/<id>` (`image_url`) | the original PNG 3840x2160, **5–9 MB** | an explicit "Open full resolution" link only |

`w` snaps up to 160/320/480/640/960/1280/1600/1920. Responses are cacheable (`Cache-Control: public,
max-age=86400`, ETag). 404 when the segment has no stored image (`skip`, `not_covered`). Always use the
URLs from the properties; never build image paths yourself.

### 5.7 `GET /gis/citizen/assessments.geojson`

The citizen uploads, from whatever backend our server runs on, as Point Features with **the same
property names** for everything the markers and filters use (`labels`, `final_labels`,
`marker_label`, `final_severity`, `confidence`, `vlm_class`, `needs_expert_review`, `expert_reviewed`,
`description`, `image_url`/`preview_url`/`thumb_url`) plus `id`, `photo_id`, `assessment_status`,
`latitude`, `longitude`, `address`, `created_at` (upload time), `processed_at`, `layer: "citizen"`.
Filters: `date_from/date_to` (on `created_at`), `label`, `severity`, `vlm_class`, `reviewed`,
`needs_review`, `min_confidence`, `bbox`, `limit/offset`, and `include_review` (default `false`: only
`classified`/`done`; `true` adds `expert_review` rows: show those with a "pending review" badge).
Citizen images are Cloudinary URLs (internet needed); the thumb/preview URLs are Cloudinary resizes.
The server caches citizen rows for 60 s, so polling more often gains nothing.
In the sample, addresses are blanked for privacy (`"address": null`); the real server returns them.

## 6. Property reference (video features)

| property | type | notes |
|---|---|---|
| `layer` | `"video"` | `"citizen"` on the other layer |
| `segment_id`, `session_id`, `k` | str, str, int | `k` = position along the drive |
| `status` | str | gate status (section 2) |
| `gate_mode`, `gate_decision`, `gate_decision_if_gated`, `gate_reasons` | str, str, str, [str] | with `gate_mode = "all"` every segment is sent; `gate_decision_if_gated` is what the detector would have done |
| `gate_probe_frac`, `gate_probe_max_logit`, `gate_evidence`, `gate_crack_mm` | num | detector scores (show in an "advanced" section) |
| `s0_m`, `s1_m`, `length_m`, `partial` | num, bool | path distance from the drive start |
| `chainage_start_km`, `chainage_end_km` | num or null | road chainage when known; else show `s0_m` as "m from start" |
| `utc`, `local_time` | ISO str | frame time; null for not_covered |
| `centre_lat`, `centre_lon`, `position_sigma_m` | num | segment centre and its 1-sigma error (m) |
| `camera_lat`, `camera_lon`, `camera_heading_deg`, `camera_speed_mps`, `camera_speed_kmh`, `camera_h_sigma_m` | num | where the phone was |
| `road_id`, `road_name`, `direction`, `lane`, `device_model` | str | from the survey form |
| `answered`, `assessment_id`, `assessment_status` | bool, str, str | `classified` or `expert_review` |
| `vlm_class` | str | `distress`, `normal`, `unknown`, `pending`, `not_sent`, `not_covered` |
| `stage1_label`, `stage1_confidence`, `stage2_confidence`, `confidence` | str, num | `confidence` = Stage 2's when distress, else Stage 1's |
| `labels`, `primary_label`, `severity`, `description`, `condition_indicators` | | the model's answer (`labels[0]` is the model's listing order, not "most important") |
| `needs_expert_review`, `expert_reviewed`, `expert_corrected_types`, `expert_corrected_severity`, `reviewed_at` | | expert state |
| `final_labels`, `final_primary_label`, `final_severity` | | **what to display**: expert correction if present, else the model |
| `marker_label` | str | the label that colours the marker |
| `quality_sharpness`, `quality_sharpness_rank`, `quality_n_candidates`, `quality_blur_px`, `quality_reason`, `low_light` | | frame quality |
| `image_width`, `image_height`, `has_image`, `image_url`, `preview_url`, `thumb_url` | | section 5.6 |

## 7. Developing offline with the sample bundle

`gis_sample.zip` (~146 MB) unzips to `gis_sample/`:

```
gis_sample/
  README.md                      what is in it, mock vs real differences
  mock_server.py, gis_core.py    the mock API (Python 3.9+, standard library only)
  api/gis/meta.json              exact API responses for all 813 segments, 3 sessions, 344 citizen uploads
  api/gis/video/segments.geojson (+ _point, _line), sessions.json, summary.json, segments.csv
  api/gis/citizen/assessments.geojson
  examples/                      pretty-printed features (one per vlm_class), a session, filtered queries
                                 with their answers (queries.json), including a 400 error
  images/w480/<segment_id>.jpg   every frame, 480 px      (= /gis/image/<id>?w=480)
  images/w1280/<segment_id>.jpg  every frame, 1280 px     (= /gis/image/<id>?w=1280)
  images/full/                   10 full-resolution PNGs  (= /gis/image/<id>)
  schema.sql                     database DDL (reference only)
  manifest.json                  every file with bytes and SHA-256
```

Run:

```
cd gis_sample
python mock_server.py --app ..\webgis        # your app at http://127.0.0.1:8765/gis/app/
                                             # API at http://127.0.0.1:8765/gis/meta
```

The mock answers the same URLs, filters, paging, errors and JSON shapes as the real server, because
`gis_core.py` is the server's own filter/aggregate code (our test suite checks mock and server give
identical answers). Differences: `generated_at` is the export time; previews exist only at 480 and
1280 px (`w` <= 480 gets 480, larger gets 1280); the full PNG exists for 10 segments (others get the
1280 JPEG). `python -m http.server` can serve the files under `api/` but cannot filter, page or map
`?w=` image URLs, so use `mock_server.py`. If you prefer a JS dev server (Vite etc.), run it on another
port and point `apiBase` at `http://127.0.0.1:8765` (the mock sends `Access-Control-Allow-Origin: *`);
switch back to `""` before delivery.

**URL mapping (for understanding only; your code uses the API URLs):**
`/gis/image/<id>?w=480` -> `images/w480/<id>.jpg`; `?w=1280` -> `images/w1280/<id>.jpg`;
`/gis/video/segments.geojson?geom=point` -> `api/gis/video/segments_point.geojson` then filtered.

## 8. Two layers, one map

- **Same marker semantics** on both layers: colour by `marker_label` (from `/gis/meta`), radius by
  `final_severity`, Normal green and small, `pending`/`unknown` grey. Distinguish the layers by marker
  *shape*: citizen = circle; video = the footprint polygon at street zoom and a small square/diamond at
  city zoom (use `geom=point` or the polygon centroid). A legend explains colours, sizes and shapes.
- **Video-specific layers** (each toggleable):
  1. *Condition ribbon*: footprint polygons filled by class/label (street zoom, >= 16).
  2. *Drive track*: `sessions?track=true` lines, one colour per drive, thin, under everything.
  3. *Coverage*: segments drawn with `geom=line` coloured by coverage: classified (by class colour),
     not sent (light grey), pending (grey dashed), **not covered (dark, dashed: "no data", never shown
     as Normal)**.
- **Citizen layer**: points from `/gis/citizen/assessments.geojson`; clustering; `include_review`
  toggle; same detail panel layout.
- One filter bar drives both layers (date, label, severity, review state, min confidence); session /
  status / gate filters apply to the video layer only. The stats bar shows video and citizen counts
  separately and never adds them up (a segment is 5 m of road; an upload is one photo).

## 9. Required features

**Parity with the citizen WebGIS brief (all required):**
1. Leaflet + OpenStreetMap (or MapLibre) map of Bengaluru, initial view = data `bbox` from `/gis/meta`
   (fallback 12.9716, 77.5946).
2. Markers coloured by distress type and sized by severity (section 8), legend.
3. Filters: date range, distress type (multi-select, grouped by IRC category, from `/gis/meta`),
   severity (multi-toggle), review state (all / needs review / expert-reviewed), "high confidence only
   (>= 0.80)".
4. Detail panel on click: the **image** (`preview_url`, with "Open full resolution" -> `image_url` in a
   new tab), final labels (main = `final_primary_label`, others beside it, expert badge when corrected:
   show the model's original labels struck through), severity, description, confidence bar with the
   0.80 line, review state, local date/time, position with ± sigma; for video also session/road,
   distance from start, gate status, frame quality.
5. Heatmap toggle (weight by severity: Low 1, Medium 2, High 3; Normal 0).
6. Clustering at low zoom (citizen points and video points).
7. CSV export of the current filter (video: `/gis/video/segments.csv?...`; citizen: build it in the
   browser from the loaded features with the same column idea).
8. Bengaluru ward boundaries overlay (public BBMP ward GeoJSON, loaded from a local file you ship in
   `webgis/data/`), with per-ward counts / distressed length on hover.
9. Stats bar from `/gis/video/summary` (+ citizen counts), updating with the filters.
10. Works on a phone (single column, bottom-sheet detail panel).

**Video-specific (required):**
11. Session / drive filter: list from `/gis/video/sessions` (road, date, length, % distressed, state);
    selecting zooms to its `bbox`.
12. **Segment strip view**: for the selected drive, a horizontal strip of `thumb_url` images in `k`
    order (virtualised: render only what is visible), each with a coloured class bar under it;
    clicking a thumbnail selects the segment on the map and vice versa; keyboard left/right steps.
13. **Play along route**: play/pause that steps through the drive's segments (adjustable speed, e.g.
    1–5 segments/s), moving a car marker along the track, showing the current frame in the detail panel
    and highlighting the footprint; preload the next few previews.
14. **% of length per label per km**: for the selected drive, a chart/table from `per_km` (stacked bar
    per km: distress labels in metres, Normal, not covered) and the drive's `labels` table
    (% of surveyed length); for all drives together, from `/gis/video/summary`.
15. Gate status display: filter by `status`; show `gate_decision_if_gated` when `gate_mode = "all"`
    (label it "detector would have: send/skip/audit").
16. Coverage display (section 8, item 3) with surveyed vs not-covered metres in the stats.

**Nice to have:** time slider, URL that encodes the current view and filters (shareable link),
print/PDF view of a drive report (map + per-km table), dark basemap toggle.

## 10. UI/UX expectations

- Audience: road engineers. Dense, calm, legible; no gradients or animations for decoration. Your own
  design language is fine (the other project UIs use a monochrome style with green/red/amber accents
  and an "SK Modernist"-like sans; matching it is welcome, not required).
- Vanilla HTML/CSS/JS with no build step is preferred (all our UIs are), served as static files.
  Libraries from a CDN with pinned versions (Leaflet 1.9.x, leaflet.markercluster, leaflet.heat,
  Chart.js or similar). If you use a framework, commit the **built** static output into `webgis/`.
- Always state uncertainty honestly: "model result, awaiting expert review" badges; ±m on positions;
  "not covered" is never drawn like "no distress".
- Empty states and errors: a 503 shows "Survey data temporarily unavailable" and retries every 30 s;
  a failed image shows a placeholder, never a broken-image icon.
- Accessibility: colour is never the only cue (labels in tooltips/legend), keyboard navigable strip,
  contrast AA.

## 11. Performance limits

- Now: 813 segments, ~1.8 MB GeoJSON with footprints. Plan for **20,000 segments** and 5,000 citizen
  points.
- Load segments **per selected drive(s) or date range**, not everything at once, when the total exceeds
  ~5,000 (`numberMatched` tells you); use `geom=point` at zoom < 16 and footprints only at street zoom.
- Filter client-side on the loaded features for instant response (the property names are flat); use
  server filters for the initial load and for CSV.
- Images: never load `image_url` (5–9 MB PNG) in lists, popups, the strip or while playing. Thumbnails
  (`?w=480`) for lists/strip/popups, previews (`?w=1280`) for the detail panel; lazy-load
  (`loading="lazy"`/IntersectionObserver); at most ~6 concurrent image requests.
- Render polygons on a canvas renderer (`L.canvas()` / `preferCanvas: true`).
- Do not poll faster than every 60 s.

## 12. What NOT to do

- No writes of any kind (there is no write endpoint; do not ask for one). Expert review happens in a
  separate tool.
- No secrets in the browser, no API keys, no Supabase keys, no database connection strings. You never
  need them: everything comes from `/gis/*`.
- No direct database or Supabase access from the browser; only the `/gis/*` API.
- Do not load full-resolution PNGs except via the explicit "Open full resolution" link.
- Do not hard-code labels, colours, sizes or thresholds: read `/gis/meta`.
- Do not show heights as MSL, do not present positions as more precise than ±`position_sigma_m`, do not
  show `not_covered` as Normal, do not sum segments and citizen uploads into one count.
- Do not change anything in the `gis_sample` files or depend on file paths inside it: use API URLs.
- Do not add a backend of your own; if something is missing from the API, write it down in your README
  ("API requests") and we add it.

## 13. Acceptance tests (we run these on the VLM PC with the real data)

1. Served at `http://<vlm-pc>:8000/gis/app/` with `apiBase = ""`: loads with no console errors; works
   identically against `mock_server.py --app` (same features, same counts).
2. Initial view fits the data bbox; all 3 drives listed with road, date, length, % distressed equal to
   `/gis/video/sessions`.
3. With no filter the stats bar equals `/gis/video/summary` (813 segments, 4,060 m, 20.1 % distressed,
   389 awaiting review on the sample).
4. Filter label = Potholes shows exactly 131 video segments (sample); Medium+High distress = 75;
   confidence >= 0.80 distress = 47; needs review = 389; the counts shown equal the API's
   `numberMatched` for the same query.
5. Clicking any segment shows its 1280 preview within 1 s on the LAN, the final labels, severity,
   confidence bar with the 0.80 line, review badge, local time, ±m; "Open full resolution" opens the PNG.
6. A segment without an image (`skip`/`not_covered`: none in the current sample, so test it by editing
   a copy of `segments.geojson` to set `has_image: false` and the three URLs to `null`) shows the status
   and no broken image; a `pending` segment (frame, no answer yet) shows the frame and "awaiting
   classification".
7. Strip view: scrolls smoothly through the 528-segment drive; selecting a thumbnail and a map segment
   stay in sync; left/right keys step.
8. Play along route runs through a drive at 5 segments/s without loading any full PNG (check the
   network panel: only `?w=` image requests).
9. Per-km chart for `..._d47c` shows 3 km bins whose metres sum to the drive's `surveyed_m` (2,637.5 m).
10. CSV export downloads `/gis/video/segments.csv` with the current filters; opens in Excel with correct
    characters (§ in descriptions).
11. Citizen layer toggles on/off; with "include pending review" off it shows 131 points, on 344
    (sample); a citizen point's detail panel shows its Cloudinary image.
12. Ward overlay shows ward names and counts on hover.
13. Phone (375 px wide): map, filters and detail panel usable; no horizontal scroll.
14. With the API stopped: a clear "data unavailable" message, no crash; it recovers when the API returns.
15. No secret, key or absolute local path anywhere in `webgis/` (we grep).

## 14. Deliverables

1. A `webgis/` folder: `index.html` (entry), `config.js` (`window.GIS_CONFIG = { apiBase: "" }`), your
   JS/CSS, `data/` (ward GeoJSON with its source and licence noted), vendored or CDN libraries with
   pinned versions. Static files only; relative paths only (it is served under `/gis/app/`).
2. `webgis/README.md`: how to run with the mock, what is done vs optional, known limitations, any "API
   requests", and screenshots (desktop + phone) of the map, detail panel, strip view and per-km chart.
3. A short test log: the acceptance tests above run against the mock, with results.

## 15. How we integrate it on the VLM PC

1. Copy your `webgis/` folder into the Capstone repository root (`Capstone/webgis/index.html`).
2. The FastAPI server mounts it automatically at `/gis/app/` on its next start (`app/gis_routes.py`
   `install()`: it serves `webgis/` when `index.html` exists; `GIS_APP_DIR` can point elsewhere).
3. `apiBase` stays `""`; the API is the same origin. Open `http://<vlm-pc>:8000/gis/app/` (or the
   Cloudflare tunnel URL + `/gis/app/`).
4. We run the acceptance tests and our API test suite (`scripts/tests/test_gis_routes.py`, which also
   checks that the mock and the server answer identically). Expected changes on our side: none, or
   small API additions you list under "API requests".

## 16. Ask the user before you start

1. Leaflet + OSM tiles OK, or another basemap?
2. Which ward boundary file to use (BBMP 198 wards vs the newer delimitation)?
3. Scope of nice-to-haves (time slider, shareable URLs, drive report print view).
4. Whether the citizen layer should show `expert_review` rows by default.
