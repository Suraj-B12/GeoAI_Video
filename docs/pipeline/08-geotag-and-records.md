# 08 · Geotag and Records

> **Implemented (pipeline v0.1):** crops are **PNG** (not JPEG; see [07](07-best-frame-and-crop.md)). EXIF is written with Pillow (GPS = camera, GPSDest = anomaly, IST + UTC times, ImageUniqueID, Orientation = 1), plus a `geoai` PNG text chunk with the full record, a sidecar JSON, `candidates.csv` (UTF-8 with BOM) and `anomalies.geojson`. The SQLite database and ExifTool/XMP below are not used yet. `GPSAltitude` is written only from the phone's MSL altitude, never the ellipsoidal height.

Goal: every crop carries its exact capture time and location (both the camera's and the distress's), readable by GIS tools, the VLM machine, and the database. Metadata never touches the pixels.

## 1. What "time" and "position" mean for a crop

| Field | Definition |
|---|---|
| Capture time | The chosen observation's **row time** at the distress box centre ([02](02-time-and-trajectory.md) §6), converted to UTC with the session's UTC fit. Millisecond resolution |
| Camera position | Trajectory at that time: lat, lon, heading, speed, 1σ horizontal uncertainty, and both heights (below) |
| Height | Ellipsoidal height (WGS-84, `alt_ellipsoid_m`) and mean-sea-level altitude. MSL comes from `alt_msl_m` when the phone recorded it (API 34+). Otherwise it is the ellipsoidal height minus the EGM2008 geoid undulation, from a geoid grid file stored locally. In India the geoid lies tens of metres below the ellipsoid, up to about 100 m in the south, so **the two heights must never be mixed up** |
| Distress position | Distress centroid ([06](06-association.md) §7): median of observation centroids, lat/lon, with its own 1σ |
| Distance and bearing camera → distress | From the two positions; typically 0.5–4 m |

Both positions are always stored. GIS consumers want the **distress** position; the EXIF standard defines `GPSLatitude/Longitude` as where the **camera** was. The EXIF `GPSDest*` tags exist exactly for "the position of the photographed subject", so we use them.

## 2. EXIF and XMP (written with ExifTool)

| Tag | Value |
|---|---|
| `DateTimeOriginal`, `SubSecTimeOriginal`, `OffsetTimeOriginal` | Local IST (`+05:30`), milliseconds |
| `GPSDateStamp`, `GPSTimeStamp` | UTC date and time (with fractional seconds) |
| `GPSLatitude/Ref`, `GPSLongitude/Ref`, `GPSMapDatum = WGS-84` | Camera position |
| `GPSAltitude`, `GPSAltitudeRef` (0 above / 1 below sea level) | **MSL** altitude of the camera. Omit both if no MSL value is available; never write the ellipsoidal height here |
| `GPSHPositioningError` | Camera 1σ horizontal error (m) |
| `GPSSpeed` (`GPSSpeedRef = K`), `GPSTrack` (`GPSTrackRef = T`) | Speed (km/h) and direction of travel (true north) |
| `GPSImgDirection` (`GPSImgDirectionRef = T`) | Camera direction = heading + mount yaw (≈ heading for a forward camera, heading + 180° for a rearward one) |
| `GPSDestLatitude/Ref`, `GPSDestLongitude/Ref` | **Distress** position |
| `GPSDestBearing` (`Ref = T`), `GPSDestDistance` (`Ref = K`, in km) | Camera → distress |
| `ImageUniqueID` | `distress_id` (UUID without dashes, 32 hex characters) |
| `Make`, `Model`, `Software` | Phone make and model; `GeoAI pipeline <version>` |
| `ColorSpace` | 1 (sRGB) |
| XMP `geoai:` namespace | `distress_id, session_id, segment, frame_in_segment, frame_global, t_ns, utc_ms, detector_model_id, detector_class, detector_score, crop_box, distress_box, scale, truncated, gsd_mm_px, crop_width_m, crop_height_m, area_m2, length_m, distress_sigma_m, ellipsoidal_height_m, msl_source (phone \| egm2008 \| none), long_group_id, piece_index, intrinsics_id, mount_calibration_id, pipeline_version` |

Writing rules (ExifTool):
- Define the `geoai` namespace in a config file (`config/exiftool_geoai.config`), passed with `-config`.
- Write numbers with the per-tag `#` suffix, e.g. `-GPSLatitude#=12.9715987`. It turns off ExifTool's print conversion for that tag only, so no value goes through a degrees-minutes text form.
- Always write the `…Ref` tags explicitly (`N`/`S`, `E`/`W`) and the absolute value in the main tag.
- Use `-stay_open` batch mode (pyexiftool does this). Starting ExifTool once per file costs about 0.2 s each.
- Read every crop back with `-n` and compare with the intended values. A crop whose metadata fails this check is an error, not a warning.

## 3. Sidecar JSON

`sidecars/<distress_id>.json` holds everything in §2, plus the scoring terms from [07](07-best-frame-and-crop.md), the class vote, and the list of observation frames. Sidecars travel with crops to the VLM machine ([09](09-vlm-interface.md)), so nothing depends on another program's EXIF reader.

## 4. Database (SQLite, `db/geoai.sqlite`, WAL mode)

```sql
CREATE TABLE sessions (session_id TEXT PRIMARY KEY, session_type TEXT, device_id TEXT, mount_id TEXT,
  road_id TEXT, direction TEXT, lane TEXT, start_chainage_km REAL, start_utc TEXT, end_utc TEXT,
  duration_s REAL, distance_m REAL, trusted INTEGER, qc_status TEXT, spec_version TEXT, app_version TEXT,
  intrinsics_ids TEXT, mount_calibration_id TEXT, latency_s REAL, ingested_at TEXT);

CREATE TABLE distresses (distress_id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions,
  detector_class TEXT, class_shares TEXT, n_obs INTEGER, single_observation INTEGER,
  long_group_id TEXT, piece_index INTEGER,
  utc TEXT, t_ns INTEGER, frame_global INTEGER, segment INTEGER, frame_in_segment INTEGER,
  cam_lat REAL, cam_lon REAL, cam_h_ell_m REAL, cam_alt_msl_m REAL, cam_sigma_m REAL,
  heading_deg REAL, speed_mps REAL,
  lat REAL, lon REAL, sigma_m REAL, area_m2 REAL, length_m REAL, lateral_m REAL, s_m REAL,
  chainage_km REAL, gsd_mm_px REAL, truncated INTEGER, possible_duplicate_of TEXT,
  crop_path TEXT, crop_sha256 TEXT, context_path TEXT, flags TEXT,
  detector_model_id TEXT, pipeline_version TEXT, created_at TEXT);

CREATE TABLE observations (distress_id TEXT REFERENCES distresses, frame_global INTEGER,
  cls TEXT, score REAL, box TEXT, range_m REAL, gsd_mm_px REAL, chosen INTEGER);

CREATE TABLE classifications (classification_id TEXT PRIMARY KEY, distress_id TEXT REFERENCES distresses,
  request_id TEXT, model_id TEXT, prompt_version TEXT, label TEXT, severity TEXT, score REAL,
  rationale TEXT, raw_output TEXT, valid INTEGER, created_at TEXT);   -- append-only, never updated

CREATE TABLE reviews (review_id TEXT PRIMARY KEY, distress_id TEXT REFERENCES distresses,
  reviewer TEXT, decision TEXT, label TEXT, severity TEXT, note TEXT, created_at TEXT);  -- append-only

CREATE TABLE models (model_id TEXT PRIMARY KEY, kind TEXT, path TEXT, sha256 TEXT, card TEXT, added_at TEXT);
```

- **Append-only** classifications and reviews: a new VLM version or a reviewer's correction adds rows. Queries choose the latest by rule (reviewer decision > latest valid VLM result for the chosen model).
- `distress_id` is a UUIDv7 (time-ordered, RFC 9562), generated once and never reused. Python's `uuid.uuid7()` exists only from 3.14; on 3.11/3.12 use the `uuid6` package from the wheelhouse.
- Times in `TEXT` columns are ISO 8601 UTC with milliseconds and a `Z` suffix. IST appears only in EXIF and in reports.
- `chainage_km` is filled only when a road centreline and km-stone markers allow it ([10](10-review-and-export.md) §2).

## 5. Tests

- Write and read back all EXIF tags on a test crop; lat/lon round-trip error < 1e-7°.
- `GPSDest*` and `GPSLatitude/Longitude` differ by the expected camera → distress distance (± 1 cm).
- `exiftool -G1 -a` on a crop shows the `XMP-geoai` group with every field.
- IST and UTC tags differ by exactly 5 h 30 min.
- A session without `alt_msl_m` gets `GPSAltitude` from the geoid grid (`msl_source = egm2008`). On a known benchmark, `GPSAltitude` and `ellipsoidal_height_m` differ by the published undulation ± 0.5 m.
- A southern-hemisphere / western-longitude test point (signs flipped) round-trips with the correct `…Ref` letters.
