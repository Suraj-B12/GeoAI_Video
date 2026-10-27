"""Segment-mode records (schema `geoai.segment.v1`, docs/pipeline/13 §5): building, checking, writing.

There is one record per road segment, whatever happened to it:
- `send` and `audit` segments have their best frame on disk (`frames/<id>.png`, full resolution,
  lossless) and go to the VLM;
- `skip` segments were checked by the gate, which found nothing, and are not sent;
- `not_covered` segments had no usable frame (GNSS gap, black frames, standing still).

All statuses share one schema, so "% of surveyed length" can be computed from the records alone: the
denominator never has to be rebuilt from logs.

Where the numbers come from (the orchestrator fills them in from the pipeline objects):
- positions on the path (segment start, centre and end) and the camera pose: `path_points` and
  `camera_pose` on the Trajectory;
- the footprint: SegmentGeometry.footprint (a ground ring in the camera frame), placed on the map with
  `footprint_lonlat`;
- the frame: its frames.csv row and the selector's FrameScore (rows of the segment, distances ahead);
- quality: SegmentResult.quality(); gate: the decision dict of gate.py.
`build_segment_record` assembles these and checks the result. Nothing here reads video or runs a model.

Imports. `vlm-import` and `analyze` also use this module on the VLM machine, where only numpy and
pandas are installed. So Pillow (records.py) and OpenCV are imported only inside the functions that
write or read images, never at module level.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from . import __version__
from .config import Config
from .util import canonical_json, sha256_file, to_json, write_json_atomic

if TYPE_CHECKING:  # only for annotations: trajectory.py pulls in scipy and pymap3d
    from .trajectory import Trajectory

SCHEMA = "geoai.segment.v1"
KIND = "segment"
STATUSES = ("send", "audit", "skip", "not_covered")
SENT_STATUSES = ("send", "audit")      # statuses whose frame is written and sent to the VLM
GATE_MODES = ("gate", "all")
GATE_DECISIONS = ("send", "audit", "skip")

# Field names of every block, in the order of the spec. A record has exactly these keys: the validator
# rejects missing and unknown ones, so a typo can't turn into a silent extra column.
TOP_FIELDS = ("id", "kind", "schema", "status", "session_id", "pipeline_version", "config_sha256", "segment",
              "frame", "camera", "target", "capture", "quality", "gate", "survey", "device", "geometry", "flags")
SEGMENT_FIELDS = ("k", "length_m", "s0_m", "s1_m", "partial", "chainage_start_km", "chainage_end_km", "start",
                  "end", "centre", "footprint_lonlat")
FRAME_FIELDS = ("file", "sha256", "width", "height", "segment", "frame_in_segment", "frame_global", "t_ns",
                "utc_ms", "utc_iso", "local_iso", "rows_of_segment", "near_edge_ahead_m", "far_edge_ahead_m")
CAMERA_FIELDS = ("lat", "lon", "alt_ellipsoid_m", "alt_msl_m", "heading_deg", "view_direction_deg", "speed_mps",
                 "h_sigma_m", "utc_ms")
TARGET_FIELDS = ("lat", "lon", "bearing_deg", "distance_m", "sigma_m")
CAPTURE_FIELDS = ("exposure_ns", "iso", "frame_duration_ns", "rs_skew_ns", "focus_diopters", "focus_mode",
                  "ois_mode", "low_light")
QUALITY_FIELDS = ("sharpness", "sharpness_rank", "n_candidates", "blur_motion_px", "blur_rotation_px", "blur_px",
                  "gyro_rad_s", "reason")
GATE_FIELDS = ("mode", "decision", "decision_if_gated", "reasons", "scores", "thresholds", "audit_draw")
GATE_SCORE_FIELDS = ("probe_frac", "probe_max_logit", "evidence", "crack_mm", "z_max", "anomalous_patch_frac",
                     "n_patches")
SURVEY_FIELDS = ("road_id", "road_name", "direction", "lane", "start_chainage_km", "operator", "vehicle", "weather")
DEVICE_FIELDS = ("manufacturer", "model", "device_id")
GEOMETRY_FIELDS = ("source", "pitch_deg", "roll_deg", "height_m")
# gate.thresholds holds numbers (or null = signal off), except these provenance texts (gate.gate_thresholds): where
# the thresholds came from (config | calibration) and which calibration file, so every record names its decider.
THRESHOLD_TEXT_FIELDS = ("source", "calibration_file", "calibration_sha256")
# Added by vlm-import, only to segments the VLM answered: the spec's seven fields, the second confidence
# and the review flag that /classify also returns, and the batch the answer came from.
VLM_FIELDS = ("labels", "stage1_label", "stage1_confidence", "severity", "description", "valid", "problem",
              "stage2_confidence", "needs_expert_review", "batch_id")
# Flags, in the order they are listed in a record:
#   partial             the shorter last segment of a recording (its real length is used everywhere)
#   low_light           taken under the app's low-light exposure
#   over_blur_limit     predicted blur above segments.max_blur_px (no sharper frame was available)
#   geometry_nominal    camera model from the session's own metadata, not a calibration
#   latency_uncorrected GNSS latency could not be estimated for this session
#   heading_from_gyro   heading from integrated gyro (too slow for a GNSS course)
#   no_utc              no UTC fit: times are on the session clock only
FLAGS = ("partial", "low_light", "over_blur_limit", "geometry_nominal", "latency_uncorrected", "heading_from_gyro",
         "no_utc")

_CHAINAGE_SIGN = {"increasing_chainage": 1.0, "decreasing_chainage": -1.0}
# Config fields holding file paths, reduced to file names for the config hash (see config_digest).
_CONFIG_PATHS = (("anomaly", "weights"), ("anomaly", "probe"), ("gate", "calibration"))
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SegmentLike(Protocol):
    """What `build_segment_record` reads from a segment (segments.Segment has exactly these fields)."""
    k: int
    s0: float
    s1: float
    partial: bool


def segment_id(session_id: str, k: int) -> str:
    """Record id `<session_id>_seg<k:05d>`. Ids of one session sort in route order."""
    if int(k) < 0:
        raise ValueError(f"segment index must be >= 0, got {k}")
    return f"{session_id}_seg{int(k):05d}"


def config_digest(cfg: Config) -> str:
    """SHA-256 of the settings (canonical JSON of `cfg.to_dict()`), recorded in every record.

    File paths (model weights, probe, gate calibration) are reduced to their file names, so the hash
    identifies the settings and not the folder the pipeline is installed in. The weights' own SHA-256
    is a setting (anomaly.weights_sha256), and each record carries the gate thresholds actually used.
    """
    d = cfg.to_dict()
    for section, key in _CONFIG_PATHS:
        v = d[section][key]
        if v:
            d[section][key] = PureWindowsPath(str(v)).name  # splits on both / and \
    return hashlib.sha256(canonical_json(d)).hexdigest()


# ---- positions -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class PathPoint:
    """A point of the GNSS path at path distance `s_m`: local ENU (m), WGS-84, and the trajectory there."""
    s_m: float
    e: float
    n: float
    lat: float
    lon: float
    h_sigma_m: float
    valid: bool        # inside the trajectory and not in a GNSS gap


def path_points(traj: Trajectory, s_values) -> list[PathPoint]:
    """Positions on the path at path distances `s_values` (m, on the Trajectory's `s` axis).

    `s` is integrated speed, so it stays flat while the vehicle stands still. It is inverted on the grid
    points where it increases, i.e. at the moment each distance is first reached. Between grid points
    (10 ms apart, under 0.2 m at 60 km/h) the position is interpolated. Beyond either end of the
    trajectory a point is extrapolated along the end heading and marked invalid: the last segment's
    far edge lies up to near_m beyond the last camera position.
    """
    s_q = np.atleast_1d(np.asarray(s_values, dtype=np.float64))
    s = np.asarray(traj.s, dtype=np.float64)
    knot = np.concatenate([[True], np.diff(s) > 0])
    sk, tk = s[knot], np.asarray(traj.t, dtype=np.float64)[knot]
    inside = np.clip(s_q, sk[0], sk[-1])
    p = traj.sample(np.interp(inside, sk, tk))
    beyond = s_q - inside  # metres past the start (< 0) or the end (> 0) of the trajectory
    psi = np.radians(p["heading"])
    e = p["e"] + beyond * np.sin(psi)
    n = p["n"] + beyond * np.cos(psi)
    lat, lon, _ = traj.enu_to_geodetic(e, n)
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
    valid = p["valid"] & (beyond == 0)
    return [PathPoint(float(s_q[i]), float(e[i]), float(n[i]), float(lat[i]), float(lon[i]), float(p["h_sigma"][i]),
                      bool(valid[i])) for i in range(len(s_q))]


@dataclass(frozen=True)
class CameraPose:
    """The camera at one instant: the exposure middle of the image's centre row (extract._RowClock)."""
    t_s: float                 # trajectory seconds
    s_m: float                 # path distance: the frame's camera position c
    e: float
    n: float
    lat: float
    lon: float
    alt_ellipsoid_m: float
    heading_deg: float         # vehicle heading, clockwise from north
    speed_mps: float
    h_sigma_m: float
    utc_ms: float | None       # None without a UTC fit
    valid: bool
    heading_from_gyro: bool


def camera_pose(traj: Trajectory, t_s: float) -> CameraPose:
    """The trajectory sampled at `t_s`, with WGS-84 position and UTC."""
    t = float(t_s)
    p = traj.sample(np.array([t]))
    lat, lon, alt = (np.atleast_1d(v) for v in traj.enu_to_geodetic(p["e"], p["n"], p["u"]))
    utc = float(traj.utc.utc_ms(t)) if traj.utc is not None else None
    return CameraPose(t, float(p["s"][0]), float(p["e"][0]), float(p["n"][0]), float(lat[0]), float(lon[0]),
                      float(alt[0]), float(p["heading"][0]), float(p["speed"][0]), float(p["h_sigma"][0]), utc,
                      bool(p["valid"][0]), bool(p["heading_from_gyro"][0]))


def footprint_lonlat(traj: Trajectory, ring_xy, pose: CameraPose) -> list[list[float]]:
    """A ground ring (x right, y ahead of the camera nadir, m) as `[[lon, lat], ...]`.

    The ring is rotated by the vehicle heading ψ (clockwise from north) and moved to the camera:
    E = e + y·sin ψ + x·cos ψ, N = n + y·cos ψ − x·sin ψ, as for crop footprints (extract.py). A
    rotation keeps the orientation, so a counter-clockwise ring (SegmentGeometry.footprint) stays
    counter-clockwise, as RFC 7946 wants for exterior rings. For a not-covered segment, pass the pose
    the camera had (or would have had) at the segment's candidate window: the shape is then the view
    the segment would have been classified from.
    """
    xy = np.asarray(ring_xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 4 or not np.array_equal(xy[0], xy[-1]):
        raise ValueError("a footprint ring needs at least 3 corners and must be closed (first point repeated last)")
    psi = math.radians(pose.heading_deg)
    E = pose.e + xy[:, 1] * math.sin(psi) + xy[:, 0] * math.cos(psi)
    N = pose.n + xy[:, 1] * math.cos(psi) - xy[:, 0] * math.sin(psi)
    lat, lon, _ = traj.enu_to_geodetic(E, N)
    return [[float(a), float(b)] for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))]


# ---- building ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameShot:
    """The chosen frame of a covered segment, as the selector measured it.

    row: its frames.csv row (a dict of Session.frames). width, height: decoded size, display orientation.
    rows_of_segment: (v_top, v_bottom), the display rows of the segment's far and near edge on the centre
    line (SegmentGeometry.rows_for: floats, not clipped to the image).
    near_edge_ahead_m, far_edge_ahead_m: s0 − c and s1 − c for the frame's camera position c.
    low_light: Session.low_light_at(t_ns) for this frame.
    """
    row: Mapping[str, Any]
    width: int
    height: int
    rows_of_segment: tuple[float, float]
    near_edge_ahead_m: float
    far_edge_ahead_m: float
    low_light: bool


def geometry_block(model) -> dict:
    """The record's `geometry` block from a CameraModel (geometry.py)."""
    return {"source": str(model.source), "pitch_deg": float(model.pitch_deg), "roll_deg": float(model.roll_deg),
            "height_m": float(model.height_m)}


def build_segment_record(*, session_id: str, session_meta: Mapping[str, Any], segment: SegmentLike,
                         start: PathPoint, centre: PathPoint, end: PathPoint, footprint: Sequence[Sequence[float]],
                         shot: FrameShot | None, camera: CameraPose | None, quality: Mapping[str, Any],
                         gate: Mapping[str, Any] | None, geometry: Mapping[str, Any], latency_method: str,
                         chainage_origin_m: float | None, cfg: Config, config_sha256: str) -> dict:
    """One `geoai.segment.v1` record (docs/pipeline/13 §5).

    Covered segment: `shot`, `camera` and `gate` are all given, and the status is `gate["decision"]`
    (send | audit | skip). Not covered: all three are None. The status is then `not_covered`, and frame,
    camera, capture and gate are null. The target keeps the segment centre, without bearing or distance
    (there is no camera position).

    frame.file is "frames/<id>.png" for send and audit (the caller writes the PNG there, e.g. with
    records.write_png) and null for skip. frame.sha256 stays null for the caller to fill in from the
    written file: the PNG embeds this record, so the record can't contain the PNG's own hash.

    footprint: the closed [[lon, lat], ...] ring from `footprint_lonlat`.
    quality: SegmentResult.quality(); gate: gate.py's decision, exactly the fields of the schema.
    chainage_origin_m: the camera's path distance s at the first frame of the recording, where the
        survey form's start chainage applies. Chainage = start ± (s − origin) / 1000 (minus for
        `decreasing_chainage`). It is null without a start chainage, a known direction or an origin, and
        it is not corrected at km stones.
    Raises ValueError when the inputs are inconsistent or the result fails `validate_segment_record`.
    """
    covered = shot is not None
    if not (covered == (camera is not None) == (gate is not None)):
        raise ValueError("a covered segment needs shot, camera and gate; a not-covered one none of them")
    if set(quality) != set(QUALITY_FIELDS):
        raise ValueError(f"quality must have exactly the fields {QUALITY_FIELDS}, got {sorted(quality)}")
    rid = segment_id(session_id, segment.k)
    s0, s1 = float(segment.s0), float(segment.s1)
    status = str(gate["decision"]) if covered else "not_covered"

    survey_meta = session_meta.get("survey") or {}
    start_km = _num(survey_meta.get("start_chainage_km"))
    sign = _CHAINAGE_SIGN.get(survey_meta.get("direction"))
    origin = _num(chainage_origin_m)
    ch0 = ch1 = None
    if start_km is not None and sign is not None and origin is not None:
        ch0 = start_km + sign * (s0 - origin) / 1000.0
        ch1 = start_km + sign * (s1 - origin) / 1000.0
    seg_block = {"k": int(segment.k), "length_m": s1 - s0, "s0_m": s0, "s1_m": s1, "partial": bool(segment.partial),
                 "chainage_start_km": ch0, "chainage_end_km": ch1, "start": _latlon(start), "end": _latlon(end),
                 "centre": _latlon(centre), "footprint_lonlat": [[float(x), float(y)] for x, y in footprint]}
    target = {"lat": float(centre.lat), "lon": float(centre.lon), "bearing_deg": None, "distance_m": None,
              "sigma_m": _num(centre.h_sigma_m)}
    flags = set()
    if segment.partial:
        flags.add("partial")
    if geometry.get("source") == "nominal":
        flags.add("geometry_nominal")
    if latency_method == "none":
        flags.add("latency_uncorrected")

    frame = cam = capture = gate_block = None
    if covered:
        from .records import utc_strings  # Pillow; see the module docstring

        row = shot.row
        utc = _num(camera.utc_ms)
        times = utc_strings(utc, cfg.output.utc_offset_minutes) if utc is not None else {}
        frame = {"file": f"frames/{rid}.png" if status in SENT_STATUSES else None, "sha256": None,
                 "width": int(shot.width), "height": int(shot.height), "segment": _int(row.get("segment")),
                 "frame_in_segment": _int(row.get("frame_in_segment")), "frame_global": _int(row.get("frame_global")),
                 "t_ns": _int(row.get("t_ns")), "utc_ms": utc, "utc_iso": times.get("utc_iso"),
                 "local_iso": times.get("local_iso"),
                 "rows_of_segment": [float(shot.rows_of_segment[0]), float(shot.rows_of_segment[1])],
                 "near_edge_ahead_m": float(shot.near_edge_ahead_m), "far_edge_ahead_m": float(shot.far_edge_ahead_m)}
        rearward = str((session_meta.get("mount") or {}).get("camera_facing") or "forward") == "rearward"
        cam = {"lat": camera.lat, "lon": camera.lon, "alt_ellipsoid_m": _num(camera.alt_ellipsoid_m),
               "alt_msl_m": None,  # no geoid model here; never the ellipsoidal height (docs/pipeline/08 §2)
               "heading_deg": _wrap360(camera.heading_deg),
               "view_direction_deg": _wrap360(camera.heading_deg + (180.0 if rearward else 0.0)),
               "speed_mps": camera.speed_mps, "h_sigma_m": camera.h_sigma_m, "utc_ms": utc}
        dE, dN = centre.e - camera.e, centre.n - camera.n
        target["bearing_deg"] = _wrap360(math.degrees(math.atan2(dE, dN)))
        target["distance_m"] = math.hypot(dE, dN)
        capture = {"exposure_ns": _int(row.get("exposure_ns")), "iso": _int(row.get("iso")),
                   "frame_duration_ns": _int(row.get("frame_duration_ns")), "rs_skew_ns": _int(row.get("rs_skew_ns")),
                   "focus_diopters": _num(row.get("focus_diopters")),
                   "focus_mode": (session_meta.get("capture_settings") or {}).get("focus_mode"),
                   "ois_mode": _int(row.get("ois_mode")), "low_light": bool(shot.low_light)}
        unknown = sorted(set(gate) - set(GATE_FIELDS))
        if unknown:
            raise ValueError(f"gate result has fields outside the schema: {unknown}")
        gate_block = {f: _plain(gate.get(f)) for f in GATE_FIELDS}
        if shot.low_light:
            flags.add("low_light")
        if camera.heading_from_gyro:
            flags.add("heading_from_gyro")
        if utc is None:
            flags.add("no_utc")
    quality_block = {f: _plain(quality[f]) for f in QUALITY_FIELDS}
    if covered and "over_blur_limit" in str(quality_block["reason"]):
        flags.add("over_blur_limit")

    device = session_meta.get("device") or {}
    rec = {
        "id": rid, "kind": KIND, "schema": SCHEMA, "status": status, "session_id": session_id,
        "pipeline_version": __version__, "config_sha256": config_sha256,
        "segment": seg_block, "frame": frame, "camera": cam, "target": target, "capture": capture,
        "quality": quality_block, "gate": gate_block,
        "survey": {f: _plain(survey_meta.get(f)) for f in SURVEY_FIELDS},
        "device": {f: _plain(device.get(f)) for f in DEVICE_FIELDS},
        "geometry": {f: _plain(geometry.get(f)) for f in GEOMETRY_FIELDS},
        "flags": [f for f in FLAGS if f in flags],
    }
    problems = validate_segment_record(rec, embedded=True)
    if problems:
        raise ValueError(f"segment record {rid} is invalid: " + "; ".join(problems))
    return rec


def _latlon(p: PathPoint) -> dict:
    return {"lat": float(p.lat), "lon": float(p.lon)}


def _wrap360(deg: float) -> float:
    """Angle in [0, 360). Python's % alone returns 360.0 for tiny negative angles (-1e-20 % 360)."""
    v = float(deg) % 360.0
    return 0.0 if v >= 360.0 else v


def _num(x) -> float | None:
    """A finite float, or None for missing values (None, NaN, pandas NA, text)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(x) -> int | None:
    v = _num(x)
    return None if v is None else int(round(v))


def _plain(x):
    """A JSON-ready copy: numpy scalars and arrays to Python values, NaN and infinities to None."""
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


# ---- checking ------------------------------------------------------------------------------------

def validate_segment_record(rec: Mapping[str, Any], *, embedded: bool = False) -> list[str]:
    """Problems of a record against `geoai.segment.v1`; an empty list means valid.

    Checks the exact key set of every block, types, ranges and the consistency rules: the id matches the
    session and k, length = s1 − s0, chainage spans the length, the status matches the gate (mode `all`
    sends everything; mode `gate` sends what it decides), frame/camera/capture/gate are null exactly for
    `not_covered`, sent segments name their frame file, the camera and frame times agree, the target is
    the segment centre, the footprint is a closed ring, flags agree with the record, and the whole
    record is JSON without NaN.
    embedded: the copy inside the PNG, which can't contain the PNG's own hash; frame.sha256 may then be null.
    """
    p: list[str] = []
    if not isinstance(rec, Mapping):
        return ["record is not an object"]
    _keys(p, "record", rec, TOP_FIELDS, optional=("vlm",))
    try:
        to_json(rec, indent=None)
    except (TypeError, ValueError) as e:
        p.append(f"record is not plain JSON ({e})")
    if rec.get("kind") != KIND:
        p.append(f"kind is {rec.get('kind')!r}, not {KIND!r}")
    if rec.get("schema") != SCHEMA:
        p.append(f"schema is {rec.get('schema')!r}, not {SCHEMA!r}")
    status = rec.get("status")
    if status not in STATUSES:
        p.append(f"status {status!r} is not one of {STATUSES}")
    if not isinstance(rec.get("session_id"), str) or not rec.get("session_id"):
        p.append("session_id must be a non-empty string")
    if not isinstance(rec.get("pipeline_version"), str):
        p.append("pipeline_version must be a string")
    if not (isinstance(rec.get("config_sha256"), str) and _HEX64.match(rec["config_sha256"])):
        p.append("config_sha256 must be 64 lowercase hex characters")
    covered = status != "not_covered"

    seg = rec.get("segment")
    length = None
    if _keys(p, "segment", seg, SEGMENT_FIELDS):
        k = seg.get("k")
        if not _is_int(k) or k < 0:
            p.append("segment.k must be an integer >= 0")
        elif isinstance(rec.get("session_id"), str) and rec.get("id") != segment_id(rec["session_id"], k):
            p.append(f"id {rec.get('id')!r} does not match session_id and segment.k")
        s0, s1 = seg.get("s0_m"), seg.get("s1_m")
        if _is_num(s0) and _is_num(s1) and s1 > s0:
            length = s1 - s0
            if not (_is_num(seg.get("length_m")) and abs(seg["length_m"] - length) <= 1e-6):
                p.append("segment.length_m is not s1_m - s0_m")
        else:
            p.append("segment.s0_m and s1_m must be numbers with s1_m > s0_m")
        if not isinstance(seg.get("partial"), bool):
            p.append("segment.partial must be true or false")
        ch0, ch1 = seg.get("chainage_start_km"), seg.get("chainage_end_km")
        if (ch0 is None) != (ch1 is None):
            p.append("segment chainage must be given at both ends or at neither")
        elif ch0 is not None:
            if not (_is_num(ch0) and _is_num(ch1)):
                p.append("segment chainage must be numbers")
            elif length is not None and abs(abs(ch1 - ch0) - length / 1000.0) > 1e-9:
                p.append("segment chainage does not span the segment length")
        for name in ("start", "end", "centre"):
            _latlon_ok(p, f"segment.{name}", seg.get(name))
        _ring_ok(p, seg.get("footprint_lonlat"))

    frame = rec.get("frame")
    if not covered:
        for name in ("frame", "camera", "capture", "gate"):
            if rec.get(name) is not None:
                p.append(f"{name} must be null for a not_covered segment")
    elif _keys(p, "frame", frame, FRAME_FIELDS):
        if status in SENT_STATUSES:
            if frame.get("file") != f"frames/{rec.get('id')}.png":
                p.append("frame.file must be frames/<id>.png for a sent segment")
            sha = frame.get("sha256")
            if sha is None and not embedded:
                p.append("frame.sha256 is missing for a sent segment")
            elif sha is not None and not (isinstance(sha, str) and _HEX64.match(sha)):
                p.append("frame.sha256 must be 64 lowercase hex characters")
        elif frame.get("file") is not None or frame.get("sha256") is not None:
            p.append("frame.file and frame.sha256 must be null for a skipped segment (no image is written)")
        for name in ("width", "height"):
            if not _is_int(frame.get(name)) or frame[name] <= 0:
                p.append(f"frame.{name} must be a positive integer")
        if not _is_int(frame.get("segment")) or frame["segment"] < 1:
            p.append("frame.segment (video segment) must be an integer >= 1")
        for name in ("frame_in_segment", "frame_global", "t_ns"):
            if not _is_int(frame.get(name)) or frame[name] < 0:
                p.append(f"frame.{name} must be an integer >= 0")
        utc = frame.get("utc_ms")
        if utc is not None and not _is_num(utc):
            p.append("frame.utc_ms must be a number or null")
        for name in ("utc_iso", "local_iso"):
            if (utc is None) != (frame.get(name) is None) or (utc is not None and not isinstance(frame[name], str)):
                p.append(f"frame.{name} must be a string exactly when utc_ms is known")
        rows = frame.get("rows_of_segment")
        if not (isinstance(rows, list) and len(rows) == 2 and all(_is_num(v) for v in rows) and rows[0] < rows[1]):
            p.append("frame.rows_of_segment must be [v_top, v_bottom] with v_top < v_bottom")
        near, far = frame.get("near_edge_ahead_m"), frame.get("far_edge_ahead_m")
        if not (_is_num(near) and _is_num(far) and near > 0):
            p.append("frame.near_edge_ahead_m and far_edge_ahead_m must be numbers, the near edge ahead of the camera")
        elif length is not None and abs((far - near) - length) > 1e-6:
            p.append("frame.far_edge_ahead_m - near_edge_ahead_m is not the segment length")

    cam = rec.get("camera")
    if covered and _keys(p, "camera", cam, CAMERA_FIELDS):
        _latlon_ok(p, "camera", cam)
        _num_ok(p, "camera.alt_ellipsoid_m", cam.get("alt_ellipsoid_m"), nullable=True)
        _num_ok(p, "camera.alt_msl_m", cam.get("alt_msl_m"), nullable=True)
        _num_ok(p, "camera.heading_deg", cam.get("heading_deg"), lo=0.0, below=360.0)
        _num_ok(p, "camera.view_direction_deg", cam.get("view_direction_deg"), lo=0.0, below=360.0)
        _num_ok(p, "camera.speed_mps", cam.get("speed_mps"), lo=0.0)
        _num_ok(p, "camera.h_sigma_m", cam.get("h_sigma_m"), lo=0.0)
        if isinstance(frame, Mapping) and cam.get("utc_ms") != frame.get("utc_ms"):
            p.append("camera.utc_ms differs from frame.utc_ms (both are the frame's centre-row time)")

    tgt = rec.get("target")
    if _keys(p, "target", tgt, TARGET_FIELDS):
        _latlon_ok(p, "target", tgt)
        if isinstance(seg, Mapping) and isinstance(seg.get("centre"), Mapping) and \
                (tgt.get("lat"), tgt.get("lon")) != (seg["centre"].get("lat"), seg["centre"].get("lon")):
            p.append("target is not the segment centre")
        _num_ok(p, "target.sigma_m", tgt.get("sigma_m"), lo=0.0, nullable=True)
        if covered:
            _num_ok(p, "target.bearing_deg", tgt.get("bearing_deg"), lo=0.0, below=360.0)
            _num_ok(p, "target.distance_m", tgt.get("distance_m"), lo=0.0)
        elif tgt.get("bearing_deg") is not None or tgt.get("distance_m") is not None:
            p.append("target.bearing_deg and distance_m must be null without a camera position")

    cap = rec.get("capture")
    if covered and _keys(p, "capture", cap, CAPTURE_FIELDS):
        for name in ("exposure_ns", "iso", "frame_duration_ns", "rs_skew_ns", "ois_mode"):
            if cap.get(name) is not None and not _is_int(cap[name]):
                p.append(f"capture.{name} must be an integer or null")
        _num_ok(p, "capture.focus_diopters", cap.get("focus_diopters"), nullable=True)
        if cap.get("focus_mode") is not None and not isinstance(cap["focus_mode"], str):
            p.append("capture.focus_mode must be a string or null")
        if not isinstance(cap.get("low_light"), bool):
            p.append("capture.low_light must be true or false")

    q = rec.get("quality")
    if _keys(p, "quality", q, QUALITY_FIELDS):
        if not isinstance(q.get("reason"), str) or not q.get("reason"):
            p.append("quality.reason must be a non-empty string")
        n = q.get("n_candidates")
        if not _is_int(n) or n < 0:
            p.append("quality.n_candidates must be an integer >= 0")
        blur = ("blur_motion_px", "blur_rotation_px", "blur_px", "gyro_rad_s")
        if covered:
            _num_ok(p, "quality.sharpness", q.get("sharpness"), lo=0.0)
            rank = q.get("sharpness_rank")
            if not (_is_int(rank) and _is_int(n) and 1 <= rank <= n):
                p.append("quality.sharpness_rank must be an integer in 1..n_candidates")
            for name in blur:
                _num_ok(p, f"quality.{name}", q.get(name), lo=0.0, nullable=True)
        else:
            if any(q.get(name) is not None for name in ("sharpness", "sharpness_rank", *blur)) or n != 0:
                p.append("quality of a not_covered segment must have no frame measures and n_candidates 0")

    g = rec.get("gate")
    if covered and _keys(p, "gate", g, GATE_FIELDS):
        mode, dec, dif = g.get("mode"), g.get("decision"), g.get("decision_if_gated")
        if mode not in GATE_MODES:
            p.append(f"gate.mode {mode!r} is not one of {GATE_MODES}")
        if dec not in GATE_DECISIONS or dif not in GATE_DECISIONS:
            p.append(f"gate.decision and decision_if_gated must be one of {GATE_DECISIONS}")
        if dec != status:
            p.append("status differs from gate.decision")
        if mode == "all" and dec != "send":
            p.append("gate mode `all` must send every covered segment")
        if mode == "gate" and dif != dec:
            p.append("gate mode `gate`: decision_if_gated must equal decision")
        if not (isinstance(g.get("reasons"), list) and all(isinstance(r, str) for r in g["reasons"])):
            p.append("gate.reasons must be a list of strings")
        if _keys(p, "gate.scores", g.get("scores"), GATE_SCORE_FIELDS):
            for name in GATE_SCORE_FIELDS:
                v = g["scores"].get(name)
                integer = name == "n_patches"
                if v is not None and not (_is_int(v) if integer else _is_num(v)):
                    p.append(f"gate.scores.{name} must be {'an integer' if integer else 'a number'} or null")
        th = g.get("thresholds")
        if not (isinstance(th, Mapping) and all(
                isinstance(k, str) and (isinstance(v, str) if k in THRESHOLD_TEXT_FIELDS else (v is None or _is_num(v)))
                for k, v in th.items())):
            p.append(f"gate.thresholds must map names to numbers or null ({', '.join(THRESHOLD_TEXT_FIELDS)}: text)")
        _num_ok(p, "gate.audit_draw", g.get("audit_draw"), lo=0.0, hi=1.0, nullable=True)

    _keys(p, "survey", rec.get("survey"), SURVEY_FIELDS)
    _keys(p, "device", rec.get("device"), DEVICE_FIELDS)
    geo = rec.get("geometry")
    if _keys(p, "geometry", geo, GEOMETRY_FIELDS):
        if not isinstance(geo.get("source"), str):
            p.append("geometry.source must be a string")
        for name in ("pitch_deg", "roll_deg", "height_m"):
            _num_ok(p, f"geometry.{name}", geo.get(name))

    flags = rec.get("flags")
    if not (isinstance(flags, list) and all(f in FLAGS for f in flags) and len(set(flags)) == len(flags)):
        p.append(f"flags must be a list of distinct names from {FLAGS}")
    else:
        if isinstance(seg, Mapping) and ("partial" in flags) != (seg.get("partial") is True):
            p.append("flag `partial` disagrees with segment.partial")
        if covered and isinstance(cap, Mapping) and ("low_light" in flags) != (cap.get("low_light") is True):
            p.append("flag `low_light` disagrees with capture.low_light")

    if "vlm" in rec and _keys(p, "vlm", rec["vlm"], VLM_FIELDS):
        v = rec["vlm"]
        if not (isinstance(v.get("labels"), list) and all(isinstance(x, str) for x in v["labels"])):
            p.append("vlm.labels must be a list of strings")
        if not isinstance(v.get("valid"), bool) or not isinstance(v.get("problem"), str):
            p.append("vlm.valid must be true or false and vlm.problem a string")
    return p


def _keys(p: list[str], name: str, obj, fields: Sequence[str], optional: Sequence[str] = ()) -> bool:
    if not isinstance(obj, Mapping):
        p.append(f"{name} must be an object")
        return False
    missing = [f for f in fields if f not in obj]
    extra = [k for k in obj if k not in fields and k not in optional]
    if missing:
        p.append(f"{name} is missing {missing}")
    if extra:
        p.append(f"{name} has unknown fields {extra}")
    return True


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num_ok(p: list[str], name: str, v, *, lo: float | None = None, hi: float | None = None,
            below: float | None = None, nullable: bool = False) -> None:
    """A finite number with lo <= v (<= hi) (< below); None also passes when nullable."""
    if v is None and nullable:
        return
    if _is_num(v) and (lo is None or v >= lo) and (hi is None or v <= hi) and (below is None or v < below):
        return
    bounds = [f"{op} {b:g}" for op, b in ((">=", lo), ("<=", hi), ("<", below)) if b is not None]
    p.append(f"{name} must be a number{' ' + ' and '.join(bounds) if bounds else ''}"
             f"{' or null' if nullable else ''}, got {v!r}")


def _latlon_ok(p: list[str], name: str, obj) -> None:
    if not isinstance(obj, Mapping):
        p.append(f"{name} must be an object with lat and lon")
        return
    if name.startswith("segment.") and set(obj) != {"lat", "lon"}:
        p.append(f"{name} must have exactly lat and lon")
    _num_ok(p, f"{name}.lat", obj.get("lat"), lo=-90.0, hi=90.0)
    _num_ok(p, f"{name}.lon", obj.get("lon"), lo=-180.0, hi=180.0)


def _ring_ok(p: list[str], ring) -> None:
    ok = isinstance(ring, list) and len(ring) >= 4 and all(
        isinstance(c, list) and len(c) == 2 and _is_num(c[0]) and _is_num(c[1])
        and -180.0 <= c[0] <= 180.0 and -90.0 <= c[1] <= 90.0 for c in ring)
    if not ok:
        p.append("segment.footprint_lonlat must be a list of at least 4 [lon, lat] points")
    elif ring[0] != ring[-1]:
        p.append("segment.footprint_lonlat is not closed (first point must be repeated last)")


def _comparable(rec: Mapping[str, Any]) -> dict:
    """A record without what legitimately differs between a sidecar and the PNG's embedded copy."""
    r = json.loads(to_json(rec, indent=None))
    r.pop("vlm", None)  # added to the sidecar by vlm-import
    if isinstance(r.get("frame"), dict):
        r["frame"]["sha256"] = None  # the PNG can't contain its own hash
    return r


def record_matches_embedded(sidecar: Mapping[str, Any], embedded: Mapping[str, Any] | None) -> list[str]:
    """Differences between a sidecar and the record embedded in its PNG (empty list = the same).

    Two differences are expected and ignored: frame.sha256, which the embedded copy can't contain, and
    the `vlm` block, which vlm-import adds to the sidecar.
    """
    if embedded is None:
        return ["the PNG has no embedded geoai record"]
    a, b = _comparable(sidecar), _comparable(embedded)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return [f"embedded record differs from the sidecar in {diff}"] if diff else []


def check_segment_png(path: Path, sidecar: Mapping[str, Any], *, offset_minutes: int) -> list[str]:
    """Problems of a written segment frame against its sidecar (docs/pipeline/13 §7, real sessions).

    The file's SHA-256 and size match frame.sha256, width and height; the embedded record matches the
    sidecar (`record_matches_embedded`) and is valid; the EXIF matches the record: GPS = camera position,
    GPSDest = segment centre with bearing and distance, DateTimeOriginal + SubSec + Offset = local time,
    GPS date and time = UTC, ImageUniqueID from the id.
    """
    from PIL import ExifTags, Image

    from .records import dms_to_deg, read_png_record, unique_id, utc_strings

    path = Path(path)
    p: list[str] = []
    frame = sidecar.get("frame") or {}
    if sha256_file(path) != frame.get("sha256"):
        p.append("file SHA-256 differs from frame.sha256")
    embedded, gps = read_png_record(path)
    with Image.open(path) as im:
        size = im.size
        exif = dict(im.getexif().get_ifd(ExifTags.IFD.Exif))
    if size != (frame.get("width"), frame.get("height")):
        p.append(f"image size {size} differs from frame width x height")
    p += record_matches_embedded(sidecar, embedded)
    if embedded is not None:
        p += [f"embedded record: {x}" for x in validate_segment_record(embedded, embedded=True)]
    G = ExifTags.GPS
    cam, tgt = sidecar.get("camera") or {}, sidecar.get("target") or {}

    def close(tag_ref, tag, want, tol, name):
        try:
            got = dms_to_deg(gps[tag], gps[tag_ref])
        except KeyError:
            p.append(f"EXIF {name} missing")
            return
        if abs(got - want) > tol:
            p.append(f"EXIF {name} {got} differs from the record's {want}")

    close(G.GPSLatitudeRef, G.GPSLatitude, cam.get("lat"), 1e-7, "GPSLatitude")
    close(G.GPSLongitudeRef, G.GPSLongitude, cam.get("lon"), 1e-7, "GPSLongitude")
    close(G.GPSDestLatitudeRef, G.GPSDestLatitude, tgt.get("lat"), 1e-7, "GPSDestLatitude")
    close(G.GPSDestLongitudeRef, G.GPSDestLongitude, tgt.get("lon"), 1e-7, "GPSDestLongitude")
    # EXIF stores the distance in km to 1e-5 (1 cm) and the bearing to 0.01 degrees (records.build_exif).
    dist, bearing = _num(tgt.get("distance_m")), _num(tgt.get("bearing_deg"))
    if G.GPSDestDistance not in gps or dist is None or abs(float(gps[G.GPSDestDistance]) - dist / 1000.0) > 0.6e-5:
        p.append("EXIF GPSDestDistance differs from target.distance_m")
    if G.GPSDestBearing not in gps or bearing is None or abs(float(gps[G.GPSDestBearing]) - bearing) > 0.006:
        p.append("EXIF GPSDestBearing differs from target.bearing_deg")
    if exif.get(ExifTags.Base.ImageUniqueID) != unique_id(sidecar["id"]):
        p.append("EXIF ImageUniqueID does not derive from the record id")
    if frame.get("utc_ms") is not None:
        ts = utc_strings(frame["utc_ms"], offset_minutes)
        want = (ts["exif_local"], ts["exif_subsec"], ts["exif_offset"], ts["gps_date"])
        got = (exif.get(ExifTags.Base.DateTimeOriginal), exif.get(ExifTags.Base.SubsecTimeOriginal),
               exif.get(ExifTags.Base.OffsetTimeOriginal), gps.get(G.GPSDateStamp))
        if got != want:
            p.append(f"EXIF date/time {got} differs from the record's {want}")
    return p


# ---- tables and maps -----------------------------------------------------------------------------

# (column, decimals): floats are written with this many decimals; None = as is (integers, text).
SEGMENT_COLUMNS: tuple[tuple[str, int | None], ...] = (
    ("id", None), ("session_id", None), ("k", None), ("status", None), ("s0_m", 3), ("s1_m", 3), ("length_m", 3),
    ("partial", None), ("chainage_start_km", 6), ("chainage_end_km", 6), ("start_lat", 9), ("start_lon", 9),
    ("centre_lat", 9), ("centre_lon", 9), ("end_lat", 9), ("end_lon", 9), ("frame_file", None),
    ("video_segment", None), ("frame_in_segment", None), ("frame_global", None), ("t_ns", None), ("utc_iso", None),
    ("local_iso", None), ("rows_top", 2), ("rows_bottom", 2), ("near_edge_ahead_m", 3), ("far_edge_ahead_m", 3),
    ("camera_lat", 9), ("camera_lon", 9), ("heading_deg", 2), ("speed_mps", 2), ("h_sigma_m", 2),
    ("target_bearing_deg", 2), ("target_distance_m", 3), ("target_sigma_m", 2), ("exposure_ns", None),
    ("iso", None), ("low_light", None), ("sharpness", 6), ("sharpness_rank", None), ("n_candidates", None),
    ("blur_motion_px", 3), ("blur_rotation_px", 3), ("blur_px", 3), ("quality_reason", None), ("gate_mode", None),
    ("gate_decision", None), ("gate_decision_if_gated", None), ("gate_reasons", None), ("probe_frac", 4),
    ("probe_max_logit", 3), ("evidence", 2), ("crack_mm", 1), ("z_max", 3), ("anomalous_patch_frac", 4),
    ("n_patches", None), ("audit_draw", 6), ("flags", None), ("vlm_labels", None), ("vlm_valid", None),
    ("vlm_severity", None), ("vlm_batch_id", None),
)
SEGMENT_CSV_FIELDS = tuple(name for name, _ in SEGMENT_COLUMNS)


def segment_row(rec: Mapping[str, Any]) -> dict:
    """One record as a flat row (typed values; lists stay lists) for CSV columns and GeoJSON properties."""
    seg = rec["segment"]
    f, c, t = rec.get("frame") or {}, rec.get("camera") or {}, rec.get("target") or {}
    cap, q, g = rec.get("capture") or {}, rec.get("quality") or {}, rec.get("gate") or {}
    sc, v = g.get("scores") or {}, rec.get("vlm") or {}
    rows = f.get("rows_of_segment") or [None, None]
    return {
        "id": rec["id"], "session_id": rec["session_id"], "k": seg["k"], "status": rec["status"],
        "s0_m": seg["s0_m"], "s1_m": seg["s1_m"], "length_m": seg["length_m"], "partial": seg["partial"],
        "chainage_start_km": seg["chainage_start_km"], "chainage_end_km": seg["chainage_end_km"],
        "start_lat": seg["start"]["lat"], "start_lon": seg["start"]["lon"],
        "centre_lat": seg["centre"]["lat"], "centre_lon": seg["centre"]["lon"],
        "end_lat": seg["end"]["lat"], "end_lon": seg["end"]["lon"],
        "frame_file": f.get("file"), "video_segment": f.get("segment"), "frame_in_segment": f.get("frame_in_segment"),
        "frame_global": f.get("frame_global"), "t_ns": f.get("t_ns"), "utc_iso": f.get("utc_iso"),
        "local_iso": f.get("local_iso"), "rows_top": rows[0], "rows_bottom": rows[1],
        "near_edge_ahead_m": f.get("near_edge_ahead_m"), "far_edge_ahead_m": f.get("far_edge_ahead_m"),
        "camera_lat": c.get("lat"), "camera_lon": c.get("lon"), "heading_deg": c.get("heading_deg"),
        "speed_mps": c.get("speed_mps"), "h_sigma_m": c.get("h_sigma_m"),
        "target_bearing_deg": t.get("bearing_deg"), "target_distance_m": t.get("distance_m"),
        "target_sigma_m": t.get("sigma_m"), "exposure_ns": cap.get("exposure_ns"), "iso": cap.get("iso"),
        "low_light": cap.get("low_light"), "sharpness": q.get("sharpness"), "sharpness_rank": q.get("sharpness_rank"),
        "n_candidates": q.get("n_candidates"), "blur_motion_px": q.get("blur_motion_px"),
        "blur_rotation_px": q.get("blur_rotation_px"), "blur_px": q.get("blur_px"), "quality_reason": q.get("reason"),
        "gate_mode": g.get("mode"), "gate_decision": g.get("decision"), "gate_decision_if_gated": g.get("decision_if_gated"),
        "gate_reasons": g.get("reasons"), "probe_frac": sc.get("probe_frac"), "probe_max_logit": sc.get("probe_max_logit"),
        "evidence": sc.get("evidence"), "crack_mm": sc.get("crack_mm"), "z_max": sc.get("z_max"),
        "anomalous_patch_frac": sc.get("anomalous_patch_frac"), "n_patches": sc.get("n_patches"),
        "audit_draw": g.get("audit_draw"), "flags": list(rec.get("flags") or []), "vlm_labels": v.get("labels"),
        "vlm_valid": v.get("valid"), "vlm_severity": v.get("severity"), "vlm_batch_id": v.get("batch_id"),
    }


def _cell(v, decimals: int | None) -> str:
    """CSV text of a value: empty for unknown, 1/0 for booleans, '+'-joined lists, fixed decimals for floats."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return "+".join(str(x) for x in v)
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return ""
        return f"{float(v):.{decimals}f}" if decimals is not None else repr(float(v))
    return str(v)


def write_table_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[tuple[str, int | None]]) -> None:
    """A CSV table (UTF-8 with BOM so Excel reads it; '\\n' line ends on every OS) of the given columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([name for name, _ in columns])
        for r in rows:
            w.writerow([_cell(r.get(name), nd) for name, nd in columns])
    tmp.replace(path)


def segment_feature(rec: Mapping[str, Any], properties: Mapping[str, Any] | None = None) -> dict:
    """A GeoJSON feature: the segment's footprint polygon with `properties` (default: `segment_row`)."""
    ring = rec["segment"].get("footprint_lonlat")
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]} if ring else None,
            "properties": dict(segment_row(rec) if properties is None else properties)}


def write_feature_collection(path: Path, features: Sequence[Mapping[str, Any]]) -> None:
    """RFC 7946 FeatureCollection (WGS-84, [longitude, latitude]), written atomically."""
    write_json_atomic(Path(path), {"type": "FeatureCollection", "features": list(features)})


def write_segment_outputs(run_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """sidecars/<id>.json, segments.csv and segments.geojson of a run, for records of every status.

    Every record is validated first (as a sidecar: sent segments must carry frame.sha256, so call this
    after the frame PNGs are written and hashed); nothing is written if any record fails. Output is in
    route order (session, k), so reruns give identical files.
    """
    run_dir = Path(run_dir)
    recs = sorted(records, key=lambda r: (r["session_id"], r["segment"]["k"]))
    bad = {r.get("id"): pr for r in recs if (pr := validate_segment_record(r))}
    if bad:
        first = next(iter(bad.items()))
        raise ValueError(f"{len(bad)} invalid segment records, e.g. {first[0]}: {'; '.join(first[1])}")
    ids = [r["id"] for r in recs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate segment record ids")
    for r in recs:
        write_json_atomic(run_dir / "sidecars" / f"{r['id']}.json", r)
    write_table_csv(run_dir / "segments.csv", (segment_row(r) for r in recs), SEGMENT_COLUMNS)
    write_feature_collection(run_dir / "segments.geojson", [segment_feature(r) for r in recs])


def load_segment_records(folder: Path) -> list[dict]:
    """Every `*.json` record in a folder (a run's sidecars/ or a batch's records/), in route order."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.json"))]
    return sorted(recs, key=lambda r: (r["session_id"], r["segment"]["k"]))


def write_contact_sheet(run_dir: Path, records: Sequence[Mapping[str, Any]], *, max_items: int = 48, cols: int = 8,
                        cell_w: int = 320) -> Path | None:
    """contact_sheet.jpg: thumbnails of the first `max_items` sent frames (send and audit), in route order.

    Each thumbnail marks the segment's rows (green for send, orange for audit), so the strip of road the
    VLM is asked about is visible at a glance; the caption gives k, status, path distance and the gate's
    strongest signals. Frames are read from run_dir/frames, so call it after the PNG writer has
    finished. Returns the sheet's path, or None when nothing was sent.
    """
    import cv2

    run_dir = Path(run_dir)
    sent = sorted((r for r in records if r["status"] in SENT_STATUSES),
                  key=lambda r: (r["session_id"], r["segment"]["k"]))[:max_items]
    if not sent:
        return None
    f0 = sent[0]["frame"]
    cell_h = int(round(cell_w * f0["height"] / f0["width"]))
    caption = 36
    n_rows = (len(sent) + cols - 1) // cols
    sheet = np.full((n_rows * (cell_h + caption), cols * cell_w, 3), 255, np.uint8)
    for i, rec in enumerate(sent):
        fr = rec["frame"]
        img = cv2.imread(str(run_dir / fr["file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {run_dir / fr['file']}")
        sc = min(cell_w / img.shape[1], cell_h / img.shape[0])
        th = cv2.resize(img, (max(1, int(round(img.shape[1] * sc))), max(1, int(round(img.shape[0] * sc)))),
                        interpolation=cv2.INTER_AREA)
        colour = (0, 160, 0) if rec["status"] == "send" else (0, 140, 255)  # BGR
        for v in fr["rows_of_segment"]:
            y = int(round(min(max(v, 0.0), img.shape[0] - 1) * sc))
            cv2.line(th, (0, y), (th.shape[1] - 1, y), colour, 2, cv2.LINE_AA)
        r, c = divmod(i, cols)
        y0, x0 = r * (cell_h + caption), c * cell_w
        sheet[y0:y0 + th.shape[0], x0:x0 + th.shape[1]] = th
        seg, scores = rec["segment"], rec["gate"]["scores"]
        line1 = f"k{seg['k']} {rec['status']} {seg['s0_m']:.0f}-{seg['s1_m']:.0f} m"
        line2 = (f"probe {_short(scores.get('probe_frac'), 2)}  ev {_short(scores.get('evidence'), 0)}"
                 f"  crack {_short(scores.get('crack_mm'), 0)}")
        cv2.putText(sheet, line1, (x0 + 3, y0 + cell_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(sheet, line2, (x0 + 3, y0 + cell_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    out = run_dir / "contact_sheet.jpg"
    if not cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"cannot write {out}")
    return out


def _short(v, nd: int) -> str:
    return "-" if v is None else f"{float(v):.{nd}f}"
