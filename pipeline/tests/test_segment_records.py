"""Segment records (schema geoai.segment.v1): building, positions, validation, EXIF, CSV/GeoJSON, contact sheet.

Expected values are computed here independently: positions from the synthetic route's own truth
(pymap3d on the straight northbound line), times with datetime, ids with hashlib, key lists typed
from the spec (docs/pipeline/13 §5), not taken from the module's constants.
"""
import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pymap3d
import pytest
from PIL import ExifTags, Image

from geoai.config import Config, GnssConfig
from geoai.records import build_exif, dms_to_deg, read_png_record, write_png
from geoai.segment_records import (SEGMENT_CSV_FIELDS, CameraPose, FrameShot, MslOffset, PathPoint,
                                   build_segment_record, camera_pose, check_segment_png, config_digest,
                                   footprint_lonlat, geometry_block, load_segment_records, path_points,
                                   record_matches_embedded, validate_segment_record, write_contact_sheet,
                                   write_segment_outputs)
from geoai.trajectory import build_trajectory, clean_fixes
from geoai.util import canonical_json

from synth_gnss import ORIGIN, T0_NS, fixes_frame

UTC0_MS = 1_790_000_000_000  # synth_gnss default: utc = UTC0_MS + 1000 * tau
SPEED = 10.0
META = {
    "session_id": "S1",
    "survey": {"road_id": "TST-1", "road_name": "test 1 ", "direction": "increasing_chainage", "lane": "L1",
               "start_chainage_km_text": "12", "start_chainage_km": 12.0, "operator": "op", "vehicle": "car",
               "weather": "dry_overcast", "notes": "not in the record"},
    "device": {"device_id": "SM-G991B-F0D1", "manufacturer": "samsung", "model": "SM-G991B", "soc_model": "x"},
    "mount": {"camera_facing": "forward", "nominal_height_m": 1.35},
    "capture_settings": {"focus_mode": "LOCKED_MANUAL"},
}
GEOMETRY = {"source": "nominal", "pitch_deg": 23.0, "roll_deg": -4.4, "height_m": 1.35}
# A ground trapezoid in the camera frame (x right, y ahead), counter-clockwise, closed.
RING_XY = [[-1.5, 1.6], [1.5, 1.6], [4.0, 6.6], [-4.0, 6.6], [-1.5, 1.6]]

# Field lists as written in docs/pipeline/13 §5.
SPEC_TOP = ["id", "kind", "schema", "status", "session_id", "pipeline_version", "config_sha256", "segment", "frame",
            "camera", "target", "capture", "quality", "gate", "survey", "device", "geometry", "flags"]
SPEC_SEGMENT = ["k", "length_m", "s0_m", "s1_m", "partial", "chainage_start_km", "chainage_end_km", "start", "end",
                "centre", "footprint_lonlat"]
SPEC_FRAME = ["file", "sha256", "width", "height", "segment", "frame_in_segment", "frame_global", "t_ns", "utc_ms",
              "utc_iso", "local_iso", "rows_of_segment", "near_edge_ahead_m", "far_edge_ahead_m"]
SPEC_CAMERA = ["lat", "lon", "alt_ellipsoid_m", "alt_msl_m", "heading_deg", "view_direction_deg", "speed_mps",
               "h_sigma_m", "utc_ms"]
SPEC_TARGET = ["lat", "lon", "bearing_deg", "distance_m", "sigma_m"]
SPEC_CAPTURE = ["exposure_ns", "iso", "frame_duration_ns", "rs_skew_ns", "focus_diopters", "focus_mode", "ois_mode",
                "low_light"]
SPEC_QUALITY = ["sharpness", "sharpness_rank", "n_candidates", "blur_motion_px", "blur_rotation_px", "blur_px",
                "gyro_rad_s", "reason"]
SPEC_GATE = ["mode", "decision", "decision_if_gated", "reasons", "scores", "thresholds", "audit_draw"]
SPEC_SCORES = ["probe_frac", "probe_max_logit", "evidence", "crack_mm", "z_max", "anomalous_patch_frac", "n_patches"]
SPEC_SURVEY = ["road_id", "road_name", "direction", "lane", "start_chainage_km", "operator", "vehicle", "weather"]


@dataclass(frozen=True)
class Seg:
    k: int
    s0: float
    s1: float
    partial: bool = False


class FlatTraj:
    """Just what footprint_lonlat needs: ENU of the synthetic origin -> WGS-84."""
    origin = ORIGIN

    def enu_to_geodetic(self, e, n, u=None):
        e = np.asarray(e, float)
        return pymap3d.enu2geodetic(e, np.asarray(n, float), np.zeros_like(e), *ORIGIN)


def _ll(e, n):
    lat, lon, _ = pymap3d.enu2geodetic(e, n, 0.0, *ORIGIN)
    return float(lat), float(lon)


def frame_row(k: int) -> dict:
    return {"segment": 1, "frame_in_segment": 30 * k + 7, "frame_global": 30 * k + 7, "t_ns": T0_NS + k * 10**9,
            "exposure_ns": 1_000_000, "iso": 1279, "frame_duration_ns": 33_333_000, "rs_skew_ns": 16_566_040,
            "focus_diopters": 0.46077293, "ois_mode": 1}


def gate_dict(mode="gate", decision="send", dif=None, probe=0.12, evidence=40.0, crack=0.0) -> dict:
    return {"mode": mode, "decision": decision, "decision_if_gated": decision if dif is None else dif,
            "reasons": ["probe_frac"] if decision == "send" else [],
            "scores": {"probe_frac": probe, "probe_max_logit": 2.5, "evidence": evidence, "crack_mm": crack,
                       "z_max": 3.1, "anomalous_patch_frac": 0.04, "n_patches": 812},
            "thresholds": {"probe_frac": 0.05, "evidence": 30.0, "crack_mm": 300.0}, "audit_draw": 0.734}


def quality_dict(covered=True, reason="sharpest") -> dict:
    if not covered:
        return {"sharpness": None, "sharpness_rank": None, "n_candidates": 0, "blur_motion_px": None,
                "blur_rotation_px": None, "blur_px": None, "gyro_rad_s": None, "reason": reason}
    return {"sharpness": 0.0123, "sharpness_rank": 1, "n_candidates": 3, "blur_motion_px": 1.2,
            "blur_rotation_px": 0.4, "blur_px": float(np.hypot(1.2, 0.4)), "gyro_rad_s": 0.01, "reason": reason}


def record_args(session_id="S1", k=0, *, status="send", dif=None, mode="gate", length=5.0, partial=False,
                base_s=100.0, size=(112, 63), meta=None, gate_scores=None, low_light=False) -> dict:
    """build_segment_record arguments for a segment of a straight northbound road (e = 0, n = s).

    status: send | audit | skip | not_covered. mode `all`: status must be send, dif is the would-be decision.
    Segment k spans [base_s + 5k, base_s + 5k + length); the camera is 1.6 m before its near edge, at
    10 m/s, with UTC = UTC0_MS + 100 ms per metre of path.
    """
    s0 = base_s + 5.0 * k
    s1 = s0 + length
    pts = []
    for s in (s0, (s0 + s1) / 2, s1):
        lat, lon = _ll(0.0, s)
        pts.append(PathPoint(s, 0.0, s, lat, lon, 2.5, True))
    c = s0 - 1.6
    clat, clon = _ll(0.0, c)
    pose = CameraPose(c / SPEED, c, 0.0, c, clat, clon, 900.0, 0.0, SPEED, 2.0, UTC0_MS + 100.0 * c, True, False)
    covered = status != "not_covered"
    shot = gate = camera = None
    if covered:
        shot = FrameShot(frame_row(k), size[0], size[1], (10.0, 55.0), 1.6, 1.6 + length, low_light)
        camera = pose
        gate = gate_dict(mode, status, dif, **(gate_scores or {}))
    return dict(session_id=session_id, session_meta=dict(META if meta is None else meta),
                segment=Seg(k, s0, s1, partial), start=pts[0], centre=pts[1], end=pts[2],
                footprint=footprint_lonlat(FlatTraj(), RING_XY, pose), shot=shot, camera=camera,
                quality=quality_dict(covered, "sharpest" if covered else "gnss_gap"), gate=gate, geometry=GEOMETRY,
                latency_method="none", chainage_origin_m=50.0, cfg=Config(), config_sha256="ab" * 32)


def make_record(session_id="S1", k=0, **kw) -> dict:
    """A record built through build_segment_record from `record_args`."""
    return build_segment_record(**record_args(session_id, k, **kw))


def write_frame(run: Path, rec: dict, rgb: np.ndarray | None = None) -> None:
    """Write a sent record's frame PNG (with EXIF and the embedded record) and set frame.sha256."""
    w, h = rec["frame"]["width"], rec["frame"]["height"]
    if rgb is None:
        rng = np.random.default_rng(rec["segment"]["k"])
        rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    rec["frame"]["sha256"] = write_png(run / rec["frame"]["file"], rgb, rec, make="samsung", model="SM-G991B",
                                       compress_level=1)


# ---- positions on a real trajectory ----------------------------------------------------------------

@pytest.fixture(scope="module")
def straight_traj():
    tau = np.arange(0.0, 61.0, 1.0)
    df = fixes_frame(tau, np.zeros_like(tau), SPEED * tau, np.full_like(tau, SPEED), np.zeros_like(tau))
    cfg = GnssConfig()
    return build_trajectory(clean_fixes(df, T0_NS, cfg), None, T0_NS, cfg)


def test_path_points_camera_pose_and_footprint_follow_the_route(straight_traj):
    traj = straight_traj
    s_start = float(traj.sample(np.array([0.0]))["s"][0])
    # Truth: at path distance s the vehicle is (s - s_start) m north of the synthetic origin.
    qs = [s_start + 37.0, s_start + 251.5, float(traj.s[-1]) + 3.0]
    pts = path_points(traj, qs)
    for p, s in zip(pts[:2], qs[:2]):
        lat, lon = _ll(0.0, s - s_start)
        assert abs(p.lat - lat) < 1e-7 and abs(p.lon - lon) < 1e-7  # ~1 cm
        assert p.valid and p.s_m == s
    end = traj.sample(np.array([traj.t[-1]]))
    lat_end, lon_end, _ = traj.enu_to_geodetic(end["e"], end["n"] + 3.0)  # 3 m beyond the end, heading north
    assert not pts[2].valid and abs(pts[2].lat - float(lat_end[0])) < 1e-8

    pose = camera_pose(traj, 20.0)
    lat, lon = _ll(0.0, SPEED * 20.0)
    assert abs(pose.lat - lat) < 1e-7 and abs(pose.lon - lon) < 1e-7
    assert min(pose.heading_deg, 360 - pose.heading_deg) < 0.5 and abs(pose.speed_mps - SPEED) < 0.05
    assert pose.utc_ms == pytest.approx(UTC0_MS + 20_000.0, abs=1.0)
    ring = footprint_lonlat(traj, RING_XY, pose)
    assert ring[0] == ring[-1] and len(ring) == len(RING_XY)
    for (lon_r, lat_r), (x, y) in zip(ring, RING_XY):
        lat_t, lon_t = _ll(x, SPEED * 20.0 + y)
        assert abs(lat_r - lat_t) < 3e-7 and abs(lon_r - lon_t) < 3e-7  # heading error < 0.5 deg over 7 m
    # Counter-clockwise in (lon, lat), as RFC 7946 wants for an exterior ring (shoelace area > 0).
    a = np.array(ring)
    assert np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]) > 0
    with pytest.raises(ValueError):
        footprint_lonlat(traj, RING_XY[:-1], pose)


# ---- building ------------------------------------------------------------------------------------

def test_send_record_has_exactly_the_spec_fields_and_values():
    rec = make_record(k=7)
    assert validate_segment_record(rec, embedded=True) == []
    assert validate_segment_record(rec) == ["frame.sha256 is missing for a sent segment"]
    assert list(rec) == SPEC_TOP
    assert list(rec["segment"]) == SPEC_SEGMENT and list(rec["frame"]) == SPEC_FRAME
    assert list(rec["camera"]) == SPEC_CAMERA and list(rec["target"]) == SPEC_TARGET
    assert list(rec["capture"]) == SPEC_CAPTURE and list(rec["quality"]) == SPEC_QUALITY
    assert list(rec["gate"]) == SPEC_GATE and list(rec["gate"]["scores"]) == SPEC_SCORES
    assert list(rec["survey"]) == SPEC_SURVEY
    assert list(rec["device"]) == ["manufacturer", "model", "device_id"]
    assert list(rec["geometry"]) == ["source", "pitch_deg", "roll_deg", "height_m"]

    assert rec["id"] == "S1_seg00007" and rec["kind"] == "segment" and rec["schema"] == "geoai.segment.v1"
    assert rec["status"] == "send" and rec["session_id"] == "S1"
    seg = rec["segment"]
    s0 = 100.0 + 35.0
    assert (seg["k"], seg["s0_m"], seg["s1_m"], seg["length_m"], seg["partial"]) == (7, s0, s0 + 5.0, 5.0, False)
    # Chainage: start 12 km at path distance 50 m, increasing.
    assert seg["chainage_start_km"] == pytest.approx(12.0 + (s0 - 50.0) / 1000.0, abs=1e-12)
    assert seg["chainage_end_km"] == pytest.approx(12.0 + (s0 + 5.0 - 50.0) / 1000.0, abs=1e-12)
    for name, s in (("start", s0), ("centre", s0 + 2.5), ("end", s0 + 5.0)):
        assert (seg[name]["lat"], seg[name]["lon"]) == pytest.approx(_ll(0.0, s), abs=1e-12)
    ring = seg["footprint_lonlat"]
    assert ring[0] == ring[-1] and len(ring) == 5
    for (lon, lat), (x, y) in zip(ring, RING_XY):  # heading 0: x east, y north of the camera
        assert (lat, lon) == pytest.approx(_ll(x, s0 - 1.6 + y), abs=1e-10)

    fr = rec["frame"]
    utc = UTC0_MS + 100.0 * (s0 - 1.6)
    t = datetime.fromtimestamp(utc / 1000, tz=timezone.utc)
    assert fr["file"] == "frames/S1_seg00007.png" and fr["sha256"] is None
    assert (fr["width"], fr["height"], fr["segment"], fr["frame_in_segment"], fr["frame_global"]) == (112, 63, 1, 217, 217)
    assert fr["t_ns"] == T0_NS + 7 * 10**9 and fr["utc_ms"] == utc
    assert fr["utc_iso"] == t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"
    loc = t + timedelta(hours=5, minutes=30)
    assert fr["local_iso"] == loc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{loc.microsecond // 1000:03d}+05:30"
    assert fr["rows_of_segment"] == [10.0, 55.0] and (fr["near_edge_ahead_m"], fr["far_edge_ahead_m"]) == (1.6, 6.6)

    cam = rec["camera"]
    assert (cam["lat"], cam["lon"]) == pytest.approx(_ll(0.0, s0 - 1.6), abs=1e-12)
    assert cam["alt_msl_m"] is None and cam["alt_ellipsoid_m"] == 900.0 and cam["utc_ms"] == utc
    assert (cam["heading_deg"], cam["view_direction_deg"], cam["speed_mps"], cam["h_sigma_m"]) == (0.0, 0.0, SPEED, 2.0)
    tgt = rec["target"]
    assert (tgt["lat"], tgt["lon"]) == (seg["centre"]["lat"], seg["centre"]["lon"])
    assert tgt["distance_m"] == pytest.approx(1.6 + 2.5, abs=1e-9) and tgt["bearing_deg"] == 0.0
    assert tgt["sigma_m"] == 2.5
    assert rec["capture"] == {"exposure_ns": 1_000_000, "iso": 1279, "frame_duration_ns": 33_333_000,
                              "rs_skew_ns": 16_566_040, "focus_diopters": 0.46077293, "focus_mode": "LOCKED_MANUAL",
                              "ois_mode": 1, "low_light": False}
    assert rec["quality"] == quality_dict() and rec["gate"] == gate_dict()
    assert rec["survey"] == {k: META["survey"][k] for k in SPEC_SURVEY}
    assert rec["device"] == {"manufacturer": "samsung", "model": "SM-G991B", "device_id": "SM-G991B-F0D1"}
    assert rec["geometry"] == GEOMETRY
    assert rec["flags"] == ["geometry_nominal", "latency_uncorrected"]


def test_rearward_view_direction_and_bearing_behind():
    meta = copy.deepcopy(META)
    meta["mount"]["camera_facing"] = "rearward"
    rec = make_record(k=1, meta=meta)
    assert rec["camera"]["view_direction_deg"] == 180.0 and rec["camera"]["heading_deg"] == 0.0


def test_not_covered_skip_and_send_all_records():
    nc = make_record(k=3, status="not_covered", partial=True, length=3.0)
    assert validate_segment_record(nc) == []
    assert nc["status"] == "not_covered"
    assert nc["frame"] is None and nc["camera"] is None and nc["capture"] is None and nc["gate"] is None
    assert nc["target"]["bearing_deg"] is None and nc["target"]["distance_m"] is None
    assert nc["quality"]["n_candidates"] == 0 and nc["quality"]["reason"] == "gnss_gap"
    assert nc["segment"]["length_m"] == 3.0 and nc["flags"] == ["partial", "geometry_nominal", "latency_uncorrected"]

    skip = make_record(k=4, status="skip")
    assert validate_segment_record(skip) == []  # no image: no sha256 needed
    assert skip["frame"]["file"] is None and skip["frame"]["sha256"] is None

    allmode = make_record(k=5, status="send", mode="all", dif="skip")
    assert validate_segment_record(allmode, embedded=True) == []
    assert allmode["gate"]["decision_if_gated"] == "skip"
    with pytest.raises(ValueError, match="mode `all` must send"):
        make_record(k=6, status="audit", mode="all")

    low = make_record(k=8, low_light=True)
    assert low["capture"]["low_light"] is True and "low_light" in low["flags"]


def test_no_chainage_without_start_or_direction():
    meta = copy.deepcopy(META)
    meta["survey"]["start_chainage_km"] = None
    assert make_record(meta=meta)["segment"]["chainage_start_km"] is None
    meta = copy.deepcopy(META)
    meta["survey"]["direction"] = "decreasing_chainage"
    seg = make_record(k=2, meta=meta)["segment"]
    assert seg["chainage_start_km"] == pytest.approx(12.0 - (110.0 - 50.0) / 1000.0, abs=1e-12)
    assert seg["chainage_end_km"] == pytest.approx(seg["chainage_start_km"] - 0.005, abs=1e-12)


def test_builder_rejects_inconsistent_inputs():
    args = record_args(k=0)
    with pytest.raises(ValueError, match="needs shot, camera and gate"):
        build_segment_record(**{**args, "gate": None})
    with pytest.raises(ValueError, match="quality must have the fields"):
        build_segment_record(**{**args, "quality": {**args["quality"], "extra": 1}})
    with pytest.raises(ValueError, match="outside the schema"):
        build_segment_record(**{**args, "gate": {**args["gate"], "debug": 1}})
    with pytest.raises(ValueError, match="gate.scores is missing"):
        build_segment_record(**{**args, "gate": {**args["gate"], "scores": {"probe_frac": 0.1}}})
    with pytest.raises(ValueError, match="footprint_lonlat is not closed"):
        build_segment_record(**{**args, "footprint": args["footprint"][:-1]})


@pytest.mark.parametrize("mutate, expect", [
    (lambda r: r.pop("survey"), "record is missing ['survey']"),
    (lambda r: r.__setitem__("extra", 1), "record has unknown fields ['extra']"),
    (lambda r: r.__setitem__("status", "audit"), "status differs from gate.decision"),
    (lambda r: r["segment"].__setitem__("length_m", 5.1), "segment.length_m is not s1_m - s0_m"),
    (lambda r: r["segment"]["footprint_lonlat"].pop(), "not closed"),
    (lambda r: r["frame"].__setitem__("sha256", None), "frame.sha256 is missing"),
    (lambda r: r["frame"].__setitem__("file", "x.png"), "frame.file must be frames/<id>.png"),
    (lambda r: r["flags"].insert(0, "partial"), "flag `partial` disagrees"),
    (lambda r: r["target"].__setitem__("lat", r["target"]["lat"] + 1e-6), "target is not the segment centre"),
    (lambda r: r["camera"].__setitem__("utc_ms", r["camera"]["utc_ms"] + 1), "camera.utc_ms differs"),
    (lambda r: r["gate"]["scores"].__setitem__("probe_frac", float("nan")), "not plain JSON"),
    (lambda r: r["frame"].__setitem__("far_edge_ahead_m", 7.6), "is not the segment length"),
    (lambda r: r.__setitem__("id", "S1_seg00008"), "does not match session_id and segment.k"),
    (lambda r: r["gate"].__setitem__("decision_if_gated", "skip"), "decision_if_gated must equal decision"),
    (lambda r: r.__setitem__("vlm", {"labels": ["Potholes"]}), "vlm is missing"),
    (lambda r: r["segment"].__setitem__("chainage_end_km", None), "at both ends or at neither"),
    (lambda r: r["quality"].__setitem__("sharpness_rank", 5), "sharpness_rank"),
    (lambda r: r["camera"].__setitem__("heading_deg", 360.0), "camera.heading_deg must be a number >= 0 and < 360"),
    (lambda r: r["gate"]["scores"].__setitem__("n_patches", 1.5), "gate.scores.n_patches must be an integer"),
    (lambda r: r.__setitem__("config_sha256", "x"), "config_sha256"),
])
def test_validator_finds_each_problem(mutate, expect):
    rec = make_record(k=0)
    rec["frame"]["sha256"] = "c" * 64
    assert validate_segment_record(rec) == []
    mutate(rec)
    problems = validate_segment_record(rec)
    assert any(expect in p for p in problems), problems


def test_validator_not_covered_must_not_carry_a_frame():
    rec = make_record(k=1, status="not_covered")
    rec["camera"] = make_record(k=1)["camera"]
    assert "camera must be null for a not_covered segment" in validate_segment_record(rec)


def test_config_digest_ignores_install_folder_only():
    a, b = Config(), Config()
    b.anomaly.weights = r"D:\elsewhere\models\dinov2\dinov2_vits14_reg4_pretrain.pth"
    b.anomaly.probe = "/mnt/x/models/probe/attain_dinov2_probe_v1.npz"
    assert config_digest(a) == config_digest(b)
    d = a.to_dict()
    d["anomaly"]["weights"] = "dinov2_vits14_reg4_pretrain.pth"
    d["anomaly"]["probe"] = "attain_dinov2_probe_v1.npz"
    assert config_digest(a) == hashlib.sha256(canonical_json(d)).hexdigest()
    c = Config()
    c.gate.probe_frac = 0.07
    assert config_digest(c) != config_digest(a)


# ---- EXIF ----------------------------------------------------------------------------------------

def test_segment_png_exif_round_trip(tmp_path):
    rec = make_record(k=7)
    write_frame(tmp_path, rec)
    path = tmp_path / rec["frame"]["file"]
    assert validate_segment_record(rec) == []
    embedded, gps = read_png_record(path)
    assert embedded["frame"]["sha256"] is None and record_matches_embedded(rec, embedded) == []
    G = ExifTags.GPS
    cam, tgt = rec["camera"], rec["target"]
    assert abs(dms_to_deg(gps[G.GPSLatitude], gps[G.GPSLatitudeRef]) - cam["lat"]) < 1e-9
    assert abs(dms_to_deg(gps[G.GPSLongitude], gps[G.GPSLongitudeRef]) - cam["lon"]) < 1e-9
    assert abs(dms_to_deg(gps[G.GPSDestLatitude], gps[G.GPSDestLatitudeRef]) - tgt["lat"]) < 1e-9
    assert abs(dms_to_deg(gps[G.GPSDestLongitude], gps[G.GPSDestLongitudeRef]) - tgt["lon"]) < 1e-9
    assert abs(float(gps[G.GPSDestDistance]) - 4.1 / 1000) < 1e-5 and gps[G.GPSDestDistanceRef] == "K"
    assert float(gps[G.GPSDestBearing]) == 0.0 and gps[G.GPSDestBearingRef] == "T"
    assert abs(float(gps[G.GPSSpeed]) - SPEED * 3.6) < 0.01 and float(gps[31]) == 2.0  # GPSHPositioningError
    assert G.GPSAltitude not in gps  # no MSL altitude: the ellipsoidal height is never written there
    t = datetime.fromtimestamp(rec["frame"]["utc_ms"] / 1000, tz=timezone.utc)
    loc = t + timedelta(minutes=330)
    assert gps[G.GPSDateStamp] == t.strftime("%Y:%m:%d")
    hms = [float(x) for x in gps[G.GPSTimeStamp]]
    assert hms[:2] == [t.hour, t.minute] and abs(hms[2] - (t.second + t.microsecond / 1e6)) < 1e-3
    with Image.open(path) as im:
        ex = im.getexif()
        exif = ex.get_ifd(ExifTags.IFD.Exif)
        assert (ex[ExifTags.Base.Make], ex[ExifTags.Base.Model]) == ("samsung", "SM-G991B")
    assert exif[ExifTags.Base.DateTimeOriginal] == loc.strftime("%Y:%m:%d %H:%M:%S")
    assert exif[ExifTags.Base.SubsecTimeOriginal] == f"{loc.microsecond // 1000:03d}"
    assert exif[ExifTags.Base.OffsetTimeOriginal] == "+05:30"
    assert exif[ExifTags.Base.ImageUniqueID] == hashlib.sha256(rec["id"].encode()).hexdigest()[:32]
    assert check_segment_png(path, rec, offset_minutes=330) == []

    moved = copy.deepcopy(rec)
    moved["target"]["lat"] += 1e-5
    moved["segment"]["centre"]["lat"] += 1e-5
    problems = check_segment_png(path, moved, offset_minutes=330)
    assert any("GPSDestLatitude" in p for p in problems) and any("embedded record differs" in p for p in problems)
    answered = {**rec, "vlm": {"labels": ["Normal"]}}  # vlm-import adds this to the sidecar only
    assert record_matches_embedded(answered, embedded) == []


def test_msl_offset_from_the_phones_fixes():
    import pandas as pd

    # Hand values: offset = alt_msl - alt_ellipsoid = 86.0 at t = 1 s and 86.4 at t = 11 s after T0.
    gnss = pd.DataFrame({"t_ns": [T0_NS + 1_000_000_000, T0_NS + 11_000_000_000, T0_NS + 12_000_000_000],
                         "alt_ellipsoid_m": [785.0, 790.0, ""], "alt_msl_m": [871.0, 876.4, 880.0]})
    m = MslOffset.from_fixes(gnss, T0_NS)
    assert m(6.0) == pytest.approx(86.2, abs=1e-9)          # halfway between the two usable fixes
    assert m(0.0) == pytest.approx(86.0) and m(20.0) == pytest.approx(86.4)   # held outside, not extrapolated
    assert MslOffset.from_fixes(gnss.drop(columns=["alt_msl_m"]), T0_NS) is None   # no MSL column
    assert MslOffset.from_fixes(gnss.assign(alt_msl_m=""), T0_NS) is None           # no MSL value
    assert MslOffset.from_fixes(None, T0_NS) is None


def test_msl_altitude_reaches_the_record_and_exif(tmp_path):
    from dataclasses import replace

    args = record_args(k=7)
    args["camera"] = replace(args["camera"], alt_msl_m=812.345)
    rec = build_segment_record(**args)
    assert rec["camera"]["alt_msl_m"] == 812.345 and rec["camera"]["alt_ellipsoid_m"] == 900.0
    write_frame(tmp_path, rec)
    _, gps = read_png_record(tmp_path / rec["frame"]["file"])
    G = ExifTags.GPS
    assert gps[G.GPSAltitudeRef] == b"\x00" and float(gps[G.GPSAltitude]) == pytest.approx(812.345, abs=0.005)
    assert check_segment_png(tmp_path / rec["frame"]["file"], rec, offset_minutes=330) == []
    wrong = copy.deepcopy(rec)
    wrong["camera"]["alt_msl_m"] = 812.5   # sidecar no longer matches the PNG's EXIF (nor its embedded record)
    assert any("GPSAltitude" in p for p in check_segment_png(tmp_path / rec["frame"]["file"], wrong, offset_minutes=330))


def test_crop_exif_still_reads_the_anomaly():
    crop = {"id": "SESSION_0000123_00", "utc_ms": 1_792_650_123_456.0,
            "camera": {"lat": 12.9715987, "lon": 77.5945627, "heading_deg": 271.5},
            "anomaly": {"lat": 12.9716102, "lon": 77.5945401, "bearing_deg": 300.0, "distance_m": 2.7}}
    ex = build_exif(crop, "samsung", "SM-G991B", "GeoAI pipeline", 330)
    gps = ex[ExifTags.IFD.GPSInfo]  # set as a dict by build_exif (get_ifd reads only parsed files)
    assert abs(dms_to_deg(gps[ExifTags.GPS.GPSDestLatitude], "N") - 12.9716102) < 1e-9
    # A crop's own utc_ms decides its time, whatever else the record holds.
    other = build_exif({**crop, "frame": {"utc_ms": 0.0}}, "samsung", "SM-G991B", "GeoAI pipeline", 330)
    assert other.tobytes() == ex.tobytes()


# ---- tables, maps, contact sheet -------------------------------------------------------------------

def _four_records(run: Path) -> list[dict]:
    recs = [make_record(k=0, status="send"), make_record(k=1, status="audit"), make_record(k=2, status="skip"),
            make_record(k=3, status="not_covered", partial=True, length=2.5)]
    for r in recs[:2]:
        write_frame(run, r)
    return recs


def test_write_segment_outputs_csv_geojson_sidecars(tmp_path):
    run = tmp_path / "S1"
    recs = _four_records(run)
    write_segment_outputs(run, list(reversed(recs)))  # written in route order whatever the input order
    assert [json.loads((run / "sidecars" / f"{r['id']}.json").read_text(encoding="utf-8")) for r in recs] == recs
    assert load_segment_records(run / "sidecars") == recs
    raw = (run / "segments.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw
    with open(run / "segments.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(SEGMENT_CSV_FIELDS)
    table = [dict(zip(rows[0], r)) for r in rows[1:]]
    assert [r["id"] for r in table] == [r["id"] for r in recs]
    assert [r["status"] for r in table] == ["send", "audit", "skip", "not_covered"]
    t0 = table[0]
    assert t0["s0_m"] == "100.000" and t0["length_m"] == "5.000" and t0["partial"] == "0"
    assert t0["frame_file"] == "frames/S1_seg00000.png" and t0["frame_global"] == "7"
    assert t0["centre_lat"] == f"{recs[0]['segment']['centre']['lat']:.9f}"
    assert t0["gate_reasons"] == "probe_frac" and t0["flags"] == "geometry_nominal+latency_uncorrected"
    assert t0["probe_frac"] == "0.1200" and t0["n_patches"] == "812" and t0["vlm_labels"] == ""
    t3 = table[3]
    assert t3["partial"] == "1" and t3["length_m"] == "2.500" and t3["frame_file"] == "" and t3["gate_mode"] == ""
    assert table[2]["frame_file"] == "" and table[2]["frame_global"] == "67"
    gj = json.loads((run / "segments.geojson").read_text(encoding="utf-8"))
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 4
    for f, r in zip(gj["features"], recs):
        assert f["geometry"] == {"type": "Polygon", "coordinates": [r["segment"]["footprint_lonlat"]]}
        assert f["properties"]["id"] == r["id"] and f["properties"]["status"] == r["status"]
    before = {p.name: p.read_bytes() for p in run.glob("segments.*")}
    write_segment_outputs(run, recs)
    assert {p.name: p.read_bytes() for p in run.glob("segments.*")} == before  # deterministic

    bad = copy.deepcopy(recs)
    bad[1]["frame"]["sha256"] = None
    with pytest.raises(ValueError, match="invalid segment records"):
        write_segment_outputs(run, bad)
    assert {p.name: p.read_bytes() for p in run.glob("segments.*")} == before  # nothing written


def test_contact_sheet_shows_sent_frames_in_route_order(tmp_path):
    run = tmp_path / "S1"
    colours = {0: (220, 30, 30), 1: (30, 200, 30), 4: (30, 30, 220)}  # RGB
    recs = []
    for k, status in ((4, "send"), (2, "skip"), (1, "audit"), (0, "send")):
        r = make_record(k=k, status=status, size=(160, 90))
        if k in colours:
            write_frame(run, r, np.full((90, 160, 3), colours[k], np.uint8))
        recs.append(r)
    sheet_path = write_contact_sheet(run, recs, cols=2, cell_w=80)
    sheet = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)[:, :, ::-1]  # to RGB
    cell_h, caption = 45, 36
    assert sheet.shape == (2 * (cell_h + caption), 160, 3)
    for i, k in enumerate((0, 1, 4)):  # route order; the skipped k=2 has no frame and is not shown
        r, c = divmod(i, 2)
        px = sheet[r * (cell_h + caption) + 15, c * 80 + 40].astype(int)
        assert np.abs(px - np.array(colours[k])).max() < 40, (k, px)
    assert write_contact_sheet(run, recs, cols=2, cell_w=80, max_items=2) == sheet_path
    assert cv2.imread(str(sheet_path)).shape[0] == cell_h + caption  # first 2 only
    assert write_contact_sheet(run, [recs[1]]) is None
