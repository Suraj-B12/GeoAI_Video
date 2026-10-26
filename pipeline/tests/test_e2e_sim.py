"""End to end on a simulated session package (spec v1.1): known distresses, markings, a shadow,
1 Hz GNSS with noise and 100 ms latency, rolling shutter, H.264 fragmented MP4.

Slow (several minutes on a CPU): run with `pytest -m slow`.
"""
import json
from pathlib import Path

import numpy as np
import pymap3d
import pytest
from PIL import ExifTags, Image

from geoai.config import AnomalyConfig, load_config
from geoai.evaluate import evaluate, load_records
from geoai.extract import process_session
from geoai.qwen import MAX_PIXELS, smart_resize
from geoai.records import dms_to_deg, read_png_record
from geoai.sim import SimSpec, simulate_session

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not Path(AnomalyConfig().weights).exists(), reason="DINOv2 weights not downloaded")]

# Probe off for the accuracy checks: procedural asphalt is outside the domain of the probe trained on
# real roads (see test_anomaly.py); test_probe_path_end_to_end runs it once on the same data.
OVERRIDES = {"geometry": {"max_gsd_mm": 8.0},  # the simulated 960x540 camera is coarser than 4K
             "gnss": {"default_latency_s": 0.10, "default_latency_measured": True},
             "anomaly": {"probe": None}}


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    res = simulate_session(root / "sessions", SimSpec(seed=1, length_m=64.0))
    cfg = load_config(overrides=OVERRIDES)
    rep = process_session(res["session_dir"], root / "out", cfg)
    run = root / "out" / res["session_dir"].name
    return {"root": root, "res": res, "rep": rep, "run": run, "cfg": cfg}


def test_every_distress_found_and_placed(sim):
    ev = evaluate(sim["run"], sim["res"]["truth"])
    assert ev["distress_features"] == 5
    assert ev["detected"] == 5, ev["per_feature"]
    assert ev["max_outside_distance_m"] < 0.35       # GNSS noise 0.25 m + geometry
    assert ev["records_elsewhere"] <= 1, ev["unmatched"]  # plain asphalt


def test_lookalikes_rank_below_distress(sim):
    ev = evaluate(sim["run"], sim["res"]["truth"])
    distress = [e for k, v in ev["per_feature"].items() for s, e in v if k.split(":")[1] not in ("marking", "shadow")]
    looks = [e for k, v in ev["per_feature"].items() for s, e in v if k.split(":")[1] in ("marking", "shadow")]
    top = sorted(distress + looks, reverse=True)[:5]
    assert all(t in distress for t in top), (sorted(distress), sorted(looks))


def test_no_cap_every_non_duplicate_is_sent_and_hashed(sim):
    import hashlib
    recs = load_records(sim["run"])
    assert sim["rep"]["budget"] is None
    assert {r["status"] for r in recs} <= {"sent", "duplicate"}
    for r in recs:
        if r["status"] == "duplicate":
            assert any(k["id"] == r["duplicate_of"] and k["status"] == "sent" for k in recs)
    for r in (r for r in recs if r["status"] == "sent"):
        f = sim["run"] / "crops" / r["crop"]["file"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == r["crop"]["sha256"]


def test_crops_fit_the_vlm_input_rules_and_carry_geotags(sim):
    recs = {r["id"]: r for r in load_records(sim["run"])}
    crops = sorted((sim["run"] / "crops").glob("*.png"))
    assert crops
    for p in crops:
        with Image.open(p) as im:
            w, h = im.size
        assert w % 28 == 0 and h % 28 == 0 and w * h <= MAX_PIXELS
        assert smart_resize(h, w) == (h, w)                # reaches the model unchanged
        assert (w // 28) * (h // 28) >= 324                # within the VLM team's tested range
        rec, gps = read_png_record(p)
        side = recs[rec["id"]]
        G = ExifTags.GPS
        assert abs(dms_to_deg(gps[G.GPSDestLatitude], gps[G.GPSDestLatitudeRef]) - side["anomaly"]["lat"]) < 1e-7
        assert abs(dms_to_deg(gps[G.GPSDestLongitude], gps[G.GPSDestLongitudeRef]) - side["anomaly"]["lon"]) < 1e-7
        assert side["utc_iso"].endswith("Z") and side["local_iso"].endswith("+05:30")
    for f in (sim["run"] / "frames").glob("*.png"):
        with Image.open(f) as im:
            assert smart_resize(im.size[1], im.size[0]) == (im.size[1], im.size[0])


def test_capture_time_matches_truth(sim):
    """Record UTC must equal the simulated UTC of the frame (GNSS time, latency removed)."""
    spec = sim["res"]["truth"]["spec"]
    from geoai.sim import T0_NS, UTC0_MS
    for r in load_records(sim["run"]):
        expected = UTC0_MS + (r["t_ns"] - T0_NS) / 1e6
        # Record time is the anomaly's own image row (rolling shutter): within one readout of the frame start.
        assert expected - 1 <= r["utc_ms"] <= expected + spec["rs_skew_ns"] / 1e6 + spec["exposure_ns"] / 1e6 + 1


def test_report_is_complete(sim):
    rep = json.loads((sim["run"] / "report.json").read_text())
    assert rep["trajectory"]["latency_method"] == "device_default"
    assert rep["road_not_covered_m"] < 1.0
    assert rep["bank"]["960x540"]["frames"] >= 12
    gj = json.loads((sim["run"] / "anomalies.geojson").read_text())
    lon, lat = gj["features"][0]["geometry"]["coordinates"]
    assert 77 < lon < 78 and 12 < lat < 13      # [longitude, latitude] order


def test_deterministic(sim):
    rep2 = process_session(sim["res"]["session_dir"], sim["root"] / "out2", sim["cfg"])
    run2 = sim["root"] / "out2" / sim["res"]["session_dir"].name
    a = (sim["run"] / "candidates.csv").read_text(encoding="utf-8-sig")
    b = (run2 / "candidates.csv").read_text(encoding="utf-8-sig")
    assert a == b
    for p in (sim["run"] / "crops").glob("*.png"):
        assert p.read_bytes() == (run2 / "crops" / p.name).read_bytes()


def test_gnss_gap_is_skipped_and_reported(tmp_path):
    """No fixes for 5 s: frames there have no position, so they are skipped, reported, never geolocated."""
    # 80 m at 8 m/s = 10 s of frames, so frames resume after the gap (fixes at 1 s and 7 s remain).
    spec = SimSpec(seed=3, length_m=80.0, width=640, height=360, gnss_gap_s=(1.5, 6.5))
    res = simulate_session(tmp_path / "s", spec)
    rep = process_session(res["session_dir"], tmp_path / "o", load_config(overrides=OVERRIDES))
    assert len(rep["trajectory"]["gaps_s"]) == 1
    assert rep["road_not_covered_m"] > 20           # ~5 s x 8 m/s of road had no usable position
    run = tmp_path / "o" / res["session_dir"].name
    from geoai.sim import ORIGIN
    for r in load_records(run):
        e, n, _ = pymap3d.geodetic2enu(r["anomaly"]["lat"], r["anomaly"]["lon"], 0.0, *ORIGIN)
        # The gap covers along-route ~[12, 52] m minus the camera's look-ahead: nothing placed inside it.
        assert not (8.0 * 1.5 + 3.0 < n < 8.0 * 6.5 - 1.0), (r["id"], n)


def test_probe_path_end_to_end(sim, tmp_path):
    """With the probe on: the run completes and probe candidates become valid, geotagged records."""
    if not Path(AnomalyConfig().probe).exists():
        pytest.skip("probe not trained")
    over = {k: dict(v) for k, v in OVERRIDES.items()}
    over["anomaly"] = {}
    cfg = load_config(overrides=over)
    rep = process_session(sim["res"]["session_dir"], tmp_path / "p", cfg)
    assert rep["probe"]["name"] == "attain_dinov2_probe_v1"
    recs = load_records(tmp_path / "p" / sim["res"]["session_dir"].name)
    assert any("probe" in r["kinds"] for r in recs)
    assert all(r["anomaly"]["lat"] is not None for r in recs)
    ev = evaluate(tmp_path / "p" / sim["res"]["session_dir"].name, sim["res"]["truth"])
    assert ev["detected"] == ev["distress_features"]   # the probe must not hide any distress
