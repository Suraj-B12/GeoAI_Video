"""Segment mode end to end on simulated session packages (docs/pipeline/13 §7).

The simulated camera is the S21 mount (23 deg below horizontal, 1.35 m high, 16:9, 67 deg HFOV) at
960 x 540, so the road from about 1.43 m to 30 m ahead is in view and a 5 m segment fits. Expected values
come from the simulator's own inputs (camera, speed, frame times, GNSS gap, where each distress was
drawn), from plain geometry written out here, and from re-decoding the video, never from the code
under test. Probe off, as in test_e2e_sim.py (procedural asphalt is outside the probe's domain).

Chain: process (gate mode, twice, and send-all) -> vlm-export -> the real run_vlm_batch.py against a fake
server that answers like the VLM system (Stage 1 Normal: stage1_label "Normal", distress_types []) ->
vlm-import -> analyze -> gate-calibrate (the last three in a Python without torch, OpenCV, PyAV, SciPy and
shapely, as on the VLM machine) -> process again with the calibration.

Slow (about 3 minutes with a GPU, longer on a CPU): run with `pytest -m slow`.
"""
import hashlib
import json
import math
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pymap3d
import pytest
from PIL import Image

from geoai.config import AnomalyConfig, load_config
from geoai.frames import process_session_segments
from geoai.segment_records import check_segment_png, load_segment_records, validate_segment_record
from geoai.sim import ORIGIN, T0_NS, UTC0_MS, SimSpec, simulate_session
from geoai.video import iter_segment

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not Path(AnomalyConfig().weights).exists(), reason="DINOv2 weights not downloaded")]

PIPELINE = Path(__file__).resolve().parent.parent
S21 = {"pitch_deg": 23.0, "height_m": 1.35}
OVERRIDES = {"gnss": {"default_latency_s": 0.10, "default_latency_measured": True}, "anomaly": {"probe": None}}
# Modules the VLM machine doesn't have: vlm-import, analyze and gate-calibrate must run without them.
HEAVY = ("torch", "cv2", "av", "shapely", "scipy", "skimage", "pymap3d")
LIGHT_CLI = ("import sys\n"
             f"for m in {HEAVY!r}:\n"
             "    sys.modules[m] = None\n"
             "from geoai.cli import main\n"
             "sys.exit(main(sys.argv[1:]))\n")


def near_m(spec: SimSpec) -> float:
    """Closest road in view: the bottom row's ray, (H - 1) / 2 rows below the optical axis, meets the road."""
    f = (spec.width / 2) / math.tan(math.radians(spec.hfov_deg) / 2)
    below = math.radians(spec.pitch_deg) + math.atan(((spec.height - 1) / 2) / f)
    return spec.height_m / math.tan(below)


def north(lat: float, lon: float) -> float:
    """Distance along the simulated route: it runs due north from ORIGIN."""
    return float(pymap3d.geodetic2enu(lat, lon, ORIGIN[2], *ORIGIN)[1])


def seg_north(rec: dict) -> tuple[float, float]:
    sg = rec["segment"]
    return north(sg["start"]["lat"], sg["start"]["lon"]), north(sg["end"]["lat"], sg["end"]["lon"])


def run_dir(out: Path, res: dict) -> Path:
    return out / res["session_dir"].name


def distress_ks(recs: list[dict], truth: dict) -> set[int]:
    """Segments whose span along the route holds the centroid of a simulated distress."""
    ks = set()
    for ft in truth["features"]:
        if not ft["is_distress"]:
            continue
        y = ft["centroid_enu"][1]
        ks |= {r["segment"]["k"] for r in recs if seg_north(r)[0] <= y < seg_north(r)[1]}
    return ks


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_seg")
    spec = SimSpec(seed=1, length_m=64.0, **S21)
    res = simulate_session(root / "sessions", spec)
    cfg = load_config(overrides=OVERRIDES)
    rep = process_session_segments(res["session_dir"], root / "out", cfg)
    rep2 = process_session_segments(res["session_dir"], root / "out2", cfg)
    cfg_all = load_config(overrides={**OVERRIDES, "gate": {"mode": "all"}})
    rep_all = process_session_segments(res["session_dir"], root / "out_all", cfg_all)
    return {"root": root, "spec": spec, "res": res, "truth": res["truth"], "rep": rep, "rep2": rep2, "rep_all": rep_all,
            "run": run_dir(root / "out", res), "run2": run_dir(root / "out2", res),
            "run_all": run_dir(root / "out_all", res), "recs": load_segment_records(run_dir(root / "out", res) / "sidecars")}


# ---- plan, coverage, geometry -------------------------------------------------------------------

def test_plan_follows_the_recording_and_the_view(sim):
    spec, recs, rep = sim["spec"], sim["recs"], sim["rep"]
    near = near_m(spec)
    assert rep["plan"]["near_m"] == pytest.approx(near, abs=1e-4)   # attitude from gravity written to 5 decimals
    # Camera positions span (frames - 1) frame periods at the simulated speed: 12 whole segments and a
    # partial one of ~3.5 m (kept: at least half a segment).
    n = int(spec.length_m / spec.speed_mps * spec.fps)
    total = spec.speed_mps * (n - 1) / spec.fps
    assert len(recs) == int(total // 5) + 1 == 13
    assert [r["segment"]["k"] for r in recs] == list(range(13))
    assert all(r["segment"]["length_m"] == pytest.approx(5.0, abs=1e-9) for r in recs[:-1])
    assert recs[-1]["segment"]["partial"] and recs[-1]["segment"]["length_m"] == pytest.approx(total - 60, abs=0.3)
    for a, b in zip(recs, recs[1:]):
        assert a["segment"]["s1_m"] == pytest.approx(b["segment"]["s0_m"], abs=1e-9)
    # The first segment starts near_m + window_m ahead of the first frame's camera (at the route start,
    # moved by the centre row's exposure-middle time); its chainage is the form's 12.000 km plus that.
    t_centre = (spec.exposure_ns / 2 + spec.rs_skew_ns * ((spec.height - 1) / 2) / spec.height) / 1e9
    y0 = spec.speed_mps * t_centre + near + 1.0
    assert seg_north(recs[0])[0] == pytest.approx(y0, abs=0.5)           # GNSS noise 0.25 m
    for k, r in enumerate(recs):
        assert r["segment"]["chainage_start_km"] == pytest.approx(12.0 + (near + 1.0 + 5.0 * k) / 1000, abs=1e-7)
        a, b = seg_north(r)
        assert b - a == pytest.approx(r["segment"]["length_m"], abs=0.3)


def test_every_segment_covered_from_its_window(sim):
    spec, recs = sim["spec"], sim["recs"]
    near = near_m(spec)
    assert {r["status"] for r in recs} <= {"send", "audit", "skip"}
    rep = sim["rep"]
    assert rep["coverage"]["not_covered_m"] == 0.0
    assert rep["coverage"]["covered_m"] == pytest.approx(sum(r["segment"]["length_m"] for r in recs))
    for r in recs:
        f = r["frame"]
        # Candidate window: the segment's near edge between near_m and near_m + 1 m ahead of the camera.
        assert near - 1e-6 <= f["near_edge_ahead_m"] <= near + 1.0 + 1e-6
        # Frames 0.53 m apart (8 m/s, 15 fps): one or two in each 1 m window.
        assert r["quality"]["n_candidates"] in (1, 2) and r["quality"]["sharpness_rank"] == 1
        assert r["quality"]["reason"] == "sharpest"
        # UTC of the frame's centre row (GNSS time with the 100 ms latency removed).
        t_ns = f["t_ns"]
        centre_ms = (spec.exposure_ns / 2 + spec.rs_skew_ns * ((spec.height - 1) / 2) / spec.height) / 1e6
        assert f["utc_ms"] == pytest.approx(UTC0_MS + (t_ns - T0_NS) / 1e6 + centre_ms, abs=1.0)
        assert f["frame_global"] == (t_ns - T0_NS) // round(1e9 / spec.fps)
        # The camera stood near_edge_ahead_m before the segment start.
        cam = north(r["camera"]["lat"], r["camera"]["lon"])
        assert seg_north(r)[0] - cam == pytest.approx(f["near_edge_ahead_m"], abs=0.3)


def test_every_simulated_distress_is_in_a_sent_segment(sim):
    recs = sim["recs"]
    n_distress = 0
    for ft in sim["truth"]["features"]:
        if not ft["is_distress"]:
            continue
        n_distress += 1
        y = ft["centroid_enu"][1]
        # The segment holding the centroid; within GNSS noise of a boundary, either neighbour.
        near_ks = [r for r in recs if seg_north(r)[0] - 0.35 <= y < seg_north(r)[1] + 0.35]
        assert near_ks, (ft["kind"], y)
        assert any(r["status"] == "send" for r in near_ks), (ft["kind"], y, [(r["id"], r["status"]) for r in near_ks])
    assert n_distress == 5


# ---- records and files --------------------------------------------------------------------------

def test_sidecars_valid_pngs_match_and_frames_are_the_video_frames(sim):
    run, recs, spec = sim["run"], sim["recs"], sim["spec"]
    sent = [r for r in recs if r["status"] in ("send", "audit")]
    assert sent
    for r in recs:
        assert validate_segment_record(r) == [], r["id"]
    for r in sent:
        png = run / r["frame"]["file"]
        assert check_segment_png(png, r, offset_minutes=330) == [], r["id"]
        assert hashlib.sha256(png.read_bytes()).hexdigest() == r["frame"]["sha256"]
    for r in recs:
        if r["status"] == "skip":
            assert r["frame"]["file"] is None and not (run / "frames" / f"{r['id']}.png").exists()
    # Full resolution and lossless: the PNG is exactly the decoded video frame.
    want = {r["frame"]["frame_in_segment"]: r for r in sent}
    sess = sim["res"]["session_dir"]
    import pandas as pd
    rows = pd.read_csv(sess / "frames.csv")
    seen = 0
    for fr in iter_segment(sess / "video" / "seg_0001.mp4", rows, segment=1, want=lambda i, row: i in want):
        if fr.rgb is None:
            continue
        with Image.open(run / want[fr.index]["frame"]["file"]) as im:
            assert im.size == (spec.width, spec.height)
            assert np.array_equal(np.asarray(im.convert("RGB")), fr.rgb)
        seen += 1
    assert seen == len(sent)
    assert (run / "contact_sheet.jpg").exists() and (run / "segments.csv").exists()
    gj = json.loads((run / "segments.geojson").read_text(encoding="utf-8"))
    assert len(gj["features"]) == len(recs)
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 77 < lon < 78 and 12 < lat < 13


def test_reruns_are_byte_identical(sim):
    a, b = sim["run"], sim["run2"]
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for rel in files:
        if rel.name != "report.json":
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    ra, rb = (json.loads((x / "report.json").read_text(encoding="utf-8")) for x in (a, b))
    for r in (ra, rb):
        for k in ("created_utc", "timings_s"):
            r.pop(k)
    assert ra == rb


def test_report_and_structured_log(sim):
    rep, recs = sim["rep"], sim["recs"]
    on_disk = json.loads((sim["run"] / "report.json").read_text(encoding="utf-8"))
    assert on_disk["mode"] == "segments" and on_disk["records"] == rep["records"]
    counts = {s: sum(1 for r in recs if r["status"] == s) for s in ("send", "audit", "skip", "not_covered")}
    assert rep["records"] == counts
    n_sent = counts["send"] + counts["audit"]
    assert rep["estimated_vlm_hours"] == pytest.approx(n_sent * 14 / 3600)
    assert rep["gate"]["pass_rate"] == pytest.approx(sum(r["gate"]["decision_if_gated"] == "send" for r in recs) / len(recs))
    assert rep["quality"]["candidates_per_segment"]["n"] == len(recs)
    assert rep["quality"]["low_light_share_chosen"] == 0.0
    assert "hairline" in rep["limits"]
    for k in ("bank_total", "decode_pass", "gate", "dinov2", "total"):
        assert rep["timings_s"][k] >= 0
    logs = list((sim["root"] / "out" / "logs").glob(f"*/{sim['res']['session_dir'].name}.jsonl"))
    assert len(logs) == 1
    lines = [json.loads(x) for x in logs[0].read_text(encoding="utf-8").splitlines()]
    events = [(x["stage"], x["event"]) for x in lines]
    for stage in ("load", "trajectory", "geometry", "bank", "decode", "outputs"):
        assert (stage, "start") in events and (stage, "end") in events
    assert events[0] == ("run", "start") and events[-1] == ("run", "end")
    assert {x["run_id"] for x in lines} == {lines[0]["run_id"]}


# ---- GNSS gap -----------------------------------------------------------------------------------

def test_gnss_gap_segments_not_covered_with_reason(tmp_path):
    """No fixes between 1.5 s and 6.5 s: fixes remain at 1 s and 7 s (8 m and 56 m along the route), and the
    6 s between them is longer than gnss.max_gap_s, so camera positions in between have no valid position."""
    spec = SimSpec(seed=3, length_m=80.0, width=640, height=360, gnss_gap_s=(1.5, 6.5), **S21)
    res = simulate_session(tmp_path / "s", spec)
    rep = process_session_segments(res["session_dir"], tmp_path / "o", load_config(overrides=OVERRIDES))
    recs = load_segment_records(run_dir(tmp_path / "o", res) / "sidecars")
    near = near_m(spec)
    gap = (8.0, 56.0)
    step = spec.speed_mps / spec.fps                     # 0.53 m between frames
    n_nc = 0
    for r in recs:
        a, _ = seg_north(r)
        lo, hi = a - near - 1.0, a - near                 # the camera window along the route
        assert validate_segment_record(r) == []
        if r["status"] == "not_covered":
            n_nc += 1
            assert r["quality"]["reason"] == "gnss_gap", r["id"]
            assert gap[0] - 0.5 <= lo and hi <= gap[1] + 0.5, (r["id"], lo, hi)
        else:
            cam = north(r["camera"]["lat"], r["camera"]["lon"])
            assert not gap[0] + 0.5 < cam < gap[1] - 0.5, (r["id"], cam)
        if gap[0] + step + 0.5 < lo and hi < gap[1] - step - 0.5:
            assert r["status"] == "not_covered", r["id"]  # window wholly inside the gap
    assert n_nc >= 7
    nc = rep["coverage"]["not_covered_by_reason"]["gnss_gap"]
    assert nc["segments"] == n_nc
    assert nc["length_m"] == pytest.approx(sum(r["segment"]["length_m"] for r in recs if r["status"] == "not_covered"))
    assert rep["coverage"]["not_covered_m"] == pytest.approx(nc["length_m"])


# ---- VLM hand-off, analysis and calibration --------------------------------------------------

class _VLM(BaseHTTPRequestHandler):
    """Answers like the VLM system: /health as it reports itself; Stage 1 Normal gives no distress_types."""
    distress: set[str] = set()
    calls: list[str] = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"status": "ok", "model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "prompts_version": "v2",
                    "quantization_bits": 4, "max_image_pixels": 1048576})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        name = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
        _VLM.calls.append(name)
        if name[:-4] in _VLM.distress:
            ans = {"stage1_label": "Distress", "stage1_confidence": 0.93, "distress_types": ["Potholes"],
                   "severity": "Medium", "stage2_confidence": 0.85, "needs_expert_review": False, "description": "d"}
        else:
            ans = {"stage1_label": "Normal", "stage1_confidence": 0.91, "distress_types": [], "severity": "None",
                   "stage2_confidence": 0.0, "needs_expert_review": False, "description": ""}
        self._send(ans)


def _cli(*args, light=False, check=True):
    cmd = [sys.executable, "-c", LIGHT_CLI, *map(str, args)] if light else [sys.executable, "-m", "geoai", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=PIPELINE)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_export_runner_import_analyze_calibrate(sim, tmp_path):
    run = sim["run_all"]
    recs = load_segment_records(run / "sidecars")
    assert all(r["status"] == "send" and r["gate"]["mode"] == "all" for r in recs)  # send-all: every segment
    pos = distress_ks(recs, sim["truth"])
    assert 3 <= len(pos) < len(recs)
    batch = tmp_path / "batch_001"
    _cli("vlm-export", run, "--batch", batch)
    reqs = [json.loads(x) for x in (batch / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [q["request_id"] for q in reqs] == [r["id"] for r in recs]

    _VLM.distress = {r["id"] for r in recs if r["segment"]["k"] in pos}
    _VLM.calls = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _VLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = subprocess.run([sys.executable, str(batch / "run_vlm_batch.py"), str(batch), "--url",
                            f"http://127.0.0.1:{srv.server_address[1]}", "--min-interval", "0"],
                           capture_output=True, text=True, timeout=300)
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(_VLM.calls) == sorted(f"{x['id']}.png" for x in recs)

    imp = json.loads(_cli("vlm-import", batch, run, light=True).stdout)
    assert imp["answered"] == imp["requests"] == len(recs) and imp["valid"] == len(recs)
    assert imp["classes"] == {"distress": len(pos), "normal": len(recs) - len(pos), "unknown": 0}
    assert imp["model_matches"] is True and imp["prompt_matches"] is True
    for rec in load_segment_records(run / "sidecars"):
        assert validate_segment_record(rec) == []
        assert rec["vlm"]["labels"] == (["Potholes"] if rec["segment"]["k"] in pos else ["Normal"])
        assert check_segment_png(run / rec["frame"]["file"], rec, offset_minutes=330) == []

    _cli("analyze", batch, light=True)
    ana = batch / "analysis"
    import csv
    with open(ana / "label_summary.csv", encoding="utf-8-sig", newline="") as f:
        summ = {(x["scope"], x["label"]): x for x in csv.DictReader(f)}
    dist_m = sum(r["segment"]["length_m"] for r in recs if r["segment"]["k"] in pos)
    all_m = sum(r["segment"]["length_m"] for r in recs)
    assert float(summ[("all", "Any distress")]["length_m"]) == pytest.approx(dist_m, abs=1e-3)
    assert float(summ[("all", "Potholes")]["pct_of_surveyed"]) == pytest.approx(100 * dist_m / all_m, abs=1e-4)
    ge = json.loads((ana / "gate_eval.json").read_text(encoding="utf-8"))["send_all"]
    gp = {r["segment"]["k"] for r in recs if r["gate"]["decision_if_gated"] == "send"}
    every = {r["segment"]["k"] for r in recs}
    assert ge["confusion"] == {"tp": len(gp & pos), "fp": len(gp - pos), "fn": len(pos - gp),
                               "tn": len(every - gp - pos)}

    cal_path = tmp_path / "gate_calibration.json"
    _cli("gate-calibrate", batch, "--target-recall", "1.0", "--out", cal_path, light=True)
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    assert cal["n_rows"] == len(recs) and cal["n_positive"] == len(pos)
    assert cal["source"]["gated_rows"] == 0 and cal["source"]["by_mode_and_status"] == {"all:send": len(recs)}
    th = cal["any_of"]["thresholds"]
    passes = {r["segment"]["k"] for r in recs
              if any(th[s] is not None and r["gate"]["scores"][s] is not None and r["gate"]["scores"][s] >= th[s]
                     for s in ("probe_frac", "evidence", "crack_mm"))}
    assert pos <= passes and cal["any_of"]["n_pass"] == len(passes) and cal["any_of"]["recall"] == 1.0

    # Gate mode with the calibration: same frames, same scores, so every VLM-distress segment is sent.
    cfg = load_config(overrides={**OVERRIDES, "gate": {"calibration": str(cal_path)}})
    process_session_segments(sim["res"]["session_dir"], tmp_path / "out_cal", cfg)
    cal_recs = load_segment_records(run_dir(tmp_path / "out_cal", sim["res"]) / "sidecars")
    sha = hashlib.sha256(cal_path.read_bytes()).hexdigest()
    for rec in cal_recs:
        g = rec["gate"]
        assert g["thresholds"]["source"] == "calibration" and g["thresholds"]["calibration_sha256"] == sha
        assert (g["decision_if_gated"] == "send") == (rec["segment"]["k"] in passes)
        if rec["segment"]["k"] in pos:
            assert rec["status"] == "send"


def test_cli_process_send_all_and_mode_checks(sim, tmp_path):
    over = tmp_path / "o.json"
    over.write_text(json.dumps(OVERRIDES), encoding="utf-8")
    r = _cli("--config", over, "process", sim["res"]["session_dir"], "--out", tmp_path / "o", "--send-all")
    assert "covered" in r.stdout
    recs = load_segment_records(run_dir(tmp_path / "o", sim["res"]) / "sidecars")
    # The send-all run's sidecars, without the vlm blocks the import test may have added to them.
    assert recs == [{k: v for k, v in x.items() if k != "vlm"} for x in load_segment_records(sim["run_all"] / "sidecars")]
    bad = _cli("process", sim["res"]["session_dir"], "--out", tmp_path / "c", "--mode", "crops", "--send-all", check=False)
    assert bad.returncode == 2 and "--send-all" in bad.stderr
