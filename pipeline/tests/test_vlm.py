"""Folder-batch round trip against a fake classifier server that mimics the VLM team's API."""
import email.parser
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from geoai.config import Config
from geoai.records import write_png, write_sidecar
from geoai.vlm import _parse, answer_class, export_batch, health_identity, import_batch, verify_sums

RUNNER = Path(__file__).resolve().parent.parent / "tools" / "run_vlm_batch.py"

# Fake answers keyed by the uploaded file name.
ANSWERS = {
    "S_0000010_00.png": {"stage1_label": "Distress", "distress_types": ["Potholes"], "severity": "Medium"},
    "S_0000020_00.png": {"stage1_label": "Normal", "distress_types": ["Normal - No distress detected"], "severity": "None"},
    "S_0000030_00.png": {"stage1_label": "Distress", "distress_types": ["N/A"], "severity": "Unknown"},
    "S_0000010_frame.png": {"stage1_label": "Distress", "distress_types": ["Potholes"], "severity": "Medium"},
    "S_0000020_frame.png": {"stage1_label": "Distress", "distress_types": ["Alligator Cracking"], "severity": "Low"},
    "S_0000030_frame.png": {"stage1_label": "Normal", "distress_types": [], "severity": "None"},
}


class _Handler(BaseHTTPRequestHandler):
    calls = []
    bodies = []            # (Content-Type header, raw body) of every POST
    echo_profile = False   # True: answer like the VLM server that knows request profiles

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version": "v2:9da1601cfec54f76",
                             "quantization": "nf4", "max_pixels": 1048576})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = self.rfile.read(n)
        name = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
        assert b"\x89PNG" in body
        _Handler.calls.append(name)
        _Handler.bodies.append((self.headers["Content-Type"], body))
        ans = ANSWERS.get(name, {"distress_types": ["Normal"]})
        extra = {}
        if _Handler.echo_profile:
            fields = _form_fields(self.headers["Content-Type"], body)
            if "profile" in fields:
                extra = {"profile": fields["profile"].decode(), "views_recorded": fields["profile"] != b"video"}
        self._send(200, {"stage1_confidence": 0.9, "stage2_confidence": 0.8, "needs_expert_review": False,
                         "description": "fake", **ans, **extra})


def _form_fields(content_type: str, body: bytes) -> dict:
    """Multipart form fields by name, parsed with the standard library's MIME parser (independent of the runner)."""
    msg = email.parser.BytesParser().parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in msg.get_payload()}


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.calls, _Handler.bodies, _Handler.echo_profile = [], [], False
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _fake_run(root: Path) -> Path:
    run = root / "run" / "S"
    for sub in ("crops", "frames", "sidecars"):
        (run / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for k, fg in enumerate((10, 20, 30)):
        rid = f"S_{fg:07d}_00"
        fid = f"S_{fg:07d}_frame"
        rec = {"id": rid, "status": "sent", "session_id": "S", "frame_global": fg, "frame_request": fid,
               "utc_ms": 1.79e12 + k, "camera": {"lat": 12.97, "lon": 77.59},
               "anomaly": {"lat": 12.97 + k * 1e-5, "lon": 77.59}, "evidence": 10.0 * (k + 1), "kinds": ["dino"],
               "crop": {"file": f"{rid}.png"}}
        write_png(run / "crops" / f"{rid}.png", (rng.random((504, 504, 3)) * 255).astype(np.uint8), rec)
        write_png(run / "frames" / f"{fid}.png", (rng.random((756, 1344, 3)) * 255).astype(np.uint8), None)
        write_sidecar(run / "sidecars" / f"{rid}.json", rec)
    return run


def _run_runner(batch: Path, url: str, *extra):
    return subprocess.run([sys.executable, str(batch / "run_vlm_batch.py"), str(batch), "--url", url,
                           "--min-interval", "0", *extra], capture_output=True, text=True, timeout=120)


def test_round_trip_with_resume(tmp_path, server):
    run = _fake_run(tmp_path)
    batch = tmp_path / "batch_001"
    info = export_batch([run], batch, Config())
    assert info["n_crops"] == 3 and info["n_frames"] == 3
    assert verify_sums(batch) == []
    # Interrupted after 2 images, then resumed: every image answered exactly once.
    r1 = _run_runner(batch, server, "--limit", "2")
    assert r1.returncode == 1, r1.stderr
    r2 = _run_runner(batch, server)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    lines = [json.loads(l) for l in (batch / "responses.jsonl").read_text().splitlines()]
    assert len(lines) == 6 and len({l["request_id"] for l in lines}) == 6
    assert sorted(_Handler.calls) == sorted(ANSWERS)
    run_info = json.loads((batch / "vlm_run.json").read_text())
    assert run_info["health"]["prompt_version"] == "v2:9da1601cfec54f76"

    summary = import_batch(batch, [run], Config())
    assert summary["answered"] == 6 and summary["missing"] == 0
    assert summary["valid"] == 5  # "N/A" (crop 30) is not a valid label; frame 30 (Stage 1 Normal, no Stage 2) is Normal
    rows = (run / "vlm_results_batch_001.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 4  # header + 3 crops
    by_id = {r.split(",")[0]: r for r in rows[1:]}
    assert ",Potholes," in by_id["S_0000010_00"] and ",1," in by_id["S_0000010_00"]  # agrees with its frame
    assert ",Normal," in by_id["S_0000020_00"]
    assert "outside the IRC:82 list" in by_id["S_0000030_00"]
    gj = json.loads((run / "vlm_results_batch_001.geojson").read_text())
    assert len(gj["features"]) == 3 and gj["features"][0]["geometry"]["coordinates"][0] == pytest.approx(77.59)


def test_runner_refuses_altered_batch(tmp_path, server):
    run = _fake_run(tmp_path)
    batch = tmp_path / "batch_002"
    export_batch([run], batch, Config())
    img = next((batch / "images").glob("*.png"))
    data = bytearray(img.read_bytes())
    data[-20] ^= 0xFF
    img.write_bytes(bytes(data))
    r = _run_runner(batch, server)
    assert r.returncode == 2 and "SHA256SUMS" in r.stderr
    with pytest.raises(ValueError):
        import_batch(batch, [run], Config())


# What the VLM system's server really returns (its PavementClassifier.predict and GET /health), not a fake's
# idealised version: an answer for every Stage 1 outcome, and its /health identity fields.
@pytest.mark.parametrize("resp, cls, labels, valid", [
    ({"stage1_label": "Normal", "distress_types": [], "severity": "None"}, "normal", ["Normal"], True),
    ({"stage1_label": "Distress", "distress_types": ["Potholes", "Ravelling"]}, "distress", ["Potholes", "Ravelling"], True),
    ({"stage1_label": "Distress", "distress_types": []}, "unknown", [], False),   # Stage 2 named nothing
    ({"stage1_label": "Normal", "distress_types": ["N/A"]}, "unknown", ["N/A"], False),  # Stage 2 forced on Normal
    ({}, "unknown", [], False),
])
def test_answer_of_each_server_outcome(resp, cls, labels, valid):
    allowed = set(Config().vlm.labels)
    res = _parse({"response": resp}, {"request_id": "r", "kind": "segment"}, allowed)
    assert res["labels"] == labels and res["valid"] is valid
    assert answer_class(res, allowed) == (cls, [lab for lab in labels if lab in allowed and lab != "Normal"])


def test_health_identity_of_the_real_server():
    cfg = Config()
    real = {"model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "prompts_version": "v2", "quantization_bits": 4}
    assert health_identity({"health": real}, cfg) == {
        "model_reported": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version_reported": "v2", "model_matches": True,
        "prompt_matches": True, "consistent": True}
    ident = health_identity({"health": dict(real, prompts_version="v1")}, cfg)
    assert ident["prompt_matches"] is False
    assert health_identity({"health": dict(real, prompts_version="v2:0000000000000000")}, cfg)["prompt_matches"] is False
    assert health_identity({"health": {}}, cfg)["prompt_matches"] is None


def _batch(tmp_path: Path, name: str) -> Path:
    batch = tmp_path / name
    export_batch([_fake_run(tmp_path)], batch, Config())
    return batch


def _runner_module(batch: Path):
    spec = importlib.util.spec_from_file_location("runner_opts", batch / "run_vlm_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_profile_field(tmp_path, server):
    batch = _batch(tmp_path, "batch_p")
    _Handler.echo_profile = True
    # No --min-interval: the URL is 127.0.0.1, so the runner does not pace (the server exempts loopback).
    r = subprocess.run([sys.executable, str(batch / "run_vlm_batch.py"), str(batch), "--url", server,
                        "--profile", "video"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "did not confirm" not in r.stderr
    images = {p.name: p.read_bytes() for p in (batch / "images").glob("*.png")}
    assert len(_Handler.bodies) == 6
    for ctype, body in _Handler.bodies:
        fields = _form_fields(ctype, body)
        assert fields["profile"] == b"video"
        name = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
        assert fields["file"] == images[name]     # the image arrives byte-exact next to the field
    lines = [json.loads(x) for x in (batch / "responses.jsonl").read_text().splitlines()]
    assert {ln["response"]["profile"] for ln in lines} == {"video"} and all(ln["error"] is None for ln in lines)
    run = json.loads((batch / "vlm_run.json").read_text())["runs"][-1]
    assert (run["profile"], run["min_interval_s"], run["batch_check"], run["profile_not_confirmed"]) == (
        "video", 0.0, "verified", 0)
    assert run["runner_version"] == "3"


def test_runner_without_profile_sends_only_the_file(tmp_path, server):
    batch = _batch(tmp_path, "batch_np")
    r = _run_runner(batch, server, "--limit", "2")
    assert r.returncode == 1, r.stdout + r.stderr
    assert len(_Handler.bodies) == 2
    for ctype, body in _Handler.bodies:
        assert list(_form_fields(ctype, body)) == ["file"]
    run = json.loads((batch / "vlm_run.json").read_text())["runs"][-1]
    assert run["profile"] is None and "profile_not_confirmed" not in run


def test_runner_warns_when_server_ignores_profile(tmp_path, server):
    batch = _batch(tmp_path, "batch_old")
    r = _run_runner(batch, server, "--profile", "video", "--limit", "3")   # this fake does not echo it
    assert r.returncode == 1 and "did not confirm profile 'video'" in r.stderr
    assert r.stderr.count("did not confirm") == 1
    run = json.loads((batch / "vlm_run.json").read_text())["runs"][-1]
    assert run["profile_not_confirmed"] == 3 and run["ok"] == 3


@pytest.mark.parametrize("url, extra, interval", [
    ("http://127.0.0.1:8000", [], 0.0),
    ("http://localhost:8000", [], 0.0),
    ("http://[::1]:8000", [], 0.0),
    ("http://127.0.0.1:8000", ["--min-interval", "3"], 3.0),   # an explicit value wins
    ("http://192.168.1.20:8000", [], 6.5),                      # another machine: 60 s / 10 requests
    ("https://abc.trycloudflare.com", [], 6.5),
])
def test_runner_default_interval(tmp_path, url, extra, interval):
    batch = _batch(tmp_path, "batch_i")
    runner = _runner_module(batch)
    seen = []
    runner.run = lambda d, a: seen.append(a) or 0
    assert runner.main([str(batch), "--url", url, *extra]) == 0
    assert seen[0].min_interval == interval and seen[0].profile is None and seen[0].skip_batch_check is False


def test_runner_skip_batch_check(tmp_path, server):
    batch = _batch(tmp_path, "batch_s")
    imgs = sorted((batch / "images").glob("*.png"))
    data = bytearray(imgs[0].read_bytes())
    data[-20] ^= 0xFF
    imgs[0].write_bytes(bytes(data))
    r = _run_runner(batch, server)
    assert r.returncode == 2 and "SHA256SUMS" in r.stderr and not _Handler.calls   # default: verified, refused
    r = _run_runner(batch, server, "--skip-batch-check")
    assert "against SHA256SUMS ..." not in r.stdout and "not re-checking SHA256SUMS" in r.stdout
    # The changed image is still caught by the per-upload hash: recorded as an error, never as answered.
    lines = {json.loads(x)["request_id"]: json.loads(x) for x in (batch / "responses.jsonl").read_text().splitlines()}
    reqs = {json.loads(x)["image"]: json.loads(x)["request_id"]
            for x in (batch / "requests.jsonl").read_text().splitlines()}
    bad_id = reqs[f"images/{imgs[0].name}"]
    assert lines[bad_id]["error"] == "image checksum differs from requests.jsonl (file changed on disk)"
    assert sum(ln["error"] is None for ln in lines.values()) == 5
    assert r.returncode == 1
    run = json.loads((batch / "vlm_run.json").read_text())["runs"][-1]
    assert run["batch_check"] == "skipped"
