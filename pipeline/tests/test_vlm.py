"""Folder-batch round trip against a fake classifier server that mimics the VLM team's API."""
import json
import importlib.util
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from geoai.config import Config
from geoai.records import write_png, write_sidecar
from geoai.vlm import export_batch, import_batch, verify_sums

RUNNER = Path(__file__).resolve().parent.parent / "tools" / "run_vlm_batch.py"
_spec = importlib.util.spec_from_file_location("run_vlm_batch", RUNNER)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

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
        ans = ANSWERS.get(name, {"distress_types": ["Normal"]})
        self._send(200, {"stage1_confidence": 0.9, "stage2_confidence": 0.8, "needs_expert_review": False,
                         "description": "fake", **ans})


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.calls = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


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
    assert summary["valid"] == 4  # "N/A" (crop 30) and the empty frame-30 answer are not valid labels
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


@pytest.mark.parametrize("tail", [b'{"request_id":', b'{"description":"\xe2\x82'])
def test_resume_repairs_torn_response_without_losing_next_answer(tmp_path, server, tail):
    run = _fake_run(tmp_path)
    batch = tmp_path / "batch_torn"
    export_batch([run], batch, Config())
    first = _run_runner(batch, server, "--limit", "1")
    assert first.returncode == 1
    response_path = batch / "responses.jsonl"
    good = response_path.read_bytes()
    response_path.write_bytes(good + tail)
    resumed = _run_runner(batch, server)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    rows = [json.loads(line) for line in response_path.read_text().splitlines()]
    assert len(rows) == 6
    assert len({r["request_id"] for r in rows}) == 6
    assert import_batch(batch, [run], Config())["missing"] == 0


def test_complete_response_without_newline_is_preserved(tmp_path):
    path = tmp_path / "responses.jsonl"
    record = {"request_id": "one", "http_status": 200, "response": {"distress_types": ["Normal"]}}
    original = json.dumps(record).encode()
    path.write_bytes(original)
    runner.repair_response_tail(path)
    assert path.read_bytes() == original + b"\n"
    assert runner.answered(path) == {"one"}


@pytest.mark.parametrize("status,response", [(429, {"detail": "rate limited"}), (200, None), (200, [])])
def test_unsuccessful_responses_remain_retryable(tmp_path, monkeypatch, status, response):
    batch = tmp_path / "batch_retry"
    batch.mkdir()
    image = batch / "image.png"
    image.write_bytes(b"fake image")
    req = {"request_id": "one", "kind": "crop", "image": image.name, "image_sha256": runner.sha256(image)}
    (batch / "requests.jsonl").write_text(json.dumps(req) + "\n")
    (batch / "SHA256SUMS").write_text("")
    calls = []

    def request(url, *args, **kwargs):
        if url.endswith("/health"):
            return 200, {"model": "fake"}
        calls.append(url)
        return status, response

    monkeypatch.setattr(runner, "_request", request)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["run_vlm_batch.py", str(batch), "--min-interval", "0"])
    assert runner.main() == 1
    assert len(calls) == (6 if status == 429 else 1)
    saved = json.loads((batch / "responses.jsonl").read_text())
    assert saved["error"]
    assert saved["http_status"] == status
    assert runner.answered(batch / "responses.jsonl") == set()


def test_answered_ignores_malformed_records(tmp_path):
    path = tmp_path / "responses.jsonl"
    rows = [None, [], {}, {"request_id": "old-bug", "error": None, "http_status": None, "response": None}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert runner.answered(path) == set()


def test_import_reports_malformed_payloads_and_keeps_duplicate_conflicts(tmp_path):
    run = _fake_run(tmp_path)
    batch = tmp_path / "batch_malformed"
    export_batch([run], batch, Config())
    reqs = [json.loads(line) for line in (batch / "requests.jsonl").read_text().splitlines()]
    rows = [None, [], {"request_id": []}]
    for req, payload in zip(reqs[1:], [[], {"distress_types": 7}, {"distress_types": {}},
                                      {"distress_types": [None]}, {"distress_types": ["Normal"]}]):
        rows.append({"request_id": req["request_id"], "http_status": 200, "response": payload})
    # The third response agrees with the second, but does not erase the earlier conflict.
    for label in ("Potholes", "Normal", "Normal"):
        rows.append({"request_id": reqs[0]["request_id"], "http_status": 200,
                     "response": {"distress_types": [label]}})
    (batch / "responses.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = import_batch(batch, [run], Config())
    assert summary["answered"] == 6
    assert summary["valid"] == 1
    assert summary["problems"] == {"bad_json": 2, "unknown_request": 1, "duplicate_conflict": 1}
    assert "conflicting duplicate responses" in (run / "vlm_results_batch_malformed.csv").read_text()
