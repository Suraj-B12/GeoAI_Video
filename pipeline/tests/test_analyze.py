"""Segment batches end to end: export -> answers (fake responses or the real runner) -> import -> analyze.

Two synthetic runs with known VLM answers: SESS_A in gate mode (send, audit, skip, not covered, a partial
last segment) and SESS_B in send-all mode (every covered segment sent, the gate's would-be decision
recorded). Every number in label_summary.csv, per_km.csv, gate_eval.json and vlm_timing.json is
recomputed here from the truth table below with plain loops; Wilson intervals by solving the score
equation's quadratic (not the closed form the module uses) and z quantiles from scipy.
"""
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.stats import norm

from geoai import vlm
from geoai.analyze import Z95, Z_BONF, analyze, load_batch, per_km
from geoai.config import Config
from geoai.records import read_png_record, write_png, write_sidecar
from geoai.segment_records import load_segment_records, validate_segment_record, write_segment_outputs
from geoai.util import sha256_file, write_json_atomic
from geoai.vlm import export_batch, import_batch, verify_sums

from test_segment_records import make_record, write_frame

PIPELINE = Path(__file__).resolve().parent.parent
LABELS = list(Config().vlm.labels)
DISTRESS_LABELS = [lab for lab in LABELS if lab != "Normal"]

# (k, status, decision_if_gated, length_m, partial)
SEGS_A = [(0, "send", None, 5.0, False), (1, "send", None, 5.0, False), (2, "send", None, 5.0, False),
          (3, "skip", None, 5.0, False), (4, "skip", None, 5.0, False), (5, "audit", None, 5.0, False),
          (6, "skip", None, 5.0, False), (7, "send", None, 5.0, False), (8, "audit", None, 5.0, False),
          (9, "skip", None, 5.0, False), (10, "not_covered", None, 5.0, False), (11, "send", None, 3.0, True)]
SEGS_B = [(0, "send", "send", 5.0, False), (1, "send", "skip", 5.0, False), (2, "send", "send", 5.0, False),
          (3, "send", "audit", 5.0, False), (4, "send", "skip", 5.0, False), (5, "send", "send", 5.0, False),
          (6, "send", "send", 5.0, False), (7, "not_covered", None, 5.0, False)]
MODES = {"SESS_A": "gate", "SESS_B": "all"}
# The VLM's answers (distress_types as the server returns them) and what they must mean. [] is what /classify
# returns when Stage 1 says Normal: Stage 2 never runs, stage1_label is "Normal" and distress_types is empty.
ANSWERS = {
    ("SESS_A", 0): (["Potholes"], "distress", ["Potholes"]),
    ("SESS_A", 1): ("Alligator Cracking, Longitudinal Cracking", "distress", ["Alligator Cracking", "Longitudinal Cracking"]),
    ("SESS_A", 2): (["Normal - No distress detected"], "normal", []),
    ("SESS_A", 5): (["Potholes", "N/A"], "distress", ["Potholes"]),          # flagged invalid, Potholes still counts
    ("SESS_A", 7): (["Ravelling"], "distress", ["Ravelling"]),                # after an HTTP 500 line
    ("SESS_A", 8): (["Normal"], "normal", []),
    ("SESS_A", 11): (["N/A"], "unknown", []),                                 # no valid label
    ("SESS_B", 0): (["Normal", "Potholes"], "distress", ["Potholes"]),        # mixed: flagged, distress counts
    ("SESS_B", 1): ([], "normal", []),
    ("SESS_B", 2): (["Hairline Cracks"], "distress", ["Hairline Cracks"]),
    ("SESS_B", 3): (["Transverse Cracking"], "distress", ["Transverse Cracking"]),
    ("SESS_B", 4): (["Edge Cracking"], "distress", ["Edge Cracking"]),
    ("SESS_B", 5): ([], "normal", []),
}
VALID = {("SESS_A", 0), ("SESS_A", 1), ("SESS_A", 2), ("SESS_A", 7), ("SESS_A", 8), ("SESS_B", 1), ("SESS_B", 2),
         ("SESS_B", 3), ("SESS_B", 4), ("SESS_B", 5)}


def rid(sid: str, k: int) -> str:
    return f"{sid}_seg{k:05d}"


def truth() -> list[dict]:
    """One row per segment: what every analysis number must be computed from."""
    rows = []
    for sid, segs in (("SESS_A", SEGS_A), ("SESS_B", SEGS_B)):
        for k, status, dif, length, partial in segs:
            covered = status != "not_covered"
            if not covered:
                cls, labels = "not_covered", []
            elif status == "skip":
                cls, labels = "not_sent", []
            elif (sid, k) in ANSWERS:
                _, cls, labels = ANSWERS[(sid, k)]
            else:
                cls, labels = "not_answered", []
            gate_pos = None if not covered else ((dif or status) == "send")
            rows.append({"sid": sid, "k": k, "status": status, "length": length, "covered": covered, "cls": cls,
                         "labels": labels, "gate_positive": gate_pos, "mode": MODES[sid],
                         "s0": 100.0 + 5 * k, "s1": 100.0 + 5 * k + length})
    return rows


def build_runs(root: Path) -> dict[str, Path]:
    runs = {}
    for sid, segs in (("SESS_A", SEGS_A), ("SESS_B", SEGS_B)):
        run = root / "runs" / sid
        recs = []
        for k, status, dif, length, partial in segs:
            rec = make_record(sid, k, status=status, dif=dif, mode=MODES[sid], length=length, partial=partial)
            if status in ("send", "audit"):
                write_frame(run, rec)
            recs.append(rec)
        write_segment_outputs(run, recs)
        write_json_atomic(run / "report.json", {"mode": "segments", "session_id": sid, "pipeline_version": "0.1.0"})
        runs[sid] = run
    return runs


def _line(request_id, seconds, start, response=None, error=None, status=200):
    return {"request_id": request_id, "kind": "segment", "http_status": status, "seconds": seconds,
            "started_utc": start, "attempts": 1, "response": response, "error": error}


def write_answers(batch: Path) -> list[dict]:
    """responses.jsonl as a runner would leave it after problems; returns the successful lines written."""
    lines, ok = [], []
    minute = 0

    def stamp():
        nonlocal minute
        minute += 1
        return f"2026-10-28T10:{minute:02d}:00Z"

    order = [("SESS_A", 0), ("SESS_A", 1), ("SESS_A", 2), ("SESS_A", 5)]
    lines.append(_line(rid("SESS_A", 7), 3.0, stamp(), {"detail": "CUDA out of memory"}, "HTTP 500", 500))
    for i, key in enumerate(order + [("SESS_A", 7), ("SESS_A", 8), ("SESS_A", 11)] +
                            [("SESS_B", k) for k in range(6)]):
        resp = {"stage1_label": "Distress", "stage1_confidence": 0.9, "distress_types": ANSWERS[key][0],
                "severity": "Medium", "stage2_confidence": 0.8, "needs_expert_review": False, "description": "d"}
        if ANSWERS[key][0] == []:  # Stage 1 Normal, exactly as the VLM system's predict() builds it
            resp.update(stage1_label="Normal", severity="None", stage2_confidence=0.0, description="")
        ln = _line(rid(*key), 10.0 + i, stamp(), resp)
        lines.append(ln)
        ok.append(ln)
        if key == ("SESS_A", 1):
            lines.append("{not json")  # a corrupted line in the middle
    dup = dict(ok[0], seconds=11.5, started_utc=stamp())  # the same answer again: not a conflict
    lines.append(dup)
    ok.append(dup)
    lines.append(_line("NOT_IN_BATCH_seg00001", 9.0, stamp(), {"distress_types": ["Potholes"]}))
    lines.append(_line(rid("SESS_B", 6), 0.5, stamp(), None, "URLError: <urlopen error [WinError 10061] refused>", None))
    text = "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n"
    text += '{"request_id": "SESS_B_seg00006", "kind": "segment", "http_st'  # cut off by a power cut
    (batch / "responses.jsonl").write_text(text, encoding="utf-8", newline="\n")
    write_json_atomic(batch / "vlm_run.json", {
        "runs": [{"started_utc": "2026-10-28T10:00:30Z", "finished_utc": "2026-10-28T10:10:30Z", "sent": 8, "ok": 7,
                  "health": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version": "v2:9da1601cfec54f76"},
                  "runner_version": "2"},
                 {"started_utc": "2026-10-28T10:10:40Z", "finished_utc": "2026-10-28T10:18:40Z", "sent": 9, "ok": 7,
                  "health": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version": "v2:9da1601cfec54f76"},
                  "runner_version": "2"}],
        "health": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version": "v2:9da1601cfec54f76"}})
    return ok


def wilson_ref(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson interval as the roots of (k/n - p)^2 = z^2 p (1 - p) / n (the score test's acceptance region)."""
    ph = k / n
    roots = np.roots([1 + z * z / n, -(2 * ph + z * z / n), ph * ph])
    lo, hi = sorted(float(np.real(r)) for r in roots)
    return max(0.0, lo), min(1.0, hi)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def answered_batch(tmp_path):
    runs = build_runs(tmp_path)
    batch = tmp_path / "batch_seg_01"
    info = export_batch([runs["SESS_A"], runs["SESS_B"]], batch, Config())
    ok_lines = write_answers(batch)
    return runs, batch, info, ok_lines


# ---- export --------------------------------------------------------------------------------------

def test_segment_export_contents(answered_batch):
    runs, batch, info, _ = answered_batch
    t = truth()
    sent = [r for r in t if r["status"] in ("send", "audit")]
    assert info["mode"] == "segments" and info["n_requests"] == info["n_segments"] == len(sent) == 14
    assert (info["n_crops"], info["n_frames"]) == (0, 0) and info["sessions"] == ["SESS_A", "SESS_B"]
    assert info["records_by_status"] == {"send": 12, "audit": 2, "skip": 4, "not_covered": 2}
    assert info["requests_by_status"] == {"send": 12, "audit": 2}
    # Qwen smart_resize with the server's min_pixels 256 x 28 = 7168 (the VLM system's MIN_IMAGE_PIXELS default):
    # rounding gives 112 x 56 = 6272 px, below the minimum, so both sides scale by sqrt(7168 / (112 x 63)) and
    # round up to multiples of 28: 63 -> 84, 112 -> 140 (transformers' own smart_resize gives the same).
    assert info["model_input_sizes"] == {"112x63": [140, 84]}
    assert "resizes it itself" in info["image_rules"] and info["not_exported"] == [] and info["reencoded"] == []
    reqs = [json.loads(x) for x in (batch / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["request_id"] for r in reqs] == [rid(r["sid"], r["k"]) for r in sent]  # route order per session
    for r in reqs:
        src = runs[r["session_id"]] / "frames" / f"{r['request_id']}.png"
        img = batch / r["image"]
        assert r["kind"] == "segment" and r["record_id"] == r["request_id"] and r["model_input"] == [140, 84]
        assert img.read_bytes() == src.read_bytes() and r["image_sha256"] == sha256_file(src)
    # Every record of every status travels with the batch, exactly as extracted.
    copied = sorted(p.name for p in (batch / "records").glob("*.json"))
    assert copied == sorted(f"{rid(r['sid'], r['k'])}.json" for r in t)
    for sid, run in runs.items():
        for p in (run / "sidecars").glob("*.json"):
            assert (batch / "records" / p.name).read_bytes() == p.read_bytes()
        assert (batch / "runs" / sid / "report.json").read_bytes() == (run / "report.json").read_bytes()
    assert verify_sums(batch) == []
    running = (batch / "RUNNING.txt").read_text(encoding="utf-8")
    assert "14 road-segment frames from 2 session(s)" in running and "112x63 -> 140x84" in running


def test_segment_export_refusals_leave_no_batch(tmp_path):
    runs = build_runs(tmp_path)
    crop_run = _fake_crop_run(tmp_path)
    with pytest.raises(ValueError, match="either crop runs or segment runs"):
        export_batch([runs["SESS_A"], crop_run], tmp_path / "b1", Config())
    with pytest.raises(ValueError, match="can send only statuses"):
        export_batch([runs["SESS_A"]], tmp_path / "b2", Config(), statuses=("send", "skip"))
    assert not (tmp_path / "b2").exists()
    with pytest.raises(ValueError, match="two of the runs"):
        export_batch([runs["SESS_A"], runs["SESS_A"]], tmp_path / "b3", Config())
    assert not (tmp_path / "b3").exists()
    # A frame changed after processing: refused, and the half-written batch is removed.
    frame = runs["SESS_B"] / "frames" / f"{rid('SESS_B', 2)}.png"
    data = bytearray(frame.read_bytes())
    data[-30] ^= 0xFF
    frame.write_bytes(bytes(data))
    (tmp_path / "b4").mkdir()
    with pytest.raises(ValueError, match="does not match frame.sha256"):
        export_batch([runs["SESS_A"], runs["SESS_B"]], tmp_path / "b4", Config())
    assert (tmp_path / "b4").exists() and not any((tmp_path / "b4").iterdir())


def test_oversize_frames_reencoded_losslessly_or_listed(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "SESS_C"
    recs = []
    gradient = np.dstack([np.tile(np.arange(112, dtype=np.uint8), (63, 1))] * 3)  # compresses better at level 9
    for k, status in ((0, "send"), (1, "send"), (2, "skip")):
        rec = make_record("SESS_C", k, status=status)
        if status == "send":
            write_frame(run, rec, gradient if k == 0 else None)
        recs.append(rec)
    write_segment_outputs(run, recs)
    write_json_atomic(run / "report.json", {"mode": "segments"})
    f0, f1 = (run / recs[k]["frame"]["file"] for k in (0, 1))
    with Image.open(f0) as im:
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=9)
    level9 = buf.tell()
    assert level9 < f0.stat().st_size < f1.stat().st_size
    monkeypatch.setattr(vlm, "SERVER_MAX_UPLOAD_BYTES", f0.stat().st_size - 1)
    info = export_batch([run], tmp_path / "b", Config())
    assert info["reencoded"] == [rid("SESS_C", 0)] and [d["id"] for d in info["not_exported"]] == [rid("SESS_C", 1)]
    assert info["n_requests"] == 1
    img = tmp_path / "b" / "images" / f"{rid('SESS_C', 0)}.png"
    with Image.open(img) as a, Image.open(f0) as b:
        assert np.array_equal(np.asarray(a), np.asarray(b))  # same pixels
    emb, gps = read_png_record(img)
    assert emb == read_png_record(f0)[0] and gps == read_png_record(f0)[1]  # metadata kept
    req = json.loads((tmp_path / "b" / "requests.jsonl").read_text(encoding="utf-8"))
    assert req["image_sha256"] == sha256_file(img)
    data = load_batch(tmp_path / "b", Config())
    row = data.table.set_index("id").loc[rid("SESS_C", 1)]
    assert row["vlm_class"] == "not_answered" and row["vlm_missing"].startswith("not exported")


# ---- crop mode unchanged -------------------------------------------------------------------------

def _fake_crop_run(root: Path) -> Path:
    run = root / "croprun" / "S"
    for sub in ("crops", "frames", "sidecars"):
        (run / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for k, fg in enumerate((10, 20)):
        r, fid = f"S_{fg:07d}_00", f"S_{fg:07d}_frame"
        rec = {"id": r, "status": "sent", "session_id": "S", "frame_global": fg, "frame_request": fid,
               "utc_ms": 1.79e12 + k, "camera": {"lat": 12.97, "lon": 77.59}, "anomaly": {"lat": 12.97, "lon": 77.59},
               "evidence": 10.0, "kinds": ["dino"], "crop": {"file": f"{r}.png"}}
        write_png(run / "crops" / f"{r}.png", (rng.random((56, 56, 3)) * 255).astype(np.uint8), rec)
        write_png(run / "frames" / f"{fid}.png", (rng.random((56, 112, 3)) * 255).astype(np.uint8), None)
        write_sidecar(run / "sidecars" / f"{r}.json", rec)
    return run


def test_crop_export_and_import_unchanged(tmp_path):
    run = _fake_crop_run(tmp_path)
    batch = tmp_path / "crop_batch"
    info = export_batch([run], batch, Config())
    assert set(info) == {"schema", "batch_id", "created_utc", "pipeline_version", "sessions", "n_requests", "n_crops",
                         "n_frames", "expected_model_id", "expected_prompt_version", "image_rules"}
    assert (info["n_crops"], info["n_frames"], info["sessions"]) == (2, 2, ["S"])
    assert sorted(p.relative_to(batch).as_posix() for p in batch.rglob("*") if p.is_file()) == sorted(
        ["RUNNING.txt", "SHA256SUMS", "batch.json", "requests.jsonl", "run_vlm_batch.py", "images/S_0000010_00.png",
         "images/S_0000010_frame.png", "images/S_0000020_00.png", "images/S_0000020_frame.png"])
    reqs = [json.loads(x) for x in (batch / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["request_id"] for r in reqs] == ["S_0000010_00", "S_0000010_frame", "S_0000020_00", "S_0000020_frame"]
    assert all(set(r) == {"request_id", "kind", "image", "image_sha256", "width", "height", "session_id",
                          "record_id", "frame_request_id"} for r in reqs)
    assert (batch / "RUNNING.txt").read_text(encoding="utf-8").startswith(
        "GeoAI VLM batch crop_batch - 4 images (2 crops, 2 full frames)")
    with open(batch / "responses.jsonl", "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(_line(r["request_id"], 12.0, "2026-10-28T10:00:00Z",
                                     {"distress_types": ["Potholes"]})) + "\n")
    s = import_batch(batch, [run], Config())
    assert (s["requests"], s["answered"], s["valid"], s["missing"]) == (4, 4, 4, 0)
    rows = read_csv(run / "vlm_results_crop_batch.csv")
    assert [r["request_id"] for r in rows] == ["S_0000010_00", "S_0000020_00"]
    assert all(r["crop_frame_agree"] == "1" and r["labels"] == "Potholes" for r in rows)
    assert not (batch / "vlm_results.csv").exists()  # crop batches write per run only
    with pytest.raises(ValueError, match="not a segment-mode batch"):
        analyze(batch)


# ---- import --------------------------------------------------------------------------------------

def test_import_from_the_batch_alone_then_into_the_runs(answered_batch):
    runs, batch, _, _ = answered_batch
    s = import_batch(batch, [], Config())
    assert (s["mode"], s["requests"], s["answered"], s["missing"], s["valid"]) == ("segments", 14, 13, 1, len(VALID))
    assert s["problems"] == {"unknown_request": 1, "duplicate_conflict": 0, "bad_json": 1}
    assert s["classes"] == {"distress": 8, "normal": 4, "unknown": 1} and s["errors_unresolved"] == 1
    assert (s["model_matches"], s["prompt_matches"], s["consistent"]) == (True, True, True)
    rows = {r["request_id"]: r for r in read_csv(batch / "vlm_results.csv")}
    assert len(rows) == 14
    a1 = rows[rid("SESS_A", 1)]
    assert a1["labels"] == "Alligator Cracking+Longitudinal Cracking" and a1["vlm_class"] == "distress"
    assert a1["valid"] == "1" and a1["answered"] == "1" and a1["seconds"] == "11.000"
    a5 = rows[rid("SESS_A", 5)]
    assert a5["valid"] == "0" and "N/A" in a5["problem"] and a5["distress_labels"] == "Potholes"
    b6 = rows[rid("SESS_B", 6)]
    assert b6["answered"] == "0" and b6["error"].startswith("URLError") and b6["vlm_class"] == ""
    assert rows[rid("SESS_A", 7)]["vlm_class"] == "distress"  # the HTTP 500 line was superseded
    gj = json.loads((batch / "vlm_results.geojson").read_text(encoding="utf-8"))
    assert len(gj["features"]) == 14 and gj["features"][0]["geometry"]["type"] == "Polygon"
    assert not any("vlm" in r for r in load_segment_records(runs["SESS_A"] / "sidecars"))  # runs untouched

    s2 = import_batch(batch, list(runs.values()), Config())
    assert s2["runs_updated"] == {"SESS_A": 7, "SESS_B": 6} and s2["run_problems"] == []
    for sid, run in runs.items():
        recs = load_segment_records(run / "sidecars")
        assert all(validate_segment_record(r) == [] for r in recs)
        for r in recs:
            key = (sid, r["segment"]["k"])
            assert ("vlm" in r) == (key in ANSWERS)
            if key in ANSWERS:
                assert r["vlm"]["batch_id"] == "batch_seg_01" and r["vlm"]["valid"] == (key in VALID)
        table = {r["id"]: r for r in read_csv(run / "segments.csv")}
        assert len(table) == len(recs)
        assert (run / f"vlm_results_batch_seg_01.csv").exists()
    assert table[rid("SESS_B", 0)]["vlm_labels"] == "Normal+Potholes" and table[rid("SESS_B", 0)]["vlm_valid"] == "0"
    # A new batch from an imported run carries the records as extracted, without the earlier answers.
    export_batch([runs["SESS_A"]], batch.parent / "batch_seg_02", Config())
    assert not any("vlm" in r for r in load_segment_records(batch.parent / "batch_seg_02" / "records"))


# ---- analysis ------------------------------------------------------------------------------------

def _expected_summary(rows: list[dict], scope: str) -> dict[str, dict]:
    sel = [r for r in rows if r["covered"] and scope in ("all", r["sid"])]
    surveyed = sum(r["length"] for r in sel)
    classified = sum(r["length"] for r in sel if r["cls"] in ("distress", "normal"))
    out = {}
    for lab in DISTRESS_LABELS:
        hit = [r for r in sel if lab in r["labels"]]
        out[lab] = (len(hit), sum(r["length"] for r in hit), surveyed, classified, True)
    for cls, name in (("distress", "Any distress"), ("normal", "Normal"), ("unknown", "No valid label"),
                      ("not_answered", "Not answered"), ("not_sent", "Not sent (gate skip)")):
        hit = [r for r in sel if r["cls"] == cls]
        out[name] = (len(hit), sum(r["length"] for r in hit), surveyed, classified, cls in ("distress", "normal"))
    return out


def test_analyze_numbers_match_an_independent_calculation(answered_batch):
    runs, batch, _, ok_lines = answered_batch
    result = analyze(batch)
    t = truth()
    out = batch / "analysis"
    assert sorted(p.name for p in out.iterdir()) == ["REPORT.md", "gate_eval.json", "label_summary.csv", "per_km.csv",
                                                      "segments_results.csv", "segments_results.geojson", "vlm_timing.json"]
    assert result["surveyed_m"] == 88.0 and result["classified_m"] == 60.0 and result["distress_m"] == 40.0

    # label_summary.csv: every row.
    summary = read_csv(out / "label_summary.csv")
    assert len(summary) == 3 * (len(DISTRESS_LABELS) + 5)
    for scope in ("all", "SESS_A", "SESS_B"):
        exp = _expected_summary(t, scope)
        got = [r for r in summary if r["scope"] == scope]
        assert [r["label"] for r in got] == list(exp)
        for r in got:
            n, length, surveyed, classified, of_cls = exp[r["label"]]
            assert int(r["segments"]) == n and float(r["length_m"]) == pytest.approx(length, abs=1e-3)
            assert float(r["surveyed_m"]) == pytest.approx(surveyed, abs=1e-3)
            assert float(r["classified_m"]) == pytest.approx(classified, abs=1e-3)
            assert float(r["pct_of_surveyed"]) == pytest.approx(100 * length / surveyed, abs=1e-4)
            if of_cls:
                assert float(r["pct_of_classified"]) == pytest.approx(100 * length / classified, abs=1e-4)
            else:
                assert r["pct_of_classified"] == ""
        cats = [r for r in got if r["kind"] == "category"]
        assert sum(int(r["segments"]) for r in cats) == sum(1 for r in t if r["covered"] and scope in ("all", r["sid"]))
        assert sum(float(r["pct_of_surveyed"]) for r in cats) == pytest.approx(100.0, abs=1e-3)
    a_pot = next(r for r in summary if r["scope"] == "SESS_A" and r["label"] == "Potholes")
    assert (a_pot["segments"], a_pot["length_m"], a_pot["pct_of_surveyed"]) == ("2", "10.000", f"{1000 / 53:.4f}")

    # per_km.csv: one stretch per session here (58 m and 40 m of road).
    km = {r["session_id"]: r for r in read_csv(out / "per_km.csv")}
    for sid in ("SESS_A", "SESS_B"):
        rows = [r for r in t if r["sid"] == sid]
        r = km[sid]
        base = min(x["s0"] for x in rows)
        assert (r["km"], float(r["from_m"])) == ("0", 0.0)
        assert float(r["to_m"]) == pytest.approx(max(x["s1"] for x in rows) - base)
        sums = {"surveyed_m": sum(x["length"] for x in rows if x["covered"]),
                "not_covered_m": sum(x["length"] for x in rows if not x["covered"]),
                "sent_m": sum(x["length"] for x in rows if x["status"] in ("send", "audit")),
                "classified_m": sum(x["length"] for x in rows if x["cls"] in ("distress", "normal"))}
        for cls in ("distress", "normal", "unknown", "not_answered", "not_sent"):
            sums[f"{cls}_m"] = sum(x["length"] for x in rows if x["cls"] == cls)
        for lab in DISTRESS_LABELS:
            sums[lab.lower().replace(" ", "_") + "_m"] = sum(x["length"] for x in rows if lab in x["labels"])
        for key, v in sums.items():
            assert float(r[key]) == pytest.approx(v, abs=1e-3), key
        assert float(r["pct_distress_of_surveyed"]) == pytest.approx(100 * sums["distress_m"] / sums["surveyed_m"], abs=1e-4)
        assert float(r["pct_distress_of_classified"]) == pytest.approx(100 * sums["distress_m"] / sums["classified_m"], abs=1e-4)
        # Chainage: 12 km at path distance 50 m, increasing (test_segment_records.META).
        assert float(r["chainage_from_km"]) == pytest.approx(12.0 + (base - 50.0) / 1000, abs=1e-6)
        assert float(r["chainage_to_km"]) == pytest.approx(12.0 + (max(x["s1"] for x in rows) - 50.0) / 1000, abs=1e-6)

    # gate_eval.json.
    assert Z95 == pytest.approx(norm.ppf(0.975), abs=1e-12) and Z_BONF == pytest.approx(norm.ppf(1 - 0.025 / 2), abs=1e-12)
    ge = json.loads((out / "gate_eval.json").read_text(encoding="utf-8"))

    def check(p, k, n, z=Z95):
        assert (p["k"], p["n"]) == (k, n) and p["estimate"] == pytest.approx(k / n)
        assert p["ci95"] == pytest.approx(list(wilson_ref(k, n, z)), abs=1e-9)

    known = ("distress", "normal")
    b = [r for r in t if r["sid"] == "SESS_B" and r["covered"]]
    bk = [r for r in b if r["cls"] in known]
    tp = sum(1 for r in bk if r["gate_positive"] and r["cls"] == "distress")
    fp = sum(1 for r in bk if r["gate_positive"] and r["cls"] == "normal")
    fn = sum(1 for r in bk if not r["gate_positive"] and r["cls"] == "distress")
    tn = sum(1 for r in bk if not r["gate_positive"] and r["cls"] == "normal")
    sa = ge["send_all"]
    assert (tp, fp, fn, tn) == (2, 1, 2, 1) and sa["confusion"] == {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    assert (sa["segments"], sa["classified"], sa["not_classified"]) == (7, 6, 1)
    check(sa["recall"], tp, tp + fn)
    check(sa["precision"], tp, tp + fp)
    check(sa["pass_rate"], sum(1 for r in b if r["gate_positive"]), len(b))
    check(sa["vlm_distress_rate"], tp + fn, len(bk))

    a = [r for r in t if r["sid"] == "SESS_A" and r["covered"]]
    pos = [r for r in a if r["gate_positive"]]
    neg = [r for r in a if not r["gate_positive"]]
    pk = [r for r in pos if r["cls"] in known]
    ak = [r for r in neg if r["status"] == "audit" and r["cls"] in known]
    kp, km_ = sum(r["cls"] == "distress" for r in pk), sum(r["cls"] == "distress" for r in ak)
    g = ge["gated"]
    assert (g["segments"], g["gate_positive"], g["gate_negative"], g["audited"], g["skipped"]) == (11, 5, 6, 2, 4)
    assert (kp, len(pk), km_, len(ak)) == (3, 4, 1, 2)
    check(g["precision"], kp, len(pk))
    check(g["miss_rate_among_rejected"], km_, len(ak))
    check(g["pass_rate"], len(pos), len(a))
    P, N = len(pos), len(neg)
    rec = P * kp / len(pk) / (P * kp / len(pk) + N * km_ / len(ak))
    p_lo, p_hi = wilson_ref(kp, len(pk), Z_BONF)
    m_lo, m_hi = wilson_ref(km_, len(ak), Z_BONF)
    assert g["recall"]["estimate"] == pytest.approx(rec) and rec == pytest.approx(3.75 / 6.75)
    assert g["recall"]["ci95"] == pytest.approx([P * p_lo / (P * p_lo + N * m_hi), P * p_hi / (P * p_hi + N * m_lo)])
    skip_m = sum(r["length"] for r in neg if r["status"] == "skip")
    miss_lo, miss_hi = wilson_ref(km_, len(ak), Z95)
    assert g["estimated_missed_distress_in_skipped"]["length_m"]["estimate"] == pytest.approx(skip_m * km_ / len(ak))
    assert g["estimated_missed_distress_in_skipped"]["length_m"]["ci95"] == pytest.approx([skip_m * miss_lo, skip_m * miss_hi])
    assert set(ge["per_session"]) == {"SESS_A", "SESS_B"}
    assert ge["per_session"]["SESS_A"]["gate"] == g and ge["per_session"]["SESS_B"]["all"] == sa

    # vlm_timing.json.
    tm = json.loads((out / "vlm_timing.json").read_text(encoding="utf-8"))
    secs = [ln["seconds"] for ln in ok_lines]
    assert (tm["requests"], tm["answered"], tm["remaining"], tm["error_lines"]) == (14, 13, 1, 2)
    assert tm["errors_by_type"] == {"HTTP 500": 1, "URLError": 1}
    sp = tm["seconds_per_answer"]
    assert sp["n"] == len(secs) == 14 and sp["median"] == pytest.approx(statistics.median(secs))
    assert sp["mean"] == pytest.approx(statistics.fmean(secs)) and sp["max"] == max(secs)
    assert sp["p10"] == pytest.approx(statistics.quantiles(secs, n=10, method="inclusive")[0])
    assert tm["runner_busy_h"] == pytest.approx((600 + 480) / 3600)
    assert tm["answers_per_hour"] == pytest.approx(13 / ((600 + 480) / 3600))
    assert tm["estimated_hours_remaining"] == pytest.approx(statistics.median(secs) / 3600)
    assert tm["first_start_utc"] == "2026-10-28T10:01:00Z"

    # segments_results: one row and one footprint per segment, joined to the answers.
    res = {r["id"]: r for r in read_csv(out / "segments_results.csv")}
    assert len(res) == len(t)
    for r in t:
        row = res[rid(r["sid"], r["k"])]
        assert row["vlm_class"] == r["cls"] and row["vlm_distress_labels"] == "+".join(r["labels"])
        assert row["frame_global"] in ("", str(30 * r["k"] + 7))  # integers stay integers
    assert res[rid("SESS_B", 6)]["vlm_missing"].startswith("error: URLError")
    assert res[rid("SESS_A", 3)]["gate_positive"] == "0" and res[rid("SESS_A", 10)]["gate_positive"] == ""
    gj = json.loads((out / "segments_results.geojson").read_text(encoding="utf-8"))
    assert len(gj["features"]) == len(t)

    # REPORT.md: the methods facts and the limits.
    rep = (out / "REPORT.md").read_text(encoding="utf-8")
    sums_sha = hashlib.sha256((batch / "SHA256SUMS").read_bytes()).hexdigest()
    for text in ("Partial batch: 13 of 14 requests answered", Config().vlm.model_id, "v2:9da1601cfec54f76", sums_sha,
                 "2.5 mm/px at 1.5 m ahead", "51.7 %", "| Potholes | 3 | 15.0 |", "Labels not found:",
                 "model matches: True"):
        assert text in rep, text

    # Deterministic, and the batch folder alone is enough.
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    elsewhere = batch.parent / "moved" / batch.name
    shutil.copytree(batch, elsewhere, ignore=shutil.ignore_patterns("analysis"))
    shutil.rmtree(batch.parent / "runs")
    analyze(elsewhere)
    second = {p.name: p.read_bytes() for p in (elsewhere / "analysis").iterdir()}
    assert second == first


def test_per_km_splits_segments_across_stretches(answered_batch):
    _, batch, _, _ = answered_batch
    data = load_batch(batch, Config())
    km = per_km(data.table, LABELS, bin_m=12.0)
    a = [r for r in truth() if r["sid"] == "SESS_A"]
    base, end = 100.0, 158.0
    edges = [(i * 12.0, min((i + 1) * 12.0, end - base)) for i in range(5)]
    got = km[km["session_id"] == "SESS_A"].reset_index(drop=True)
    assert len(got) == 5
    for i, (lo, hi) in enumerate(edges):
        def overlap(rows):
            return sum(max(0.0, min(hi, r["s1"] - base) - max(lo, r["s0"] - base)) for r in rows)
        assert (got.loc[i, "from_m"], got.loc[i, "to_m"]) == (lo, hi)
        assert got.loc[i, "surveyed_m"] == pytest.approx(overlap([r for r in a if r["covered"]]))
        assert got.loc[i, "distress_m"] == pytest.approx(overlap([r for r in a if r["cls"] == "distress"]))
        assert got.loc[i, "potholes_m"] == pytest.approx(overlap([r for r in a if "Potholes" in r["labels"]]))
        assert got.loc[i, "not_covered_m"] == pytest.approx(overlap([r for r in a if not r["covered"]]))
    assert got["surveyed_m"].sum() == pytest.approx(53.0) and got["not_covered_m"].sum() == pytest.approx(5.0)


def test_analyze_before_any_answer(tmp_path):
    runs = build_runs(tmp_path)
    batch = tmp_path / "b"
    export_batch(list(runs.values()), batch, Config())
    s = analyze(batch)
    assert (s["answered"], s["classified_m"], s["distress_m"]) == (0, 0.0, 0.0)
    ge = json.loads((batch / "analysis" / "gate_eval.json").read_text(encoding="utf-8"))
    assert ge["gated"]["recall"]["estimate"] is None and ge["gated"]["estimated_missed_distress_in_skipped"] is None
    assert ge["send_all"]["recall"] == {"k": 0, "n": 0, "estimate": None, "ci95": None}
    tm = json.loads((batch / "analysis" / "vlm_timing.json").read_text(encoding="utf-8"))
    assert tm["remaining"] == 14 and tm["estimated_hours_remaining"] == pytest.approx(14 * 14.0 / 3600)
    rep = (batch / "analysis" / "REPORT.md").read_text(encoding="utf-8")
    assert "Partial batch: 0 of 14" in rep and "the gate's misses are unknown" in rep


def test_import_and_analyze_need_only_numpy_and_pandas(answered_batch):
    """vlm-import and analyze on the VLM machine: torch, OpenCV, Pillow, SciPy etc. are not installed there."""
    _, batch, _, _ = answered_batch
    script = (
        "import importlib.abc, sys\n"
        "BLOCK = {'torch', 'cv2', 'PIL', 'scipy', 'shapely', 'pymap3d', 'av', 'skimage'}\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from pathlib import Path\n"
        "from geoai.config import Config\n"
        "from geoai.vlm import import_batch\n"
        "from geoai.analyze import analyze\n"
        f"s = import_batch(Path({str(batch)!r}), [], Config())\n"
        f"a = analyze(Path({str(batch)!r}))\n"
        "print(s['answered'], a['distress_m'])\n")
    r = subprocess.run([sys.executable, "-c", script], cwd=PIPELINE, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["13", "40.0"]


# ---- the runner against a fake server --------------------------------------------------------------

class _Server(BaseHTTPRequestHandler):
    """The VLM team's API as the runner sees it, checking every upload byte for byte."""
    images: dict[str, bytes] = {}
    calls: list[str] = []
    fail_first: dict[str, int] = {}       # file name -> HTTP status of its first attempt
    always: int | None = None             # status for every /classify call (None = answer)
    labels = ["Potholes"]

    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version": "v2:9da1601cfec54f76"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        boundary = self.headers["Content-Type"].split("boundary=", 1)[1].encode()
        name = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
        start = body.index(b"\r\n\r\n") + 4
        assert body.endswith(b"\r\n--" + boundary + b"--\r\n")
        payload = body[start:len(body) - len(boundary) - 8]
        assert payload == self.images[name], name  # the streamed upload is the exact file
        _Server.calls.append(name)
        if self.always is not None:
            return self._send(self.always, {"detail": "failing"})
        first = self.fail_first.pop(name, None)
        if first == 429:
            return self._send(429, {"detail": "rate limited"}, {"Retry-After": "0"})
        if first is not None:
            return self._send(first, {"detail": "busy"})
        self._send(200, {"stage1_label": "Distress", "distress_types": self.labels, "severity": "Low",
                         "stage1_confidence": 0.9, "stage2_confidence": 0.85, "needs_expert_review": False,
                         "description": "fake", "timings": {"stage1_ms": 1300.0, "stage2_ms": 8200.0}})


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _Server.calls, _Server.fail_first, _Server.always, _Server.images = [], {}, None, {}
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _runner(batch: Path, url: str, *extra):
    return subprocess.run([sys.executable, str(batch / "run_vlm_batch.py"), str(batch), "--url", url,
                           "--min-interval", "0", "--retry-wait", "0", *extra], capture_output=True, text=True,
                          timeout=120)


def _segment_batch(tmp_path: Path) -> Path:
    runs = build_runs(tmp_path)
    batch = tmp_path / "batch_rt"
    export_batch(list(runs.values()), batch, Config())
    _Server.images = {p.name: p.read_bytes() for p in (batch / "images").glob("*.png")}
    return batch


def test_runner_streams_retries_and_resumes(tmp_path, server):
    batch = _segment_batch(tmp_path)
    _Server.fail_first = {f"{rid('SESS_A', 0)}.png": 503, f"{rid('SESS_A', 1)}.png": 429}
    r1 = _runner(batch, server, "--limit", "3")
    assert r1.returncode == 1, r1.stdout + r1.stderr
    assert "left ~" in r1.stdout and "retry in 0 s" in r1.stdout
    r2 = _runner(batch, server)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    lines = [json.loads(x) for x in (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 14 and len({ln["request_id"] for ln in lines}) == 14 and all(ln["error"] is None for ln in lines)
    att = {ln["request_id"]: ln["attempts"] for ln in lines}
    assert att[rid("SESS_A", 0)] == 2 and att[rid("SESS_A", 1)] == 2 and att[rid("SESS_A", 2)] == 1
    assert sorted(set(_Server.calls)) == sorted(_Server.images) and len(_Server.calls) == 16
    log = json.loads((batch / "vlm_run.json").read_text(encoding="utf-8"))
    assert len(log["runs"]) == 2 and [x["ok"] for x in log["runs"]] == [3, 11] and log["runs"][1]["runner_version"] == "3"
    s = import_batch(batch, [], Config())
    assert (s["answered"], s["missing"], s["valid"]) == (14, 0, 14)
    a = analyze(batch)
    assert a["distress_m"] == 88.0 - 20.0  # every sent segment Potholes; only the skipped 20 m are not
    tm = json.loads((batch / "analysis" / "vlm_timing.json").read_text(encoding="utf-8"))
    assert tm["server_timings_median"] == {"stage1_ms": 1300.0, "stage2_ms": 8200.0}


def test_runner_repairs_a_cut_off_line(tmp_path, server):
    batch = _segment_batch(tmp_path)
    first = json.dumps(_line(rid("SESS_A", 0), 12.0, "2026-10-28T10:00:00Z", {"distress_types": ["Normal"]}))
    (batch / "responses.jsonl").write_text(first + "\n" + '{"request_id": "SESS_A_seg00001", "ki', encoding="utf-8")
    r = _runner(batch, server, "--limit", "1")
    assert "removed a cut-off last line" in r.stdout
    lines = (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["request_id"] for x in lines] == [rid("SESS_A", 0), rid("SESS_A", 1)]
    # A complete line that lost only its newline keeps its answer.
    text = (batch / "responses.jsonl").read_text(encoding="utf-8").rstrip("\n")
    (batch / "responses.jsonl").write_text(text, encoding="utf-8")
    r = _runner(batch, server, "--limit", "1")
    assert "added the missing newline" in r.stdout
    ids = [json.loads(x)["request_id"] for x in (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ids == [rid("SESS_A", 0), rid("SESS_A", 1), rid("SESS_A", 2)]


def test_runner_ctrl_c_saves_what_was_answered(tmp_path, server):
    batch = _segment_batch(tmp_path)
    spec = importlib.util.spec_from_file_location("runner_under_test", batch / "run_vlm_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    real_post, calls = runner.post_file, []

    def post_then_interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    runner.post_file = post_then_interrupt
    code = runner.main([str(batch), "--url", server, "--min-interval", "0"])
    assert code == 130
    lines = (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["error"] is None
    log = json.loads((batch / "vlm_run.json").read_text(encoding="utf-8"))
    assert log["runs"][0]["interrupted"] is True and log["runs"][0]["ok"] == 1
    assert _runner(batch, server).returncode == 0
    assert len((batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 14


def test_runner_stops_on_auth_errors_and_a_failing_server(tmp_path, server):
    batch = _segment_batch(tmp_path)
    _Server.always = 401
    r = _runner(batch, server)
    assert r.returncode == 2 and "--api-key" in r.stderr
    assert not (batch / "responses.jsonl").read_text(encoding="utf-8").strip()
    _Server.always = 500
    r = _runner(batch, server, "--retries", "1", "--max-failures", "2")
    assert r.returncode == 1 and "2 images in a row failed" in r.stderr
    lines = [json.loads(x) for x in (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [ln["error"] for ln in lines] == ["HTTP 500", "HTTP 500"] and [ln["attempts"] for ln in lines] == [2, 2]
    r = subprocess.run([sys.executable, str(batch / "run_vlm_batch.py"), str(batch), "--url", "http://127.0.0.1:9"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "Is the classifier server running?" in r.stderr


def test_multipart_body_streams_exact_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location("runner_mp", PIPELINE / "tools" / "run_vlm_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    p = tmp_path / "x.png"
    data = np.random.default_rng(3).integers(0, 256, 3 * runner.CHUNK + 17, dtype=np.uint8).tobytes()
    p.write_bytes(data)
    body = runner.MultipartFile(p)
    chunks = []
    for n in (5, 100, runner.CHUNK, 8192, 7):
        chunks.append(body.read(n))
    while part := body.read(8192):
        chunks.append(part)
    body.close()
    sent = b"".join(chunks)
    assert len(sent) == body.length and body.complete and body.hexdigest() == hashlib.sha256(data).hexdigest()
    boundary = body.content_type.split("boundary=")[1]
    assert sent.startswith(f"--{boundary}\r\n".encode()) and sent.endswith(f"\r\n--{boundary}--\r\n".encode())
    assert data in sent and len(sent) - len(data) == len(sent.replace(data, b""))
    p.write_bytes(data)
    short = runner.MultipartFile(p)
    p.write_bytes(data[:100])  # the file shrinks after the length was announced
    with pytest.raises(OSError, match="shorter"):
        while short.read(8192):
            pass
    short.close()
