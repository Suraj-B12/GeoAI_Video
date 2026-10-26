"""Offline hand-off to the Qwen2.5-VL classifier on the VLM machine (docs/pipeline/09).

Folder batch (pilot default, no network):
  geoai vlm-export RUN_DIR... BATCH_DIR   -> requests.jsonl + images/ + SHA256SUMS + runner script
  (USB drive) -> on the VLM machine: python run_vlm_batch.py BATCH_DIR  (calls their existing
                 POST /classify on 127.0.0.1:8000; standard library only; resumable)
  (USB drive) -> geoai vlm-import BATCH_DIR RUN_DIR...  -> validated vlm_results.csv / .geojson

Facts used from the VLM team's answers (vlm_handoff/vlm-information-response.md):
- /classify takes multipart field "file" and returns JSON with stage1_label, stage1_confidence,
  distress_types, severity, description, stage2_confidence, needs_expert_review and raw outputs.
- Valid labels are the 18 IRC:82-2015 names plus "Normal"; anything else (e.g. "N/A",
  "Unknown") means "no valid label" and goes to review.
- 10 requests per minute per client; one image at a time (~11-18 s each).
- For week 1 they asked for the full frame of every crop as its own request.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from . import __version__
from .config import Config
from .util import sha256_file, to_json, write_json_atomic

RUNNER = Path(__file__).resolve().parent.parent / "tools" / "run_vlm_batch.py"
BATCH_SCHEMA = "geoai.vlm_batch.v1"


def export_batch(run_dirs: list[Path], batch_dir: Path, cfg: Config, *, include_frames: bool | None = None,
                 statuses: tuple[str, ...] = ("sent",)) -> dict:
    batch_dir = Path(batch_dir)
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"{batch_dir} is not empty")
    (batch_dir / "images").mkdir(parents=True, exist_ok=True)
    include_frames = cfg.crop.full_frame if include_frames is None else include_frames
    requests = []
    seen_frames: set[str] = set()
    sessions = []
    for run in run_dirs:
        run = Path(run)
        sessions.append(run.name)
        for side in sorted((run / "sidecars").glob("*.json")):
            rec = json.loads(side.read_text(encoding="utf-8"))
            if rec.get("status") not in statuses:
                continue
            src = run / "crops" / rec["crop"]["file"]
            requests.append(_request(src, batch_dir, rec["id"], "crop", rec, rec.get("frame_request")))
            fid = rec.get("frame_request")
            if include_frames and fid and fid not in seen_frames:
                fsrc = run / "frames" / f"{fid}.png"
                if fsrc.exists():
                    requests.append(_request(fsrc, batch_dir, fid, "frame", rec, None))
                    seen_frames.add(fid)
    with open(batch_dir / "requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    batch = {
        "schema": BATCH_SCHEMA, "batch_id": batch_dir.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_version": __version__, "sessions": sessions, "n_requests": len(requests),
        "n_crops": sum(1 for r in requests if r["kind"] == "crop"), "n_frames": sum(1 for r in requests if r["kind"] == "frame"),
        "expected_model_id": cfg.vlm.model_id, "expected_prompt_version": cfg.vlm.prompt_version,
        "image_rules": "PNG, sRGB, upright; both sides multiples of 28, at most 1,048,576 px: the model receives these pixels unchanged",
    }
    write_json_atomic(batch_dir / "batch.json", batch)
    shutil.copyfile(RUNNER, batch_dir / "run_vlm_batch.py")
    (batch_dir / "RUNNING.txt").write_text(_running_txt(batch), encoding="utf-8", newline="\n")
    _write_sums(batch_dir)
    return batch


def _request(src: Path, batch_dir: Path, rid: str, kind: str, rec: dict, frame_request: str | None) -> dict:
    dst = batch_dir / "images" / f"{rid}.png"
    shutil.copyfile(src, dst)
    from PIL import Image
    with Image.open(dst) as im:
        w, h = im.size
    return {"request_id": rid, "kind": kind, "image": f"images/{rid}.png", "image_sha256": sha256_file(dst),
            "width": w, "height": h, "session_id": rec.get("session_id"), "record_id": rec["id"] if kind == "crop" else None,
            "frame_request_id": frame_request}


def _write_sums(batch_dir: Path) -> None:
    lines = []
    for p in sorted(batch_dir.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS", "responses.jsonl", "vlm_run.json"):
            lines.append(f"{sha256_file(p)}  {p.relative_to(batch_dir).as_posix()}")
    (batch_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_sums(batch_dir: Path) -> list[str]:
    """Files whose checksum no longer matches (empty list = intact)."""
    bad = []
    for line in (batch_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = batch_dir / rel
        if not p.exists() or sha256_file(p) != digest:
            bad.append(rel)
    return bad


def _running_txt(batch: dict) -> str:
    return f"""GeoAI VLM batch {batch['batch_id']} - {batch['n_requests']} images ({batch['n_crops']} crops, {batch['n_frames']} full frames)

On the VLM machine, with the classifier server running (POST /classify on port 8000):

    python run_vlm_batch.py .

The script needs only the Python standard library. It:
  1. checks SHA256SUMS (stops if any image was altered in transit),
  2. reads GET /health and records the model, prompt and settings in vlm_run.json,
  3. sends each image in requests.jsonl to /classify, one at a time (the server's rate limit
     is respected), and appends one line per image to responses.jsonl,
  4. can be stopped and restarted at any time: answered images are skipped.

Options: --url http://127.0.0.1:8000  --api-key KEY  --timeout 300
Expected model: {batch['expected_model_id']}
Expected prompt version: {batch['expected_prompt_version']}

Images are PNG and already sized for the model (sides multiples of 28, <= 1,048,576 px).
Please don't re-encode or resize them. Copy the whole folder back when done; please delete
the images from the VLM machine afterwards (the full frames may show vehicles or people).
"""


# ---- import ------------------------------------------------------------------------------------

RESULT_FIELDS = ["request_id", "kind", "record_id", "frame_request_id", "valid", "problem", "stage1_label",
                 "stage1_confidence", "labels", "severity", "stage2_confidence", "needs_expert_review",
                 "description", "frame_labels", "crop_frame_agree", "http_status", "seconds"]


def _labels(resp: dict) -> list[str]:
    v = resp.get("distress_types")
    if v is None:
        return []
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    out = []
    for x in v:
        x = str(x).strip()
        if x.lower().startswith("normal"):
            x = "Normal"
        out.append(x)
    return out


def import_batch(batch_dir: Path, run_dirs: list[Path], cfg: Config) -> dict:
    batch_dir = Path(batch_dir)
    bad = verify_sums(batch_dir)
    if bad:
        raise ValueError(f"batch files changed since export: {bad[:5]}")
    reqs = {}
    for line in (batch_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            reqs[r["request_id"]] = r
    run_info = {}
    if (batch_dir / "vlm_run.json").exists():
        run_info = json.loads((batch_dir / "vlm_run.json").read_text(encoding="utf-8"))
    allowed = set(cfg.vlm.labels)
    results: dict[str, dict] = {}
    problems = {"unknown_request": 0, "duplicate_conflict": 0, "bad_json": 0}
    resp_path = batch_dir / "responses.jsonl"
    lines = resp_path.read_text(encoding="utf-8").splitlines() if resp_path.exists() else []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # half-written last line after an interruption
            problems["bad_json"] += 1
            continue
        rid = r.get("request_id")
        if rid not in reqs:
            problems["unknown_request"] += 1
            continue
        if r.get("error"):
            continue  # not classified; will be re-sent
        res = _parse(r, reqs[rid], allowed)
        if rid in results and results[rid]["labels"] != res["labels"]:
            problems["duplicate_conflict"] += 1
            res["valid"] = False
            res["problem"] = "conflicting duplicate responses"
        results[rid] = res
    # Crop vs its full frame (the VLM team's week-1 comparison).
    for res in results.values():
        if res["kind"] == "crop" and res.get("frame_request_id") in results:
            fl = results[res["frame_request_id"]]["labels"]
            res["frame_labels"] = "+".join(fl)
            res["crop_frame_agree"] = int(set(fl) == set(res["labels"]))
    missing = [rid for rid in reqs if rid not in results]
    # Join to the extraction records and write outputs per run directory.
    records = {}
    for run in run_dirs:
        for side in (Path(run) / "sidecars").glob("*.json"):
            rec = json.loads(side.read_text(encoding="utf-8"))
            records[rec["id"]] = (Path(run), rec)
    by_run: dict[Path, list[dict]] = {}
    for res in results.values():
        if res["kind"] == "crop" and res["record_id"] in records:
            run, rec = records[res["record_id"]]
            by_run.setdefault(run, []).append({**res, "rec": rec})
    summary = {"batch_id": batch_dir.name, "requests": len(reqs), "answered": len(results), "missing": len(missing),
               "valid": sum(1 for r in results.values() if r["valid"]), "problems": problems,
               "vlm_run": run_info, "model_matches": run_info.get("model_id") in (None, cfg.vlm.model_id)}
    for run, items in by_run.items():
        _write_results(run, batch_dir.name, items, summary)
    summary["feedback"] = feedback(results.values(), records)
    write_json_atomic(batch_dir / "import_summary.json", summary)
    return summary


def _parse(r: dict, req: dict, allowed: set[str]) -> dict:
    resp = r.get("response") or {}
    labels = _labels(resp)
    problem = ""
    if not labels:
        problem = "no labels"
    elif any(l not in allowed for l in labels):
        problem = "label outside the IRC:82 list: " + ",".join(l for l in labels if l not in allowed)
    elif "Normal" in labels and len(labels) > 1:
        problem = "Normal together with distress labels"
    return {"request_id": req["request_id"], "kind": req["kind"], "record_id": req.get("record_id"),
            "frame_request_id": req.get("frame_request_id"), "valid": not problem, "problem": problem,
            "stage1_label": resp.get("stage1_label"), "stage1_confidence": resp.get("stage1_confidence"),
            "labels": labels, "severity": resp.get("severity"), "stage2_confidence": resp.get("stage2_confidence"),
            "needs_expert_review": resp.get("needs_expert_review"), "description": resp.get("description"),
            "http_status": r.get("http_status"), "seconds": r.get("seconds"), "raw": resp}


def _write_results(run: Path, batch_id: str, items: list[dict], summary: dict) -> None:
    out_csv = run / f"vlm_results_{batch_id}.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS + ["anomaly_lat", "anomaly_lon", "utc_iso", "evidence", "crop_file"],
                           extrasaction="ignore")
        w.writeheader()
        for it in items:
            a = it["rec"].get("anomaly") or {}
            row = {k: it.get(k) for k in RESULT_FIELDS}
            row["labels"] = "+".join(it["labels"])
            row.update({"anomaly_lat": a.get("lat"), "anomaly_lon": a.get("lon"), "utc_iso": it["rec"].get("utc_iso"),
                        "evidence": it["rec"].get("evidence"), "crop_file": (it["rec"].get("crop") or {}).get("file")})
            w.writerow(row)
    feats = []
    for it in items:
        a = it["rec"].get("anomaly") or {}
        if a.get("lat") is None:
            continue
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
                      "properties": {"id": it["record_id"], "labels": it["labels"], "severity": it["severity"],
                                     "valid": it["valid"], "needs_expert_review": it["needs_expert_review"],
                                     "utc": it["rec"].get("utc_iso"), "local_time": it["rec"].get("local_iso"),
                                     "crop_file": (it["rec"].get("crop") or {}).get("file"),
                                     "description": it["description"]}})
    (run / f"vlm_results_{batch_id}.geojson").write_text(to_json({"type": "FeatureCollection", "features": feats}),
                                                         encoding="utf-8")


def feedback(results, records: dict) -> dict:
    """How often the VLM confirmed a distress, by evidence band and detector signal.

    A recommendation only (never applied automatically): the VLM's own Stage 1 misses about
    half of all distress (response notes 5.1), so its "Normal" is not reliable ground truth.
    """
    rows = []
    for res in results:
        if res["kind"] != "crop" or not res["valid"] or res["record_id"] not in records:
            continue
        rec = records[res["record_id"]][1]
        rows.append((float(rec.get("evidence", 0)), "+".join(rec.get("kinds", [])), "Normal" not in res["labels"]))
    if not rows:
        return {}
    rows.sort()
    bands = []
    n = len(rows)
    for q in range(5):
        part = rows[q * n // 5:(q + 1) * n // 5]
        if part:
            bands.append({"evidence_from": part[0][0], "evidence_to": part[-1][0], "n": len(part),
                          "vlm_distress_share": sum(1 for r in part if r[2]) / len(part)})
    kinds = {}
    for _, k, d in rows:
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        kinds[k][1] += int(d)
    return {"by_evidence_quintile": bands,
            "by_signal": {k: {"n": v[0], "vlm_distress_share": v[1] / v[0]} for k, v in kinds.items()},
            "note": "Recommendation only. If the lowest-evidence band is almost always Normal, raising "
                    "anomaly.min_mass saves VLM time; check the frames before changing it."}
