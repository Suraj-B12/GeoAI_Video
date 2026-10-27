"""Offline hand-off to the Qwen2.5-VL classifier on the VLM machine (docs/pipeline/09, 13 §6).

Folder batch (pilot default, no network):
  geoai vlm-export RUN_DIR... --batch BATCH_DIR -> requests.jsonl + images/ + SHA256SUMS + runner script
  (USB drive) -> on the VLM machine: python run_vlm_batch.py BATCH_DIR  (calls their existing
                 POST /classify on 127.0.0.1:8000; standard library only; resumable)
  (USB drive) -> geoai vlm-import BATCH_DIR [RUN_DIR...]  -> validated vlm_results.csv / .geojson

A batch holds runs of one mode (report.json `mode`), never both:
- crops (documents 05-07): one request per sent crop, plus each crop's full frame, already
  downscaled to what the model receives (1344x756);
- segments (document 13): one request per `send` and `audit` segment, kind "segment". The image is
  the best frame at full resolution, lossless; the VLM's processor resizes it itself, exactly as in
  its evaluation on whole road frames. The batch also carries every segment record of every status
  (records/<id>.json) and each run's report.json (runs/<session_id>/report.json). `vlm-import` and
  `analyze` therefore need only the batch folder.

Facts used from the VLM team's answers (vlm_handoff/vlm-information-response.md):
- /classify takes multipart field "file" and returns JSON with stage1_label, stage1_confidence,
  distress_types, severity, description, stage2_confidence, needs_expert_review and raw outputs.
- Valid labels are the 18 IRC:82-2015 names plus "Normal"; anything else (e.g. "N/A",
  "Unknown") means "no valid label" and goes to review.
- 10 requests per minute per client; one image at a time (~11-18 s each).
- Uploads are limited to 10 MB and to 4096 x 4096 px per image (notes 1.3). S21 4K frames measured
  4.6-6.5 MB as PNG. A larger one is re-encoded losslessly at the highest zlib level (same pixels);
  if it is still too large, it is listed in batch.json `not_exported` instead of failing on the VLM
  machine.
- For week 1 they asked for the full frame of every crop as its own request.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import __version__
from .config import Config
from .qwen import frame_input_size
from .segment_records import (SENT_STATUSES, STATUSES, load_segment_records, segment_feature, segment_row,
                              validate_segment_record, write_feature_collection, write_segment_outputs, write_table_csv)
from .util import sha256_file, to_json, write_json_atomic

RUNNER = Path(__file__).resolve().parent.parent / "tools" / "run_vlm_batch.py"
BATCH_SCHEMA = "geoai.vlm_batch.v1"
SEGMENT_STATUSES = SENT_STATUSES       # segment statuses a batch can carry (skip and not_covered have no image)
SERVER_MAX_UPLOAD_BYTES = 10_000_000   # /classify upload limit "10 MB" (notes 1.3), read as the smaller decimal MB
SERVER_MAX_SIDE_PX = 4096              # /classify accepts 32x32 .. 4096x4096 (notes 1.3)
SECONDS_PER_IMAGE = 14.0               # middle of the measured 11-18 s per image (notes 1.6)
_COPY_CHUNK = 1 << 20


def run_mode(run_dir: Path) -> str:
    """'segments' or 'crops', from the run's report.json (crop runs written before the field have none)."""
    rep = Path(run_dir) / "report.json"
    mode = json.loads(rep.read_text(encoding="utf-8")).get("mode") if rep.exists() else None
    if mode in (None, "crops"):
        return "crops"
    if mode == "segments":
        return "segments"
    raise ValueError(f"{rep}: unknown mode {mode!r}")


def export_batch(run_dirs: list[Path], batch_dir: Path, cfg: Config, *, include_frames: bool | None = None,
                 statuses: Sequence[str] | None = None) -> dict:
    """Write a VLM batch for the runs (see the module docstring); returns batch.json.

    statuses: record statuses to send. Default: ("sent",) for crop runs, ("send", "audit") for segment
        runs, which may only use those two.
    include_frames: crop runs only; default cfg.crop.full_frame.
    A failed export removes what it wrote (the folder was empty or absent), so it can simply be rerun.
    """
    batch_dir = Path(batch_dir)
    existed = batch_dir.exists()
    if existed and any(batch_dir.iterdir()):
        raise FileExistsError(f"{batch_dir} is not empty")
    runs = [Path(r) for r in run_dirs]
    modes = {run_mode(r) for r in runs}
    if len(modes) > 1:
        raise ValueError("a batch holds either crop runs or segment runs, not both")
    try:
        if modes == {"segments"}:
            return _export_segments(runs, batch_dir, cfg, SEGMENT_STATUSES if statuses is None else tuple(statuses))
        return _export_crops(runs, batch_dir, cfg, include_frames, ("sent",) if statuses is None else tuple(statuses))
    except BaseException:  # including Ctrl-C: never leave a half-written batch that looks complete
        if existed:
            for child in batch_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        elif batch_dir.exists():
            shutil.rmtree(batch_dir)
        raise


def _export_crops(run_dirs: list[Path], batch_dir: Path, cfg: Config, include_frames: bool | None,
                  statuses: tuple[str, ...]) -> dict:
    (batch_dir / "images").mkdir(parents=True, exist_ok=True)
    include_frames = cfg.crop.full_frame if include_frames is None else include_frames
    requests = []
    seen_frames: set[str] = set()
    sessions = []
    for run in run_dirs:
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
    _write_requests(batch_dir, requests)
    batch = {
        "schema": BATCH_SCHEMA, "batch_id": batch_dir.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_version": __version__, "sessions": sessions, "n_requests": len(requests),
        "n_crops": sum(1 for r in requests if r["kind"] == "crop"), "n_frames": sum(1 for r in requests if r["kind"] == "frame"),
        "expected_model_id": cfg.vlm.model_id, "expected_prompt_version": cfg.vlm.prompt_version,
        "image_rules": "PNG, sRGB, upright; both sides multiples of 28, at most 1,048,576 px: the model receives these pixels unchanged",
    }
    _finish_batch(batch_dir, batch, _running_txt(batch))
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


def _write_requests(batch_dir: Path, requests: list[dict]) -> None:
    with open(batch_dir / "requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _finish_batch(batch_dir: Path, batch: dict, running_txt: str) -> None:
    write_json_atomic(batch_dir / "batch.json", batch)
    shutil.copyfile(RUNNER, batch_dir / "run_vlm_batch.py")
    (batch_dir / "RUNNING.txt").write_text(running_txt, encoding="utf-8", newline="\n")
    _write_sums(batch_dir)


# ---- segment export ------------------------------------------------------------------------------

def _export_segments(runs: list[Path], batch_dir: Path, cfg: Config, statuses: tuple[str, ...]) -> dict:
    if not statuses or any(s not in SEGMENT_STATUSES for s in statuses):
        raise ValueError(f"segment runs can send only statuses {SEGMENT_STATUSES}, got {statuses}")
    (batch_dir / "images").mkdir(parents=True, exist_ok=True)
    requests: list[dict] = []
    sessions: list[str] = []
    by_status = {s: 0 for s in STATUSES}
    reencoded: list[str] = []
    not_exported: list[dict] = []
    for run in runs:
        recs = load_segment_records(run / "sidecars")
        if not recs:
            raise ValueError(f"{run}: no segment records in sidecars/")
        bad = {r.get("id"): pr for r in recs if (pr := validate_segment_record(r))}
        if bad:
            rid, pr = next(iter(bad.items()))
            raise ValueError(f"{run}: {len(bad)} invalid segment records, e.g. {rid}: {'; '.join(pr)}")
        sids = {r["session_id"] for r in recs}
        if len(sids) != 1:
            raise ValueError(f"{run}: records of several sessions {sorted(sids)}")
        sid = sids.pop()
        if sid in sessions:
            raise ValueError(f"session {sid} is in two of the runs given")
        sessions.append(sid)
        rep_dst = batch_dir / "runs" / sid / "report.json"
        rep_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "report.json", rep_dst)
        for rec in recs:
            by_status[rec["status"]] += 1
            # The record as extracted: an answer from an earlier batch must not travel with a new one.
            write_json_atomic(batch_dir / "records" / f"{rec['id']}.json", {k: v for k, v in rec.items() if k != "vlm"})
            if rec["status"] in statuses:
                req = _segment_request(run, rec, batch_dir, reencoded, not_exported)
                if req is not None:
                    requests.append(req)
    _write_requests(batch_dir, requests)
    sizes = sorted({(r["width"], r["height"]) for r in requests})
    nbytes = [(batch_dir / r["image"]).stat().st_size for r in requests]
    batch = {
        "schema": BATCH_SCHEMA, "batch_id": batch_dir.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_version": __version__, "mode": "segments", "sessions": sessions, "statuses": list(statuses),
        "n_records": sum(by_status.values()), "records_by_status": by_status,
        "n_requests": len(requests), "n_segments": len(requests), "n_crops": 0, "n_frames": 0,
        "requests_by_status": {s: sum(1 for r in requests if r["status"] == s) for s in statuses},
        "expected_model_id": cfg.vlm.model_id, "expected_prompt_version": cfg.vlm.prompt_version,
        "image_rules": _segment_image_rules(sizes),
        "model_input_sizes": {f"{w}x{h}": list(frame_input_size(w, h)) for w, h in sizes},
        "max_upload_bytes": SERVER_MAX_UPLOAD_BYTES, "image_bytes_total": sum(nbytes),
        "image_bytes_max": max(nbytes, default=0), "reencoded": reencoded, "not_exported": not_exported,
        "estimated_vlm_hours": len(requests) * SECONDS_PER_IMAGE / 3600,
    }
    _finish_batch(batch_dir, batch, _running_txt_segments(batch, sizes))
    return batch


def _segment_request(run: Path, rec: dict, batch_dir: Path, reencoded: list[str], not_exported: list[dict]) -> dict | None:
    """Copy one segment frame into the batch (checked against its record) and describe the request."""
    from PIL import Image

    rid, fr = rec["id"], rec["frame"]
    src = run / fr["file"]
    dst = batch_dir / "images" / f"{rid}.png"
    if _copy_hashing(src, dst) != fr["sha256"]:
        raise ValueError(f"{src} does not match frame.sha256 of its record (was the run re-processed?)")
    with Image.open(dst) as im:
        size = im.size
    if size != (fr["width"], fr["height"]):
        raise ValueError(f"{src} is {size[0]}x{size[1]}, its record says {fr['width']}x{fr['height']}")
    w, h = size
    if max(w, h) > SERVER_MAX_SIDE_PX:
        dst.unlink()
        not_exported.append({"id": rid, "reason": f"{w}x{h} is larger than the server's {SERVER_MAX_SIDE_PX} px limit"})
        return None
    if dst.stat().st_size > SERVER_MAX_UPLOAD_BYTES:
        _reencode_png(dst)
        if dst.stat().st_size > SERVER_MAX_UPLOAD_BYTES:
            not_exported.append({"id": rid, "reason": f"{dst.stat().st_size} bytes after lossless re-encoding, "
                                                      f"over the server's {SERVER_MAX_UPLOAD_BYTES} byte limit"})
            dst.unlink()
            return None
        reencoded.append(rid)
    mw, mh = frame_input_size(w, h)
    return {"request_id": rid, "kind": "segment", "image": f"images/{rid}.png", "image_sha256": sha256_file(dst),
            "width": w, "height": h, "session_id": rec["session_id"], "record_id": rid, "frame_request_id": None,
            "status": rec["status"], "model_input": [mw, mh]}


def _copy_hashing(src: Path, dst: Path) -> str:
    """Copy src to dst in one read, returning the SHA-256 of the bytes copied."""
    import hashlib

    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(_COPY_CHUNK):
            h.update(chunk)
            fo.write(chunk)
    return h.hexdigest()


def _reencode_png(path: Path) -> None:
    """Re-encode a PNG losslessly at zlib level 9, keeping its text chunks and EXIF; the pixels are checked equal."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    tmp = path.with_suffix(".tmp.png")
    with Image.open(path) as im:
        im.load()
        pixels = np.asarray(im)
        info = PngInfo()
        for key, value in im.text.items():
            info.add_text(key, value)
        kwargs = {"format": "PNG", "compress_level": 9, "pnginfo": info}
        if im.info.get("exif"):
            kwargs["exif"] = im.info["exif"]
        im.save(tmp, **kwargs)
    with Image.open(tmp) as im2:
        same = np.array_equal(np.asarray(im2), pixels)
    if not same:
        tmp.unlink()
        raise RuntimeError(f"lossless re-encoding of {path} changed its pixels")
    tmp.replace(path)


def _segment_image_rules(sizes: list[tuple[int, int]]) -> str:
    resized = ", ".join(f"{w}x{h} -> {mw}x{mh}" for (w, h), (mw, mh) in
                        ((s, frame_input_size(*s)) for s in sizes)) or "no images"
    return ("PNG, sRGB, upright, lossless: the whole camera frame at full resolution. The VLM's processor resizes "
            "it itself (aspect ratio kept, both sides multiples of 28, at most 1,048,576 px; "
            f"{resized}), as in its evaluation on whole road frames. Do not resize or re-encode.")


def _write_sums(batch_dir: Path) -> None:
    lines = []
    for p in sorted(batch_dir.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS", "responses.jsonl", "vlm_run.json"):
            lines.append(f"{sha256_file(p)}  {p.relative_to(batch_dir).as_posix()}")
    (batch_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_sums(batch_dir: Path, *, skip_prefixes: Sequence[str] = ()) -> list[str]:
    """Files whose checksum no longer matches (empty list = intact).

    skip_prefixes: relative paths starting with these are not checked (analysis skips "images/": the
    numbers depend only on the records and answers, and the runner checked each image as it sent it).
    """
    bad = []
    for line in (Path(batch_dir) / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        if any(rel.startswith(pre) for pre in skip_prefixes):
            continue
        p = Path(batch_dir) / rel
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


def _running_txt_segments(batch: dict, sizes: list[tuple[int, int]]) -> str:
    n = batch["n_requests"]
    dims = ", ".join(f"{w}x{h}" for w, h in sizes) or "none"
    model_in = ", ".join(f"{w}x{h} -> {mw}x{mh}" for (w, h), (mw, mh) in ((s, frame_input_size(*s)) for s in sizes))
    return f"""GeoAI VLM batch {batch['batch_id']} - {n} road-segment frames from {len(batch['sessions'])} session(s)

On the VLM machine, with the classifier server running (POST /classify on port 8000):

    python run_vlm_batch.py .

The script needs only the Python standard library. It:
  1. checks SHA256SUMS (stops if any file was altered in transit),
  2. reads GET /health and records the model, prompt and settings in vlm_run.json,
  3. uploads each image in requests.jsonl to /classify, one at a time (at most 10 per minute),
     and appends one line per image to responses.jsonl,
  4. can be stopped with Ctrl-C and restarted at any time: answered images are skipped.

At about 11-18 s per image this batch needs about {n * 11 / 3600:.1f}-{n * 18 / 3600:.1f} hours.
Options: --url http://127.0.0.1:8000  --api-key KEY  --timeout 300  --limit N
Expected model: {batch['expected_model_id']}
Expected prompt version: {batch['expected_prompt_version']}

Images are whole camera frames at full resolution ({dims}), lossless PNG, at most
{batch['image_bytes_max'] / 1e6:.1f} MB each. Your processor resizes them itself ({model_in or 'none'}), exactly as in
your evaluation on whole road frames: please don't resize or re-encode them. Copy the whole folder
back when done, and please delete the images from the VLM machine afterwards (the frames may show
vehicles or people).
"""


# ---- answers -------------------------------------------------------------------------------------

RESULT_FIELDS = ["request_id", "kind", "record_id", "frame_request_id", "valid", "problem", "stage1_label",
                 "stage1_confidence", "labels", "severity", "stage2_confidence", "needs_expert_review",
                 "description", "frame_labels", "crop_frame_agree", "http_status", "seconds"]


@dataclass
class Answers:
    """A batch's requests and what the runner recorded for them.

    requests: request_id -> request, in requests.jsonl order.
    results: request_id -> parsed answer (`_parse`), for answered requests only.
    errors: request_id -> the error of its last error line, for requests never answered.
    problems: counts of response lines that could not be used.
    lines: every readable response line, in file order (for timing).
    run_info: vlm_run.json ({} before the runner has run).
    """
    requests: dict[str, dict]
    results: dict[str, dict]
    errors: dict[str, str]
    problems: dict[str, int]
    lines: list[dict] = field(default_factory=list)
    run_info: dict = field(default_factory=dict)


def _labels(resp: dict) -> list[str]:
    """The answer's labels. /classify runs Stage 2 only on frames Stage 1 called Distress; for a frame
    Stage 1 called Normal it returns stage1_label "Normal" and an empty distress_types (the VLM
    system's PavementClassifier.predict), and that answer is the label Normal."""
    v = resp.get("distress_types")
    if not v:
        return ["Normal"] if str(resp.get("stage1_label") or "").strip().lower() == "normal" else []
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    out = []
    for x in v:
        x = str(x).strip()
        if x.lower().startswith("normal"):
            x = "Normal"
        out.append(x)
    return out


def read_answers(batch_dir: Path, cfg: Config) -> Answers:
    """Parse requests.jsonl, responses.jsonl and vlm_run.json of a batch (checksums: see verify_sums).

    A request counts as answered by its response lines without an error; error lines leave it
    unanswered (the runner sends it again). The last line is ignored if it is cut off (an interrupted
    write); other unreadable lines, lines for unknown requests and answered requests whose repeated
    answers disagree are counted in `problems` (the disagreeing answer is marked invalid).
    """
    batch_dir = Path(batch_dir)
    reqs: dict[str, dict] = {}
    for line in (batch_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            reqs[r["request_id"]] = r
    run_info = {}
    if (batch_dir / "vlm_run.json").exists():
        run_info = json.loads((batch_dir / "vlm_run.json").read_text(encoding="utf-8"))
    allowed = set(cfg.vlm.labels)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    problems = {"unknown_request": 0, "duplicate_conflict": 0, "bad_json": 0}
    parsed_lines: list[dict] = []
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
        parsed_lines.append(r)
        rid = r.get("request_id")
        if rid not in reqs:
            problems["unknown_request"] += 1
            continue
        if r.get("error"):
            errors[rid] = str(r["error"])
            continue  # not classified; will be re-sent
        res = _parse(r, reqs[rid], allowed)
        if rid in results and results[rid]["labels"] != res["labels"]:
            problems["duplicate_conflict"] += 1
            res["valid"] = False
            res["problem"] = "conflicting duplicate responses"
        results[rid] = res
    for rid in results:
        errors.pop(rid, None)
    return Answers(reqs, results, errors, problems, parsed_lines, run_info)


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


def answer_class(res: Mapping[str, Any] | None, allowed: Iterable[str]) -> tuple[str, list[str]]:
    """(class, distress labels) of one answer: class "distress", "normal" or "unknown".

    VLM distress = any valid non-Normal label (docs/pipeline/13 §4). Validity is judged label by label,
    so an answer flagged for an extra unknown label or for "Normal" mixed with distress still reports
    the valid distress it names; the flag stays in `problem`. "normal" = valid labels, Normal only.
    "unknown" = no valid label (e.g. "N/A", an empty answer), or repeated answers that disagree.
    """
    if res is None or res.get("problem") == "conflicting duplicate responses":
        return "unknown", []
    allowed = set(allowed)
    valid = [lab for lab in res.get("labels") or [] if lab in allowed]
    distress = sorted({lab for lab in valid if lab != "Normal"})
    if distress:
        return "distress", distress
    return ("normal", []) if "Normal" in valid else ("unknown", [])


def vlm_block(res: Mapping[str, Any], batch_id: str) -> dict:
    """The record's `vlm` block (segment_records.VLM_FIELDS) from one parsed answer."""
    return {"labels": list(res["labels"]), "stage1_label": res.get("stage1_label"),
            "stage1_confidence": res.get("stage1_confidence"), "severity": res.get("severity"),
            "description": res.get("description"), "valid": bool(res["valid"]), "problem": res["problem"],
            "stage2_confidence": res.get("stage2_confidence"), "needs_expert_review": res.get("needs_expert_review"),
            "batch_id": batch_id}


def health_identity(run_info: Mapping[str, Any], cfg: Config) -> dict:
    """What GET /health (vlm_run.json) reported about the model and prompt, against the expected ids.

    The response's field names are documented only as "model, prompt version, quantization, pixel cap,
    probe status" (notes 1.3), so the usual names are looked up. A value that isn't reported gives None
    (unknown), not a mismatch. A model name without "@revision" can match only by name. Every runner
    session records its own /health, and they must all agree (`consistent`): a server changed in the
    middle of a batch would mix two models in one result.
    """
    healths = [r.get("health") or {} for r in run_info.get("runs") or []] or [run_info.get("health") or {}]
    ids = [(_first_str(h, ("model_id", "model", "model_name")), _first_str(h, ("prompt_version", "prompts_version")))
           for h in healths]
    model, prompt = ids[-1]
    exp_model, exp_prompt = cfg.vlm.model_id, cfg.vlm.prompt_version
    if model is None:
        model_ok = None
    elif "@" in model:
        model_ok = exp_model == model or exp_model.startswith(model + "/")
    else:
        model_ok = model == exp_model.split("@")[0]
    # The VLM system's /health reports prompts_version as the name alone ("v2"), without the prompt id
    # after the colon: such a value can match only by name, like a model without "@revision".
    if prompt is None:
        prompt_ok = None
    elif ":" in prompt:
        prompt_ok = prompt == exp_prompt
    else:
        prompt_ok = prompt == exp_prompt.split(":")[0]
    return {"model_reported": model, "prompt_version_reported": prompt, "model_matches": model_ok,
            "prompt_matches": prompt_ok, "consistent": len(set(ids)) <= 1}


def _first_str(d: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for k in keys:
        if isinstance(d.get(k), str) and d[k]:
            return d[k]
    return None


# ---- import --------------------------------------------------------------------------------------

def import_batch(batch_dir: Path, run_dirs: list[Path], cfg: Config) -> dict:
    """Validate an answered batch and write its results; returns the summary (also import_summary.json).

    Crop batches: vlm_results_<batch>.csv / .geojson in each run folder given.
    Segment batches: vlm_results.csv / .geojson in the batch folder (works without any run folder); for
    each segment run folder given, the answers are also added to its sidecars as `vlm` blocks, its
    segments.csv / .geojson are rewritten and vlm_results_<batch>.csv / .geojson are written beside them.
    Raises ValueError if any batch file changed since export.
    """
    batch_dir = Path(batch_dir)
    bad = verify_sums(batch_dir)
    if bad:
        raise ValueError(f"batch files changed since export: {bad[:5]}")
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    ans = read_answers(batch_dir, cfg)
    if batch.get("mode") == "segments":
        return _import_segments(batch_dir, batch, ans, [Path(r) for r in run_dirs], cfg)
    reqs, results = ans.requests, ans.results
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
    ident = health_identity(ans.run_info, cfg)
    summary = {"batch_id": batch_dir.name, "requests": len(reqs), "answered": len(results), "missing": len(missing),
               "valid": sum(1 for r in results.values() if r["valid"]), "problems": ans.problems,
               "vlm_run": ans.run_info, "model_matches": ident["model_matches"], "prompt_matches": ident["prompt_matches"]}
    for run, items in by_run.items():
        _write_results(run, batch_dir.name, items, summary)
    summary["feedback"] = feedback(results.values(), records)
    write_json_atomic(batch_dir / "import_summary.json", summary)
    return summary


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


# Columns of a segment batch's vlm_results.csv: one row per request, answered or not.
SEGMENT_RESULT_COLUMNS: tuple[tuple[str, int | None], ...] = (
    ("request_id", None), ("session_id", None), ("k", None), ("status", None), ("s0_m", 3), ("s1_m", 3),
    ("length_m", 3), ("chainage_start_km", 6), ("chainage_end_km", 6), ("centre_lat", 9), ("centre_lon", 9),
    ("utc_iso", None), ("local_iso", None), ("frame_file", None), ("answered", None), ("error", None),
    ("valid", None), ("problem", None), ("vlm_class", None), ("labels", None), ("distress_labels", None),
    ("stage1_label", None), ("stage1_confidence", 4), ("severity", None), ("stage2_confidence", 4),
    ("needs_expert_review", None), ("description", None), ("gate_decision_if_gated", None), ("probe_frac", 4),
    ("evidence", 2), ("crack_mm", 1), ("http_status", None), ("seconds", 3),
)


def load_batch_records(batch_dir: Path) -> dict[str, dict]:
    """The segment records a batch carries (records/), by id."""
    return {r["id"]: r for r in load_segment_records(Path(batch_dir) / "records")}


def _segment_result_row(req: Mapping[str, Any], rec: Mapping[str, Any], res: Mapping[str, Any] | None,
                        error: str | None, allowed: set[str]) -> dict:
    base = segment_row(rec)
    cls, distress = answer_class(res, allowed)
    r = res or {}
    return {"request_id": req["request_id"], **{k: base[k] for k in (
        "session_id", "k", "status", "s0_m", "s1_m", "length_m", "chainage_start_km", "chainage_end_km",
        "centre_lat", "centre_lon", "utc_iso", "local_iso", "frame_file", "gate_decision_if_gated", "probe_frac",
        "evidence", "crack_mm")},
        "answered": res is not None, "error": None if res is not None else error,
        "valid": r.get("valid"), "problem": r.get("problem"), "vlm_class": cls if res is not None else None,
        "labels": r.get("labels"), "distress_labels": distress if res is not None else None,
        "stage1_label": r.get("stage1_label"), "stage1_confidence": r.get("stage1_confidence"),
        "severity": r.get("severity"), "stage2_confidence": r.get("stage2_confidence"),
        "needs_expert_review": r.get("needs_expert_review"), "description": r.get("description"),
        "http_status": r.get("http_status"), "seconds": r.get("seconds")}


def _write_segment_results(folder: Path, name: str, rows: list[dict], recs: Mapping[str, Mapping[str, Any]]) -> None:
    """<name>.csv and <name>.geojson (footprint polygons) of segment result rows, in `folder`."""
    write_table_csv(folder / f"{name}.csv", rows, SEGMENT_RESULT_COLUMNS)
    write_feature_collection(folder / f"{name}.geojson", [segment_feature(recs[r["request_id"]], r) for r in rows])


def _import_segments(batch_dir: Path, batch: dict, ans: Answers, run_dirs: list[Path], cfg: Config) -> dict:
    batch_id = batch_dir.name
    allowed = set(cfg.vlm.labels)
    records = load_batch_records(batch_dir)
    lost = [rid for rid in ans.requests if rid not in records]
    if lost:
        raise ValueError(f"batch has requests without records/: {lost[:5]}")
    rows = [_segment_result_row(req, records[rid], ans.results.get(rid), ans.errors.get(rid), allowed)
            for rid, req in ans.requests.items()]
    _write_segment_results(batch_dir, "vlm_results", rows, records)

    runs_updated: dict[str, int] = {}
    run_problems: list[str] = []
    for run in run_dirs:
        if run_mode(run) != "segments":
            raise ValueError(f"{run} is not a segment run")
        recs = load_segment_records(run / "sidecars")
        n = 0
        for rec in recs:
            res = ans.results.get(rec["id"])
            if res is None:
                continue
            if (rec.get("frame") or {}).get("sha256") != (records[rec["id"]].get("frame") or {}).get("sha256"):
                run_problems.append(f"{rec['id']}: the run's frame differs from the exported one; not updated")
                continue
            rec["vlm"] = vlm_block(res, batch_id)
            n += 1
        write_segment_outputs(run, recs)
        ids = {r["id"] for r in recs}
        _write_segment_results(run, f"vlm_results_{batch_id}", [r for r in rows if r["request_id"] in ids], records)
        runs_updated[run.name] = n

    ident = health_identity(ans.run_info, cfg)
    classes = {c: sum(1 for r in rows if r["vlm_class"] == c) for c in ("distress", "normal", "unknown")}
    summary = {"batch_id": batch_id, "mode": "segments", "requests": len(ans.requests), "answered": len(ans.results),
               "missing": len(ans.requests) - len(ans.results),
               "valid": sum(1 for r in ans.results.values() if r["valid"]), "problems": ans.problems,
               "classes": classes, "not_exported": len(batch.get("not_exported") or []),
               "errors_unresolved": len(ans.errors), "vlm_run": ans.run_info, **ident,
               "runs_updated": runs_updated, "run_problems": run_problems}
    write_json_atomic(batch_dir / "import_summary.json", summary)
    return summary


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
