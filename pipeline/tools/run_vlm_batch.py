#!/usr/bin/env python3
"""Run a GeoAI VLM batch on the VLM machine, against its existing classifier server.

Standard library only (Python 3.8+). Usage, from inside the batch folder:

    python run_vlm_batch.py .
    python run_vlm_batch.py . --url http://127.0.0.1:8000 --api-key KEY

For every image in requests.jsonl it POSTs the file to /classify (multipart field "file") and
appends one JSON line to responses.jsonl:
    {"request_id", "kind", "http_status", "seconds", "started_utc", "response": <server JSON>, "error"}
It verifies SHA256SUMS first, records GET /health in vlm_run.json, respects the server's rate
limit (10 requests per minute per client), and can be stopped and restarted: images already
answered without error are skipped. A power cut loses at most the line being written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

RUNNER_VERSION = "1"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sums(d: Path) -> list:
    bad = []
    for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = d / rel
        if not p.exists() or sha256(p) != digest:
            bad.append(rel)
    return bad


def _request(url, data=None, headers=None, timeout=300.0):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, json.loads(body.decode("utf-8")) if body else None
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body.decode("utf-8"))
        except ValueError:
            return e.code, {"text": body.decode("utf-8", "replace")[:2000]}


def multipart(path: Path):
    boundary = "geoai" + uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode("utf-8")
    body = head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def answered(resp_path: Path) -> set:
    done = set()
    if not resp_path.exists():
        return done
    lines = resp_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue  # a half-written last line after an interruption
        if not r.get("error"):
            done.add(r["request_id"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--min-interval", type=float, default=6.5, help="seconds between request starts (10/min limit)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new requests (0 = all)")
    a = ap.parse_args()
    d = a.batch_dir.resolve()
    bad = check_sums(d)
    if bad:
        print(f"STOP: {len(bad)} files differ from SHA256SUMS (copy the batch again): {bad[:5]}", file=sys.stderr)
        return 2
    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    status, health = _request(a.url.rstrip("/") + "/health", headers=headers, timeout=30)
    if status != 200:
        print(f"STOP: GET /health returned {status}: {health}", file=sys.stderr)
        return 2
    run_path = d / "vlm_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {"runs": []}
    this = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": a.url, "health": health,
            "runner_version": RUNNER_VERSION, "host": platform.node(), "python": platform.python_version()}
    run["runs"].append(this)
    run["health"] = health
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    reqs = [json.loads(l) for l in (d / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    resp_path = d / "responses.jsonl"
    done = answered(resp_path)
    todo = [r for r in reqs if r["request_id"] not in done]
    print(f"{len(reqs)} requests, {len(done)} already answered, {len(todo)} to send")
    sent = ok = 0
    last_start = 0.0
    with open(resp_path, "a", encoding="utf-8", newline="\n") as out:
        for req in todo:
            if a.limit and sent >= a.limit:
                break
            img = d / req["image"]
            rec = {"request_id": req["request_id"], "kind": req.get("kind"), "http_status": None, "seconds": None,
                   "started_utc": None, "response": None, "error": None}
            if sha256(img) != req["image_sha256"]:
                rec["error"] = "image checksum mismatch"
            else:
                body, ctype = multipart(img)
                for attempt in range(6):
                    wait = a.min_interval - (time.time() - last_start)
                    if wait > 0:
                        time.sleep(wait)
                    last_start = time.time()
                    rec["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    try:
                        status, resp = _request(a.url.rstrip("/") + "/classify", body,
                                                {**headers, "Content-Type": ctype}, a.timeout)
                    except Exception as e:  # noqa: BLE001 - recorded, retried next run
                        status, resp = None, None
                        rec["error"] = repr(e)
                        break
                    if status == 429:  # rate limited: back off and retry
                        time.sleep(10 * (attempt + 1))
                        continue
                    rec["http_status"] = status
                    rec["response"] = resp
                    rec["error"] = None if status == 200 else f"HTTP {status}"
                    break
                rec["seconds"] = round(time.time() - last_start, 3)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            sent += 1
            ok += rec["error"] is None
            print(f"[{sent}/{len(todo)}] {req['request_id']} {'ok' if rec['error'] is None else rec['error']}", flush=True)
    this["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    this["sent"], this["ok"] = sent, ok
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    remaining = len(todo) - ok
    print(f"done: {ok} answered, {remaining} still to do" + (" (run again to retry)" if remaining else ""))
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
