#!/usr/bin/env python3
"""Run a GeoAI VLM batch on the VLM machine, against its existing classifier server.

Standard library only (Python 3.8+). Usage, from inside the batch folder:

    python run_vlm_batch.py .
    python run_vlm_batch.py . --url http://127.0.0.1:8000 --api-key KEY
    python run_vlm_batch.py . --profile video        # drive-video frames (see --profile)

For every image in requests.jsonl it POSTs the file to /classify (multipart field "file") and
appends one JSON line to responses.jsonl:
    {"request_id", "kind", "http_status", "seconds", "started_utc", "attempts", "response": <server JSON>, "error"}

- It verifies SHA256SUMS first and stops if any file differs. --skip-batch-check skips that full re-hash:
  safe only when an earlier call already verified the same batch on the same disk (the toggle verifies
  on its first chunk of a session and skips on the later ones). Every image is still hashed while it is
  uploaded and checked against requests.jsonl, so a changed image is never recorded as answered.
- --profile NAME adds the form field "profile" to every request. The server then also returns
  "profile" and "views_recorded". "video" makes the classifier skip the probe's record-only tile views
  (about 9 s of a 13 s Normal answer); its labels, confidences and review flag are unchanged. Without
  --profile the request is exactly as before.
- It records GET /health in vlm_run.json (one entry per run) and warns when /health does not mention
  the model or prompt version the batch expects.
- Uploads are streamed from disk. Each image is read once, hashed while it is sent, and never held in
  memory whole (a full 4K frame is several MB). The hash is checked against requests.jsonl after the
  upload, so an image that changed on disk is never recorded as answered.
- It sends one request at a time, starting them at least --min-interval apart. The default is 6.5 s, which
  keeps within the server's 10 per minute, except when --url is this machine (127.0.0.1, ::1,
  localhost): the server does not rate-limit unproxied loopback requests, so the default there is 0.
- Retries:
  - HTTP 429 waits for Retry-After (or backs off);
  - HTTP 5xx, timeouts and connection errors are retried with exponential backoff (--retries,
    --retry-wait);
  - other HTTP errors (e.g. a rejected image) are recorded, and the next image is sent.
  It stops on 401/403 (wrong API key) and after --max-failures images in a row have failed (server
  down).
- It can be stopped and restarted: images already answered without error are skipped. A last line cut
  off by a power cut is removed before new lines are appended. Ctrl-C stops cleanly after recording
  what was answered.
- It prints one progress line per image: its time, the average, and the time left.

Exit codes: 0 every request answered; 1 some still to do (run again); 2 stopped before sending
(checksums, /health) or on an authentication error; 130 interrupted by Ctrl-C.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

RUNNER_VERSION = "3"
REMOTE_MIN_INTERVAL_S = 6.5   # 60 s / 10 requests per minute (the server's limit for remote clients)
CHUNK = 1 << 20
MAX_WAIT_S = 300.0       # longest single backoff or Retry-After wait


class Stop(Exception):
    """A condition under which sending more images is pointless (wrong API key, server down)."""

    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sums(d: Path) -> list:
    """Files whose SHA-256 differs from SHA256SUMS, or that are missing (empty list = intact)."""
    entries = [line.split("  ", 1) for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line.strip()]
    total = sum((d / rel).stat().st_size for _, rel in entries if (d / rel).exists())
    print(f"checking {len(entries)} files ({total / 1e9:.2f} GB) against SHA256SUMS ...", flush=True)
    return [rel for digest, rel in entries if not (d / rel).exists() or sha256(d / rel) != digest]


def is_loopback_url(url: str) -> bool:
    """True when the URL's host is this machine (127.0.0.0/8, ::1 or 'localhost')."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def get_json(url: str, headers: dict, timeout: float):
    """(status, parsed JSON or None) of a GET."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, _parse_json(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _parse_json(e.read())


def _parse_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        return None


class MultipartFile:
    """A multipart/form-data body with one file field (after optional text fields), produced in chunks.

    http.client sends any body that has read() block by block, so the image is streamed from disk
    instead of being joined into one bytes object (the old runner held each image twice in memory).
    The file's bytes are hashed as they go out; `complete` and `hexdigest()` tell afterwards whether
    exactly the expected file was sent.
    """

    def __init__(self, path: Path, field: str = "file", content_type: str = "image/png", fields: dict | None = None):
        boundary = "geoai" + uuid.uuid4().hex
        self.content_type = "multipart/form-data; boundary=" + boundary
        text = "".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                       for k, v in (fields or {}).items())
        self._head = (text + f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                      f"filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n").encode("utf-8")
        self._tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        self._size = path.stat().st_size
        self.length = len(self._head) + self._size + len(self._tail)
        self._f = open(path, "rb")
        self._hash = hashlib.sha256()
        self._sent_file = 0
        self._pos = 0  # position in head + file + tail

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.length
        out = bytearray()
        head, size = len(self._head), self._size
        while len(out) < n and self._pos < self.length:
            want = n - len(out)
            if self._pos < head:
                part = self._head[self._pos:self._pos + want]
            elif self._pos < head + size:
                part = self._f.read(min(want, CHUNK, head + size - self._pos))
                if not part:
                    raise OSError("image became shorter during upload")
                self._hash.update(part)
                self._sent_file += len(part)
            else:
                start = self._pos - head - size
                part = self._tail[start:start + want]
            out += part
            self._pos += len(part)
        return bytes(out)

    @property
    def complete(self) -> bool:
        return self._sent_file == self._size

    def hexdigest(self) -> str:
        return self._hash.hexdigest()

    def close(self) -> None:
        self._f.close()


def post_file(url: str, path: Path, headers: dict, timeout: float, fields: dict | None = None):
    """POST one image; returns (status, parsed JSON or None, raw text excerpt, response headers, body).

    Connection errors and timeouts are raised (OSError, http.client.HTTPException) for the caller to retry.
    """
    body = MultipartFile(path, fields=fields)
    try:
        req = urllib.request.Request(url, data=body, method="POST", headers={
            **headers, "Content-Type": body.content_type, "Content-Length": str(body.length)})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, raw, resp_headers = r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            status, raw, resp_headers = e.code, e.read(), e.headers
        return status, _parse_json(raw), raw[:2000].decode("utf-8", "replace"), resp_headers, body
    finally:
        body.close()


class Pacer:
    """Keeps request starts at least `interval` seconds apart (the server's rate limit)."""

    def __init__(self, interval: float):
        self.interval = interval
        self._last = None

    def wait(self) -> None:
        if self._last is not None:
            delay = self.interval - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
        self._last = time.monotonic()


def backoff(attempt: int, base: float) -> float:
    return min(MAX_WAIT_S, base * 2 ** (attempt - 1))


def retry_after(resp_headers, default: float) -> float:
    try:
        return min(MAX_WAIT_S, max(0.0, float(resp_headers.get("Retry-After"))))
    except (TypeError, ValueError, AttributeError):
        return default


def send_one(d: Path, req: dict, url: str, headers: dict, a, pacer: Pacer) -> dict:
    """Classify one image, with retries; returns its responses.jsonl line."""
    line = {"request_id": req["request_id"], "kind": req.get("kind"), "http_status": None, "seconds": None,
            "started_utc": None, "attempts": 0, "response": None, "error": None}
    img = d / req["image"]
    if not img.exists():
        line["error"] = "image missing"
        return line
    size = img.stat().st_size
    if size > a.max_upload_mb * 1e6:
        line["error"] = f"image is {size} bytes, over the server's upload limit ({a.max_upload_mb} MB)"
        return line
    for attempt in range(1, a.retries + 2):
        pacer.wait()
        line["attempts"] = attempt
        line["started_utc"] = utc_now()
        t0 = time.monotonic()
        try:
            if a.profile:
                status, resp, text, resp_headers, body = post_file(url, img, headers, a.timeout,
                                                                   fields={"profile": a.profile})
            else:
                status, resp, text, resp_headers, body = post_file(url, img, headers, a.timeout)
        except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets are all OSError
            line.update(seconds=round(time.monotonic() - t0, 3), http_status=None, response=None,
                        error=f"{type(e).__name__}: {e}")
            if attempt <= a.retries:
                wait = backoff(attempt, a.retry_wait)
                print(f"    {req['request_id']}: {line['error']}; retry in {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            return line
        line["seconds"] = round(time.monotonic() - t0, 3)
        line["http_status"] = status
        if not body.complete or body.hexdigest() != req["image_sha256"]:
            line.update(response=None, error="image checksum differs from requests.jsonl (file changed on disk)")
            return line
        line["response"] = resp if resp is not None else {"text": text}
        if status == 200 and isinstance(resp, dict):
            line["error"] = None
            return line
        line["error"] = f"HTTP {status}" if status != 200 else "HTTP 200 without a JSON object"
        if status in (401, 403):
            raise Stop(f"the server refused the request (HTTP {status}): check --api-key", 2)
        if (status == 429 or status >= 500) and attempt <= a.retries:
            wait = backoff(attempt, a.retry_wait)
            if status == 429:
                wait = retry_after(resp_headers, wait)
            print(f"    {req['request_id']}: HTTP {status}; retry in {wait:.0f} s", flush=True)
            time.sleep(wait)
            continue
        return line
    return line


def answered(resp_path: Path) -> set:
    """Request ids with a response line without an error (unreadable lines are ignored)."""
    done = set()
    if not resp_path.exists():
        return done
    for line in resp_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("request_id") and not r.get("error"):
            done.add(r["request_id"])
    return done


def repair_tail(resp_path: Path) -> str | None:
    """Make responses.jsonl end with a newline before appending; returns what was done, or None.

    Lines are written whole and then synced, so a missing final newline means the write was cut off.
    A cut-off fragment is removed (its image is sent again). A complete JSON line that lost only its
    newline keeps its answer: the newline is added.
    """
    if not resp_path.exists():
        return None
    data = resp_path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    try:
        json.loads(data[cut:].decode("utf-8"))
    except ValueError:
        with open(resp_path, "r+b") as f:
            f.truncate(cut)
        return f"removed a cut-off last line ({len(data) - cut} bytes)"
    with open(resp_path, "ab") as f:
        f.write(b"\n")
    return "added the missing newline after the last line"


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fmt_h(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds to wait on the server per request")
    ap.add_argument("--min-interval", type=float, default=None,
                    help=f"seconds between request starts (default {REMOTE_MIN_INTERVAL_S}, the 10/min limit; "
                         "0 when --url is this machine, which the server does not rate-limit)")
    ap.add_argument("--profile", default=None,
                    help="send this classifier profile with every request ('video' for drive-video frames: "
                         "skips the server's record-only probe tiles, same labels); default: none, as before")
    ap.add_argument("--skip-batch-check", action="store_true",
                    help="do not re-hash the whole batch against SHA256SUMS. Safe only when an earlier call "
                         "verified this batch on this disk; each image is still hashed as it is uploaded")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new requests (0 = all)")
    ap.add_argument("--only-kind", choices=("segment", "segment_band", "crop", "frame"), default=None,
                    help="send only requests of this kind")
    ap.add_argument("--retries", type=int, default=4, help="retries per image after timeouts, 429 and 5xx")
    ap.add_argument("--retry-wait", type=float, default=5.0, help="first backoff in seconds (doubles per retry)")
    ap.add_argument("--max-failures", type=int, default=5, help="stop after this many failed images in a row")
    ap.add_argument("--max-upload-mb", type=float, default=10.0, help="the server's upload limit (MB)")
    a = ap.parse_args(argv)
    if a.min_interval is None:
        a.min_interval = 0.0 if is_loopback_url(a.url) else REMOTE_MIN_INTERVAL_S
    a.profile = (a.profile or "").strip() or None
    d = a.batch_dir.resolve()
    try:
        return run(d, a)
    except KeyboardInterrupt:
        print("\ninterrupted before any image was sent", file=sys.stderr)
        return 130


def run(d: Path, a) -> int:
    if a.skip_batch_check:
        print("not re-checking SHA256SUMS (--skip-batch-check); each image is hashed as it is sent", flush=True)
        bad = []
    else:
        bad = check_sums(d)
    if bad:
        print(f"STOP: {len(bad)} files differ from SHA256SUMS (copy the batch again): {bad[:5]}", file=sys.stderr)
        return 2
    batch = json.loads((d / "batch.json").read_text(encoding="utf-8")) if (d / "batch.json").exists() else {}
    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    base = a.url.rstrip("/")
    try:
        status, health = get_json(base + "/health", headers, 30.0)
    except (OSError, http.client.HTTPException) as e:
        print(f"STOP: cannot reach {base}/health ({e}). Is the classifier server running?", file=sys.stderr)
        return 2
    if status != 200:
        print(f"STOP: GET /health returned {status}: {health}", file=sys.stderr)
        return 2
    said = json.dumps(health)
    # The server may report the model without "@revision" and the prompt version without ":<prompt id>"
    # (the VLM system's /health says "prompts_version": "v2"). A name as short as "v2" must be a whole
    # JSON value, or the start of one, so that it is not found inside another word.
    prompt = str(batch.get("expected_prompt_version") or "").split(":")[0]
    model = str(batch.get("expected_model_id") or "").split("@")[0]
    found = {"expected_prompt_version": f'"{prompt}"' in said or f'"{prompt}:' in said,
             "expected_model_id": model in said}
    for key, what in (("expected_prompt_version", "prompt version"), ("expected_model_id", "model")):
        want = str(batch.get(key) or "")
        if want and not found[key]:
            print(f"WARNING: /health does not mention the expected {what} {want!r}: {said[:300]}", file=sys.stderr)

    run_path = d / "vlm_run.json"
    log = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {"runs": []}
    this = {"started_utc": utc_now(), "url": a.url, "health": health, "runner_version": RUNNER_VERSION,
            "host": platform.node(), "python": platform.python_version(), "profile": a.profile,
            "min_interval_s": a.min_interval, "batch_check": "skipped" if a.skip_batch_check else "verified"}
    resp_path = d / "responses.jsonl"
    fixed = repair_tail(resp_path)
    if fixed:
        print(f"responses.jsonl: {fixed}")
        this["repaired"] = fixed
    log["runs"].append(this)
    log["health"] = health
    write_json(run_path, log)

    reqs = [json.loads(x) for x in (d / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.only_kind:
        reqs = [r for r in reqs if r.get("kind") == a.only_kind]
    done = answered(resp_path)
    todo = [r for r in reqs if r["request_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(reqs)} requests, {len(done & {r['request_id'] for r in reqs})} already answered, "
          f"{len(todo)} to send now (about {fmt_h(len(todo) * 14)} at ~14 s each)", flush=True)
    sent = ok = failed_in_row = not_confirmed = 0
    pacer = Pacer(a.min_interval)
    t_start = time.monotonic()
    code = None
    try:
        with open(resp_path, "a", encoding="utf-8", newline="\n") as out:
            for req in todo:
                line = send_one(d, req, base + "/classify", headers, a, pacer)
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                sent += 1
                ok += line["error"] is None
                if a.profile and line["error"] is None and line["response"].get("profile") != a.profile:
                    # An older server ignores the field and runs its default: valid answers, no speed-up.
                    not_confirmed += 1
                    if not_confirmed == 1:
                        print(f"WARNING: the server did not confirm profile {a.profile!r}; it ran its default",
                              file=sys.stderr, flush=True)
                failed_in_row = 0 if line["error"] is None else failed_in_row + 1
                per = (time.monotonic() - t_start) / sent
                state = "ok" if line["error"] is None else line["error"]
                print(f"[{sent}/{len(todo)}] {req['request_id']} {state} {line['seconds'] or 0:.1f} s | "
                      f"{per:.1f} s/image | left ~{fmt_h(per * (len(todo) - sent))}", flush=True)
                if failed_in_row >= a.max_failures:
                    raise Stop(f"{failed_in_row} images in a row failed (last: {line['error']}); is the server up?", 1)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        this["stopped"] = str(e)
        code = e.exit_code
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl-C): every answer so far is saved; run the same command again to continue",
              file=sys.stderr)
        this["interrupted"] = True
        code = 130
    this["finished_utc"] = utc_now()
    this["sent"], this["ok"] = sent, ok
    if a.profile:
        this["profile_not_confirmed"] = not_confirmed
    write_json(run_path, log)
    remaining = len([r for r in reqs if r["request_id"] not in answered(resp_path)])
    print(f"done: {ok} answered in this run, {remaining} still to do" + (" (run again to retry)" if remaining else ""))
    if code is not None:
        return code
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
