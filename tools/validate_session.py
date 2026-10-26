#!/usr/bin/env python3
"""Validate a GeoAI Capture session folder against docs/session-package-spec.md.

Reference implementation of the ingest checks (spec section 14). Standard library only, plus:
  - ffprobe on PATH (for the video checks)
  - optional: the `cryptography` package (to verify the manifest signature)

Usage:
  python tools/validate_session.py <session_dir> [--json report.json]
Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = cannot read the session.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SUPPORTED_MAJOR = 1


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, level: str, check: str, detail: str) -> None:
        self.items.append({"level": level, "check": check, "detail": detail})

    def ok(self, check: str, detail: str = "") -> None:
        self.add("PASS", check, detail)

    def warn(self, check: str, detail: str) -> None:
        self.add("WARN", check, detail)

    def fail(self, check: str, detail: str) -> None:
        self.add("FAIL", check, detail)

    def info(self, check: str, detail: str) -> None:
        self.add("INFO", check, detail)

    @property
    def failed(self) -> bool:
        return any(i["level"] == "FAIL" for i in self.items)


def canonical(obj) -> bytes:
    """Same bytes as the app's CanonicalJson for strings/ints/bools/null/objects/arrays."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p: Path) -> list[dict]:
    """Reads a session CSV, dropping an incomplete last line (allowed after a crash)."""
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    return list(csv.DictReader(text.splitlines()))


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_manifest(d: Path, r: Report) -> None:
    mpath = d / "manifest.json"
    if not mpath.is_file():
        r.fail("manifest", "manifest.json missing (session not finalised; finalise it in the app)")
        return
    m = json.loads(mpath.read_text(encoding="utf-8"))
    listed = {f["path"]: f for f in m.get("files", [])}
    bad = []
    for rel, f in listed.items():
        p = d / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
        elif p.stat().st_size != f["bytes"]:
            bad.append(f"{rel}: size {p.stat().st_size} != {f['bytes']}")
        elif sha256_file(p) != f["sha256"]:
            bad.append(f"{rel}: sha256 mismatch")
    actual = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
    extra = sorted(actual - set(listed) - {"manifest.json", "pc_receipt.json"})
    if bad:
        r.fail("checksums", "; ".join(bad[:10]))
    else:
        r.ok("checksums", f"{len(listed)} files match manifest")
    if extra:
        r.warn("unlisted files", ", ".join(extra[:10]))

    sig = m.get("signature")
    if not sig:
        r.warn("signature", f"absent ({m.get('signature_error', 'no reason given')})")
        return
    payload = {k: m[k] for k in ("spec_version", "session_id", "created_wall_ms", "files")}
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError:
        r.info("signature", "present; install `cryptography` to verify it")
        return
    try:
        key = serialization.load_der_public_key(base64.b64decode(sig["public_key_x509_b64"]))
        key.verify(base64.b64decode(sig["value_b64"]), canonical(payload), ec.ECDSA(hashes.SHA256()))
        r.ok("signature", "ECDSA P-256 signature valid for the manifest payload")
    except InvalidSignature:
        r.fail("signature", "INVALID: manifest was modified after signing")


def ffprobe_packets(video: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,time_base",
         "-show_entries", "packet=pts,dts,flags", "-of", "json", str(video)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:300])
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    tb = Fraction(stream["time_base"])
    pkts = [p for p in data.get("packets", []) if p.get("pts") is not None]
    return stream, tb, pkts


def check_video(d: Path, frames: list[dict], r: Report) -> None:
    segs = sorted((d / "video").glob("seg_*.mp4"))
    if not segs:
        r.fail("video", "no video segments")
        return
    if shutil.which("ffprobe") is None:
        r.warn("video", "ffprobe not on PATH: video checks skipped")
        return
    by_seg: dict[int, list[dict]] = {}
    for row in frames:
        by_seg.setdefault(to_int(row["segment"]), []).append(row)
    for seg in segs:
        idx = int(seg.stem.split("_")[1])
        rows = by_seg.get(idx, [])
        try:
            stream, tb, pkts = ffprobe_packets(seg)
        except Exception as e:  # noqa: BLE001
            r.fail(f"{seg.name}", f"ffprobe failed: {e}")
            continue
        name = f"{seg.name} ({stream.get('codec_name')} {stream.get('width')}x{stream.get('height')})"
        pkts.sort(key=lambda p: int(p["pts"]))
        if len(pkts) != len(rows):
            # The last fragment of a crashed session may be lost: fewer packets than rows is a WARN.
            level = r.warn if len(pkts) < len(rows) else r.fail
            level(name, f"{len(pkts)} video samples but {len(rows)} frames.csv rows")
        else:
            r.ok(name, f"{len(pkts)} samples == frames.csv rows")
        if pkts and "K" not in pkts[0].get("flags", ""):
            r.fail(name, "first sample is not a keyframe")
        # Container PTS is a sample KEY, not a time: fragmented-MP4 muxers estimate the last sample
        # duration of each fragment, so file PTS drift slightly from the true timestamps.
        # Required: unique, strictly increasing PTS and a 1:1 sample <-> row mapping by order.
        pts = [int(p["pts"]) for p in pkts]
        if any(b <= a for a, b in zip(pts, pts[1:])):
            r.fail(f"{seg.name} PTS order", "container PTS not strictly increasing (B-frames or muxer error)")
        else:
            r.ok(f"{seg.name} PTS order", "strictly increasing, unique")
        n = min(len(pkts), len(rows))
        if n >= 2:
            p0 = pts[0]
            u0 = to_int(rows[0]["pts_us"])
            worst = max(abs((pts[i] - p0) * float(tb) * 1e6 - (to_int(rows[i]["pts_us"]) - u0)) for i in range(n))
            r.info(f"{seg.name} PTS drift", f"max {worst / 1000:.2f} ms vs frames.csv (expected; never use container PTS as time)")


def check_frames(frames: list[dict], r: Report) -> tuple[int | None, int | None]:
    if not frames:
        r.fail("frames.csv", "no rows")
        return None, None
    t = [to_int(f["t_ns"]) for f in frames]
    if any(v is None for v in t):
        r.fail("frames.csv t_ns", f"{sum(v is None for v in t)} rows without t_ns")
        t = [v for v in t if v is not None]
    inc = all(b > a for a, b in zip(t, t[1:]))
    (r.ok if inc else r.fail)("frames.csv t_ns", "strictly increasing" if inc else "NOT strictly increasing")
    unmatched = sum(1 for f in frames if not f["sensor_ts_ns"])
    if unmatched:
        r.warn("frame metadata join", f"{unmatched} of {len(frames)} frames have no camera metadata")
    else:
        r.ok("frame metadata join", f"all {len(frames)} frames matched")
    res = [abs(to_int(f["match_residual_ns"])) for f in frames if to_int(f["match_residual_ns"]) is not None]
    if res:
        mx = max(res)
        (r.ok if mx < 1_000_000 else r.warn)("match residual", f"max {mx / 1e3:.1f} us")
    gaps = [b - a for a, b in zip(t, t[1:])]
    if gaps:
        med = statistics.median(gaps)
        drops = [g for g in gaps if g > 1.5 * med]
        pct = 100.0 * len(drops) / len(gaps)
        (r.ok if pct < 0.1 else r.warn)("dropped frames", f"{len(drops)} gaps > 1.5x median ({med / 1e6:.2f} ms); {pct:.2f}%")
        r.info("frame rate", f"{1e9 / med:.2f} fps median")
    exp = [to_int(f["exposure_ns"]) for f in frames if to_int(f["exposure_ns"])]
    if exp:
        r.info("exposure", f"median 1/{1e9 / statistics.median(exp):.0f} s, longest 1/{1e9 / max(exp):.0f} s")
    return (t[0], t[-1]) if t else (None, None)


def check_gnss(d: Path, t_first: int | None, t_last: int | None, r: Report) -> None:
    fixes = read_csv(d / "gnss_fixes.csv")
    if not fixes:
        r.fail("gnss", "no fixes")
        return
    t = sorted(to_int(f["t_ns"]) for f in fixes if to_int(f["t_ns"]) is not None)
    mock = sum(1 for f in fixes if f.get("is_mock") == "1")
    (r.fail if mock else r.ok)("mock locations", f"{mock} mock fixes" if mock else "none")
    if t_first is not None:
        before = [x for x in t if x <= t_first]
        after = [x for x in t if x >= t_last]
        if before and after:
            r.ok("GNSS brackets video", "fixes exist before the first and after the last frame")
        else:
            r.warn("GNSS brackets video", "missing a fix before the first or after the last frame (interpolation edges)")
        inside = [x for x in t if t_first <= x <= t_last]
        span = (t_last - t_first) / 1e9
        if span > 0:
            r.info("GNSS rate", f"{len(inside) / span:.2f} fixes/s during recording")
    gaps = [(b - a) / 1e9 for a, b in zip(t, t[1:])]
    if gaps:
        mx = max(gaps)
        (r.ok if mx <= 2.0 else r.warn)("GNSS max gap", f"{mx:.2f} s")
    acc = sorted(to_float(f["h_acc_m"]) for f in fixes if to_float(f["h_acc_m"]) is not None)
    if acc:
        r.info("GNSS accuracy", f"median {acc[len(acc) // 2]:.1f} m (68%)")
    lag = [(to_int(f["receipt_ns"]) - to_int(f["t_ns"])) / 1e6 for f in fixes if to_int(f["receipt_ns"]) and to_int(f["t_ns"])]
    if lag:
        r.info("GNSS fix->receipt", f"median {statistics.median(lag):.1f} ms (0 ms suggests fixes are stamped on delivery)")


def check_imu(d: Path, r: Report) -> None:
    rows = read_csv(d / "imu.csv")
    if not rows:
        r.fail("imu", "no samples")
        return
    by: dict[str, list[int]] = {}
    for row in rows:
        v = to_int(row["t_ns"])
        if v is not None:
            by.setdefault(row["sensor"], []).append(v)
    parts = []
    for name, ts in sorted(by.items()):
        ts.sort()
        span = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0
        parts.append(f"{name} {len(ts) / span:.0f} Hz" if span > 0 else f"{name} n={len(ts)}")
    r.info("imu rates", ", ".join(parts))


def check_clock(d: Path, r: Report) -> None:
    rows = read_csv(d / "clock_sync.csv")
    offs = []
    for row in rows:
        a, m, b = to_int(row["t_ns_before"]), to_int(row["mono_ns"]), to_int(row["t_ns_after"])
        if None not in (a, m, b):
            offs.append((a + b) // 2 - m)
    if len(offs) >= 2:
        spread = (max(offs) - min(offs)) / 1e6
        (r.ok if spread < 1.0 else r.warn)("elapsed-monotonic offset", f"spread {spread:.3f} ms over {len(offs)} samples (non-zero = device slept)")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--json", type=Path, help="write the report as JSON")
    a = ap.parse_args()
    d = a.session_dir
    r = Report()
    try:
        session = json.loads((d / "session.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"cannot read session.json: {e}", file=sys.stderr)
        return 2
    major = int(str(session.get("spec_version", "0")).split(".")[0])
    (r.ok if major == SUPPORTED_MAJOR else r.fail)("spec_version", str(session.get("spec_version")))
    bs, be = session.get("boot_count_start"), session.get("boot_count_end")
    if be is None:
        r.warn("boot count", "no end boot count (session not finalised normally)")
    elif bs != be:
        r.fail("boot count", f"phone rebooted during the session ({bs} -> {be})")
    else:
        r.ok("boot count", "no reboot during the session")
    end = session.get("end", {})
    r.info("session", f"{session.get('session_id')} type={session.get('session_type')} end={end.get('reason')}")
    cam = session.get("camera", {})
    ts = cam.get("timestamp_source")
    (r.ok if ts == "REALTIME" else r.warn)("camera timestamp source", str(ts))
    cs = session.get("capture_settings", {})
    mode = cs.get("exposure_mode")
    (r.ok if mode == "MANUAL_SHUTTER_AUTO_ISO" else r.warn)("exposure mode", str(mode))
    r.info("streams", f"{cs.get('streams_while_recording')} timestamp base {cs.get('encoder_timestamp_base')}")
    orient = session.get("orientation", {}).get("verdict")
    (r.ok if orient == "UPRIGHT" else r.warn)("mount orientation", str(orient))

    check_manifest(d, r)
    frames = read_csv(d / "frames.csv")
    check_video(d, frames, r)
    t0, t1 = check_frames(frames, r)
    check_gnss(d, t0, t1, r)
    check_imu(d, r)
    check_clock(d, r)

    width = max(len(i["check"]) for i in r.items)
    for i in r.items:
        print(f"{i['level']:4}  {i['check']:<{width}}  {i['detail']}")
    if a.json:
        a.json.write_text(json.dumps({"session": str(d), "items": r.items}, indent=2), encoding="utf-8")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
