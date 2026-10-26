#!/usr/bin/env python3
"""Copy finalised GeoAI Capture sessions from a USB-connected phone to this PC, offline.

For each session on the phone that has a manifest.json and is not yet on the PC:
  1. adb pull into <dest>/.incoming/<session_id>
  2. verify every file's size and SHA-256 against the manifest, plus the signature if the
     `cryptography` package is installed
  3. move it to <dest>/<session_id> (only verified sessions ever appear there)
  4. write pc_receipt.json next to it and push the same receipt to the phone, so the app shows
     the session as "VERIFIED_ON_PC" (safe to delete)

Nothing is deleted from the phone by this tool. Deletion happens in the app, which shows a separate
warning before deleting a session that has no matching receipt.

Requirements: Android platform-tools (adb) on PATH or --adb; USB debugging enabled and authorised.
Usage:
  python tools/pull_sessions.py D:/geoai/sessions
  python tools/pull_sessions.py D:/geoai/sessions --serial R5CR10XXXX --phone-root /sdcard/GeoAI/sessions
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOTS = [
    "/sdcard/GeoAI/sessions",  # "All files access" granted (normal)
    "/sdcard/Android/data/com.geoai.capture/files/GeoAI/sessions",  # fallback storage
]
TOOL = "pull_sessions.py v1"


def adb(args: list[str], adb_path: str, serial: str | None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [adb_path] + (["-s", serial] if serial else []) + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(d: Path) -> tuple[bool, str, int]:
    m_path = d / "manifest.json"
    m = json.loads(m_path.read_text(encoding="utf-8"))
    files = m.get("files", [])
    for f in files:
        p = d / f["path"]
        if not p.is_file():
            return False, f"missing {f['path']}", 0
        if p.stat().st_size != f["bytes"]:
            return False, f"size mismatch {f['path']}", 0
        if sha256_file(p) != f["sha256"]:
            return False, f"sha256 mismatch {f['path']}", 0
    sig = m.get("signature")
    note = "checksums ok; signature absent"
    if sig:
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec

            payload = {k: m[k] for k in ("spec_version", "session_id", "created_wall_ms", "files")}
            data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            key = serialization.load_der_public_key(base64.b64decode(sig["public_key_x509_b64"]))
            key.verify(base64.b64decode(sig["value_b64"]), data, ec.ECDSA(hashes.SHA256()))
            note = "checksums ok; signature valid"
        except ImportError:
            note = "checksums ok; signature not checked (pip install cryptography)"
        except Exception as e:  # noqa: BLE001  (InvalidSignature or malformed key)
            return False, f"signature invalid: {e.__class__.__name__}", 0
    return True, note, len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path, help="PC folder that holds verified sessions")
    ap.add_argument("--adb", default=shutil.which("adb") or "adb")
    ap.add_argument("--serial", help="device serial if several are connected")
    ap.add_argument("--phone-root", action="append", help="session folder on the phone (repeatable)")
    ap.add_argument("--no-receipt", action="store_true", help="do not push pc_receipt.json back to the phone")
    a = ap.parse_args()

    a.dest.mkdir(parents=True, exist_ok=True)
    incoming = a.dest / ".incoming"
    incoming.mkdir(exist_ok=True)
    roots = a.phone_root or DEFAULT_ROOTS
    try:
        adb(["get-state"], a.adb, a.serial)
    except Exception as e:  # noqa: BLE001
        print(f"adb device not available: {e}", file=sys.stderr)
        return 2

    pulled = 0
    failures = 0
    for root in roots:
        ls = adb(["shell", "ls", "-1", root], a.adb, a.serial, check=False)
        if ls.returncode != 0:
            continue
        for sid in [x.strip() for x in ls.stdout.splitlines() if x.strip()]:
            remote = f"{root}/{sid}"
            if adb(["shell", "ls", f"{remote}/manifest.json"], a.adb, a.serial, check=False).returncode != 0:
                print(f"skip {sid}: not finalised on the phone (finalise it in the app)")
                continue
            final = a.dest / sid
            if final.exists():
                print(f"skip {sid}: already on PC")
                continue
            tmp = incoming / sid
            if tmp.exists():
                shutil.rmtree(tmp)
            print(f"pull {sid} ...", flush=True)
            r = adb(["pull", remote, str(incoming)], a.adb, a.serial, check=False)
            if r.returncode != 0:
                print(f"  FAILED adb pull: {r.stderr.strip()[:200]}")
                failures += 1
                continue
            ok, note, n = verify(tmp)
            if not ok:
                print(f"  FAILED verification: {note} (left in {tmp})")
                failures += 1
                continue
            manifest_sha = sha256_file(tmp / "manifest.json")
            receipt = {
                "status": "verified",
                "session_id": sid,
                "manifest_sha256": manifest_sha,
                "files_verified": n,
                "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pc_host": platform.node(),
                "tool": TOOL,
                "note": note,
            }
            (tmp / "pc_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            os.replace(tmp, final)
            if not a.no_receipt:
                push = adb(["push", str(final / "pc_receipt.json"), f"{remote}/pc_receipt.json"], a.adb, a.serial, check=False)
                if push.returncode != 0:
                    print(f"  warning: receipt not pushed to phone: {push.stderr.strip()[:200]}")
            pulled += 1
            print(f"  ok: {n} files, {note}")
    print(f"done: {pulled} session(s) verified, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
