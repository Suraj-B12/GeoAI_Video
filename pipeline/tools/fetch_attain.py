#!/usr/bin/env python3
"""Download the Attain pavement distress dataset (Mendeley Data, doi:10.17632/nykrzdm74f.1, CC BY 4.0).

Standard library only. Every file is checked against the SHA-256 that Mendeley publishes; files
already present with the right hash are skipped, so the script can be re-run to resume.

    python tools/fetch_attain.py data/attain

Layout: <dest>/<subset>/<file> (subset = Attain_SMP_WS_v2, ...); the class-list YAML files and
the dataset metadata (dataset.json) go in <dest>/. Attribution is written to <dest>/ATTRIBUTION.txt.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/nykrzdm74f"
# Mendeley refuses urllib's default User-Agent (HTTP 403).
HEADERS = {"User-Agent": "GeoAI-pipeline/0.1 (dataset download for research)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str, tries: int = 4) -> str:
    if dest.exists() and sha256(dest) == expected:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
            if sha256(tmp) != expected:
                raise ValueError("checksum mismatch")
            tmp.replace(dest)
            return "downloaded"
        except Exception as e:  # noqa: BLE001 - retried, then reported
            if tmp.exists():
                tmp.unlink()
            if attempt == tries - 1:
                return f"FAILED: {e}"
            time.sleep(2 * (attempt + 1))
    return "FAILED"


def subset_of(name: str) -> str | None:
    m = re.match(r"(Attain_[A-Z]+_[A-Z]+_v\d+)_", name)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    dest = a.dest
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(API, headers={**HEADERS, "Accept": "application/json"}), timeout=120) as r:
        meta = json.loads(r.read().decode("utf-8"))
    (dest / "dataset.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    files = meta["files"]
    names = [f["filename"] for f in files]
    if len(set(names)) != len(names):
        print("STOP: duplicate file names in the listing", file=sys.stderr)
        return 2
    jobs = []
    for f in files:
        sub = subset_of(f["filename"])
        target = dest / sub / f["filename"] if sub else dest / f["filename"]
        cd = f["content_details"]
        jobs.append((cd["download_url"], target, cd["sha256_hash"]))
    total = sum(f["size"] for f in files)
    print(f"{len(jobs)} files, {total / 1e6:.1f} MB, licence {meta['data_licence']['short_name']}")
    results = {"downloaded": 0, "skipped": 0, "failed": []}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = {ex.submit(fetch, *j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            res = fut.result()
            if res.startswith("FAILED"):
                results["failed"].append((futs[fut][1].name, res))
            else:
                results[res] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    (dest / "ATTRIBUTION.txt").write_text(
        "Attain: An inclusive pavement distress dataset (version 1)\n"
        "Rezaeimanesh, Golroo, Fahmani, Amani, Hasanitabar, Entezari, Karimi - Amirkabir University of Technology\n"
        "Mendeley Data, doi:10.17632/nykrzdm74f.1\n"
        "Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), http://creativecommons.org/licenses/by/4.0\n"
        "Used unchanged to train the GeoAI DINOv2 distress probe (pipeline/models/probe).\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s: {results['downloaded']} downloaded, {results['skipped']} already present, "
          f"{len(results['failed'])} failed")
    for name, why in results["failed"][:20]:
        print("  FAILED", name, why)
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
