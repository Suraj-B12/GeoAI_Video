"""Command line: `geoai <command>` (or `python -m geoai <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .util import setup_logging, to_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geoai", description="Offline extraction of non-healthy pavement for the VLM")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--config", type=Path, help="JSON file overriding defaults (see geoai/config.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="sessions -> geotagged crops")
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--allow-test", action="store_true", help="also process `test` sessions")
    p.add_argument("--allow-untrusted", action="store_true", help="process sessions with mock locations")

    p = sub.add_parser("vlm-export", help="runs -> VLM batch folder for the USB drive")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--no-frames", action="store_true", help="don't send the full frame of each crop")
    p.add_argument("--include-over-budget", action="store_true")

    p = sub.add_parser("vlm-import", help="answered VLM batch -> validated results")
    p.add_argument("batch", type=Path)
    p.add_argument("runs", nargs="+", type=Path)

    p = sub.add_parser("simulate", help="write a synthetic session with known distresses")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=float, default=80.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)

    p = sub.add_parser("evaluate", help="score a run against a simulator truth.json")
    p.add_argument("run", type=Path)
    p.add_argument("truth", type=Path)

    a = ap.parse_args(argv)
    setup_logging(a.verbose)
    cfg = load_config(a.config)

    if a.cmd == "process":
        from .extract import process_session
        if a.device:
            cfg.anomaly.device = a.device
        status = 0
        for s in a.sessions:
            try:
                rep = process_session(s, a.out, cfg, allow_test=a.allow_test, allow_untrusted=a.allow_untrusted)
                print(f"{rep['session_id']}: {rep['records']}  (~{rep['estimated_vlm_hours']:.1f} h of VLM time)")
            except Exception as e:  # noqa: BLE001 - report and continue with the next session
                print(f"{s}: FAILED: {e}", file=sys.stderr)
                status = 1
        return status
    if a.cmd == "vlm-export":
        from .vlm import export_batch
        statuses = ("sent", "over_budget") if a.include_over_budget else ("sent",)
        b = export_batch(a.runs, a.batch, cfg, include_frames=False if a.no_frames else None, statuses=statuses)
        print(f"batch {b['batch_id']}: {b['n_crops']} crops + {b['n_frames']} frames -> {a.batch}")
        return 0
    if a.cmd == "vlm-import":
        from .vlm import import_batch
        s = import_batch(a.batch, a.runs, cfg)
        print(to_json({k: s[k] for k in ("requests", "answered", "valid", "missing", "problems", "model_matches")}))
        return 0
    if a.cmd == "simulate":
        from .sim import SimSpec, simulate_session
        r = simulate_session(a.out, SimSpec(seed=a.seed, length_m=a.length, width=a.width, height=a.height))
        print(r["session_dir"])
        return 0
    if a.cmd == "evaluate":
        from .evaluate import evaluate
        ev = evaluate(a.run, json.loads(a.truth.read_text(encoding="utf-8")))
        print(to_json(ev))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
