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

    p = sub.add_parser("process", help="sessions -> best frame per road segment (segments) or anomaly crops (crops)")
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("segments", "crops"), default="segments",
                   help="segments: one best full frame per segment (docs/pipeline/13); crops: anomaly crops")
    p.add_argument("--send-all", action="store_true",
                   help="segments: send every covered segment (gate.mode = all; calibration runs)")
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--allow-test", action="store_true", help="also process `test` sessions")
    p.add_argument("--allow-untrusted", action="store_true", help="process sessions with mock locations")

    p = sub.add_parser("vlm-export", help="runs -> VLM batch folder for the USB drive")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--no-frames", action="store_true", help="crop runs: don't send the full frame of each crop")
    p.add_argument("--include-over-budget", action="store_true", help="crop runs: also send over-budget crops")
    p.add_argument("--image", choices=("frame", "band"), default=None,
                   help="segment runs: send the whole frame or the segment's band (default: config vlm.image)")
    p.add_argument("--ids", type=Path, default=None,
                   help="segment runs: send only the record ids listed in this file (one per line)")

    p = sub.add_parser("vlm-import", help="answered VLM batch -> validated results")
    p.add_argument("batch", type=Path)
    p.add_argument("runs", nargs="*", type=Path,
                   help="run folders to update (crop batches: required; segment batches: optional)")

    p = sub.add_parser("analyze", help="segment batch -> survey tables, gate evaluation and REPORT.md (BATCH/analysis)")
    p.add_argument("batch", type=Path)

    p = sub.add_parser("gate-calibrate", help="answered segment batches -> gate thresholds (gate_calibration.json)")
    p.add_argument("batches", nargs="+", type=Path)
    p.add_argument("--target-recall", type=float, default=0.9,
                   help="share of VLM-distress segments the gate must keep (default 0.9)")
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: gate_calibration.json in the first batch folder)")

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
        return _process(a, cfg)
    if a.cmd == "vlm-export":
        from .vlm import export_batch, run_mode
        if {run_mode(r) for r in a.runs} == {"segments"}:
            if a.no_frames or a.include_over_budget:
                ap.error("--no-frames and --include-over-budget apply to crop runs only")
            if a.image:
                cfg.vlm.image = a.image
            ids = None
            if a.ids:
                ids = [x.strip() for x in a.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
            b = export_batch(a.runs, a.batch, cfg, ids=ids)
            print(f"batch {b['batch_id']}: {b['n_segments']} segment {b['image']}s {b['requests_by_status']} from "
                  f"{len(b['sessions'])} session(s), {b['image_bytes_total'] / 1e9:.2f} GB, "
                  f"~{b['estimated_vlm_hours']:.1f} h of VLM time -> {a.batch}")
            if b["not_exported"]:
                print(f"not exported: {b['not_exported']}", file=sys.stderr)
            return 0
        if a.image or a.ids:
            ap.error("--image and --ids apply to segment runs only")
        statuses = ("sent", "over_budget") if a.include_over_budget else ("sent",)
        b = export_batch(a.runs, a.batch, cfg, include_frames=False if a.no_frames else None, statuses=statuses)
        print(f"batch {b['batch_id']}: {b['n_crops']} crops + {b['n_frames']} frames -> {a.batch}")
        return 0
    if a.cmd == "vlm-import":
        from .vlm import import_batch
        mode = json.loads((a.batch / "batch.json").read_text(encoding="utf-8")).get("mode") or "crops"
        if mode != "segments" and not a.runs:
            ap.error("a crop batch needs the run folders its crops came from")
        s = import_batch(a.batch, a.runs, cfg)
        keys = ("requests", "answered", "valid", "missing", "problems", "model_matches")
        if mode == "segments":
            keys += ("prompt_matches", "classes", "errors_unresolved", "not_exported", "runs_updated", "run_problems")
        print(to_json({k: s[k] for k in keys}))
        return 0
    if a.cmd == "analyze":
        from .analyze import analyze
        print(to_json(analyze(a.batch, cfg)))
        return 0
    if a.cmd == "gate-calibrate":
        return _gate_calibrate(a, cfg)
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


def _process(a: argparse.Namespace, cfg) -> int:
    """`process`: each session in turn; a failed session is reported (and logged) and the next one runs."""
    if a.device:
        cfg.anomaly.device = a.device
    if a.send_all:
        if a.mode != "segments":
            print("--send-all applies to --mode segments only", file=sys.stderr)
            return 2
        cfg.gate.mode = "all"
    status = 0
    for s in a.sessions:
        try:
            if a.mode == "segments":
                from .frames import process_session_segments
                rep = process_session_segments(s, a.out, cfg, allow_test=a.allow_test, allow_untrusted=a.allow_untrusted)
                cov, g = rep["coverage"], rep["gate"]
                rate = "-" if g["pass_rate"] is None else f"{100 * g['pass_rate']:.1f} %"
                print(f"{rep['session_id']}: {rep['records']}  covered {cov['covered_m']:.0f} of {cov['planned_m']:.0f} m, "
                      f"gate pass rate {rate}, {rep['timings_s']['total']:.0f} s  "
                      f"(~{rep['estimated_vlm_hours']:.1f} h of VLM time)")
            else:
                rep = _process_crops(s, a, cfg)
                print(f"{rep['session_id']}: {rep['records']}  (~{rep['estimated_vlm_hours']:.1f} h of VLM time)")
        except Exception as e:  # noqa: BLE001 - report and continue with the next session
            print(f"{s}: FAILED: {e}", file=sys.stderr)
            status = 1
    return status


def _process_crops(session: Path, a: argparse.Namespace, cfg) -> dict:
    """Crop mode, with a run-level structured log (start, end or failure with its stack trace)."""
    from .extract import process_session
    from .runlog import RunLog

    rl = RunLog(a.out, Path(session).name, "process", mode="crops", session_dir=str(session))
    try:
        rep = process_session(session, a.out, cfg, allow_test=a.allow_test, allow_untrusted=a.allow_untrusted)
    except BaseException as e:
        rl.error("run", e)
        rl.finish(False, error_type=type(e).__name__, error=str(e))
        raise
    rl.finish(True, records=rep["records"], total_s=rep["timings_s"]["total"])
    return rep


def _gate_calibrate(a: argparse.Namespace, cfg) -> int:
    """`gate-calibrate`: thresholds from the VLM's answers (docs/pipeline/13 §4), written with their source."""
    from .analyze import calibration_rows
    from .gate import calibrate, write_calibration

    rows, source = calibration_rows(a.batches, cfg)
    if not rows:
        print("no answered segment with a valid label in these batches: nothing to calibrate", file=sys.stderr)
        return 1
    if source["gated_rows"]:
        print(f"WARNING: {source['gated_rows']} of {source['rows']} rows come from gate-mode runs, which over-represent "
              "gate-positive segments; calibrate on send-all runs (process --send-all)", file=sys.stderr)
    cal = calibrate(rows, a.target_recall)
    cal["source"] = source
    out = a.out or (a.batches[0] / "gate_calibration.json")
    write_calibration(out, cal)
    ao = cal["any_of"]
    rate = "-" if ao["pass_rate"] is None else f"{100 * ao['pass_rate']:.1f} %"
    rec = "-" if ao["recall"] is None else f"{100 * ao['recall']:.1f} %"
    print(f"{out}: {cal['n_rows']} segments, {cal['n_positive']} VLM-positive; any-of {ao['status']}: "
          f"thresholds {ao['thresholds']}, recall {rec}, pass rate {rate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
