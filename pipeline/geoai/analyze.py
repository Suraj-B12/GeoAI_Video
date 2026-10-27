"""Survey results of a segment-mode VLM batch (docs/pipeline/13 §6).

`analyze(BATCH)` reads only the batch folder: every segment record (records/), the requests and the
VLM's answers (requests.jsonl, responses.jsonl), the runner's log (vlm_run.json) and each run's
report.json (runs/). It needs numpy and pandas, but no GPU and no model, so it runs on the VLM machine
right after the runner. It can be run again whenever more answers arrive: an unfinished batch is
analysed as far as it goes, and the rest is counted as not answered.

Definitions (repeated in REPORT.md, because every number depends on them):
- Surveyed length: the summed length of covered segments (send, audit, skip). `not_covered`
  segments were never imaged and are reported separately. Lengths are real lengths: a partial last
  segment counts with its own length, not 5 m.
- VLM classes come from the answer's labels, each checked against the 18 IRC:82-2015 names and
  Normal:
  - "distress": at least one valid non-Normal label (the definition gate calibration uses, §4);
  - "normal": valid labels, Normal only;
  - "unknown": answered, but no valid label.
  A segment with several distress labels counts once for each of them in the label rows, and once
  in the "Any distress" row.
- Classified length: segments of class distress or normal.
- Gate positive: decision_if_gated == "send", the gate's own rule (send-all runs record it too).
- The VLM is the reference here, not ground truth. On the VLM team's Attain test split, its Stage 1
  found 51.7 % of distressed frames (response notes 5.1).

Outputs in BATCH/analysis/: segments_results.csv / .geojson, label_summary.csv, per_km.csv,
gate_eval.json, vlm_timing.json, REPORT.md. Each is a deterministic function of the batch (REPORT.md
also names the software versions it ran with).
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .config import Config
from .qwen import frame_input_size
from .segment_records import (SEGMENT_COLUMNS, SENT_STATUSES, segment_feature, segment_row, write_feature_collection,
                              write_table_csv)
from .util import write_json_atomic
from .vlm import (SECONDS_PER_IMAGE, Answers, answer_class, health_identity, load_batch_records, read_answers,
                  verify_sums)

Z95 = 1.959963984540054        # two-sided 95 % normal quantile
Z_BONF = 2.241402727604947     # two-sided 97.5 %: two such intervals hold together with >= 95 % (Bonferroni)
KNOWN = ("distress", "normal")  # classes with a usable VLM answer
CATEGORY_ROWS = (("distress", "Any distress"), ("normal", "Normal"), ("unknown", "No valid label"),
                 ("not_answered", "Not answered"), ("not_sent", "Not sent (gate skip)"))
RUNNER_MIN_INTERVAL_S = 6.5    # run_vlm_batch.py default: 10 requests per minute at most
LIMITS = ("After the VLM's resize, along-track ground resolution is about 2.5 mm/px at 1.5 m ahead, 4 mm at "
          "2 m, 13 mm at 4 m and 25-30 mm at 6-7 m (S21, 23 deg tilt). Potholes, patches, ravelling, alligator "
          "and wide cracks stay visible; hairline and fine cracks beyond about 2 m do not.")
REFERENCE = ("The VLM (Qwen2.5-VL-7B, zero-shot, IRC:82-2015 prompts) is the reference, not ground truth: on "
             "the VLM team's Attain test split its Stage 1 found 51.7 % of distressed frames (response notes 5.1).")

RESULT_COLUMNS: tuple[tuple[str, int | None], ...] = tuple(c for c in SEGMENT_COLUMNS if not c[0].startswith("vlm_")) + (
    ("gate_positive", None), ("requested", None), ("answered", None), ("vlm_class", None), ("vlm_missing", None),
    ("vlm_valid", None), ("vlm_problem", None), ("vlm_labels", None), ("vlm_distress_labels", None),
    ("vlm_stage1_label", None), ("vlm_stage1_confidence", 4), ("vlm_severity", None), ("vlm_stage2_confidence", 4),
    ("vlm_needs_expert_review", None), ("vlm_description", None), ("vlm_batch_id", None),
)
SUMMARY_COLUMNS: tuple[tuple[str, int | None], ...] = (
    ("scope", None), ("kind", None), ("label", None), ("segments", None), ("length_m", 3), ("surveyed_m", 3),
    ("classified_m", 3), ("pct_of_surveyed", 4), ("pct_of_classified", 4),
)
PER_KM_BIN_M = 1000.0


# ---- statistics ----------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval of a proportion k/n (None for n = 0).

    Chosen over the normal (Wald) interval because the counts here are small and the proportions often
    near 0 or 1, where Wald collapses to zero width.
    """
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(k: int, n: int) -> dict:
    """{k, n, estimate, ci95}: a proportion with its Wilson 95 % interval (estimate and ci95 null for n = 0)."""
    ci = wilson(k, n)
    return {"k": int(k), "n": int(n), "estimate": k / n if n else None, "ci95": None if ci is None else list(ci)}


def _fsum(values: Iterable[float]) -> float:
    return math.fsum(float(v) for v in values)


def _pct(part: float, whole: float) -> float | None:
    return 100.0 * part / whole if whole > 0 else None


# ---- loading -------------------------------------------------------------------------------------

@dataclass
class BatchData:
    """Everything `analyze` uses: batch.json, the records by id, the answers, one row per segment.

    rows keep the values' own types and are what the files are written from; `table` is the same rows
    as a DataFrame for selecting and summing (pandas turns integer columns with nulls into floats, so it
    is never written out).
    """
    batch_dir: Path
    batch: dict
    records: dict[str, dict]
    answers: Answers
    rows: list[dict]
    table: pd.DataFrame


def load_batch(batch_dir: Path, cfg: Config) -> BatchData:
    """Read a segment batch and join every record with its VLM answer (if any).

    Checks the batch's checksums except the images (the numbers depend only on records, requests and
    answers; the runner checked each image as it uploaded it). Raises ValueError for a crop batch, a
    changed file or a batch without records.
    """
    batch_dir = Path(batch_dir)
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    if batch.get("mode") != "segments":
        raise ValueError(f"{batch_dir} is not a segment-mode batch (analyze needs `geoai process --mode segments` runs)")
    bad = verify_sums(batch_dir, skip_prefixes=("images/",))
    if bad:
        raise ValueError(f"batch files changed since export: {bad[:5]}")
    records = load_batch_records(batch_dir)
    if not records:
        raise ValueError(f"{batch_dir} has no segment records")
    ans = read_answers(batch_dir, cfg)
    allowed = set(cfg.vlm.labels)
    not_exported = {d["id"]: d["reason"] for d in batch.get("not_exported") or []}
    rows = [_segment_state(rec, ans, allowed, not_exported, batch_dir.name) for rec in records.values()]
    return BatchData(batch_dir, batch, records, ans, rows, pd.DataFrame(rows))


def _segment_state(rec: Mapping[str, Any], ans: Answers, allowed: set[str], not_exported: Mapping[str, str],
                   batch_id: str) -> dict:
    """The record's table row (segment_row) with its VLM class and answer."""
    rid, status = rec["id"], rec["status"]
    res = ans.results.get(rid)
    distress: list[str] = []
    missing = None
    if status == "not_covered":
        cls = "not_covered"
    elif status == "skip":
        cls = "not_sent"
    elif res is None:
        cls = "not_answered"
        if rid in not_exported:
            missing = f"not exported: {not_exported[rid]}"
        elif rid not in ans.requests:
            missing = "not requested in this batch"
        elif rid in ans.errors:
            missing = f"error: {ans.errors[rid]}"
        else:
            missing = "no answer yet"
    else:
        cls, distress = answer_class(res, allowed)
    r = res or {}
    gate = rec.get("gate")
    row = segment_row(rec)
    row.update({
        "covered": status != "not_covered", "sent": status in SENT_STATUSES, "requested": rid in ans.requests,
        "answered": res is not None, "gate_positive": None if gate is None else gate["decision_if_gated"] == "send",
        "vlm_class": cls, "vlm_missing": missing, "vlm_valid": r.get("valid"), "vlm_problem": r.get("problem"),
        "vlm_labels": r.get("labels"), "vlm_distress_labels": distress if res is not None else None,
        "vlm_stage1_label": r.get("stage1_label"), "vlm_stage1_confidence": r.get("stage1_confidence"),
        "vlm_severity": r.get("severity"), "vlm_stage2_confidence": r.get("stage2_confidence"),
        "vlm_needs_expert_review": r.get("needs_expert_review"), "vlm_description": r.get("description"),
        "vlm_batch_id": batch_id if res is not None else None, "vlm_seconds": r.get("seconds"),
    })
    return row


def _scopes(t: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """("all", every segment), then each session in id order."""
    return [("all", t)] + [(sid, t[t["session_id"] == sid]) for sid in sorted(t["session_id"].unique())]


def _distress_labels(labels: Sequence[str]) -> list[str]:
    return [lab for lab in labels if lab != "Normal"]


def label_key(label: str) -> str:
    """Column-safe name of a label: "Alligator Cracking" -> "alligator_cracking"."""
    return label.lower().replace(" ", "_")


# ---- tables --------------------------------------------------------------------------------------

def label_summary(t: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Per IRC label and per class: segments, length and % of surveyed and classified length.

    One block for all sessions ("all"), then one per session. Label rows list every distress label of
    the IRC list (zeros included, so tables of different sessions line up); category rows partition the
    surveyed segments, so their segments and pct_of_surveyed add up to the totals.
    """
    out = []
    for scope, d in _scopes(t):
        cov = d[d["covered"]]
        surveyed = _fsum(cov["length_m"])
        classified = _fsum(cov.loc[cov["vlm_class"].isin(KNOWN), "length_m"])
        for lab in _distress_labels(labels):
            has = cov["vlm_distress_labels"].map(lambda ls, lab=lab: isinstance(ls, list) and lab in ls)
            out.append(_summary_row(scope, "label", lab, cov[has], surveyed, classified, True))
        for cls, name in CATEGORY_ROWS:
            out.append(_summary_row(scope, "category", name, cov[cov["vlm_class"] == cls], surveyed, classified,
                                    cls in KNOWN))
    return pd.DataFrame(out, columns=[c for c, _ in SUMMARY_COLUMNS])


def _summary_row(scope: str, kind: str, label: str, sel: pd.DataFrame, surveyed: float, classified: float,
                 of_classified: bool) -> dict:
    length = _fsum(sel["length_m"])
    return {"scope": scope, "kind": kind, "label": label, "segments": int(len(sel)), "length_m": length,
            "surveyed_m": surveyed, "classified_m": classified, "pct_of_surveyed": _pct(length, surveyed),
            "pct_of_classified": _pct(length, classified) if of_classified else None}


def per_km(t: pd.DataFrame, labels: Sequence[str], bin_m: float = PER_KM_BIN_M) -> pd.DataFrame:
    """Lengths per class and label in consecutive 1 km stretches of each session.

    Stretches start at the session's first segment (from_m / to_m: path distance from there; the last
    stretch is as long as the road that is left). A segment that straddles a boundary is split by
    length, so every column sums to the session total. Chainage at the stretch ends is given when the
    records carry one.
    """
    dist = _distress_labels(labels)
    rows = []
    for sid in sorted(t["session_id"].unique()):
        d = t[t["session_id"] == sid].sort_values("k")
        base, end = float(d["s0_m"].min()), float(d["s1_m"].max())
        n_bins = max(1, int(math.ceil((end - base) / bin_m - 1e-9)))
        acc = [defaultdict(list) for _ in range(n_bins)]
        for r in d.itertuples(index=False):
            a, b = r.s0_m - base, r.s1_m - base
            for i in range(int(a // bin_m), min(n_bins, int(math.ceil(b / bin_m - 1e-12)))):
                ov = min(b, (i + 1) * bin_m) - max(a, i * bin_m)
                if ov <= 0:
                    continue
                for key in _length_keys(r):
                    acc[i][key].append(ov)
        chain = _chainage_map(d, base)
        for i, a in enumerate(acc):
            lo, hi = i * bin_m, min((i + 1) * bin_m, end - base)
            row = {"session_id": sid, "km": i, "from_m": lo, "to_m": hi,
                   "chainage_from_km": chain(lo) if chain else None, "chainage_to_km": chain(hi) if chain else None}
            for key in ("surveyed_m", "not_covered_m", "sent_m", "classified_m", "distress_m", "normal_m", "unknown_m",
                        "not_answered_m", "not_sent_m", *(f"{label_key(lab)}_m" for lab in dist)):
                row[key] = math.fsum(a[key])
            row["pct_distress_of_surveyed"] = _pct(row["distress_m"], row["surveyed_m"])
            row["pct_distress_of_classified"] = _pct(row["distress_m"], row["classified_m"])
            rows.append(row)
    return pd.DataFrame(rows, columns=[c for c, _ in per_km_columns(labels)])


def per_km_columns(labels: Sequence[str]) -> tuple[tuple[str, int | None], ...]:
    """Columns of per_km.csv, with the decimals written."""
    fixed = (("session_id", None), ("km", None), ("from_m", 3), ("to_m", 3), ("chainage_from_km", 6),
             ("chainage_to_km", 6), ("surveyed_m", 3), ("not_covered_m", 3), ("sent_m", 3), ("classified_m", 3),
             ("distress_m", 3), ("normal_m", 3), ("unknown_m", 3), ("not_answered_m", 3), ("not_sent_m", 3))
    return fixed + tuple((f"{label_key(lab)}_m", 3) for lab in _distress_labels(labels)) + \
        (("pct_distress_of_surveyed", 4), ("pct_distress_of_classified", 4))


def _length_keys(r) -> list[str]:
    """The per_km columns a segment's length counts towards."""
    if not r.covered:
        return ["not_covered_m"]
    keys = ["surveyed_m", f"{r.vlm_class}_m"]
    if r.sent:
        keys.append("sent_m")
    if r.vlm_class in KNOWN:
        keys.append("classified_m")
    if r.vlm_class == "distress":
        keys += [f"{label_key(lab)}_m" for lab in r.vlm_distress_labels]
    return keys


def _chainage_map(d: pd.DataFrame, base: float):
    """x (m from the session's first segment) -> chainage km, from the first record that has one; None if none do."""
    ref = d[d["chainage_start_km"].notna()]
    if ref.empty:
        return None
    r = ref.iloc[0]
    sign = 1.0 if r["chainage_end_km"] > r["chainage_start_km"] else -1.0
    return lambda x: float(r["chainage_start_km"] + sign * (base + x - r["s0_m"]) / 1000.0)


def gate_eval(t: pd.DataFrame) -> dict:
    """The gate against the VLM, on the segments where that can be measured without bias.

    send_all (runs with gate.mode = all): every covered segment was sent, so recall and precision of the
    gate's would-be decision come straight from the confusion matrix of the classified segments.
    gated (runs with gate.mode = gate): precision from the sent segments; recall needs the distress the
    gate rejected, which only the random audit sample shows: its distress share m estimates the share
    among all rejected segments (method in recall.method). Skipped segments count as gate-negative.
    Proportions carry Wilson 95 % intervals; segments without a usable answer are left out and counted.
    """
    cov = t[t["covered"]]
    out = {"reference": REFERENCE,
           "definitions": {
               "vlm_distress": "at least one valid non-Normal IRC:82 label in the VLM's answer",
               "classified": "the answer has at least one valid label (distress or Normal); others are not_classified",
               "gate_positive": "gate.decision_if_gated == 'send' (the gate's threshold rule)",
               "intervals": "Wilson score, 95 %"},
           "send_all": _eval_send_all(cov[cov["gate_mode"] == "all"]),
           "gated": _eval_gated(cov[cov["gate_mode"] == "gate"]),
           "per_session": {}}
    for sid in sorted(cov["session_id"].unique()):
        d = cov[cov["session_id"] == sid]
        for mode in sorted(d["gate_mode"].unique()):
            dm = d[d["gate_mode"] == mode]
            out["per_session"].setdefault(sid, {})[mode] = _eval_send_all(dm) if mode == "all" else _eval_gated(dm)
    return out


def _eval_send_all(d: pd.DataFrame) -> dict:
    known = d[d["vlm_class"].isin(KNOWN)]
    pos = known["gate_positive"].astype(bool).to_numpy()
    vd = (known["vlm_class"] == "distress").to_numpy()
    tp, fp = int((pos & vd).sum()), int((pos & ~vd).sum())
    fn, tn = int((~pos & vd).sum()), int((~pos & ~vd).sum())
    return {"sessions": sorted(d["session_id"].unique().tolist()), "segments": int(len(d)),
            "classified": int(len(known)), "not_classified": int(len(d) - len(known)),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "recall": proportion(tp, tp + fn), "precision": proportion(tp, tp + fp),
            "pass_rate": proportion(int(d["gate_positive"].astype(bool).sum()), int(len(d))),
            "vlm_distress_rate": proportion(tp + fn, int(len(known)))}


def _eval_gated(d: pd.DataFrame) -> dict:
    positive = d["gate_positive"].astype(bool)
    pos, neg = d[positive], d[~positive]
    audit, skip = neg[neg["status"] == "audit"], neg[neg["status"] == "skip"]
    pk, ak = pos[pos["vlm_class"].isin(KNOWN)], audit[audit["vlm_class"].isin(KNOWN)]
    kp, np_ = int((pk["vlm_class"] == "distress").sum()), int(len(pk))
    km, nm = int((ak["vlm_class"] == "distress").sum()), int(len(ak))
    miss = proportion(km, nm)
    skip_m = _fsum(skip["length_m"])
    missed = None
    if miss["estimate"] is not None:
        est, (lo, hi) = miss["estimate"], miss["ci95"]
        missed = {"segments": {"estimate": est * len(skip), "ci95": [lo * len(skip), hi * len(skip)]},
                  "length_m": {"estimate": est * skip_m, "ci95": [lo * skip_m, hi * skip_m]},
                  "note": "skipped segments x the distress share of the answered audit sample"}
    return {"sessions": sorted(d["session_id"].unique().tolist()), "segments": int(len(d)),
            "gate_positive": int(len(pos)), "gate_negative": int(len(neg)), "audited": int(len(audit)),
            "skipped": int(len(skip)), "audit_share_of_negatives": len(audit) / len(neg) if len(neg) else None,
            "sent_classified": np_, "audit_classified": nm,
            "pass_rate": proportion(int(len(pos)), int(len(d))), "precision": proportion(kp, np_),
            "miss_rate_among_rejected": miss, "recall": _gated_recall(len(pos), len(neg), kp, np_, km, nm),
            "estimated_missed_distress_in_skipped": missed}


def _gated_recall(n_pos: int, n_neg: int, kp: int, np_: int, km: int, nm: int) -> dict:
    """Recall of a gated run: P·p / (P·p + N·m), with its interval from the extremes of p and m.

    P, N: gate-positive and gate-negative segments; p: the VLM-distress share of the answered sent
    segments; m: that of the answered audit segments (a random sample of the negatives). The recall
    falls as m rises and rises with p, so the interval's ends are (p_lo, m_hi) and (p_hi, m_lo), with p
    and m from 97.5 % Wilson intervals: both hold at once with probability >= 95 % (Bonferroni).
    """
    method = ("recall = P*p / (P*p + N*m); P, N = gate-positive, gate-negative segments; p = VLM-distress share "
              "of answered sent segments; m = VLM-distress share of answered audit segments; interval from the "
              "97.5 % Wilson limits of p and m (Bonferroni, joint coverage >= 95 %)")
    if np_ == 0:
        return {"estimate": None, "ci95": None, "method": method}
    if n_neg == 0:  # nothing rejected, so nothing missed (undefined if the VLM saw no distress at all)
        return {"estimate": 1.0 if kp else None, "ci95": [1.0, 1.0] if kp else None, "method": method}
    if nm == 0:
        return {"estimate": None, "ci95": None, "method": method}
    p_lo, p_hi = wilson(kp, np_, Z_BONF)
    m_lo, m_hi = wilson(km, nm, Z_BONF)

    def recall(p: float, m: float) -> float | None:
        den = n_pos * p + n_neg * m
        return n_pos * p / den if den > 0 else None

    # Both ends have a positive denominator: a Wilson upper limit is > 0 for n > 0 (m_hi, p_hi > 0).
    return {"estimate": recall(kp / np_, km / nm), "ci95": [recall(p_lo, m_hi), recall(p_hi, m_lo)], "method": method}


def vlm_timing(ans: Answers) -> dict:
    """How long the VLM took: per answer, per runner session, and what is left.

    seconds_per_answer: the runner's wall time per answered request (upload, inference, response), so
    the server's own stage times are reported separately when its responses carry them (`timings`).
    The remaining time assumes the median per answer, but never less than the runner's 6.5 s spacing.
    """
    ok = [ln for ln in ans.lines if not ln.get("error") and ln.get("request_id") in ans.requests]
    errors = [ln for ln in ans.lines if ln.get("error")]
    secs = np.array([v for v in (_num(ln.get("seconds")) for ln in ok) if v is not None], dtype=np.float64)
    by_kind: dict[str, list[float]] = defaultdict(list)
    for ln in ok:
        if _num(ln.get("seconds")) is not None:
            by_kind[str(ans.requests[ln["request_id"]].get("kind"))].append(float(ln["seconds"]))
    err_types: dict[str, int] = defaultdict(int)
    for ln in errors:
        err_types[_error_type(str(ln["error"]))] += 1
    spans = [(t, t + (_num(ln.get("seconds")) or 0.0)) for ln in ans.lines if (t := _utc_s(ln.get("started_utc"))) is not None]
    server: dict[str, list[float]] = defaultdict(list)
    for ln in ok:
        timings = (ln.get("response") or {}).get("timings")
        if isinstance(timings, Mapping):
            for k, v in timings.items():
                if _num(v) is not None:
                    server[str(k)].append(float(v))
    sessions = [{k: run.get(k) for k in ("started_utc", "finished_utc", "sent", "ok", "interrupted", "runner_version",
                                         "host", "python")} for run in ans.run_info.get("runs") or []]
    durations = [_utc_s(s["finished_utc"]) - _utc_s(s["started_utc"]) for s in sessions
                 if _utc_s(s.get("started_utc")) is not None and _utc_s(s.get("finished_utc")) is not None]
    busy_h = math.fsum(durations) / 3600 if durations and len(durations) == len(sessions) else None
    remaining = sum(1 for rid in ans.requests if rid not in ans.results)
    per_image = max(float(np.median(secs)) if secs.size else SECONDS_PER_IMAGE, RUNNER_MIN_INTERVAL_S)
    return {
        "requests": len(ans.requests), "answered": len(ans.results), "remaining": remaining,
        "response_lines": len(ans.lines), "error_lines": len(errors), "errors_by_type": dict(sorted(err_types.items())),
        "seconds_per_answer": _stats(secs),
        "seconds_per_answer_by_kind": {k: _stats(np.array(v)) for k, v in sorted(by_kind.items())},
        "server_timings_median": {k: float(np.median(v)) for k, v in sorted(server.items())},
        "first_start_utc": _iso(min(s for s, _ in spans)) if spans else None,
        "last_end_utc": _iso(max(e for _, e in spans)) if spans else None,
        "span_h": (max(e for _, e in spans) - min(s for s, _ in spans)) / 3600 if spans else None,
        "runner_sessions": sessions, "runner_busy_h": busy_h,
        "answers_per_hour": len(ans.results) / busy_h if busy_h else None,
        "estimated_hours_remaining": remaining * per_image / 3600,
    }


def _stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "min": float(x.min()),
            "max": float(x.max()), "total_h": float(x.sum() / 3600)}


def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _utc_s(text) -> float | None:
    """Seconds since 1970 of the runner's "YYYY-mm-ddTHH:MM:SSZ" stamps (None if absent or malformed)."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error_type(err: str) -> str:
    """'HTTP 503' stays as is; "URLError(...)" -> "URLError"; other text up to its first colon."""
    if err.startswith("HTTP "):
        return err.split(":")[0].strip()
    head = err.split("(", 1)[0].split(":", 1)[0].strip()
    return head or "error"


# ---- report --------------------------------------------------------------------------------------

def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _f(v, nd: int = 1, unit: str = "") -> str:
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}{unit}"


def _ci(p: Mapping[str, Any] | None, scale: float = 100.0, nd: int = 1) -> str:
    """"12.3 [4.5, 30.1]" for an estimate with its interval, scaled to per cent; "-" when there is none."""
    if not p or p.get("estimate") is None:
        return "-"
    lo, hi = p["ci95"]
    return f"{p['estimate'] * scale:.{nd}f} [{lo * scale:.{nd}f}, {hi * scale:.{nd}f}]"


def _resolution_mm(rec: Mapping[str, Any]) -> float | None:
    """Mean along-track ground size of one model-input pixel over the segment (mm), for a sent segment.

    (far - near) metres span (v_bottom - v_top) frame rows; the VLM's processor scales rows by
    model_height / height (qwen.smart_resize). A computed counterpart of the spec's quoted figures.
    """
    f = rec.get("frame") or {}
    rows = f.get("rows_of_segment")
    if not rows or rows[1] <= rows[0]:
        return None
    _, mh = frame_input_size(f["width"], f["height"])
    return (f["far_edge_ahead_m"] - f["near_edge_ahead_m"]) * 1000.0 / ((rows[1] - rows[0]) * mh / f["height"])


def report_md(data: BatchData, summary: pd.DataFrame, km: pd.DataFrame, gate: dict, timing: dict, cfg: Config) -> str:
    """REPORT.md: methods, results and limits of the batch, written for the thesis."""
    t, b, recs = data.table, data.batch, list(data.records.values())
    ident = health_identity(data.answers.run_info, cfg)
    n_req, n_ans = len(data.answers.requests), len(data.answers.results)
    sums_sha = hashlib.sha256((data.batch_dir / "SHA256SUMS").read_bytes()).hexdigest()
    cov = t[t["covered"]]
    L: list[str] = [f"# Segment survey: VLM results of batch {data.batch_dir.name}", ""]
    if n_ans < n_req:
        L += [f"**Partial batch: {n_ans} of {n_req} requests answered.** Unanswered segments are counted as "
              "\"Not answered\"; run the runner again, then `geoai vlm-import` and `geoai analyze`.", ""]
    else:
        L += [f"All {n_req} requests answered.", ""]

    L += ["## Data", ""]
    rows = []
    for sid in sorted(t["session_id"].unique()):
        d = t[t["session_id"] == sid]
        r0 = data.records[d.iloc[0]["id"]]
        local = next((x for x in d["local_iso"] if isinstance(x, str)), None)
        sv = r0["survey"]
        rows.append([sid, local[:10] if local else "-", sv.get("road_id") or "-", sv.get("lane") or "-",
                     sv.get("direction") or "-", len(d), _f(_fsum(d.loc[d["covered"], "length_m"]), 0),
                     _f(_fsum(d.loc[~d["covered"], "length_m"]), 0), int(d["sent"].sum()), int(d["answered"].sum())])
    L += _md_table(["Session", "Date", "Road", "Lane", "Direction", "Segments", "Surveyed m", "Not covered m",
                    "Sent", "Answered"], rows)
    devices = sorted({f"{r['device'].get('manufacturer')} {r['device'].get('model')} ({r['device'].get('device_id')})"
                      for r in recs})
    caps = [r["capture"] for r in recs if r.get("capture")]
    exp_ms = [c["exposure_ns"] / 1e6 for c in caps if c.get("exposure_ns") is not None]
    isos = [c["iso"] for c in caps if c.get("iso") is not None]
    L += ["", f"- Device: {', '.join(devices)}.",
          f"- Exposure of the chosen frames: median {_f(float(np.median(exp_ms)) if exp_ms else None, 2)} ms, "
          f"ISO median {_f(float(np.median(isos)) if isos else None, 0)}; focus mode "
          f"{', '.join(sorted({str(c.get('focus_mode')) for c in caps})) or '-'}; "
          f"{sum(1 for c in caps if c.get('low_light'))} chosen frames under low-light exposure.", ""]

    L += ["## Methods", ""]
    lengths = sorted({round(r["segment"]["length_m"], 6) for r in recs if not r["segment"]["partial"]})
    sent_recs = [r for r in recs if r["status"] in SENT_STATUSES]
    near = [r["frame"]["near_edge_ahead_m"] for r in sent_recs]
    far = [r["frame"]["far_edge_ahead_m"] for r in sent_recs]
    res_mm = [v for v in (_resolution_mm(r) for r in sent_recs) if v is not None]
    modes = sorted({r["gate"]["mode"] for r in recs if r.get("gate")})
    thresholds = sorted({json.dumps(r["gate"]["thresholds"], sort_keys=True) for r in recs if r.get("gate")})
    L += [f"- **Segments.** The road is cut into segments of {', '.join(_f(x, 2, ' m') for x in lengths) or '-'} "
          "of GNSS path distance; a shorter last segment is kept when at least half as long and is weighted "
          "by its real length. For each segment the sharpest frame among those that show the whole segment "
          "nearest the camera is chosen (normalised Tenengrad of the segment's rows), avoiding predicted "
          "motion and rotation blur.",
          f"- **Viewing distance.** In the sent frames the segment's near edge was a median "
          f"{_f(float(np.median(near)) if near else None, 2)} m and its far edge {_f(float(np.median(far)) if far else None, 2)} m "
          "ahead of the camera. Mean along-track size of one model-input pixel over the segment: median "
          f"{_f(float(np.median(res_mm)) if res_mm else None, 1)} mm "
          f"(range {_f(min(res_mm) if res_mm else None, 1)}-{_f(max(res_mm) if res_mm else None, 1)} mm).",
          f"- **Gate.** Mode(s): {', '.join(modes) or '-'}. A segment is sent when any signal reaches its threshold "
          f"(thresholds used: {'; '.join(thresholds) or '-'}); a deterministic share of the rejected segments "
          "(SHA-256 of the record id) is sent as an audit sample.",
          f"- **VLM.** Expected model `{b.get('expected_model_id')}`, prompt version `{b.get('expected_prompt_version')}`. "
          f"Reported by GET /health: model `{ident['model_reported']}`, prompt `{ident['prompt_version_reported']}` "
          f"(model matches: {ident['model_matches']}, prompt matches: {ident['prompt_matches']}, same in every runner "
          f"session: {ident['consistent']}). Images: {b.get('image_rules')}",
          "- **Definitions.** Surveyed length = covered segments (send, audit, skip); not-covered segments are "
          "reported apart. VLM distress = at least one valid non-Normal IRC:82-2015 label; Normal = valid labels, "
          "Normal only; other answers have no valid label. A segment with several distress labels counts once "
          "per label in the label rows and once in \"Any distress\". Classified length = distress + Normal. "
          "Gate positive = the gate's own decision (recorded also when every segment was sent).",
          f"- **Reproducibility.** Pipeline {', '.join(sorted({r['pipeline_version'] for r in recs}))}; config SHA-256 "
          f"{', '.join(sorted({r['config_sha256'][:16] for r in recs}))}; batch `{b.get('batch_id')}` (SHA-256 of "
          f"SHA256SUMS: `{sums_sha}`); analysis with geoai {__version__}, Python {platform.python_version()}, "
          f"numpy {np.__version__}, pandas {pd.__version__}.", ""]

    L += ["## Results", "", "### Distress by length (all sessions)", ""]
    allrows = summary[summary["scope"] == "all"]
    shown = allrows[(allrows["kind"] == "category") | (allrows["segments"] > 0)]
    L += _md_table(["Label", "Segments", "Length m", "% of surveyed", "% of classified"],
                   [[r.label, r.segments, _f(r.length_m, 1), _f(r.pct_of_surveyed, 2), _f(r.pct_of_classified, 2)]
                    for r in shown.itertuples(index=False)])
    absent = allrows[(allrows["kind"] == "label") & (allrows["segments"] == 0)]["label"].tolist()
    total = allrows.iloc[0]
    L += ["", f"Surveyed {_f(total.surveyed_m, 1)} m, classified {_f(total.classified_m, 1)} m. "
          f"Labels not found: {', '.join(absent) if absent else 'none'}.", "", "### Per session", ""]
    prow = []
    for scope in sorted(set(summary["scope"]) - {"all"}):
        s = summary[(summary["scope"] == scope) & (summary["kind"] == "category")].set_index("label")
        prow.append([scope, _f(s.loc["Any distress", "surveyed_m"], 1), _f(s.loc["Any distress", "classified_m"], 1),
                     _f(s.loc["Any distress", "length_m"], 1), _f(s.loc["Any distress", "pct_of_surveyed"], 2),
                     _f(s.loc["Any distress", "pct_of_classified"], 2)])
    L += _md_table(["Session", "Surveyed m", "Classified m", "Distress m", "% of surveyed", "% of classified"], prow)
    L += ["", f"Per-km figures: `per_km.csv` ({len(km)} stretches).", "", "### Gate against the VLM", ""]
    sa, gt = gate["send_all"], gate["gated"]
    grows = []
    if sa["segments"]:
        grows.append(["Send-all runs", sa["segments"], sa["classified"], _ci(sa["recall"]), _ci(sa["precision"]),
                      _ci(sa["pass_rate"])])
    if gt["segments"]:
        grows.append(["Gated runs", gt["segments"], gt["sent_classified"] + gt["audit_classified"],
                      _ci(gt["recall"]), _ci(gt["precision"]), _ci(gt["pass_rate"])])
    L += _md_table(["Runs", "Segments", "Classified", "Recall % [95 % CI]", "Precision % [95 % CI]",
                    "Pass rate % [95 % CI]"], grows) if grows else ["No covered segments."]
    if gt["segments"]:
        L += ["", f"Gated runs: {gt['audited']} of {gt['gate_negative']} rejected segments were audited; the VLM found "
              f"distress in {_ci(gt['miss_rate_among_rejected'])} % of the answered audits. "
              + ("Estimated distress missed in the skipped segments: "
                 f"{_f(gt['estimated_missed_distress_in_skipped']['length_m']['estimate'], 1)} m "
                 f"[{_f(gt['estimated_missed_distress_in_skipped']['length_m']['ci95'][0], 1)}, "
                 f"{_f(gt['estimated_missed_distress_in_skipped']['length_m']['ci95'][1], 1)}]."
                 if gt["estimated_missed_distress_in_skipped"] else "No answered audit segment yet: the gate's misses are unknown.")]
    spa = timing["seconds_per_answer"]
    L += ["", "### VLM time", "",
          f"{timing['answered']} answers, {_f(spa.get('median'), 1)} s median per image "
          f"(p10 {_f(spa.get('p10'), 1)}, p90 {_f(spa.get('p90'), 1)}); {timing['error_lines']} error lines "
          f"{dict(timing['errors_by_type']) or ''}; about {_f(timing['estimated_hours_remaining'], 1)} h left.", ""]

    cov_m, nc_m = _fsum(cov["length_m"]), _fsum(t.loc[~t["covered"], "length_m"])
    h_sig = [r["camera"]["h_sigma_m"] for r in recs if r.get("camera")]
    L += ["## Limits", "",
          f"- **Resolution.** {LIMITS}",
          f"- **Reference.** {REFERENCE} Its labels are single answers per segment, not measurements of extent: "
          "\"% of length\" is the share of segments with the label, each segment counted with its full length.",
          "- **Gate.** In gate mode, skipped segments are not classified; the label shares are then lower bounds, and "
          "the audit sample estimates what the gate missed (above).",
          f"- **Coverage.** {_f(nc_m, 1)} m of road ({_f(_pct(nc_m, cov_m + nc_m), 1)} %) had no usable frame "
          "(GNSS gap, black frames, standing still) and is not in any percentage.",
          f"- **Position.** Positions are GNSS (median horizontal sigma {_f(float(np.median(h_sig)) if h_sig else None, 1)} m); "
          "chainage, where given, is the survey form's start chainage plus path distance, not corrected at km stones.",
          ""]
    return "\n".join(L)


# ---- gate calibration input ----------------------------------------------------------------------

def calibration_rows(batch_dirs: Sequence[Path], cfg: Config) -> tuple[list[dict], dict]:
    """The rows `gate.calibrate` needs, from answered segment batches, and where they came from.

    One row per segment the VLM answered with a usable class (distress or normal, `answer_class`):
    {"scores": the record's gate scores, "vlm_distress": class == "distress"}. Answers without a valid
    label, unanswered requests and segments without a gate block are left out and counted. A segment
    in several batches is used once, with the answer of the last batch given (a re-run batch replaces
    an earlier answer). The source block lists, per batch, the rows used by gate mode and status:
    rows from gate-mode runs over-represent gate-positive segments (only a random audit share of the
    negatives is sent), so a calibration should come from send-all runs; `gated_rows` makes a mixed
    input visible.
    """
    allowed = set(cfg.vlm.labels)
    by_id: dict[str, tuple[str, dict, dict]] = {}
    per_batch = []
    for b in batch_dirs:
        data = load_batch(Path(b), cfg)
        used = skipped_unknown = 0
        for row in data.rows:
            if not row["answered"]:
                continue
            rec = data.records[row["id"]]
            cls, _ = answer_class(data.answers.results[row["id"]], allowed)
            if cls not in KNOWN or rec.get("gate") is None:
                skipped_unknown += 1
                continue
            by_id[row["id"]] = (data.batch_dir.name, rec, {"scores": dict(rec["gate"]["scores"]),
                                                            "vlm_distress": cls == "distress"})
            used += 1
        per_batch.append({"batch_id": data.batch_dir.name, "requests": len(data.answers.requests),
                          "answered": len(data.answers.results), "usable": used, "no_valid_label": skipped_unknown})
    ids = sorted(by_id, key=lambda i: (by_id[i][1]["session_id"], by_id[i][1]["segment"]["k"]))
    rows = [by_id[i][2] for i in ids]
    composition: dict[str, int] = defaultdict(int)
    for i in ids:
        g = by_id[i][1]["gate"]
        composition[f"{g['mode']}:{by_id[i][1]['status']}"] += 1
    gated = sum(v for k, v in composition.items() if k.startswith("gate:"))
    source = {"batches": per_batch, "rows": len(rows), "duplicates_replaced": sum(p["usable"] for p in per_batch) - len(rows),
              "by_mode_and_status": dict(sorted(composition.items())), "gated_rows": gated,
              "sessions": sorted({by_id[i][1]["session_id"] for i in ids})}
    return rows, source


# ---- entry point ---------------------------------------------------------------------------------

def analyze(batch_dir: Path, cfg: Config | None = None) -> dict:
    """Write BATCH/analysis/ (see the module docstring) and return a short summary."""
    cfg = cfg or Config()
    data = load_batch(batch_dir, cfg)
    out = data.batch_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    t = data.table
    labels = list(cfg.vlm.labels)
    write_table_csv(out / "segments_results.csv", data.rows, RESULT_COLUMNS)
    names = [c for c, _ in RESULT_COLUMNS]
    write_feature_collection(out / "segments_results.geojson",
                             [segment_feature(data.records[r["id"]], {k: r[k] for k in names}) for r in data.rows])
    summary = label_summary(t, labels)
    write_table_csv(out / "label_summary.csv", summary.to_dict("records"), SUMMARY_COLUMNS)
    km = per_km(t, labels)
    write_table_csv(out / "per_km.csv", km.to_dict("records"), per_km_columns(labels))
    gate = gate_eval(t)
    write_json_atomic(out / "gate_eval.json", gate)
    timing = vlm_timing(data.answers)
    write_json_atomic(out / "vlm_timing.json", timing)
    (out / "REPORT.md").write_text(report_md(data, summary, km, gate, timing, cfg), encoding="utf-8", newline="\n")
    allrow = summary[(summary["scope"] == "all") & (summary["label"] == "Any distress")].iloc[0]
    return {"batch_id": data.batch_dir.name, "segments": int(len(t)), "requests": len(data.answers.requests),
            "answered": len(data.answers.results), "surveyed_m": float(allrow["surveyed_m"]),
            "classified_m": float(allrow["classified_m"]), "distress_m": float(allrow["length_m"]),
            "pct_distress_of_surveyed": _num(allrow["pct_of_surveyed"]), "output": str(out)}
