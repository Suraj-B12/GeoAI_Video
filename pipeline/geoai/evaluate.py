"""Scoring an extraction run against ground truth (simulator truth.json, or any list of polygons).

A record matches a feature when its anomaly centroid lies within `tol_m` of the feature's
footprint (positions compared in metres, local ENU).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymap3d
from shapely.geometry import Point, Polygon


def load_records(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(run_dir) / "sidecars").glob("*.json"))]


def evaluate(run_dir: Path, truth: dict, tol_m: float = 0.5, statuses=("sent", "over_budget")) -> dict:
    origin = truth["origin"]
    feats = []
    for i, f in enumerate(truth["features"]):
        lat, lon = np.array(f["polygon_latlon"]).T
        e, n, _ = pymap3d.geodetic2enu(lat, lon, np.zeros_like(lat), *origin)
        feats.append({"i": i, "kind": f["kind"], "is_distress": f["is_distress"],
                      "poly": Polygon(np.column_stack([e, n])).buffer(0)})
    recs = [r for r in load_records(run_dir) if r.get("status") in statuses]
    per_feature = {f["i"]: [] for f in feats}
    unmatched = []
    for r in recs:
        a = r.get("anomaly") or {}
        if a.get("lat") is None:
            continue
        e, n, _ = pymap3d.geodetic2enu(a["lat"], a["lon"], 0.0, *origin)
        pt = Point(float(e), float(n))
        hits = [f for f in feats if f["poly"].distance(pt) <= tol_m]
        if hits:
            best = min(hits, key=lambda f: f["poly"].distance(pt))
            per_feature[best["i"]].append(r)
        else:
            unmatched.append(r)
    distress = [f for f in feats if f["is_distress"]]
    detected = [f for f in distress if per_feature[f["i"]]]
    sent_detected = [f for f in distress if any(r["status"] == "sent" for r in per_feature[f["i"]])]
    lookalike_hits = sum(len(per_feature[f["i"]]) for f in feats if not f["is_distress"])
    errors = []
    for f in detected:
        for r in per_feature[f["i"]]:
            e, n, _ = pymap3d.geodetic2enu(r["anomaly"]["lat"], r["anomaly"]["lon"], 0.0, *origin)
            errors.append(f["poly"].distance(Point(float(e), float(n))))
    return {
        "distress_features": len(distress),
        "detected": len(detected),
        "detected_in_sent": len(sent_detected),
        "recall": len(detected) / max(len(distress), 1),
        "records_on_distress": sum(len(per_feature[f["i"]]) for f in distress),
        "records_on_lookalikes": lookalike_hits,
        "records_elsewhere": len(unmatched),
        "per_feature": {f"{f['i']}:{f['kind']}": [(r["status"], round(r["evidence"], 1)) for r in per_feature[f["i"]]]
                        for f in feats},
        "unmatched": [(r["id"], r["status"], round(r["evidence"], 1), round(r["anomaly"].get("ahead_m", 0), 2),
                       round(r["anomaly"].get("lateral_m", 0), 2)) for r in unmatched],
        "max_outside_distance_m": float(max(errors)) if errors else None,
    }
