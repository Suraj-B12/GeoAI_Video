"""Extraction: a verified session -> geotagged crops of road surface that doesn't look healthy.

How the road is covered exactly once (no tracking needed):
- Each recording resolution gets a fixed analysis window: the image rows that show the road
  from `band_y0` to `band_y0 + band_length_m` ahead of (or behind) the camera, plus context.
- A frame is selected every `band_length_m` of travel (path distance s). Selected frame i is
  responsible for the road s in [s_i + band_y0, s_i + band_y0 + L), so consecutive bands abut.
- An anomaly found in frame i is kept only if its ground centroid lies in frame i's band (with
  a small tolerance at both edges).
- The same stretch of road seen from several frames (band edges, a long crack whose parts fall in
  different bands) is merged by non-maximum suppression on the ground: strongest evidence first, a
  record is a duplicate of a kept one if their ground footprints overlap by `dedupe_overlap` of the
  smaller (centroids within `dedupe_dist_m` when a record has no footprint). It is not transitive, so separate distresses
  stay separate however dense they are. No cap on the count unless `budget_per_km` is set.

Pass 1 builds the "normal pavement" bank from frames spread over the whole session (keyframes,
jittered spacing). Pass 2 decodes every frame, analyses the selected ones and writes crops.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from . import __version__
from .anomaly import AnomalyDetector, Candidate
from .config import Config
from .dinov2 import PATCH, PatchFeatures
from .geometry import CameraModel, nominal_model
from .qwen import CropPlan, frame_input_size, plan_crop
from .records import PngPool, unique_id, utc_strings, write_csv, write_geojson, write_sidecar
from .session import Session
from .trajectory import Trajectory, build_trajectory, clean_fixes
from .util import log, write_json_atomic
from .video import iter_segment, segment_info


class ExtractionError(RuntimeError):
    pass


# ---- per-resolution analysis view ------------------------------------------------------------

@dataclass
class View:
    width: int
    height: int
    model: CameraModel            # full resolution, display orientation
    v0: int                       # analysis window rows [v0, v1) of the full frame
    v1: int
    sx: float                     # analysis scale (analysis px per frame px)
    sy: float
    ana_w: int
    ana_h: int
    model_ana: CameraModel
    band_y0: float                # band start, metres ahead of the camera nadir (negative = behind)
    band_len: float
    gsd_rows_mm: np.ndarray       # along-track ground resolution per analysis row
    rotate180: bool
    notes: dict = field(default_factory=dict)

    def analysis_image(self, rgb: np.ndarray) -> np.ndarray:
        win = rgb[self.v0:self.v1]
        w = int(round(self.width * self.sx))
        h = int(round((self.v1 - self.v0) * self.sy))
        small = cv2.resize(win, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (win.shape[1], win.shape[0]) else win
        return np.ascontiguousarray(small[: self.ana_h, : self.ana_w])

    def to_frame_px(self, x, y):
        """Analysis-image pixel coordinates -> full-frame pixel coordinates."""
        return (np.asarray(x, float) + 0.5) / self.sx - 0.5, (np.asarray(y, float) + 0.5) / self.sy - 0.5 + self.v0

    def summary(self) -> dict:
        return {"width": self.width, "height": self.height, "window_rows": [self.v0, self.v1],
                "analysis_size": [self.ana_w, self.ana_h], "scale": [self.sx, self.sy],
                "band_y0_m": self.band_y0, "band_length_m": self.band_len,
                "gsd_mm_range": [float(np.nanmin(self.gsd_rows_mm)), float(np.nanmax(self.gsd_rows_mm))],
                "rotate180": self.rotate180, "camera": self.model.to_dict(), **self.notes}


def plan_view(model: CameraModel, cfg: Config, rotate180: bool) -> View:
    geo, ex = cfg.geometry, cfg.extract
    W, H = model.width, model.height
    vs = np.arange(0, H, 2, dtype=float)
    uc = np.full_like(vs, (W - 1) / 2)
    _, y, ok = model.pixel_to_ground(uc, vs, geo.max_range_m)
    along, _ = model.gsd_mm(uc, vs)
    _, _, okl = model.pixel_to_ground(np.zeros_like(vs), vs, geo.max_range_m * 1.6)
    _, _, okr = model.pixel_to_ground(np.full_like(vs, W - 1), vs, geo.max_range_m * 1.6)
    usable = ok & okl & okr & (along <= geo.max_gsd_mm) & (vs <= H * (1 - geo.bottom_exclude_frac) - 1)
    if not usable.any():
        raise ExtractionError("no usable road rows: check the mount pitch/height (camera may see no road)")
    yu = y[usable]
    forward = float(np.nanmedian(yu)) > 0
    y_near = float(np.nanmin(yu)) if forward else float(np.nanmax(yu))
    extent = float(np.nanmax(yu) - np.nanmin(yu))
    L = ex.band_length_m
    notes = {}
    if L + ex.context_m > 0.9 * extent:
        L = max(0.3, 0.9 * extent - ex.context_m)
        notes["band_length_reduced"] = f"usable road extent {extent:.2f} m; band reduced to {L:.2f} m"
        log.warning(notes["band_length_reduced"])
    band_y0 = y_near if forward else y_near - L
    lo, hi = band_y0 - ex.context_m, band_y0 + L + ex.context_m
    rows = vs[usable & (y >= lo) & (y <= hi)]
    v0 = int(max(0, rows.min() - 4))
    v1 = int(min(H, rows.max() + 6))
    med_gsd = float(np.nanmedian(along[usable & (y >= lo) & (y <= hi)]))
    # Downscale only when pixels are finer than the target (4K: ~1 mm/px -> 0.5); never upscale.
    ds = min(1.0, max(0.25, med_gsd / ex.target_gsd_mm))
    w = int(round(W * ds))
    h = int(round((v1 - v0) * ds))
    sx, sy = w / W, h / (v1 - v0)
    ana_w, ana_h = (w // PATCH) * PATCH, (h // PATCH) * PATCH
    if ana_w < 4 * PATCH or ana_h < 4 * PATCH:
        raise ExtractionError(f"analysis window too small ({ana_w}x{ana_h})")
    model_ana = model.cropped(0, v0, W, v1 - v0).scaled(sx, sy)
    model_ana = CameraModel(ana_w, ana_h, model_ana.K, model_ana.dist, model_ana.R_v_from_c, model_ana.C_v,
                            model_ana.source, model_ana.notes)
    rr = np.arange(ana_h, dtype=float)
    gsd_rows, _ = model_ana.gsd_mm(np.full_like(rr, (ana_w - 1) / 2), rr)
    gsd_rows = np.where(np.isfinite(gsd_rows), gsd_rows, np.nanmax(gsd_rows))
    notes.update({"forward_view": forward, "usable_extent_m": extent, "downscale": ds})
    return View(W, H, model, v0, v1, sx, sy, ana_w, ana_h, model_ana, band_y0, L, gsd_rows, rotate180, notes)


# ---- helpers ---------------------------------------------------------------------------------

def _median_gravity(session: Session) -> np.ndarray:
    imu = session.imu
    if not imu.empty and "sensor" in imu.columns:
        g = imu[imu["sensor"] == "gravity"]
        if not session.frames.empty:
            t0, t1 = int(session.frames["t_ns"].min()), int(session.frames["t_ns"].max())
            g = g[(g["t_ns"] >= t0) & (g["t_ns"] <= t1)]
        if len(g) >= 10:
            return np.median(g[["x", "y", "z"]].to_numpy(float), axis=0)
    gv = (session.meta.get("orientation") or {}).get("gravity_device_xyz")
    if gv:
        return np.asarray(gv, float)
    raise ExtractionError("no gravity data: cannot derive the camera attitude")


class _RowClock:
    """Exposure-middle time of a display row, in trajectory seconds (rolling shutter aware)."""

    def __init__(self, t0_ns: int, height: int, rotate180: bool):
        self.t0_ns = t0_ns
        self.h = height
        self.rot = rotate180

    def __call__(self, row: dict, display_row: float) -> float:
        r = (self.h - 1 - display_row) if self.rot else display_row
        return (int(row["t_ns"]) - self.t0_ns + _num0(row.get("exposure_ns")) / 2
                + _num0(row.get("rs_skew_ns")) * r / self.h) / 1e9


def _num0(x) -> float:
    """Float value, or 0 for missing (None, NaN, pandas NA)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _in_intervals(t_ns: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= t_ns < b for a, b in spans)


@dataclass
class _Rec:
    rec: dict
    plan: CropPlan
    region: tuple[float, float, float, float]
    evidence: float
    frame_order: int
    enu: tuple[float, float]
    poly: Polygon | None = None        # ground footprint, local ENU metres


# ---- main ------------------------------------------------------------------------------------

def process_session(session_dir: Path, out_root: Path, cfg: Config, *, allow_test: bool = False,
                    allow_untrusted: bool = False, features: PatchFeatures | None = None,
                    bank_keyframes_only: bool = True) -> dict:
    t_start = time.time()
    s = Session.load(session_dir)
    if s.session_type not in ("survey",) and not (allow_test and s.session_type == "test"):
        raise ExtractionError(f"session type {s.session_type!r} is not a survey")
    if not s.trusted and not allow_untrusted:
        raise ExtractionError("session has mock locations (untrusted); refusing to geolocate")
    verdict = s.orientation_verdict
    if verdict == "PORTRAIT":
        raise ExtractionError("phone was mounted in portrait: camera geometry unusable")
    rotate180 = verdict == "UPSIDE_DOWN"
    out = Path(out_root) / s.session_id
    if out.exists():
        shutil.rmtree(out)
    (out / "crops").mkdir(parents=True)
    (out / "frames").mkdir()
    (out / "sidecars").mkdir()

    # Time and trajectory.
    t0 = s.t0_ns
    fixes = clean_fixes(s.gnss, t0, cfg.gnss)
    traj = build_trajectory(fixes, s.imu, t0, cfg.gnss)
    log.info("trajectory: %.0f m, latency %.3f s (%s), %d gaps", traj.s[-1] - traj.s[0],
             traj.latency.latency_s, traj.latency.method, len(traj.gaps))

    # Views per resolution.
    gravity = _median_gravity(s)
    seg_files = s.video_files()
    seg_res = {}
    for seg, path in seg_files.items():
        if not path.exists():
            raise ExtractionError(f"missing video file {path}")
        info = segment_info(path)
        seg_res[seg] = (info["width"], info["height"])
    views: dict[tuple[int, int], View] = {}
    for res in sorted(set(seg_res.values())):
        model = nominal_model(s.meta, res[0], res[1], gravity, cfg.geometry, rotate180=rotate180)
        views[res] = plan_view(model, cfg, rotate180)
        log.info("view %dx%d: pitch %.1f deg, height %.2f m, band %.2f..%.2f m, analysis %dx%d",
                 res[0], res[1], model.pitch_deg, model.height_m, views[res].band_y0,
                 views[res].band_y0 + views[res].band_len, views[res].ana_w, views[res].ana_h)

    if features is None:
        features = PatchFeatures(Path(cfg.anomaly.weights), cfg.anomaly.device, cfg.anomaly.tile_patches,
                                 cfg.anomaly.tile_overlap_patches, cfg.anomaly.batch_tiles,
                                 cfg.anomaly.weights_sha256, cfg.anomaly.layers)
    if features.device.type != "cuda":
        log.warning("features on %s: several times slower than a GPU (is the CUDA build of torch installed?)",
                    features.device)
    detectors ={res: AnomalyDetector(features, v.ana_h, v.ana_w, cfg.anomaly, cfg.cracks) for res, v in views.items()}
    black = s.black_intervals_ns()

    def frame_ok(row: dict, clock: _RowClock, v: View):
        t = clock(row, (v.v0 + v.v1) / 2)
        p = traj.sample(np.array([t]))
        if not p["valid"][0] or p["reversing"][0]:
            return None
        if _in_intervals(int(row["t_ns"]), black):
            return None
        if cfg.extract.skip_low_light and s.low_light_at(np.array([int(row["t_ns"])]))[0]:
            return None
        return t, p

    # Pass 1: bank.
    t_bank = time.time()
    route = float(traj.s[-1] - traj.s[0])
    rng = np.random.default_rng(0)
    bank_counts = {res: 0 for res in views}

    def bank_pass(keyframes_only: bool, spacing: float, only_res=None):
        next_s = {res: -math.inf for res in views}
        for seg, path in seg_files.items():
            res = seg_res[seg]
            if only_res is not None and res not in only_res:
                continue
            v = views[res]
            clock = _RowClock(t0, v.height, v.rotate180)
            chosen: dict[int, float] = {}

            def want(idx, row, v=v, clock=clock, res=res, chosen=chosen):
                if bank_counts[res] + len(chosen) >= cfg.anomaly.bank_max_frames:
                    return False
                ok = frame_ok(row, clock, v)
                if ok is None or ok[1]["speed"][0] < 1.0:
                    return False
                s_m = float(ok[1]["s"][0])
                if s_m < next_s[res]:
                    return False
                next_s[res] = s_m + spacing * (1 + rng.uniform(-cfg.anomaly.bank_jitter, cfg.anomaly.bank_jitter))
                chosen[idx] = s_m
                return True

            rows = s.frames[s.frames["segment"] == seg]
            for fr in iter_segment(path, rows, segment=seg, keyframes_only=keyframes_only,
                                   rotate180=rotate180, want=want, allow_extra_rows=True):
                if fr.rgb is None:
                    continue
                detectors[res].add_bank_frame(v.analysis_image(fr.rgb), chosen[fr.index], v.gsd_rows_mm)
                bank_counts[res] += 1

    spacing = max(cfg.anomaly.bank_spacing_m, route / max(cfg.anomaly.bank_max_frames, 1))
    bank_pass(bank_keyframes_only, spacing)
    short = [res for res, n in bank_counts.items() if n < cfg.anomaly.bank_min_frames]
    if short:
        # Too few keyframe positions: use every frame, with spacing for the minimum count.
        log.info("bank: %s too few frames from keyframes; using all frames", short)
        for res in short:
            detectors[res] = AnomalyDetector(features, views[res].ana_h, views[res].ana_w, cfg.anomaly, cfg.cracks)
            bank_counts[res] = 0
        bank_pass(False, max(0.5, route / (2 * cfg.anomaly.bank_min_frames)), only_res=set(short))
    for res, det in detectors.items():
        if bank_counts[res] < 2:
            raise ExtractionError(f"only {bank_counts[res]} usable bank frames at {res[0]}x{res[1]}")
        det.finalize()
    log.info("bank: %s frames in %.1f s", bank_counts, time.time() - t_bank)

    # Pass 2: query.
    t_query = time.time()
    lat0 = traj.origin
    tz = cfg.output.utc_offset_minutes
    dev = s.meta.get("device", {}) or {}
    recs: list[_Rec] = []
    frame_order = 0
    next_q = -math.inf
    selected = 0
    frame_stats = []
    pool = PngPool()
    for seg, path in seg_files.items():
        res = seg_res[seg]
        v = views[res]
        det = detectors[res]
        clock = _RowClock(t0, v.height, v.rotate180)
        sel_info: dict[int, tuple] = {}

        def want(idx, row, v=v, clock=clock, sel_info=sel_info):
            nonlocal next_q
            ok = frame_ok(row, clock, v)
            if ok is None:
                return False
            s_m = float(ok[1]["s"][0])
            if s_m < next_q:
                return False
            # Next target on a 1.5 m grid; after a gap (already past the next target) restart the
            # grid one band ahead, instead of selecting the very next frame too.
            next_q = next_q + v.band_len if s_m < next_q + v.band_len else s_m + v.band_len
            sel_info[idx] = (ok[0], s_m)
            return True

        rows = s.frames[s.frames["segment"] == seg]
        for fr in iter_segment(path, rows, segment=seg, rotate180=rotate180, want=want, allow_extra_rows=True):
            if fr.rgb is None:
                continue
            selected += 1
            t_mid, s_anchor = sel_info.pop(fr.index)
            ana = v.analysis_image(fr.rgb)
            cands, diag = det.detect(ana, s_anchor, v.gsd_rows_mm)
            frame_stats.append({"frame_global": int(fr.row["frame_global"]), "s_m": s_anchor, **diag,
                                "candidates": len(cands), "band_len": v.band_len})
            frame_recs = _frame_records(fr, v, cands, t_mid, s_anchor, traj, s, cfg, clock, frame_order)
            if frame_recs:
                frame_id = f"{s.session_id}_{int(fr.row['frame_global']):07d}_frame"
                frame_recs = _merge_in_frame(frame_recs, v, cfg)
                for r in frame_recs:
                    r.rec["frame_request"] = frame_id if cfg.crop.full_frame else None
                    _write_crop(out, fr.rgb, r, s, dev, cfg, pool)
                if cfg.crop.full_frame:
                    fw, fh = frame_input_size(v.width, v.height)
                    small = cv2.resize(fr.rgb, (fw, fh), interpolation=cv2.INTER_AREA)
                    first = frame_recs[0].rec
                    frame_rec = {"id": frame_id, "kind": "frame", "session_id": s.session_id,
                                 "frame_global": first["frame_global"], "utc_ms": first["camera"].get("utc_ms"),
                                 "camera": first["camera"], "anomaly": {}}
                    pool.submit(out / "frames" / f"{frame_id}.png", small, frame_rec, make=dev.get("manufacturer", ""),
                                model=dev.get("model", ""), offset_minutes=tz, compress_level=cfg.crop.png_compress_level)
                recs.extend(frame_recs)
            frame_order += 1
    log.info("query: %d frames analysed, %d raw records in %.1f s", selected, len(recs), time.time() - t_query)

    pool.close()  # every crop and frame is on disk (and hashed) before files are moved below
    # Merge duplicates (strongest evidence kept), then apply the optional budget.
    recs.sort(key=lambda r: (r.frame_order, -r.evidence))
    _dedupe(recs, cfg)
    live = [r for r in recs if r.rec["status"] == "candidate"]
    km = max(route / 1000.0, 1e-3)
    budget = None if cfg.extract.budget_per_km is None else max(10, int(math.ceil(cfg.extract.budget_per_km * km)))
    live.sort(key=lambda r: -r.evidence)
    for k, r in enumerate(live):
        r.rec["status"] = "sent" if budget is None or k < budget else "over_budget"
        r.rec["rank"] = k + 1

    # Move what is not sent out of crops/; drop frames nothing sent refers to.
    (out / "not_sent").mkdir()
    sent_frames = set()
    for r in recs:
        if r.rec["status"] == "sent":
            sent_frames.add(r.rec.get("frame_request"))
        else:
            src = out / "crops" / r.rec["crop"]["file"]
            if src.exists():
                src.replace(out / "not_sent" / r.rec["crop"]["file"])
            r.rec["crop"]["file"] = f"../not_sent/{r.rec['crop']['file']}"
    for f in (out / "frames").glob("*.png"):
        if f.stem not in sent_frames:
            f.unlink()
    for r in recs:
        if r.rec.get("frame_request") not in sent_frames:
            r.rec["frame_request"] = None
    for r in recs:
        write_sidecar(out / "sidecars" / f"{r.rec['id']}.json", r.rec)
    all_recs = [r.rec for r in recs]
    write_csv(out / "candidates.csv", all_recs)
    write_geojson(out / "anomalies.geojson", [r for r in all_recs if r["status"] in ("sent", "over_budget")])

    uncovered = 0.0
    for a, b in zip(frame_stats, frame_stats[1:]):
        uncovered += max(0.0, b["s_m"] - a["s_m"] - a["band_len"] - cfg.extract.band_tolerance_m)
    counts = {}
    for r in all_recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report = {
        "pipeline_version": __version__, "session_id": s.session_id, "session_dir": str(session_dir),
        "created_utc": utc_strings(time.time() * 1000, 0)["utc_iso"],
        "route_m": route, "frames_analysed": selected, "records": counts, "budget": budget,
        "road_not_covered_m": uncovered,
        "estimated_vlm_hours": (counts.get("sent", 0) * (2 if cfg.crop.full_frame else 1)) * 14 / 3600,
        "trajectory": traj.to_dict(), "views": {f"{k[0]}x{k[1]}": v.summary() for k, v in views.items()},
        "bank": {f"{k[0]}x{k[1]}": {"frames": bank_counts[k], **detectors[k].bank.stats} for k in views},
        "timings_s": {"total": time.time() - t_start, "bank": t_query - t_bank, "query": time.time() - t_query},
        "config": cfg.to_dict(), "features_device": str(features.device),
        "probe": _probe_summary(next(iter(detectors.values())).probe),
        "frame_stats": frame_stats,
    }
    write_json_atomic(out / "report.json", report)
    _contact_sheet(out, [r.rec for r in sorted(recs, key=lambda r: -r.evidence) if r.rec["status"] == "sent"][:48])
    log.info("done: %s in %.1f s", counts, time.time() - t_start)
    return report


def _contact_sheet(out: Path, recs: list[dict], cell: int = 256, cols: int = 8) -> None:
    """contact_sheet.jpg: the highest-evidence sent crops, for a quick look at a run."""
    if not recs:
        return
    rows = (len(recs) + cols - 1) // cols
    sheet = np.full((rows * (cell + 22), cols * cell, 3), 255, np.uint8)
    for k, rec in enumerate(recs):
        img = cv2.imread(str(out / "crops" / rec["crop"]["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = cell / max(h, w)
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        y, x = r * (cell + 22), c * cell
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
        label = f"#{rec.get('rank', k + 1)} ev {rec['evidence']:.0f} {'+'.join(rec['kinds'])}"
        cv2.putText(sheet, label, (x + 3, y + cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _frame_records(fr, v: View, cands: list[Candidate], t_mid: float, s_anchor: float, traj: Trajectory,
                   s: Session, cfg: Config, clock: _RowClock, frame_order: int) -> list[_Rec]:
    out: list[_Rec] = []
    model = v.model
    tol = cfg.extract.band_tolerance_m
    cam = traj.sample(np.array([t_mid]))
    # The next selected frame lands up to one frame of travel beyond its target, so this band
    # also accepts that overshoot; anything kept twice is removed by the position de-duplication.
    overshoot = 1.5 * float(cam["speed"][0]) * fr_period_s(s)
    band_lo = s_anchor + v.band_y0 - tol
    band_hi = s_anchor + v.band_y0 + v.band_len + tol + overshoot
    clat, clon, calt = traj.enu_to_geodetic(cam["e"], cam["n"], cam["u"])
    utc_mid = float(traj.utc.utc_ms(t_mid)) if traj.utc is not None else None
    low_light = bool(s.low_light_at(np.array([int(fr.row["t_ns"])]))[0])
    view_dir = (cam["heading"][0] + (180.0 if not v.notes.get("forward_view", True) else 0.0)) % 360
    for k, c in enumerate(cands):
        x0, y0 = v.to_frame_px(c.bbox[0], c.bbox[1])
        x1, y1 = v.to_frame_px(c.bbox[2], c.bbox[3])
        x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
        x1, y1 = min(float(v.width), float(x1)), min(float(v.height), float(y1))
        cu, cv_ = (x0 + x1) / 2, (y0 + y1) / 2
        gx, gy, ok = model.pixel_to_ground(np.array([cu]), np.array([cv_]))
        if not ok[0]:
            continue
        t_c = clock(fr.row, cv_)
        pose = traj.sample(np.array([t_c]))
        s_c = float(pose["s"][0] + gy[0])
        if not band_lo <= s_c < band_hi:
            continue
        psi = math.radians(float(pose["heading"][0]))
        E = float(pose["e"][0] + gy[0] * math.sin(psi) + gx[0] * math.cos(psi))
        N = float(pose["n"][0] + gy[0] * math.cos(psi) - gx[0] * math.sin(psi))
        alat, alon, _ = traj.enu_to_geodetic(np.array([E]), np.array([N]))
        # Footprint: box corners (and edge midpoints) on the ground.
        us = np.array([x0, cu, x1, x1, x1, cu, x0, x0])
        vs = np.array([y0, y0, y0, cv_, y1, y1, y1, cv_])
        fx, fy, fok = model.pixel_to_ground(us, vs)
        area = None
        poly = None
        poly_ll = None
        if fok.all():
            pe = pose["e"][0] + fy * math.sin(psi) + fx * math.cos(psi)
            pn = pose["n"][0] + fy * math.cos(psi) - fx * math.sin(psi)
            poly = Polygon(np.column_stack([pe, pn]))
            if not poly.is_valid:
                poly = poly.buffer(0)
            area = float(poly.area)
            plat, plon, _ = traj.enu_to_geodetic(pe, pn)
            poly_ll = [[float(a), float(b)] for a, b in zip(plon, plat)]
        if area is not None and area < cfg.extract.min_area_m2:
            continue
        dE, dN = E - float(cam["e"][0]), N - float(cam["n"][0])
        sigma = float(pose["h_sigma"][0])
        if model.source == "nominal":
            sigma = math.hypot(sigma, 0.1 * math.hypot(gx[0], gy[0]))  # nominal height/yaw: ~10 % of range
        utc_c = float(traj.utc.utc_ms(t_c)) if traj.utc is not None else None
        flags = []
        if low_light:
            flags.append("low_light")
        if bool(pose["heading_from_gyro"][0]):
            flags.append("heading_from_gyro")
        if model.source == "nominal":
            flags.append("geometry_nominal")
        if traj.latency.method == "none":
            flags.append("latency_uncorrected")
        rec_id = f"{s.session_id}_{int(fr.row['frame_global']):07d}_{k:02d}"
        rec = {
            "id": rec_id, "kind": "crop", "status": "candidate", "session_id": s.session_id,
            "segment": fr.segment, "frame_in_segment": fr.index, "frame_global": int(fr.row["frame_global"]),
            "t_ns": int(fr.row["t_ns"]), "utc_ms": utc_c,
            **({k2: v2 for k2, v2 in utc_strings(utc_c, cfg.output.utc_offset_minutes).items() if k2 in ("utc_iso", "local_iso")} if utc_c else {}),
            "camera": {"lat": float(clat[0]), "lon": float(clon[0]), "alt_ellipsoid_m": float(calt[0]),
                       "alt_msl_m": None, "heading_deg": float(cam["heading"][0]), "view_direction_deg": view_dir,
                       "speed_mps": float(cam["speed"][0]), "h_sigma_m": float(cam["h_sigma"][0]),
                       "utc_ms": utc_mid},
            "anomaly": {"lat": float(alat[0]), "lon": float(alon[0]), "sigma_m": sigma,
                        "lateral_m": float(gx[0]), "ahead_m": float(gy[0]), "s_m": s_c,
                        "distance_m": math.hypot(dE, dN), "bearing_deg": math.degrees(math.atan2(dE, dN)) % 360,
                        "area_m2": area, "footprint_lonlat": poly_ll},
            "evidence": float(c.score), "kinds": list(c.kinds), "peak_u": c.z_dino, "peak_crack_z": c.z_crack,
            "peak_probe_logit": c.probe_max,
            "patches": c.patches, "crack_length_mm": c.crack_length_mm, "flags": flags,
            "geometry": model.source, "pipeline_version": __version__,
        }
        out.append(_Rec(rec, None, (x0, y0, x1, y1), float(c.score), frame_order, (E, N), poly))
    return out


def _probe_summary(probe) -> dict | None:
    if probe is None:
        return None
    m = probe.meta
    return {k: m.get(k) for k in ("name", "training_data", "test_auroc", "seed_rule", "test_fpr_at_seed")} |         {"seed_logit": probe.seed_logit}


def fr_period_s(s: Session) -> float:
    """Median frame period of the session (s)."""
    if getattr(s, "_period", None) is None:
        d = np.diff(s.frames["t_ns"].to_numpy(dtype=np.int64)) / 1e9
        d = d[d > 0]
        s._period = float(np.median(d)) if d.size else 1 / 30
    return s._period


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _plan(region, v: View, cfg: Config) -> CropPlan:
    c = cfg.crop
    return plan_crop(region, v.width, v.height, margin_frac=c.margin_frac, margin_min_px=c.margin_min_px,
                     min_tokens=c.min_tokens, max_pixels=c.max_pixels, max_aspect=c.max_aspect)


def _merge_in_frame(recs: list[_Rec], v: View, cfg: Config) -> list[_Rec]:
    """Anomalies close together in one frame become ONE crop of the pavement around all of them.

    Two anomalies merge when their ground footprints are within `crop.merge_gap_m` of each other, or
    (no footprint) their crops overlap by IoU >= `crop.merge_iou`. Merging repeats until nothing more
    merges, so a cluster of distresses becomes one larger segment. A merge is skipped when the merged
    crop would have to be downscaled below `crop.merge_min_scale` (fine cracks would blur away).
    The kept record has the highest evidence; it lists every member (id, position, evidence, area),
    so nothing measured about the members is lost.
    """
    c = cfg.crop
    for r in recs:
        r.plan = _plan(r.region, v, cfg)
        r.rec.setdefault("members", [_member(r)])

    def close(a: _Rec, b: _Rec) -> bool:
        if a.poly is not None and b.poly is not None and a.poly.distance(b.poly) <= c.merge_gap_m:
            return True
        return _iou(a.plan.box, b.plan.box) >= c.merge_iou

    merged = True
    while merged and len(recs) > 1:
        merged = False
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if not close(a, b):
                    continue
                region = (min(a.region[0], b.region[0]), min(a.region[1], b.region[1]),
                          max(a.region[2], b.region[2]), max(a.region[3], b.region[3]))
                plan = _plan(region, v, cfg)
                if plan.out_w / plan.box_w < c.merge_min_scale:
                    continue
                keep, other = (a, b) if a.evidence >= b.evidence else (b, a)
                keep.region = region
                keep.plan = plan
                keep.evidence = a.evidence + b.evidence
                keep.rec["evidence"] = keep.evidence
                keep.rec["kinds"] = sorted(set(a.rec["kinds"]) | set(b.rec["kinds"]))
                keep.rec.setdefault("merged", []).extend([other.rec["id"], *other.rec.get("merged", [])])
                keep.rec["members"] = keep.rec["members"] + other.rec["members"]
                if keep.poly is not None and other.poly is not None:
                    keep.poly = keep.poly.union(other.poly)
                    keep.rec["anomaly"]["merged_area_m2"] = float(keep.poly.area)
                recs.pop(j if keep is a else i)
                merged = True
                break
            if merged:
                break
    return recs


def _member(r: _Rec) -> dict:
    a = r.rec["anomaly"]
    return {"id": r.rec["id"], "lat": a["lat"], "lon": a["lon"], "evidence": r.evidence,
            "kinds": list(r.rec["kinds"]), "area_m2": a.get("area_m2")}


def _dedupe(recs: list[_Rec], cfg: Config) -> None:
    """Ground-plane non-maximum suppression across frames (see the module docstring)."""
    ex = cfg.extract
    cell = 2.0  # m; spatial hash for the neighbour search
    kept: dict[tuple[int, int], list[_Rec]] = {}

    def same(r: _Rec, k: _Rec) -> bool:
        if k.frame_order == r.frame_order:
            return False  # same frame: already merged by crop overlap
        if r.poly is None or k.poly is None:
            # No footprint (part of the box off the road plane): fall back to centroid distance.
            return math.hypot(k.enu[0] - r.enu[0], k.enu[1] - r.enu[1]) < ex.dedupe_dist_m
        if not r.poly.intersects(k.poly):
            return False
        small = min(r.poly.area, k.poly.area)
        return small > 0 and r.poly.intersection(k.poly).area >= ex.dedupe_overlap * small

    for r in sorted((r for r in recs if r.rec["status"] == "candidate"), key=lambda r: -r.evidence):
        reach = ex.dedupe_dist_m
        if r.poly is not None:
            x0, y0, x1, y1 = r.poly.bounds
            reach = max(reach, x1 - x0, y1 - y0)
        n = int(math.ceil(reach / cell))
        ci, cj = int(math.floor(r.enu[0] / cell)), int(math.floor(r.enu[1] / cell))
        winner = next((k for di in range(-n, n + 1) for dj in range(-n, n + 1)
                       for k in kept.get((ci + di, cj + dj), ()) if same(r, k)), None)
        if winner is not None:
            r.rec["status"] = "duplicate"
            r.rec["duplicate_of"] = winner.rec["id"]
            winner.rec.setdefault("duplicates", []).append(r.rec["id"])
        else:
            kept.setdefault((ci, cj), []).append(r)


def _write_crop(out: Path, rgb: np.ndarray, r: _Rec, s: Session, dev: dict, cfg: Config, pool: PngPool) -> None:
    p = r.plan
    x0, y0, x1, y1 = p.box
    crop = rgb[y0:y1, x0:x1]
    if (p.out_w, p.out_h) != (crop.shape[1], crop.shape[0]):
        crop = cv2.resize(crop, (p.out_w, p.out_h), interpolation=cv2.INTER_AREA)
    name = f"{r.rec['id']}.png"
    r.rec["crop"] = {"file": name, "box": list(p.box), "region": [float(v) for v in r.region],
                     "out_w": p.out_w, "out_h": p.out_h, "tokens": p.tokens, "scale": p.scale,
                     "truncated": p.truncated}
    if p.truncated:
        r.rec["flags"].append("truncated")
    crop_info = r.rec["crop"]
    pool.submit(out / "crops" / name, crop, r.rec, on_done=lambda sha, c=crop_info: c.__setitem__("sha256", sha),
                make=dev.get("manufacturer", ""), model=dev.get("model", ""),
                offset_minutes=cfg.output.utc_offset_minutes, compress_level=cfg.crop.png_compress_level)
    r.rec["image_unique_id"] = unique_id(r.rec["id"])
