"""Segment mode, the orchestrator: a verified session -> one best full frame per 5 m of road (docs/pipeline/13).

`process_session_segments` runs, in order:
1. Loading and checks, trajectory, camera model and output folder, exactly as the crop mode
   (extract.process_session): survey sessions only, trusted locations, no portrait mount; an
   UPSIDE_DOWN mount is decoded rotated; attitude from the median gravity vector; nominal intrinsics.
2. Geometry and plan: a SegmentGeometry per recording size, then segments from the first to the last
   frame's camera position, shifted by near_m + window_m so every candidate window lies inside the
   recording (segments.plan_segments with t_span and offset_m = plan_offset_m).
3. Bank pass: frames spread over the route (keyframes first, all frames if that gives too few, as
   in the crop mode) build the session bank of each gate view (gate.plan_segment_view, far edge
   gate.gate_far_m of the selector's near_m).
4. One decode pass: every frame is described (camera position c at its centre-row time, exposure,
   low light, gyro rate) and offered to the BestFrameSelector, which asks for RGB only inside
   candidate windows. Each finished segment is scored by the gate on its best frame, decided, turned
   into a geoai.segment.v1 record, and for `send` and `audit` its frame is written in full
   resolution as lossless PNG (record embedded, EXIF) on the PNG writer's threads.
5. Outputs: sidecars, segments.csv, segments.geojson, contact_sheet.jpg and report.json.

Determinism. Everything a rerun writes into the run folder is byte-identical, except report.json's
`created_utc` and `timings_s` (wall-clock measurements) and `features_device`. The bank's spacing
jitter is seeded; the audit draw is SHA-256 of the record id; PNG, JSON, CSV and JPEG encoders are
deterministic. Structured logs (runlog.py) go to <out_root>/logs/, outside the run folder.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .analyze import LIMITS
from .anomaly import AnomalyDetector
from .config import Config
from .dinov2 import PatchFeatures
from .extract import ExtractionError, View, _in_intervals, _median_gravity, _probe_summary, _RowClock
from .gate import SegmentGate, gate_far_m, plan_segment_view
from .geometry import CameraModel, cam_from_device, nominal_model
from .records import PngPool, utc_strings
from .runlog import RunLog
from .segment_records import (SENT_STATUSES, STATUSES, FrameShot, build_segment_record, camera_pose, config_digest,
                              footprint_lonlat, geometry_block, path_points, segment_id, write_contact_sheet,
                              write_segment_outputs)
from .segments import (BestFrameSelector, FrameInfo, GyroImageRates, SegmentGeometry, SegmentResult, plan_offset_m,
                       plan_segments)
from .session import Session
from .trajectory import Trajectory, build_trajectory, clean_fixes
from .util import log, write_json_atomic
from .video import iter_segment, segment_info
from .vlm import SECONDS_PER_IMAGE

# Bank frames are taken only while moving at least this fast (as in the crop mode): standing still
# would put many copies of one place into the bank.
BANK_MIN_SPEED_MPS = 1.0
_PCTS = (10, 50, 90)
# Progress log lines at most this often. The VLM system's video toggle tails this log to show the
# extraction live; a 2.6 km session runs ~10 min, so 2 s is smooth without flooding the log.
PROGRESS_EVERY_S = 2.0


class _Progress:
    """Throttled `progress {json}` log lines for a watcher that tails the log.

    One line per phase change and at most one per `every_s` within a phase; `force` writes one
    regardless (a phase's last count). Lines go to the log only, never into the run folder, so the
    outputs stay byte-identical across reruns. Fields: phase ("plan" | "bank" | "bank_finalize" |
    "decode" | "outputs"), done, total (an estimate for the bank), and phase-specific counts.
    """

    def __init__(self, every_s: float = PROGRESS_EVERY_S):
        self.every_s = float(every_s)
        self._phase: str | None = None
        self._last = -math.inf

    def update(self, phase: str, done: int, total: int | None, *, force: bool = False, **fields) -> None:
        now = time.monotonic()
        if not force and phase == self._phase and now - self._last < self.every_s:
            return
        self._phase, self._last = phase, now
        log.info("progress %s", json.dumps({"phase": phase, "done": int(done), "total": total, **fields},
                                           sort_keys=True, default=str))


class _TimedFeatures:
    """PatchFeatures with a stopwatch: the DINOv2 time of the bank and of the gate, for report.json.

    The features come back as numpy arrays (copied from the device), so each call has finished when it
    returns and the wall time is the real cost.
    """

    def __init__(self, features: PatchFeatures):
        self._f = features
        self.seconds = 0.0
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        out = self._f(rgb)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return out


@dataclass
class _Res:
    """Everything that depends on the recording size."""
    model: CameraModel
    geom: SegmentGeometry
    view: View
    gate: SegmentGate | None = None
    bank_frames: int = 0


def _time_at_s(traj: Trajectory, s_m: float) -> float:
    """Trajectory time at which the path distance first reaches s_m (clipped to the trajectory), the
    inversion segment_records.path_points uses."""
    s = np.asarray(traj.s, dtype=np.float64)
    knot = np.concatenate([[True], np.diff(s) > 0])
    sk, tk = s[knot], np.asarray(traj.t, dtype=np.float64)[knot]
    return float(np.interp(min(max(float(s_m), sk[0]), sk[-1]), sk, tk))


def _dist(values) -> dict:
    """n, min, p10, median, p90, max and mean of the finite values (n only when there are none)."""
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    p = np.percentile(x, _PCTS)
    return {"n": int(x.size), "min": float(x.min()), "p10": float(p[0]), "median": float(p[1]), "p90": float(p[2]),
            "max": float(x.max()), "mean": float(x.mean())}


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


# ---- main ------------------------------------------------------------------------------------

def process_session_segments(session_dir: Path, out_root: Path, cfg: Config, *, allow_test: bool = False,
                             allow_untrusted: bool = False, features: PatchFeatures | None = None) -> dict:
    """Segment mode for one session (see the module docstring); returns report.json's content.

    Writes <out_root>/<session_id>/ (replaced if it exists) and appends to the structured log
    <out_root>/logs/<UTC date>/<session folder name>.jsonl. Raises ExtractionError when the session
    cannot be processed (wrong type, mock locations, portrait mount, no usable geometry, no bank).
    """
    session_dir = Path(session_dir)
    rl = RunLog(Path(out_root), session_dir.name, "process", mode="segments", session_dir=str(session_dir),
                pipeline_version=__version__, gate_mode=cfg.gate.mode)
    try:
        report = _process(session_dir, Path(out_root), cfg, allow_test, allow_untrusted, features, rl)
    except BaseException as e:
        rl.finish(False, error_type=type(e).__name__, error=str(e))
        raise
    rl.finish(True, records=report["records"], covered_m=report["coverage"]["covered_m"],
              not_covered_m=report["coverage"]["not_covered_m"], total_s=report["timings_s"]["total"])
    return report


def _process(session_dir: Path, out_root: Path, cfg: Config, allow_test: bool, allow_untrusted: bool,
             features: PatchFeatures | None, rl: RunLog) -> dict:
    t_start = time.time()
    timings: dict[str, float] = {}

    # ---- 1. loading, checks, trajectory (as extract.process_session) ----
    with rl.stage("load") as st:
        s = Session.load(session_dir)
        if s.session_type not in ("survey",) and not (allow_test and s.session_type == "test"):
            raise ExtractionError(f"session type {s.session_type!r} is not a survey")
        if not s.trusted and not allow_untrusted:
            raise ExtractionError("session has mock locations (untrusted); refusing to geolocate")
        verdict = s.orientation_verdict
        if verdict == "PORTRAIT":
            raise ExtractionError("phone was mounted in portrait: camera geometry unusable")
        if s.frames.empty:
            raise ExtractionError("session has no frames")
        rotate180 = verdict == "UPSIDE_DOWN"
        out = out_root / s.session_id
        if out.exists():
            shutil.rmtree(out)
        (out / "frames").mkdir(parents=True)
        (out / "sidecars").mkdir()
        st.update(session_id=s.session_id, frames=int(len(s.frames)), orientation=verdict)

    with rl.stage("trajectory") as st:
        t0 = s.t0_ns
        fixes = clean_fixes(s.gnss, t0, cfg.gnss)
        traj = build_trajectory(fixes, s.imu, t0, cfg.gnss)
        route = float(traj.s[-1] - traj.s[0])
        log.info("trajectory: %.0f m, latency %.3f s (%s), %d gaps", route, traj.latency.latency_s,
                 traj.latency.method, len(traj.gaps))
        st.update(route_m=route, latency_s=traj.latency.latency_s, latency_method=traj.latency.method,
                  gaps=len(traj.gaps))

    # ---- 2. geometry, gate views, segment plan ----
    with rl.stage("geometry") as st:
        gravity = _median_gravity(s)
        seg_files = s.video_files()
        seg_res: dict[int, tuple[int, int]] = {}
        for seg, path in seg_files.items():
            if not path.exists():
                raise ExtractionError(f"missing video file {path}")
            info = segment_info(path)
            seg_res[seg] = (info["width"], info["height"])
        models = {res: nominal_model(s.meta, res[0], res[1], gravity, cfg.geometry, rotate180=rotate180)
                  for res in sorted(set(seg_res.values()))}
        geoms = {}
        for res, model in models.items():
            try:
                geoms[res] = SegmentGeometry.build(model, cfg)
            except ValueError as e:
                raise ExtractionError(f"{res[0]}x{res[1]}: {e}") from e
        frames = s.frames.sort_values("t_ns", kind="stable")
        first, last = frames.iloc[0].to_dict(), frames.iloc[-1].to_dict()

        def centre_time(row: dict) -> float:
            H = seg_res[int(row["segment"])][1]
            return _RowClock(t0, H, rotate180)(row, (H - 1) / 2)

        t_span = (centre_time(first), centre_time(last))
        offset = plan_offset_m(list(geoms.values()), cfg)
        segments = plan_segments(traj, cfg, t_span=t_span, offset_m=offset)
        if not segments:
            raise ExtractionError("no valid trajectory while recording: no segment can be planned")
        sel = BestFrameSelector(segments, list(geoms.values()), cfg)
        far_needed = gate_far_m(sel.near_m, cfg)
        rs: dict[tuple[int, int], _Res] = {}
        for res, model in models.items():
            view = plan_segment_view(model, geoms[res].near_m, far_needed, cfg, rotate180)
            rs[res] = _Res(model, geoms[res], view)
            log.info("segments %dx%d: pitch %.1f deg, height %.2f m, road %.2f..%.2f m ahead; gate view rows %d..%d, "
                     "analysis %dx%d", res[0], res[1], model.pitch_deg, model.height_m, geoms[res].near_m,
                     geoms[res].far_m, view.v0, view.v1, view.ana_w, view.ana_h)
        chainage_origin = float(traj.sample(np.array([t_span[0]]))["s"][0])
        log.info("plan: %d segments of %.1f m from s = %.1f m (offset %.2f m), windows %.2f m",
                 len(segments), cfg.segments.length_m, segments[0].s0, offset, cfg.segments.window_m)
        st.update(segments=len(segments), near_m=sel.near_m, far_m=sel.far_m, offset_m=offset,
                  sizes=[f"{w}x{h}" for w, h in rs])
    prog = _Progress()
    prog.update("plan", 0, len(segments), route_m=round(route, 1), length_m=cfg.segments.length_m)

    if features is None:
        features = PatchFeatures(Path(cfg.anomaly.weights), cfg.anomaly.device, cfg.anomaly.tile_patches,
                                 cfg.anomaly.tile_overlap_patches, cfg.anomaly.batch_tiles,
                                 cfg.anomaly.weights_sha256, cfg.anomaly.layers)
    if features.device.type != "cuda":
        log.warning("features on %s: several times slower than a GPU (is the CUDA build of torch installed?)",
                    features.device)
    timed = _TimedFeatures(features)
    black = s.black_intervals_ns()
    timings["setup"] = time.time() - t_start

    # ---- 3. bank ----
    with rl.stage("bank") as st:
        bank = _bank_pass(s, traj, cfg, rs, seg_files, seg_res, rotate180, black, timed, route, prog)
        timings.update({f"bank_{k}": v for k, v in bank.items()})
        st.update(frames={f"{w}x{h}": r.bank_frames for (w, h), r in rs.items()}, **bank)

    # ---- 4. decode, select, gate, records ----
    with rl.stage("decode") as st:
        dec = _Decode(s, traj, cfg, rs, sel, out, chainage_origin, prog)
        t_dec = time.time()
        dino_before = timed.seconds
        for seg, path in seg_files.items():
            dec.run_segment(seg, path, seg_res[seg], rotate180, black)
        dec.finish()
        timings["decode_pass"] = time.time() - t_dec
        timings["gate"] = dec.t_gate
        timings["gate_dinov2"] = timed.seconds - dino_before
        timings["png_wait"] = dec.t_png
        st.update(frames_decoded=dec.n_decoded, frames_rgb=dec.n_rgb, segments=len(dec.records),
                  statuses={k: sum(1 for r in dec.records if r["status"] == k) for k in STATUSES})
    recs = dec.records
    log.info("decode: %d frames (%d converted to RGB), %d segments in %.1f s (gate %.1f s)", dec.n_decoded, dec.n_rgb,
             len(recs), timings["decode_pass"], dec.t_gate)

    # ---- 5. outputs ----
    prog.update("outputs", len(recs), len(segments), force=True)
    with rl.stage("outputs") as st:
        t_out = time.time()
        write_segment_outputs(out, recs)
        write_contact_sheet(out, recs)
        timings["outputs"] = time.time() - t_out
        timings["dinov2"] = timed.seconds
        timings["total"] = time.time() - t_start
        report = _report(s, session_dir, cfg, traj, route, rs, sel, segments, offset, chainage_origin, recs, dec,
                         timings, timed, features)
        write_json_atomic(out / "report.json", report)
        st.update(files=len(recs))
    counts = report["records"]
    log.info("done: %s; covered %.0f of %.0f m; pass rate %s; ~%.1f h of VLM time; %.1f s", counts,
             report["coverage"]["covered_m"], report["coverage"]["planned_m"],
             _fmt_pct(report["gate"]["pass_rate"]), report["estimated_vlm_hours"], timings["total"])
    return report


def _fmt_pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


# ---- bank pass -------------------------------------------------------------------------------

def _bank_pass(s: Session, traj: Trajectory, cfg: Config, rs: dict[tuple[int, int], _Res], seg_files: dict,
               seg_res: dict, rotate180: bool, black: list, features: _TimedFeatures, route: float,
               prog: _Progress) -> dict:
    """Build every gate's bank: frames spread along the route, as extract.process_session's pass 1.

    A frame qualifies when the trajectory is valid there, it is not reversing, not in a black interval,
    not low-light when segments.skip_low_light, and the vehicle moves (BANK_MIN_SPEED_MPS). One frame is
    taken per `bank_spacing_m` of road (randomly jittered by +-bank_jitter, seeded, so markings can't
    phase-lock), and at most bank_max_frames per recording size. Keyframes first, which needs no full
    decode; if a size gets fewer than bank_min_frames, its bank is rebuilt from all frames at the
    spacing that gives the minimum. Returns the timings (s): decode, features (DINOv2 + crack filter
    + bank bookkeeping per frame), finalize, total.
    """
    an = cfg.anomaly
    rng = np.random.default_rng(0)
    t_feat = 0.0
    t_bank = time.time()
    t0 = s.t0_ns

    def new_gate(r: _Res) -> SegmentGate:
        det = AnomalyDetector(features, r.view.ana_h, r.view.ana_w, an, cfg.cracks)
        return SegmentGate(det, r.view, cfg)

    for r in rs.values():
        r.gate = new_gate(r)
        r.bank_frames = 0

    def frame_s(row: dict, r: _Res) -> float | None:
        """Path position of a usable bank frame (time of the gate view's middle row), else None."""
        clock = _RowClock(t0, r.view.height, rotate180)
        p = traj.sample(np.array([clock(row, (r.view.v0 + r.view.v1) / 2)]))
        if not p["valid"][0] or p["reversing"][0] or p["speed"][0] < BANK_MIN_SPEED_MPS:
            return None
        if _in_intervals(int(row["t_ns"]), black):
            return None
        if cfg.segments.skip_low_light and s.low_light_at(np.array([int(row["t_ns"])]))[0]:
            return None
        return float(p["s"][0])

    def run(keyframes_only: bool, spacing: float, only: set | None) -> None:
        nonlocal t_feat
        next_s = {res: -math.inf for res in rs}
        # Expected bank size (progress only): one frame per `spacing` of road, capped, per recording size.
        target = sum(min(an.bank_max_frames, max(1, math.ceil(route / spacing)))
                     for res in rs if only is None or res in only)
        for seg, path in seg_files.items():
            res = seg_res[seg]
            if only is not None and res not in only:
                continue
            r = rs[res]
            chosen: dict[int, float] = {}

            def want(idx, row, r=r, res=res, chosen=chosen):
                if r.bank_frames + len(chosen) >= an.bank_max_frames:
                    return False
                s_m = frame_s(row, r)
                if s_m is None or s_m < next_s[res]:
                    return False
                next_s[res] = s_m + spacing * (1 + rng.uniform(-an.bank_jitter, an.bank_jitter))
                chosen[idx] = s_m
                return True

            rows = s.frames[s.frames["segment"] == seg]
            for fr in iter_segment(path, rows, segment=seg, keyframes_only=keyframes_only, rotate180=rotate180,
                                   want=want, allow_extra_rows=True):
                if fr.rgb is None:
                    continue
                t = time.perf_counter()
                r.gate.add_bank_frame(fr.rgb, chosen.pop(fr.index))
                t_feat += time.perf_counter() - t
                r.bank_frames += 1
                prog.update("bank", sum(x.bank_frames for x in rs.values()), target)

    run(True, max(an.bank_spacing_m, route / max(an.bank_max_frames, 1)), None)
    short = {res for res, r in rs.items() if r.bank_frames < an.bank_min_frames}
    if short:
        log.info("bank: %s too few frames from keyframes; using all frames", sorted(short))
        for res in short:
            rs[res].gate = new_gate(rs[res])
            rs[res].bank_frames = 0
        run(False, max(0.5, route / (2 * an.bank_min_frames)), short)
    t_fin = time.time()
    n_bank = sum(r.bank_frames for r in rs.values())
    prog.update("bank_finalize", n_bank, n_bank, force=True)
    for res, r in rs.items():
        if r.bank_frames < 2:
            raise ExtractionError(f"only {r.bank_frames} usable bank frames at {res[0]}x{res[1]}")
        r.gate.finalize()
    end = time.time()
    total = end - t_bank
    log.info("bank: %s frames in %.1f s (features %.1f s, finalize %.1f s)",
             {f"{w}x{h}": r.bank_frames for (w, h), r in rs.items()}, total, t_feat, end - t_fin)
    return {"total": total, "decode": total - t_feat - (end - t_fin), "features": t_feat, "finalize": end - t_fin}


# ---- decode pass -----------------------------------------------------------------------------

class _Decode:
    """The one decode pass: frames -> selector -> finished segments -> gate -> records and PNGs."""

    def __init__(self, s: Session, traj: Trajectory, cfg: Config, rs: dict[tuple[int, int], _Res],
                 sel: BestFrameSelector, out: Path, chainage_origin: float, prog: _Progress):
        self.s, self.traj, self.cfg, self.rs, self.sel, self.out = s, traj, cfg, rs, sel, out
        self.prog = prog
        self.status_counts = {k: 0 for k in STATUSES}
        self.chainage_origin = chainage_origin
        self.main = rs[max(rs, key=lambda res: res[0] * res[1])]  # the largest size: its geometry for uncovered segments
        self.config_sha = config_digest(cfg)
        self.dev = s.meta.get("device", {}) or {}
        cam = s.meta.get("camera", {}) or {}
        so = int(cam.get("sensor_orientation_deg") or (s.meta.get("orientation") or {}).get("sensor_orientation_deg") or 90)
        self.gyro = GyroImageRates(s.imu, cam_from_device(so), s.t0_ns)
        self.pool = PngPool()
        self.records: list[dict] = []
        self.results: list[SegmentResult] = []
        self.n_decoded = self.n_rgb = 0
        self.n_low_light = 0
        self.t_gate = 0.0
        self.t_png = 0.0
        if cfg.segments.export_segment_view:
            (out / "segment_views").mkdir()

    def _infos(self, rows, res: tuple[int, int], rotate180: bool, black: list) -> dict[int, FrameInfo]:
        """FrameInfo of every frames.csv row of one video segment, by sample index (vectorised over rows)."""
        recs = rows.sort_values("frame_in_segment").to_dict("records")
        if not recs:
            return {}
        clock = _RowClock(self.s.t0_ns, res[1], rotate180)
        t = np.array([clock(r, (res[1] - 1) / 2) for r in recs], dtype=np.float64)
        t_ns = np.array([int(r["t_ns"]) for r in recs], dtype=np.int64)
        p = self.traj.sample(t)
        low = self.s.low_light_at(t_ns)
        omega = self.gyro(t)
        out = {}
        for i, r in enumerate(recs):
            out[int(r["frame_in_segment"])] = FrameInfo(
                t=float(t[i]), c=float(p["s"][i]), speed_mps=float(p["speed"][i]), valid=bool(p["valid"][i]),
                reversing=bool(p["reversing"][i]), black=_in_intervals(int(t_ns[i]), black), low_light=bool(low[i]),
                exposure_ns=r.get("exposure_ns"), omega_perp_rad_s=float(omega[i]), ref=r)
        return out

    def run_segment(self, seg: int, path: Path, res: tuple[int, int], rotate180: bool, black: list) -> None:
        rows = self.s.frames[self.s.frames["segment"] == seg]
        infos = self._infos(rows, res, rotate180, black)

        def want(idx, row):
            info = infos.get(idx)
            return info is not None and self.sel.wants(info)

        for fr in iter_segment(path, rows, segment=seg, rotate180=rotate180, want=want, allow_extra_rows=True):
            info = infos[fr.index]
            self.n_decoded += 1
            self.n_low_light += int(info.low_light)
            if fr.rgb is not None:
                self.n_rgb += 1
            self.sel.offer(info, fr.rgb)
            for r in self.sel.pop_ready(info.c):
                self._segment(r)

    def finish(self) -> None:
        for r in self.sel.flush():
            self._segment(r)
        t = time.time()
        self.pool.close()  # every frame is on disk and hashed (frame.sha256 filled) before the outputs
        self.t_png = time.time() - t

    def _segment(self, res: SegmentResult) -> None:
        seg, cfg, traj = res.segment, self.cfg, self.traj
        rid = segment_id(self.s.session_id, seg.k)
        start, centre, end = path_points(traj, [seg.s0, (seg.s0 + seg.s1) / 2, seg.s1])
        if res.covered:
            b = res.best
            r = self.rs[(b.width, b.height)]
            t = time.perf_counter()
            scores = r.gate.score(res.rgb, b.info.c, b.rows)
            self.t_gate += time.perf_counter() - t
            gate = r.gate.decide(scores, rid)
            pose = camera_pose(traj, b.info.t)
            ring = r.geom.footprint(b.near_edge_ahead_m, b.far_edge_ahead_m)
            shot = FrameShot(b.info.ref, b.width, b.height, b.rows, b.near_edge_ahead_m, b.far_edge_ahead_m,
                             b.info.low_light)
            foot_pose = pose
        else:
            # The view the segment would have been classified from: camera at the window's near end.
            r = self.main
            gate = shot = pose = None
            foot_pose = camera_pose(traj, _time_at_s(traj, res.window[1]))
            ring = r.geom.footprint(self.sel.near_m, self.sel.near_m + seg.length_m)
        rec = build_segment_record(
            session_id=self.s.session_id, session_meta=self.s.meta, segment=seg, start=start, centre=centre, end=end,
            footprint=footprint_lonlat(traj, ring, foot_pose), shot=shot, camera=pose, quality=res.quality(),
            gate=gate, geometry=geometry_block(r.model),
            latency_method=traj.latency.method, chainage_origin_m=self.chainage_origin, cfg=cfg,
            config_sha256=self.config_sha)
        if rec["status"] in SENT_STATUSES:
            frame = rec["frame"]
            kw = {"make": self.dev.get("manufacturer", ""), "model": self.dev.get("model", ""),
                  "software": cfg.output.software, "offset_minutes": cfg.output.utc_offset_minutes,
                  "compress_level": cfg.crop.png_compress_level}
            self.pool.submit(self.out / frame["file"], res.rgb, rec, on_done=lambda sha, f=frame: f.__setitem__("sha256", sha),
                             **kw)
            if cfg.segments.export_segment_view:
                r0, r1 = res.best.row_slice
                self.pool.submit(self.out / "segment_views" / f"{rid}.png", res.rgb[r0:r1], rec, **kw)
        self.records.append(rec)
        self.status_counts[rec["status"]] += 1
        # Progress: the frame named here is written by the PNG pool shortly after (sent segments only).
        self.prog.update("decode", len(self.records), len(self.sel.segments), segment=rid, k=seg.k,
                         status=rec["status"], frames_decoded=self.n_decoded, **self.status_counts)
        # The result without its frame: report statistics only, the RGB is released here.
        self.results.append(SegmentResult(res.segment, res.window, res.best, None, res.n_candidates,
                                          res.sharpness_rank, res.reason, res.excluded))


# ---- report ----------------------------------------------------------------------------------

def _report(s: Session, session_dir: Path, cfg: Config, traj: Trajectory, route: float, rs: dict, sel: BestFrameSelector,
            segments: list, offset: float, chainage_origin: float, recs: list[dict], dec: _Decode, timings: dict,
            timed: _TimedFeatures, features: PatchFeatures) -> dict:
    """report.json (docs/pipeline/13 §5, HANDOFF_PLAN §9): coverage, gate, quality, timings, VLM estimate."""
    L = cfg.segments.length_m
    counts = {k: sum(1 for r in recs if r["status"] == k) for k in STATUSES}
    cov = [r for r in recs if r["status"] != "not_covered"]
    unc = [r for r in recs if r["status"] == "not_covered"]
    planned_m = math.fsum(r["segment"]["length_m"] for r in recs)
    covered_m = math.fsum(r["segment"]["length_m"] for r in cov)
    by_reason: dict[str, dict] = {}
    for r in unc:
        d = by_reason.setdefault(r["quality"]["reason"], {"segments": 0, "length_m": 0.0})
        d["segments"] += 1
        d["length_m"] += r["segment"]["length_m"]
    gated = {k: sum(1 for r in cov if r["gate"]["decision_if_gated"] == k) for k in ("send", "audit", "skip")}
    reasons: dict[str, int] = {}
    for r in cov:
        for x in r["gate"]["reasons"]:
            reasons[x] = reasons.get(x, 0) + 1
    n_sent = counts["send"] + counts["audit"]
    km = planned_m / 1000.0      # the road the plan covers (the GNSS route also spans before and after recording)
    q = [r["quality"] for r in cov]
    ncand = [r["quality"]["n_candidates"] for r in recs]
    hist: dict[str, int] = {}
    for n in ncand:
        hist[str(n)] = hist.get(str(n), 0) + 1
    excluded: dict[str, int] = {}
    for res in dec.results:
        for k, v in res.excluded.items():
            excluded[k] = excluded.get(k, 0) + v
    low_chosen = sum(1 for r in cov if r["capture"]["low_light"])
    over = sum(1 for r in cov if "over_blur_limit" in r["flags"])
    speeds = [r["camera"]["speed_mps"] for r in cov]
    gsd = [res.best.gsd_along_mm for res in dec.results if res.best is not None]
    return {
        "mode": "segments", "pipeline_version": __version__, "session_id": s.session_id, "session_dir": str(session_dir),
        "created_utc": utc_strings(time.time() * 1000, 0)["utc_iso"], "config_sha256": dec.config_sha,
        "route_m": route, "limits": LIMITS,
        "plan": {"length_m": L, "window_m": cfg.segments.window_m, "offset_m": offset, "near_m": sel.near_m,
                 "far_m": sel.far_m, "segments": len(segments), "partial_last": bool(segments[-1].partial),
                 "s_first_m": segments[0].s0, "s_last_m": segments[-1].s1, "chainage_origin_s_m": chainage_origin},
        "records": counts,
        "coverage": {"planned_m": planned_m, "covered_m": covered_m, "not_covered_m": planned_m - covered_m,
                     "covered_share": _share(covered_m, planned_m),
                     "segments_covered": len(cov), "segments_not_covered": len(unc),
                     "not_covered_by_reason": dict(sorted(by_reason.items()))},
        "gate": {"mode": cfg.gate.mode, "decision_if_gated": gated, "pass_rate": _share(gated["send"], len(cov)),
                 "audit_rate": _share(gated["audit"], len(cov)), "skip_rate": _share(gated["skip"], len(cov)),
                 "sent_rate": _share(n_sent, len(cov)), "reasons": dict(sorted(reasons.items())),
                 "thresholds": next(iter(rs.values())).gate.thresholds,
                 "score_distribution": {k: _dist(r["gate"]["scores"][k] for r in cov)
                                        for k in ("probe_frac", "evidence", "crack_mm", "z_max", "anomalous_patch_frac")}},
        "quality": {"candidates_per_segment": {**_dist(ncand), "histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0])))},
                    "candidates_per_covered_segment": _dist(x["n_candidates"] for x in q),
                    "frames_excluded_in_windows": dict(sorted(excluded.items())),
                    "sharpness": _dist(x["sharpness"] for x in q),
                    "sharpness_rank": _dist(x["sharpness_rank"] for x in q),
                    "blur_px": _dist(x["blur_px"] for x in q), "blur_motion_px": _dist(x["blur_motion_px"] for x in q),
                    "blur_rotation_px": _dist(x["blur_rotation_px"] for x in q),
                    "over_blur_limit_share": _share(over, len(cov)), "max_blur_px": cfg.segments.max_blur_px,
                    "gsd_along_mm_segment_centre": _dist(gsd), "speed_mps": _dist(speeds),
                    "low_light_share_chosen": _share(low_chosen, len(cov)),
                    "low_light_share_frames": _share(dec.n_low_light, dec.n_decoded),
                    "chosen_reasons": {k: sum(1 for x in q if x["reason"] == k) for k in sorted({x["reason"] for x in q})}},
        "frames": {"decoded": dec.n_decoded, "converted_rgb": dec.n_rgb, "written": n_sent},
        "estimated_vlm_hours": n_sent * SECONDS_PER_IMAGE / 3600,
        "vlm_estimate": {"seconds_per_image": SECONDS_PER_IMAGE, "images": n_sent,
                         "hours": n_sent * SECONDS_PER_IMAGE / 3600,
                         "hours_per_km": n_sent * SECONDS_PER_IMAGE / 3600 / km if km > 0 else None,
                         "hours_if_all_covered_sent": len(cov) * SECONDS_PER_IMAGE / 3600},
        "trajectory": traj.to_dict(),
        "geometry": {f"{w}x{h}": {**r.geom.summary(L), "gate_view": r.view.summary()} for (w, h), r in rs.items()},
        "bank": {f"{w}x{h}": {"frames": r.bank_frames, **r.gate.detector.bank.stats} for (w, h), r in rs.items()},
        "probe": _probe_summary(next(iter(rs.values())).gate.detector.probe),
        "timings_s": dict(timings), "dinov2_calls": timed.calls, "features_device": str(features.device),
        "config": cfg.to_dict(),
    }
