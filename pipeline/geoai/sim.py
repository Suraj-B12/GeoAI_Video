"""Session simulator: writes a complete GeoAI Capture session package (spec v1.1) of a synthetic
road with known distresses, plus `truth.json`.

Used by the end-to-end tests: every stage after ingest runs on data whose answers are known
(where each distress is, which features are look-alikes, what the GNSS latency is).

Rendering is exact for the camera model: each pixel ray is intersected with the road plane at
its own row time (rolling shutter), and the road raster is sampled there (cv2.remap).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pymap3d
from av.video.reformatter import VideoReformatter

from .geometry import CameraModel, cam_from_device, vehicle_from_camera
from .simtexture import (RoadRaster, asphalt, draw_alligator, draw_crack, draw_marking, draw_patch,
                         draw_pothole, draw_shadow, to_rgb)
from .util import sha256_file

ORIGIN = (12.9715987, 77.5945627, 900.0)
T0_NS = 7_000_000_000_000
UTC0_MS = 1_792_650_000_000


@dataclass
class SimSpec:
    seed: int = 0
    width: int = 960
    height: int = 540
    fps: float = 15.0
    speed_mps: float = 8.0
    length_m: float = 80.0
    height_m: float = 1.6
    pitch_deg: float = 45.0
    hfov_deg: float = 67.0
    exposure_ns: int = 250_000
    rs_skew_ns: int = 12_000_000
    latency_s: float = 0.10          # GNSS fixes stamped this late
    gnss_noise_m: float = 0.25
    raster_res_m: float = 0.003
    bitrate: int = 8_000_000
    vehicle_shadow: bool = True      # a dark area fixed in the image (recurs in every frame)
    distresses: bool = True
    lookalikes: bool = True
    session_type: str = "survey"
    gnss_gap_s: tuple[float, float] | None = None   # no fixes in this interval (seconds from the first frame)
    extra: dict = field(default_factory=dict)


def _camera(spec: SimSpec) -> CameraModel:
    W, H = spec.width, spec.height
    f = (W / 2) / math.tan(math.radians(spec.hfov_deg) / 2)
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    th = math.radians(spec.pitch_deg)
    up_cam = np.array([0.0, -math.cos(th), -math.sin(th)])
    R = vehicle_from_camera(up_cam, "forward")
    return CameraModel(W, H, K, np.zeros(5), R, np.array([0.0, 0.0, spec.height_m]), "sim")


def _gravity_device(spec: SimSpec) -> np.ndarray:
    th = math.radians(spec.pitch_deg)
    up_cam = np.array([0.0, -math.cos(th), -math.sin(th)])
    return 9.80665 * cam_from_device(90).T @ up_cam


def build_road(spec: SimSpec) -> RoadRaster:
    rng = np.random.default_rng(spec.seed + 100)
    road = asphalt(5.0, spec.length_m + 12.0, spec.raster_res_m, seed=spec.seed, lateral0_m=-2.5, along0_m=-4.0)
    if spec.lookalikes:
        draw_marking(road, 1.05, 0.12, dash_m=3.0, gap_m=4.5)        # broken lane line (recurring)
        draw_marking(road, -1.35, 0.15)                              # continuous edge line
    L = spec.length_m
    if spec.distresses:
        draw_pothole(road, (0.1, 0.22 * L), 0.16, rng)
        draw_crack(road, (-0.7, 0.34 * L), 90.0, 1.3, 6.0, rng)                 # transverse
        draw_crack(road, (0.35, 0.44 * L), 2.0, 2.6, 5.0, rng)                  # longitudinal
        draw_alligator(road, (-0.2, 0.58 * L), (0.9, 1.2), 0.16, 4.0, rng)
        draw_patch(road, (0.25, 0.72 * L), (0.8, 0.6), 8.0, tone=0.62)
    if spec.lookalikes:
        draw_shadow(road, (-0.3, 0.86 * L), (1.4, 0.9), rng, depth=0.55)
    return road


def _maps(cam: CameraModel):
    u, v = np.meshgrid(np.arange(cam.width, dtype=np.float64), np.arange(cam.height, dtype=np.float64))
    x, y, ok = cam.pixel_to_ground(u, v, max_range_m=60.0)
    return x, y, ok


def _render(road: RoadRaster, gx, gy, ok, along_cam_rows: np.ndarray, shadow_mask, rng) -> np.ndarray:
    """Grey image: camera nadir at along-route position along_cam_rows[row] (row-specific time)."""
    along = gy + along_cam_rows[:, None]
    mx = ((gx - road.lateral0_m) / road.res_m).astype(np.float32)
    my = ((along - road.along0_m) / road.res_m).astype(np.float32)
    mx[~ok] = -10
    my[~ok] = -10
    g = cv2.remap(road.img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.75)
    g[~ok] = 0.8  # sky / above the horizon
    if shadow_mask is not None:
        g = g * (1 - 0.35 * shadow_mask)
    return g


def simulate_session(out_dir: Path, spec: SimSpec | None = None) -> dict:
    spec = spec or SimSpec()
    rng = np.random.default_rng(spec.seed)
    session_id = f"20261022T120000Z_SIM-{spec.seed:04d}_{spec.seed:04x}"
    d = Path(out_dir) / session_id
    (d / "video").mkdir(parents=True, exist_ok=True)
    cam = _camera(spec)
    road = build_road(spec)
    gx, gy, ok = _maps(cam)
    W, H = spec.width, spec.height
    shadow = None
    if spec.vehicle_shadow:
        yy, xx = np.mgrid[0:H, 0:W]
        shadow = ((yy > 0.88 * H) & (xx < 0.35 * W)).astype(np.float32)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 6)

    duration = spec.length_m / spec.speed_mps
    n_frames = int(duration * spec.fps)
    period_ns = int(round(1e9 / spec.fps))
    t_frames = T0_NS + np.arange(n_frames, dtype=np.int64) * period_ns

    # ---- video (H.264, fragmented MP4, keyframe every second, no B-frames) ----
    path = d / "video" / "seg_0001.mp4"
    container = av.open(str(path), mode="w", format="mp4",
                        options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
    stream = container.add_stream("libx264", rate=Fraction(spec.fps).limit_denominator(1000))
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = spec.bitrate
    stream.time_base = Fraction(1, 90000)
    stream.codec_context.time_base = Fraction(1, 90000)
    stream.codec_context.gop_size = int(round(spec.fps))
    stream.codec_context.max_b_frames = 0
    stream.options = {"bf": "0", "keyint_min": str(int(round(spec.fps))), "sc_threshold": "0",
                      "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"}
    ref = VideoReformatter()
    frames_rows = []
    sizes = []
    for k, t_ns in enumerate(t_frames):
        t_s = (int(t_ns) - T0_NS) / 1e9
        rows = np.arange(H, dtype=np.float64)
        acc = np.zeros((H, W), np.float32)
        for sub in (0.25, 0.75):  # motion blur: two samples within the exposure
            t_rows = t_s + (spec.exposure_ns * sub + spec.rs_skew_ns * rows / H) / 1e9
            acc += _render(road, gx, gy, ok, spec.speed_mps * t_rows, shadow, rng)
        grey = acc / 2
        rgb = to_rgb(grey)
        rgb = np.clip(rgb.astype(np.int16) + rng.normal(0, 1.5, rgb.shape).astype(np.int16), 0, 255).astype(np.uint8)
        vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        vf = ref.reformat(vf, format="yuv420p", src_colorspace="ITU709", dst_colorspace="ITU709",
                          src_color_range="JPEG", dst_color_range="MPEG")
        pts_us = int(t_ns) // 1000
        vf.pts = int(round(pts_us * 0.09))
        vf.time_base = Fraction(1, 90000)
        for pkt in stream.encode(vf):
            container.mux(pkt)
            sizes.append(pkt.size)
        frames_rows.append({
            "segment": 1, "frame_in_segment": k, "frame_global": k, "pts_us": pts_us, "sensor_ts_ns": int(t_ns),
            "t_ns": int(t_ns), "offset_ns": 0, "match_residual_ns": 0, "capture_frame_number": 1000 + k,
            "exposure_ns": spec.exposure_ns, "frame_duration_ns": period_ns, "iso": 200,
            "rs_skew_ns": spec.rs_skew_ns, "focus_diopters": 0.6, "focal_length_mm": 5.4, "lens_state": 0,
            "ois_mode": 0, "eis_mode": 0, "ae_state": 0, "af_state": 4, "awb_gains": "2.0;1.0;1.0;1.6",
            "zoom_ratio": 1.0, "is_keyframe": int(k % int(round(spec.fps)) == 0), "encoded_bytes": 0,
        })
    for pkt in stream.encode():
        container.mux(pkt)
        sizes.append(pkt.size)
    container.close()
    for row, sz in zip(frames_rows, sizes):
        row["encoded_bytes"] = sz

    _write_csv(d / "frames.csv", frames_rows)
    _write_csv(d / "segments.csv", [{
        "segment": 1, "file": "video/seg_0001.mp4", "start_t_ns": int(t_frames[0]) - 50_000_000,
        "end_t_ns": int(t_frames[-1]) + 100_000_000, "first_pts_us": frames_rows[0]["pts_us"],
        "last_pts_us": frames_rows[-1]["pts_us"], "frames": n_frames, "bytes": path.stat().st_size,
        "width": W, "height": H, "mime": "video/avc", "bitrate_bps": spec.bitrate, "close_reason": "user_stop"}])

    # ---- GNSS: 1 Hz from 3 s before the first frame to 3 s after the last ----
    taus = np.arange(-3.0, duration + 3.0, 1.0)
    noise = np.cumsum(rng.normal(0, spec.gnss_noise_m * 0.3, (len(taus), 2)), axis=0)
    noise = noise - noise.mean(0)
    noise = noise / max(noise.std(), 1e-9) * spec.gnss_noise_m if len(taus) > 2 else noise
    e = noise[:, 0]
    n = spec.speed_mps * taus + noise[:, 1]
    lat, lon, alt = pymap3d.enu2geodetic(e, n, np.zeros_like(e), *ORIGIN)
    fixes = []
    for i, tau in enumerate(taus):
        if spec.gnss_gap_s and spec.gnss_gap_s[0] <= tau <= spec.gnss_gap_s[1]:
            continue
        fixes.append({
            "t_ns": T0_NS + int(round((tau + spec.latency_s) * 1e9)), "t_unc_ns": 1000,
            "receipt_ns": T0_NS + int(round((tau + spec.latency_s + 0.015) * 1e9)),
            "utc_ms": UTC0_MS + int(round(tau * 1000)), "lat_deg": f"{lat[i]:.9f}", "lon_deg": f"{lon[i]:.9f}",
            "alt_ellipsoid_m": f"{alt[i]:.3f}", "alt_msl_m": f"{alt[i] + 86.0:.3f}", "h_acc_m": 3.0, "v_acc_m": 5.0,
            "speed_mps": f"{spec.speed_mps + rng.normal(0, 0.03):.3f}", "speed_acc_mps": 0.2,
            "bearing_deg": f"{rng.normal(0, 0.3) % 360:.2f}", "bearing_acc_deg": 2.0, "is_mock": 0,
            "provider": "gps", "n_sats_used": 18,
        })
    _write_csv(d / "gnss_fixes.csv", fixes)

    # ---- IMU (100 Hz): stationary mount, constant speed ----
    g_dev = _gravity_device(spec)
    t_imu = T0_NS + np.arange(int((-3.0) * 1e9), int((duration + 3.0) * 1e9), 10_000_000, dtype=np.int64)
    imu_rows = []
    for t in t_imu:
        a = g_dev + rng.normal(0, 0.05, 3)
        w = rng.normal(0, 0.002, 3)
        imu_rows.append({"t_ns": int(t), "sensor": "acc", "x": f"{a[0]:.5f}", "y": f"{a[1]:.5f}", "z": f"{a[2]:.5f}", "w": "", "accuracy": 3})
        imu_rows.append({"t_ns": int(t) + 1, "sensor": "gyro", "x": f"{w[0]:.6f}", "y": f"{w[1]:.6f}", "z": f"{w[2]:.6f}", "w": "", "accuracy": 3})
        imu_rows.append({"t_ns": int(t) + 2, "sensor": "gravity", "x": f"{g_dev[0]:.5f}", "y": f"{g_dev[1]:.5f}", "z": f"{g_dev[2]:.5f}", "w": "", "accuracy": 3})
    _write_csv(d / "imu.csv", imu_rows)

    _write_csv(d / "events.csv", [
        {"t_ns": int(t_frames[0]) - 2_000_000_000, "type": "session_start", "value_json": "{}"},
        {"t_ns": int(t_frames[0]) - 10_000_000, "type": "recording_started", "value_json": "{}"},
        {"t_ns": int(t_frames[n_frames // 2]), "type": "operator_marker",
         "value_json": json.dumps({"marker_type": "km_stone", "note": "12"})},
        {"t_ns": int(t_frames[-1]) + 3_000_000_000, "type": "session_end", "value_json": "{}"}])
    _write_csv(d / "exposure_control.csv", [{"t_ns": int(t_frames[0]) - 5_000_000, "reason": "metered",
                                              "exposure_ns": spec.exposure_ns, "iso": 200, "low_light": 0,
                                              "speed_mps": spec.speed_mps, "mean_luma": 110}])
    _write_csv(d / "clock_sync.csv", [{"t_ns_before": int(t), "mono_ns": int(t) - 5_000_000_000, "t_ns_after": int(t) + 2000,
                                        "wall_ms": UTC0_MS + int((int(t) - T0_NS) / 1e6) + 2500}
                                       for t in (t_frames[0], t_frames[-1])])
    _write_csv(d / "thermal.csv", [{"t_ns": int(t_frames[0]), "thermal_status": 0, "headroom_10s": 0.2,
                                     "battery_temp_c": 33.0, "battery_pct": 80, "charging": 1}])

    f = cam.K[0, 0]
    meta = {
        "spec_version": "1.1", "session_id": session_id, "session_type": spec.session_type,
        "boot_count_start": 7, "boot_count_end": 7,
        "app": {"package": "geoai.sim", "version_name": "sim", "version_code": 0, "build_type": "sim", "settings": {}},
        "device": {"device_id": f"SIM-{spec.seed:04d}", "manufacturer": "simulator", "model": "SIM", "sdk_int": 35},
        "survey": {"road_id": "SIM", "road_name": "Simulated road", "direction": "increasing_chainage", "lane": "L1",
                   "start_chainage_km": 12.0},
        "mount": {"mount_id": "M1", "camera_facing": "forward", "nominal_height_m": spec.height_m,
                  "nominal_pitch_deg": spec.pitch_deg},
        "camera": {"camera_id": "0", "lens_facing": "back", "sensor_orientation_deg": 90, "timestamp_source": "REALTIME",
                   "active_array_size": [0, 0, W, H], "pixel_array_size": [W, H],
                   "lens_intrinsic_calibration": [f, f, (W - 1) / 2, (H - 1) / 2, 0.0], "lens_distortion": None},
        "clock": {"camera_timestamp_source": "REALTIME", "encoder_timestamp_base": "SENSOR",
                  "sensor_to_session_offset_ns": 0, "encoder_pts_offset_ns": 0},
        "start": {"t_ns": int(t_frames[0]) - 2_000_000_000},
        "capture_settings": {"exposure_mode": "MANUAL_SHUTTER_AUTO_ISO", "streams_while_recording": "encoder+analysis",
                             "focus_mode": "LOCKED_AF", "eis": "OFF", "zoom_ratio": 1.0, "fps": spec.fps},
        "encoders": [{"mime": "video/avc", "width": W, "height": H, "config_level": 0,
                      "color": {"standard": "BT709", "range": "LIMITED", "transfer": "SDR"}}],
        "orientation": {"sensor_orientation_deg": 90, "gravity_device_xyz": [float(v) for v in g_dev], "verdict": "UPRIGHT"},
        "recording": {"first_frame_t_ns": int(t_frames[0]), "last_frame_t_ns": int(t_frames[-1]), "segments": 1},
        "end": {"t_ns": int(t_frames[-1]) + 3_000_000_000, "reason": "user_stop"},
        "stats": {"frames_encoded": n_frames, "mock_location_seen": False},
    }
    (d / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name not in ("manifest.json", "pc_receipt.json", "truth.json"))
    manifest = {"spec_version": "1.1", "session_id": session_id, "created_wall_ms": UTC0_MS,
                "files": [{"path": p.relative_to(d).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
                "signature": None, "signature_error": "simulated session"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- truth ----
    truth = {"session_id": session_id, "spec": asdict(spec), "origin": ORIGIN, "camera": cam.to_dict(), "features": []}
    for ft in road.features:
        poly = np.array(ft.polygon_m)
        # Road frame (lateral, along) == ENU (E, N) for this straight northbound route.
        plat, plon, _ = pymap3d.enu2geodetic(poly[:, 0], poly[:, 1], np.zeros(len(poly)), *ORIGIN)
        c = poly.mean(0)
        clat, clon, _ = pymap3d.enu2geodetic(c[0], c[1], 0.0, *ORIGIN)
        truth["features"].append({"kind": ft.kind, "is_distress": ft.is_distress, "params": ft.params,
                                  "polygon_enu": poly.tolist(), "polygon_latlon": np.column_stack([plat, plon]).tolist(),
                                  "centroid_enu": c.tolist(), "centroid_latlon": [float(clat), float(clon)]})
    (d / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return {"session_dir": d, "truth": truth}


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
