"""All tunable parameters, with defaults. Override with a JSON file (`--config my.json`).

Unknown keys in an override are an error, so a typo can't silently leave a default in place.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class GnssConfig:
    max_h_acc_m: float = 25.0          # fixes worse than this are dropped
    max_gap_s: float = 3.0             # longer gaps make the trajectory undefined
    outlier_min_m: float = 5.0
    outlier_sigma: float = 3.0
    max_accel_mps2: float = 6.0
    default_latency_s: float = 0.0     # device default from the marker test (docs/pipeline/11 §4)
    default_latency_measured: bool = False  # True once the marker test has set default_latency_s
    latency_search_s: float = 0.5
    latency_min_corr: float = 0.6
    latency_min_driving_s: float = 60.0
    latency_max_disagreement_s: float = 0.1
    min_speed_for_heading_mps: float = 1.5
    reversing_flip_deg: float = 120.0


@dataclass
class GeometryConfig:
    default_height_m: float = 1.5      # used only when session.json has no nominal_height_m
    default_hfov_deg: float = 67.0     # used only when session.json has no usable intrinsics
    max_range_m: float = 6.0           # ground points further than this are not used
    max_gsd_mm: float = 3.0            # ROI ends where along-track resolution gets coarser than this
    bottom_exclude_frac: float = 0.0   # image rows to ignore at the bottom (vehicle body in view)


@dataclass
class ExtractConfig:
    band_length_m: float = 1.5         # road length each selected frame is responsible for
    band_tolerance_m: float = 0.3      # overlap kept at band edges, then de-duplicated
    context_m: float = 0.5             # extra road analysed beyond each band edge
    target_gsd_mm: float = 2.0         # analysis resolution; frames are downscaled to about this
    dedupe_overlap: float = 0.5        # same anomaly seen from two frames: ground footprints overlap this share of the smaller
    dedupe_dist_m: float = 0.35        # ... or, for a record without a footprint, centroids this close
    budget_per_km: float | None = None  # optional cap on crops sent per km (highest evidence first); None = send all
    min_area_m2: float = 0.0015        # smaller anomalies are ignored (about 4 x 4 cm)
    skip_low_light: bool = False


@dataclass
class AnomalyConfig:
    weights: str = str(PIPELINE_DIR / "models" / "dinov2" / "dinov2_vits14_reg4_pretrain.pth")
    weights_sha256: str = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
    device: str = "auto"               # auto | cpu | cuda
    # Mid-level blocks: the last block mixes in context from the whole tile, so one pothole made
    # 62-90 % of its neighbours look anomalous in tests; blocks 4+8 keep the score local.
    layers: tuple[int, ...] = (4, 8)
    tile_patches: int = 32             # DINOv2 tile = 32 x 14 = 448 px
    tile_overlap_patches: int = 4      # overlap between tiles; only tile centres are kept
    bank_spacing_m: float = 12.0       # one bank frame per this much road
    bank_max_frames: int = 400
    bank_min_frames: int = 12
    bank_size_per_row: int = 4096      # coreset size per patch row
    knn_k: int = 3
    trim_frac: float = 0.05            # most unusual bank patches dropped as likely distress
    exclude_window_m: float = 10.0     # bank patches this close to the query are ignored (same distress)
    bank_jitter: float = 0.35          # random +-35 % on bank spacing: periodic markings must not phase-lock
    # Patch score u = (d - median) / (tail_factor x (q90 - median)), from the bank's own
    # leave-place-out distances in the same rows. Median and 90th percentile come from the bulk,
    # which a few per cent of distress in the bank barely moves; an upper quantile would be set by
    # that distress (too high) or cut off by trimming (too low). On normal road the 99.9th
    # percentile of d was 2.6-3.2 x the (q90 - median) spread in tests, so u >= 1 is rare.
    tail_factor: float = 4.0
    # Divide brightness by its local mean (Gaussian, this many analysis pixels) before DINOv2, so
    # a shadow - mostly a multiplicative brightness change - looks like the road around it.
    illum_sigma_px: float | None = 24.0
    # Road paint (white/yellow markings) is not pavement distress: patches mostly covered by paint
    # add no evidence. Paint = brighter than paint_ratio x the local road brightness.
    mask_paint: bool = True
    paint_ratio: float = 1.45
    paint_patch_frac: float = 0.3
    score_threshold: float = 1.0       # seed level for an anomaly
    grow_fraction: float = 0.6         # anomaly grows into neighbours with u >= 0.6 x threshold
    # Smallest evidence kept: sum over patches of (u - grow level). Synthetic tests: distress
    # scored 31-136, texture flukes 0.4-8.3. Ranking under the crop budget does the rest.
    min_mass: float = 6.0
    # Absolute distress probe (tools/train_probe.py, Attain CC BY 4.0). Used when the file exists:
    # patches it calls distress stay out of the bank, and it adds its own candidates.
    probe: str | None = str(PIPELINE_DIR / "models" / "probe" / "attain_dinov2_probe_v1.npz")
    probe_bank_exclude: bool = True
    probe_candidates: bool = True
    probe_grow_logit: float = 1.0      # probe anomalies grow into patches within this of the seed level
    batch_tiles: int = 8


@dataclass
class CrackConfig:
    enabled: bool = True
    sigmas_px: tuple[float, ...] = (1.0, 2.0)
    z_threshold: float = 6.0           # ridge strength vs the session's background, robust z
    min_length_mm: float = 150.0       # shorter dark lines are texture, not cracks
    bank_mask_frac: float = 0.02       # patches with more crack pixels than this stay out of the bank
    as_candidates: bool = True


@dataclass
class CropConfig:
    margin_frac: float = 0.25
    margin_min_px: float = 64.0
    min_tokens: int = 324
    max_pixels: int = 1_048_576
    max_aspect: float = 3.0
    # Distresses close together in one frame become one larger crop of that pavement segment.
    merge_gap_m: float = 0.5           # ground footprints this close (m) merge
    merge_iou: float = 0.25            # ... or, without a footprint, crops overlapping this much
    merge_min_scale: float = 0.5       # never merge into a crop downscaled more than this (detail loss)
    full_frame: bool = True            # also send the whole frame (VLM team's week-1 request)
    png_compress_level: int = 1        # lossless at any level; measured on S21 crops: 5.3x faster than 6, files 16 % larger


@dataclass
class OutputConfig:
    utc_offset_minutes: int = 330      # IST, for EXIF local time only
    software: str = "GeoAI pipeline"


@dataclass
class VlmConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5/nf4/max_pixels=1048576"
    prompt_version: str = "v2:9da1601cfec54f76"
    labels: tuple[str, ...] = (
        "Bleeding", "Streaking", "Hungry Surface", "Hairline Cracks", "Alligator Cracking",
        "Longitudinal Cracking", "Transverse Cracking", "Edge Cracking", "Slippage", "Rutting",
        "Corrugation", "Shoving", "Shallow Depression", "Settlement", "Stripping", "Ravelling",
        "Potholes", "Edge Breaking", "Normal",
    )


@dataclass
class SegmentConfig:
    """Segment mode (docs/pipeline/13): one best full frame per fixed length of road."""
    length_m: float = 5.0              # road segment length along the path (GNSS path distance s)
    # A frame is a candidate for a segment when the segment's near edge lies between the closest road the
    # camera sees (near_m, from geometry) and near_m + window_m ahead of it, and its far edge is in view.
    window_m: float = 1.0              # 30 fps: ~3 candidates at 40 km/h, ~5 at 20 km/h
    sharp_cols: tuple[float, float] = (0.2, 0.8)  # columns used for sharpness (lane, not roadside)
    sharp_downscale: float = 0.5       # sharpness measured on a half-size grey image (speed)
    max_blur_px: float = 4.0           # predicted blur (motion + rotation) above this: used only if nothing better
    skip_low_light: bool = False       # True: low-light frames never chosen (else only when nothing else)
    min_speed_mps: float = 0.5         # standing still: no segment progress, no candidates
    export_segment_view: bool = False  # also write a fixed native-resolution image of the segment rows (research)


@dataclass
class GateConfig:
    """Frame-level distress check on the segment area of the best frame (docs/pipeline/13 §4)."""
    mode: str = "gate"                 # gate: send segments that pass | all: send every segment (calibration)
    # A segment passes when ANY signal reaches its threshold. Defaults are conservative starting points;
    # `geoai gate-calibrate` sets them from VLM answers (label-free) and writes gate_calibration.json.
    probe_frac: float = 0.05           # share of segment road patches the Attain probe calls distress
    evidence: float = 30.0             # summed bank + crack evidence of anomalies inside the segment
    crack_mm: float = 300.0            # total crack length inside the segment
    audit_frac: float = 0.10           # rejected segments also sent, to measure the gate's misses
    calibration: str | None = None     # path to gate_calibration.json; its thresholds override the above


@dataclass
class Config:
    gnss: GnssConfig = field(default_factory=GnssConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    cracks: CrackConfig = field(default_factory=CrackConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(obj: Any, override: dict, path: str) -> None:
    names = {f.name: f for f in dataclasses.fields(obj)}
    for k, v in override.items():
        if k not in names:
            raise KeyError(f"unknown config key {path}{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            if not isinstance(v, dict):
                raise TypeError(f"{path}{k} must be an object")
            _merge(cur, v, f"{path}{k}.")
        else:
            if isinstance(cur, tuple):
                v = tuple(v)
            setattr(obj, k, v)


def load_config(path: Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if path is not None:
        _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")), "")
    if overrides:
        _merge(cfg, overrides, "")
    return cfg
