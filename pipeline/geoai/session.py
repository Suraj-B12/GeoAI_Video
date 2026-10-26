"""Reading a GeoAI Capture session package (docs/session-package-spec.md, v1.x)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .util import read_csv_safe

SUPPORTED_MAJOR = 1

# Columns that hold session-clock nanoseconds: read as exact integers (nullable).
_INT_NS = {
    "frames": ["t_ns", "sensor_ts_ns", "pts_us", "offset_ns", "match_residual_ns", "exposure_ns",
               "frame_duration_ns", "rs_skew_ns", "capture_frame_number"],
    "gnss": ["t_ns", "t_unc_ns", "receipt_ns", "utc_ms"],
    "imu": ["t_ns"],
    "events": ["t_ns"],
    "exposure": ["t_ns", "exposure_ns"],
    "clock": ["t_ns_before", "mono_ns", "t_ns_after", "wall_ms"],
}


def _read(path: Path, kind: str) -> pd.DataFrame:
    # A missing value makes CSV inference use float64, irreversibly rounding large clock
    # integers before the later Int64 conversion. Parse clock columns as strings first.
    df = read_csv_safe(path, dtype={c: "string" for c in _INT_NS.get(kind, [])})
    for c in _INT_NS.get(kind, []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


@dataclass
class Session:
    path: Path
    meta: dict
    frames: pd.DataFrame
    segments: pd.DataFrame
    gnss: pd.DataFrame
    imu: pd.DataFrame
    events: pd.DataFrame
    exposure: pd.DataFrame
    clock: pd.DataFrame

    @classmethod
    def load(cls, path: Path) -> "Session":
        path = Path(path)
        meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
        major = int(str(meta.get("spec_version", "0")).split(".")[0])
        if major != SUPPORTED_MAJOR:
            raise ValueError(f"unsupported spec_version {meta.get('spec_version')} in {path}")
        frames = _read(path / "frames.csv", "frames")
        if not frames.empty:
            frames = frames.dropna(subset=["segment", "frame_in_segment", "t_ns"]).copy()
            frames["segment"] = frames["segment"].astype(int)
            frames["frame_in_segment"] = frames["frame_in_segment"].astype(int)
            frames["frame_global"] = frames["frame_global"].astype(int)
        return cls(
            path=path,
            meta=meta,
            frames=frames,
            segments=_read(path / "segments.csv", "segments"),
            gnss=_read(path / "gnss_fixes.csv", "gnss"),
            imu=_read(path / "imu.csv", "imu"),
            events=_read(path / "events.csv", "events"),
            exposure=_read(path / "exposure_control.csv", "exposure"),
            clock=_read(path / "clock_sync.csv", "clock"),
        )

    # ---- identity -------------------------------------------------------------------------
    @property
    def session_id(self) -> str:
        return str(self.meta.get("session_id", self.path.name))

    @property
    def session_type(self) -> str:
        return str(self.meta.get("session_type", "survey"))

    @property
    def device_id(self) -> str:
        return str(self.meta.get("device", {}).get("device_id", ""))

    @property
    def camera(self) -> dict:
        return self.meta.get("camera", {}) or {}

    @property
    def mount(self) -> dict:
        return self.meta.get("mount", {}) or {}

    @property
    def survey(self) -> dict:
        return self.meta.get("survey", {}) or {}

    @property
    def orientation_verdict(self) -> str:
        return str((self.meta.get("orientation") or {}).get("verdict", "UNKNOWN"))

    @property
    def trusted(self) -> bool:
        if self.gnss.empty or "is_mock" not in self.gnss.columns:
            return True
        return not bool((pd.to_numeric(self.gnss["is_mock"], errors="coerce").fillna(0) > 0).any())

    @property
    def t0_ns(self) -> int:
        """Origin of the pipeline's internal time axis: the first frame's exposure start."""
        if not self.frames.empty:
            return int(self.frames["t_ns"].min())
        return int((self.meta.get("start") or {}).get("t_ns") or 0)

    # ---- files ----------------------------------------------------------------------------
    def video_files(self) -> dict[int, Path]:
        segs = sorted(set(int(s) for s in self.frames["segment"].unique())) if not self.frames.empty else []
        return {s: self.path / "video" / f"seg_{s:04d}.mp4" for s in segs}

    # ---- intervals from events and exposure ------------------------------------------------
    def _events_of(self, *types: str) -> pd.DataFrame:
        if self.events.empty or "type" not in self.events.columns:
            return pd.DataFrame(columns=["t_ns", "type", "value_json"])
        return self.events[self.events["type"].isin(types)].sort_values("t_ns")

    def black_intervals_ns(self) -> list[tuple[int, int]]:
        """Spans between `image_black` and the next `image_ok` (lens covered, camera muted)."""
        out: list[tuple[int, int]] = []
        start: int | None = None
        for _, r in self._events_of("image_black", "image_ok").iterrows():
            if r["type"] == "image_black" and start is None:
                start = int(r["t_ns"])
            elif r["type"] == "image_ok" and start is not None:
                out.append((start, int(r["t_ns"])))
                start = None
        if start is not None:
            out.append((start, np.iinfo(np.int64).max))
        return out

    def km_markers(self) -> list[tuple[int, str]]:
        out = []
        for _, r in self._events_of("operator_marker").iterrows():
            try:
                v = json.loads(r.get("value_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                v = {}
            if v.get("marker_type") == "km_stone":
                out.append((int(r["t_ns"]), str(v.get("note", ""))))
        return out

    def low_light_at(self, t_ns: np.ndarray) -> np.ndarray:
        """`low_light` flag of the exposure decision in force at each time (step function)."""
        t_ns = np.asarray(t_ns, dtype=np.int64)
        ex = self.exposure
        if ex.empty or "low_light" not in ex.columns:
            return np.zeros(t_ns.shape, dtype=bool)
        ex = ex.dropna(subset=["t_ns"]).sort_values("t_ns")
        times = ex["t_ns"].to_numpy(dtype=np.int64)
        flags = pd.to_numeric(ex["low_light"], errors="coerce").fillna(0).to_numpy() > 0
        idx = np.searchsorted(times, t_ns, side="right") - 1
        res = np.zeros(t_ns.shape, dtype=bool)
        ok = idx >= 0
        res[ok] = flags[idx[ok]]
        return res
