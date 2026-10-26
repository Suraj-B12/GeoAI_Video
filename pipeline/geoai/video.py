"""Decoding session video and joining every sample to its frames.csv row (docs/pipeline/03).

Decoded sample i of seg_NNNN.mp4 <-> frames.csv row (segment = NNNN, frame_in_segment = i).
Container PTS are used only as keys to find i (they drift from capture times); times come from
frames.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pandas as pd
from av.video.reformatter import VideoReformatter

from .util import log


@dataclass
class DecodedFrame:
    segment: int
    index: int              # 0-based sample index in the segment file
    row: dict               # the frames.csv row
    rgb: np.ndarray | None  # H x W x 3 uint8 in display orientation (None if not requested)
    keyframe: bool
    width: int
    height: int


class JoinError(RuntimeError):
    pass


def segment_info(path: Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return {"width": s.codec_context.width, "height": s.codec_context.height,
                "codec": s.codec_context.name, "time_base": str(s.time_base)}


def to_rgb(frame: av.VideoFrame, reformatter: VideoReformatter) -> np.ndarray:
    """BT.709 limited range -> full-range RGB, explicitly (untagged streams would default to BT.601)."""
    rgb = reformatter.reformat(frame, format="rgb24", src_colorspace="ITU709",
                               src_color_range="MPEG", dst_color_range="JPEG")
    return rgb.to_ndarray()


def iter_segment(path: Path, rows: pd.DataFrame, *, segment: int, keyframes_only: bool = False,
                 rotate180: bool = False, want=None, allow_extra_rows: bool = False,
                 threads: str = "AUTO") -> Iterator[DecodedFrame]:
    """Yield decoded frames of one segment, each joined to its frames.csv row.

    want: optional callable(index, row) -> bool. RGB is produced only for frames it accepts
          (decoding still visits every frame; conversion is the expensive part at 4K).
    allow_extra_rows: a crash-recovered last segment may have more rows than samples.
    """
    rows = rows.sort_values("frame_in_segment")
    if not rows["frame_in_segment"].is_monotonic_increasing or rows["frame_in_segment"].duplicated().any():
        raise JoinError(f"segment {segment}: frame_in_segment not unique/increasing")
    by_index = {int(r["frame_in_segment"]): r for r in rows.to_dict("records")}
    reformatter = VideoReformatter()
    index_of_pts: dict[int, int] = {}
    n_samples = 0
    seen = 0
    last_pts = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = threads
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        for packet in container.demux(stream):
            if packet.pts is not None:
                if packet.pts in index_of_pts:
                    raise JoinError(f"segment {segment}: duplicate container PTS {packet.pts}")
                if last_pts is not None and packet.pts <= last_pts:
                    raise JoinError(f"segment {segment}: container PTS not strictly increasing (B-frames/reordering)")
                last_pts = packet.pts
                index_of_pts[packet.pts] = n_samples
                n_samples += 1
            for frame in packet.decode():
                idx = index_of_pts.get(frame.pts)
                if idx is None:
                    raise JoinError(f"segment {segment}: decoded frame with unknown PTS {frame.pts}")
                row = by_index.get(idx)
                if row is None:
                    raise JoinError(f"segment {segment}: sample {idx} has no frames.csv row "
                                    f"({len(by_index)} rows): more samples than rows")
                seen += 1
                rgb = None
                if want is None or want(idx, row):
                    rgb = to_rgb(frame, reformatter)
                    if rotate180:
                        rgb = np.ascontiguousarray(rgb[::-1, ::-1])
                yield DecodedFrame(segment=segment, index=idx, row=row, rgb=rgb,
                                   keyframe=bool(frame.key_frame), width=frame.width, height=frame.height)
    if n_samples > len(by_index):
        raise JoinError(f"segment {segment}: {n_samples} samples but {len(by_index)} rows")
    if n_samples < len(by_index):
        msg = f"segment {segment}: {len(by_index) - n_samples} frames.csv rows without a sample (trimmed)"
        if not allow_extra_rows:
            raise JoinError(msg)
        log.warning(msg)
    if not keyframes_only and seen != n_samples:
        log.warning("segment %d: %d of %d samples could not be decoded", segment, n_samples - seen, n_samples)


def count_samples(path: Path) -> int:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return sum(1 for p in c.demux(s) if p.pts is not None)
