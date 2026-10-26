"""Reject frame reordering before associating video pixels with capture metadata."""
from fractions import Fraction
import av
import numpy as np
import pandas as pd
import pytest

from geoai.video import JoinError, iter_segment


def _video(path, b_frames):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"bf": str(b_frames), "g": "12", "b-adapt": "0", "sc_threshold": "0"}
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 20, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


@pytest.mark.parametrize("keyframes_only", [False, True])
def test_b_frames_are_rejected_before_join(tmp_path, keyframes_only):
    video = tmp_path / "reordered.mp4"
    _video(video, b_frames=2)
    with av.open(str(video)) as container:
        pts = [p.pts for p in container.demux(video=0) if p.pts is not None]
    assert any(b < a for a, b in zip(pts, pts[1:])), "fixture must contain reordered packets"
    rows = pd.DataFrame({"frame_in_segment": range(12)})
    with pytest.raises(JoinError, match="PTS.*increasing"):
        list(iter_segment(video, rows, segment=0, keyframes_only=keyframes_only, want=lambda *_: False))


def test_normal_video_preserves_every_sample_join(tmp_path):
    video = tmp_path / "ordered.mp4"
    _video(video, b_frames=0)
    rows = pd.DataFrame({"frame_in_segment": range(12), "capture_key": range(100, 112)})
    frames = list(iter_segment(video, rows, segment=0, want=lambda *_: False))
    assert [f.index for f in frames] == list(range(12))
    assert [f.row["capture_key"] for f in frames] == list(range(100, 112))
