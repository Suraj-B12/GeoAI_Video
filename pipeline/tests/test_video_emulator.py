"""Join test on a real app recording (from the Android emulator).

Set GEOAI_EMULATOR_SESSION to a pulled session folder to run it; skipped otherwise.
"""
import os
from pathlib import Path

import pytest

from geoai.session import Session
from geoai.video import iter_segment

SESSION = os.environ.get("GEOAI_EMULATOR_SESSION")
pytestmark = pytest.mark.skipif(not SESSION or not Path(SESSION).exists(), reason="no emulator session")


def test_every_sample_joins_to_its_row():
    s = Session.load(Path(SESSION))
    for seg, path in s.video_files().items():
        rows = s.frames[s.frames["segment"] == seg]
        got = [f.index for f in iter_segment(path, rows, segment=seg, want=lambda i, r: i % 50 == 0)]
        assert got == list(range(len(rows)))


def test_keyframes_only_decode_returns_keyframe_rows():
    s = Session.load(Path(SESSION))
    seg, path = next(iter(s.video_files().items()))
    rows = s.frames[s.frames["segment"] == seg]
    frames = list(iter_segment(path, rows, segment=seg, keyframes_only=True))
    assert frames, "no keyframes decoded"
    assert all(int(f.row["is_keyframe"]) == 1 for f in frames)
    assert frames[0].rgb.shape == (frames[0].height, frames[0].width, 3)
