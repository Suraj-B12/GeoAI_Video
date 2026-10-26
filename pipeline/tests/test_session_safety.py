"""Session inputs must survive invalid output paths and retain their exact clock values."""
import json

import pandas as pd
import pytest

from geoai.config import Config
from geoai.extract import ExtractionError, process_session
from geoai.session import _read


@pytest.mark.parametrize("layout", ["same", "ancestor", "inside", "symlink"])
def test_extraction_refuses_source_output_overlap(tmp_path, layout):
    source = tmp_path / "S"
    if layout == "ancestor":
        source /= "source"
    source.mkdir(parents=True)
    metadata = source / "session.json"
    original = json.dumps({"spec_version": "1.1", "session_type": "survey", "session_id": "S"})
    metadata.write_text(original)
    output = source / "runs" if layout == "inside" else tmp_path
    if layout == "symlink":
        output = tmp_path / "runs"
        output.mkdir()
        (output / "S").symlink_to(source, target_is_directory=True)
    with pytest.raises(ExtractionError, match="overlap"):
        process_session(source, output, Config())
    assert metadata.read_text() == original


@pytest.mark.parametrize("session_id", ["", ".", "..", "../elsewhere", "/absolute", "..\\elsewhere", "C:elsewhere"])
def test_extraction_refuses_session_id_paths(tmp_path, session_id):
    source = tmp_path / "source"
    source.mkdir()
    metadata = source / "session.json"
    metadata.write_text(json.dumps({"spec_version": "1.1", "session_type": "survey", "session_id": session_id}))
    with pytest.raises(ExtractionError, match="single folder name"):
        process_session(source, tmp_path / "runs", Config())
    assert metadata.is_file()


def test_nullable_session_timestamps_keep_nanosecond_precision(tmp_path):
    path = tmp_path / "imu.csv"
    ns = 10_000_000_000_000_001
    path.write_text(f"t_ns,sensor,x,y,z\n{ns},acc,1,2,3\n,acc,1,2,3\n{ns + 1},acc,1,2,3\n")
    rows = _read(path, "imu")
    assert rows["t_ns"].dtype == pd.Int64Dtype()
    assert int(rows.loc[0, "t_ns"]) == ns
    assert pd.isna(rows.loc[1, "t_ns"])
    assert int(rows.loc[2, "t_ns"]) == ns + 1
