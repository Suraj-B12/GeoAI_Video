"""The diagnostic CLI must report corrupt sessions instead of crashing or approving them."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "validate_session.py"
_spec = importlib.util.spec_from_file_location("validate_session", SCRIPT)
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


@pytest.mark.parametrize("manifest", [None, [], {}, {"files": []}, {"files": [None]}, {"files": [{}]}])
def test_malformed_manifest_reports_failure(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = validator.Report()
    validator.check_manifest(tmp_path, report)
    assert report.failed


def test_manifest_cannot_validate_an_external_file(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    data = b"outside"
    (tmp_path / "outside").write_bytes(data)
    manifest = {"files": [{"path": "../outside", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
    (session / "manifest.json").write_text(json.dumps(manifest))
    report = validator.Report()
    validator.check_manifest(session, report)
    assert report.failed


def test_valid_manifest_still_verifies_local_file(tmp_path):
    data = b'{"spec_version":"1.1"}'
    (tmp_path / "session.json").write_bytes(data)
    manifest = {"files": [{"path": "session.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = validator.Report()
    validator.check_manifest(tmp_path, report)
    assert not report.failed
    assert any(i["level"] == "PASS" and i["check"] == "checksums" for i in report.items)


def test_duplicate_frame_times_report_failure_without_dividing_by_zero():
    report = validator.Report()
    row = {"t_ns": "100", "sensor_ts_ns": "100", "match_residual_ns": "0", "exposure_ns": "1"}
    assert validator.check_frames([row, row], report) == (100, 100)
    assert report.failed


def test_csv_preserves_newlines_inside_quoted_fields(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text('t_ns,note\n100,"first\nsecond"\n')
    assert validator.read_csv(path) == [{"t_ns": "100", "note": "first\nsecond"}]


def test_csv_ignores_partial_utf8_in_torn_final_line(tmp_path):
    path = tmp_path / "events.csv"
    path.write_bytes(b't_ns,note\n100,complete\n200,\xe2\x82')
    assert validator.read_csv(path) == [{"t_ns": "100", "note": "complete"}]


def test_video_validator_preserves_packet_order(tmp_path, monkeypatch):
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "seg_0000.mp4").touch()
    monkeypatch.setattr(validator.shutil, "which", lambda _: "ffprobe")
    monkeypatch.setattr(validator, "ffprobe_packets", lambda _: (
        {"codec_name": "h264", "width": 64, "height": 64}, Fraction(1, 30),
        [{"pts": 0, "flags": "K"}, {"pts": 2}, {"pts": 1}],
    ))
    frames = [{"segment": "0", "pts_us": str(i * 33333)} for i in range(3)]
    report = validator.Report()
    validator.check_video(tmp_path, frames, report)
    assert any(i["level"] == "FAIL" and "PTS order" in i["check"] for i in report.items)


@pytest.mark.parametrize("metadata", [None, [], {"spec_version": "invalid"},
                                      {"spec_version": "1.1", "end": None, "camera": []}])
def test_cli_malformed_input_exits_cleanly(tmp_path, metadata):
    (tmp_path / "session.json").write_text(json.dumps(metadata))
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode in (1, 2)
    assert "Traceback" not in result.stderr


def test_cli_writes_report_when_csv_is_missing_required_columns(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"spec_version": "1.1"}))
    (tmp_path / "frames.csv").write_text("wrong_header\n123\n")
    target = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--json", str(target)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert any(item["level"] == "FAIL" for item in json.loads(target.read_text())["items"])
