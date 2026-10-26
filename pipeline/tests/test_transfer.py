"""The USB verification gate must never approve malformed or out-of-session manifests."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "pull_sessions", Path(__file__).resolve().parents[2] / "tools" / "pull_sessions.py"
)
pull = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pull)


def _entry(name, data=b"session data"):
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("manifest", [None, [], {}, {"files": []}, {"files": [None]}, {"files": [{}]}])
def test_malformed_manifest_is_not_verified(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ok, note, count = pull.verify(tmp_path)
    assert not ok and count == 0


def test_invalid_json_is_a_verification_failure(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b'{"files":')
    assert pull.verify(tmp_path)[0] is False


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "C:/outside.txt", "..\\outside.txt"])
def test_manifest_cannot_verify_files_outside_session(tmp_path, name):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [_entry(name)]}))
    assert pull.verify(tmp_path)[0] is False


def test_manifest_cannot_follow_symlink_outside_session(tmp_path):
    source = tmp_path / "session"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"session data")
    (source / "linked.txt").symlink_to(outside)
    (source / "manifest.json").write_text(json.dumps({"files": [_entry("linked.txt")]}))
    assert pull.verify(source)[0] is False


def test_valid_manifest_and_checksum_mismatch(tmp_path):
    data = tmp_path / "session.json"
    data.write_bytes(b"session data")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [_entry("session.json")]}))
    assert pull.verify(tmp_path) == (True, "checksums ok; signature absent", 1)
    data.write_bytes(b"corrupt data")
    assert pull.verify(tmp_path)[0] is False


def test_duplicate_manifest_entries_are_not_verified(tmp_path):
    (tmp_path / "session.json").write_bytes(b"session data")
    entry = _entry("session.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [entry, entry]}))
    assert pull.verify(tmp_path)[0] is False
