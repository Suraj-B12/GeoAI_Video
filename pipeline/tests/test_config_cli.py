import json
import subprocess
import sys
from pathlib import Path

import pytest

from geoai.config import load_config


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"extract": {"budget_per_kmm": 10}}))
    with pytest.raises(KeyError, match="budget_per_kmm"):
        load_config(p)


def test_overrides_apply_and_tuples_convert():
    cfg = load_config(overrides={"extract": {"budget_per_km": 12.5}, "anomaly": {"layers": [6, 9]}})
    assert cfg.extract.budget_per_km == 12.5
    assert cfg.anomaly.layers == (6, 9)
    assert cfg.crop.max_pixels == 1_048_576  # untouched defaults stay


def test_cli_help_runs():
    r = subprocess.run([sys.executable, "-m", "geoai", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "vlm-export" in r.stdout


def test_relative_paths_in_a_config_file_are_relative_to_it(tmp_path, monkeypatch):
    """gate.calibration, anomaly.weights and anomaly.probe given relative in a JSON file resolve against the file's
    folder (wherever the CLI runs); absolute paths, null and the overrides dict are taken as given."""
    d = tmp_path / "calibration"
    d.mkdir()
    p = d / "c.json"
    p.write_text(json.dumps({"gate": {"calibration": "gate_x.json"}, "anomaly": {"probe": None}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(p)
    assert cfg.gate.calibration == str((d / "gate_x.json").resolve()) and cfg.anomaly.probe is None
    assert cfg.anomaly.weights == load_config().anomaly.weights               # not in the file: default
    absolute = str((tmp_path / "elsewhere" / "w.pth").resolve())
    p.write_text(json.dumps({"anomaly": {"weights": absolute}, "gate": {"calibration": "../g.json"}}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.anomaly.weights == absolute and cfg.gate.calibration == str((tmp_path / "g.json").resolve())
    assert load_config(overrides={"gate": {"calibration": "rel.json"}}).gate.calibration == "rel.json"


def test_committed_probe_gate_override_loads_its_calibration(monkeypatch, tmp_path):
    """pipeline/calibration/segments_probe_gate.json: the S21 calibration, probe_frac only, from any directory."""
    from geoai.config import PIPELINE_DIR
    from geoai.gate import gate_thresholds, read_calibration

    monkeypatch.chdir(tmp_path)
    cfg = load_config(PIPELINE_DIR / "calibration" / "segments_probe_gate.json")
    cal = read_calibration(Path(cfg.gate.calibration))
    th = gate_thresholds(cfg)
    assert th["source"] == "calibration" and th["evidence"] is None and th["crack_mm"] is None
    assert th["probe_frac"] == cal["any_of"]["thresholds"]["probe_frac"] > 0
    assert cal["n_rows"] == 813 and cal["n_positive"] == 164 and cal["source"]["gated_rows"] == 0
