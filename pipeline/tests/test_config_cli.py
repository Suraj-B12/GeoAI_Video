import json
import subprocess
import sys

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
