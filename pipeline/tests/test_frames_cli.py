"""Fast checks of the segment-mode CLI glue: gate-calibrate's input rows, the structured run log on failure,
and the commands the VLM machine runs without torch.

The answered batch is test_analyze's (two runs, known answers); the expected rows are counted by hand from
its ANSWERS table: 13 answered segments, one without a valid label ("N/A"), so 12 rows, 8 of them
VLM-distress; SESS_A is a gate-mode run (4 send + 2 audit answered and usable), SESS_B a send-all run (6).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from geoai.analyze import calibration_rows
from geoai.cli import main
from geoai.config import Config

from test_analyze import answered_batch  # noqa: F401 - pytest fixture

PIPELINE = Path(__file__).resolve().parent.parent


def test_calibration_rows_from_an_answered_batch(answered_batch):  # noqa: F811
    _, batch, _, _ = answered_batch
    rows, src = calibration_rows([batch], Config())
    assert len(rows) == 12 and sum(r["vlm_distress"] for r in rows) == 8
    assert src["by_mode_and_status"] == {"all:send": 6, "gate:audit": 2, "gate:send": 4}
    assert src["gated_rows"] == 6 and src["sessions"] == ["SESS_A", "SESS_B"]
    assert src["batches"][0]["no_valid_label"] == 1 and src["batches"][0]["usable"] == 12
    assert all(set(r["scores"]) >= {"probe_frac", "evidence", "crack_mm"} for r in rows)
    # The same batch twice: every segment once.
    rows2, src2 = calibration_rows([batch, batch], Config())
    assert rows2 == rows and src2["duplicates_replaced"] == 12


def test_gate_calibrate_cli_without_torch(answered_batch, tmp_path):  # noqa: F811
    _, batch, _, _ = answered_batch
    script = ("import sys\n"
              "for m in ('torch', 'cv2', 'av', 'shapely', 'scipy', 'skimage', 'pymap3d'):\n"
              "    sys.modules[m] = None\n"
              "from geoai.cli import main\n"
              "sys.exit(main(sys.argv[1:]))\n")
    r = subprocess.run([sys.executable, "-c", script, "gate-calibrate", str(batch), "--target-recall", "0.75"],
                       capture_output=True, text=True, timeout=120, cwd=PIPELINE)
    assert r.returncode == 0, r.stderr
    assert "gate-mode runs" in r.stderr                      # the mixed input is pointed out
    cal = json.loads((batch / "gate_calibration.json").read_text(encoding="utf-8"))
    assert cal["n_rows"] == 12 and cal["n_positive"] == 8 and cal["target_recall"] == 0.75
    assert cal["source"]["gated_rows"] == 6


def test_failed_run_is_logged_with_its_stack_trace(tmp_path, capsys):
    status = main(["process", str(tmp_path / "no_such_session"), "--out", str(tmp_path / "out")])
    assert status == 1 and "FAILED" in capsys.readouterr().err
    logs = list((tmp_path / "out" / "logs").glob("*/no_such_session.jsonl"))
    assert len(logs) == 1
    lines = [json.loads(x) for x in logs[0].read_text(encoding="utf-8").splitlines()]
    assert [(x["stage"], x["event"]) for x in lines] == [("run", "start"), ("load", "start"), ("load", "error"),
                                                         ("run", "failed")]
    err = lines[2]
    assert err["error_type"] == "FileNotFoundError" and "Traceback" in err["traceback"]
    assert "session.json" in err["error"]


def test_cli_mode_checks(tmp_path, capsys):
    assert main(["process", str(tmp_path), "--out", str(tmp_path / "o"), "--mode", "crops", "--send-all"]) == 2
    with pytest.raises(SystemExit):
        main(["vlm-import", str(_crop_batch(tmp_path))])    # a crop batch needs its runs


def _crop_batch(tmp_path: Path) -> Path:
    b = tmp_path / "crop_batch"
    b.mkdir()
    (b / "batch.json").write_text(json.dumps({"schema": "geoai.vlm_batch.v1"}), encoding="utf-8")
    return b
