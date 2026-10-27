"""frames._Progress: the throttled `progress {json}` log lines the VLM system's video toggle tails."""
import json
import logging

import pytest

from geoai import frames


def _lines(caplog) -> list[dict]:
    out = []
    for r in caplog.records:
        msg = r.getMessage()
        if msg.startswith("progress "):
            out.append(json.loads(msg[len("progress "):]))
    return out


def test_progress_is_throttled_within_a_phase_and_written_on_phase_change(caplog, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(frames.time, "monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger=frames.log.name)
    p = frames._Progress(every_s=2.0)
    p.update("bank", 1, 90)            # first line of a phase: written
    clock[0] += 1.0
    p.update("bank", 2, 90)            # 1 s later, same phase: dropped
    clock[0] += 1.0
    p.update("bank", 3, 90)            # 2 s after the last line: written
    p.update("bank_finalize", 3, 3)    # phase change: written at once
    p.update("decode", 1, 528, segment="S_seg00000", k=0, status="send", send=1)
    p.update("decode", 2, 528, segment="S_seg00001", k=1, status="skip", send=1, skip=1)   # dropped
    p.update("decode", 2, 528, force=True, send=1, skip=1)                                 # forced
    got = [(d["phase"], d["done"], d["total"]) for d in _lines(caplog)]
    assert got == [("bank", 1, 90), ("bank", 3, 90), ("bank_finalize", 3, 3), ("decode", 1, 528), ("decode", 2, 528)]
    first_decode = _lines(caplog)[3]
    assert first_decode["segment"] == "S_seg00000" and first_decode["status"] == "send" and first_decode["send"] == 1


@pytest.mark.parametrize("total", [None, 0, 12])
def test_progress_line_is_one_json_object(caplog, total):
    caplog.set_level(logging.INFO, logger=frames.log.name)
    frames._Progress().update("plan", 0, total, route_m=1149.8)
    (d,) = _lines(caplog)
    assert d == {"phase": "plan", "done": 0, "total": total, "route_m": 1149.8}
