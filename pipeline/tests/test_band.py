"""Segment bands (vlm.image = band, docs/pipeline/13 §6): planning, export, import and analysis.

Band sizes and the processor's sizes are worked out by hand below (smart_resize written out with the
numbers), not with qwen.smart_resize; pixels are compared with the frame PNG's own rows.
"""
import copy
import hashlib
import json
import math

import numpy as np
import pytest
from PIL import ExifTags, Image

from geoai.analyze import analyze
from geoai.config import Config
from geoai.qwen import plan_band
from geoai.records import read_png_record
from geoai.segment_records import (check_segment_png, load_segment_records, record_matches_embedded,
                                   validate_segment_record)
from geoai.util import sha256_file
from geoai.vlm import export_batch, import_batch, verify_sums

from test_analyze import ANSWERS, build_runs, rid, write_answers


# ---- planning ------------------------------------------------------------------------------------

def test_band_of_a_near_segment_keeps_its_rows_and_a_tenth_above():
    # Rows 801..2159 (centres in [800.3, 2170.0], cut at the frame's bottom): 1359 rows; margin ceil(135.9) = 136.
    p = plan_band(3840, 2160, (800.3, 2170.0), margin_frac=0.10, max_aspect=2.94)
    assert (p.y0, p.y1, p.height, p.margin_rows, p.extra_rows_below, p.extra_rows_above) == (665, 2160, 1495, 136, 0, 0)
    assert p.box == (0, 665, 3840, 2160)
    # 1495 x 3840 > 1,048,576: beta = sqrt(1495 * 3840 / 1048576) = 2.33984; 1495 / beta / 28 = 22.82 -> 616;
    # 3840 / beta / 28 = 58.61 -> 1624. 616 x 1624 = 1,000,384 px, 22 x 58 = 1276 tokens; scale 0.4229 (frame: 0.35).
    assert (p.model_w, p.model_h, p.tokens) == (1624, 616, 1276)


def test_short_band_grows_down_to_the_frame_bottom_then_up_to_the_aspect_limit():
    # Rows 1500..2100 (601) + margin 61 = 1439..2101, 662 rows; 3840 / 2.94 = 1306.1 -> at least 1307 rows.
    # Down first: the 59 rows to the bottom (2160); then up 1307 - 721 = 586 rows: 853..2160.
    p = plan_band(3840, 2160, (1500.0, 2100.0), margin_frac=0.10, max_aspect=2.94)
    assert (p.y0, p.y1, p.margin_rows, p.extra_rows_below, p.extra_rows_above) == (853, 2160, 61, 59, 586)
    assert 3840 / p.height <= 2.94
    # beta = sqrt(1307 * 3840 / 1048576) = 2.18778: 1307 / beta / 28 = 21.34 -> 588; 3840 / beta / 28 = 62.69 -> 1736.
    assert (p.model_w, p.model_h, p.tokens) == (1736, 588, 21 * 62)


def test_band_margin_stops_at_the_top_and_bad_input_is_refused():
    p = plan_band(3840, 2160, (20.0, 1800.0))
    assert (p.y0, p.y1, p.margin_rows) == (0, 1801, 20)  # 178 rows wanted, 20 exist
    with pytest.raises(ValueError):
        plan_band(3840, 2160, (900.0, 800.0))
    with pytest.raises(ValueError):
        plan_band(3840, 2160, (2200.0, 2300.0))   # below the frame
    with pytest.raises(ValueError):
        plan_band(3840, 2160, (math.nan, 800.0))


# ---- export, import, analysis --------------------------------------------------------------------

def _band_cfg() -> Config:
    cfg = Config()
    cfg.vlm.image = "band"
    return cfg


@pytest.fixture()
def band_batch(tmp_path):
    runs = build_runs(tmp_path)
    batch = tmp_path / "batch_band"
    info = export_batch([runs["SESS_A"], runs["SESS_B"]], batch, _band_cfg())
    return runs, batch, info


def test_band_export_cuts_the_frame_rows_losslessly(band_batch):
    runs, batch, info = band_batch
    assert info["image"] == "band" and info["band"] == {"margin_frac": 0.10, "max_aspect": 2.94}
    assert info["n_requests"] == 14 and "full-width band" in info["image_rules"]
    reqs = [json.loads(x) for x in (batch / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in reqs:
        rec = json.loads((batch / "records" / f"{r['request_id']}.json").read_text(encoding="utf-8"))
        # Test frames are 112 x 63 with the segment on rows [10, 55]: rows 10..55 (46) + ceil(4.6) = 5 above
        # -> 5..56, 51 rows (112 / 2.94 = 38.1, so no growth). The processor: 56 x 112 = 6272 < 7168 px, so
        # beta = sqrt(7168 / (51 x 112)) = 1.12022: 51 x beta / 28 = 2.04 -> 84, 112 x beta / 28 = 4.48 -> 140.
        assert r["kind"] == "segment_band" and (r["width"], r["height"]) == (112, 51) and r["model_input"] == [140, 84]
        b = rec["vlm_input"]
        assert b["box"] == [0, 5, 112, 56] and b["tokens"] == 15 and b["rows_of_segment"] == [5.0, 50.0]
        assert (b["margin_rows"], b["extra_rows_below"], b["extra_rows_above"]) == (5, 0, 0)
        assert b["sha256"] == r["image_sha256"] == sha256_file(batch / r["image"]) and b["file"] == r["image"]
        assert validate_segment_record(rec) == []
        with Image.open(runs[r["session_id"]] / rec["frame"]["file"]) as f, Image.open(batch / r["image"]) as im:
            assert np.array_equal(np.asarray(im), np.asarray(f)[5:56]) and im.mode == "RGB"
            uid = im.getexif().get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.ImageUniqueID]
        assert uid == hashlib.sha256(f"{rec['id']}:band".encode()).hexdigest()[:32]
        assert check_segment_png(batch / r["image"], rec, offset_minutes=330) == []
        embedded, _ = read_png_record(batch / r["image"])
        assert embedded["vlm_input"]["sha256"] is None and record_matches_embedded(rec, embedded) == []
    # Records of segments without a request carry no band; the runs are untouched.
    for rec in load_segment_records(batch / "records"):
        assert ("vlm_input" in rec) == (rec["status"] in ("send", "audit"))
    for run in runs.values():
        assert not any("vlm_input" in r for r in load_segment_records(run / "sidecars"))
    assert verify_sums(batch) == []
    assert "14 road-segment bands" in (batch / "RUNNING.txt").read_text(encoding="utf-8")


def test_band_record_checks(band_batch):
    _, batch, _ = band_batch
    rec = json.loads(next((batch / "records").glob("SESS_A_seg00000.json")).read_text(encoding="utf-8"))
    for mutate, expect in [
        (lambda r: r["vlm_input"].__setitem__("box", [0, 5, 100, 56]), "full-width band"),
        (lambda r: r["vlm_input"].__setitem__("model_input", [112, 56]), "model_input"),
        (lambda r: r["vlm_input"].__setitem__("rows_of_segment", [10.0, 55.0]), "rows_of_segment"),
        (lambda r: r["vlm_input"].__setitem__("sha256", None), "vlm_input.sha256"),
        (lambda r: r["vlm_input"].pop("tokens"), "missing"),
        (lambda r: r.__setitem__("status", "skip"), "only for sent segments"),
    ]:
        bad = copy.deepcopy(rec)
        mutate(bad)
        assert any(expect in p for p in validate_segment_record(bad)), expect


def test_band_export_of_chosen_ids(tmp_path):
    runs = build_runs(tmp_path)
    ids = [rid("SESS_A", 1), rid("SESS_B", 3)]
    info = export_batch(list(runs.values()), tmp_path / "b", _band_cfg(), ids=ids)
    assert info["n_requests"] == 2 and info["ids_requested"] == 2 and info["n_records"] == 20
    with pytest.raises(ValueError, match="not sent records"):
        export_batch(list(runs.values()), tmp_path / "b2", _band_cfg(), ids=[rid("SESS_A", 3)])  # a skip segment
    assert not (tmp_path / "b2").exists()
    frames = export_batch(list(runs.values()), tmp_path / "b3", Config(), ids=ids)  # ids work for frames too
    assert frames["image"] == "frame" and frames["n_requests"] == 2
    cfg = Config()
    cfg.vlm.image = "crop"
    with pytest.raises(ValueError, match="vlm.image"):
        export_batch(list(runs.values()), tmp_path / "b4", cfg)


def test_band_answers_import_and_analyze(band_batch):
    runs, batch, _ = band_batch
    write_answers(batch)
    s = import_batch(batch, list(runs.values()), _band_cfg())
    assert s["runs_updated"] == {"SESS_A": 7, "SESS_B": 6} and s["run_problems"] == []
    for sid, run in runs.items():
        for r in load_segment_records(run / "sidecars"):
            assert validate_segment_record(r) == []
            if (sid, r["segment"]["k"]) in ANSWERS:
                assert r["vlm"]["image"] == "band"
    analyze(batch, _band_cfg())
    report = (batch / "analysis" / "REPORT.md").read_text(encoding="utf-8")
    # 5 m over rows 10..55 (45 rows) of a 51-row band the processor makes 84 rows high: 5000 / (45 x 84 / 51).
    assert f"{5000 / (45 * 84 / 51):.1f} mm" in report and "full-width band" in report
