"""Duplicate merging on the ground, the GPU Gaussian blur, and the parallel PNG writer."""
import hashlib

import cv2
import numpy as np
import pytest
import torch
from shapely.geometry import box

from geoai import blur
from geoai.config import Config
from geoai.extract import _dedupe, _Rec
from geoai.records import PngPool, write_png


def _rec(i, frame, x, y, w, h, ev):
    """A record whose ground footprint is a w x h m box with its lower-left corner at (x, y)."""
    poly = box(x, y, x + w, y + h)
    c = poly.centroid
    return _Rec({"id": f"r{i}", "status": "candidate"}, None, (0, 0, 1, 1), ev, frame, (c.x, c.y), poly)


def test_same_road_seen_from_several_frames_merges_to_the_strongest():
    # One 0.3 x 1.4 m crack seen from 3 consecutive frames, the views shifted by up to 0.2 m.
    recs = [_rec(0, 0, 0.0, 0.0, 0.3, 1.4, 20.0), _rec(1, 1, 0.05, 0.1, 0.3, 1.4, 35.0),
            _rec(2, 2, 0.1, 0.2, 0.3, 1.2, 12.0)]
    _dedupe(recs, Config())
    status = {r.rec["id"]: r.rec["status"] for r in recs}
    assert status == {"r0": "duplicate", "r1": "candidate", "r2": "duplicate"}
    assert recs[0].rec["duplicate_of"] == "r1" and recs[2].rec["duplicate_of"] == "r1"
    assert sorted(recs[1].rec["duplicates"]) == ["r0", "r2"]


def test_dense_separate_distresses_are_all_kept():
    # 300 potholes of 0.2 x 0.2 m, 0.33 m apart along 100 m of road: 3000 per km, none merged.
    recs = [_rec(i, i // 3, 0.0, i * 0.33, 0.2, 0.2, 10.0 + i % 7) for i in range(300)]
    _dedupe(recs, Config())
    assert all(r.rec["status"] == "candidate" for r in recs)


def test_chain_of_overlaps_is_not_transitive():
    # A carpet of alligator cracking: 1 m boxes every 0.6 m, each overlapping its neighbours by 40 %.
    recs = [_rec(i, i, 0.0, i * 0.6, 1.0, 1.0, 50.0 - i) for i in range(20)]
    _dedupe(recs, Config())
    kept = [r for r in recs if r.rec["status"] == "candidate"]
    assert len(kept) == 20  # 40 % < dedupe_overlap 50 %: separate stretches of road


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("sigma", [12.0, 24.0, 40.0])
def test_gpu_blur_matches_opencv(sigma):
    rng = np.random.default_rng(0)
    img = (rng.random((700, 2758)) * 255).astype(np.float32)
    ref = cv2.GaussianBlur(img, (0, 0), sigma)
    blur.set_device("cuda")
    try:
        got = blur.gaussian_blur(img, sigma)
    finally:
        blur.set_device(None)
    assert got.shape == ref.shape and got.dtype == np.float32
    assert np.abs(got - ref).max() < 1e-3  # values 0-255: float32 rounding only


def test_cpu_blur_is_opencv_exactly():
    img = np.random.default_rng(1).random((300, 400)).astype(np.float32)
    blur.set_device(None)
    assert np.array_equal(blur.gaussian_blur(img, 24.0), cv2.GaussianBlur(img, (0, 0), 24.0))


def test_png_pool_writes_the_same_files_as_write_png(tmp_path):
    rng = np.random.default_rng(2)
    imgs = [(rng.random((280, 336, 3)) * 255).astype(np.uint8) for _ in range(12)]
    recs = [{"id": f"x{i}", "anomaly": {"lat": 12.9, "lon": 77.5}, "camera": {"lat": 12.9, "lon": 77.5}} for i in range(12)]
    shas = {}
    pool = PngPool(workers=4, max_pending=3)
    for i, (im, r) in enumerate(zip(imgs, recs)):
        pool.submit(tmp_path / "pool" / f"{i}.png", im, r, on_done=lambda sha, i=i: shas.__setitem__(i, sha))
        r["status"] = "changed after submit"  # must not reach the file
        im[:] = 0
    pool.close()
    rng = np.random.default_rng(2)
    for i in range(12):
        im = (rng.random((280, 336, 3)) * 255).astype(np.uint8)
        r = {"id": f"x{i}", "anomaly": {"lat": 12.9, "lon": 77.5}, "camera": {"lat": 12.9, "lon": 77.5}}
        ref = write_png(tmp_path / "ref" / f"{i}.png", im, r)
        assert shas[i] == ref
        assert hashlib.sha256((tmp_path / "pool" / f"{i}.png").read_bytes()).hexdigest() == ref


# ---- one crop for distresses close together in a frame ----------------------------------------

from types import SimpleNamespace

from geoai.extract import _merge_in_frame

VIEW = SimpleNamespace(width=3840, height=2160)


def _frame_rec(i, px_box, ground_box, ev):
    poly = box(*ground_box)
    rec = {"id": f"f{i}", "status": "candidate", "kinds": ["probe"],
           "anomaly": {"lat": 12.9 + i * 1e-6, "lon": 77.5, "area_m2": float(poly.area)}}
    c = poly.centroid
    return _Rec(rec, None, px_box, ev, 0, (c.x, c.y), poly)


def test_close_distresses_become_one_bigger_crop():
    # Two potholes 0.3 m apart on the ground, crops not overlapping much in the image.
    a = _frame_rec(0, (1000, 1500, 1200, 1650), (0.0, 2.0, 0.2, 2.2), 30.0)
    b = _frame_rec(1, (1450, 1500, 1650, 1650), (0.5, 2.0, 0.7, 2.2), 12.0)
    out = _merge_in_frame([a, b], VIEW, Config())
    assert len(out) == 1
    m = out[0]
    assert m.rec["id"] == "f0" and m.evidence == 42.0
    x0, y0, x1, y1 = m.plan.box
    assert x0 <= 1000 and x1 >= 1650 and y0 <= 1500 and y1 >= 1650  # covers both
    assert [mm["id"] for mm in m.rec["members"]] == ["f0", "f1"]
    assert m.rec["merged"] == ["f1"]
    assert abs(m.rec["anomaly"]["merged_area_m2"] - 0.08) < 1e-9


def test_distresses_far_apart_stay_separate():
    a = _frame_rec(0, (300, 1500, 500, 1650), (-1.5, 2.0, -1.3, 2.2), 30.0)
    b = _frame_rec(1, (3300, 1500, 3500, 1650), (1.3, 2.0, 1.5, 2.2), 12.0)
    out = _merge_in_frame([a, b], VIEW, Config())
    assert len(out) == 2


def test_cluster_merges_transitively_into_one_segment():
    # Five patches in a row, each 0.4 m from the next: one segment, all members kept.
    recs = [_frame_rec(i, (400 + 300 * i, 1600, 550 + 300 * i, 1700), (0.6 * i, 2.0, 0.6 * i + 0.2, 2.2), 10.0 + i)
            for i in range(5)]
    out = _merge_in_frame(recs, VIEW, Config())
    assert len(out) == 1
    assert sorted(mm["id"] for mm in out[0].rec["members"]) == [f"f{i}" for i in range(5)]
    assert out[0].evidence == sum(10.0 + i for i in range(5))


def test_merge_never_downscales_below_the_limit():
    # Close on the ground but the union would span most of a 4K frame: scale under 0.5 -> no merge.
    a = _frame_rec(0, (0, 0, 300, 300), (0.0, 2.0, 0.2, 2.2), 30.0)
    b = _frame_rec(1, (3500, 1800, 3840, 2160), (0.4, 2.0, 0.6, 2.2), 12.0)
    cfg = Config()
    out = _merge_in_frame([a, b], VIEW, cfg)
    assert len(out) == 2
    for r in out:
        assert r.plan.out_w / r.plan.box_w >= cfg.crop.merge_min_scale or len(r.rec["members"]) == 1
