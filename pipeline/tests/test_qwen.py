import random

import pytest

from geoai.qwen import (MAX_PIXELS, MIN_TESTED_TOKENS, frame_input_size, plan_crop,
                        smart_resize, visual_tokens)

# "You send -> model receives" table measured by the VLM team with their production processor
# (vlm-information-response.md, notes 2.2).
VLM_TEAM_TABLE = [
    ((3840, 2160), (1344, 756), 1296),
    ((1920, 1080), (1344, 756), 1296),
    ((1344, 896), (1232, 812), 1276),
    ((1479, 508), (1484, 504), 954),
    ((1024, 683), (1036, 672), 888),
    ((640, 640), (644, 644), 529),
    ((672, 448), (672, 448), 384),
    ((336, 224), (336, 224), 96),
]


@pytest.mark.parametrize("sent,received,tokens", VLM_TEAM_TABLE)
def test_smart_resize_matches_vlm_team_measurements(sent, received, tokens):
    w, h = sent
    rh, rw = smart_resize(h, w)
    assert (rw, rh) == received
    assert visual_tokens(rw, rh) == tokens


def test_frame_input_size_for_survey_frames():
    assert frame_input_size(3840, 2160) == (1344, 756)
    assert frame_input_size(1920, 1080) == (1344, 756)


def _assert_valid(plan, region, frame_w, frame_h):
    assert plan.out_w % 28 == 0 and plan.out_h % 28 == 0
    assert plan.out_w * plan.out_h <= MAX_PIXELS
    assert smart_resize(plan.out_h, plan.out_w) == (plan.out_h, plan.out_w)
    assert plan.out_w <= plan.box_w and plan.out_h <= plan.box_h
    x0, y0, x1, y1 = plan.box
    assert 0 <= x0 < x1 <= frame_w and 0 <= y0 < y1 <= frame_h
    # Isotropic scale (no distortion beyond integer rounding).
    sx, sy = plan.out_w / plan.box_w, plan.out_h / plan.box_h
    assert abs(sx - sy) / sy < 0.01
    if not plan.truncated:
        assert x0 <= region[0] and y0 <= region[1] and x1 >= region[2] and y1 >= region[3]


def test_small_region_gets_context_not_enlargement():
    region = (1900.0, 1500.0, 1960.0, 1540.0)  # a 60 x 40 px crack box
    p = plan_crop(region, 3840, 2160)
    _assert_valid(p, region, 3840, 2160)
    assert p.scale == 1.0
    assert p.tokens >= MIN_TESTED_TOKENS
    cx, cy = (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2
    assert abs(cx - 1930) <= 1 and abs(cy - 1520) <= 1


def test_large_region_is_downscaled_not_cut():
    region = (200.0, 900.0, 3600.0, 2100.0)
    p = plan_crop(region, 3840, 2160)
    _assert_valid(p, region, 3840, 2160)
    assert p.scale < 1.0
    assert not p.truncated


def test_region_at_edge_shifts_inward():
    region = (3790.0, 1000.0, 3839.0, 1100.0)
    p = plan_crop(region, 3840, 2160)
    _assert_valid(p, region, 3840, 2160)
    assert p.box[2] == 3840 and not p.truncated


def test_very_wide_region_respects_aspect_limit():
    region = (0.0, 1800.0, 3840.0, 1830.0)  # a transverse crack across the whole frame
    p = plan_crop(region, 3840, 2160)
    _assert_valid(p, region, 3840, 2160)
    assert max(p.out_w / p.out_h, p.out_h / p.out_w) <= 3.0 + 0.15


def test_region_larger_than_frame_is_truncated():
    region = (-500.0, -100.0, 4500.0, 2300.0)
    p = plan_crop(region, 3840, 2160)
    assert p.truncated
    assert p.box == (0, 0, 3840, 2160) or (p.box[2] - p.box[0]) <= 3840


def test_random_regions_always_valid():
    rng = random.Random(7)
    for frame_w, frame_h in [(3840, 2160), (1920, 1080), (1280, 720)]:
        for _ in range(400):
            w = rng.uniform(4, frame_w * 1.2)
            h = rng.uniform(4, frame_h * 1.2)
            x0 = rng.uniform(-0.1 * frame_w, frame_w - 2)
            y0 = rng.uniform(-0.1 * frame_h, frame_h - 2)
            region = (x0, y0, x0 + w, y0 + h)
            p = plan_crop(region, frame_w, frame_h, margin_frac=rng.uniform(0, 0.5),
                          margin_min_px=rng.uniform(0, 100))
            _assert_valid(p, region, frame_w, frame_h)
            assert p.tokens >= MIN_TESTED_TOKENS


def test_tall_region_near_frame_edge_keeps_token_minimum():
    # Found by the end-to-end test: height capped by a 540-row frame left 323 tokens.
    region = (400.0, 20.0, 440.0, 520.0)
    p = plan_crop(region, 960, 540)
    _assert_valid(p, region, 960, 540)
    assert p.tokens >= MIN_TESTED_TOKENS


def test_deterministic():
    region = (1000.3, 1200.7, 1333.1, 1500.9)
    assert plan_crop(region, 3840, 2160) == plan_crop(region, 3840, 2160)
