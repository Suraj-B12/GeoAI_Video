"""Mechanics of the label-free detector on synthetic asphalt (top-down windows of one road).

These check that the method behaves as designed: distress is flagged, recurring markings and
plain texture are not, and a distress can't hide by matching itself in the bank. They do not
measure real-world accuracy (that needs real footage).
"""
from pathlib import Path

import numpy as np
import pytest

from geoai.anomaly import AnomalyDetector
from geoai.config import AnomalyConfig, CrackConfig
from geoai.dinov2 import PatchFeatures
from geoai.simtexture import (asphalt, draw_crack, draw_marking, draw_patch, draw_pothole, to_rgb)

# The Attain-trained probe is off here: these tests check the bank mechanics on procedural texture,
# which the probe (trained on real roads) reads as distressed: 16 % of synthetic plain-asphalt patches
# score above its threshold, against 0-1.8 % of real clean road. The probe is tested on real
# held-out frames in test_probe.py.
CFG = AnomalyConfig(exclude_window_m=3.0, probe=None)
pytestmark = pytest.mark.skipif(not Path(CFG.weights).exists(), reason="DINOv2 weights not downloaded")

RES = 0.0025          # m per pixel
WIN_H, WIN_W = 224, 392


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(5)
    road = asphalt(1.0, 42.0, RES, seed=3, lateral0_m=-0.5)
    draw_marking(road, 0.35, 0.10, dash_m=1.0, gap_m=2.0)          # recurring look-alike
    pot = draw_pothole(road, (-0.1, 20.6), 0.12, rng)
    crk = draw_crack(road, (-0.4, 30.4), 90.0, 0.6, 5.0, rng)      # transverse crack
    pat = draw_patch(road, (-0.05, 10.6), (0.35, 0.30), 10.0, tone=0.65)
    return road, {"pothole": pot, "crack": crk, "patch": pat}


def window(road, along_m):
    r0 = int(round((along_m - road.along0_m) / RES))
    return to_rgb(road.img[r0:r0 + WIN_H, 0:WIN_W])


@pytest.fixture(scope="module")
def detector(scene):
    road, _ = scene
    feats = PatchFeatures(Path(CFG.weights), "cpu", tile_patches=32, overlap=4, layers=CFG.layers)
    det = AnomalyDetector(feats, WIN_H, WIN_W, CFG, CrackConfig(min_length_mm=150.0))
    gsd = np.full(WIN_H, RES * 1000)
    # Jittered spacing, as the pipeline selects bank frames (periodic markings must not phase-lock).
    rng = np.random.default_rng(0)
    along = 0.3
    while along < 41.0:
        det.add_bank_frame(window(road, along), float(along), gsd)
        along += 0.6 * (1 + rng.uniform(-CFG.bank_jitter, CFG.bank_jitter))
    det.finalize()
    return det, gsd


def _hits(cands, road, along_m, feature):
    """Candidates whose box overlaps the feature's footprint (in window pixels)."""
    poly = np.array(feature.polygon_m)
    cols = (poly[:, 0] - road.lateral0_m) / RES
    rows = (poly[:, 1] - along_m) / RES
    fx0, fx1, fy0, fy1 = cols.min(), cols.max(), rows.min(), rows.max()
    return [c for c in cands if c.bbox[0] <= fx1 and c.bbox[2] >= fx0 and c.bbox[1] <= fy1 and c.bbox[3] >= fy0]


@pytest.mark.parametrize("name,along", [("pothole", 20.45), ("crack", 30.2), ("patch", 10.4)])
def test_distress_is_flagged(scene, detector, name, along):
    road, feats = scene
    det, gsd = detector
    cands, diag = det.detect(window(road, along), along, gsd)
    assert _hits(cands, road, along, feats[name]), (name, diag, [(c.bbox, c.score, c.kinds) for c in cands])


def test_plain_asphalt_and_recurring_marking_rarely_flagged(scene, detector):
    """Clean windows along the route (many contain marking dashes): at most 1 false candidate in ~45."""
    road, feats = scene
    det, gsd = detector
    false, windows = 0, 0
    for along in np.arange(1.0, 41.0, 0.77):
        if any(abs(along + 0.28 - p[1]) < 0.9 for ft in feats.values() for p in ft.polygon_m):
            continue  # window overlaps a distress
        cands, _ = det.detect(window(road, along), along, gsd)
        false += len(cands)
        windows += 1
    assert windows >= 40
    assert false <= 1, false


def test_distress_evidence_far_above_threshold(scene, detector):
    road, feats = scene
    det, gsd = detector
    for name, along in (("pothole", 20.45), ("crack", 30.2), ("patch", 10.4)):
        cands, _ = det.detect(window(road, along), along, gsd)
        assert max(c.score for c in _hits(cands, road, along, feats[name])) > 3 * CFG.min_mass


def test_distress_cannot_match_itself_in_the_bank(scene, detector):
    """The pothole window is also (almost) a bank frame; the exclusion window must ignore it."""
    road, feats = scene
    det, gsd = detector
    cands, _ = det.detect(window(road, 20.5), 20.5, gsd)
    assert _hits(cands, road, 20.5, feats["pothole"])
