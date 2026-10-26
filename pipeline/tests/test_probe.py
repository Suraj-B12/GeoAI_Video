"""The Attain-trained distress probe: metadata, compatibility, and behaviour on held-out real frames.

The 27 frames in vlm_handoff/sample_images were forced into the probe's test split, so they
were never used for training.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoai.anomaly import AnomalyDetector, normalise_illumination
from geoai.config import AnomalyConfig, CrackConfig
from geoai.dinov2 import PatchFeatures
from geoai.probe import DistressProbe

CFG = AnomalyConfig()
PROBE = Path(CFG.probe) if CFG.probe else None
SAMPLES = Path(__file__).resolve().parents[2] / "vlm_handoff" / "sample_images"
pytestmark = pytest.mark.skipif(PROBE is None or not PROBE.exists() or not Path(CFG.weights).exists(),
                                reason="probe or DINOv2 weights missing")


def test_probe_matches_pipeline_features():
    p = DistressProbe.load(PROBE)
    assert tuple(p.layers) == tuple(CFG.layers)
    assert p.illum_sigma_px == CFG.illum_sigma_px
    assert p.w.shape == (384 * len(CFG.layers),)
    assert p.meta["features"]["weights_sha256"] == CFG.weights_sha256


def test_held_out_metrics_recorded_and_sane():
    m = DistressProbe.load(PROBE).meta
    assert m["test_auroc"] >= 0.85
    assert m["test_fpr_at_seed"] <= 0.03          # seed chosen at 1 % on validation patches


def test_mismatched_feature_config_is_refused():
    feats = PatchFeatures(Path(CFG.weights), "cpu", layers=(6, 9))
    bad = AnomalyConfig(layers=(6, 9))
    with pytest.raises(ValueError, match="trained on layers"):
        AnomalyDetector(feats, 224, 224, bad, CrackConfig())


def _road(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    return a[int(a.shape[0] * 0.45):]


def test_flags_held_out_distress_frames_more_than_clean_ones():
    samples = sorted(SAMPLES.glob("0[1-8]_*/*.png"))
    required = {"01_linear_crack", "02_alligator_crack", "07_block_crack", "08_no_distress_annotated"}
    if not required.issubset({f.parent.name for f in samples}):
        pytest.skip("held-out sample images are not included in this checkout")
    p = DistressProbe.load(PROBE)
    pf = PatchFeatures(Path(CFG.weights), "cpu", layers=CFG.layers)
    frac = {}
    for f in samples:
        logit = p.logits(pf(normalise_illumination(_road(f), CFG.illum_sigma_px)))
        frac.setdefault(f.parent.name, []).append(float((logit >= p.seed_logit).mean()))
    cracks = [v for k, vs in frac.items() for v in vs if k in ("01_linear_crack", "02_alligator_crack", "07_block_crack")]
    clean = frac["08_no_distress_annotated"]
    # Most crack frames have some flagged road; clean frames have little.
    assert np.mean([v > 0.005 for v in cracks]) >= 0.6, frac
    assert np.median(cracks) > np.median(clean), frac
