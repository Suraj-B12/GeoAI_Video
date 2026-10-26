from pathlib import Path

import numpy as np
import pytest
import torch

from geoai.config import AnomalyConfig
from geoai.dinov2 import IMAGENET_MEAN, IMAGENET_STD, PatchFeatures, load_dinov2

CFG = AnomalyConfig()
FIXTURE = Path(__file__).parent / "data" / "dinov2_reference_112x140.npz"
pytestmark = pytest.mark.skipif(not Path(CFG.weights).exists(), reason="DINOv2 weights not downloaded")


def test_matches_official_implementation_output():
    """The fixture holds patch tokens computed by facebookresearch/dinov2 itself (same weights)."""
    ref = np.load(FIXTURE)
    model, dev, dtype = load_dinov2(Path(CFG.weights), "cpu", CFG.weights_sha256)
    t = torch.from_numpy(ref["image"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with torch.inference_mode():
        out = model((t - mean) / std).numpy()[0]
    assert np.abs(out - ref["patch_tokens"]).max() < 1e-4


def test_tiled_features_cover_the_grid_and_are_normalised():
    pf = PatchFeatures(Path(CFG.weights), "cpu", tile_patches=16, overlap=2)
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (14 * 20 + 5, 14 * 37 + 9, 3), dtype=np.uint8)
    f = pf(img)
    assert f.shape == (20, 37, 384)
    norms = np.linalg.norm(f, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    # Deterministic.
    assert np.array_equal(f, pf(img))


def test_tile_layout_keeps_every_patch_once():
    pf = PatchFeatures.__new__(PatchFeatures)
    pf.tile, pf.overlap = 16, 2
    for n in (5, 16, 17, 40, 100):
        rows, _ = pf._tiles(n, 1)
        kept = []
        for s0, s1, k0, k1 in rows:
            assert s0 <= k0 < k1 <= s1
            kept.extend(range(k0, k1))
        assert kept == list(range(n))
