"""The GPU paths of normalise_illumination, ridge_strength and paint_fraction give the numpy results bit for
bit, crack components keep their whole-image peak z, and the CUDA-graph k-center picks the loop's indices.

The references below are verbatim copies of the numpy code as it was before the GPU paths existed. The
images are random noise (every value and rounding case), a smooth gradient (large uniform regions, an
even pixel count, so the median is the mean of two values) and a simulated road with a crack and a
pothole, at the gate's analysis size.
"""
import cv2
import numpy as np
import pytest
import torch

from geoai import blur
from geoai.anomaly import _kcenter, normalise_illumination, paint_fraction
from geoai.cracks import CrackDetector, normalised_gray, ridge_strength
from geoai.simtexture import asphalt, draw_crack, draw_pothole, to_rgb

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def ref_normalise_illumination(rgb, sigma):
    f = rgb.astype(np.float32)
    lum = f.mean(-1)
    local = blur.gaussian_blur(lum, sigma)
    target = float(np.median(lum))
    return np.clip(f * (target / np.maximum(local, 1.0))[..., None], 0, 255).astype(np.uint8)


def ref_ridge_strength(gray, sigmas):
    out = np.zeros_like(gray, dtype=np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
        l1 = 0.5 * (dxx + dyy + tmp)
        l2 = 0.5 * (dxx + dyy - tmp)
        line = np.maximum(l1 - np.abs(l2), 0.0) * (s * s)
        np.maximum(out, line, out=out)
    return out


def _images():
    rng = np.random.default_rng(7)
    yield "noise", rng.integers(0, 256, (713, 1403, 3), dtype=np.uint8)          # odd pixel count
    yy, xx = np.mgrid[0:700, 0:1400]
    yield "gradient", np.dstack([(xx * 255 // 1399).astype(np.uint8), (yy * 255 // 699).astype(np.uint8),
                                 np.full_like(xx, 90, dtype=np.uint8)])
    road = asphalt(3.0, 6.0, 0.003, seed=3, lateral0_m=-1.5, along0_m=0.0)
    draw_crack(road, (0.0, 2.0), 30.0, 1.5, 4.0, rng)
    draw_pothole(road, (0.5, 4.0), 0.2, rng)
    yield "road", to_rgb(road.img)


@pytest.fixture()
def gpu():
    blur.set_device("cuda")
    yield
    blur.set_device(None)


@pytest.mark.parametrize("sigma", [12.0, 24.0])
def test_normalise_illumination_gpu_is_numpy_exactly(gpu, sigma):
    for name, rgb in _images():
        assert blur.device_for(rgb.shape[:2], sigma) is not None, name   # the GPU path really runs
        got = normalise_illumination(rgb, sigma)
        want = ref_normalise_illumination(rgb, sigma)
        assert got.dtype == np.uint8 and np.array_equal(got, want), (name, int((got != want).sum()))


def test_ridge_strength_gpu_is_numpy_exactly(gpu):
    for name, rgb in _images():
        g = normalised_gray(rgb)
        got = ridge_strength(g, (1.0, 2.0))
        want = ref_ridge_strength(g, (1.0, 2.0))
        assert got.dtype == np.float32 and np.array_equal(got, want), (name, int((got != want).sum()))


def ref_paint_fraction(rgb, grid_h, grid_w, ratio, sigma=40.0):
    lum = rgb.astype(np.float32).mean(-1)
    ref0 = blur.gaussian_blur(lum, sigma)
    clipped = np.minimum(lum, ref0 * 1.15)
    ref = blur.gaussian_blur(clipped, sigma)
    paint = (lum > ratio * ref).astype(np.uint8)
    paint = cv2.dilate(paint, np.ones((14, 14), np.uint8)).astype(np.float32)
    h, w = grid_h * 14, grid_w * 14
    return paint[:h, :w].reshape(grid_h, 14, grid_w, 14).mean(axis=(1, 3))


def test_paint_fraction_gpu_is_numpy_exactly(gpu):
    rng = np.random.default_rng(3)
    for name, rgb in _images():
        rgb = rgb.copy()
        rgb[100:110, 200:900] = 250                      # a marking, so the mask is not empty
        rgb[rng.random(rgb.shape[:2]) < 0.002] = 255     # and specks at every brightness ratio
        gh, gw = rgb.shape[0] // 14, rgb.shape[1] // 14
        for ratio in (1.45, 1.3):
            got = paint_fraction(rgb, gh, gw, ratio)
            want = ref_paint_fraction(rgb, gh, gw, ratio)
            assert got.any() and np.array_equal(got, want), (name, ratio)


def test_crack_component_peaks_are_the_whole_image_ones(gpu):
    """detect() takes each component's peak z inside its box; the reference takes it over the whole image."""
    rng = np.random.default_rng(1)
    det = CrackDetector()
    images = [rgb for _, rgb in _images()]
    for rgb in images:
        det.add_background(rgb, rng)
    det.finalize()
    det.z_threshold, det.min_length_mm, det.min_elongation = 2.0, 10.0, 1.0   # many components, of every shape
    n = 0
    for rgb in images:
        gsd = np.full(rgb.shape[0], 2.0)
        mask, comps = det.detect(rgb, gsd)
        z = det.zmap(rgb)
        m = cv2.morphologyEx((z > det.z_threshold).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        by_box = {tuple(int(v) for v in (s[0], s[1], s[0] + s[2], s[1] + s[3])): i for i, s in enumerate(stats) if i}
        for c in comps:
            assert c.z_max == float(z[labels == by_box[c.bbox]].max())
            n += 1
    assert n > 20


def test_kcenter_graph_is_the_step_loop_at_bank_size():
    """A real bank row's shape (16,380 patches x 768, coreset 4,096): the CUDA-graph replay picks exactly the
    indices of the step-by-step loop (written out here as in the first version)."""
    rng = np.random.default_rng(0)
    c = rng.standard_normal((32, 768))
    x = c[rng.integers(0, 32, 16380)] + 0.8 * rng.standard_normal((16380, 768))
    X = torch.from_numpy((x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)).cuda()
    m, seed = 4096, 5
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(len(X), (1,), generator=g))
    idx = [first]
    mind = 1.0 - X @ X[first]
    for _ in range(1, m):
        j = int(torch.argmax(mind))
        idx.append(j)
        mind = torch.minimum(mind, 1.0 - X @ X[j])
    assert _kcenter(X, m, seed).tolist() == idx


def test_cpu_paths_unchanged():
    blur.set_device(None)
    for name, rgb in _images():
        assert np.array_equal(normalise_illumination(rgb, 24.0), ref_normalise_illumination(rgb, 24.0)), name
        g = normalised_gray(rgb)
        assert np.array_equal(ridge_strength(g, (1.0, 2.0)), ref_ridge_strength(g, (1.0, 2.0))), name
        gh, gw = rgb.shape[0] // 14, rgb.shape[1] // 14
        assert np.array_equal(paint_fraction(rgb, gh, gw, 1.45), ref_paint_fraction(rgb, gh, gw, 1.45)), name
