"""Large-sigma Gaussian blur, on the GPU when the pipeline runs there.

cv2.GaussianBlur with sigma 12-40 on a 2758 x 700 analysis window takes about 30 ms on a laptop
CPU and ran four times per frame: the largest CPU cost after PNG writing. This computes the same
filter: OpenCV's own kernel (cv2.getGaussianKernel, size round(8 sigma + 1) | 1 for float32),
the same BORDER_REFLECT_101 border, as a separable fp32 convolution. tests/test_blur.py checks it
against cv2.

The device is set once by the detector (`set_device`); on the CPU, or for small images, cv2 is used.
`device_for` tells other per-image filters (anomaly.normalise_illumination, cracks.ridge_strength)
whether this blur runs on the GPU for an image, so they can keep their pixels there (`blur_tensor`).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_device: torch.device | None = None
_kernels: dict[tuple[float, str], torch.Tensor] = {}


def set_device(device: torch.device | str | None) -> None:
    global _device
    d = torch.device(device) if device is not None else None
    _device = d if d is not None and d.type == "cuda" else None
    if _device is not None:
        # TF32 would round the fp32 convolution to a 10-bit mantissa.
        torch.backends.cudnn.allow_tf32 = False


def _ksize(sigma: float) -> int:
    return int(round(sigma * 8 + 1)) | 1


def device_for(shape: tuple[int, ...], sigma: float) -> torch.device | None:
    """The GPU a float32 image of this (height, width) is blurred on with this sigma, or None (cv2)."""
    h, w = int(shape[0]), int(shape[1])
    if _device is None or _ksize(sigma) // 2 >= min(h, w) or h * w < 100_000:
        return None
    return _device


def blur_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """The GPU blur of a 2-D float32 tensor already on the device (use only where device_for says so)."""
    k = _ksize(sigma)
    key = (float(sigma), str(x.device))
    ker = _kernels.get(key)
    if ker is None:
        ker = torch.from_numpy(cv2.getGaussianKernel(k, sigma).astype(np.float32).ravel()).to(x.device)
        _kernels[key] = ker
    p = k // 2
    with torch.inference_mode():
        y = F.conv2d(F.pad(x[None, None], (p, p, 0, 0), mode="reflect"), ker.view(1, 1, 1, k))
        y = F.conv2d(F.pad(y, (0, 0, p, p), mode="reflect"), ker.view(1, 1, k, 1))
        return y[0, 0]


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """cv2.GaussianBlur(img, (0, 0), sigma) for a 2-D float32 image."""
    dev = device_for(img.shape, sigma) if img.dtype == np.float32 else None
    if dev is None:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    with torch.inference_mode():
        return blur_tensor(torch.from_numpy(np.ascontiguousarray(img)).to(dev), sigma).cpu().numpy()
