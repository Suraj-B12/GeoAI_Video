"""Absolute crack-likeness: thin dark curvilinear structures (no training, no labels).

The DINOv2 detector finds what is unusual *for this road*. On a road that is cracked almost
everywhere, cracks are not unusual, so this detector supplies an absolute signal that does not
depend on how common cracks are. It is also used to keep cracked patches out of the DINOv2
"normal" bank.

Method: multi-scale Hessian line filter on illumination-normalised grey (shadows divide out),
scale-normalised, then a robust z-score against the session's own background, then connected
components that are long and thin in millimetres on the ground.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .blur import device_for, gaussian_blur


def normalised_gray(rgb: np.ndarray, illum_sigma: float = 12.0) -> np.ndarray:
    """Grey divided by its local mean: removes shading and most of a shadow's effect."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local = gaussian_blur(g, illum_sigma)
    return g / np.maximum(local, 0.02)


def ridge_strength(gray: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    """Dark-line strength: the largest Hessian eigenvalue minus the smaller one's magnitude.

    The derivatives are always OpenCV's (CPU). Where the pipeline's blur runs on the GPU for an image this
    size (blur.device_for), the eigenvalue arithmetic does too, bit for bit (`_ridge_gpu`): in numpy it
    was ~215 ms of float32 temporaries per 4 MP gate image, against ~60 ms for the derivatives.
    """
    dev = device_for(gray.shape, min(sigmas)) if gray.dtype == np.float32 and sigmas else None
    out = np.zeros_like(gray, dtype=np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        if dev is not None:
            out = _ridge_gpu(out, dxx, dyy, dxy, s, dev)
            continue
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
        l1 = 0.5 * (dxx + dyy + tmp)     # across a dark line: large and positive
        l2 = 0.5 * (dxx + dyy - tmp)     # along the line: near zero
        line = np.maximum(l1 - np.abs(l2), 0.0) * (s * s)
        np.maximum(out, line, out=out)
    return out


@torch.inference_mode()
def _ridge_gpu(out: np.ndarray, dxx: np.ndarray, dyy: np.ndarray, dxy: np.ndarray, s: float,
               dev: torch.device) -> np.ndarray:
    """One scale of ridge_strength on the GPU, in numpy's operation order (each a separate, correctly rounded
    float32 kernel: x**2 is x*x, 4*x is a float32 product, sqrt is IEEE), so the result is the numpy one."""
    xx, yy, xy, o = (torch.from_numpy(a).to(dev) for a in (dxx, dyy, dxy, out))
    d = xx - yy
    q = xy * xy
    tmp = torch.sqrt(d * d + q * 4)
    tr = xx + yy
    l1 = (tr + tmp) * 0.5
    l2 = (tr - tmp) * 0.5
    line = torch.clamp_min(l1 - torch.abs(l2), 0.0) * (s * s)
    return torch.maximum(o, line).cpu().numpy()


@dataclass
class CrackComponent:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive) in analysis-image pixels
    length_mm: float
    z_max: float
    pixels: int


class CrackDetector:
    def __init__(self, sigmas=(1.0, 2.0), z_threshold=6.0, min_length_mm=150.0, min_elongation=6.0):
        self.sigmas = tuple(sigmas)
        self.z_threshold = z_threshold
        self.min_length_mm = min_length_mm
        self.min_elongation = min_elongation
        self.med: float | None = None
        self.scale: float | None = None
        self._samples: list[np.ndarray] = []

    # Background statistics come from the bank frames (most pixels of any road are not cracks,
    # so pixel-level statistics stay those of the background even on badly cracked roads).
    def add_background(self, rgb: np.ndarray, rng: np.random.Generator, n: int = 20000) -> None:
        r = ridge_strength(normalised_gray(rgb), self.sigmas).ravel()
        self._samples.append(rng.choice(r, size=min(n, r.size), replace=False))

    def finalize(self) -> None:
        if not self._samples:
            raise RuntimeError("crack detector has no background samples")
        x = np.concatenate(self._samples)
        self.med = float(np.median(x))
        q = np.quantile(x, [0.5, 0.75, 0.9])
        # The distribution is one-sided (>= 0): use the upper quartile spread.
        self.scale = float(max(q[2] - q[0], 1e-6))
        self._samples = []

    def zmap(self, rgb: np.ndarray) -> np.ndarray:
        r = ridge_strength(normalised_gray(rgb), self.sigmas)
        return (r - self.med) / self.scale

    def detect(self, rgb: np.ndarray, gsd_along_mm: np.ndarray) -> tuple[np.ndarray, list[CrackComponent]]:
        """Returns the crack mask and long, thin components.

        gsd_along_mm: per-row ground resolution (mm/px) of the analysis image (length H).
        """
        z = self.zmap(rgb)
        mask = (z > self.z_threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps: list[CrackComponent] = []
        keep = np.zeros(n, bool)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            row = min(int(y + h / 2), len(gsd_along_mm) - 1)
            g = float(gsd_along_mm[row])
            length_mm = float(np.hypot(w, h)) * g
            elong = (np.hypot(w, h) ** 2) / max(area, 1)
            if length_mm >= self.min_length_mm and elong >= self.min_elongation:
                keep[i] = True
                # Peak z inside the component's own box (the whole-image comparison per component cost
                # ~0.3 s per 4 MP gate image on real roads, which have hundreds of components).
                box = (slice(y, y + h), slice(x, x + w))
                comps.append(CrackComponent((int(x), int(y), int(x + w), int(y + h)), length_mm,
                                            float(z[box][labels[box] == i].max()), int(area)))
        return keep[labels], comps
