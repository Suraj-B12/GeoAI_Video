"""Absolute "this looks like distress" signal: a linear probe on DINOv2 patch features.

Trained on the public Attain dataset (CC BY 4.0) by tools/train_probe.py, with the same features
the anomaly detector uses (same blocks, same illumination normalisation). Unlike the session
bank, it doesn't depend on how common a distress is on the road being surveyed, so a road that
is cracked almost everywhere is still flagged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DistressProbe:
    w: np.ndarray            # (D,)
    b: float
    layers: tuple[int, ...]
    illum_sigma_px: float | None
    seed_logit: float        # patches at or above this start a probe anomaly (chosen on held-out data)
    meta: dict

    @classmethod
    def load(cls, path: Path) -> "DistressProbe":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        sigma = meta.get("illum_sigma_px")
        return cls(z["w"].astype(np.float32), float(z["b"]), tuple(int(x) for x in z["layers"]),
                   None if sigma is None else float(sigma), float(z["seed_logit"]), meta)

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """feats: (gh, gw, D) L2-normalised features from PatchFeatures -> (gh, gw) logits."""
        if feats.shape[-1] != self.w.shape[0]:
            raise ValueError(f"probe expects {self.w.shape[0]}-d features, got {feats.shape[-1]}")
        return feats @ self.w + self.b

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w, b=np.float32(self.b), layers=np.array(self.layers),
                 seed_logit=np.float32(self.seed_logit), meta=np.array(json.dumps(self.meta)))
