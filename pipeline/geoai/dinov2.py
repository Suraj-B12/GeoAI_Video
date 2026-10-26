"""DINOv2 ViT-S/14 with registers, inference only, loading Meta's official weights.

A self-contained re-implementation (about 100 lines) so the offline pipeline needs neither
torch.hub nor the DINOv2 repository at run time. It matches facebookresearch/dinov2
(`dinov2_vits14_reg`: 4 register tokens, bicubic antialiased position-embedding interpolation,
no interpolation offset); tests/test_dinov2.py checks it against outputs of the official code.

Weights: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
(Apache-2.0).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .util import sha256_file

PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * 4)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)


class DinoV2Small(nn.Module):
    dim = 384
    depth = 12
    heads = 6
    registers = 4

    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed(self.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, self.dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.registers, self.dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.dim))
        self.blocks = nn.ModuleList([_Block(self.dim, self.heads) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)

    def _pos(self, n_patches: int, h0: int, w0: int, dtype) -> torch.Tensor:
        N = self.pos_embed.shape[1] - 1
        if n_patches == N and h0 == w0:
            return self.pos_embed.to(dtype)
        pos = self.pos_embed.float()
        M = int(math.sqrt(N))
        patch = F.interpolate(pos[:, 1:].reshape(1, M, M, self.dim).permute(0, 3, 1, 2),
                              size=(h0, w0), mode="bicubic", antialias=True)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, self.dim)
        return torch.cat((pos[:, :1], patch), dim=1).to(dtype)

    def forward(self, img: torch.Tensor, layers: tuple[int, ...] = (12,)) -> torch.Tensor:
        """img: B x 3 x H x W, ImageNet-normalised, H and W multiples of 14.

        Returns patch tokens of the requested blocks (1-based; 12 = last), each passed through the
        final norm as in the official `get_intermediate_layers(norm=True)`, concatenated:
        B x (H/14) x (W/14) x (384 * len(layers)).
        """
        B, _, H, W = img.shape
        if H % PATCH or W % PATCH:
            raise ValueError(f"image size {W}x{H} not a multiple of {PATCH}")
        h0, w0 = H // PATCH, W // PATCH
        x = self.patch_embed.proj(img).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1).to(x.dtype), x), dim=1)
        x = x + self._pos(x.shape[1] - 1, h0, w0, x.dtype)
        x = torch.cat((x[:, :1], self.register_tokens.expand(B, -1, -1).to(x.dtype), x[:, 1:]), dim=1)
        outs = []
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i in layers:
                outs.append(self.norm(x)[:, 1 + self.registers:].reshape(B, h0, w0, self.dim))
            if i >= max(layers):
                break
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


def load_dinov2(weights: Path, device: str = "auto", expected_sha256: str | None = None) -> tuple[DinoV2Small, torch.device, torch.dtype]:
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"DINOv2 weights not found: {weights}")
    if expected_sha256 and sha256_file(weights) != expected_sha256:
        raise ValueError(f"DINOv2 weights checksum mismatch: {weights}")
    model = DinoV2Small()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = model.to(dev, dtype).eval()
    return model, dev, dtype


class PatchFeatures:
    """Dense, L2-normalised DINOv2 patch features of an image, computed tile by tile.

    Tiles overlap by `overlap` patches on each side and only their centres are kept, so every
    kept patch has context around it. The tiling is a deterministic function of the image size,
    so the same image position always gets the same tile context (needed for position-wise
    comparison across frames).
    """

    def __init__(self, weights: Path, device: str = "auto", tile_patches: int = 32, overlap: int = 4,
                 batch: int = 8, expected_sha256: str | None = None, layers: tuple[int, ...] = (12,)):
        self.model, self.device, self.dtype = load_dinov2(weights, device, expected_sha256)
        self.tile = tile_patches
        self.overlap = overlap
        self.batch = batch
        self.layers = tuple(sorted(layers))
        self.dim = DinoV2Small.dim * len(self.layers)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def grid_shape(h: int, w: int) -> tuple[int, int]:
        return h // PATCH, w // PATCH

    def _tiles(self, gh: int, gw: int):
        """Tile origins (in patches) along one axis, with kept ranges."""
        def axis(n: int):
            t = min(self.tile, n)
            if t == n:
                return [(0, n, 0, n)]
            step = t - 2 * self.overlap
            starts = list(range(0, n - t + 1, step))
            if starts[-1] != n - t:
                starts.append(n - t)
            out = []
            for i, s0 in enumerate(starts):
                keep0 = 0 if i == 0 else s0 + self.overlap
                keep1 = n if i == len(starts) - 1 else s0 + t - self.overlap
                if i > 0:
                    keep0 = max(keep0, out[-1][3])
                out.append((s0, s0 + t, keep0, keep1))
            return out
        return axis(gh), axis(gw)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: H x W x 3 uint8. Returns float32 features of shape (H//14, W//14, 384), L2-normalised.

        Pixels beyond the last full patch (right and bottom remainder) are ignored.
        """
        gh, gw = self.grid_shape(*rgb.shape[:2])
        if gh < 1 or gw < 1:
            raise ValueError("image smaller than one patch")
        img = torch.from_numpy(np.ascontiguousarray(rgb[: gh * PATCH, : gw * PATCH])).to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        img = ((img - self.mean) / self.std).to(self.dtype)
        rows, cols = self._tiles(gh, gw)
        out = np.zeros((gh, gw, self.dim), np.float32)
        jobs = [(r, c) for r in rows for c in cols]
        nl = len(self.layers)
        for i in range(0, len(jobs), self.batch):
            chunk = jobs[i:i + self.batch]
            # Tiles in a batch must share a size; the last row/column may differ, so group.
            groups: dict[tuple[int, int], list] = {}
            for r, c in chunk:
                groups.setdefault((r[1] - r[0], c[1] - c[0]), []).append((r, c))
            for (th, tw), items in groups.items():
                batch = torch.cat([img[:, :, r[0] * PATCH:r[1] * PATCH, c[0] * PATCH:c[1] * PATCH] for r, c in items])
                feats = self.model(batch, self.layers).float()
                # Unit length per layer, then overall: cosine distance = mean of per-layer distances.
                b, h, w, _ = feats.shape
                feats = F.normalize(feats.reshape(b, h, w, nl, DinoV2Small.dim), dim=-1).reshape(b, h, w, self.dim)
                feats = (feats / math.sqrt(nl)).cpu().numpy()
                for k, (r, c) in enumerate(items):
                    out[r[2]:r[3], c[2]:c[3]] = feats[k, r[2] - r[0]:r[3] - r[0], c[2] - c[0]:c[3] - c[0]]
        return out
