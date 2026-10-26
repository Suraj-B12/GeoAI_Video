"""Input-size rules of the Qwen2.5-VL classifier on the VLM machine.

Source: vlm_handoff/vlm-information-response.md, notes 2.2 and §8. The server's processor keeps
the aspect ratio, rounds each side to a multiple of 28 and downscales (bicubic) only above
1,048,576 pixels. A crop whose sides are already multiples of 28 and whose area is within the
limits reaches the model pixel for pixel, so every crop planned here is a fixed point of
`smart_resize`.

The VLM team's evidence covers inputs of about 324 to 1,300 visual tokens (one token per 28 x 28
block). Small distresses therefore get more real context around them instead of being enlarged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FACTOR = 28
MIN_PIXELS = 7168              # server setting (vlm-information-response.yaml, preferred_size_px)
MAX_PIXELS = 1_048_576
TOKEN_PIXELS = FACTOR * FACTOR
MIN_TESTED_TOKENS = 324        # lowest input size in the VLM team's evaluations
MAX_ASPECT = 3.0               # widest evaluated input was 1484 x 504 (2.94:1)


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """The Qwen2-VL / Qwen2.5-VL processor rule (transformers `image_processing_qwen2_vl.smart_resize`).

    Returns (height, width) as the model receives them. Python's round() is banker's rounding,
    exactly as in the original.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid size {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def visual_tokens(width: int, height: int) -> int:
    """Visual tokens for an input the model receives unchanged (both sides multiples of 28)."""
    return (width // FACTOR) * (height // FACTOR)


def frame_input_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    """(width, height) to which a full frame is resized before sending: what the server would do anyway."""
    h, w = smart_resize(frame_h, frame_w)
    return w, h


@dataclass(frozen=True)
class CropPlan:
    """A crop box in frame pixels and the exact output size sent to the VLM."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (x1, y1 exclusive), inside the frame
    out_w: int
    out_h: int
    truncated: bool                 # the region plus its margin did not fit inside the frame

    @property
    def box_w(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def box_h(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def scale(self) -> float:
        """Output pixels per frame pixel (1.0 = pixels unchanged; never above 1)."""
        return self.out_w / self.box_w

    @property
    def tokens(self) -> int:
        return visual_tokens(self.out_w, self.out_h)


def _floor28(v: float) -> int:
    return max(FACTOR, int(math.floor(v / FACTOR + 1e-9)) * FACTOR)


def _ceil28(v: float) -> int:
    return max(FACTOR, int(math.ceil(v / FACTOR - 1e-9)) * FACTOR)


def plan_crop(region: tuple[float, float, float, float], frame_w: int, frame_h: int, *,
              margin_frac: float = 0.25, margin_min_px: float = 64.0,
              min_tokens: int = MIN_TESTED_TOKENS, max_pixels: int = MAX_PIXELS,
              max_aspect: float = MAX_ASPECT) -> CropPlan:
    """Plan a crop around `region` (x0, y0, x1, y1 in frame pixels).

    Rules, in order:
    1. Add a margin of max(margin_frac x long side, margin_min_px) on every side.
    2. Keep the aspect ratio within max_aspect, by adding context to the short side.
    3. Reach at least min_tokens of real pixels, by adding context, never by enlarging.
    4. Fit inside the frame: shift inward first, clip only if larger than the frame (truncated).
    5. Output sides are multiples of 28 and the area is at most max_pixels. Larger crops are
       downscaled (the caller uses area interpolation); smaller ones keep their pixels exactly.
    """
    if frame_w < FACTOR or frame_h < FACTOR:
        raise ValueError("frame too small")
    x0, y0, x1, y1 = (float(v) for v in region)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty region {region}")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    m = max(margin_frac * max(w, h), margin_min_px)
    bw, bh = w + 2 * m, h + 2 * m
    if bw > max_aspect * bh:
        bh = bw / max_aspect
    elif bh > max_aspect * bw:
        bw = bh / max_aspect
    area_min = min_tokens * TOKEN_PIXELS
    if bw * bh < area_min:
        k = math.sqrt(area_min / (bw * bh))
        bw, bh = bw * k, bh * k
    bw, bh = min(bw, float(frame_w)), min(bh, float(frame_h))

    # Output size and the exact box that maps onto it with one isotropic scale.
    s = min(1.0, math.sqrt(max_pixels / (bw * bh)))
    out_w, out_h = _ceil28(bw * s), _ceil28(bh * s)
    for _ in range(8):
        if out_w * out_h > max_pixels:
            # Rounding up overshot the pixel budget: round the larger side down instead.
            if out_w >= out_h:
                out_w -= FACTOR
            else:
                out_h -= FACTOR
            continue
        box_w, box_h = round(out_w / s), round(out_h / s)
        if box_w > frame_w:
            out_w = _floor28(frame_w * s)
            continue
        if box_h > frame_h:
            out_h = _floor28(frame_h * s)
            continue
        break
    else:  # pragma: no cover - the loop always converges for frames >= 28 px
        raise RuntimeError("crop planning did not converge")
    if s == 1.0:
        # Clamping one side to the frame can leave the crop under the token minimum: grow the
        # other side (more real context), within the frame and the aspect limit.
        while out_w * out_h < area_min:
            if out_w + FACTOR <= frame_w and (out_w + FACTOR) <= max_aspect * out_h:
                out_w += FACTOR
            elif out_h + FACTOR <= frame_h and (out_h + FACTOR) <= max_aspect * out_w:
                out_h += FACTOR
            else:
                break
        box_w, box_h = out_w, out_h
    box_w, box_h = min(box_w, frame_w), min(box_h, frame_h)

    bx0 = int(round(cx - box_w / 2))
    by0 = int(round(cy - box_h / 2))
    bx0 = min(max(bx0, 0), frame_w - box_w)
    by0 = min(max(by0, 0), frame_h - box_h)
    box = (bx0, by0, bx0 + box_w, by0 + box_h)
    truncated = x0 < box[0] - 0.5 or y0 < box[1] - 0.5 or x1 > box[2] + 0.5 or y1 > box[3] + 0.5
    plan = CropPlan(box=box, out_w=out_w, out_h=out_h, truncated=truncated)
    _check(plan, frame_w, frame_h, max_pixels)
    return plan


def _check(p: CropPlan, frame_w: int, frame_h: int, max_pixels: int) -> None:
    assert p.out_w % FACTOR == 0 and p.out_h % FACTOR == 0, p
    assert p.out_w * p.out_h <= max_pixels, p
    assert p.out_w <= p.box_w and p.out_h <= p.box_h, f"would enlarge: {p}"
    assert 0 <= p.box[0] < p.box[2] <= frame_w and 0 <= p.box[1] < p.box[3] <= frame_h, p
    assert smart_resize(p.out_h, p.out_w) == (p.out_h, p.out_w), f"not a fixed point: {p}"
