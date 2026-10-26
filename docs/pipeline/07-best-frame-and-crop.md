# 07 · Crops for Qwen2.5-VL

Goal: each anomaly cut out slightly larger than itself, in a form the VLM receives without any change.

Implementation: [`geoai/qwen.py`](../../pipeline/geoai/qwen.py), `extract._write_crop`. Facts below come from the VLM team's answers ([`vlm_handoff`](../../vlm_handoff/vlm-information-response.md), notes 2.2, 2.7, 2.8, §8).

## 1. What the VLM's processor does

Qwen2.5-VL keeps the aspect ratio, rounds each side to a multiple of 28, and downscales (bicubic) only above 1,048,576 pixels. `qwen.smart_resize` reproduces their measured table exactly (8 sizes, from 3840×2160 → 1344×756 to 336×224 → 336×224).

**Every crop is a fixed point of that rule.** Both sides are multiples of 28, and the area is at most 1,048,576 px and at least 324 visual tokens (a 504×504 square). So the model receives our pixels unchanged.

## 2. Crop rule

```
margin m = max(0.25 x long side, 64 px) on every side
aspect   <= 3:1 (the widest input they evaluated was 1484x504), by adding context to the short side
size     >= 324 tokens: small anomalies get more real context, never enlargement
fit      shift inward at frame edges; clip only if larger than the frame (flagged truncated)
large    > 1,048,576 px: downscale isotropically (area interpolation), sides multiples of 28
```

- The road runs up the image (frames are stored upright), so the VLM can tell longitudinal from transverse cracking.
- Edges and shoulders are visible whenever they are in the band.
- `tests/test_qwen.py` checks these invariants on 1,200 random regions, plus a case found end to end: a tall region near the frame edge whose height was capped by the frame.

## 3. Files

| File | Content | Sent? |
|---|---|---|
| `crops/<id>.png` | The crop. **PNG**: frames are already video-compressed, and the VLM team found that re-encoding as JPEG q95 changed labels on 9 of 49 photos. EXIF + a `geoai` text chunk ([08](08-geotag-and-records.md)) | Yes |
| `frames/<frame_id>.png` | The whole frame at the size the VLM would resize it to (1344×756 for 4K and 1080p) | Yes, as a separate request (their week-1 request) |
| `contact_sheet.jpg` | The 48 highest-evidence crops | No |

**Nothing is drawn on crops or frames.** There are no boxes, text or GPS stamps.

## 4. Tests

- `tests/test_qwen.py`: the VLM team's table, and invariants on random regions.
- `tests/test_e2e_sim.py`: every written crop and frame passes `smart_resize` unchanged and has at least 324 tokens.
- `tests/test_records.py`: PNG pixels are lossless and bytes are identical on rerun.
