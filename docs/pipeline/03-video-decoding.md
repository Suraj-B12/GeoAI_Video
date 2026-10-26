# 03 · Video Decoding and Frame Quality

Goal: decode every sample of every segment exactly once, attach it to the right `frames.csv` row, convert it to correct RGB, and decide whether it is worth detecting on.

## 1. The join (the most important rule in this stage)

**Decoded sample `i` of `video/seg_NNNN.mp4` ⇔ the `frames.csv` row with `segment = NNNN` and `frame_in_segment = i`.**

- **Never use container PTS as a time.** The fragmented-MP4 muxer estimates the last sample duration of each 2 s fragment, so file PTS drift from the true capture times (measured on the emulator: 5–14 ms after 20 s, growing per fragment). Times come only from `frames.csv` ([02](02-time-and-trajectory.md)).
- **Use container PTS as a key.** They are unique and strictly increasing (the app writes no B-frames; ingest verified it). So:

```python
container = av.open(path)
stream = container.streams.video[0]
stream.thread_type = "AUTO"
index_of_pts = {}
for i, packet in enumerate(p for p in container.demux(stream) if p.pts is not None):
    index_of_pts[packet.pts] = i              # sample index in file order
    for frame in packet.decode():             # decoders may emit frames a few packets later
        idx = index_of_pts[frame.pts]         # exact: maps the frame back to its sample
        row = rows[(segment, idx)]
        yield idx, row, frame
for frame in stream.codec_context.decode(None):   # flush frames still buffered in the decoder
    ...                                           # same mapping as above
```

Counting decoded frames instead would shift every later frame by one if a single sample fails to decode. The PTS-to-index map makes a decode failure lose only that frame.

- If the file has **fewer** samples than rows (crash-recovered session), drop the extra rows ([01](01-ingest.md)). If it has **more**, stop with an error: that never happens with a valid package.
- Segments can change resolution mid-session (thermal step-down to 1080p). Select the intrinsics by each segment's `width × height`.

## 2. Colour conversion

The app tags every stream BT.709, limited ("TV") range, SDR (when `encoders[].config_level ≤ 1`). Convert with exactly those coefficients:

```python
rgb = frame.reformat(format="rgb24",
                     src_colorspace="ITU709", src_color_range="MPEG",
                     dst_color_range="JPEG").to_ndarray()
```

(Check the argument names against the installed PyAV version.) Converting with BT.601 or full-range assumptions shifts colours and contrast slightly. That doesn't break detection, but it changes what the VLM sees between sessions. Verify once with the colour test in [11](11-testing-and-acceptance.md) §3.

If `config_level ≥ 2` (no colour tags written), assume BT.709 limited and note it in QC.

## 3. Orientation

Frames are stored in **sensor orientation**. If `orientation.verdict = UPSIDE_DOWN`, rotate each frame 180° before detection and cropping. Keep the mapping so that rolling-shutter timing uses the sensor row: `r_sensor = H − 1 − r_display` (and likewise for columns). `PORTRAIT` sessions are flagged **bad** at ingest and are not processed: the camera geometry doesn't fit the calibration.

## 4. Frame quality

For each decoded frame, computed on a 2× downscaled grey image inside the mount ROI ([04](04-calibration-and-geometry.md)):

| Metric | Definition | Use |
|---|---|---|
| `sharpness` | Variance of the Laplacian (3×3) | Relative; compare within a session |
| `mean_luma`, `clip_hi`, `clip_lo` | Mean of Y; share of pixels > 250 / < 5 | Exposure problems |
| `black` | `mean_luma < 3` and `clip_lo > 0.98`, or inside an `image_black` → `image_ok` span in events.csv | Exclude |
| `smear_mm` | `speed × exposure_ns` | Exclude above 2× the blur budget; down-weight above 1× |
| `low_light` | From `exposure_control.csv` at the frame time | Down-weight |
| `bump` | From [02](02-time-and-trajectory.md) §7 | Down-weight |
| `stationary` | From [02](02-time-and-trajectory.md) | Thin to 1 frame per 2 s |
| `pos_quality` | From [02](02-time-and-trajectory.md) | Exclude `gap` frames from geolocated output |

Frames that are `black`, over the smear limit, or in a GNSS gap skip detection entirely. Everything else is detected; the scores feed best-frame selection in [07](07-best-frame-and-crop.md).

## 5. Performance

- 4K HEVC decoding on the CPU with FFmpeg is about 60–150 fps on a modern 8–16 core CPU, which is enough for real time. GPU decoding (TorchCodec on CUDA, or PyNvVideoCodec) is an optional optimisation; on Windows, TorchCodec's CUDA support is experimental. Verify it before relying on it.
- Never hold decoded 4K frames beyond the current window (24 MB each as RGB). Stage 07 keeps only the best candidate crop per open track and piece, and spools finished candidates to disk ([07](07-best-frame-and-crop.md) §1).
- Decode each segment once. Detection, association and cropping all consume the same stream of frames.

## 6. Tests

- On the emulator session in the test set: 1:1 join for every sample; the frame decoded at index `i` must match the frame extracted by `ffmpeg -vf select=eq(n\,i)` (pixel-equal).
- Corrupt one sample in the middle of a copied segment: every other frame must keep its correct row.
- A recovered session with 26 extra rows: rows trimmed, no error.
- Colour test: [11](11-testing-and-acceptance.md) §3.
