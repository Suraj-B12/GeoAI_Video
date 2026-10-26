# Focus diagnosis: S21 sessions of 26 Oct 2026

**Verdict: defocus, mainly in session 2. Motion blur is secondary.** Autofocus locked on different distances for the same mount: 0.46 dpt (2.17 m) in session 1 and 0.80 dpt (1.25 m) in session 2. 1.25 m is nearer than any road the camera sees, so session 2's analysed road lies largely outside the depth of field.

## Measurements

Thirty to forty frames per session were decoded, using the 4K luma of the central half of the road band.

**Direction of blur.** Motion blur would weaken the gradients along travel (image y) more than those across it (x).

| Session | Median y/x gradient energy | Correlation of that ratio with smear |
|---|---|---|
| 1 | 0.95 | −0.23 |
| 2 | 0.98 | −0.30 |

The blur is isotropic, so motion is not the main cause.

**Fine-detail ratio.** This is the variance of a difference-of-Gaussians at σ 0.7/1.4 divided by the variance at σ 2/4. It is robust to texture contrast and lower when out of focus.

| Session (focus) | Near 1.5–1.9 m | Mid 1.9–2.6 m | Far 2.6–3.5 m |
|---|---|---|---|
| 1 (0.46 dpt), all frames | 0.25 | 0.30 | 0.34 |
| 2 (0.80 dpt), all frames | 0.27 | 0.31 | 0.28 |
| 1, stationary frames | 0.27 | 0.39 | 0.41 |
| 2, stationary frames | 0.20 | 0.20 | 0.22 |

- Fine-detail energy (DoG 0.7/1.4 variance) is 1.2–2.5× higher in session 1 in every region: mid 0.435 against 0.173, far 0.198 against 0.144, near 0.310 against 0.254.
- At 100 %, session 2's crops are visibly soft, worst beyond 2 m.
- Both sessions are also generally soft: the 4K Laplacian variance is about 8–12. That comes from ISO 1000–1600 noise reduction and HEVC, which focus cannot fix.

**Depth of field.** f = 5.4 mm, f/1.8, circle of confusion = 2 output pixels = 3.78 µm, giving a hyperfocal distance of 4.29 m.

- Focus at 2.17 m (session 1): sharp from 1.44 to 4.39 m.
- Focus at 1.25 m (session 2): sharp from 0.97 to 1.76 m.

**Road actually seen.** Mount 1.35 m, tilt 25.1° at start (gravity), S21 intrinsics, 16:9 crop, road up to 4 m ahead.

- Depth along the optical axis runs from 1.86 to 4.18 m.
- The best focus is the dioptre midpoint: 0.389 dpt (2.57 m), with at most 1.28 px of defocus blur at the ends.
- Blur that the recorded focus caused:

| Focus | Near end | Far end |
|---|---|---|
| Session 1, 0.46 dpt | 0.66 px | 1.90 px |
| Session 2, 0.80 dpt | 2.25 px | 4.81 px |

**Lens.** `LENS_INFO_FOCUS_DISTANCE_CALIBRATION` = CALIBRATED and minimum focus 10 dpt, so metric focus distances can be set directly.

## Fix (app)

- **Geometric target.** Focus comes from geometry (`camera/FocusPlanner.kt`): gravity tilt, mount height from the survey form and the Camera2 intrinsics give the road rows and their depths. Focus is set to the dioptre midpoint with autofocus off.
- **Sweep check.** A sweep of ±0.6 dpt, scored only on analysis frames whose capture result shows the lens settled at that step, can override the target only when one position is clearly sharper (by 15 % or more, at least 0.25 dpt away). The 640×360 analysis stream is about 6× coarser than 4K, so fine differences there are noise.
- **Road-only metering.** AF regions, AE regions and the luma/sharpness region of the analysis stream are the road rows only (columns 20–80 %), never sky, buildings or vehicles.
- **Fallback.** Without a calibrated lens or usable geometry, autofocus runs on the road region and then holds.
- **Logging.** Everything is logged in a `focus` event and in `session.json → capture_settings.focus` (plan, sweep scores, chosen dioptres, method). The live screen shows "Focus set by".
