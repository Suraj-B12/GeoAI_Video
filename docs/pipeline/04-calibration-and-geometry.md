# 04 · Calibration and Geometry

Goal: turn a pixel into a point on the road, in metres, relative to the camera. Then (with the trajectory from [02](02-time-and-trajectory.md)) turn it into a latitude/longitude. Without this, the "GPS coordinate of the crop" is just where the phone was, which can be 0.5–16 m from the distress.

Two calibrations, both computed on the PC from sessions the app records:

| Calibration | Recorded as | Valid for | Redo when |
|---|---|---|---|
| Lens (intrinsics) | `calibration_intrinsics` session | One phone unit, one recording resolution, one focus setting | Different phone, resolution or focus lock; after a drop or repair |
| Mount (extrinsics) | `calibration_mount` session | One mount ID on one vehicle | Every survey day, and whenever the mount is touched |

## 1. Lens calibration

**Board.** ChArUco, printed and glued flat on a rigid board, about 1.2 m × 0.8 m (it is used at 1.5–3.5 m from the camera). Suggested: 9 × 6 squares of 130 mm, markers 100 mm, dictionary `DICT_5X5_100`. **Measure the printed square size with a steel rule**; printers scale. Enter the spec in the app's "Target spec" field, e.g. `charuco 9x6 sq=130.2mm mk=100.1mm DICT_5X5_100`.

**Capture.**
1. Put the phone on its survey mount, set to the resolution being calibrated.
2. Focus must lock exactly as it does for surveying. Start the `calibration_intrinsics` session while the camera looks at the road: Prepare locks focus and exposure automatically.
3. Only then bring the board into view. Don't press "Re-lock" (focus and exposure) while the board is in view.
4. Move the board slowly through the whole frame, especially the corners, tilted ±30–45°, for about 60–90 s.
5. Repeat at 1080p (Settings → Resolution), so segments recorded after a thermal step-down also have a calibration.

**Compute** (`geoai calibrate-intrinsics`):
1. Decode every 5th frame. Reject frames with low sharpness or motion blur.
2. Detect with `cv2.aruco.CharucoDetector` (OpenCV ≥ 4.8). Keep views with ≥ 20 corners.
3. Choose 40–80 views that cover the image evenly (grid of 6 × 4 cells; at least 2 views per cell where possible).
4. `cv2.calibrateCamera` with the standard 5-parameter model `(k1, k2, p1, p2, k3)`. The S21 main lens has mild distortion; don't use the fisheye or rational model unless residuals demand it.
5. Accept if RMS reprojection error ≤ 0.5 px at 4K (≤ 0.35 px at 1080p) and no single view exceeds 3× the RMS. Otherwise drop the worst views and repeat.

**Output** `data/calibration/intrinsics/<device_id>/<W>x<H>/<session_id>.json`:
`{device_id, camera_id, width, height, focus_diopters, K (3×3), dist [k1,k2,p1,p2,k3], rms_px, n_views, coverage, board, created_utc}`.

**Prior and check:** `camera.lens_intrinsic_calibration` in session.json is the factory value for the sensor array. After scaling to the recording resolution and crop it should agree with the result within a few per cent. A big disagreement means a calibration mistake.

## 2. Mount calibration

**Vehicle frame V:** origin on the ground under the **centre of the rear axle**, X to the right, Y forward, Z up. Measure two reference points on the vehicle (left and right rear wheel centres projected to the ground) to set it out. A string line or chalk line marks the vehicle's centre line on the ground.

**Ground target.** Six or more ArUco markers (`DICT_5X5_100`, 200 mm, IDs 0–11), printed on matte, non-reflective sheets and laid flat in the camera's view. Measure each marker centre's (X, Y) in the vehicle frame with a tape, to ±5 mm. Alternative: one large ChArUco mat (2 m × 1 m) whose origin and axis are measured relative to the vehicle frame. Enter the measurements in "Target placement" (or as a CSV alongside), e.g. `id0 X=-0.80 Y=3.10; id1 X=0.00 Y=3.10; …`.

**Capture.** Vehicle parked on flat, level ground, engine running (suspension at operating height), driver seated, nothing moving, 10 s recording.

**Compute** (`geoai calibrate-mount`):
1. Detect markers in 20 frames; average corner positions; undistort with the intrinsics.
2. Solve the camera pose with `cv2.solvePnP` (IPPE or SQPNP), all marker corners, Z = 0, then refine with LM.
3. Report `height_m`, `pitch_deg`, `roll_deg`, `yaw_deg` (yaw relative to the vehicle's forward axis; ≈ 180° for a rearward camera), the camera centre (X, Y), and the reprojection RMS (accept ≤ 1.5 px at 4K).
4. **Check with held-out markers:** hold 2 markers out of the solve and compare their projected positions with the tape values (accept < 3 cm at ≤ 3 m range).
5. Average the gravity vector (`imu gravity`) over the 10 s as `gravity_ref_device`.
6. Compute the ROI polygon: image area where the road is visible (exclude the vehicle body) and where the along-track ground resolution is better than `max_gsd_mm_per_px` (default 3.0). The far field beyond that is too coarse to classify.

**Output** `data/calibration/mount/<mount_id>/<session_id>.json`:
`{mount_id, vehicle, created_utc, intrinsics_id, R_cam_from_vehicle (3×3), t_cam_in_vehicle_m (3), height_m, pitch_deg, roll_deg, yaw_deg, camera_faces, H_ground_from_undistorted (3×3), roi_polygon_px, max_gsd_mm_per_px, reprojection_rms_px, check_errors_m, gravity_ref_device}`.

## 3. Pixel → road → map

For a pixel `(u, v)` in a frame (display orientation, see [03](03-video-decoding.md) §3):

1. **Undistort:** `cv2.undistortPoints([[u, v]], K, dist, P=K)` gives an ideal pinhole pixel.
2. **Ray:** `d_cam = K⁻¹ [u', v', 1]ᵀ`; in the vehicle frame `d_V = Rᵀ d_cam` (R = `R_cam_from_vehicle`), from camera centre `c_V = t_cam_in_vehicle_m`. Careful: `solvePnP` returns `rvec, tvec` with `X_cam = R·X_V + tvec`, so the camera centre is `c_V = −Rᵀ·tvec`, not `tvec`.
3. **Intersect Z = 0:** `λ = −c_V.z / d_V.z` (requires `d_V.z < 0`: below the horizon). Ground point `P_V = c_V + λ d_V`.
4. **Relative to the camera nadir:** `x = P_V.x − c_V.x` (right), `y = P_V.y − c_V.y` (forward), `range = |(x, y)|`.
5. **To map coordinates** at the detection's own time (row-specific, [02](02-time-and-trajectory.md) §6), with camera ENU position `(E₀, N₀)` and heading ψ (clockwise from north):
   ```
   E = E₀ + y·sin ψ + x·cos ψ
   N = N₀ + y·cos ψ − x·sin ψ
   ```
   Then `pymap3d.enu2geodetic` → lat/lon. The phone's GNSS antenna is a few centimetres from its camera, so the lever arm is taken as zero.
6. **Ground resolution** at a pixel: project `(u±1, v)` and `(u, v±1)`; the distances are the across-track and along-track mm per pixel.

Distress footprint: project the detection's polygon (mask outline, or the four box corners) point by point. In perspective, a box becomes a trapezoid on the road. Area, length (long side of the minimum rotated rectangle) and centroid come from that ground polygon in metres (shapely).

**Validity:** only pixels inside the ROI, below the horizon, and with range ≤ `max_range_m` (default 6 m) are projected. Footprints that extend beyond those limits are clipped and flagged `partial`.

## 4. Error budget (what "accurate" means here)

| Source | Size (S21 phone GNSS) |
|---|---|
| GNSS absolute position | 2–5 m typical (68 %), worse under trees, by buildings or near flyovers |
| Residual timing lag × speed | ≈ 0.2–0.5 m at 40 km/h after correction |
| Heading error × range | 1° × 3 m ≈ 5 cm |
| Calibration (pose, focal length) | ≈ 2–5 cm at 3 m |
| Suspension pitch (not corrected, [02](02-time-and-trajectory.md) §7) | ≈ 5–10 cm |
| Road crown and grade (flat-road assumption) | ≈ 2–5 cm |

Distress positions are therefore dominated by GNSS: about 3–5 m absolute, but **relative** positions within a short stretch are good to about 0.2–0.5 m. That relative accuracy is what makes the de-duplication in [06](06-association.md) work. For sub-metre absolute positions, add RTK or PPK ([PLAN](../PLAN.md)).

## 5. Tests

- Synthetic camera at a known pose: project → back-project a grid of road points; error < 1 mm.
- A real `calibration_mount` session: held-out marker error < 3 cm.
- Rearward camera (yaw 180°): a point 2 m behind the camera must map 2 m *behind* the phone's position on the map.
- Round trip on a curve: with ψ changing, `E, N` of a fixed road point seen in two frames 0.5 s apart agree within the trajectory's relative accuracy (< 0.3 m).
