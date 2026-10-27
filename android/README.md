# GeoAI Capture (Android)

The capture app for the pavement survey. It records 4K pavement video from a vehicle-mounted phone. For every encoded frame it logs the exact exposure time and camera metadata, on the same clock as GNSS fixes, IMU samples, thermal state and operator markers. It writes a signed **session package**, which the PC pulls over USB.

- **Fully offline:** the app has no INTERNET permission, and nothing leaves the phone except over the USB cable.
- **Data contract:** [../docs/session-package-spec.md](../docs/session-package-spec.md) (spec version 1.1).
- **Design background:** [../docs/PLAN.md](../docs/PLAN.md) §5.

## 1. Build

Requirements: JDK 17 or newer, and the Android SDK with platform 36. `local.properties` points `sdk.dir` at the SDK. The Gradle 8.14 wrapper is included.

```bash
cd android
./gradlew :app:assembleDebug :app:testDebugUnitTest
```

- If `ANDROID_HOME` is set to a different SDK than `local.properties`, unset it for the build: `env -u ANDROID_HOME ./gradlew …` in Git Bash.
- The APK is written to `app/build/outputs/apk/debug/app-debug.apk`.
- `assembleRelease` is signed with the debug key, which is acceptable for the pilot only. Use a real keystore before any wider distribution.
- Versions: AGP 8.11.1, Kotlin 2.2.20, Jetpack Compose, Media3 muxer 1.11.1; compileSdk 36, targetSdk 35, minSdk 31.

## 2. Install (offline, over USB)

1. On the S21, enable developer options: **Settings → About phone → Software information → tap "Build number" 7 times**.
2. Then turn on **Settings → Developer options → USB debugging**, connect a data-capable USB-C cable (charge-only cables don't work), and accept the PC's key ("Always allow from this computer").
3. Install:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

**What survives updates and uninstalls:**
- Updating (`-r`) keeps all sessions and settings.
- Sessions in `/sdcard/GeoAI` survive even an uninstall.
- **Uninstalling deletes the phone's signing key**, which lives in the Android Keystore. A reinstall creates a new one. Old sessions stay verifiable, because every manifest carries its own public key and attestation chain, but record the change in the device log.

## 3. First launch

Grant, in order:
- Camera.
- **Precise** location ("While using the app" is enough; the app records from a foreground service).
- Notifications.
- **All files access**: a settings page opens. It lets sessions live in `/sdcard/GeoAI/sessions`, where ADB can reach them and where they survive an uninstall.
- The battery-optimisation exemption dialog.

### Samsung (One UI) settings: essential on the S21

Without these, One UI may stop the recording when the screen turns off.

| Setting | Path (names vary slightly between One UI versions) |
|---|---|
| App battery: **Unrestricted** | Settings → Apps → GeoAI Capture → Battery |
| **Never sleeping apps**: add GeoAI Capture | Settings → Battery (or Battery and device care) → Background usage limits |
| Battery protection on (85 %, or "Maximum 80 %" on One UI 6.1+) | Settings → Battery → Battery protection. Less heat while the phone is on constant power in the vehicle |
| Software auto-update off | Settings → Software update. An OS update can change camera behaviour; re-run the Device check after any update |
| Location on | Quick settings or Settings → Location |

**GNSS without the internet.** The app never uses the network. If the phone itself has mobile data, Android downloads satellite assistance data and the first fix takes seconds. In airplane mode, or with no SIM, the first fix of the day can take about a minute in open sky, and longer under trees or if the phone hasn't been used for days. Start the app outdoors a few minutes before driving. The app won't start a survey recording until it has a fix (§6).

## 4. First day with the S21: checklist

Do these in order. Keep the results with the pilot data.

1. **Device check** (Home → Device check, about 2 minutes, outdoors for the GNSS part). Save the report. It lands in `/sdcard/GeoAI/device_reports/`, and you can pull it with `adb pull /sdcard/GeoAI/device_reports`.

   | Expected | If not |
   |---|---|
   | Camera level FULL or LEVEL_3, MANUAL_SENSOR available | No manual exposure: frames will be blurred. Stop and report |
   | Timestamp source REALTIME | `UNKNOWN` still works (the app measures the offset), but timing is approximate; the marker test measures the lag |
   | Minimum exposure ≤ 1/8000 s, 4K 30 fps HEVC encoder, EIS can be turned off | Note it |
   | GNSS: about 1 fix per second, satellites from several constellations, L5/E5a signals | Note it |
   | Thermal status and headroom report values | Thermal step-down still works from the status alone |

2. **Desk test:** record a 10-minute `test` session, then pull and validate it (§8). Expect exit code 0, 0 dropped frames, and a join residual under 1 ms.
3. **Kill test:** start a `test` recording, then force-stop the app (Settings → Apps → GeoAI Capture → Force stop). Relaunch it. The session appears as INCOMPLETE: tap **Finalise**, pull and validate. The video must be readable up to the last 2-second fragment.
4. **Heat soak:** 60 minutes of recording on the mount in the sun, with the charger connected. Note when the first thermal warning and the 1080p step-down happen. There must be no crash.
5. **Drive:** 15 minutes at 20–50 km/h. Check that the shutter follows speed (Exposure row in the status panel) and that there's no LOW LIGHT in daylight.
6. **Calibration sessions** (§7): intrinsics at 4K and at 1080p, then the mount.
7. **Marker test for GNSS latency:** record it now and analyse it once the pipeline exists ([../docs/pipeline/11-testing-and-acceptance.md](../docs/pipeline/11-testing-and-acceptance.md) §4).

## 5. Mounting

- **Landscape**, camera looking at the road, pitched down as computed with [`tools/mount_geometry.py`](../tools/mount_geometry.py) (recommended: exterior bracket, 1.6–2.0 m high, 45–55° down; [PLAN §4.3, §5.9](../docs/PLAN.md)).
- The status panel's **Orientation** row must read **upright ✓**.
  - If it says UPSIDE DOWN, turn the phone 180°. The PC can correct it, but don't rely on that.
  - PORTRAIT sessions are rejected by the PC.
- Segment mode (one frame per 5 m, [../docs/pipeline/13-segment-frames.md](../docs/pipeline/13-segment-frames.md)) needs the frame to show 6 m of road beyond the nearest visible road. The status panel checks this live from gravity and the survey form's height:
  - **Tilt**: the camera's angle below the horizon;
  - **Road in view**: nearest to farthest road along the image's centre column ("60+" with the horizon in view);
  - **5 m segment** must read **fits**. On the S21's 16:9 4K frame this holds up to about 31° of tilt at 1.35 m (33° at 1.6 m, 36° at 2.0 m), so the steep 45–55° exterior mount suits crop mode only. The pilot mount (23°, 1.35 m) shows 1.45–33 m of road.
- Take the phone out of its case. Shade it, keep air flowing, and power it with a USB-C PD charger. Keep metal away from the top of the phone (GNSS antenna). Use vibration isolators and a safety tether.
- Clean the lens before every run.

## 6. Workflow

1. **Setup:**
   - **Session type:** `survey` / `test` / `calibration_intrinsics` / `calibration_mount`.
   - **Survey form:** road ID and name, direction (increasing or decreasing chainage), lane, start chainage (km), operator, vehicle, weather, notes.
   - **Mount:** mount ID, which way the camera faces, nominal height and pitch.
   - **Calibration sessions only:** the target description.
2. **Prepare:** opens the camera and locks focus and exposure on the road. Keep the vehicle still for a few seconds.
   - **Focus is set from geometry**, not left to autofocus (`camera/FocusPlanner.kt`): gravity gives the camera's tilt and the survey form gives the mount height. From these the app computes the road a segment-mode frame must show sharply ([../docs/pipeline/13-segment-frames.md](../docs/pipeline/13-segment-frames.md) §2): from the nearest visible road to 6 m beyond it (a 5 m segment whose near edge is up to 1 m beyond that road), cut at the farthest visible road. Focus goes to the dioptre midpoint of that road's depth range, so its near and far ends are equally sharp. On the pilot mount (23° tilt, 1.35 m) the road is 1.45–7.45 m ahead and focus is 0.34 dpt (3.0 m), with at most 1.7 px of defocus blur at 4K (0.6 px after the VLM's resize). The former 4 m range focused at 0.38 dpt and blurred a segment's far edge by 2.1 px.
   - A five-step sweep (±0.6 dpt) scores the road's sharpness at each setting. It overrides the geometry only if another setting is clearly sharper (by at least 15 %, and at least 0.25 dpt away).
   - Autofocus, metering and the sharpness measure look only at the road rows (the middle 60 % of the width), never at sky, buildings or vehicles.
   - If the lens doesn't report calibrated focus distances, or there is no gravity reading, the app falls back to autofocus on the road area only, then holds that focus.
   - The plan, the sweep scores and the decision are saved in `session.json` (`capture_settings.focus`) and in `events.csv`. The live screen shows "Focus set by".
   - Before 26 Oct 2026, autofocus decided alone. On the same mount it chose 0.46 dpt in one session and 0.80 dpt in another; 0.80 dpt blurred the far half of the road ([field/focus_diagnosis.md](../field/focus_diagnosis.md)).
3. **Record:**
   - A **survey** session can't record until GNSS has a fix ≤ 10 m that is no older than 2 s.
   - **Record anyway** starts without the fix and is logged as `recording_forced_without_gnss`. Use it only for tests.
   - If location is off, a banner offers **Turn on**.
4. **While driving:**
   - Keep within the speed band: the speed turns amber above 40 km/h and red above 50.
   - Press **Marker** at every km stone (`km_stone`, with the chainage), and for notable distresses or events.
   - Alerts (vibration and notification) report lost GNSS, heat, storage and camera problems.
5. **Stop:** the app records 3 s more, finalises the files, and writes and signs `manifest.json`.
6. **Sessions screen:**
   - **Finalise** recovers sessions interrupted by a crash or power loss.
   - After the PC has pulled a session, it shows **VERIFIED_ON_PC**. Delete only those. Deleting anything else asks for a separate "Delete UNVERIFIED session?" confirmation, because that data would be lost for good.

**Automatic protections:**
- **Storage:** the app needs 5 GB free to start and stops cleanly at 2 GB.
- **Heat:** MODERATE gives a warning; SEVERE steps down to 1080p in a new segment; CRITICAL stops cleanly.
- **Camera errors:** if the camera is interrupted (e.g. another app takes it), the app retries up to 3 times after 3, 6 and 10 s, at most 10 times per session.
- **GNSS:** it restarts the request after 15 s without a fix.

## 7. Calibration sessions

Both are computed on the PC ([../docs/pipeline/04-calibration-and-geometry.md](../docs/pipeline/04-calibration-and-geometry.md)). The app only records them.

- **`calibration_intrinsics`**: once per phone and per resolution, and again after a drop or repair.
  1. Phone on its mount, looking at the road. **Prepare** locks focus on the road.
  2. Only then bring in the ChArUco board, and don't press "Re-lock" while it is in view.
  3. Move the board slowly through the whole frame at 1.5–3.5 m, corners included, tilted ±30–45°, for 60–90 s.
  4. Enter the board as measured with a steel rule, e.g. `charuco 9x6 sq=130.2mm mk=100.1mm DICT_5X5_100`.
- **`calibration_mount`**: at the start **and the end** of every survey day, and whenever the mount is touched.
  1. Park on flat, level ground with the engine running and the driver seated.
  2. Lay six or more ArUco markers in view.
  3. Measure each marker centre in the vehicle frame (origin on the ground under the rear-axle centre, X right, Y forward), and enter them as `id0 X=-0.80 Y=3.10; id1 …`.
  4. Record 10 s with nothing moving.

## 8. Transfer to the PC (USB)

On the PC, with [Android platform-tools](https://developer.android.com/tools/releases/platform-tools) (`adb`) and Python 3.9 or newer:

```bash
python tools/pull_sessions.py D:/geoai/sessions
```

The tool:
1. pulls each finalised session into `.incoming/`;
2. verifies every file's size and SHA-256, plus the signature if the `cryptography` package is installed (install it);
3. moves the session into place;
4. pushes `pc_receipt.json` back to the phone.

It never deletes anything on the phone.

To check a session by hand:

```bash
python tools/validate_session.py D:/geoai/sessions/<session_id>
```

Exit code 0 = no FAIL, 1 = at least one FAIL, 2 = the session can't be read. The video checks need `ffprobe` (FFmpeg) on PATH; without it they are skipped with a warning.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "Waiting for GNSS fix" | Go outdoors with open sky and wait (see §3, GNSS without the internet). Check that location is on and precise |
| Recording stops when the screen is off | The Samsung battery settings in §3 are missing |
| Exposure mode `AUTO_FALLBACK` (red) | The camera refused manual exposure; frames will be motion-blurred. Re-run the Device check |
| "LOW LIGHT" in the Exposure row | ISO hit 1600 and the shutter was lengthened. Blur risk: avoid deep shade, dusk and dawn |
| Unmatched frames > 0 | Some frames had no camera metadata; their times are estimated. Report it, with the session |
| Session shows INCOMPLETE | The app was interrupted. Tap **Finalise** in Sessions |
| `adb devices` shows nothing or `unauthorized` | Use a data cable, re-enable USB debugging, accept the key prompt on the phone |
| Orientation not "upright ✓" | Re-mount in landscape; turn the phone 180° if it says UPSIDE DOWN |

## 10. Debug intents (debug builds only)

```bash
adb shell am start -n com.geoai.capture/.ui.MainActivity --ez debug_autostart true --ei debug_duration_s 60 --es debug_session_type test
```

```bash
adb shell am start -n com.geoai.capture/.ui.MainActivity --ez debug_recover true
```

```bash
adb shell am start -n com.geoai.capture/.ui.MainActivity --ez debug_dump_threads true
```

- The first records a 60 s session without touching the screen. Add `--ei debug_min_free_mb 300` on emulators, which have little storage.
- The second finalises interrupted sessions.
- The third logs the app's thread stacks (`adb logcat -s GeoAI-threads`).
- All app logs use the tag `GeoAI`.

In Git Bash, `export MSYS_NO_PATHCONV=1` first, or paths like `/sdcard/...` get rewritten.

## 11. Test status

**Verified on the Android emulator (API 36):**
- 44 JVM unit tests (exposure control, frame matching, road focus plan and live mount check, road coverage, locale-proof formatting);
- sessions validate end to end: checksums, signature, 1:1 frame join, residual < 1 µs, 0 drops;
- thermal critical stop;
- crash recovery, with the fragmented MP4 readable to its last fragment;
- USB pull and receipt round trip; Cancel deleting the session folder;
- the GNSS gate blocking Record;
- GNSS resuming after location is switched off and on.

**Not yet verified:** everything that needs the real S21. That covers the real camera (manual exposure, focus lock, 4K HEVC encoder, timestamp source), real GNSS, heat behaviour, and recovery after a camera disconnect, which the emulator couldn't trigger. The checklist in §4 covers these.

## 12. Code map

| Path (`app/src/main/java/com/geoai/capture/`) | Role |
|---|---|
| `service/SurveyService.kt` | Foreground service: session lifecycle, GNSS gate, monitoring, thermal ladder, recovery |
| `camera/` | Camera2 recorder, manual exposure loop, frame ↔ metadata matcher, encoder, fragmented-MP4 segment writer, analysis stream |
| `sensors/` | GNSS (fixes, status, NMEA, optional raw), IMU, thermal, clock sync, orientation check |
| `io/` | Session store and recovery, manifest and Keystore signing, buffered CSV writers, events |
| `devicecheck/` | Device qualification report |
| `ui/` | Compose screens: permissions, home, setup, live, sessions, device check, settings |
| `util/` | Clocks, locale-proof formatting, canonical JSON, diagnostics |
