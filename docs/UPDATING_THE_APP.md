# Updating the capture app on the S21 from another PC

Android only installs an update over an existing app if the new APK is signed with **the same key**. The app on the phone is a debug build signed with the original PC's debug keystore. Without that key, `adb install -r` fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, and the only fix is to uninstall the app, which **deletes its settings and any sessions not yet copied to the PC**.

## Key (not in git; copy it by USB)

| Item | Value |
|---|---|
| File | `debug.keystore` (on the original PC: `C:\Users\suraj\.android\debug.keystore`) |
| Store password | `android` |
| Key alias / key password | `androiddebugkey` / `android` |
| Valid | 22 Aug 2025 to 15 Aug 2055 |

Put it at `%USERPROFILE%\.android\debug.keystore` on the other PC **before the first build**. Gradle picks it up on its own; the build files need no change. If the other PC has already created its own debug keystore there, replace it.

The phone's *session-signing* key is separate. It lives in the phone's hardware-backed Android Keystore and cannot be copied. It stays on the phone and survives updates (not uninstalls).

## Build and install

1. Android Studio (or the SDK and JDK 17+), and `android/local.properties` with `sdk.dir=<this PC's SDK path>`. That file is machine-specific and not in git.
2. Phone: USB debugging on, authorised for this PC (`adb devices` shows `device`).
3. From `android/`:

```bash
env -u ANDROID_HOME ./gradlew :app:assembleDebug :app:testDebugUnitTest --no-daemon --console=plain -q
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

`-r` keeps the app's data. Never `adb uninstall` before copying sessions off (`tools/pull_sessions.py`).

The installed app's settings migrate through `DEFAULTS_VERSION` in `settings/AppSettings.kt`; bump it only when a default must change on phones in the field.
