package com.geoai.capture.settings

import android.content.Context
import android.media.MediaFormat
import android.os.Build
import org.json.JSONObject
import java.util.Locale
import java.util.UUID

enum class Resolution(val width: Int, val height: Int, val label: String) {
    UHD_2160(3840, 2160, "3840x2160 (4K)"),
    FHD_1080(1920, 1080, "1920x1080"),
}

enum class Codec(val mime: String, val label: String) {
    HEVC(MediaFormat.MIMETYPE_VIDEO_HEVC, "HEVC (H.265)"),
    AVC(MediaFormat.MIMETYPE_VIDEO_AVC, "AVC (H.264)"),
}

/** Capture settings. Changing any of these changes the data; they are all recorded in session.json. */
data class AppSettings(
    val deviceId: String,
    val resolution: Resolution = Resolution.UHD_2160,
    val codec: Codec = Codec.HEVC,
    val bitrateMbps: Int = 60,
    val fps: Int = 30,
    /** Ground smear allowed during one exposure, in millimetres (sets shutter from speed). */
    val blurBudgetMm: Double = 2.5,
    /** Longest exposure in normal light, as 1/x s. */
    val maxExposureDenominator: Int = 1000,
    /** Longest exposure once ISO is maxed (low light; blur risk is flagged), as 1/x s. */
    val lowLightExposureDenominator: Int = 500,
    val maxIso: Int = 1600,
    /** Target mean luma (0–255, gamma-encoded) of the road region. */
    val targetLuma: Int = 110,
    /** Speed assumed for the shutter when GNSS speed is unavailable. */
    val unknownSpeedKmh: Double = 50.0,
    val oisEnabled: Boolean = false,
    val segmentMinutes: Int = 5,
    val thermalStepDown: Boolean = true,
    val rawGnss: Boolean = false,
    val gnssFullTracking: Boolean = false,
    val keepScreenOn: Boolean = true,
    val alerts: Boolean = true,
    /** Start recording by itself once the camera is locked and GNSS is good (surveys and tests). */
    val autoRecord: Boolean = true,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("device_id", deviceId)
        .put("resolution", resolution.name)
        .put("codec", codec.name)
        .put("bitrate_mbps", bitrateMbps)
        .put("fps", fps)
        .put("blur_budget_mm", blurBudgetMm)
        .put("max_exposure_denominator", maxExposureDenominator)
        .put("low_light_exposure_denominator", lowLightExposureDenominator)
        .put("max_iso", maxIso)
        .put("target_luma", targetLuma)
        .put("unknown_speed_kmh", unknownSpeedKmh)
        .put("ois_enabled", oisEnabled)
        .put("segment_minutes", segmentMinutes)
        .put("thermal_step_down", thermalStepDown)
        .put("raw_gnss", rawGnss)
        .put("gnss_full_tracking", gnssFullTracking)
        .put("keep_screen_on", keepScreenOn)
        .put("alerts", alerts)
        .put("auto_record", autoRecord)

    companion object {
        fun fromJson(o: JSONObject, fallbackDeviceId: String): AppSettings {
            val d = AppSettings(fallbackDeviceId)
            return AppSettings(
                deviceId = o.optString("device_id", d.deviceId).ifBlank { d.deviceId },
                resolution = runCatching { Resolution.valueOf(o.getString("resolution")) }.getOrDefault(d.resolution),
                codec = runCatching { Codec.valueOf(o.getString("codec")) }.getOrDefault(d.codec),
                bitrateMbps = o.optInt("bitrate_mbps", d.bitrateMbps).coerceIn(8, 200),
                fps = 30,
                blurBudgetMm = o.optDouble("blur_budget_mm", d.blurBudgetMm).coerceIn(0.5, 20.0),
                maxExposureDenominator = o.optInt("max_exposure_denominator", d.maxExposureDenominator).coerceIn(60, 16000),
                lowLightExposureDenominator = o.optInt("low_light_exposure_denominator", d.lowLightExposureDenominator).coerceIn(30, 16000),
                maxIso = o.optInt("max_iso", d.maxIso).coerceIn(50, 12800),
                targetLuma = o.optInt("target_luma", d.targetLuma).coerceIn(40, 200),
                unknownSpeedKmh = o.optDouble("unknown_speed_kmh", d.unknownSpeedKmh).coerceIn(5.0, 120.0),
                oisEnabled = o.optBoolean("ois_enabled", d.oisEnabled),
                segmentMinutes = o.optInt("segment_minutes", d.segmentMinutes).coerceIn(1, 30),
                thermalStepDown = o.optBoolean("thermal_step_down", d.thermalStepDown),
                rawGnss = o.optBoolean("raw_gnss", d.rawGnss),
                gnssFullTracking = o.optBoolean("gnss_full_tracking", d.gnssFullTracking),
                keepScreenOn = o.optBoolean("keep_screen_on", d.keepScreenOn),
                alerts = o.optBoolean("alerts", d.alerts),
                autoRecord = o.optBoolean("auto_record", d.autoRecord),
            )
        }
    }
}

/** Operator-entered survey details. Numbers stay strings here; session.json stores raw + parsed. */
data class SurveyParams(
    val sessionType: String = TYPE_SURVEY,
    val roadId: String = "",
    val roadName: String = "",
    val direction: String = "increasing_chainage",
    val lane: String = "L1",
    val startChainageKm: String = "",
    val operator: String = "",
    val vehicle: String = "",
    val weather: String = "dry_sunny",
    val notes: String = "",
    val mountId: String = "M1",
    val cameraFacing: String = "forward",
    /** Measured on the pilot S21 mount (1.35 m); kept from the last session once changed. */
    val nominalHeightM: String = "1.35",
    val nominalPitchDeg: String = "",
    val targetSpec: String = "",
    val targetPlacement: String = "",
) {
    fun toJson(): JSONObject = JSONObject()
        .put("session_type", sessionType)
        .put("road_id", roadId)
        .put("road_name", roadName)
        .put("direction", direction)
        .put("lane", lane)
        .put("start_chainage_km", startChainageKm)
        .put("operator", operator)
        .put("vehicle", vehicle)
        .put("weather", weather)
        .put("notes", notes)
        .put("mount_id", mountId)
        .put("camera_facing", cameraFacing)
        .put("nominal_height_m", nominalHeightM)
        .put("nominal_pitch_deg", nominalPitchDeg)
        .put("target_spec", targetSpec)
        .put("target_placement", targetPlacement)

    companion object {
        const val TYPE_SURVEY = "survey"
        const val TYPE_TEST = "test"
        const val TYPE_CAL_INTRINSICS = "calibration_intrinsics"
        const val TYPE_CAL_MOUNT = "calibration_mount"
        val TYPES = listOf(TYPE_SURVEY, TYPE_TEST, TYPE_CAL_INTRINSICS, TYPE_CAL_MOUNT)

        fun fromJson(o: JSONObject): SurveyParams = SurveyParams(
            sessionType = o.optString("session_type", TYPE_SURVEY).takeIf { it in TYPES } ?: TYPE_SURVEY,
            roadId = o.optString("road_id"),
            roadName = o.optString("road_name"),
            direction = o.optString("direction", "increasing_chainage"),
            lane = o.optString("lane", "L1"),
            startChainageKm = o.optString("start_chainage_km"),
            operator = o.optString("operator"),
            vehicle = o.optString("vehicle"),
            weather = o.optString("weather", "dry_sunny"),
            notes = o.optString("notes"),
            mountId = o.optString("mount_id", "M1"),
            cameraFacing = o.optString("camera_facing", "forward"),
            nominalHeightM = o.optString("nominal_height_m"),
            nominalPitchDeg = o.optString("nominal_pitch_deg"),
            targetSpec = o.optString("target_spec"),
            targetPlacement = o.optString("target_placement"),
        )
    }
}

/** Bump when [AppSettings] defaults change in a way phones in the field should pick up. */
const val DEFAULTS_VERSION = 2

class SettingsRepository(context: Context) {
    private val prefs = context.getSharedPreferences("geoai_settings", Context.MODE_PRIVATE)

    private val defaultDeviceId: String by lazy {
        prefs.getString("generated_device_id", null) ?: run {
            val model = Build.MODEL.uppercase(Locale.ROOT).replace(Regex("[^A-Z0-9-]"), "")
            val id = "$model-" + UUID.randomUUID().toString().substring(0, 4).uppercase(Locale.ROOT)
            prefs.edit().putString("generated_device_id", id).apply()
            id
        }
    }

    fun load(): AppSettings {
        // Settings saved under older defaults are reset to the current recommended ones (device ID kept),
        // so an app update needs no manual changes on the phone.
        if (prefs.getInt("defaults_version", 0) < DEFAULTS_VERSION) {
            val old = prefs.getString("settings_json", null)
                ?.let { runCatching { AppSettings.fromJson(JSONObject(it), defaultDeviceId) }.getOrNull() }
            prefs.edit().putInt("defaults_version", DEFAULTS_VERSION).apply()
            save(AppSettings(old?.deviceId ?: defaultDeviceId))
        }
        val raw = prefs.getString("settings_json", null) ?: return AppSettings(defaultDeviceId)
        return runCatching { AppSettings.fromJson(JSONObject(raw), defaultDeviceId) }.getOrDefault(AppSettings(defaultDeviceId))
    }

    /** Restores every capture setting to the recommended default, keeping the device ID. */
    fun resetToDefaults() = save(AppSettings(load().deviceId))

    fun save(s: AppSettings) {
        val clean = s.copy(deviceId = s.deviceId.trim().replace(Regex("[^A-Za-z0-9_-]"), "-").ifBlank { defaultDeviceId })
        prefs.edit().putString("settings_json", clean.toJson().toString()).apply()
    }

    fun loadLastSurvey(): SurveyParams =
        prefs.getString("last_survey_json", null)?.let { runCatching { SurveyParams.fromJson(JSONObject(it)) }.getOrNull() }
            ?: SurveyParams()

    fun saveLastSurvey(p: SurveyParams) {
        prefs.edit().putString("last_survey_json", p.toJson().toString()).apply()
    }
}
