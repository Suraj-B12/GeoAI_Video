package com.geoai.capture.service

import android.content.Context
import android.os.Build
import android.provider.Settings
import com.geoai.capture.BuildConfig
import com.geoai.capture.io.ManifestBuilder
import com.geoai.capture.settings.AppSettings
import com.geoai.capture.settings.SurveyParams
import com.geoai.capture.util.Clocks
import org.json.JSONArray
import org.json.JSONObject

/** Builds session.json. Written once at the start and rewritten (atomically) at the end. */
object SessionMetadata {

    fun bootCount(context: Context): Int =
        runCatching { Settings.Global.getInt(context.contentResolver, Settings.Global.BOOT_COUNT) }.getOrDefault(-1)

    private fun numberOrNull(s: String): Any = s.trim().replace(',', '.').toDoubleOrNull() ?: JSONObject.NULL

    fun device(deviceId: String): JSONObject = JSONObject()
        .put("device_id", deviceId)
        .put("manufacturer", Build.MANUFACTURER)
        .put("brand", Build.BRAND)
        .put("model", Build.MODEL)
        .put("device", Build.DEVICE)
        .put("product", Build.PRODUCT)
        .put("hardware", Build.HARDWARE)
        .put("soc_manufacturer", Build.SOC_MANUFACTURER)
        .put("soc_model", Build.SOC_MODEL)
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("build_incremental", Build.VERSION.INCREMENTAL)
        .put("android_release", Build.VERSION.RELEASE)
        .put("sdk_int", Build.VERSION.SDK_INT)
        .put("security_patch", Build.VERSION.SECURITY_PATCH)

    fun initial(
        context: Context,
        sessionId: String,
        params: SurveyParams,
        settings: AppSettings,
        camera: JSONObject,
        gnssInfo: Map<String, String>,
        imu: JSONArray,
        storageRoot: String,
        sharedStorage: Boolean,
        start: Clocks.ClockPair,
        timestampSource: String,
    ): JSONObject {
        val survey = JSONObject()
            .put("road_id", params.roadId)
            .put("road_name", params.roadName)
            .put("direction", params.direction)
            .put("lane", params.lane)
            .put("start_chainage_km_text", params.startChainageKm)
            .put("start_chainage_km", numberOrNull(params.startChainageKm))
            .put("operator", params.operator)
            .put("vehicle", params.vehicle)
            .put("weather", params.weather)
            .put("notes", params.notes)
        val mount = JSONObject()
            .put("mount_id", params.mountId)
            .put("camera_facing", params.cameraFacing)
            .put("nominal_height_m", numberOrNull(params.nominalHeightM))
            .put("nominal_pitch_deg", numberOrNull(params.nominalPitchDeg))
        val gnss = JSONObject()
            .put("provider", "gps")
            .put("raw_measurements_logged", settings.rawGnss)
            .put("full_tracking", settings.rawGnss && settings.gnssFullTracking)
        gnssInfo.forEach { (k, v) -> gnss.put(k, v) }
        val o = JSONObject()
            .put("spec_version", ManifestBuilder.SPEC_VERSION)
            .put("session_id", sessionId)
            .put("session_type", params.sessionType)
            .put("boot_count_start", bootCount(context))
            .put(
                "app",
                JSONObject()
                    .put("package", BuildConfig.APPLICATION_ID)
                    .put("version_name", BuildConfig.VERSION_NAME)
                    .put("version_code", BuildConfig.VERSION_CODE)
                    .put("build_type", BuildConfig.BUILD_TYPE)
                    .put("settings", settings.toJson()),
            )
            .put("device", device(settings.deviceId))
            .put("survey", survey)
            .put("mount", mount)
            .put("camera", camera)
            .put("gnss", gnss)
            .put("imu", JSONObject().put("sensors", imu))
            .put("storage", JSONObject().put("root", storageRoot).put("shared_storage", sharedStorage))
            .put(
                "clock",
                JSONObject()
                    .put("session_clock", "elapsedRealtimeNanos (CLOCK_BOOTTIME)")
                    .put("camera_timestamp_source", timestampSource)
                    .put("elapsed_minus_mono_ns_at_start", start.elapsedMinusMonoNs),
            )
            .put("start", JSONObject().put("t_ns", start.elapsedMidNs).put("mono_ns", start.monoNs).put("wall_ms", start.wallMs))
        if (params.sessionType == SurveyParams.TYPE_CAL_INTRINSICS || params.sessionType == SurveyParams.TYPE_CAL_MOUNT) {
            o.put("calibration_target", JSONObject().put("spec", params.targetSpec).put("placement", params.targetPlacement))
        }
        return o
    }

    fun complete(
        context: Context,
        base: JSONObject,
        capture: JSONObject?,
        encoders: JSONArray,
        clockExtra: JSONObject,
        orientation: JSONObject,
        recording: JSONObject,
        end: Clocks.ClockPair,
        reason: String,
        stats: JSONObject,
    ): JSONObject {
        val o = JSONObject(base.toString())
        o.remove("recording_in_progress")
        o.put("boot_count_end", bootCount(context))
        capture?.let { o.put("capture_settings", it) }
        o.put("encoders", encoders)
        val clock = o.optJSONObject("clock") ?: JSONObject()
        clockExtra.keys().forEach { k -> clock.put(k, clockExtra.get(k)) }
        o.put("clock", clock)
        o.put("orientation", orientation)
        o.put("recording", recording)
        o.put("end", JSONObject().put("t_ns", end.elapsedMidNs).put("mono_ns", end.monoNs).put("wall_ms", end.wallMs).put("reason", reason))
        o.put("stats", stats)
        return o
    }
}
