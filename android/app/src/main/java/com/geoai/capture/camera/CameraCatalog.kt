package com.geoai.capture.camera

import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.media.MediaCodec
import android.util.Range
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.hypot

/** Picks the main back camera and describes what it can do (for session.json and the device check). */
object CameraCatalog {

    fun levelRank(level: Int?): Int = when (level) {
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> 4
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> 3
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> 2
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> 1
        else -> 0
    }

    fun levelName(level: Int?): String = when (level) {
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
        CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
        else -> "UNKNOWN($level)"
    }

    private val capabilityNames = mapOf(
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE to "BACKWARD_COMPATIBLE",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR to "MANUAL_SENSOR",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING to "MANUAL_POST_PROCESSING",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW to "RAW",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING to "PRIVATE_REPROCESSING",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_READ_SENSOR_SETTINGS to "READ_SENSOR_SETTINGS",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BURST_CAPTURE to "BURST_CAPTURE",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_YUV_REPROCESSING to "YUV_REPROCESSING",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT to "DEPTH_OUTPUT",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_CONSTRAINED_HIGH_SPEED_VIDEO to "CONSTRAINED_HIGH_SPEED_VIDEO",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MOTION_TRACKING to "MOTION_TRACKING",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA to "LOGICAL_MULTI_CAMERA",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MONOCHROME to "MONOCHROME",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_SECURE_IMAGE_DATA to "SECURE_IMAGE_DATA",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_SYSTEM_CAMERA to "SYSTEM_CAMERA",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_OFFLINE_PROCESSING to "OFFLINE_PROCESSING",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_ULTRA_HIGH_RESOLUTION_SENSOR to "ULTRA_HIGH_RESOLUTION_SENSOR",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_REMOSAIC_REPROCESSING to "REMOSAIC_REPROCESSING",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_DYNAMIC_RANGE_TEN_BIT to "DYNAMIC_RANGE_TEN_BIT",
        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_STREAM_USE_CASE to "STREAM_USE_CASE",
    )

    fun capabilities(c: CameraCharacteristics): List<String> =
        (c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)).map { capabilityNames[it] ?: "CAP_$it" }

    fun hasManualSensor(c: CameraCharacteristics): Boolean =
        c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true

    fun timestampSourceName(c: CameraCharacteristics): String =
        when (c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)) {
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN"
            else -> "UNREPORTED"
        }

    /** 35 mm-equivalent focal length from the physical focal length and sensor diagonal. */
    fun equivalentFocalMm(c: CameraCharacteristics): Double? {
        val f = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
        val s = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
        val diag = hypot(s.width.toDouble(), s.height.toDouble())
        return if (diag > 0) f * 43.27 / diag else null
    }

    /** Back camera with the best hardware level, then the focal length closest to a ~26 mm main lens. */
    fun chooseMainBackCamera(manager: CameraManager): String? =
        manager.cameraIdList.mapNotNull { id ->
            val c = runCatching { manager.getCameraCharacteristics(id) }.getOrNull() ?: return@mapNotNull null
            if (c.get(CameraCharacteristics.LENS_FACING) != CameraMetadata.LENS_FACING_BACK) return@mapNotNull null
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
            if (!caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE)) return@mapNotNull null
            Triple(id, levelRank(c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)), abs((equivalentFocalMm(c) ?: 26.0) - 26.0))
        }.sortedWith(compareByDescending<Triple<String, Int, Double>> { it.second }.thenBy { it.third }).firstOrNull()?.first

    /** Encoder output sizes the camera supports, largest first. */
    fun recordSizes(c: CameraCharacteristics): List<Size> {
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return emptyList()
        return (map.getOutputSizes(MediaCodec::class.java) ?: emptyArray()).sortedByDescending { it.width.toLong() * it.height }
    }

    /** Best record size ≤ the requested one with the same 16:9 aspect. */
    fun pickRecordSize(c: CameraCharacteristics, wantW: Int, wantH: Int): Size? {
        val sizes = recordSizes(c)
        sizes.firstOrNull { it.width == wantW && it.height == wantH }?.let { return it }
        return sizes.filter { it.width <= wantW && it.height <= wantH && it.width * 9 == it.height * 16 }
            .maxByOrNull { it.width.toLong() * it.height }
    }

    /** Smallest 16:9 YUV size at least [minWidth] wide, same field of view as the record stream. */
    fun pickAnalysisSize(c: CameraCharacteristics, minWidth: Int = 640): Size? {
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val yuv = map.getOutputSizes(ImageFormat.YUV_420_888) ?: return null
        return yuv.filter { it.width * 9 == it.height * 16 && it.width >= minWidth }.minByOrNull { it.width }
            ?: yuv.filter { it.width >= minWidth }.minByOrNull { it.width }
    }

    fun minFrameDurationNs(c: CameraCharacteristics, size: Size): Long? {
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        return runCatching { map.getOutputMinFrameDuration(MediaCodec::class.java, size) }.getOrNull()
    }

    /** Fixed [fps, fps] AE range if available, else the range ending at fps with the highest floor. */
    fun pickFpsRange(c: CameraCharacteristics, fps: Int): Range<Int>? {
        val ranges = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: return null
        return ranges.firstOrNull { it.lower == fps && it.upper == fps }
            ?: ranges.filter { it.upper == fps }.maxByOrNull { it.lower }
    }

    private fun intArray(a: IntArray?): JSONArray = JSONArray().also { j -> a?.forEach { j.put(it) } }
    private fun floatArray(a: FloatArray?): Any = a?.let { arr -> JSONArray().also { j -> arr.forEach { j.put(it.toDouble()) } } } ?: JSONObject.NULL

    fun describe(id: String, c: CameraCharacteristics): JSONObject {
        val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        val pixel = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        val phys = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val exp = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val iso = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val zoom = c.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
        return JSONObject()
            .put("camera_id", id)
            .put("lens_facing", when (c.get(CameraCharacteristics.LENS_FACING)) {
                CameraMetadata.LENS_FACING_BACK -> "back"; CameraMetadata.LENS_FACING_FRONT -> "front"; else -> "external"
            })
            .put("physical_camera_ids", JSONArray(c.physicalCameraIds.toList()))
            .put("hardware_level", levelName(c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)))
            .put("capabilities", JSONArray(capabilities(c)))
            .put("timestamp_source", timestampSourceName(c))
            .put("sensor_orientation_deg", c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: JSONObject.NULL)
            .put("focal_lengths_mm", floatArray(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)))
            .put("equivalent_focal_mm", equivalentFocalMm(c) ?: JSONObject.NULL)
            .put("apertures", floatArray(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)))
            .put("sensor_physical_size_mm", phys?.let { JSONArray().put(it.width.toDouble()).put(it.height.toDouble()) } ?: JSONObject.NULL)
            .put("pixel_array_size", pixel?.let { JSONArray().put(it.width).put(it.height) } ?: JSONObject.NULL)
            .put("active_array_size", active?.let { JSONArray().put(it.left).put(it.top).put(it.right).put(it.bottom) } ?: JSONObject.NULL)
            .put("lens_intrinsic_calibration", floatArray(c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)))
            .put("lens_distortion", floatArray(c.get(CameraCharacteristics.LENS_DISTORTION)))
            .put("lens_pose_rotation", floatArray(c.get(CameraCharacteristics.LENS_POSE_ROTATION)))
            .put("lens_pose_translation", floatArray(c.get(CameraCharacteristics.LENS_POSE_TRANSLATION)))
            .put("exposure_time_range_ns", exp?.let { JSONArray().put(it.lower).put(it.upper) } ?: JSONObject.NULL)
            .put("sensitivity_range", iso?.let { JSONArray().put(it.lower).put(it.upper) } ?: JSONObject.NULL)
            .put("max_frame_duration_ns", c.get(CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION) ?: JSONObject.NULL)
            .put("focus_distance_calibration", when (c.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION)) {
                CameraMetadata.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_CALIBRATED -> "CALIBRATED"
                CameraMetadata.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_APPROXIMATE -> "APPROXIMATE"
                else -> "UNCALIBRATED"
            })
            .put("min_focus_distance_diopters", c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)?.toDouble() ?: JSONObject.NULL)
            .put("video_stabilization_modes", intArray(c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)))
            .put("ois_modes", intArray(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)))
            .put("zoom_ratio_range", zoom?.let { JSONArray().put(it.lower.toDouble()).put(it.upper.toDouble()) } ?: JSONObject.NULL)
            .put("record_sizes", JSONArray(recordSizes(c).take(12).map { "${it.width}x${it.height}" }))
            .put("ae_fps_ranges", JSONArray((c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: emptyArray()).map { "${it.lower}-${it.upper}" }))
    }
}
