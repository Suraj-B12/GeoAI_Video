package com.geoai.capture.devicecheck

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.Sensor
import android.hardware.SensorManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.location.LocationRequest
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.PowerManager
import android.util.Size
import androidx.core.content.ContextCompat
import com.geoai.capture.camera.CameraCatalog
import com.geoai.capture.camera.VideoEncoder
import com.geoai.capture.io.SessionStore
import com.geoai.capture.service.SessionMetadata
import com.geoai.capture.settings.AppSettings
import com.geoai.capture.util.Clocks
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.Executor
import kotlin.coroutines.resume

/** Checks the pilot phone against the capture requirements and saves a JSON report. */
object DeviceCheck {

    enum class Level { PASS, WARN, FAIL, INFO }
    data class Item(val name: String, val level: Level, val detail: String)
    data class Report(val items: List<Item>, val file: File?)

    suspend fun run(
        context: Context,
        store: SessionStore,
        settings: AppSettings,
        gnssSeconds: Int,
        progress: (String) -> Unit,
    ): Report = withContext(Dispatchers.Default) {
        val items = mutableListOf<Item>()
        val json = JSONObject().put("created_wall_ms", Clocks.wallMs()).put("device", SessionMetadata.device(settings.deviceId))
        fun add(name: String, level: Level, detail: String) { items += Item(name, level, detail) }

        // ---- permissions & storage
        progress("Permissions and storage")
        val camOk = granted(context, Manifest.permission.CAMERA)
        val locOk = granted(context, Manifest.permission.ACCESS_FINE_LOCATION)
        val shared = Environment.isExternalStorageManager()
        add("Camera permission", if (camOk) Level.PASS else Level.FAIL, if (camOk) "granted" else "missing")
        add("Precise location permission", if (locOk) Level.PASS else Level.FAIL, if (locOk) "granted" else "missing")
        add("All files access", if (shared) Level.PASS else Level.WARN,
            if (shared) "sessions in ${store.root}" else "sessions fall back to app-specific storage (deleted on uninstall)")
        val freeGb = store.freeBytes() / 1e9
        add("Free storage", if (freeGb >= 50) Level.PASS else if (freeGb >= 10) Level.WARN else Level.FAIL, fmt("%.1f GB", freeGb))
        val pm = context.getSystemService(PowerManager::class.java)
        val ignoring = pm.isIgnoringBatteryOptimizations(context.packageName)
        add("Battery optimisation", if (ignoring) Level.PASS else Level.WARN,
            if (ignoring) "app is exempt" else "not exempt: also set the app to 'Unrestricted' and add it to 'Never sleeping apps'")
        json.put("storage", JSONObject().put("root", store.root.absolutePath).put("shared", shared).put("free_gb", freeGb))

        // ---- cameras
        progress("Cameras")
        val cm = context.getSystemService(CameraManager::class.java)
        val cams = JSONArray()
        for (id in cm.cameraIdList) runCatching { cams.put(CameraCatalog.describe(id, cm.getCameraCharacteristics(id))) }
        json.put("cameras", cams)
        val mainId = CameraCatalog.chooseMainBackCamera(cm)
        json.put("main_camera_id", mainId ?: JSONObject.NULL)
        var recordSize: Size? = null
        if (mainId == null) {
            add("Main back camera", Level.FAIL, "none found")
        } else {
            val c = cm.getCameraCharacteristics(mainId)
            val level = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            val rank = CameraCatalog.levelRank(level)
            add("Camera $mainId hardware level", if (rank >= 3) Level.PASS else if (rank == 2) Level.WARN else Level.FAIL, CameraCatalog.levelName(level))
            val manual = CameraCatalog.hasManualSensor(c)
            add("Manual shutter (MANUAL_SENSOR)", if (manual) Level.PASS else Level.FAIL,
                if (manual) "supported" else "NOT supported: shutter can't be forced, motion blur uncontrolled")
            val ts = CameraCatalog.timestampSourceName(c)
            add("Camera timestamp source", if (ts == "REALTIME") Level.PASS else Level.WARN,
                if (ts == "REALTIME") "REALTIME (same clock as GNSS/IMU)" else "$ts: frame/GNSS sync only approximate")
            val exp = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val minExp = exp?.lower
            add("Shortest exposure", if (minExp != null && minExp <= 125_000L) Level.PASS else Level.WARN,
                minExp?.let { fmt("1/%.0f s", 1e9 / it) } ?: "unknown")
            val iso = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            add("ISO range", Level.INFO, iso?.let { "${it.lower}–${it.upper}" } ?: "unknown")
            recordSize = CameraCatalog.pickRecordSize(c, 3840, 2160)
            val is4k = recordSize?.width == 3840 && recordSize.height == 2160
            add("4K encoder stream", if (is4k) Level.PASS else Level.WARN, recordSize?.let { "${it.width}x${it.height}" } ?: "no 16:9 record size")
            recordSize?.let { rs ->
                val minDur = CameraCatalog.minFrameDurationNs(c, rs)
                add("30 fps at ${rs.width}x${rs.height}", if (minDur != null && minDur <= 33_366_666L) Level.PASS else Level.FAIL,
                    minDur?.let { fmt("min frame duration %.1f ms", it / 1e6) } ?: "unknown")
            }
            val eis = c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES) ?: IntArray(0)
            add("EIS can be off", if (eis.isEmpty() || eis.contains(CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)) Level.PASS else Level.FAIL,
                "modes ${eis.joinToString()}")
            val ois = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION) ?: IntArray(0)
            add("OIS", Level.INFO, if (ois.contains(CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON)) "present (app default: off)" else "absent")
            val minFocus = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
            add("Manual focus lock", if (manual && minFocus > 0f) Level.PASS else Level.WARN,
                if (manual && minFocus > 0f) "LOCKED_MANUAL available" else "falls back to AF lock")
            add("Analysis stream", Level.INFO, CameraCatalog.pickAnalysisSize(c)?.let { "${it.width}x${it.height} YUV" } ?: "none")
            add("Sensor orientation", Level.INFO, "${c.get(CameraCharacteristics.SENSOR_ORIENTATION)}°")
        }

        // ---- encoders
        progress("Encoders")
        val encJson = JSONArray()
        for ((mime, label) in listOf(MediaFormat.MIMETYPE_VIDEO_HEVC to "HEVC", MediaFormat.MIMETYPE_VIDEO_AVC to "AVC")) {
            val info = VideoEncoder.pickEncoder(mime, 3840, 2160, 30)
            if (info == null) {
                add("$label 4K30 encoder", if (label == "HEVC") Level.WARN else Level.INFO, "not available")
            } else {
                val caps = info.getCapabilitiesForType(mime)
                val br = caps.videoCapabilities?.bitrateRange
                add("$label 4K30 encoder", Level.PASS,
                    "${info.name} hw=${info.isHardwareAccelerated} bitrate " +
                        (br?.let { "${it.lower / 1_000_000}–${it.upper / 1_000_000} Mbps" } ?: "unknown"))
                encJson.put(
                    JSONObject().put("mime", mime).put("name", info.name).put("hardware", info.isHardwareAccelerated)
                        .put("bitrate_min", br?.lower ?: JSONObject.NULL).put("bitrate_max", br?.upper ?: JSONObject.NULL)
                        .put("vbr", caps.encoderCapabilities?.isBitrateModeSupported(MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR) == true),
                )
            }
        }
        json.put("encoders", encJson)

        // ---- stream combination
        if (mainId != null && camOk && recordSize != null) {
            progress("Recording stream combination")
            val combo = streamComboSupported(context, mainId, recordSize)
            add("4K encoder + analysis stream together",
                when (combo["both"]) { "true" -> Level.PASS; "false" -> Level.WARN; else -> Level.INFO },
                "query: ${combo["both"]} (if not supported, recording runs encoder-only with frozen exposure)")
            json.put("stream_combination", JSONObject(combo as Map<*, *>))
        }

        // ---- GNSS
        progress("GNSS")
        val lm = context.getSystemService(LocationManager::class.java)
        val gnss = JSONObject()
            .put("hardware_model", lm.gnssHardwareModelName ?: "")
            .put("year_of_hardware", lm.gnssYearOfHardware)
            .put("capabilities", runCatching { lm.gnssCapabilities.toString() }.getOrDefault(""))
        add("GNSS chipset", Level.INFO, "${lm.gnssHardwareModelName ?: "?"} (${lm.gnssYearOfHardware})")
        if (gnssSeconds > 0 && locOk) {
            progress("GNSS live test (${gnssSeconds}s, go outdoors)")
            val live = gnssLiveTest(context, gnssSeconds)
            gnss.put("live_test", live)
            val rate = live.optDouble("fix_rate_hz", 0.0)
            add("GNSS fix rate", if (rate >= 0.9) Level.PASS else Level.WARN, fmt("%.2f Hz over %d fixes", rate, live.optInt("fixes")))
            add("GNSS median accuracy", Level.INFO, fmt("%.1f m (68%%)", live.optDouble("median_accuracy_m", Double.NaN)))
            add("Dual-frequency (L5/E5a)", if (live.optBoolean("l5_seen")) Level.PASS else Level.WARN,
                if (live.optBoolean("l5_seen")) "L5-band signals tracked" else "no L5-band signal seen (check outdoors)")
            add("NavIC (IRNSS)", Level.INFO, if (live.optBoolean("navic_seen")) "tracked" else "not seen")
            add("Mock locations", if (live.optBoolean("mock_seen")) Level.FAIL else Level.PASS,
                if (live.optBoolean("mock_seen")) "mock location detected: disable mock apps" else "none")
        }
        json.put("gnss", gnss)

        // ---- sensors & thermal
        progress("Sensors and thermal")
        val sm = context.getSystemService(SensorManager::class.java)
        val sensors = JSONArray()
        for (type in listOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_GRAVITY, Sensor.TYPE_GAME_ROTATION_VECTOR, Sensor.TYPE_MAGNETIC_FIELD)) {
            val s = sm.getDefaultSensor(type)
            if (s == null) {
                add("Sensor type $type", Level.WARN, "missing")
                continue
            }
            sensors.put(JSONObject().put("type", type).put("name", s.name).put("vendor", s.vendor).put("min_delay_us", s.minDelay))
        }
        json.put("sensors", sensors)
        val headroom = runCatching { pm.getThermalHeadroom(10) }.getOrNull()
        add("Thermal API", if (headroom != null && !headroom.isNaN()) Level.PASS else Level.WARN,
            "status ${pm.currentThermalStatus}, headroom ${headroom ?: "n/a"}")
        json.put("thermal", JSONObject().put("status", pm.currentThermalStatus).put("headroom", headroom?.toDouble()?.takeIf { !it.isNaN() } ?: JSONObject.NULL))

        json.put("items", JSONArray(items.map { JSONObject().put("name", it.name).put("level", it.level.name).put("detail", it.detail) }))
        val file = runCatching {
            File(store.reportsDir, "device_check_${Clocks.compactUtc(Clocks.wallMs())}.json").also { it.writeText(json.toString(2)) }
        }.getOrNull()
        progress("Done")
        Report(items, file)
    }

    private fun granted(c: Context, p: String) = ContextCompat.checkSelfPermission(c, p) == PackageManager.PERMISSION_GRANTED
    private fun fmt(f: String, vararg a: Any?) = String.format(Locale.ROOT, f, *a)

    @SuppressLint("MissingPermission")
    private suspend fun streamComboSupported(context: Context, cameraId: String, recordSize: Size): Map<String, String> {
        val thread = HandlerThread("geoai-check-camera").apply { start() }
        val handler = Handler(thread.looper)
        val executor = Executor { handler.post(it) }
        val result = mutableMapOf<String, String>()
        var codec: MediaCodec? = null
        var reader: ImageReader? = null
        var device: CameraDevice? = null
        try {
            val cm = context.getSystemService(CameraManager::class.java)
            val chars = cm.getCameraCharacteristics(cameraId)
            val analysis = CameraCatalog.pickAnalysisSize(chars) ?: Size(640, 360)
            val mime = if (VideoEncoder.pickEncoder(MediaFormat.MIMETYPE_VIDEO_HEVC, recordSize.width, recordSize.height, 30) != null)
                MediaFormat.MIMETYPE_VIDEO_HEVC else MediaFormat.MIMETYPE_VIDEO_AVC
            val info = VideoEncoder.pickEncoder(mime, recordSize.width, recordSize.height, 30) ?: return mapOf("both" to "no_encoder")
            codec = MediaCodec.createByCodecName(info.name)
            val fmt = MediaFormat.createVideoFormat(mime, recordSize.width, recordSize.height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, 20_000_000)
                setInteger(MediaFormat.KEY_FRAME_RATE, 30)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }
            codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            val encSurface = codec.createInputSurface()
            reader = ImageReader.newInstance(analysis.width, analysis.height, ImageFormat.YUV_420_888, 2)
            device = withTimeoutOrNull(5_000) {
                suspendCancellableCoroutine<CameraDevice?> { cont ->
                    cm.openCamera(cameraId, object : CameraDevice.StateCallback() {
                        override fun onOpened(d: CameraDevice) { if (cont.isActive) cont.resume(d) else d.close() }
                        override fun onDisconnected(d: CameraDevice) { d.close(); if (cont.isActive) cont.resume(null) }
                        override fun onError(d: CameraDevice, error: Int) { d.close(); if (cont.isActive) cont.resume(null) }
                    }, handler)
                }
            }
            val dev = device ?: return mapOf("both" to "camera_unavailable")
            fun query(outputs: List<OutputConfiguration>): String = try {
                val noop = object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) = Unit
                    override fun onConfigureFailed(session: CameraCaptureSession) = Unit
                }
                @Suppress("DEPRECATION")
                dev.isSessionConfigurationSupported(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, executor, noop)).toString()
            } catch (e: UnsupportedOperationException) {
                "unknown"
            } catch (e: Exception) {
                "error: ${e.javaClass.simpleName}"
            }
            fun enc() = OutputConfiguration(encSurface).also { if (Build.VERSION.SDK_INT >= 33) it.setTimestampBase(OutputConfiguration.TIMESTAMP_BASE_SENSOR) }
            result["both"] = query(listOf(enc(), OutputConfiguration(reader.surface)))
            result["encoder_only"] = query(listOf(enc()))
            result["record_size"] = "${recordSize.width}x${recordSize.height}"
            result["analysis_size"] = "${analysis.width}x${analysis.height}"
            result["mime"] = mime
            encSurface.release()
        } catch (e: Exception) {
            result["both"] = "error: $e"
        } finally {
            runCatching { device?.close() }
            runCatching { codec?.release() }
            runCatching { reader?.close() }
            thread.quitSafely()
        }
        return result
    }

    @SuppressLint("MissingPermission")
    private suspend fun gnssLiveTest(context: Context, seconds: Int): JSONObject {
        val lm = context.getSystemService(LocationManager::class.java)
        val thread = HandlerThread("geoai-check-gnss").apply { start() }
        val executor = Executor { Handler(thread.looper).post(it) }
        val fixes = mutableListOf<Location>()
        val constellations = mutableSetOf<Int>()
        var l5 = false
        var navic = false
        val listener = LocationListener { l -> synchronized(fixes) { fixes += l } }
        val statusCb = object : GnssStatus.Callback() {
            override fun onSatelliteStatusChanged(s: GnssStatus) {
                for (i in 0 until s.satelliteCount) {
                    constellations += s.getConstellationType(i)
                    if (s.getConstellationType(i) == GnssStatus.CONSTELLATION_IRNSS) navic = true
                    if (s.hasCarrierFrequencyHz(i) && s.getCarrierFrequencyHz(i) in 1.16e9..1.19e9) l5 = true
                }
            }
        }
        try {
            lm.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                LocationRequest.Builder(0L).setQuality(LocationRequest.QUALITY_HIGH_ACCURACY).setMinUpdateIntervalMillis(0L).build(),
                executor, listener,
            )
            lm.registerGnssStatusCallback(executor, statusCb)
            delay(seconds * 1000L)
        } finally {
            runCatching { lm.removeUpdates(listener) }
            runCatching { lm.unregisterGnssStatusCallback(statusCb) }
            thread.quitSafely()
        }
        val list = synchronized(fixes) { fixes.toList() }.sortedBy { it.elapsedRealtimeNanos }
        val spanS = if (list.size >= 2) (list.last().elapsedRealtimeNanos - list.first().elapsedRealtimeNanos) / 1e9 else 0.0
        val acc = list.filter { it.hasAccuracy() }.map { it.accuracy.toDouble() }.sorted()
        val gaps = list.zipWithNext { a, b -> (b.elapsedRealtimeNanos - a.elapsedRealtimeNanos) / 1e9 }
        return JSONObject()
            .put("seconds", seconds)
            .put("fixes", list.size)
            .put("fix_rate_hz", if (spanS > 0) (list.size - 1) / spanS else 0.0)
            .put("max_gap_s", gaps.maxOrNull() ?: JSONObject.NULL)
            .put("median_accuracy_m", if (acc.isNotEmpty()) acc[acc.size / 2] else JSONObject.NULL)
            .put("mock_seen", list.any { it.isMock })
            .put("constellations", JSONArray(constellations.sorted()))
            .put("l5_seen", l5)
            .put("navic_seen", navic)
    }
}
