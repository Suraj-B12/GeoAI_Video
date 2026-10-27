package com.geoai.capture.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaFormat
import android.os.Build
import android.os.Handler
import android.util.Size
import android.view.Surface
import com.geoai.capture.settings.AppSettings
import com.geoai.capture.settings.Codec
import com.geoai.capture.util.Clocks
import com.geoai.capture.util.Diag
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.android.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs
import kotlin.math.max

/**
 * Owns the camera for one survey session.
 *
 * Setup: one YUV analysis stream (small) for the operator preview, auto-exposure metering and
 * focusing. Lock: focus is set for the ROAD from the mount geometry ([FocusPlanner]), checked by a short
 * contrast sweep, and frozen (autofocus restricted to the road is the fallback); exposure switches to
 * shutter priority with our own ISO loop, metered on the road rows only. Recording: encoder surface (4K) plus the
 * same analysis stream. If the device can't run both, it records encoder-only with exposure frozen.
 *
 * Every camera-state change runs on the camera thread (via [dispatcher]); analysis runs on its own
 * thread and only posts setting changes back to the camera thread.
 */
class CameraRecorder(
    context: Context,
    val cameraId: String,
    private val settings: AppSettings,
    private val cameraHandler: Handler,
    private val encoderHandler: Handler,
    private val analysisHandler: Handler,
    private val segmentWriter: SegmentWriter,
    private val frameMatcher: FrameMatcher,
    private val listener: Listener,
) {
    interface Listener {
        fun onPreview(bitmap: Bitmap)
        fun onAnalysis(result: AnalysisProcessor.Result, frameExposureNs: Long?, frameIso: Int?, decision: ExposureSetting?)
        fun onState(state: CameraState)
        /** The camera or encoder failed; recording (if any) has been interrupted. */
        fun onCameraError(message: String)
        fun onEvent(type: String, value: JSONObject)
        /** Latest GNSS speed if fresh (≤ 2 s old), else null. */
        fun speedMps(): Double?
    }

    data class CameraState(
        val exposureMode: String,
        val focusMode: String,
        val focusDiopters: Float?,
        val setting: ExposureSetting?,
        val recording: Boolean,
        val recordWidth: Int,
        val recordHeight: Int,
        val streams: String,
        val captureFailures: Long,
        val buffersLost: Long,
        /** One line for the operator: how focus was set and for which road distances. */
        val focusNote: String = "",
    )

    data class LockState(
        val exposureMode: String,
        val focusMode: String,
        val focusDiopters: Float?,
        val setting: ExposureSetting?,
        val road: RoadRoi? = null,
        val focusMethod: String = "none",
    )

    /** The road part of the image: [rect] in active-array pixels (AF/AE regions), rows as stream fractions. */
    data class RoadRoi(val rect: Rect, val topFrac: Double, val bottomFrac: Double, val leftFrac: Double, val rightFrac: Double)

    /** What the focus plan needs from outside the camera: gravity (device axes) and the mount height. */
    class FocusInputs(val gravity: FloatArray?, val heightM: Double, val heightSource: String)

    val dispatcher = cameraHandler.asCoroutineDispatcher()
    private val manager = context.getSystemService(CameraManager::class.java)
    val chars: CameraCharacteristics = manager.getCameraCharacteristics(cameraId)
    val manualSensor = CameraCatalog.hasManualSensor(chars)
    val sensorOrientation: Int = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
    private val minFocus = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
    private val afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: IntArray(0)
    private val oisModes = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION) ?: IntArray(0)
    private val hasZoomRatio = chars.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) != null
    private val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
    private val maxAeRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0
    private val focusCalibration = chars.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION)
    private val focusCalibrationName = when (focusCalibration) {
        CameraMetadata.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_CALIBRATED -> "CALIBRATED"
        CameraMetadata.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_APPROXIMATE -> "APPROXIMATE"
        else -> "UNCALIBRATED"
    }
    private val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
    /** Lens model for the focus plan and the live road view; null when the characteristics lack a field. */
    private val lens: FocusPlanner.Lens? by lazy { lensModel() }
    private val sensorUpDir = FocusPlanner.sensorUp(sensorOrientation)
    private val fpsRange = CameraCatalog.pickFpsRange(chars, settings.fps)
    private val frameDurationNs = 1_000_000_000L / settings.fps
    val analysisSize: Size = CameraCatalog.pickAnalysisSize(chars) ?: Size(640, 360)

    private val limits = run {
        val exp = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val iso = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        SensorLimits(exp?.lower ?: 100_000L, exp?.upper ?: 33_000_000L, iso?.lower ?: 50, iso?.upper ?: 3200)
    }
    private val exposureLock = Any()
    private val controller = ExposureController(
        ExposureConfig(
            blurBudgetMm = settings.blurBudgetMm,
            maxExposureNs = 1_000_000_000L / settings.maxExposureDenominator,
            lowLightMaxExposureNs = 1_000_000_000L / settings.lowLightExposureDenominator,
            maxIso = settings.maxIso,
            targetLuma = settings.targetLuma.toDouble(),
            unknownSpeedMps = settings.unknownSpeedKmh / 3.6,
        ),
        limits,
    )

    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var analysisReader: ImageReader? = null
    private var encoder: VideoEncoder? = null
    private var targets: List<Surface> = emptyList()
    private var encoderSurface: Surface? = null

    @Volatile var recording = false
        private set
    /** True from the first recording session until the camera closes, so trailing results still match. */
    @Volatile private var feedMatcher = false
    /** Consecutive analysed frames that were black (lens covered, camera muted by privacy toggle). */
    private var blackStreak = 0
    @Volatile var imageBlack = false
        private set
    /** AUTO_METERING (setup) | MANUAL_SHUTTER_AUTO_ISO | AUTO_FALLBACK (no MANUAL_SENSOR). */
    @Volatile var exposureMode = "AUTO_METERING"
        private set
    /** CONTINUOUS_VIDEO | AF_SCANNING | LOCKED_MANUAL | LOCKED_AF | FIXED. */
    @Volatile var focusMode = "CONTINUOUS_VIDEO"
        private set
    @Volatile var focusDiopters: Float? = null
        private set
    /** geometry_sweep_confirmed | geometry_sweep_flat | geometry | sweep_override | af_road_region | af_default_region | fixed. */
    @Volatile var focusMethod = "none"
        private set
    @Volatile private var focusNote = ""
    @Volatile private var focusReport: JSONObject? = null
    /** AF/AE metering region; null = the default lower-centre region. */
    @Volatile private var road: RoadRoi? = null
    /** While a focus sweep runs: (lens dioptres of the analysed frame, sharpness) of stationary-lens frames. */
    @Volatile private var sweepActive = false
    private val sweepSamples = ConcurrentLinkedQueue<Pair<Float, Double>>()
    @Volatile private var setting: ExposureSetting? = null
    @Volatile var streams = "analysis"
        private set
    @Volatile var encoderTimestampBase = "DEFAULT"
        private set
    @Volatile var recordSize: Size? = null
        private set
    @Volatile var activeEncoder: JSONObject? = null
        private set
    @Volatile private var latest: ResultMeta? = null
    private val recentMeta = object : LinkedHashMap<Long, ResultMeta>(256) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Long, ResultMeta>?) = size > 150
    }
    @Volatile var captureFailures = 0L
        private set
    @Volatile var buffersLost = 0L
        private set
    @Volatile private var readyDeferred: CompletableDeferred<Unit>? = null
    @Volatile private var eosDeferred: CompletableDeferred<Unit>? = null
    @Volatile private var closed = false

    private val processor = AnalysisProcessor()
    private var lastAnalysisNs = 0L
    private var lastPreviewNs = 0L
    private val cameraExecutor = Executor { cameraHandler.post(it) }

    // ---------------------------------------------------------------- lifecycle

    @SuppressLint("MissingPermission")
    suspend fun open() = withContext(dispatcher) {
        val reader = ImageReader.newInstance(analysisSize.width, analysisSize.height, ImageFormat.YUV_420_888, 4)
        reader.setOnImageAvailableListener(analysisListener, analysisHandler)
        analysisReader = reader
        camera = suspendCancellableCoroutine { cont ->
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    if (cont.isActive) cont.resume(device) else device.close()
                }

                override fun onDisconnected(device: CameraDevice) {
                    device.close()
                    if (cont.isActive) cont.resumeWithException(IllegalStateException("camera disconnected"))
                    else onDeviceLost("camera disconnected (another app, the system, or overheating)")
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                    val msg = "camera device error $error"
                    if (cont.isActive) cont.resumeWithException(IllegalStateException(msg)) else onDeviceLost(msg)
                }
            }, cameraHandler)
        }
        targets = listOf(reader.surface)
        session = createSession(listOf(OutputConfiguration(reader.surface)), targets)
        streams = "analysis"
        setRepeating()
        publishState()
    }

    /** Closes everything. Safe to call twice. */
    suspend fun close() = withContext(dispatcher) {
        if (encoder != null) stopRecording("stop")
        closed = true
        feedMatcher = false
        Diag.i("recorder.close: session")
        runCatching { session?.close() }
        session = null
        Diag.i("recorder.close: camera")
        runCatching { camera?.close() }
        camera = null
        val reader = analysisReader
        analysisReader = null
        Diag.i("recorder.close: analysis reader")
        // Close on the analysis thread so no Image is being read while its buffers are freed.
        if (reader != null) runOn(analysisHandler) { runCatching { reader.close() } }
        Diag.i("recorder.close: done")
    }

    // ---------------------------------------------------------------- focus & exposure lock

    /**
     * Sets focus for the road and meters exposure on the road, then locks both. The vehicle should be still.
     *
     * Calibrated (or approximately calibrated) lens + manual focus + usable geometry: focus = the geometric
     * target, checked by a sweep of +-0.6 dpt ([FocusPlanner.decide]). Otherwise: autofocus restricted to the
     * road region, then held. AF and AE never meter the sky, buildings or roadside when a road region exists.
     */
    suspend fun lockFocusAndExposure(inputs: FocusInputs): LockState = withContext(dispatcher) {
        exposureMode = "AUTO_METERING"
        val plan = planFocus(inputs)
        setRoad(plan)
        val metric = focusCalibration == CameraMetadata.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_CALIBRATED ||
            focusCalibration == CameraMetadata.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_APPROXIMATE
        val report = JSONObject()
            .put("lens_calibration", focusCalibrationName)
            .put("height_m", inputs.heightM)
            .put("height_source", inputs.heightSource)
            .put("plan", plan?.let { planJson(it) } ?: JSONObject.NULL)
        if (plan != null && metric && manualSensor && minFocus > 0f) {
            focusMode = "SWEEP"
            val scores = ArrayList<Pair<Double, Double>>()
            val sweep = JSONArray()
            sweepActive = true
            try {
                for (d in FocusPlanner.sweep(plan.targetDiopters, minFocus.toDouble())) {
                    focusDiopters = d.toFloat()
                    setRepeating()
                    val (score, n) = sweepScore(d)
                    scores += d to score
                    sweep.put(
                        JSONObject().put("diopters", d).put("sharpness", if (score.isFinite()) score else JSONObject.NULL).put("frames", n),
                    )
                }
            } finally {
                sweepActive = false
                sweepSamples.clear()
            }
            val decision = FocusPlanner.decide(plan.targetDiopters, scores)
            focusDiopters = decision.diopters.toFloat()
            focusMode = "LOCKED_MANUAL"
            focusMethod = decision.method
            setRepeating()
            waitFor(1000) { lensAt(decision.diopters) }
            report.put("sweep", sweep).put("chosen_diopters", decision.diopters)
        } else {
            focusMode = if (afModes.contains(CameraMetadata.CONTROL_AF_MODE_AUTO)) "AF_SCANNING" else "FIXED"
            setRepeating()
            delay(300)
            if (focusMode == "AF_SCANNING") {
                val trigger = buildRequest(targets).apply {
                    set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
                }.build()
                runCatching { session?.capture(trigger, captureCallback, cameraHandler) }
                waitFor(4000) {
                    val st = latest?.afState
                    st == CameraMetadata.CONTROL_AF_STATE_FOCUSED_LOCKED || st == CameraMetadata.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED
                }
                report.put("af_state", latest?.afState ?: JSONObject.NULL)
            }
            val d = latest?.focusDiopters
            focusMode = when {
                focusMode == "FIXED" -> "FIXED"
                manualSensor && minFocus > 0f && d != null -> "LOCKED_MANUAL"
                else -> "LOCKED_AF" // AF_MODE_AUTO keeps the lens where the trigger left it.
            }
            focusDiopters = d
            focusMethod = when {
                focusMode == "FIXED" -> "fixed"
                road != null -> "af_road_region"
                else -> "af_default_region"
            }
            report.put(
                "why_not_geometry",
                when {
                    plan == null -> "no usable geometry (gravity, height or view)"
                    !metric -> "lens focus distance is $focusCalibrationName"
                    else -> "no manual focus control"
                },
            )
        }
        waitFor(2500) {
            val st = latest?.aeState
            st == CameraMetadata.CONTROL_AE_STATE_CONVERGED || st == CameraMetadata.CONTROL_AE_STATE_FLASH_REQUIRED ||
                st == CameraMetadata.CONTROL_AE_STATE_LOCKED
        }
        val meta = latest
        if (manualSensor && meta?.exposureNs != null && meta.iso != null) {
            synchronized(exposureLock) { setting = controller.initFromMetering(meta.exposureNs, meta.iso, listener.speedMps()) }
            exposureMode = "MANUAL_SHUTTER_AUTO_ISO"
        } else {
            exposureMode = "AUTO_FALLBACK"
        }
        report.put("method", focusMethod).put("focus_mode", focusMode)
            .put("focus_diopters", focusDiopters?.toDouble() ?: JSONObject.NULL)
        focusReport = report
        focusNote = noteFor(plan)
        setRepeating()
        listener.onEvent("focus", report)
        listener.onEvent("lock", lockJson())
        publishState()
        LockState(exposureMode, focusMode, focusDiopters, setting, road, focusMethod)
    }

    private fun planFocus(inputs: FocusInputs): FocusPlanner.Plan? {
        val l = lens ?: return null
        val up = sensorUpDir ?: return null
        return FocusPlanner.plan(l, inputs.gravity, up, inputs.heightM, settings.resolution.width, settings.resolution.height)
    }

    /**
     * Tilt and the road the recorded frame shows (near..far ahead, whether a segment fits) for the current gravity:
     * the live mount check. Pure computation on immutable camera data, so any thread may call it.
     */
    fun roadView(inputs: FocusInputs): FocusPlanner.RoadView? {
        val l = lens ?: return null
        val up = sensorUpDir ?: return null
        return FocusPlanner.roadView(l, inputs.gravity, up, inputs.heightM, settings.resolution.width, settings.resolution.height)
    }

    /** Lens model from the camera characteristics; null when the needed fields are missing. */
    private fun lensModel(): FocusPlanner.Lens? {
        val a = activeArray ?: return null
        val focal = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
        val aperture = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.firstOrNull() ?: return null
        val phys = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
        val pix = chars.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE) ?: return null
        // Intrinsics [fx, fy, cx, cy, s] in active-array pixels; without them, the pinhole model from the sensor size.
        val k = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)?.takeIf { it.size >= 4 && it[1] > 0f }
        val fy = k?.get(1)?.toDouble() ?: (focal / phys.height * pix.height.toDouble())
        val cx = k?.get(2)?.toDouble() ?: (a.width() / 2.0)
        val cy = k?.get(3)?.toDouble() ?: (a.height() / 2.0)
        val pixelUm = phys.width * 1000.0 / pix.width
        return FocusPlanner.Lens(
            activeWidth = a.width(), activeHeight = a.height(), fyPx = fy, cxPx = cx, cyPx = cy,
            focalMm = focal.toDouble(), aperture = aperture.toDouble(),
            outputPixelUm = pixelUm * a.width() / settings.resolution.width,
            minFocusDiopters = minFocus.toDouble(),
        )
    }

    private fun setRoad(plan: FocusPlanner.Plan?) {
        val a = activeArray
        road = if (plan != null && a != null) {
            RoadRoi(
                Rect(a.left + plan.roiLeft, a.top + plan.roiTop, a.left + plan.roiRight, a.top + plan.roiBottom),
                plan.roiTopFrac, plan.roiBottomFrac, FocusPlanner.ROI_LEFT, FocusPlanner.ROI_RIGHT,
            )
        } else {
            null
        }
        applyRoadToAnalysis()
    }

    private fun applyRoadToAnalysis() {
        val r = road
        if (r != null) processor.setRoi(r.topFrac, r.bottomFrac, r.leftFrac, r.rightFrac) else processor.resetRoi()
    }

    /** Mean sharpness of analysed frames taken with the lens settled at [d], and how many there were. */
    private suspend fun sweepScore(d: Double): Pair<Double, Int> {
        waitFor(1000) { lensAt(d) }
        sweepSamples.clear()
        val tol = focusTolerance(d)
        val got = ArrayList<Double>()
        withTimeoutOrNull(1500) {
            while (got.size < SWEEP_FRAMES) {
                val sample = sweepSamples.poll()
                if (sample == null) {
                    delay(20)
                } else if (abs(sample.first - d) <= tol) {
                    got += sample.second
                }
            }
        }
        return (if (got.isEmpty()) Double.NaN else got.average()) to got.size
    }

    private fun focusTolerance(d: Double) = max(0.05, 0.05 * d)

    private fun lensAt(d: Double): Boolean {
        val m = latest ?: return false
        val fd = m.focusDiopters ?: return false
        val still = m.lensState == null || m.lensState == CameraMetadata.LENS_STATE_STATIONARY
        return still && abs(fd - d) <= focusTolerance(d)
    }

    private fun planJson(p: FocusPlanner.Plan) = JSONObject()
        .put("camera_depression_deg", p.view.tiltDeg)
        .put("road_ahead_near_m", p.nearRoadM)
        .put("road_ahead_far_m", p.farRoadM)
        .put("road_depth_near_m", p.nearDepthM)
        .put("road_depth_far_m", p.farDepthM)
        .put("target_diopters", p.targetDiopters)
        .put("predicted_edge_blur_px", p.edgeBlurPx)
        .put("road_rows_active_array", JSONArray().put(p.roiTop).put(p.roiBottom))
        .put("road_rows_frac", JSONArray().put(p.roiTopFrac).put(p.roiBottomFrac))
        .put("segment_reach_m", FocusPlanner.SEGMENT_REACH_M)
        .put("segment_length_m", FocusPlanner.SEGMENT_LENGTH_M)
        .put("segment_window_m", FocusPlanner.SEGMENT_WINDOW_M)
        .put("view_far_m", p.view.farM ?: JSONObject.NULL)
        .put("segment_fits", p.view.segmentFits)

    private fun noteFor(plan: FocusPlanner.Plan?): String {
        val method = focusMethod.replace('_', ' ')
        return if (plan != null) String.format(Locale.ROOT, "%s; road %.1f-%.1f m ahead", method, plan.nearRoadM, plan.farRoadM) else method
    }

    /** Re-applies a previous lock (camera reopened after an error) without re-running AF. */
    suspend fun applyLock(state: LockState) = withContext(dispatcher) {
        focusMode = state.focusMode
        focusDiopters = state.focusDiopters
        focusMethod = state.focusMethod
        road = state.road
        applyRoadToAnalysis()
        exposureMode = state.exposureMode
        val s = state.setting
        if (exposureMode == "MANUAL_SHUTTER_AUTO_ISO" && s != null) {
            synchronized(exposureLock) { setting = controller.initFromMetering(s.exposureNs, s.iso, listener.speedMps()) }
        }
        setRepeating()
        listener.onEvent("lock_restored", lockJson())
        publishState()
    }

    fun lockState() = LockState(exposureMode, focusMode, focusDiopters, setting, road, focusMethod)

    private fun lockJson() = JSONObject()
        .put("exposure_mode", exposureMode)
        .put("focus_mode", focusMode)
        .put("focus_diopters", focusDiopters?.toDouble() ?: JSONObject.NULL)
        .put("focus_method", focusMethod)
        .put("exposure_ns", setting?.exposureNs ?: JSONObject.NULL)
        .put("iso", setting?.iso ?: JSONObject.NULL)

    // ---------------------------------------------------------------- recording

    /** Starts the encoder and a recording session at [size]. */
    suspend fun startRecording(size: Size) = withContext(dispatcher) {
        check(camera != null) { "camera not open" }
        runCatching { session?.close() }
        session = null

        val bitrate = scaledBitrate(size)
        var enc = VideoEncoder(settings.codec.mime, size.width, size.height, settings.fps, bitrate, 1, encoderHandler, encoderListener)
        try {
            enc.configure()
        } catch (e: Exception) {
            if (settings.codec != Codec.HEVC) throw e
            listener.onEvent("encoder_fallback", Diag.errorJson(e).put("from", "HEVC").put("to", "AVC"))
            enc = VideoEncoder(Codec.AVC.mime, size.width, size.height, settings.fps, bitrate, 1, encoderHandler, encoderListener)
            enc.configure()
        }
        segmentWriter.streamWidth = size.width
        segmentWriter.streamHeight = size.height
        segmentWriter.streamMime = enc.mime
        segmentWriter.streamBitrate = enc.appliedBitrate
        encoder = enc
        encoderSurface = enc.inputSurface
        recordSize = size
        activeEncoder = enc.describe()

        val reader = analysisReader ?: error("analysis reader closed")
        val both = listOf(encoderOutput(enc.inputSurface), OutputConfiguration(reader.surface))
        var newSession: CameraCaptureSession? = null
        val supported = isSupported(both)
        if (supported != false) {
            targets = listOf(enc.inputSurface, reader.surface)
            newSession = runCatching { createSession(both, targets) }.getOrNull()
            streams = "encoder+analysis"
        }
        if (newSession == null) {
            listener.onEvent("analysis_stream_unavailable", JSONObject().put("query_result", supported?.toString() ?: "unknown"))
            targets = listOf(enc.inputSurface)
            newSession = createSession(listOf(encoderOutput(enc.inputSurface)), targets)
            streams = "encoder_only"
        }
        session = newSession
        enc.start()
        feedMatcher = true
        recording = true
        setRepeating()
        listener.onEvent(
            "recording_started",
            JSONObject().put("width", size.width).put("height", size.height).put("streams", streams)
                .put("encoder", activeEncoder).put("encoder_timestamp_base", encoderTimestampBase),
        )
        publishState()
    }

    /** Stops the repeating request, drains the encoder to end-of-stream and closes the segment. */
    suspend fun stopRecording(reason: String) = withContext(dispatcher) {
        Diag.i("recorder.stopRecording($reason): recording=$recording encoder=${encoder != null}")
        val s = session
        if (s != null && recording) {
            val ready = CompletableDeferred<Unit>()
            readyDeferred = ready
            runCatching { s.stopRepeating() }
            withTimeoutOrNull(1500) { ready.await() }
        }
        recording = false
        drainEncoder(reason)
        runCatching { s?.close() }
        session = null
        publishState()
    }

    /** After a device error: the session is gone, but the encoder still holds frames to flush. */
    suspend fun abandonAfterError() = withContext(dispatcher) {
        recording = false
        drainEncoder("error")
        session = null
        publishState()
    }

    private suspend fun drainEncoder(reason: String) {
        val enc = encoder ?: return
        val eos = CompletableDeferred<Unit>()
        eosDeferred = eos
        runCatching { enc.signalEndOfStream() }
        val drained = withTimeoutOrNull(3000) { eos.await() } != null
        if (!drained) listener.onEvent("encoder_eos_timeout", JSONObject().put("reason", reason))
        // Close the segment on the encoder thread, the only thread that touches the muxer.
        runOn(encoderHandler) { segmentWriter.close(reason) }
        enc.release()
        encoder = null
        encoderSurface = null
        activeEncoder = null
    }

    private fun scaledBitrate(size: Size): Int {
        val full = settings.bitrateMbps * 1_000_000L
        val scaled = full * size.width * size.height / (3840L * 2160L)
        return max(12_000_000L, scaled).toInt()
    }

    private fun encoderOutput(surface: Surface) = OutputConfiguration(surface).also {
        if (Build.VERSION.SDK_INT >= 33) {
            it.setTimestampBase(OutputConfiguration.TIMESTAMP_BASE_SENSOR)
            encoderTimestampBase = "SENSOR"
        } else {
            encoderTimestampBase = "DEFAULT"
        }
    }

    // ---------------------------------------------------------------- requests & sessions

    private suspend fun createSession(outputs: List<OutputConfiguration>, surfaces: List<Surface>): CameraCaptureSession {
        val cam = camera ?: error("camera closed")
        return suspendCancellableCoroutine { cont ->
            val cfg = SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, cameraExecutor,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) {
                        if (cont.isActive) cont.resume(s)
                    }

                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        if (cont.isActive) cont.resumeWithException(IllegalStateException("capture session configuration failed"))
                    }

                    override fun onReady(s: CameraCaptureSession) {
                        readyDeferred?.complete(Unit)
                    }
                })
            cfg.sessionParameters = buildRequest(surfaces).build()
            try {
                cam.createCaptureSession(cfg)
            } catch (e: Exception) {
                if (cont.isActive) cont.resumeWithException(e)
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun isSupported(outputs: List<OutputConfiguration>): Boolean? {
        val cam = camera ?: return null
        return try {
            val noop = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) = Unit
                override fun onConfigureFailed(session: CameraCaptureSession) = Unit
            }
            cam.isSessionConfigurationSupported(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, cameraExecutor, noop))
        } catch (e: UnsupportedOperationException) {
            null
        } catch (e: IllegalArgumentException) {
            false
        } catch (e: Exception) {
            null
        }
    }

    private fun buildRequest(surfaces: List<Surface>): CaptureRequest.Builder {
        val b = camera!!.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        surfaces.forEach { b.addTarget(it) }
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.CONTROL_SCENE_MODE, CameraMetadata.CONTROL_SCENE_MODE_DISABLED)
        b.set(CaptureRequest.CONTROL_EFFECT_MODE, CameraMetadata.CONTROL_EFFECT_MODE_OFF)
        // EIS warps and crops each frame differently: it would break the camera calibration.
        b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        if (oisModes.isNotEmpty()) {
            val on = settings.oisEnabled && oisModes.contains(CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON)
            b.set(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                if (on) CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON else CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
        }
        if (hasZoomRatio) b.set(CaptureRequest.CONTROL_ZOOM_RATIO, 1.0f)
        b.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
        b.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
        fpsRange?.let { b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
        val st = setting
        if (exposureMode == "MANUAL_SHUTTER_AUTO_ISO" && st != null) {
            b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
            b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, st.exposureNs)
            b.set(CaptureRequest.SENSOR_SENSITIVITY, st.iso)
            b.set(CaptureRequest.SENSOR_FRAME_DURATION, frameDurationNs)
        } else {
            b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
        }
        when (focusMode) {
            "LOCKED_MANUAL", "SWEEP" -> {
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                focusDiopters?.let { b.set(CaptureRequest.LENS_FOCUS_DISTANCE, it) }
            }
            "LOCKED_AF", "AF_SCANNING" -> b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            "FIXED" -> b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
            else -> b.set(
                CaptureRequest.CONTROL_AF_MODE,
                if (afModes.contains(CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)) CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                else CameraMetadata.CONTROL_AF_MODE_OFF,
            )
        }
        val a = activeArray
        val r = road
        if (r != null) {
            // The road rows from the focus plan: AF and AE never meter vehicles, buildings or sky.
            val region = arrayOf(MeteringRectangle(r.rect, MeteringRectangle.METERING_WEIGHT_MAX))
            if (maxAfRegions > 0) b.set(CaptureRequest.CONTROL_AF_REGIONS, region)
            if (maxAeRegions > 0) b.set(CaptureRequest.CONTROL_AE_REGIONS, region)
        } else if (maxAfRegions > 0 && a != null) {
            // No plan: lower-centre of the sensor frame, the road just ahead when the phone is mounted upright.
            b.set(
                CaptureRequest.CONTROL_AF_REGIONS,
                arrayOf(
                    MeteringRectangle(
                        (a.width() * 0.30).toInt(), (a.height() * 0.50).toInt(),
                        (a.width() * 0.40).toInt(), (a.height() * 0.30).toInt(),
                        MeteringRectangle.METERING_WEIGHT_MAX,
                    ),
                ),
            )
        }
        return b
    }

    private fun setRepeating() {
        val s = session ?: return
        try {
            s.setRepeatingRequest(buildRequest(targets).build(), captureCallback, cameraHandler)
        } catch (e: Exception) {
            listener.onEvent("repeating_request_error", JSONObject().put("error", e.toString()))
        }
    }

    private fun applySetting(next: ExposureSetting) {
        if (closed || exposureMode != "MANUAL_SHUTTER_AUTO_ISO") return
        setting = next
        setRepeating()
        publishState()
    }

    private suspend fun waitFor(timeoutMs: Long, cond: () -> Boolean) {
        withTimeoutOrNull(timeoutMs) { while (!cond()) delay(30) }
    }

    private suspend fun runOn(handler: Handler, block: () -> Unit) = suspendCancellableCoroutine { cont ->
        handler.post {
            try {
                block()
                cont.resume(Unit)
            } catch (e: Throwable) {
                cont.resumeWithException(e)
            }
        }
    }

    private fun onDeviceLost(message: String) {
        camera = null
        session = null
        val wasRecording = recording
        recording = false
        listener.onEvent("camera_lost", JSONObject().put("message", message).put("was_recording", wasRecording))
        publishState()
        listener.onCameraError(message)
    }

    private fun publishState() {
        val size = recordSize
        listener.onState(
            CameraState(exposureMode, focusMode, focusDiopters, setting, recording, size?.width ?: 0, size?.height ?: 0,
                streams, captureFailures, buffersLost, focusNote),
        )
    }

    /** Capture settings for session.json (what was actually configured). */
    fun describeCapture(): JSONObject = JSONObject()
        .put("exposure_mode", exposureMode)
        .put("blur_budget_mm", settings.blurBudgetMm)
        .put("max_exposure_ns", 1_000_000_000L / settings.maxExposureDenominator)
        .put("low_light_max_exposure_ns", 1_000_000_000L / settings.lowLightExposureDenominator)
        .put("max_iso", settings.maxIso)
        .put("target_luma", settings.targetLuma)
        .put("unknown_speed_mps", settings.unknownSpeedKmh / 3.6)
        .put("focus_mode", focusMode)
        .put("focus_distance_diopters", focusDiopters?.toDouble() ?: JSONObject.NULL)
        .put("focus_method", focusMethod)
        .put("focus", focusReport ?: JSONObject.NULL)
        .put("metering_region", road?.let { JSONArray().put(it.rect.left).put(it.rect.top).put(it.rect.right).put(it.rect.bottom) } ?: JSONObject.NULL)
        .put("eis", "OFF")
        .put("ois", if (oisModes.isEmpty()) "UNAVAILABLE" else if (settings.oisEnabled) "ON" else "OFF")
        .put("zoom_ratio", if (hasZoomRatio) 1.0 else JSONObject.NULL)
        .put("awb", "AUTO")
        .put("template", "TEMPLATE_RECORD")
        .put("fps", settings.fps)
        .put("frame_duration_ns", frameDurationNs)
        .put("ae_fps_range", fpsRange?.let { "${it.lower}-${it.upper}" } ?: JSONObject.NULL)
        .put("analysis_stream", JSONObject().put("width", analysisSize.width).put("height", analysisSize.height).put("format", "YUV_420_888"))
        .put("streams_while_recording", streams)
        .put("encoder_timestamp_base", encoderTimestampBase)
        .put("record_size", recordSize?.let { "${it.width}x${it.height}" } ?: JSONObject.NULL)

    // ---------------------------------------------------------------- callbacks

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
            val gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS)
            val meta = ResultMeta(
                sensorTsNs = ts,
                frameNumber = result.frameNumber,
                exposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                frameDurationNs = result.get(CaptureResult.SENSOR_FRAME_DURATION),
                iso = result.get(CaptureResult.SENSOR_SENSITIVITY),
                rsSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW),
                focusDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE),
                focalLengthMm = result.get(CaptureResult.LENS_FOCAL_LENGTH),
                lensState = result.get(CaptureResult.LENS_STATE),
                oisMode = result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE),
                eisMode = result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE),
                aeState = result.get(CaptureResult.CONTROL_AE_STATE),
                afState = result.get(CaptureResult.CONTROL_AF_STATE),
                awbGains = gains?.let { "${it.red};${it.greenEven};${it.greenOdd};${it.blue}" },
                zoomRatio = if (hasZoomRatio) result.get(CaptureResult.CONTROL_ZOOM_RATIO) else null,
            )
            latest = meta
            synchronized(recentMeta) { recentMeta[ts] = meta }
            if (feedMatcher) frameMatcher.onResult(meta)
        }

        override fun onCaptureFailed(s: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
            if (!recording) return
            captureFailures++
            listener.onEvent(
                "capture_failed",
                JSONObject().put("frame_number", failure.frameNumber).put("reason", failure.reason)
                    .put("image_captured", failure.wasImageCaptured()),
            )
        }

        override fun onCaptureBufferLost(s: CameraCaptureSession, request: CaptureRequest, target: Surface, frameNumber: Long) {
            if (!recording) return
            buffersLost++
            listener.onEvent(
                "buffer_lost",
                JSONObject().put("frame_number", frameNumber).put("target", if (target == encoderSurface) "encoder" else "analysis"),
            )
        }
    }

    private val encoderListener = object : VideoEncoder.Listener {
        override fun onFormat(format: MediaFormat) {
            segmentWriter.onFormat(format)
            listener.onEvent("encoder_output_format", JSONObject().put("format", format.toString()))
        }

        override fun onSample(buffer: ByteBuffer, info: MediaCodec.BufferInfo) {
            try {
                segmentWriter.onSample(buffer, info)
            } catch (e: Exception) {
                listener.onEvent("muxer_write_error", Diag.errorJson(e))
            }
        }

        override fun onEndOfStream() {
            eosDeferred?.complete(Unit)
        }

        override fun onError(e: Exception) {
            Diag.e("encoder error", e)
            listener.onEvent("encoder_error", Diag.errorJson(e))
            eosDeferred?.complete(Unit)
            cameraHandler.post { listener.onCameraError("encoder error: $e") }
        }
    }

    private val analysisListener = ImageReader.OnImageAvailableListener { reader ->
        if (closed) return@OnImageAvailableListener
        val image = try {
            reader.acquireLatestImage()
        } catch (e: Exception) {
            null
        } ?: return@OnImageAvailableListener
        try {
            val now = Clocks.elapsedNs()
            if (now - lastAnalysisNs < ANALYSIS_INTERVAL_NS) return@OnImageAvailableListener
            lastAnalysisNs = now
            val makePreview = now - lastPreviewNs >= PREVIEW_INTERVAL_NS
            if (makePreview) lastPreviewNs = now
            val res = processor.process(image, makePreview)
            res.preview?.let { listener.onPreview(it) }
            val meta = synchronized(recentMeta) { recentMeta[res.timestampNs] }
            val lensDiopters = meta?.focusDiopters
            if (sweepActive && lensDiopters != null &&
                (meta.lensState == null || meta.lensState == CameraMetadata.LENS_STATE_STATIONARY)
            ) {
                sweepSamples.add(lensDiopters to res.sharpness)
            }
            val black = res.meanLuma < 3.0 && res.darkFraction > 0.98
            blackStreak = if (black) blackStreak + 1 else 0
            if (!imageBlack && blackStreak >= 3) {
                imageBlack = true
                listener.onEvent("image_black", JSONObject().put("t_ns", res.timestampNs).put("mean_luma", res.meanLuma))
            } else if (imageBlack && !black) {
                imageBlack = false
                listener.onEvent("image_ok", JSONObject().put("t_ns", res.timestampNs).put("mean_luma", res.meanLuma))
            }
            var decision: ExposureSetting? = null
            // A black image says nothing about scene brightness (covered lens, muted camera): keep exposure.
            if (!black && exposureMode == "MANUAL_SHUTTER_AUTO_ISO" && meta?.exposureNs != null && meta.iso != null) {
                decision = synchronized(exposureLock) {
                    controller.update(LumaSample(res.meanLuma, res.highlightFraction, meta.exposureNs, meta.iso), listener.speedMps())
                }
                decision?.let { next -> cameraHandler.post { applySetting(next) } }
            }
            listener.onAnalysis(res, meta?.exposureNs, meta?.iso, decision)
        } finally {
            image.close()
        }
    }

    companion object {
        /** ~6.7 Hz exposure/luma analysis; 5 fps operator preview. */
        const val ANALYSIS_INTERVAL_NS = 150_000_000L
        const val PREVIEW_INTERVAL_NS = 200_000_000L
        /** Analysed frames averaged per sweep position. */
        const val SWEEP_FRAMES = 2
    }
}
