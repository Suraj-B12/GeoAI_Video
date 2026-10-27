package com.geoai.capture.service

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.camera2.CameraManager
import android.location.Location
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import android.os.Process
import android.os.VibrationEffect
import android.os.VibratorManager
import android.util.Size
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.geoai.capture.BuildConfig
import com.geoai.capture.CaptureApp
import com.geoai.capture.R
import com.geoai.capture.camera.CameraCatalog
import com.geoai.capture.camera.CameraRecorder
import com.geoai.capture.camera.ExposureSetting
import com.geoai.capture.camera.FocusPlanner
import com.geoai.capture.camera.FrameMatcher
import com.geoai.capture.camera.FrameRow
import com.geoai.capture.camera.SegmentWriter
import com.geoai.capture.camera.AnalysisProcessor
import com.geoai.capture.coverage.CoverageStore
import com.geoai.capture.coverage.RoadCoverage
import com.geoai.capture.io.ManifestBuilder
import com.geoai.capture.io.SessionStore
import com.geoai.capture.sensors.ClockSyncRecorder
import com.geoai.capture.sensors.GnssRecorder
import com.geoai.capture.sensors.ImuRecorder
import com.geoai.capture.sensors.OrientationCheck
import com.geoai.capture.sensors.ThermalRecorder
import com.geoai.capture.settings.AppSettings
import com.geoai.capture.settings.SurveyParams
import com.geoai.capture.ui.MainActivity
import com.geoai.capture.util.Clocks
import com.geoai.capture.util.Diag
import com.geoai.capture.util.Fmt
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.SecureRandom
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Foreground service (types camera + location) that runs one session from PREPARE to a finalised,
 * checksummed folder. Commands arrive as Intents from the UI or the notification. All state
 * changes run on the main thread through [scope]; camera, encoder, analysis and sensors each have
 * their own thread.
 */
class SurveyService : Service() {

    companion object {
        const val ACTION_PREPARE = "com.geoai.capture.action.PREPARE"
        const val ACTION_LOCK = "com.geoai.capture.action.LOCK"
        const val ACTION_START = "com.geoai.capture.action.START"
        const val ACTION_STOP = "com.geoai.capture.action.STOP"
        const val ACTION_CANCEL = "com.geoai.capture.action.CANCEL"
        const val ACTION_MARKER = "com.geoai.capture.action.MARKER"
        const val EXTRA_PARAMS = "params_json"
        const val EXTRA_MARKER_TYPE = "marker_type"
        const val EXTRA_MARKER_NOTE = "marker_note"
        const val EXTRA_DEBUG_MIN_FREE_MB = "debug_min_free_mb"
        /** START without waiting for a GNSS fix (operator confirmed). */
        const val EXTRA_FORCE_START = "force_start"
        private const val GNSS_READY_ACC_M = 10f
        /** Mount height used for the focus plan when the survey form has none (pilot S21 mount). */
        private const val DEFAULT_MOUNT_HEIGHT_M = 1.35
        const val CHANNEL_SURVEY = "survey"
        const val CHANNEL_ALERTS = "alerts"
        private const val NOTIF_ID = 1001
        private const val ALERT_ID = 1002
        private const val POST_ROLL_MS = 3_000L
        private const val MIN_FREE_TO_START = 5_000_000_000L
        private const val MIN_FREE_RUNNING = 2_000_000_000L
        private const val LOW_FREE_WARNING = 10_000_000_000L
        private const val MAX_RECOVERIES = 10
        private val RECOVERY_BACKOFF_MS = longArrayOf(3_000, 6_000, 10_000)
    }

    private val app get() = application as CaptureApp
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private fun thread(name: String, priority: Int): Pair<HandlerThread, Handler> {
        val t = HandlerThread(name, priority).apply { start() }
        return t to Handler(t.looper)
    }

    private lateinit var cameraThread: Pair<HandlerThread, Handler>
    private lateinit var encoderThread: Pair<HandlerThread, Handler>
    private lateinit var analysisThread: Pair<HandlerThread, Handler>
    private lateinit var sensorThread: Pair<HandlerThread, Handler>
    private val callbackExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "geoai-gnss") }

    private var settings: AppSettings? = null
    private var ctx: SessionContext? = null
    private var baseMeta: JSONObject? = null
    private var recorder: CameraRecorder? = null
    private var segmentWriter: SegmentWriter? = null
    private var matcher: FrameMatcher? = null
    private var gnss: GnssRecorder? = null
    private var imu: ImuRecorder? = null
    private var thermal: ThermalRecorder? = null
    private var clock: ClockSyncRecorder? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var cameraId: String? = null
    private var sensorToSessionNs = 0L
    private var lockState: CameraRecorder.LockState? = null
    private var recordSize: Size? = null
    private var recoveryAttempts = 0
    private var recovering = false
    private var prepareNs = 0L
    private var lastGnssRestartNs = 0L
    private var stepDownDone = false
    private var stopping = false
    private val encoders = JSONArray()
    private var orientationAtStart = JSONObject()
    private var lastCapture: JSONObject? = null
    private val firstFrameT = AtomicLong(Long.MIN_VALUE)
    private val lastFrameT = AtomicLong(Long.MIN_VALUE)
    private val framesEncoded = AtomicLong()
    private val framesUnmatched = AtomicLong()
    @Volatile private var lastFix: Location? = null
    private val recentSpeeds = ArrayDeque<Pair<Long, Double>>()
    @Volatile private var distanceM = 0.0
    @Volatile private var mockSeen = false
    private val coverageLock = Any()
    private var coverage: RoadCoverage? = null
    private var coverageStore: CoverageStore? = null
    /** Non-repeating road km of all earlier sessions, read at prepare. */
    private var priorNewM = 0.0
    private var autoStartTried = false
    private var gnssFixCount = AtomicLong()
    private var recordingStartNs = 0L
    private var gnssLostLogged = false
    private var monitorJob: Job? = null
    private val warnings = LinkedHashMap<String, Long>()
    private val lastAlertMs = HashMap<String, Long>()
    private val lastWarnLoggedMs = HashMap<String, Long>()
    private var debugMinFreeBytes: Long? = null
    private val minFreeToStart: Long get() = debugMinFreeBytes ?: MIN_FREE_TO_START
    private val minFreeRunning: Long get() = debugMinFreeBytes?.div(2) ?: MIN_FREE_RUNNING

    private val phase: Phase get() = SurveyController.status.value.phase
    private fun update(t: (SurveyStatus) -> SurveyStatus) = SurveyController.update(t)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        cameraThread = thread("geoai-camera", Process.THREAD_PRIORITY_DISPLAY)
        encoderThread = thread("geoai-encoder", Process.THREAD_PRIORITY_DISPLAY)
        analysisThread = thread("geoai-analysis", Process.THREAD_PRIORITY_DEFAULT)
        sensorThread = thread("geoai-sensors", Process.THREAD_PRIORITY_DEFAULT)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Diag.i("onStartCommand ${intent?.action} phase=$phase")
        when (intent?.action) {
            ACTION_PREPARE -> {
                if (!promoteToForeground()) return START_NOT_STICKY
                val params = intent.getStringExtra(EXTRA_PARAMS)
                // Automated emulator tests only: emulators have little free storage.
                debugMinFreeBytes = if (BuildConfig.DEBUG && intent.hasExtra(EXTRA_DEBUG_MIN_FREE_MB))
                    intent.getIntExtra(EXTRA_DEBUG_MIN_FREE_MB, 0) * 1_000_000L else null
                scope.launch { prepare(params) }
            }
            ACTION_LOCK -> scope.launch { lock() }
            ACTION_START -> {
                val force = intent.getBooleanExtra(EXTRA_FORCE_START, false)
                scope.launch { startRecording(force) }
            }
            ACTION_STOP -> scope.launch { stop("user_stop") }
            // Not named cancel(): inside launch {} that would resolve to CoroutineScope.cancel() and kill the coroutine.
            ACTION_CANCEL -> scope.launch { cancelSession() }
            ACTION_MARKER -> marker(intent.getStringExtra(EXTRA_MARKER_TYPE) ?: "marker", intent.getStringExtra(EXTRA_MARKER_NOTE) ?: "")
            else -> if (ctx == null) stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        // Normal stops clear ctx first. If the system tears us down mid-session, finalise what we can.
        if (ctx != null) {
            runBlocking { withTimeoutOrNull(8_000) { stop("service_destroyed") } }
        }
        scope.cancel()
        listOf(cameraThread, encoderThread, analysisThread, sensorThread).forEach { it.first.quitSafely() }
        callbackExecutor.shutdown()
        super.onDestroy()
    }

    // ---------------------------------------------------------------- foreground & notifications

    private fun promoteToForeground(): Boolean = try {
        ServiceCompat.startForeground(
            this, NOTIF_ID, notification("Preparing survey…"),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION,
        )
        true
    } catch (e: Exception) {
        update { it.copy(phase = Phase.ERROR, error = "Cannot start foreground service: ${e.message}") }
        stopSelf()
        false
    }

    private fun notification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, SurveyService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL_SURVEY)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("GeoAI Capture")
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(open)
            .addAction(0, "Stop", stop)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, notification(text))
    }

    private fun warn(message: String, alert: Boolean = false) {
        val nowMs = System.currentTimeMillis()
        val logIt = synchronized(warnings) {
            warnings.remove(message)
            warnings[message] = nowMs
            while (warnings.size > 6) warnings.remove(warnings.keys.first())
            // The monitor re-raises standing warnings every few seconds: log each message at most once a minute.
            val last = lastWarnLoggedMs[message] ?: 0L
            if (nowMs - last > 60_000) {
                lastWarnLoggedMs[message] = nowMs
                true
            } else false
        }
        if (logIt) ctx?.events?.log("warning", JSONObject().put("message", message))
        if (alert && settings?.alerts != false) {
            val now = System.currentTimeMillis()
            if (now - (lastAlertMs[message] ?: 0L) > 60_000) {
                lastAlertMs[message] = now
                runCatching {
                    getSystemService(VibratorManager::class.java).defaultVibrator
                        .vibrate(VibrationEffect.createWaveform(longArrayOf(0, 400, 200, 400, 200, 800), -1))
                }
                getSystemService(NotificationManager::class.java).notify(
                    ALERT_ID,
                    NotificationCompat.Builder(this, CHANNEL_ALERTS)
                        .setSmallIcon(R.drawable.ic_notification)
                        .setContentTitle("Survey alert")
                        .setContentText(message)
                        .setPriority(NotificationCompat.PRIORITY_HIGH)
                        .setAutoCancel(true)
                        .build(),
                )
            }
        }
    }

    private fun currentWarnings(): List<String> = synchronized(warnings) { warnings.keys.toList().asReversed() }

    // ---------------------------------------------------------------- session lifecycle

    private suspend fun prepare(paramsJson: String?) {
        if (ctx != null) return
        stopping = false
        prepareNs = Clocks.elapsedNs()
        lastGnssRestartNs = prepareNs
        val params = paramsJson?.let { runCatching { SurveyParams.fromJson(JSONObject(it)) }.getOrNull() } ?: SurveyParams()
        val s = app.settingsRepo.load()
        settings = s
        update { SurveyStatus(phase = Phase.PREPARING, sessionType = params.sessionType, message = "Opening camera…") }
        try {
            val manager = getSystemService(CameraManager::class.java)
            val id = CameraCatalog.chooseMainBackCamera(manager) ?: error("No usable back camera")
            cameraId = id
            val chars = manager.getCameraCharacteristics(id)
            val tsSource = CameraCatalog.timestampSourceName(chars)
            val start = Clocks.bestPair()
            // REALTIME cameras already use the session clock; UNKNOWN ones are ~monotonic.
            sensorToSessionNs = if (tsSource == "REALTIME") 0L else start.elapsedMinusMonoNs
            val sessionId = "${Clocks.compactUtc(start.wallMs)}_${s.deviceId}_${hex4()}"
            val dir = app.store.create(sessionId)
            val c = SessionContext(dir, sessionId, params, s.rawGnss)
            ctx = c
            acquireWakeLock()
            autoStartTried = false
            loadCoverage(sessionId)

            gnss = GnssRecorder(this, callbackExecutor, c.gnssFixes, c.gnssStatus, c.nmea, c.gnssRaw) { onFix(it) }
                .also { it.start(s.gnssFullTracking) }
            imu = ImuRecorder(this, sensorThread.second, c.imu).also { it.start() }
            thermal = ThermalRecorder(this, sensorThread.second, callbackExecutor, c.thermal) { st -> scope.launch { onThermal(st) } }
                .also { it.start() }
            clock = ClockSyncRecorder(sensorThread.second, c.clock).also { it.start() }

            val meta = SessionMetadata.initial(
                this, sessionId, params, s, CameraCatalog.describe(id, chars), gnss!!.hardwareInfo(), imu!!.describe(),
                app.store.root.absolutePath, app.store.usingSharedStorage, start, tsSource,
            )
            baseMeta = meta
            withContext(Dispatchers.IO) { SessionStore.writeJsonAtomic(File(dir, "session.json"), meta) }
            c.events.log("session_start", JSONObject().put("session_type", params.sessionType).put("camera_id", id))
            if (tsSource != "REALTIME") warn("Camera timestamps are not REALTIME on this phone: sync is approximate")

            val sw = SegmentWriter(File(dir, "video"), s.segmentMinutes * 60_000_000L, c.segments, c.events) { f -> matcher?.onFrame(f) }
            segmentWriter = sw
            matcher = FrameMatcher(
                toleranceNs = 2_000_000L,
                offsetCandidatesNs = listOf(0L, -start.elapsedMinusMonoNs),
                sink = { row -> onFrameRow(row) },
            )
            val r = newRecorder(id, s, sw, matcher!!)
            recorder = r
            r.open()
            update { it.copy(phase = Phase.PREVIEW, sessionId = sessionId, message = "Camera ready") }
            updateNotification("Ready: $sessionId")
            startMonitor()
            lock()
        } catch (e: Exception) {
            Diag.e("prepare failed", e)
            ctx?.events?.log("prepare_failed", Diag.errorJson(e))
            update { it.copy(phase = Phase.ERROR, error = "Could not start: ${e.message}") }
            cleanupWithoutSaving(deleteFolder = true)
        }
    }

    /** Counts any earlier sessions not counted yet, then starts this session's counter. Never blocks a survey. */
    private suspend fun loadCoverage(sessionId: String) {
        val store = CoverageStore(app.store.root)
        coverageStore = store
        val (cells, prior) = withContext(Dispatchers.IO) {
            runCatching {
                val n = store.backfill(app.store.sessionsDir, sessionId)
                if (n > 0) Diag.i("coverage: counted $n earlier session(s)")
                store.loadCells() to store.totalNewM()
            }.getOrElse {
                Diag.e("coverage: load failed", it)
                HashSet<Long>() to 0.0
            }
        }
        priorNewM = prior
        synchronized(coverageLock) { coverage = RoadCoverage(cells) }
        update { it.copy(roadNewKm = 0.0, roadTotalKm = prior / 1000.0) }
    }

    private fun newRecorder(id: String, s: AppSettings, sw: SegmentWriter, fm: FrameMatcher) = CameraRecorder(
        this, id, s, cameraThread.second, encoderThread.second, analysisThread.second, sw, fm,
        object : CameraRecorder.Listener {
            override fun onPreview(bitmap: android.graphics.Bitmap) = SurveyController.setPreview(bitmap)

            override fun onAnalysis(result: AnalysisProcessor.Result, frameExposureNs: Long?, frameIso: Int?, decision: ExposureSetting?) {
                val c = ctx ?: return
                val t = result.timestampNs + sensorToSessionNs
                val speed = speedMps()
                c.analysis.write(
                    Fmt.row(
                        t.toString(), Fmt.optFixed(result.meanLuma, 2), Fmt.optFixed(result.highlightFraction, 5),
                        Fmt.optFixed(result.darkFraction, 5), Fmt.optFixed(result.sharpness, 2), Fmt.opt(frameExposureNs),
                        Fmt.opt(frameIso), Fmt.optFixed(speed, 3),
                    ),
                )
                if (decision != null) {
                    c.exposure.write(
                        Fmt.row(
                            t.toString(), decision.reason, decision.exposureNs.toString(), decision.iso.toString(),
                            Fmt.bool(decision.lowLight), Fmt.optFixed(speed, 3), Fmt.optFixed(result.meanLuma, 2),
                        ),
                    )
                }
                update { it.copy(meanLuma = result.meanLuma, sharpness = result.sharpness, highlightFrac = result.highlightFraction) }
            }

            override fun onState(state: CameraRecorder.CameraState) = update {
                it.copy(
                    exposureMode = state.exposureMode, focusMode = state.focusMode, focusDiopters = state.focusDiopters,
                    exposureNs = state.setting?.exposureNs, iso = state.setting?.iso, lowLight = state.setting?.lowLight == true,
                    recordWidth = state.recordWidth, recordHeight = state.recordHeight, streams = state.streams,
                    captureFailures = state.captureFailures, buffersLost = state.buffersLost, focusNote = state.focusNote,
                )
            }

            override fun onCameraError(message: String) {
                scope.launch { handleCameraError(message) }
            }

            override fun onEvent(type: String, value: JSONObject) {
                ctx?.events?.log(type, value)
                if (type == "image_black") scope.launch { warn("Camera image is black: lens covered or camera blocked", alert = true) }
            }

            override fun speedMps(): Double? = this@SurveyService.speedMps()
        },
    )

    private suspend fun lock() {
        val r = recorder ?: return
        if (phase != Phase.PREVIEW && phase != Phase.LOCKING) return
        update { it.copy(phase = Phase.LOCKING, locked = false, message = "Focusing on the road and metering. Keep the vehicle still.") }
        // The focus plan needs the camera's tilt: give the gravity sensor a moment after the camera opens.
        withTimeoutOrNull(1_500) { while (imu?.latestGravity == null) delay(50) }
        try {
            lockState = r.lockFocusAndExposure(focusInputs())
            lastCapture = r.describeCapture()
            val warnManual = lockState?.exposureMode == "AUTO_FALLBACK"
            if (warnManual) warn("No manual shutter on this camera: motion blur is NOT controlled", alert = true)
            update { it.copy(phase = Phase.PREVIEW, locked = true, message = "Focus and exposure locked") }
        } catch (e: Exception) {
            update { it.copy(phase = Phase.PREVIEW, locked = false, message = "Lock failed: ${e.message}") }
        }
    }

    /** Gravity and mount height for the road focus plan (CameraRecorder / FocusPlanner). */
    private fun focusInputs(): CameraRecorder.FocusInputs {
        val form = ctx?.params?.nominalHeightM?.replace(',', '.')?.trim()?.toDoubleOrNull()?.takeIf { it in 0.3..5.0 }
        return CameraRecorder.FocusInputs(
            gravity = imu?.latestGravity,
            heightM = form ?: DEFAULT_MOUNT_HEIGHT_M,
            heightSource = if (form != null) "survey_form" else "default",
        )
    }

    private fun gnssReady(): Boolean {
        val f = lastFix ?: return false
        val age = Clocks.elapsedNs() - f.elapsedRealtimeNanos
        return age in 0..2_000_000_000L && f.hasAccuracy() && f.accuracy <= GNSS_READY_ACC_M
    }

    private suspend fun startRecording(force: Boolean = false) {
        val r = recorder ?: return
        val s = settings ?: return
        val c = ctx ?: return
        if (phase != Phase.PREVIEW) return
        if (lockState == null) lock()
        // Survey frames need GNSS fixes on both sides for interpolation: wait for a good fix unless forced.
        if (c.params.sessionType == SurveyParams.TYPE_SURVEY && !gnssReady() && !force) {
            warn("Waiting for GNSS fix (≤ ${GNSS_READY_ACC_M.toInt()} m) before recording")
            return
        }
        if (force && !gnssReady()) c.events.log("recording_forced_without_gnss", JSONObject())
        val free = withContext(Dispatchers.IO) { app.store.freeBytes() }
        if (free in 0 until minFreeToStart) {
            warn(String.format(Locale.ROOT, "Not enough free storage to start (%.1f GB)", free / 1e9), alert = true)
            return
        }
        val size = CameraCatalog.pickRecordSize(r.chars, s.resolution.width, s.resolution.height)
            ?: CameraCatalog.pickRecordSize(r.chars, 1920, 1080)
        if (size == null) {
            update { it.copy(message = "This camera has no 16:9 recording size") }
            return
        }
        if (size.width != s.resolution.width) warn("Recording at ${size.width}x${size.height} (requested size unsupported)")
        update { it.copy(phase = Phase.STARTING, message = "Starting recording…") }
        try {
            orientationAtStart = orientationJson()
            r.startRecording(size)
            recordSize = size
            r.activeEncoder?.let { encoders.put(it) }
            recordingStartNs = Clocks.elapsedNs()
            clock?.sampleNow()
            lastCapture = r.describeCapture()
            c.events.log("recording_start", JSONObject().put("width", size.width).put("height", size.height))
            writeInterimMeta()
            update { it.copy(phase = Phase.RECORDING, message = "Recording") }
            updateNotification("Recording ${size.width}x${size.height}")
            if (orientationAtStart.optString("verdict") != OrientationCheck.Verdict.UPRIGHT.name) {
                warn("Check mount orientation: ${orientationAtStart.optString("verdict")}", alert = true)
            }
        } catch (e: Exception) {
            Diag.e("recording start failed", e)
            c.events.log("recording_start_failed", Diag.errorJson(e))
            update { it.copy(phase = Phase.PREVIEW, message = "Could not start recording: ${e.message}") }
            reopenForPreview()
        }
    }

    /**
     * Rewrites session.json as soon as recording starts (and after encoder changes), so a session
     * interrupted by a crash or power loss still records its capture settings, encoders and orientation.
     */
    private suspend fun writeInterimMeta() {
        val c = ctx ?: return
        val base = baseMeta ?: return
        val o = JSONObject(base.toString())
        lastCapture?.let { o.put("capture_settings", it) }
        o.put("encoders", JSONArray(encoders.toString()))
        o.put("orientation", orientationAtStart)
        o.put("recording_in_progress", true)
        val clock = o.optJSONObject("clock") ?: JSONObject()
        clock.put("encoder_timestamp_base", lastCapture?.optString("encoder_timestamp_base") ?: JSONObject.NULL)
        clock.put("sensor_to_session_offset_ns", sensorToSessionNs)
        o.put("clock", clock)
        runCatching { withContext(Dispatchers.IO) { SessionStore.writeJsonAtomic(File(c.dir, "session.json"), o) } }
    }

    private suspend fun reopenForPreview() {
        val id = cameraId ?: return
        runCatching { recorder?.close() }
        val nr = newRecorder(id, settings!!, segmentWriter!!, matcher!!)
        recorder = nr
        runCatching {
            nr.open()
            lockState?.let { nr.applyLock(it) }
        }.onFailure { update { s -> s.copy(phase = Phase.ERROR, error = "Camera unavailable: ${it.message}") } }
    }

    private suspend fun stop(reason: String) {
        if (stopping) return
        val c = ctx ?: run { finishService(); return }
        stopping = true
        val wasRecording = phase == Phase.RECORDING || phase == Phase.STARTING
        update { it.copy(phase = Phase.STOPPING, message = "Stopping…") }
        c.events.log("stop_requested", JSONObject().put("reason", reason))
        Diag.i("stop($reason): begin, wasRecording=$wasRecording")
        recorder?.let { r ->
            runCatching { r.stopRecording(reason) }
            lastCapture = runCatching { r.describeCapture() }.getOrNull() ?: lastCapture
        }
        // Keep GNSS/IMU running briefly so the last frames have a fix after them for interpolation.
        Diag.i("stop: recorder stopped")
        if (wasRecording) delay(POST_ROLL_MS)
        runCatching { recorder?.close() }
        recorder = null
        Diag.i("stop: recorder closed")
        val fm = matcher
        fm?.flush()
        withContext(Dispatchers.IO) { segmentWriter?.finish(reason) }
        Diag.i("stop: segments finished")
        gnss?.stop()
        imu?.stop()
        thermal?.stop()
        clock?.stop()
        monitorJob?.cancel()
        Diag.i("stop: sensors stopped")
        c.events.log("session_end", JSONObject().put("reason", reason))

        val end = Clocks.bestPair()
        val clockExtra = JSONObject()
            .put("encoder_timestamp_base", lastCapture?.optString("encoder_timestamp_base") ?: JSONObject.NULL)
            .put("encoder_pts_offset_ns", fm?.offsetNs ?: JSONObject.NULL)
            .put("encoder_pts_offset_method", fm?.offsetMethod ?: JSONObject.NULL)
            .put("sensor_to_session_offset_ns", sensorToSessionNs)
            .put("elapsed_minus_mono_ns_at_end", end.elapsedMinusMonoNs)
        val recording = JSONObject()
            .put("first_frame_t_ns", firstFrameT.get().takeIf { it != Long.MIN_VALUE } ?: JSONObject.NULL)
            .put("last_frame_t_ns", lastFrameT.get().takeIf { it != Long.MIN_VALUE } ?: JSONObject.NULL)
            .put("segments", segmentWriter?.segmentIndex ?: 0)
            .put("thermal_step_down", stepDownDone)
            .put("camera_recoveries", recoveryAttempts)
        val stats = JSONObject()
            .put("frames_encoded", framesEncoded.get())
            .put("frames_unmatched", framesUnmatched.get())
            .put("frames_matched", fm?.matched ?: 0)
            .put("camera_frames_not_encoded", fm?.resultsWithoutFrame ?: 0)
            .put("samples_dropped_before_first_keyframe", segmentWriter?.droppedBeforeKeyframe ?: 0)
            .put("gnss_fixes", gnssFixCount.get())
            .put("gnss_restarts", gnss?.restarts ?: 0)
            .put("mock_location_seen", mockSeen)
            .put("max_thermal_status", thermal?.maxStatus ?: 0)
            .put("distance_m", distanceM)
            .put("warnings", JSONArray(currentWarnings()))
        val (roadNewM, roadPathM, cells) = synchronized(coverageLock) {
            val cov = coverage
            coverage = null
            Triple(cov?.newM ?: 0.0, cov?.pathM ?: 0.0, cov?.takeFresh() ?: emptyList())
        }
        stats.put("road_new_m", roadNewM)
            .put("road_counted_m", roadPathM)
            .put("road_total_new_m_after", priorNewM + roadNewM)
        if (c.params.sessionType in CoverageStore.COUNTED_TYPES) {
            withContext(Dispatchers.IO) {
                runCatching { coverageStore?.commit(c.id, roadNewM, cells) }.onFailure { Diag.e("coverage: commit failed", it) }
            }
        }
        val base = baseMeta ?: JSONObject()
        val meta = SessionMetadata.complete(this, base, lastCapture, encoders, clockExtra, orientationAtStart, recording, end, reason, stats)
        withContext(Dispatchers.IO) {
            c.closeAll()
            SessionStore.writeJsonAtomic(File(c.dir, "session.json"), meta)
        }
        update { it.copy(phase = Phase.FINALIZING, message = "Writing checksums…", finalizeProgress = 0f) }
        withContext(Dispatchers.IO) {
            ManifestBuilder.write(c.dir, c.id) { p -> update { it.copy(finalizeProgress = p) } }
        }
        ctx = null
        Diag.i("stop: finalised ${c.id}")
        SurveyController.setPreview(null)
        update { SurveyStatus(phase = Phase.IDLE, message = "Saved ${c.id} ($reason)") }
        finishService()
    }

    private suspend fun cancelSession() {
        Diag.i("cancelSession: phase=$phase")
        if (phase == Phase.RECORDING || phase == Phase.STARTING) {
            stop("user_stop")
            return
        }
        cleanupWithoutSaving(deleteFolder = true)
        update { SurveyStatus(phase = Phase.IDLE, message = "Cancelled; nothing was recorded") }
    }

    private suspend fun cleanupWithoutSaving(deleteFolder: Boolean) {
        runCatching { recorder?.close() }.onFailure { Diag.e("cleanup: recorder.close failed", it) }
        recorder = null
        gnss?.stop()
        imu?.stop()
        thermal?.stop()
        clock?.stop()
        monitorJob?.cancel()
        val c = ctx
        ctx = null
        if (c != null) withContext(Dispatchers.IO) {
            runCatching { c.closeAll() }
            if (deleteFolder) c.dir.deleteRecursively()
        }
        SurveyController.setPreview(null)
        finishService()
    }

    private fun finishService() {
        stopping = false
        runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
        wakeLock = null
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "GeoAICapture:survey")
            .apply { acquire(12 * 60 * 60 * 1000L) }
    }

    private fun marker(type: String, note: String) {
        val c = ctx ?: return
        val f = lastFix
        c.events.log(
            "operator_marker",
            JSONObject().put("marker_type", type).put("note", note)
                .put("last_fix_t_ns", f?.elapsedRealtimeNanos ?: JSONObject.NULL),
        )
        update { it.copy(message = "Marker saved: $type ${note.take(30)}") }
    }

    // ---------------------------------------------------------------- failures

    private suspend fun handleCameraError(message: String) {
        val c = ctx ?: return
        if (stopping || recovering) return
        warn("Camera interrupted: $message", alert = true)
        if (phase == Phase.RECORDING) {
            recovering = true
            try {
                val r = recorder
                runCatching { r?.stopRecording("error") }
                runCatching { r?.close() }
                recorder = null
                // Up to 3 attempts per interruption (another app may hold the camera for a while),
                // and at most MAX_RECOVERIES interruptions per session.
                var recovered = false
                var attempt = 0
                while (!recovered && attempt < 3 && recoveryAttempts < MAX_RECOVERIES && !stopping) {
                    if ((thermal?.latest?.status ?: 0) >= ThermalRecorder.STATUS_CRITICAL) break
                    attempt++
                    recoveryAttempts++
                    c.events.log("camera_recovery_attempt", JSONObject().put("attempt", attempt).put("session_total", recoveryAttempts))
                    update { it.copy(message = "Camera interrupted, retrying ($attempt/3)…") }
                    delay(RECOVERY_BACKOFF_MS[attempt - 1])
                    try {
                        val nr = newRecorder(cameraId!!, settings!!, segmentWriter!!, matcher!!)
                        recorder = nr
                        nr.open()
                        lockState?.let { nr.applyLock(it) }
                        nr.startRecording(recordSize!!)
                        nr.activeEncoder?.let { encoders.put(it) }
                        recovered = true
                        writeInterimMeta()
                        c.events.log("camera_recovered", JSONObject().put("attempt", attempt))
                        update { it.copy(phase = Phase.RECORDING, message = "Recording (recovered after camera error)") }
                    } catch (e: Exception) {
                        Diag.e("camera recovery attempt $attempt failed", e)
                        c.events.log("camera_recovery_failed", Diag.errorJson(e).put("attempt", attempt))
                        runCatching { recorder?.close() }
                        recorder = null
                    }
                }
                if (!recovered && !stopping) stop("camera_error")
            } finally {
                recovering = false
            }
        } else if (phase == Phase.PREVIEW || phase == Phase.LOCKING || phase == Phase.STARTING) {
            update { it.copy(phase = Phase.ERROR, error = "Camera error: $message") }
            cleanupWithoutSaving(deleteFolder = true)
        }
    }

    private suspend fun onThermal(status: Int) {
        val c = ctx ?: return
        c.events.log("thermal_status", JSONObject().put("status", status))
        if (phase != Phase.RECORDING) {
            if (status >= ThermalRecorder.STATUS_MODERATE) warn("Phone is hot (thermal status $status). Shade it before recording.")
            return
        }
        when {
            status >= ThermalRecorder.STATUS_CRITICAL -> {
                warn("Overheating: recording stopped", alert = true)
                stop("thermal_critical")
            }
            status >= ThermalRecorder.STATUS_SEVERE && settings?.thermalStepDown == true && !stepDownDone && (recordSize?.width ?: 0) > 1920 ->
                stepDown()
            status >= ThermalRecorder.STATUS_MODERATE -> warn("Phone is hot (thermal status $status): shade or cool it", alert = true)
        }
    }

    private suspend fun stepDown() {
        val r = recorder ?: return
        val small = CameraCatalog.pickRecordSize(r.chars, 1920, 1080) ?: return
        stepDownDone = true
        warn("Overheating: switched to ${small.width}x${small.height}", alert = true)
        try {
            r.stopRecording("thermal_step_down")
            r.startRecording(small)
            recordSize = small
            r.activeEncoder?.let { encoders.put(it) }
            ctx?.events?.log("thermal_step_down", JSONObject().put("width", small.width).put("height", small.height))
            writeInterimMeta()
            updateNotification("Recording ${small.width}x${small.height} (thermal step-down)")
        } catch (e: Exception) {
            Diag.e("thermal step-down failed", e)
            ctx?.events?.log("thermal_step_down_failed", Diag.errorJson(e))
            stop("thermal_step_down_failed")
        }
    }

    // ---------------------------------------------------------------- data callbacks

    private fun onFrameRow(row: FrameRow) {
        val c = ctx ?: return
        c.frames.write(c.frameRow(row, sensorToSessionNs))
        framesEncoded.incrementAndGet()
        if (row.meta == null) framesUnmatched.incrementAndGet()
        val ts = row.meta?.sensorTsNs ?: row.offsetNs?.let { row.frame.ptsUs * 1000L - it } ?: return
        val t = ts + sensorToSessionNs
        firstFrameT.compareAndSet(Long.MIN_VALUE, t)
        lastFrameT.set(t)
    }

    private fun onFix(l: Location) {
        gnssFixCount.incrementAndGet()
        if (l.hasSpeed()) synchronized(recentSpeeds) {
            recentSpeeds.addLast(l.elapsedRealtimeNanos to l.speed.toDouble())
            while (recentSpeeds.size > 20) recentSpeeds.removeFirst()
        }
        if (l.isMock) mockSeen = true
        val prev = lastFix
        if (prev != null && phase == Phase.RECORDING && l.hasAccuracy() && l.accuracy < 25f && l.speed > 0.5f) {
            distanceM += haversineM(prev.latitude, prev.longitude, l.latitude, l.longitude)
        }
        lastFix = l
        synchronized(coverageLock) {
            val cov = coverage ?: return
            if (phase != Phase.RECORDING || ctx?.params?.sessionType !in CoverageStore.COUNTED_TYPES) {
                cov.breakTrack()
            } else if (CoverageStore.accept(if (l.hasAccuracy()) l.accuracy else null, if (l.hasSpeed()) l.speed else null, l.isMock)) {
                cov.add(l.latitude, l.longitude)
            }
        }
    }

    /**
     * Speed used for the shutter: the highest GNSS speed of the last 3 s. Conservative on purpose:
     * one glitched 0 m/s fix while moving would otherwise allow a long, blurring exposure.
     */
    private fun speedMps(): Double? {
        val now = Clocks.elapsedNs()
        val recent = synchronized(recentSpeeds) { recentSpeeds.filter { now - it.first in 0..3_000_000_000L } }
        if (recent.isEmpty()) return null
        return recent.maxOf { it.second }
    }

    private fun orientationJson(): JSONObject {
        val g = imu?.latestGravity
        val so = recorder?.sensorOrientation ?: 90
        return JSONObject()
            .put("sensor_orientation_deg", so)
            .put("gravity_device_xyz", g?.let { JSONArray().put(it[0].toDouble()).put(it[1].toDouble()).put(it[2].toDouble()) } ?: JSONObject.NULL)
            .put("verdict", OrientationCheck.verdict(so, g).name)
            .put("rule", "UPRIGHT means sensor row 0 (top of the stored image) faces up in the world")
    }

    private fun startMonitor() {
        monitorJob?.cancel()
        monitorJob = scope.launch {
            var tick = 0
            while (isActive) {
                delay(1_000)
                tick++
                if (tick % 10 == 0) Diag.i("monitor tick $tick phase=$phase")
                val now = Clocks.elapsedNs()
                val fix = lastFix
                val ageS = fix?.let { (now - it.elapsedRealtimeNanos) / 1e9 }
                if (phase == Phase.RECORDING) {
                    if (ageS == null || ageS > 5.0) {
                        if (!gnssLostLogged) {
                            gnssLostLogged = true
                            ctx?.events?.log("gnss_lost", JSONObject().put("fix_age_s", ageS ?: JSONObject.NULL))
                        }
                        if ((ageS ?: 99.0) > 10.0) warn("GNSS fix lost", alert = true)
                    } else if (gnssLostLogged) {
                        gnssLostLogged = false
                        ctx?.events?.log("gnss_recovered", JSONObject().put("fix_age_s", ageS))
                    }
                }
                var freeGb: Double? = null
                if (tick % 5 == 1) {
                    // StatFs can block if shared storage stalls; never call it on the main thread.
                    val free = withContext(Dispatchers.IO) { app.store.freeBytes() }
                    freeGb = if (free >= 0) free / 1e9 else null
                    if (phase == Phase.RECORDING && free in 0 until minFreeRunning) {
                        warn("Storage full: recording stopped", alert = true)
                        // stop() cancels this monitor job, so it must run in its own coroutine.
                        scope.launch { stop("low_storage") }
                        return@launch
                    } else if (free in 0 until LOW_FREE_WARNING) {
                        warn("Low storage")
                    }
                }
                val th = thermal?.latest
                val verdict = OrientationCheck.verdict(recorder?.sensorOrientation ?: 90, imu?.latestGravity)
                // GNSS watchdog: no fix for 15 s while location is on → re-register the listeners.
                val g = gnss
                val locOn = g?.locationEnabled ?: true
                if (g != null) {
                    if (!locOn) {
                        warn("Location is OFF: turn on Location (GPS) in quick settings", alert = phase == Phase.RECORDING)
                    } else {
                        val sinceFix = if (g.lastFixReceiptNs > 0) now - g.lastFixReceiptNs else now - prepareNs
                        if (sinceFix > 15_000_000_000L && now - lastGnssRestartNs > 15_000_000_000L) {
                            lastGnssRestartNs = now
                            g.restart()
                            ctx?.events?.log("gnss_restart", JSONObject().put("seconds_since_fix", sinceFix / 1e9))
                        }
                    }
                }
                val ready = gnssReady()
                maybeAutoStart(ready)
                val roadNewM = synchronized(coverageLock) { coverage?.newM }
                // Live mount check (segment mode): the tilt needs only gravity; the road the frame shows, and whether a
                // segment fits, also need the lens. One gravity reading for both, so the rows never disagree.
                val mountInputs = focusInputs()
                val view = recorder?.roadView(mountInputs)
                update { s ->
                    s.copy(
                        gnssReady = ready,
                        locationEnabled = locOn,
                        gnssAccM = fix?.takeIf { it.hasAccuracy() }?.accuracy,
                        gnssSats = gnss?.satellitesUsed?.takeIf { it >= 0 },
                        gnssFixAgeS = ageS,
                        speedKmh = fix?.takeIf { it.hasSpeed() && (ageS ?: 99.0) < 3.0 }?.speed?.toDouble()?.times(3.6),
                        mock = mockSeen,
                        recordingS = if (phase == Phase.RECORDING) (now - recordingStartNs) / 1e9 else s.recordingS,
                        distanceKm = distanceM / 1000.0,
                        roadNewKm = roadNewM?.div(1000.0) ?: s.roadNewKm,
                        roadTotalKm = (priorNewM + (roadNewM ?: 0.0)) / 1000.0,
                        framesEncoded = framesEncoded.get(),
                        framesUnmatched = framesUnmatched.get(),
                        segment = segmentWriter?.segmentIndex ?: 0,
                        thermalStatus = th?.status ?: 0,
                        headroom = th?.headroom,
                        batteryTempC = th?.batteryTempC,
                        batteryPct = th?.batteryPct,
                        charging = th?.charging == true,
                        freeGb = freeGb ?: s.freeGb,
                        orientation = verdict.name,
                        tiltDeg = FocusPlanner.tiltDeg(mountInputs.gravity),
                        roadNearM = view?.nearM,
                        roadFarM = view?.farM,
                        segmentFits = view?.segmentFits,
                        warnings = currentWarnings(),
                    )
                }
            }
        }
    }

    /**
     * Auto-record: once the camera is locked and (for surveys) GNSS is good, recording starts by itself,
     * once per session. Calibration sessions stay manual.
     */
    private fun maybeAutoStart(gnssOk: Boolean) {
        val s = settings ?: return
        val c = ctx ?: return
        if (!s.autoRecord || autoStartTried || phase != Phase.PREVIEW || !SurveyController.status.value.locked) return
        val type = c.params.sessionType
        if (type != SurveyParams.TYPE_SURVEY && type != SurveyParams.TYPE_TEST) return
        if (type == SurveyParams.TYPE_SURVEY && !gnssOk) return
        autoStartTried = true
        c.events.log("auto_record_start", JSONObject())
        scope.launch { startRecording() }
    }

    private fun hex4(): String {
        val b = ByteArray(2).also { SecureRandom().nextBytes(it) }
        return SessionStore.hex(b)
    }

    private fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val r = 6_371_008.8
        val p1 = Math.toRadians(lat1)
        val p2 = Math.toRadians(lat2)
        val dp = p2 - p1
        val dl = Math.toRadians(lon2 - lon1)
        val a = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2)
        return 2 * r * asin(sqrt(a.coerceIn(0.0, 1.0)))
    }
}
