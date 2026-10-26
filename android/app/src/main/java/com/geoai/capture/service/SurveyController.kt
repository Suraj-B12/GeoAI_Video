package com.geoai.capture.service

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class Phase { IDLE, PREPARING, PREVIEW, LOCKING, STARTING, RECORDING, STOPPING, FINALIZING, ERROR }

data class SurveyStatus(
    val phase: Phase = Phase.IDLE,
    val sessionId: String? = null,
    val sessionType: String = "",
    val message: String? = null,
    val error: String? = null,
    val locked: Boolean = false,
    val exposureMode: String = "",
    val focusMode: String = "",
    val focusDiopters: Float? = null,
    /** How focus was set and for which road distances, e.g. "geometry sweep confirmed; road 1.4-4.0 m". */
    val focusNote: String = "",
    val exposureNs: Long? = null,
    val iso: Int? = null,
    val lowLight: Boolean = false,
    val meanLuma: Double? = null,
    val sharpness: Double? = null,
    val highlightFrac: Double? = null,
    val recordWidth: Int = 0,
    val recordHeight: Int = 0,
    val streams: String = "",
    val orientation: String = "UNKNOWN",
    /** Fresh fix (≤ 2 s old) with ≤ 10 m horizontal accuracy: required before a survey recording starts. */
    val gnssReady: Boolean = false,
    val locationEnabled: Boolean = true,
    val gnssAccM: Float? = null,
    val gnssSats: Int? = null,
    val gnssFixAgeS: Double? = null,
    val speedKmh: Double? = null,
    val mock: Boolean = false,
    val recordingS: Double = 0.0,
    val distanceKm: Double = 0.0,
    /** Road driven for the first time in this session (non-repeating), km. */
    val roadNewKm: Double = 0.0,
    /** Non-repeating road km over all sessions, including this one. */
    val roadTotalKm: Double = 0.0,
    val framesEncoded: Long = 0,
    val framesUnmatched: Long = 0,
    val captureFailures: Long = 0,
    val buffersLost: Long = 0,
    val segment: Int = 0,
    val thermalStatus: Int = 0,
    val headroom: Float? = null,
    val batteryTempC: Float? = null,
    val batteryPct: Int? = null,
    val charging: Boolean = false,
    val freeGb: Double? = null,
    val warnings: List<String> = emptyList(),
    val finalizeProgress: Float? = null,
) {
    val busy: Boolean get() = phase != Phase.IDLE && phase != Phase.ERROR
}

/** Process-wide state shared by the service (writer) and the UI (reader). */
object SurveyController {
    private val _status = MutableStateFlow(SurveyStatus())
    val status: StateFlow<SurveyStatus> = _status.asStateFlow()
    private val _preview = MutableStateFlow<Bitmap?>(null)
    val preview: StateFlow<Bitmap?> = _preview.asStateFlow()

    internal fun update(transform: (SurveyStatus) -> SurveyStatus) {
        while (true) {
            val cur = _status.value
            if (_status.compareAndSet(cur, transform(cur))) return
        }
    }

    internal fun setPreview(b: Bitmap?) {
        _preview.value = b
    }

    /** Leaves the ERROR screen; the service has already cleaned up. */
    fun clearError() {
        update { if (it.phase == Phase.ERROR) SurveyStatus(message = it.error) else it }
    }

    fun prepare(context: Context, paramsJson: String, debugMinFreeMb: Int? = null) {
        val i = Intent(context, SurveyService::class.java).setAction(SurveyService.ACTION_PREPARE)
            .putExtra(SurveyService.EXTRA_PARAMS, paramsJson)
        if (debugMinFreeMb != null) i.putExtra(SurveyService.EXTRA_DEBUG_MIN_FREE_MB, debugMinFreeMb)
        ContextCompat.startForegroundService(context, i)
    }

    fun send(context: Context, action: String, configure: Intent.() -> Unit = {}) {
        if (!status.value.busy && action != SurveyService.ACTION_PREPARE) return
        context.startService(Intent(context, SurveyService::class.java).setAction(action).apply(configure))
    }
}
