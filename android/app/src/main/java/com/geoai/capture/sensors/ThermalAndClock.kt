package com.geoai.capture.sensors

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Handler
import android.os.PowerManager
import androidx.core.content.ContextCompat
import com.geoai.capture.io.AsyncLineWriter
import com.geoai.capture.util.Clocks
import com.geoai.capture.util.Fmt
import java.util.concurrent.Executor

/**
 * Thermal status (event-driven) plus a 5 s sample of thermal headroom and battery state.
 * Headroom is NaN on devices that don't support it; a status stuck at 0 can also mean "unsupported".
 */
class ThermalRecorder(
    private val context: Context,
    private val handler: Handler,
    private val executor: Executor,
    private val out: AsyncLineWriter,
    private val onStatus: (Int) -> Unit,
) {
    data class Sample(val status: Int, val headroom: Float?, val batteryTempC: Float?, val batteryPct: Int?, val charging: Boolean)

    private val pm = context.getSystemService(PowerManager::class.java)
    @Volatile var latest = Sample(0, null, null, null, false)
        private set
    @Volatile var maxStatus = 0
        private set
    private var running = false

    private val statusListener = PowerManager.OnThermalStatusChangedListener { status ->
        if (status > maxStatus) maxStatus = status
        sample()
        onStatus(status)
    }

    private val tick = object : Runnable {
        override fun run() {
            if (!running) return
            sample()
            handler.postDelayed(this, 5_000)
        }
    }

    fun start() {
        if (running) return
        running = true
        pm.addThermalStatusListener(executor, statusListener)
        handler.post(tick)
    }

    fun stop() {
        if (!running) return
        running = false
        runCatching { pm.removeThermalStatusListener(statusListener) }
        handler.removeCallbacks(tick)
        sample()
    }

    private fun sample() {
        val status = pm.currentThermalStatus
        if (status > maxStatus) maxStatus = status
        val headroom = runCatching { pm.getThermalHeadroom(10) }.getOrNull()?.takeIf { !it.isNaN() }
        val battery = ContextCompat.registerReceiver(
            context, null, IntentFilter(Intent.ACTION_BATTERY_CHANGED), ContextCompat.RECEIVER_NOT_EXPORTED,
        )
        val tempTenths = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val s = Sample(
            status,
            headroom,
            if (tempTenths != Int.MIN_VALUE) tempTenths / 10f else null,
            if (level >= 0 && scale > 0) level * 100 / scale else null,
            plugged,
        )
        latest = s
        out.write(
            Fmt.row(Clocks.elapsedNs().toString(), status.toString(), Fmt.opt(s.headroom), Fmt.opt(s.batteryTempC), Fmt.opt(s.batteryPct), Fmt.bool(plugged)),
        )
    }

    companion object {
        const val HEADER = "t_ns,thermal_status,headroom_10s,battery_temp_c,battery_pct,charging"
        const val STATUS_MODERATE = PowerManager.THERMAL_STATUS_MODERATE
        const val STATUS_SEVERE = PowerManager.THERMAL_STATUS_SEVERE
        const val STATUS_CRITICAL = PowerManager.THERMAL_STATUS_CRITICAL
    }
}

/** clock_sync.csv: elapsedRealtime/monotonic/wall triples every 10 s and on demand. */
class ClockSyncRecorder(private val handler: Handler, private val out: AsyncLineWriter) {
    private var running = false
    @Volatile var latest: Clocks.ClockPair = Clocks.bestPair()
        private set

    private val tick = object : Runnable {
        override fun run() {
            if (!running) return
            sampleNow()
            handler.postDelayed(this, 10_000)
        }
    }

    fun start() {
        if (running) return
        running = true
        handler.post(tick)
    }

    fun stop() {
        running = false
        handler.removeCallbacks(tick)
        sampleNow()
    }

    fun sampleNow(): Clocks.ClockPair {
        val p = Clocks.bestPair()
        latest = p
        out.write(Fmt.row(p.elapsedBeforeNs.toString(), p.monoNs.toString(), p.elapsedAfterNs.toString(), p.wallMs.toString()))
        return p
    }

    companion object {
        const val HEADER = "t_ns_before,mono_ns,t_ns_after,wall_ms"
    }
}

/**
 * Is the recorded (sensor-native, unrotated) image upright? The back camera's SENSOR_ORIENTATION
 * tells which device axis the sensor's first row faces; the gravity sensor reading points up.
 */
object OrientationCheck {
    enum class Verdict { UPRIGHT, UPSIDE_DOWN, PORTRAIT, UNKNOWN }

    fun verdict(sensorOrientationDeg: Int, gravity: FloatArray?): Verdict {
        val g = gravity ?: return Verdict.UNKNOWN
        val (ux, uy) = when (((sensorOrientationDeg % 360) + 360) % 360) {
            0 -> 0f to 1f
            90 -> 1f to 0f
            180 -> 0f to -1f
            270 -> -1f to 0f
            else -> return Verdict.UNKNOWN
        }
        val along = g[0] * ux + g[1] * uy           // component along the sensor's "up"
        val across = g[0] * uy - g[1] * ux          // component along the sensor's rows
        return when {
            kotlin.math.abs(across) > kotlin.math.abs(along) && kotlin.math.abs(across) > 3f -> Verdict.PORTRAIT
            along > 2f -> Verdict.UPRIGHT
            along < -2f -> Verdict.UPSIDE_DOWN
            else -> Verdict.UNKNOWN
        }
    }
}
