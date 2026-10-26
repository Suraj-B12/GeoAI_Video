package com.geoai.capture.util

import android.os.SystemClock
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * The session clock is SystemClock.elapsedRealtimeNanos() (CLOCK_BOOTTIME). Camera results on
 * REALTIME-source devices, Location fixes and SensorEvents all use it. System.nanoTime() is
 * CLOCK_MONOTONIC, the default clock of video-encoder surfaces. The wall clock is display-only.
 */
object Clocks {
    fun elapsedNs(): Long = SystemClock.elapsedRealtimeNanos()
    fun monoNs(): Long = System.nanoTime()
    fun wallMs(): Long = System.currentTimeMillis()

    data class ClockPair(val elapsedBeforeNs: Long, val monoNs: Long, val elapsedAfterNs: Long, val wallMs: Long) {
        val elapsedMidNs: Long get() = elapsedBeforeNs + (elapsedAfterNs - elapsedBeforeNs) / 2
        /** elapsedRealtime − monotonic: the total deep-sleep time since boot (≥ 0). */
        val elapsedMinusMonoNs: Long get() = elapsedMidNs - monoNs
    }

    fun samplePair(): ClockPair {
        val a = elapsedNs()
        val m = monoNs()
        val b = elapsedNs()
        return ClockPair(a, m, b, wallMs())
    }

    /** Best of [n] back-to-back samples: the one with the tightest elapsed sandwich. */
    fun bestPair(n: Int = 7): ClockPair =
        (1..n).map { samplePair() }.minBy { it.elapsedAfterNs - it.elapsedBeforeNs }

    private fun utcFormat(pattern: String) =
        SimpleDateFormat(pattern, Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }

    fun compactUtc(ms: Long): String = utcFormat("yyyyMMdd'T'HHmmss'Z'").format(Date(ms))
    fun isoUtc(ms: Long): String = utcFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").format(Date(ms))
}
