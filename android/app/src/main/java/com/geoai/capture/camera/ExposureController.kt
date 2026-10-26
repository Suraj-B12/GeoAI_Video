package com.geoai.capture.camera

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

data class ExposureConfig(
    /** Allowed ground smear during one exposure, millimetres. */
    val blurBudgetMm: Double,
    /** Longest exposure allowed in normal light. */
    val maxExposureNs: Long,
    /** Longest exposure allowed once ISO is at its cap (low light, blur risk flagged). */
    val lowLightMaxExposureNs: Long,
    val maxIso: Int,
    /** Target mean luma of the road region, 0–255 (gamma-encoded). */
    val targetLuma: Double,
    /** Speed assumed when GNSS speed is missing or stale. */
    val unknownSpeedMps: Double,
    /** Below this speed the shutter is computed as if moving at this speed. */
    val minSpeedMps: Double = 1.0,
)

data class SensorLimits(val minExposureNs: Long, val maxExposureNs: Long, val minIso: Int, val maxIso: Int)

data class ExposureSetting(val exposureNs: Long, val iso: Int, val lowLight: Boolean, val reason: String) {
    val product: Double get() = exposureNs.toDouble() * iso
}

/** Luma of one analysed frame plus the exposure/ISO that frame was actually captured with. */
data class LumaSample(val meanLuma: Double, val highlightFraction: Double, val exposureNs: Long, val iso: Int)

/**
 * Shutter priority with automatic ISO, for frames taken from a moving vehicle.
 *
 * Shutter = blur budget ÷ speed (clamped), so ground smear stays under the budget at any speed.
 * ISO is then solved from the required exposure product E = t × ISO. E is updated from measured
 * luma; each update is computed from the sample's *own* t and ISO, so stale samples (pipeline
 * latency of a few frames) repeat the same target instead of compounding.
 */
class ExposureController(private val cfg: ExposureConfig, private val lim: SensorLimits) {

    var current: ExposureSetting? = null
        private set

    private val shutterMinNs: Long get() = lim.minExposureNs
    private val shutterMaxNs: Long get() = max(lim.minExposureNs, min(cfg.maxExposureNs, lim.maxExposureNs))
    private val isoCap: Int get() = max(lim.minIso, min(cfg.maxIso, lim.maxIso))

    /** Exposure time that keeps ground smear ≤ blur budget at [speedMps]. */
    fun shutterFor(speedMps: Double?): Long {
        val v = if (speedMps == null || speedMps.isNaN() || speedMps < 0) cfg.unknownSpeedMps else max(speedMps, cfg.minSpeedMps)
        val t = (cfg.blurBudgetMm / 1000.0 / v * 1e9).toLong()
        return t.coerceIn(shutterMinNs, shutterMaxNs)
    }

    /** Starting point from an auto-exposure metering pass (AE ON, before switching to manual). */
    fun initFromMetering(aeExposureNs: Long, aeIso: Int, speedMps: Double?): ExposureSetting {
        val e = max(1.0, aeExposureNs.toDouble() * aeIso)
        return solve(e, shutterFor(speedMps), "metered").also { current = it }
    }

    /** Re-solve for a new speed only (no new luma); null if the change is too small to apply. */
    fun onSpeed(speedMps: Double?): ExposureSetting? {
        val cur = current ?: return null
        return accept(cur, solve(cur.product, hysteresisShutter(cur, speedMps), "speed"))
    }

    fun update(sample: LumaSample, speedMps: Double?): ExposureSetting? {
        val cur = current ?: return null
        if (sample.exposureNs <= 0 || sample.iso <= 0) return null
        val measured = sample.exposureNs.toDouble() * sample.iso
        val y = sample.meanLuma.coerceIn(1.0, 254.0)
        var ratio = (cfg.targetLuma / y).pow(GAMMA)
        val clipped = sample.highlightFraction > HIGHLIGHT_LIMIT
        if (clipped) ratio = min(ratio, CLIP_STEP)
        val inDeadband = !clipped && abs(ln(ratio) / LN2) < DEADBAND_EV
        val target = if (inDeadband) cur.product
        else measured * ratio.pow(DAMPING).coerceIn(1.0 / MAX_STEP, MAX_STEP)
        return accept(cur, solve(target, hysteresisShutter(cur, speedMps), if (inDeadband) "speed" else "luma"))
    }

    private fun hysteresisShutter(cur: ExposureSetting, speedMps: Double?): Long {
        val wanted = shutterFor(speedMps)
        if (cur.lowLight) return wanted
        return if (abs(wanted - cur.exposureNs) < SHUTTER_HYSTERESIS * cur.exposureNs) cur.exposureNs else wanted
    }

    private fun accept(cur: ExposureSetting, next: ExposureSetting): ExposureSetting? {
        val dt = abs(next.exposureNs - cur.exposureNs).toDouble() / cur.exposureNs
        val di = abs(next.iso - cur.iso).toDouble() / cur.iso
        return if (dt > MIN_CHANGE || di > MIN_CHANGE || next.lowLight != cur.lowLight) {
            current = next
            next
        } else null
    }

    private fun solve(product: Double, shutterNs: Long, reason: String): ExposureSetting {
        var t = shutterNs
        var iso = product / t
        var lowLight = false
        if (iso > isoCap) {
            iso = isoCap.toDouble()
            val lowLightMax = max(t, min(cfg.lowLightMaxExposureNs, lim.maxExposureNs))
            t = (product / iso).toLong().coerceIn(t, lowLightMax)
            lowLight = true
        } else if (iso < lim.minIso) {
            iso = lim.minIso.toDouble()
            t = (product / iso).toLong().coerceIn(shutterMinNs, t)
        }
        return ExposureSetting(t, iso.roundToInt().coerceIn(lim.minIso, isoCap), lowLight, reason)
    }

    companion object {
        /** sRGB-like encoding: linear brightness ∝ luma^2.2. */
        const val GAMMA = 2.2
        /** Fraction of the correction applied per update (damps oscillation). */
        const val DAMPING = 0.6
        /** Largest single-update change of the exposure product (±1 EV). */
        const val MAX_STEP = 2.0
        const val DEADBAND_EV = 0.2
        /** Share of ROI pixels above 250 that counts as clipping. */
        const val HIGHLIGHT_LIMIT = 0.02
        const val CLIP_STEP = 0.8
        const val SHUTTER_HYSTERESIS = 0.15
        const val MIN_CHANGE = 0.03
        private val LN2 = ln(2.0)
    }
}
