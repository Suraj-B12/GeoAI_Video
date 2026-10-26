package com.geoai.capture.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.pow

class ExposureControllerTest {

    private val limits = SensorLimits(minExposureNs = 20_000L, maxExposureNs = 100_000_000L, minIso = 50, maxIso = 3200)
    private val cfg = ExposureConfig(
        blurBudgetMm = 2.5,
        maxExposureNs = 1_000_000L,          // 1/1000 s
        lowLightMaxExposureNs = 2_000_000L,  // 1/500 s
        maxIso = 1600,
        targetLuma = 110.0,
        unknownSpeedMps = 50 / 3.6,
    )

    @Test fun shutterKeepsSmearWithinBudget() {
        val c = ExposureController(cfg, limits)
        val v = 40 / 3.6
        val t = c.shutterFor(v)
        // 2.5 mm at 11.11 m/s = 225 µs
        assertEquals(225_000.0, t.toDouble(), 1_000.0)
        assertTrue("smear ≤ budget", v * t / 1e9 * 1000 <= 2.5 + 1e-6)
    }

    @Test fun shutterIsCappedWhenSlowAndUsesFallbackSpeedWhenUnknown() {
        val c = ExposureController(cfg, limits)
        assertEquals(1_000_000L, c.shutterFor(0.5))          // slow → capped at 1/1000 s
        assertEquals(180_000.0, c.shutterFor(null).toDouble(), 1_000.0) // 50 km/h assumed
    }

    /** Scene model: luma = 255 · min(1, (E·k))^(1/2.2) with E = exposure_s × ISO. */
    private fun luma(e: ExposureSetting, k: Double): Double = 255.0 * (e.exposureNs / 1e9 * e.iso * k).coerceAtMost(1.0).pow(1 / 2.2)

    private fun converge(k: Double, speed: Double): ExposureSetting {
        val c = ExposureController(cfg, limits)
        var s = c.initFromMetering(aeExposureNs = 10_000_000L, aeIso = 100, speedMps = speed)
        repeat(40) {
            val next = c.update(LumaSample(luma(s, k), 0.0, s.exposureNs, s.iso), speed)
            if (next != null) s = next
        }
        return s
    }

    @Test fun convergesToTargetLumaInDaylight() {
        val s = converge(k = 2.0, speed = 11.0)
        assertTrue("luma near target, got ${luma(s, 2.0)}", abs(luma(s, 2.0) - 110.0) < 12.0)
        assertTrue("short shutter kept", s.exposureNs <= 250_000L)
        assertTrue(!s.lowLight)
    }

    @Test fun lowLightRaisesIsoToCapThenExtendsExposure() {
        val s = converge(k = 0.002, speed = 11.0)
        assertEquals(1600, s.iso)
        assertTrue(s.lowLight)
        assertTrue("exposure within low-light cap", s.exposureNs <= 2_000_000L)
    }

    @Test fun clippedHighlightsPullExposureDown() {
        val c = ExposureController(cfg, limits)
        val s = c.initFromMetering(1_000_000L, 400, 11.0)
        val next = c.update(LumaSample(meanLuma = 110.0, highlightFraction = 0.1, exposureNs = s.exposureNs, iso = s.iso), 11.0)
        assertNotNull(next)
        assertTrue(next!!.product < s.product)
    }

    @Test fun staleSamplesDoNotCompound() {
        val c = ExposureController(cfg, limits)
        val s0 = c.initFromMetering(1_000_000L, 200, 11.0)
        val dark = LumaSample(meanLuma = 40.0, highlightFraction = 0.0, exposureNs = s0.exposureNs, iso = s0.iso)
        val s1 = c.update(dark, 11.0)
        assertNotNull(s1)
        // The same (stale) measurement again must not push exposure further.
        assertNull(c.update(dark, 11.0))
    }
}
