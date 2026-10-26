package com.geoai.capture.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/** Geometry and decisions of [FocusPlanner], with the pilot S21's own camera characteristics. */
class FocusPlannerTest {
    // Galaxy S21 main camera as recorded in session.json (26 Oct 2026): active array 4032 x 3024,
    // intrinsics fy 3053.8, cy 1529.7, f = 5.4 mm, f/1.8, sensor 7.2576 mm wide, min focus 10 dpt.
    private val s21 = FocusPlanner.Lens(
        activeWidth = 4032, activeHeight = 3024, fyPx = 3053.831787109375, cxPx = 2030.2230224609375,
        cyPx = 1529.724609375, focalMm = 5.4, aperture = 1.8, outputPixelUm = 7.2576 * 1000 / 4032 * 4032 / 3840,
        minFocusDiopters = 10.0,
    )
    private val up90 = FocusPlanner.sensorUp(90)!!

    /** Gravity sensor reading for a back camera tilted [depressionDeg] below the horizon, UPRIGHT landscape. */
    private fun gravity(depressionDeg: Double): FloatArray {
        val a = Math.toRadians(depressionDeg)
        return floatArrayOf((9.81 * cos(a)).toFloat(), 0f, (9.81 * sin(a)).toFloat())
    }

    @Test fun pilotMountFocusesOnTheRoadItSees() {
        // Session 1's gravity at the start of recording (25.1 degrees down), mount height 1.35 m.
        val g = floatArrayOf(8.8279390335083f, 0.9817163944244385f, 4.156202793121338f)
        val p = FocusPlanner.plan(s21, g, up90, 1.35, 3840, 2160)
        assertNotNull(p)
        p!!
        assertEquals(25.08, p.depressionDeg, 0.05)
        // Independent calculation (numpy): road depth along the axis 1.861-4.177 m, target 0.3885 dpt.
        assertEquals(1.861, p.nearDepthM, 0.01)
        assertEquals(4.177, p.farDepthM, 0.02)
        assertEquals(0.3885, p.targetDiopters, 0.002)
        assertEquals(1.28, p.edgeBlurPx, 0.05)
        assertTrue("road rows ${p.roiTop}-${p.roiBottom}", p.roiTop in 1180..1200 && p.roiBottom in 2550..2560)
        assertTrue(p.roiTopFrac in 0.35..0.37 && p.roiBottomFrac > 0.95)
    }

    @Test fun session2FocusWasOutsideTheRoad() {
        // The 0.80 dpt that autofocus chose in session 2 blurs the far road by about 4.8 output pixels,
        // the geometric target by at most 1.3.
        val g = floatArrayOf(8.8279390335083f, 0.9817163944244385f, 4.156202793121338f)
        val p = FocusPlanner.plan(s21, g, up90, 1.35, 3840, 2160)!!
        val farBlurAf = FocusPlanner.blurUm(s21, 0.8005, p.farDepthM) / s21.outputPixelUm
        assertEquals(4.81, farBlurAf, 0.05)
        assertTrue(p.edgeBlurPx < 1.5)
    }

    @Test fun steepExteriorMountSeesOnlyNearRoad() {
        // The plan's recommended mount: 1.6 m high, 45 degrees down. All visible road is within 4 m.
        val p = FocusPlanner.plan(s21, gravity(45.0), up90, 1.6, 3840, 2160)!!
        assertEquals(45.0, p.depressionDeg, 0.01)
        assertTrue(p.nearDepthM < p.farDepthM)
        // Bottom of the visible frame: 45 + atan((2646 - 1529.7) / 3053.8) degrees down.
        assertTrue(p.nearDepthM in 1.5..1.8)
        assertTrue("whole frame is road: ${p.roiTopFrac}", p.roiTopFrac < 0.02)
        assertTrue(p.edgeBlurPx < 2.0)
    }

    @Test fun cameraLookingUpOrNoGravityGivesNoPlan() {
        val up = floatArrayOf(8.8279390335083f, 0f, -4.156202793121338f)
        assertNull(FocusPlanner.plan(s21, up, up90, 1.35, 3840, 2160))
        assertNull(FocusPlanner.plan(s21, null, up90, 1.35, 3840, 2160))
        assertNull(FocusPlanner.plan(s21, gravity(25.0), up90, 0.1, 3840, 2160))
        assertNull(FocusPlanner.plan(s21, floatArrayOf(0.1f, 0.1f, 0.1f), up90, 1.35, 3840, 2160))
    }

    @Test fun upsideDownMountPutsTheRoadAtTheTopRows() {
        val g = floatArrayOf(-8.8279390335083f, 0f, 4.156202793121338f)
        val p = FocusPlanner.plan(s21, g, up90, 1.35, 3840, 2160)!!
        assertTrue("road rows ${p.roiTopFrac}-${p.roiBottomFrac}", p.roiTopFrac < 0.01 && p.roiBottomFrac < 0.7)
    }

    @Test fun visibleRowsAre16by9CentreCrop() {
        assertEquals(378 to 2646, FocusPlanner.visibleRows(4032, 3024, 3840, 2160))
        assertEquals(0 to 3024, FocusPlanner.visibleRows(4032, 3024, 4032, 3024))
    }

    @Test fun blurFollowsTheThinLensFormula() {
        // A = 5.4 / 1.8 = 3 mm; 3 mm x 5.4 mm x 0.5 dpt = 8.1 micrometres.
        assertEquals(8.1, FocusPlanner.blurUm(s21, 0.5, 1.0), 1e-9)
        assertEquals(0.0, FocusPlanner.blurUm(s21, 0.5, 2.0), 1e-9)
    }

    @Test fun sweepIncludesTargetAndStaysInLensRange() {
        assertEquals(listOf(0.0, 0.1, 0.4, 0.7, 1.0), FocusPlanner.sweep(0.4, 10.0).map { Math.round(it * 10) / 10.0 })
        assertEquals(listOf(0.0, 0.3), FocusPlanner.sweep(0.0, 0.3).map { Math.round(it * 10) / 10.0 })
        assertTrue(FocusPlanner.sweep(0.39, 10.0).any { it == 0.39 })
    }

    @Test fun decideKeepsGeometryUnlessTheSweepClearlyDisagrees() {
        val t = 0.4
        // Flat: no information at analysis resolution.
        assertEquals("geometry_sweep_flat", FocusPlanner.decide(t, listOf(0.0 to 100.0, 0.4 to 102.0, 0.7 to 101.0)).method)
        // Small gain next to the target: noise, keep geometry.
        val near = FocusPlanner.decide(t, listOf(0.1 to 100.0, 0.4 to 120.0, 0.7 to 130.0, 1.0 to 90.0))
        assertEquals("geometry_sweep_confirmed", near.method)
        assertEquals(t, near.diopters, 1e-9)
        // A clearly sharper position far from the target (mis-set height or calibration error): follow it.
        val far = FocusPlanner.decide(t, listOf(0.0 to 80.0, 0.4 to 100.0, 0.7 to 105.0, 1.0 to 140.0))
        assertEquals("sweep_override", far.method)
        assertEquals(1.0, far.diopters, 1e-9)
        // Too few valid frames: geometry alone.
        assertEquals("geometry", FocusPlanner.decide(t, listOf(0.4 to 100.0, 0.7 to Double.NaN)).method)
    }
}
