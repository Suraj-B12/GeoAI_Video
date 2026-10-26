package com.geoai.capture.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.atan
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.tan

/**
 * Geometry and decisions of [FocusPlanner], with the pilot S21's own camera characteristics.
 *
 * Expected numbers marked "numpy" come from an independent calculation with the PIPELINE's camera model
 * (geoai.geometry: K matrix, rotation from gravity via cam_from_device / vehicle_from_camera, pixel_to_ground and
 * ground_to_pixel), sampling the same rows; the zero-roll test also re-derives everything with plane trigonometry.
 */
class FocusPlannerTest {
    // Galaxy S21 main camera as recorded in session.json (26 Oct 2026): active array 4032 x 3024,
    // intrinsics fy 3053.8, cy 1529.7, f = 5.4 mm, f/1.8, sensor 7.2576 mm wide, min focus 10 dpt.
    private val s21 = FocusPlanner.Lens(
        activeWidth = 4032, activeHeight = 3024, fyPx = 3053.831787109375, cxPx = 2030.2230224609375,
        cyPx = 1529.724609375, focalMm = 5.4, aperture = 1.8, outputPixelUm = 7.2576 * 1000 / 4032 * 4032 / 3840,
        minFocusDiopters = 10.0,
    )
    private val up90 = FocusPlanner.sensorUp(90)!!
    /** Session 1's gravity at the start of recording: 25.08 degrees down, rolled -6.3 degrees. */
    private val session1 = floatArrayOf(8.8279390335083f, 0.9817163944244385f, 4.156202793121338f)

    /**
     * Gravity sensor reading for a back camera tilted [depressionDeg] below the horizon and rolled [rollDeg]
     * (pipeline convention), UPRIGHT landscape, sensor orientation 90.
     */
    private fun gravity(depressionDeg: Double, rollDeg: Double = 0.0): FloatArray {
        val a = Math.toRadians(depressionDeg)
        val r = Math.toRadians(rollDeg)
        return floatArrayOf((9.81 * cos(a) * cos(r)).toFloat(), (-9.81 * cos(a) * sin(r)).toFloat(), (9.81 * sin(a)).toFloat())
    }

    private fun blurPx(diopters: Double, depthM: Double) = FocusPlanner.blurUm(s21, diopters, depthM) / s21.outputPixelUm

    @Test fun segmentReachIsLengthPlusWindow() {
        // Pipeline SegmentConfig defaults: length_m 5.0, window_m 1.0.
        assertEquals(5.0, FocusPlanner.SEGMENT_LENGTH_M, 0.0)
        assertEquals(1.0, FocusPlanner.SEGMENT_WINDOW_M, 0.0)
        assertEquals(6.0, FocusPlanner.SEGMENT_REACH_M, 0.0)
    }

    @Test fun zeroRollMountMatchesPlaneTrigonometry() {
        // 23.0 degrees down, 1.35 m: the centre column lies in the vertical plane through the optical axis.
        val t = Math.toRadians(23.0)
        val h = 1.35
        // Angles below the axis of the last and first visible rows (active rows 378..2645 for 3840 x 2160).
        val bottom = atan((2645 + 0.5 - s21.cyPx) / s21.fyPx)
        val top = atan((378 + 0.5 - s21.cyPx) / s21.fyPx)
        val near = h / tan(t + bottom)
        val farView = h / tan(t + top)
        val farRoad = near + 6.0
        // Depth along the optical axis = slant distance x cos(angle off the axis).
        val zNear = h / sin(t + bottom) * cos(bottom)
        val zFar = hypot(h, farRoad) * cos(t - atan(h / farRoad))
        val target = (1 / zNear + 1 / zFar) / 2
        val edge = 5.4 / 1.8 * 5.4 * (1 / zNear - target) / s21.outputPixelUm
        // The trigonometry agrees with numpy: road 1.4441-7.4441 m, view to 32.9705 m, depth 1.8568-7.3798 m,
        // 0.33703 dpt, 1.7274 px.
        assertEquals(1.4441, near, 1e-4)
        assertEquals(32.9705, farView, 1e-4)
        assertEquals(1.8568, zNear, 1e-4)
        assertEquals(7.3798, zFar, 1e-4)
        assertEquals(0.33703, target, 1e-5)
        assertEquals(1.7274, edge, 1e-4)

        val v = FocusPlanner.roadView(s21, gravity(23.0), up90, h, 3840, 2160)!!
        assertEquals(23.0, v.tiltDeg, 1e-5)
        assertEquals(near, v.nearM!!, 1e-4)
        assertEquals(farView, v.farM!!, 1e-3)
        assertTrue(v.segmentFits)
        val p = FocusPlanner.plan(s21, gravity(23.0), up90, h, 3840, 2160)!!
        assertEquals(v, p.view)
        assertEquals(near, p.nearRoadM, 1e-4)
        assertEquals(farRoad, p.farRoadM, 1e-4)
        assertEquals(zNear, p.nearDepthM, 1e-4)
        assertEquals(zFar, p.farDepthM, 1e-4)
        assertEquals(target, p.targetDiopters, 1e-5)
        assertEquals(edge, p.edgeBlurPx, 1e-4)
        // Metering rows: from the first sampled row within near + 6 m down to the bottom margin (numpy: 843-2556).
        assertEquals(843, p.roiTop)
        assertEquals(2556, p.roiBottom)
    }

    @Test fun pilotMountFocusesOnTheSegmentRoad() {
        val p = FocusPlanner.plan(s21, session1, up90, 1.35, 3840, 2160)
        assertNotNull(p)
        p!!
        // numpy: tilt 25.0757 deg; road 1.3484 m (bottom row) to 17.0154 m (top row); focus covers 1.3484-7.3484 m
        // ahead = depth 1.7935-7.2280 m, target 0.34796 dpt, edge blur 1.7967 px; metering rows 728-2556.
        assertEquals(25.0757, p.view.tiltDeg, 1e-3)
        assertEquals(1.3484, p.view.nearM!!, 2e-4)
        assertEquals(17.0154, p.view.farM!!, 2e-3)
        assertTrue(p.view.segmentFits)
        assertEquals(1.3484, p.nearRoadM, 2e-4)
        assertEquals(7.3484, p.farRoadM, 2e-4)
        assertEquals(1.7935, p.nearDepthM, 2e-4)
        assertEquals(7.2280, p.farDepthM, 2e-4)
        assertEquals(0.34796, p.targetDiopters, 2e-5)
        assertEquals(1.7967, p.edgeBlurPx, 2e-4)
        // The dioptre midpoint blurs both ends of the segment road equally.
        assertEquals(blurPx(p.targetDiopters, p.nearDepthM), blurPx(p.targetDiopters, p.farDepthM), 1e-9)
        assertEquals(728, p.roiTop)
        assertEquals(2556, p.roiBottom)
        assertEquals((728 - 378) / 2268.0, p.roiTopFrac, 1e-12)
        assertEquals((2556 - 378) / 2268.0, p.roiBottomFrac, 1e-12)
    }

    @Test fun segmentFocusBeatsTheFormer4mFocusOnTheSegment() {
        // The S21 mount as the pipeline measured it: 23.0 degrees down, roll -4.4 degrees, 1.35 m.
        val p = FocusPlanner.plan(s21, gravity(23.0, -4.4), up90, 1.35, 3840, 2160)!!
        // numpy: road 1.4469-7.4469 m ahead (view to 32.2038 m), depth 1.8593-7.3824 m, 0.33664 dpt, 1.7244 px.
        assertEquals(1.4469, p.nearRoadM, 2e-4)
        assertEquals(7.4469, p.farRoadM, 2e-4)
        assertEquals(32.2038, p.view.farM!!, 2e-3)
        assertEquals(1.8593, p.nearDepthM, 2e-4)
        assertEquals(7.3824, p.farDepthM, 2e-4)
        assertEquals(0.33664, p.targetDiopters, 2e-5)
        assertEquals(1.7244, p.edgeBlurPx, 2e-4)
        // The former plan (road up to 4 m, bottom 4 % excluded) focused at 0.37765 dpt (numpy replica of the old
        // code, which reproduces its 0.3885 dpt for session 1): sharper on 1.5-4 m (1.20 px) but 2.08 px at the
        // segment's far edge and 1.37 px at its near edge.
        val former = 0.37765
        assertEquals(2.0759, blurPx(former, p.farDepthM), 2e-4)
        assertEquals(1.3729, blurPx(former, p.nearDepthM), 2e-4)
        assertTrue(p.edgeBlurPx < blurPx(former, p.farDepthM) - 0.3)
    }

    @Test fun session2FocusWasOutsideTheRoad() {
        // The 0.80 dpt that autofocus chose in session 2 blurs the segment's far road by about 5.7 output pixels
        // (numpy 5.6756) and its near road by 2.1 (2.0823); the geometric target by at most 1.8.
        val p = FocusPlanner.plan(s21, session1, up90, 1.35, 3840, 2160)!!
        assertEquals(5.6756, blurPx(0.8005, p.farDepthM), 2e-3)
        assertEquals(2.0823, blurPx(0.8005, p.nearDepthM), 2e-3)
        assertTrue(p.edgeBlurPx < 1.8)
    }

    @Test fun steepExteriorMountCannotFitASegment() {
        // 1.6 m high, 45 degrees down: the frame shows only 0.74-3.54 m of road (numpy 0.7437-3.5362), less than
        // the 6 m a segment frame needs. Focus then covers all visible road.
        val v = FocusPlanner.roadView(s21, gravity(45.0), up90, 1.6, 3840, 2160)!!
        assertEquals(45.0, v.tiltDeg, 1e-4)
        assertEquals(0.7437, v.nearM!!, 2e-4)
        assertEquals(3.5362, v.farM!!, 2e-4)
        assertFalse(v.segmentFits)
        val p = FocusPlanner.plan(s21, gravity(45.0), up90, 1.6, 3840, 2160)!!
        assertEquals(v.farM!!, p.farRoadM, 1e-12)
        // numpy: depth 1.6572-3.6319 m, 0.43938 dpt, 1.4060 px; whole frame is metered road.
        assertEquals(1.6572, p.nearDepthM, 2e-4)
        assertEquals(3.6319, p.farDepthM, 2e-4)
        assertEquals(0.43938, p.targetDiopters, 2e-5)
        assertEquals(1.4060, p.edgeBlurPx, 2e-4)
        assertEquals(0.0, p.roiTopFrac, 0.0)
    }

    @Test fun horizonInViewCapsTheFarRoad() {
        // 5 degrees down: the top rows look above the horizon, so the view reaches MAX_ROAD_M; numpy near 2.8858 m.
        val v = FocusPlanner.roadView(s21, gravity(5.0), up90, 1.35, 3840, 2160)!!
        assertEquals(2.8858, v.nearM!!, 2e-4)
        assertEquals(FocusPlanner.MAX_ROAD_M, v.farM!!, 0.0)
        assertTrue(v.segmentFits)
        val p = FocusPlanner.plan(s21, gravity(5.0), up90, 1.35, 3840, 2160)!!
        assertEquals(2.8858 + 6.0, p.farRoadM, 2e-4)
        assertEquals(1728, p.roiTop)
    }

    @Test fun cameraLookingUpOrNoGravityGivesNoPlan() {
        val up = floatArrayOf(8.8279390335083f, 0f, -4.156202793121338f)
        assertNull(FocusPlanner.plan(s21, up, up90, 1.35, 3840, 2160))
        // The tilt is still known (looking up 25.2 degrees), but no road is in view and no segment fits.
        val v = FocusPlanner.roadView(s21, up, up90, 1.35, 3840, 2160)!!
        assertEquals(-25.21, v.tiltDeg, 0.01)
        assertNull(v.nearM)
        assertNull(v.farM)
        assertFalse(v.segmentFits)
        assertNull(FocusPlanner.plan(s21, null, up90, 1.35, 3840, 2160))
        assertNull(FocusPlanner.roadView(s21, null, up90, 1.35, 3840, 2160))
        assertNull(FocusPlanner.plan(s21, gravity(25.0), up90, 0.1, 3840, 2160))
        assertNull(FocusPlanner.roadView(s21, gravity(25.0), up90, 0.1, 3840, 2160))
        assertNull(FocusPlanner.plan(s21, floatArrayOf(0.1f, 0.1f, 0.1f), up90, 1.35, 3840, 2160))
        // Straight down: "ahead" is undefined.
        assertNull(FocusPlanner.roadView(s21, floatArrayOf(0f, 0f, 9.81f), up90, 1.35, 3840, 2160))
    }

    @Test fun upsideDownMountPutsTheRoadAtTheTopRows() {
        val g = floatArrayOf(-8.8279390335083f, 0f, 4.156202793121338f)
        val p = FocusPlanner.plan(s21, g, up90, 1.35, 3840, 2160)!!
        // numpy: nearest road 1.3098 m at the top row; near + 6 m reached at row 2333 (frac 0.8642).
        assertEquals(1.3098, p.nearRoadM, 2e-4)
        assertTrue("road rows ${p.roiTopFrac}-${p.roiBottomFrac}", p.roiTopFrac < 0.01 && p.roiBottomFrac in 0.86..0.87)
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
