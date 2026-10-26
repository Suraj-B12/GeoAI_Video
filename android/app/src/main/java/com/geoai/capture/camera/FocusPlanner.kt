package com.geoai.capture.camera

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Where to focus so the ROAD is sharp, computed from the mount geometry instead of left to autofocus.
 *
 * Autofocus locks on whatever has contrast in its region: a vehicle ahead, a kerb, the bonnet. On the first
 * S21 sessions it locked at 0.46 and 0.80 dioptres for the same mount; 0.80 (1.25 m) puts the far half of the
 * analysed road outside the depth of field. With a calibrated lens the right distance follows from geometry:
 *
 * 1. Gravity gives the camera's tilt below the horizon; each image row is a ray at its own depression. Along the
 *    centre column the frame shows road from `near` (the bottom row) to `far` (the top row, or the horizon).
 * 2. Segment mode (docs/pipeline/13-segment-frames.md) classifies one frame per [SEGMENT_LENGTH_M] of road, taken
 *    while the segment's near edge is between `near` and `near + `[SEGMENT_WINDOW_M]. The road a chosen frame must
 *    render sharply is therefore `near .. near + `[SEGMENT_REACH_M] ahead, cut at `far`. Crop mode analyses at most
 *    6 m ahead (pipeline geometry.max_range_m), inside the same range, so one plan serves both.
 * 3. A road point d metres ahead lies at depth z = d cos(t) + h sin(t) along the optical axis (t = tilt, h = mount
 *    height), whatever its column and the camera's roll: the depth range is that of the planned road ahead.
 * 4. The focus that gives equal, smallest defocus at the nearest and farthest road depth is the dioptre midpoint
 *    D = (1/z_near + 1/z_far) / 2. Blur diameter on the sensor at depth z: A * f * |D - 1/z|, A = f / N.
 *
 * S21 pilot mount (23.0 deg, roll -4.4 deg, 1.35 m; f 5.4 mm, f/1.8, 1.89 um output pixel at 3840 x 2160): road
 * 1.45-7.45 m ahead, depth 1.86-7.38 m, focus 0.337 dpt, edge blur 1.72 output px (0.60 px after the VLM's resize
 * to 1344 x 756). The former 4 m limit focused at 0.378 dpt: 1.20 px on its own 1.5-4 m, but 2.08 px at a
 * segment's far edge (independent calculation with the pipeline camera model; see FocusPlannerTest).
 *
 * Pure Kotlin (no Android types) so it is unit-tested on the JVM. Coordinates are sensor active-array pixels.
 */
object FocusPlanner {
    /** Road length classified per frame in segment mode, m (pipeline SegmentConfig.length_m). */
    const val SEGMENT_LENGTH_M = 5.0
    /** Candidate window: a segment's near edge is at most this far beyond the nearest visible road, m (SegmentConfig.window_m). */
    const val SEGMENT_WINDOW_M = 1.0
    /**
     * Farthest road a segment frame must show sharply, measured from the nearest visible road: the segment's far
     * edge lies [SEGMENT_LENGTH_M] beyond its near edge, which is at most [SEGMENT_WINDOW_M] beyond that road.
     */
    const val SEGMENT_REACH_M = SEGMENT_LENGTH_M + SEGMENT_WINDOW_M
    /** Farthest road reported, m ahead, as in the pipeline: with the horizon in the frame the view reaches this. */
    const val MAX_ROAD_M = 60.0
    /** Bottom rows that may show the bonnet or bumper rather than road: kept out of the AF/AE metering region. */
    const val BOTTOM_MARGIN_FRAC = 0.04
    /** Left and right edges of the road region (fractions of the width): roadside objects stay out. */
    const val ROI_LEFT = 0.20
    const val ROI_RIGHT = 0.80
    /** At least this share of the visible rows must be planned road for a plan. */
    const val MIN_ROAD_FRAC = 0.10
    /** Contrast sweep: offsets around the target, dioptres. */
    val SWEEP_OFFSETS = doubleArrayOf(-0.6, -0.3, 0.0, 0.3, 0.6)
    /** A sweep result replaces the geometric target only if its score is this much higher ... */
    const val OVERRIDE_GAIN = 0.15
    /** ... and it lies at least this far (dioptres) from the target. */
    const val OVERRIDE_MIN_DIST = 0.25
    /** Scores within this relative spread carry no focus information (texture too fine for the analysis stream). */
    const val FLAT_SPREAD = 0.05

    /** The back camera's optical axis in device axes. */
    private val AXIS = doubleArrayOf(0.0, 0.0, -1.0)

    data class Lens(
        val activeWidth: Int,
        val activeHeight: Int,
        /** Focal length in active-array pixels (vertical) and principal point. */
        val fyPx: Double,
        val cxPx: Double,
        val cyPx: Double,
        val focalMm: Double,
        val aperture: Double,
        /** Size of one output (recorded) pixel on the sensor, micrometres. */
        val outputPixelUm: Double,
        /** LENS_INFO_MINIMUM_FOCUS_DISTANCE, dioptres. */
        val minFocusDiopters: Double,
    )

    /**
     * What the recorded frame shows of a flat road along its centre column (the principal point's column), from
     * gravity and the mount height: the pipeline's near_m and far_m (docs/pipeline/13 §2), for the live screen.
     */
    data class RoadView(
        /** Optical axis below the horizontal, degrees (negative: looking up). */
        val tiltDeg: Double,
        /** Nearest visible road, metres ahead of the camera nadir; null when no row sees road within [MAX_ROAD_M]. */
        val nearM: Double?,
        /** Farthest visible road, metres ahead; [MAX_ROAD_M] when the horizon is in the frame; null without road. */
        val farM: Double?,
    ) {
        /** A whole segment plus its candidate window is in view: near + [SEGMENT_REACH_M] <= far. */
        val segmentFits: Boolean get() = nearM != null && farM != null && nearM + SEGMENT_REACH_M <= farM
    }

    data class Plan(
        /** What the frame shows; the plan covers its first [SEGMENT_REACH_M] of road. */
        val view: RoadView,
        /** Road the focus covers, metres ahead: the nearest visible road and min(near + reach, farthest visible). */
        val nearRoadM: Double,
        val farRoadM: Double,
        /** The same road as depths along the optical axis, which is what the focus distance acts on. */
        val nearDepthM: Double,
        val farDepthM: Double,
        val targetDiopters: Double,
        /** Predicted defocus blur at the nearest/farthest road depth, in output pixels. */
        val edgeBlurPx: Double,
        /** Road region, active-array pixels: rows [roiTop, roiBottom), columns [roiLeft, roiRight). */
        val roiTop: Int,
        val roiBottom: Int,
        val roiLeft: Int,
        val roiRight: Int,
        /** The same rows as fractions of the recorded (16:9) image height, for the analysis stream. */
        val roiTopFrac: Double,
        val roiBottomFrac: Double,
        val heightM: Double,
    )

    /**
     * Device-axis direction (x, y) of sensor row 0 for a sensor orientation, as in OrientationCheck:
     * with the phone upright in landscape (UPRIGHT), this direction points up in the world.
     */
    fun sensorUp(sensorOrientationDeg: Int): Pair<Double, Double>? = when (((sensorOrientationDeg % 360) + 360) % 360) {
        0 -> 0.0 to 1.0
        90 -> 1.0 to 0.0
        180 -> 0.0 to -1.0
        270 -> -1.0 to 0.0
        else -> null
    }

    /** Rows of the active array that a stream of [outW] x [outH] shows (centre crop at zoom 1). */
    fun visibleRows(activeWidth: Int, activeHeight: Int, outW: Int, outH: Int): Pair<Int, Int> {
        val cropH = min(activeHeight.toDouble(), activeWidth.toDouble() * outH / outW)
        val top = (activeHeight - cropH) / 2
        return top.toInt() to (top + cropH).toInt()
    }

    /**
     * The road a stream of [outW] x [outH] shows, or null when the geometry can't tell (no gravity, implausible
     * height, camera pointing straight up or down). [gravity] = TYPE_GRAVITY reading in device axes (points up,
     * ~9.8 m/s2); [sensorUp] = the device-axis direction of sensor row 0 (see OrientationCheck).
     */
    fun roadView(lens: Lens, gravity: FloatArray?, sensorUp: Pair<Double, Double>, heightM: Double, outW: Int, outH: Int): RoadView? {
        if (lens.fyPx <= 0) return null
        val m = mount(gravity, sensorUp, heightM) ?: return null
        return view(m, scan(lens, m, outW, outH))
    }

    /**
     * The focus plan, or null when the geometry can't give one (no gravity, implausible height, camera not
     * looking at the road). Arguments as for [roadView]; the back camera looks along device -z.
     */
    fun plan(lens: Lens, gravity: FloatArray?, sensorUp: Pair<Double, Double>, heightM: Double, outW: Int, outH: Int): Plan? {
        if (lens.fyPx <= 0 || lens.focalMm <= 0 || lens.aperture <= 0) return null
        val m = mount(gravity, sensorUp, heightM) ?: return null
        val s = scan(lens, m, outW, outH)
        val view = view(m, s)
        val near = view.nearM ?: return null
        val far = min(near + SEGMENT_REACH_M, view.farM ?: return null)
        val rowsH = (s.visBottom - s.visTop).toDouble()
        val usableBottom = s.visBottom - (rowsH * BOTTOM_MARGIN_FRAC).toInt()
        val metered = s.road.filter { it.aheadM <= far && it.v < usableBottom }
        if (metered.isEmpty()) return null
        val rowMin = metered.minOf { it.v }
        val rowMax = min(metered.maxOf { it.v } + s.step, usableBottom)
        if (rowMax - rowMin < MIN_ROAD_FRAC * rowsH) return null
        val cosTilt = sqrt(max(0.0, 1 - m.sinTilt * m.sinTilt))
        val nearDepth = near * cosTilt + heightM * m.sinTilt
        val farDepth = far * cosTilt + heightM * m.sinTilt
        val target = ((1 / nearDepth + 1 / farDepth) / 2).coerceIn(0.0, max(0.0, lens.minFocusDiopters))
        val blurPx = max(blurUm(lens, target, nearDepth), blurUm(lens, target, farDepth)) / lens.outputPixelUm
        return Plan(
            view = view,
            nearRoadM = near,
            farRoadM = far,
            nearDepthM = nearDepth,
            farDepthM = farDepth,
            targetDiopters = target,
            edgeBlurPx = blurPx,
            roiTop = rowMin,
            roiBottom = rowMax,
            roiLeft = (lens.activeWidth * ROI_LEFT).toInt(),
            roiRight = (lens.activeWidth * ROI_RIGHT).toInt(),
            roiTopFrac = ((rowMin - s.visTop) / rowsH).coerceIn(0.0, 1.0),
            roiBottomFrac = ((rowMax - s.visTop) / rowsH).coerceIn(0.0, 1.0),
            heightM = heightM,
        )
    }

    /** Gravity and mount, resolved once: unit vectors in device axes and the sine of the axis tilt. */
    private class Mount(val up: DoubleArray, val down: DoubleArray, val forward: DoubleArray, val sinTilt: Double, val heightM: Double)

    /** A sampled image row whose centre-column ray meets the road ahead within [MAX_ROAD_M]. */
    private class RoadRow(val v: Int, val aheadM: Double)

    /** Visible rows [visTop, visBottom) sampled every [step] rows plus the last; [horizonInView]: some row sees no road within the cap. */
    private class Scan(val visTop: Int, val visBottom: Int, val step: Int, val road: List<RoadRow>, val horizonInView: Boolean)

    private fun mount(gravity: FloatArray?, sensorUp: Pair<Double, Double>, heightM: Double): Mount? {
        val g = gravity ?: return null
        if (heightM !in 0.3..5.0) return null
        val gn = sqrt(g[0].toDouble() * g[0] + g[1].toDouble() * g[1] + g[2].toDouble() * g[2])
        if (gn < 5.0) return null
        val up = doubleArrayOf(g[0] / gn, g[1] / gn, g[2] / gn)
        val axisUp = dot(AXIS, up)
        // "Ahead" = the horizontal projection of the optical axis, as the pipeline's vehicle frame (geometry.py).
        val fwd = doubleArrayOf(AXIS[0] - axisUp * up[0], AXIS[1] - axisUp * up[1], AXIS[2] - axisUp * up[2])
        val fn = sqrt(dot(fwd, fwd))
        if (fn < 1e-6) return null
        // Increasing rows run opposite to the direction of sensor row 0.
        val down = doubleArrayOf(-sensorUp.first, -sensorUp.second, 0.0)
        return Mount(up, down, doubleArrayOf(fwd[0] / fn, fwd[1] / fn, fwd[2] / fn), -axisUp, heightM)
    }

    /**
     * Metres ahead of the nadir where the centre-column ray through active-array row [v] meets the road:
     * +infinity at or above the horizon, zero or negative for road below or behind the camera.
     */
    private fun aheadM(lens: Lens, m: Mount, v: Int): Double {
        val off = (v + 0.5 - lens.cyPx) / lens.fyPx
        val ray = doubleArrayOf(AXIS[0] + off * m.down[0], AXIS[1] + off * m.down[1], AXIS[2] + off * m.down[2])
        val rn = sqrt(dot(ray, ray))
        val sinDepression = -dot(ray, m.up) / rn
        if (sinDepression <= 0) return Double.POSITIVE_INFINITY
        return m.heightM / sinDepression * dot(ray, m.forward) / rn
    }

    /** Samples every visible row (at 400-row resolution) plus the last row, so both end rows are exact. */
    private fun scan(lens: Lens, m: Mount, outW: Int, outH: Int): Scan {
        val (visTop, visBottom) = visibleRows(lens.activeWidth, lens.activeHeight, outW, outH)
        val step = max(1, (visBottom - visTop) / 400)
        val road = ArrayList<RoadRow>()
        var horizon = false
        fun visit(v: Int) {
            val ahead = aheadM(lens, m, v)
            if (ahead > MAX_ROAD_M) horizon = true else if (ahead > 0) road += RoadRow(v, ahead)
        }
        for (v in visTop until visBottom step step) visit(v)
        if (visBottom > visTop && (visBottom - 1 - visTop) % step != 0) visit(visBottom - 1)
        return Scan(visTop, visBottom, step, road, horizon)
    }

    private fun view(m: Mount, s: Scan) = RoadView(
        tiltDeg = Math.toDegrees(asin(m.sinTilt.coerceIn(-1.0, 1.0))),
        nearM = s.road.minOfOrNull { it.aheadM },
        farM = if (s.road.isEmpty()) null else if (s.horizonInView) MAX_ROAD_M else s.road.maxOf { it.aheadM },
    )

    private fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    /** Defocus blur diameter on the sensor (micrometres) for an object at [depthM] with the lens set to [diopters]. */
    fun blurUm(lens: Lens, diopters: Double, depthM: Double): Double {
        val apertureMm = lens.focalMm / lens.aperture
        // mm x mm x (1/m) = 1e-3 mm = micrometres.
        return apertureMm * lens.focalMm * abs(diopters - 1 / depthM)
    }

    /** Sweep positions around [target], within the lens range, target included. */
    fun sweep(target: Double, minFocusDiopters: Double): List<Double> =
        SWEEP_OFFSETS.map { (target + it).coerceIn(0.0, minFocusDiopters) }.distinct()

    data class Decision(val diopters: Double, val method: String)

    /**
     * Chooses the final focus from the geometric [target] and the sweep's (dioptres, sharpness) scores.
     * The analysis stream is ~6x coarser than the recording, so small differences are noise: the sweep only
     * overrides geometry when it finds a clearly sharper position well away from the target (a mis-set height
     * or a lens calibration error).
     */
    fun decide(target: Double, scores: List<Pair<Double, Double>>): Decision {
        val valid = scores.filter { it.second.isFinite() && it.second > 0 }
        if (valid.size < 3) return Decision(target, "geometry")
        val lo = valid.minOf { it.second }
        val best = valid.maxBy { it.second }
        if (best.second <= lo * (1 + FLAT_SPREAD)) return Decision(target, "geometry_sweep_flat")
        val atTarget = valid.minBy { abs(it.first - target) }
        return if (best.second > atTarget.second * (1 + OVERRIDE_GAIN) && abs(best.first - target) >= OVERRIDE_MIN_DIST)
            Decision(best.first, "sweep_override")
        else Decision(target, "geometry_sweep_confirmed")
    }
}
