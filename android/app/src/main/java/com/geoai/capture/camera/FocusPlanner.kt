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
 * 1. Gravity gives the camera's depression below the horizon; each image row is a ray at its own depression.
 * 2. A ray that meets the road plane (height h below the lens) at horizontal distance at most [FAR_ROAD_M] is road.
 *    Its depth along the optical axis is z = (h / sin(depression)) * cos(angle off axis).
 * 3. The focus that gives equal, smallest defocus at the nearest and farthest road depth is the dioptre midpoint
 *    D = (1/z_near + 1/z_far) / 2. Blur diameter on the sensor at depth z: A * f * |D - 1/z|, A = f / N.
 *
 * Pure Kotlin (no Android types) so it is unit-tested on the JVM. Coordinates are sensor active-array pixels.
 */
object FocusPlanner {
    /** Road farther than this is not what the survey analyses (the pipeline uses at most ~6 m, mostly 1-3 m). */
    const val FAR_ROAD_M = 4.0
    /** Bottom rows that may show the bonnet or bumper rather than road. */
    const val BOTTOM_MARGIN_FRAC = 0.04
    /** Left and right edges of the road region (fractions of the width): roadside objects stay out. */
    const val ROI_LEFT = 0.20
    const val ROI_RIGHT = 0.80
    /** At least this share of the visible rows must be road for a plan. */
    const val MIN_ROAD_FRAC = 0.10
    /** Contrast sweep: offsets around the target, dioptres. */
    val SWEEP_OFFSETS = doubleArrayOf(-0.6, -0.3, 0.0, 0.3, 0.6)
    /** A sweep result replaces the geometric target only if its score is this much higher ... */
    const val OVERRIDE_GAIN = 0.15
    /** ... and it lies at least this far (dioptres) from the target. */
    const val OVERRIDE_MIN_DIST = 0.25
    /** Scores within this relative spread carry no focus information (texture too fine for the analysis stream). */
    const val FLAT_SPREAD = 0.05

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

    data class Plan(
        val depressionDeg: Double,
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
     * The focus plan, or null when the geometry can't give one (no gravity, implausible height, camera not
     * looking at the road). [gravity] = TYPE_GRAVITY reading in device axes (points up, ~9.8 m/s2);
     * [sensorUp] = the device-axis direction of sensor row 0 (see OrientationCheck), back camera along -z.
     */
    fun plan(lens: Lens, gravity: FloatArray?, sensorUp: Pair<Double, Double>, heightM: Double, outW: Int, outH: Int): Plan? {
        val g = gravity ?: return null
        if (heightM !in 0.3..5.0 || lens.fyPx <= 0 || lens.focalMm <= 0 || lens.aperture <= 0) return null
        val gn = sqrt(g[0].toDouble() * g[0] + g[1].toDouble() * g[1] + g[2].toDouble() * g[2])
        if (gn < 5.0) return null
        val up = doubleArrayOf(g[0] / gn, g[1] / gn, g[2] / gn)
        // Optical axis (back camera) and the image "down" direction (increasing rows), device axes.
        val axis = doubleArrayOf(0.0, 0.0, -1.0)
        val down = doubleArrayOf(-sensorUp.first, -sensorUp.second, 0.0)
        val (visTop, visBottom) = visibleRows(lens.activeWidth, lens.activeHeight, outW, outH)
        val usableBottom = visBottom - ((visBottom - visTop) * BOTTOM_MARGIN_FRAC).toInt()
        var near = Double.MAX_VALUE
        var far = 0.0
        var rowMin = Int.MAX_VALUE
        var rowMax = Int.MIN_VALUE
        val step = max(1, (visBottom - visTop) / 400)
        var v = visTop
        while (v < usableBottom) {
            val off = (v + 0.5 - lens.cyPx) / lens.fyPx
            val ray = doubleArrayOf(axis[0] + off * down[0], axis[1] + off * down[1], axis[2] + off * down[2])
            val rn = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2])
            val depression = -(ray[0] * up[0] + ray[1] * up[1] + ray[2] * up[2]) / rn  // sin of the depression
            if (depression > 0.02) {
                val slant = heightM / depression
                val ground = slant * sqrt(max(0.0, 1 - depression * depression))
                if (ground <= FAR_ROAD_M) {
                    val depth = slant / sqrt(1 + off * off)
                    near = min(near, depth)
                    far = max(far, depth)
                    rowMin = min(rowMin, v)
                    rowMax = max(rowMax, v + step)
                }
            }
            v += step
        }
        if (rowMin == Int.MAX_VALUE || (rowMax - rowMin) < MIN_ROAD_FRAC * (visBottom - visTop)) return null
        val target = ((1 / near + 1 / far) / 2).coerceIn(0.0, max(0.0, lens.minFocusDiopters))
        val blurPx = blurUm(lens, target, near).coerceAtLeast(blurUm(lens, target, far)) / lens.outputPixelUm
        val axisDep = asin((-(axis[0] * up[0] + axis[1] * up[1] + axis[2] * up[2])).coerceIn(-1.0, 1.0))
        val rowsH = (visBottom - visTop).toDouble()
        return Plan(
            depressionDeg = Math.toDegrees(axisDep),
            nearDepthM = near,
            farDepthM = far,
            targetDiopters = target,
            edgeBlurPx = blurPx,
            roiTop = rowMin,
            roiBottom = min(rowMax, usableBottom),
            roiLeft = (lens.activeWidth * ROI_LEFT).toInt(),
            roiRight = (lens.activeWidth * ROI_RIGHT).toInt(),
            roiTopFrac = ((rowMin - visTop) / rowsH).coerceIn(0.0, 1.0),
            roiBottomFrac = ((min(rowMax, usableBottom) - visTop) / rowsH).coerceIn(0.0, 1.0),
            heightM = heightM,
        )
    }

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
