package com.geoai.capture.ui

import com.geoai.capture.camera.FocusPlanner
import java.util.Locale

/**
 * Operator texts for the live mount check: camera tilt, the road the frame shows, and whether a segment fits
 * (docs/pipeline/13-segment-frames.md §2). Pure Kotlin so the wording is unit-tested.
 *
 * Every text stays within [MAX_CHARS] monospace characters: one line of the StatusPanel's value column in the
 * narrowest layout it supports (384 dp landscape at 16:9: panel 287 dp - 112 dp label = 175 dp; 13 sp monospace
 * advances 0.6 em = 7.8 dp at font scale 1).
 */
object MountText {
    const val MAX_CHARS = 22

    private fun f(fmt: String, vararg a: Any?) = String.format(Locale.ROOT, fmt, *a)

    /** Optical axis against the horizon; looking up is spelled out because no road can be in view. */
    fun tilt(tiltDeg: Double?): String = when {
        tiltDeg == null -> "–"
        tiltDeg >= 0 -> f("%.1f° below horizon", tiltDeg)
        else -> f("%.1f° ABOVE horizon", -tiltDeg)
    }

    /** Nearest..farthest visible road ahead; "60+" when the horizon is in the frame. [tiltKnown]: gravity was read. */
    fun roadRange(nearM: Double?, farM: Double?, tiltKnown: Boolean): String = when {
        nearM == null || farM == null -> if (tiltKnown) "no road in view" else "–"
        farM >= FocusPlanner.MAX_ROAD_M -> f("%.1f-%.0f+ m ahead", nearM, FocusPlanner.MAX_ROAD_M)
        else -> f("%.1f-%.1f m ahead", nearM, farM)
    }

    /** Whether a segment and its candidate window fit, with the farthest road that requires (near + reach). */
    fun segment(nearM: Double?, fits: Boolean?): String = when {
        fits == null -> "–"
        nearM == null -> "NO: no road in view"
        fits -> f("fits (needs %.1f m)", nearM + FocusPlanner.SEGMENT_REACH_M)
        else -> f("NO: needs %.1f m", nearM + FocusPlanner.SEGMENT_REACH_M)
    }

    /** Label of the segment row, from the segment length the pipeline uses. */
    val segmentLabel: String = f("%.0f m segment", FocusPlanner.SEGMENT_LENGTH_M)
}
