package com.geoai.capture.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale

/** Operator texts of the live mount check ([MountText]). */
class MountTextTest {

    @Test fun tiltReadsBelowOrAboveTheHorizon() {
        assertEquals("23.0° below horizon", MountText.tilt(23.0))
        assertEquals("0.0° below horizon", MountText.tilt(0.0))
        assertEquals("4.5° ABOVE horizon", MountText.tilt(-4.5))
        assertEquals("–", MountText.tilt(null))
    }

    @Test fun roadRangeShowsTheHorizonCap() {
        // Pilot mount (numpy): 1.4469-32.2038 m; 5 degrees down: horizon in view, capped at 60 m.
        assertEquals("1.4-32.2 m ahead", MountText.roadRange(1.4469, 32.2038, viewKnown = true))
        assertEquals("2.9-60+ m ahead", MountText.roadRange(2.8858, 60.0, viewKnown = true))
        assertEquals("no road in view", MountText.roadRange(null, null, viewKnown = true))
        assertEquals("–", MountText.roadRange(null, null, viewKnown = false))
    }

    @Test fun segmentFitNamesTheRoadItNeeds() {
        // Needed road = near + 5 m segment + 1 m window: 1.4469 + 6 = 7.4469, 0.7437 + 6 = 6.7437.
        assertEquals("fits (needs 7.4 m)", MountText.segment(1.4469, fits = true))
        assertEquals("NO: needs 6.7 m", MountText.segment(0.7437, fits = false))
        assertEquals("NO: no road in view", MountText.segment(null, fits = false))
        assertEquals("–", MountText.segment(null, fits = null))
        assertEquals("5 m segment", MountText.segmentLabel)
    }

    @Test fun textsFitOneLineAndUseDotDecimals() {
        val saved = Locale.getDefault()
        try {
            Locale.setDefault(Locale.GERMANY)
            val texts = listOf(
                MountText.tilt(-90.0), MountText.tilt(90.0), MountText.tilt(23.04),
                MountText.roadRange(59.99, 60.0, true), MountText.roadRange(12.345, 59.99, true),
                MountText.roadRange(null, null, true),
                MountText.segment(59.99, false), MountText.segment(54.0, true), MountText.segment(null, false),
            )
            texts.forEach { t ->
                assertTrue("'$t' is ${t.length} chars", t.length <= MountText.MAX_CHARS)
                assertFalse("'$t' uses a comma decimal", t.contains(','))
            }
            assertEquals("23.0° below horizon", MountText.tilt(23.04))
        } finally {
            Locale.setDefault(saved)
        }
    }
}
