package com.geoai.capture.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class FrameMatcherTest {

    private val frameNs = 33_333_333L
    private val t0 = 20_000_000_000_000L // ~5.5 h after boot (more than the 2 h of sleep used below)

    private fun meta(ts: Long, n: Long) = ResultMeta(ts, n, 200_000L, frameNs, 100, 20_000_000L, 0.5f, 5.4f, 0, 0, 0, 0, 4, null, 1f)
    private fun frame(i: Int, ptsUs: Long) = EncodedFrame(1, i, i.toLong(), ptsUs, i % 30 == 0, 1000)

    private fun run(offsetNs: Long, candidates: List<Long>, dropFrame: Int = -1, dropResult: Int = -1, resultsFirst: Boolean = true): List<FrameRow> {
        val rows = mutableListOf<FrameRow>()
        val m = FrameMatcher(2_000_000L, candidates, { rows += it })
        val n = 90
        for (i in 0 until n) {
            val ts = t0 + i * frameNs
            val pts = (ts + offsetNs) / 1000L
            if (resultsFirst) {
                if (i != dropResult) m.onResult(meta(ts, i.toLong()))
                if (i != dropFrame) m.onFrame(frame(i, pts))
            } else {
                if (i != dropFrame) m.onFrame(frame(i, pts))
                if (i != dropResult) m.onResult(meta(ts, i.toLong()))
            }
        }
        m.flush()
        assertNotNull(m.offsetNs)
        return rows
    }

    @Test fun sensorTimestampBaseMatchesWithZeroOffset() {
        val rows = run(offsetNs = 0L, candidates = listOf(0L, -7_200_000_000_000L))
        assertEquals(90, rows.size)
        assertEquals(90, rows.count { it.meta != null })
        rows.forEach { assertEquals(it.frame.frameGlobal, it.meta!!.frameNumber) }
    }

    @Test fun monotonicEncoderBaseUsesSecondCandidate() {
        val sleep = 7_200_000_000_000L // 2 h of deep sleep since boot
        val rows = run(offsetNs = -sleep, candidates = listOf(0L, -sleep))
        assertEquals(90, rows.count { it.meta != null })
        rows.forEach { assertEquals(it.frame.frameGlobal, it.meta!!.frameNumber) }
    }

    @Test fun unknownOffsetFoundByHistogram() {
        val rows = run(offsetNs = 123_456_789L, candidates = listOf(0L))
        assertEquals(90, rows.count { it.meta != null })
    }

    @Test fun framesMayArriveBeforeResults() {
        val rows = run(offsetNs = 0L, candidates = listOf(0L), resultsFirst = false)
        assertEquals(90, rows.count { it.meta != null })
    }

    @Test fun missingResultGivesOneUnmatchedFrame() {
        val rows = run(offsetNs = 0L, candidates = listOf(0L), dropResult = 40)
        assertEquals(90, rows.size)
        val unmatched = rows.filter { it.meta == null }
        assertEquals(1, unmatched.size)
        assertEquals(40L, unmatched[0].frame.frameGlobal)
        assertNull(unmatched[0].residualNs)
    }

    @Test fun droppedEncoderFrameIsCountedAsCameraFrameNotEncoded() {
        val rows = mutableListOf<FrameRow>()
        val m = FrameMatcher(2_000_000L, listOf(0L), { rows += it })
        for (i in 0 until 90) {
            val ts = t0 + i * frameNs
            m.onResult(meta(ts, i.toLong()))
            if (i != 50) m.onFrame(frame(i, ts / 1000L))
        }
        m.flush()
        assertEquals(89, rows.size)
        assertEquals(1L, m.resultsWithoutFrame)
    }
}
