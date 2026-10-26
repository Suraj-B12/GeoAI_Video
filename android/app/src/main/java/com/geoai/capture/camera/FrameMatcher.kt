package com.geoai.capture.camera

import kotlin.math.abs

/** Per-frame camera metadata from onCaptureCompleted. */
data class ResultMeta(
    val sensorTsNs: Long,
    val frameNumber: Long,
    val exposureNs: Long?,
    val frameDurationNs: Long?,
    val iso: Int?,
    val rsSkewNs: Long?,
    val focusDiopters: Float?,
    val focalLengthMm: Float?,
    val lensState: Int?,
    val oisMode: Int?,
    val eisMode: Int?,
    val aeState: Int?,
    val afState: Int?,
    val awbGains: String?,
    val zoomRatio: Float?,
)

/** One sample written to the muxer. */
data class EncodedFrame(
    val segment: Int,
    val frameInSegment: Int,
    val frameGlobal: Long,
    val ptsUs: Long,
    val isKeyframe: Boolean,
    val bytes: Int,
)

/** [residualNs] = pts_ns − offset − sensor_ts (null when unmatched). */
data class FrameRow(val frame: EncodedFrame, val meta: ResultMeta?, val offsetNs: Long?, val residualNs: Long?)

/**
 * Joins every encoded video frame to the CaptureResult of the exposure that produced it.
 *
 * Encoder PTS and SENSOR_TIMESTAMP share a clock only when the encoder surface uses
 * TIMESTAMP_BASE_SENSOR (offset 0). With the default base, encoder timestamps are MONOTONIC and
 * the offset equals monotonic − elapsedRealtime. The offset is chosen from [offsetCandidatesNs]
 * by majority match, or from a histogram of differences if no candidate fits. It stays fixed for
 * the session, because both clocks run at the same rate while the device is awake.
 *
 * Results and frames can arrive in either order. A frame waits up to [waitNs] of newer results
 * for its match before it is written as unmatched. Thread-safe.
 */
class FrameMatcher(
    private val toleranceNs: Long,
    private val offsetCandidatesNs: List<Long>,
    private val sink: (FrameRow) -> Unit,
    private val waitNs: Long = 500_000_000L,
    private val maxResults: Int = 600,
) {
    private class Entry(val meta: ResultMeta) { var used = false }

    private val results = ArrayDeque<Entry>()
    private val pending = ArrayDeque<EncodedFrame>()

    var offsetNs: Long? = null
        private set
    var offsetMethod: String = "undetermined"
        private set
    var matched = 0L
        private set
    var unmatched = 0L
        private set
    /** Camera frames whose result never matched an encoded frame (dropped before encoding). */
    var resultsWithoutFrame = 0L
        private set

    @Synchronized fun onResult(meta: ResultMeta) {
        // Results normally arrive in order; keep the deque sorted anyway.
        if (results.isNotEmpty() && meta.sensorTsNs < results.last().meta.sensorTsNs) {
            val idx = results.indexOfFirst { it.meta.sensorTsNs > meta.sensorTsNs }
            results.add(if (idx < 0) results.size else idx, Entry(meta))
        } else {
            results.addLast(Entry(meta))
        }
        process()
        trim()
    }

    @Synchronized fun onFrame(frame: EncodedFrame) {
        pending.addLast(frame)
        process()
    }

    /** At stop: match what can be matched and write the rest as unmatched. */
    @Synchronized fun flush() {
        if (offsetNs == null) decideOffset(force = true)
        process()
        while (pending.isNotEmpty()) emitUnmatched(pending.removeFirst())
        if (offsetNs != null) resultsWithoutFrame += results.count { !it.used && isAfterFirstFrame(it) }
        results.clear()
    }

    private var firstFrameTargetNs: Long? = null
    private fun isAfterFirstFrame(e: Entry): Boolean {
        val first = firstFrameTargetNs ?: return false
        return e.meta.sensorTsNs >= first - toleranceNs
    }

    private fun process() {
        if (offsetNs == null && !decideOffset(force = false)) {
            // Bound memory if the offset can't be found (should not happen on a sane device).
            while (pending.size > 300) emitUnmatched(pending.removeFirst())
            return
        }
        val off = offsetNs ?: return
        while (pending.isNotEmpty()) {
            val f = pending.first()
            val target = f.ptsUs * 1000L - off
            if (firstFrameTargetNs == null) firstFrameTargetNs = target
            val e = nearest(target)
            if (e != null && !e.used && abs(e.meta.sensorTsNs - target) <= toleranceNs) {
                e.used = true
                pending.removeFirst()
                matched++
                sink(FrameRow(f, e.meta, off, target - e.meta.sensorTsNs))
                continue
            }
            val newest = results.lastOrNull()?.meta?.sensorTsNs
            if (newest != null && newest > target + waitNs) {
                pending.removeFirst()
                emitUnmatched(f)
                continue
            }
            break
        }
    }

    private fun emitUnmatched(f: EncodedFrame) {
        unmatched++
        sink(FrameRow(f, null, offsetNs, null))
    }

    /** Returns true once the offset is known. */
    private fun decideOffset(force: Boolean): Boolean {
        if (offsetNs != null) return true
        if (pending.isEmpty() || results.isEmpty()) return false
        if (!force && (pending.size < MIN_FRAMES_FOR_OFFSET || results.size < MIN_FRAMES_FOR_OFFSET)) return false
        val need = if (force) 1 else (pending.size / 2).coerceAtLeast(MIN_FRAMES_FOR_OFFSET / 2)
        var best: Long? = null
        var bestCount = 0
        for (c in offsetCandidatesNs) {
            val count = pending.count { f -> hasMatch(f.ptsUs * 1000L - c) }
            if (count > bestCount) { best = c; bestCount = count }
        }
        if (best != null && bestCount >= need) {
            offsetNs = best
            offsetMethod = "candidate"
            return true
        }
        if (!force && pending.size < HISTOGRAM_FRAMES) return false
        // Histogram of all pairwise differences in 1 ms bins; the mode is the offset.
        val bins = HashMap<Long, Int>()
        for (f in pending) for (r in results) {
            val d = f.ptsUs * 1000L - r.meta.sensorTsNs
            val bin = Math.floorDiv(d, 1_000_000L)
            bins[bin] = (bins[bin] ?: 0) + 1
        }
        val mode = bins.maxByOrNull { it.value } ?: return false
        // Refine: median of the differences that fall in the winning bin and its neighbours.
        val diffs = ArrayList<Long>()
        for (f in pending) for (r in results) {
            val d = f.ptsUs * 1000L - r.meta.sensorTsNs
            if (abs(Math.floorDiv(d, 1_000_000L) - mode.key) <= 1) diffs += d
        }
        diffs.sort()
        val candidate = diffs[diffs.size / 2]
        val count = pending.count { f -> hasMatch(f.ptsUs * 1000L - candidate) }
        if (count >= need) {
            offsetNs = candidate
            offsetMethod = "histogram"
            return true
        }
        return false
    }

    private fun hasMatch(targetNs: Long): Boolean {
        val e = nearest(targetNs) ?: return false
        return abs(e.meta.sensorTsNs - targetNs) <= toleranceNs
    }

    private fun nearest(targetNs: Long): Entry? {
        if (results.isEmpty()) return null
        var lo = 0
        var hi = results.size - 1
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (results[mid].meta.sensorTsNs < targetNs) lo = mid + 1 else hi = mid
        }
        val a = results[lo]
        if (lo == 0) return a
        val b = results[lo - 1]
        return if (abs(a.meta.sensorTsNs - targetNs) <= abs(b.meta.sensorTsNs - targetNs)) a else b
    }

    private fun trim() {
        while (results.size > maxResults) {
            val e = results.removeFirst()
            if (!e.used && offsetNs != null && isAfterFirstFrame(e)) resultsWithoutFrame++
        }
    }

    companion object {
        const val MIN_FRAMES_FOR_OFFSET = 10
        const val HISTOGRAM_FRAMES = 60
    }
}
