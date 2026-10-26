package com.geoai.capture.camera

import android.media.MediaCodec
import android.media.MediaFormat
import androidx.annotation.OptIn
import androidx.media3.common.C
import androidx.media3.common.Format
import androidx.media3.common.util.MediaFormatUtil
import androidx.media3.common.util.UnstableApi
import androidx.media3.muxer.BufferInfo
import androidx.media3.muxer.FragmentedMp4Muxer
import com.geoai.capture.io.AsyncLineWriter
import com.geoai.capture.io.EventLog
import com.geoai.capture.util.Clocks
import com.geoai.capture.util.Fmt
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Writes encoder output into fragmented MP4 segments (seg_0001.mp4, …).
 *
 * - A segment always starts on a keyframe; rotation happens at the first keyframe after
 *   [segmentDurationUs], so no frame is lost between segments.
 * - Fragmented MP4 stays readable up to the last complete fragment (~2 s) if the app or phone dies.
 * - All muxer calls happen on the encoder thread; finished muxers are closed on a helper thread.
 * - Segment numbering and the global frame counter continue across encoder restarts
 *   (thermal step-down, camera recovery), so a session's frames.csv is one continuous sequence.
 */
@OptIn(UnstableApi::class)
class SegmentWriter(
    private val videoDir: File,
    private val segmentDurationUs: Long,
    private val segmentsCsv: AsyncLineWriter,
    private val events: EventLog,
    private val onFrame: (EncodedFrame) -> Unit,
) {
    private var format: Format? = null
    private var muxer: FragmentedMp4Muxer? = null
    private var trackId = -1
    private val closer = Executors.newSingleThreadExecutor { r -> Thread(r, "geoai-muxer-close") }

    /** Set before each encoder (re)start so segment rows describe the stream. */
    @Volatile var streamWidth = 0
    @Volatile var streamHeight = 0
    @Volatile var streamMime = ""
    @Volatile var streamBitrate = 0

    var segmentIndex = 0
        private set
    var frameGlobal = 0L
        private set
    var droppedBeforeKeyframe = 0L
        private set

    private var segFile: File? = null
    private var segFrames = 0
    private var segBytes = 0L
    private var segFirstPtsUs = 0L
    private var segLastPtsUs = 0L
    private var segStartTNs = 0L
    private var lastPtsUs = Long.MIN_VALUE

    /** Called from the encoder thread when its output format is known (or changes). */
    fun onFormat(mf: MediaFormat) {
        val f = MediaFormatUtil.createFormatFromMediaFormat(mf)
        if (muxer != null && format != null && format != f) {
            // A new format mid-stream needs a new track, so start a new segment at the next keyframe.
            close("format_change")
        }
        format = f
    }

    fun onSample(buf: ByteBuffer, info: MediaCodec.BufferInfo) {
        val key = info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME != 0
        val pts = info.presentationTimeUs
        if (muxer == null) {
            if (!key || format == null) {
                droppedBeforeKeyframe++
                return
            }
            open(pts)
        } else if (key && pts - segFirstPtsUs >= segmentDurationUs) {
            close("rotation")
            open(pts)
        }
        if (pts <= lastPtsUs) {
            events.log("non_monotonic_pts", JSONObject().put("pts_us", pts).put("previous_pts_us", lastPtsUs))
        }
        lastPtsUs = pts
        muxer!!.writeSampleData(trackId, buf, BufferInfo(pts, info.size, if (key) C.BUFFER_FLAG_KEY_FRAME else 0))
        onFrame(EncodedFrame(segmentIndex, segFrames, frameGlobal, pts, key, info.size))
        segFrames++
        frameGlobal++
        segBytes += info.size
        segLastPtsUs = pts
    }

    private fun open(ptsUs: Long) {
        segmentIndex++
        val file = File(videoDir, String.format(Locale.ROOT, "seg_%04d.mp4", segmentIndex))
        val channel = FileOutputStream(file).channel
        val m = FragmentedMp4Muxer.Builder(channel).setFragmentDurationMs(2000).build()
        trackId = m.addTrack(format!!)
        muxer = m
        segFile = file
        segFrames = 0
        segBytes = 0
        segFirstPtsUs = ptsUs
        segLastPtsUs = ptsUs
        segStartTNs = Clocks.elapsedNs()
        events.log(
            "segment_start",
            JSONObject().put("segment", segmentIndex).put("file", "video/${file.name}").put("first_pts_us", ptsUs)
                .put("width", streamWidth).put("height", streamHeight).put("mime", streamMime),
        )
    }

    /** Closes the open segment (if any). [reason]: rotation | stop | thermal_step_down | error | format_change. */
    fun close(reason: String) {
        val m = muxer ?: return
        muxer = null
        val file = segFile
        val row = Fmt.row(
            segmentIndex.toString(),
            "video/${file?.name ?: ""}",
            segStartTNs.toString(),
            Clocks.elapsedNs().toString(),
            segFirstPtsUs.toString(),
            segLastPtsUs.toString(),
            segFrames.toString(),
            segBytes.toString(),
            streamWidth.toString(),
            streamHeight.toString(),
            streamMime,
            streamBitrate.toString(),
            reason,
        )
        segmentsCsv.write(row)
        events.log("segment_end", JSONObject().put("segment", segmentIndex).put("frames", segFrames).put("reason", reason))
        closer.execute {
            try {
                m.close()
            } catch (e: Exception) {
                events.log("muxer_close_error", JSONObject().put("segment", file?.name).put("error", e.toString()))
            }
        }
    }

    /** Close the last segment and wait until every muxer has finished writing. */
    fun finish(reason: String) {
        close(reason)
        closer.shutdown()
        closer.awaitTermination(30, TimeUnit.SECONDS)
    }

    companion object {
        const val HEADER = "segment,file,start_t_ns,end_t_ns,first_pts_us,last_pts_us,frames,bytes,width,height,mime,bitrate_bps,close_reason"
    }
}
