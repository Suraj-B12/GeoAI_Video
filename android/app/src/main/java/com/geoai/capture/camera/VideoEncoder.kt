package com.geoai.capture.camera

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecInfo.CodecCapabilities
import android.media.MediaCodecInfo.EncoderCapabilities
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Handler
import android.view.Surface
import org.json.JSONObject
import java.nio.ByteBuffer

/**
 * Hardware video encoder fed by a Surface the camera renders into.
 * B-frames are disabled: the fragmented-MP4 muxer does not support out-of-order frames, and
 * presentation order == decode order keeps the per-frame join on the PC trivial.
 */
class VideoEncoder(
    val mime: String,
    val width: Int,
    val height: Int,
    val fps: Int,
    private val requestedBitrate: Int,
    private val iFrameIntervalSec: Int,
    private val handler: Handler,
    private val listener: Listener,
) {
    interface Listener {
        fun onFormat(format: MediaFormat)
        fun onSample(buffer: ByteBuffer, info: MediaCodec.BufferInfo)
        fun onEndOfStream()
        fun onError(e: Exception)
    }

    lateinit var codec: MediaCodec
        private set
    lateinit var inputSurface: Surface
        private set
    var codecName = ""
        private set
    var hardwareAccelerated = false
        private set
    var appliedBitrate = requestedBitrate
        private set
    var bitrateMode = "default"
        private set
    var profileName = "default"
        private set
    @Volatile var released = false
        private set

    fun configure() {
        val info = pickEncoder(mime, width, height, fps)
            ?: throw IllegalStateException("No encoder for $mime ${width}x$height@$fps")
        val caps = info.getCapabilitiesForType(mime)
        val vcaps = caps.videoCapabilities ?: throw IllegalStateException("${info.name} has no video capabilities")
        appliedBitrate = vcaps.bitrateRange.clamp(requestedBitrate)
        codecName = info.name
        hardwareAccelerated = info.isHardwareAccelerated

        // Some encoders reject optional keys. Try the full format first, then progressively simpler
        // ones; every attempt and its error is recorded in describe() and session.json.
        val enc = caps.encoderCapabilities
        val wantedProfile = when (mime) {
            MediaFormat.MIMETYPE_VIDEO_HEVC -> MediaCodecInfo.CodecProfileLevel.HEVCProfileMain
            MediaFormat.MIMETYPE_VIDEO_AVC -> MediaCodecInfo.CodecProfileLevel.AVCProfileHigh
            else -> null
        }
        val profileSupported = wantedProfile != null && caps.profileLevels.any { it.profile == wantedProfile }
        var lastError: Exception? = null
        for (level in 0..3) {
            val format = MediaFormat.createVideoFormat(mime, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, appliedBitrate)
                setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameIntervalSec)
                // B-frames are not supported by the fragmented-MP4 muxer: only the last-resort level omits the key.
                if (level <= 2) setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
                if (level <= 1) {
                    bitrateMode = when {
                        enc?.isBitrateModeSupported(EncoderCapabilities.BITRATE_MODE_VBR) == true -> {
                            setInteger(MediaFormat.KEY_BITRATE_MODE, EncoderCapabilities.BITRATE_MODE_VBR); "VBR"
                        }
                        enc?.isBitrateModeSupported(EncoderCapabilities.BITRATE_MODE_CBR) == true -> {
                            setInteger(MediaFormat.KEY_BITRATE_MODE, EncoderCapabilities.BITRATE_MODE_CBR); "CBR"
                        }
                        else -> "default"
                    }
                    setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT709)
                    setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
                    setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
                } else {
                    bitrateMode = "default"
                }
                if (level == 0) {
                    setInteger(MediaFormat.KEY_PRIORITY, 0) // real time
                    // Parameter sets inside every keyframe: any segment is decodable on its own.
                    setInteger(MediaFormat.KEY_PREPEND_HEADER_TO_SYNC_FRAMES, 1)
                    if (profileSupported) setInteger(MediaFormat.KEY_PROFILE, wantedProfile!!)
                }
            }
            val c = MediaCodec.createByCodecName(info.name)
            try {
                c.setCallback(callback, handler)
                c.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                inputSurface = c.createInputSurface()
                codec = c
                configLevel = level
                profileName = if (level == 0 && profileSupported) {
                    if (mime == MediaFormat.MIMETYPE_VIDEO_HEVC) "HEVC Main" else "AVC High"
                } else "encoder default"
                return
            } catch (e: Exception) {
                lastError = e
                configFailures += "level $level: ${e.javaClass.simpleName}: ${e.message}"
                runCatching { c.release() }
            }
        }
        throw IllegalStateException("Could not configure ${info.name}: ${configFailures.joinToString("; ")}", lastError)
    }

    /** 0 = full format; higher = optional keys dropped after the encoder rejected them. */
    var configLevel = -1
        private set
    private val configFailures = mutableListOf<String>()

    fun start() = codec.start()

    fun signalEndOfStream() = codec.signalEndOfInputStream()

    fun release() {
        if (released) return
        released = true
        runCatching { codec.stop() }
        runCatching { codec.release() }
        runCatching { inputSurface.release() }
    }

    fun describe(): JSONObject = JSONObject()
        .put("mime", mime)
        .put("codec_name", codecName)
        .put("hardware_accelerated", hardwareAccelerated)
        .put("width", width)
        .put("height", height)
        .put("fps", fps)
        .put("bitrate_bps_requested", requestedBitrate)
        .put("bitrate_bps_applied", appliedBitrate)
        .put("bitrate_mode", bitrateMode)
        .put("profile", profileName)
        .put("i_frame_interval_s", iFrameIntervalSec)
        .put("config_level", configLevel)
        .put("config_failures", org.json.JSONArray(configFailures))
        .put("max_b_frames", if (configLevel <= 2) 0 else JSONObject.NULL)
        .put("prepend_parameter_sets_to_keyframes", configLevel == 0)
        .put("color", if (configLevel <= 1) JSONObject().put("standard", "BT709").put("range", "LIMITED").put("transfer", "SDR") else JSONObject.NULL)

    private val callback = object : MediaCodec.Callback() {
        override fun onInputBufferAvailable(codec: MediaCodec, index: Int) = Unit // Surface input.

        override fun onOutputBufferAvailable(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
            try {
                val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                val config = info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                if (!config && info.size > 0) {
                    val buf = codec.getOutputBuffer(index)
                    if (buf != null) {
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        listener.onSample(buf, info)
                    }
                }
                codec.releaseOutputBuffer(index, false)
                if (eos) listener.onEndOfStream()
            } catch (e: Exception) {
                listener.onError(e)
            }
        }

        override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) = listener.onError(e)

        override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) = listener.onFormat(format)
    }

    companion object {
        fun pickEncoder(mime: String, width: Int, height: Int, fps: Int): MediaCodecInfo? {
            val candidates = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { info ->
                info.isEncoder && info.supportedTypes.any { it.equals(mime, ignoreCase = true) } &&
                    runCatching {
                        info.getCapabilitiesForType(mime).videoCapabilities?.areSizeAndRateSupported(width, height, fps.toDouble()) == true
                    }.getOrDefault(false)
            }
            return candidates.firstOrNull { it.isHardwareAccelerated && !it.isSoftwareOnly } ?: candidates.firstOrNull()
        }
    }
}
