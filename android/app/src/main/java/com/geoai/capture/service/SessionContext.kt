package com.geoai.capture.service

import com.geoai.capture.camera.FrameRow
import com.geoai.capture.camera.SegmentWriter
import com.geoai.capture.io.AsyncLineWriter
import com.geoai.capture.io.EventLog
import com.geoai.capture.io.IoScheduler
import com.geoai.capture.sensors.ClockSyncRecorder
import com.geoai.capture.sensors.GnssRecorder
import com.geoai.capture.sensors.ImuRecorder
import com.geoai.capture.sensors.RawGnssLogger
import com.geoai.capture.sensors.ThermalRecorder
import com.geoai.capture.settings.SurveyParams
import com.geoai.capture.util.Fmt
import java.io.File

/** All files of one session folder. Column lists here ARE the data contract (docs/session-package-spec.md). */
class SessionContext(val dir: File, val id: String, val params: SurveyParams, rawGnss: Boolean) {
    val io = IoScheduler()
    private fun w(name: String, header: String?) = AsyncLineWriter(File(dir, name), header, io)

    val frames = w("frames.csv", FRAMES_HEADER)
    val segments = w("segments.csv", SegmentWriter.HEADER)
    val gnssFixes = w("gnss_fixes.csv", GnssRecorder.FIXES_HEADER)
    val gnssStatus = w("gnss_status.csv", GnssRecorder.STATUS_HEADER)
    val nmea = w("nmea.csv", GnssRecorder.NMEA_HEADER)
    val gnssRaw: AsyncLineWriter? = if (rawGnss) w("gnss_raw.txt", RAW_PREAMBLE) else null
    val imu = w("imu.csv", ImuRecorder.HEADER)
    val clock = w("clock_sync.csv", ClockSyncRecorder.HEADER)
    val thermal = w("thermal.csv", ThermalRecorder.HEADER)
    val analysis = w("analysis.csv", ANALYSIS_HEADER)
    val exposure = w("exposure_control.csv", EXPOSURE_HEADER)
    val events = EventLog(w("events.csv", EventLog.HEADER))

    /**
     * Formats one frames.csv row. t_ns is the frame's start-of-exposure on the session clock.
     * For an unmatched frame it is estimated from its PTS (sensor_ts_ns is then empty).
     * [sensorToSessionNs] converts the camera clock to elapsedRealtime: 0 for REALTIME cameras,
     * elapsed − monotonic for UNKNOWN-source cameras.
     */
    fun frameRow(r: FrameRow, sensorToSessionNs: Long): String {
        val m = r.meta
        val f = r.frame
        val sensorTs = m?.sensorTsNs
        val estimated = r.offsetNs?.let { f.ptsUs * 1000L - it }
        val t = (sensorTs ?: estimated)?.plus(sensorToSessionNs)
        return Fmt.row(
            f.segment.toString(),
            f.frameInSegment.toString(),
            f.frameGlobal.toString(),
            f.ptsUs.toString(),
            Fmt.opt(sensorTs),
            Fmt.opt(t),
            Fmt.opt(r.offsetNs),
            Fmt.opt(r.residualNs),
            Fmt.opt(m?.frameNumber),
            Fmt.opt(m?.exposureNs),
            Fmt.opt(m?.frameDurationNs),
            Fmt.opt(m?.iso),
            Fmt.opt(m?.rsSkewNs),
            Fmt.opt(m?.focusDiopters),
            Fmt.opt(m?.focalLengthMm),
            Fmt.opt(m?.lensState),
            Fmt.opt(m?.oisMode),
            Fmt.opt(m?.eisMode),
            Fmt.opt(m?.aeState),
            Fmt.opt(m?.afState),
            m?.awbGains ?: "",
            Fmt.opt(m?.zoomRatio),
            Fmt.bool(f.isKeyframe),
            f.bytes.toString(),
        )
    }

    fun closeAll() {
        io.shutdown()
    }

    companion object {
        const val FRAMES_HEADER =
            "segment,frame_in_segment,frame_global,pts_us,sensor_ts_ns,t_ns,offset_ns,match_residual_ns,capture_frame_number," +
                "exposure_ns,frame_duration_ns,iso,rs_skew_ns,focus_diopters,focal_length_mm,lens_state,ois_mode,eis_mode," +
                "ae_state,af_state,awb_gains,zoom_ratio,is_keyframe,encoded_bytes"
        const val ANALYSIS_HEADER = "t_ns,mean_luma,highlight_frac,dark_frac,sharpness,exposure_ns,iso,speed_mps"
        const val EXPOSURE_HEADER = "t_ns,reason,exposure_ns,iso,low_light,speed_mps,mean_luma"
        val RAW_PREAMBLE = "# GeoAI Capture raw GNSS log in GnssLogger 'Raw' record format\n#\n" + RawGnssLogger.HEADER + "\n#"
    }
}
