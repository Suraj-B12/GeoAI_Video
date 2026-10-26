package com.geoai.capture.sensors

import android.location.GnssMeasurementsEvent
import com.geoai.capture.io.AsyncLineWriter
import com.geoai.capture.util.Fmt

/**
 * Raw GNSS measurements written as Google GnssLogger "Raw" records, so existing converters
 * (GnssLogger → RINEX, gnss_lib_py, the Google smartphone-decimeter tooling) can read them.
 * Deprecated carrier cycle/phase fields are left empty, exactly as GnssLogger does.
 */
class RawGnssLogger(private val out: AsyncLineWriter) {

    fun log(event: GnssMeasurementsEvent) {
        val c = event.clock
        val leap = if (c.hasLeapSecond()) c.leapSecond else DEFAULT_LEAP_SECONDS
        val fullBias = if (c.hasFullBiasNanos()) c.fullBiasNanos else null
        val bias = if (c.hasBiasNanos()) c.biasNanos else 0.0
        val utcMillis = fullBias?.let {
            val gpsNanos = c.timeNanos - (it + bias.toLong())
            gpsNanos / 1_000_000L + GPS_EPOCH_UNIX_MS - leap * 1000L
        }
        for (m in event.measurements) {
            out.write(
                Fmt.row(
                    "Raw",
                    Fmt.opt(utcMillis),
                    c.timeNanos.toString(),
                    if (c.hasLeapSecond()) c.leapSecond.toString() else "",
                    if (c.hasTimeUncertaintyNanos()) Fmt.opt(c.timeUncertaintyNanos) else "",
                    Fmt.opt(fullBias),
                    if (c.hasBiasNanos()) Fmt.opt(c.biasNanos) else "",
                    if (c.hasBiasUncertaintyNanos()) Fmt.opt(c.biasUncertaintyNanos) else "",
                    if (c.hasDriftNanosPerSecond()) Fmt.opt(c.driftNanosPerSecond) else "",
                    if (c.hasDriftUncertaintyNanosPerSecond()) Fmt.opt(c.driftUncertaintyNanosPerSecond) else "",
                    c.hardwareClockDiscontinuityCount.toString(),
                    m.svid.toString(),
                    Fmt.opt(m.timeOffsetNanos),
                    m.state.toString(),
                    m.receivedSvTimeNanos.toString(),
                    m.receivedSvTimeUncertaintyNanos.toString(),
                    Fmt.opt(m.cn0DbHz),
                    Fmt.opt(m.pseudorangeRateMetersPerSecond),
                    Fmt.opt(m.pseudorangeRateUncertaintyMetersPerSecond),
                    m.accumulatedDeltaRangeState.toString(),
                    Fmt.opt(m.accumulatedDeltaRangeMeters),
                    Fmt.opt(m.accumulatedDeltaRangeUncertaintyMeters),
                    if (m.hasCarrierFrequencyHz()) Fmt.opt(m.carrierFrequencyHz) else "",
                    "", "", "", // CarrierCycles, CarrierPhase, CarrierPhaseUncertainty (deprecated)
                    m.multipathIndicator.toString(),
                    if (m.hasSnrInDb()) Fmt.opt(m.snrInDb) else "",
                    m.constellationType.toString(),
                    @Suppress("DEPRECATION") if (m.hasAutomaticGainControlLevelDb()) Fmt.opt(m.automaticGainControlLevelDb) else "",
                    if (m.hasBasebandCn0DbHz()) Fmt.opt(m.basebandCn0DbHz) else "",
                    if (m.hasFullInterSignalBiasNanos()) Fmt.opt(m.fullInterSignalBiasNanos) else "",
                    if (m.hasFullInterSignalBiasUncertaintyNanos()) Fmt.opt(m.fullInterSignalBiasUncertaintyNanos) else "",
                    if (m.hasSatelliteInterSignalBiasNanos()) Fmt.opt(m.satelliteInterSignalBiasNanos) else "",
                    if (m.hasSatelliteInterSignalBiasUncertaintyNanos()) Fmt.opt(m.satelliteInterSignalBiasUncertaintyNanos) else "",
                    if (m.hasCodeType()) m.codeType else "",
                    if (c.hasElapsedRealtimeNanos()) c.elapsedRealtimeNanos.toString() else "",
                ),
            )
        }
    }

    companion object {
        /** GPS epoch (1980-01-06) in Unix milliseconds. */
        const val GPS_EPOCH_UNIX_MS = 315_964_800_000L
        /** GPS−UTC leap seconds since 2017-01-01, used only if the chipset does not report it. */
        const val DEFAULT_LEAP_SECONDS = 18

        const val HEADER =
            "# Raw,utcTimeMillis,TimeNanos,LeapSecond,TimeUncertaintyNanos,FullBiasNanos,BiasNanos,BiasUncertaintyNanos," +
                "DriftNanosPerSecond,DriftUncertaintyNanosPerSecond,HardwareClockDiscontinuityCount,Svid,TimeOffsetNanos,State," +
                "ReceivedSvTimeNanos,ReceivedSvTimeUncertaintyNanos,Cn0DbHz,PseudorangeRateMetersPerSecond," +
                "PseudorangeRateUncertaintyMetersPerSecond,AccumulatedDeltaRangeState,AccumulatedDeltaRangeMeters," +
                "AccumulatedDeltaRangeUncertaintyMeters,CarrierFrequencyHz,CarrierCycles,CarrierPhase,CarrierPhaseUncertainty," +
                "MultipathIndicator,SnrInDb,ConstellationType,AgcDb,BasebandCn0DbHz,FullInterSignalBiasNanos," +
                "FullInterSignalBiasUncertaintyNanos,SatelliteInterSignalBiasNanos,SatelliteInterSignalBiasUncertaintyNanos," +
                "CodeType,ChipsetElapsedRealtimeNanos"
    }
}
