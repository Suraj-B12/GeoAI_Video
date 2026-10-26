package com.geoai.capture.sensors

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssMeasurementRequest
import android.location.GnssMeasurementsEvent
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.location.LocationRequest
import android.location.OnNmeaMessageListener
import android.os.Build
import com.geoai.capture.io.AsyncLineWriter
import com.geoai.capture.util.Clocks
import com.geoai.capture.util.Fmt
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger

/**
 * Logs every GNSS fix from GPS_PROVIDER (never the fused provider, which blends Wi-Fi/cell
 * positions and smooths), the per-satellite status, raw NMEA, and optionally raw measurements in
 * Google GnssLogger "Raw" format for later PPK against CORS data.
 */
class GnssRecorder(
    context: Context,
    private val executor: Executor,
    private val fixes: AsyncLineWriter,
    private val status: AsyncLineWriter,
    private val nmea: AsyncLineWriter,
    private val raw: AsyncLineWriter?,
    private val onFix: (Location) -> Unit,
) {
    private val lm = context.getSystemService(LocationManager::class.java)
    private val satsUsed = AtomicInteger(-1)
    @Volatile var started = false
        private set
    val satellitesUsed: Int get() = satsUsed.get()
    /** elapsedRealtime of the last fix callback (0 = none yet); drives the watchdog in the service. */
    @Volatile var lastFixReceiptNs = 0L
        private set
    @Volatile var restarts = 0
        private set
    private var fullTracking = false

    /** Location services: master switch on and the GPS provider enabled. */
    val locationEnabled: Boolean
        get() = runCatching { lm.isLocationEnabled && lm.isProviderEnabled(LocationManager.GPS_PROVIDER) }.getOrDefault(false)

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) = handleFix(location)
        override fun onLocationChanged(locations: MutableList<Location>) = locations.forEach { handleFix(it) }
        // A request made while location was off can stay inactive after it is switched on: re-register.
        override fun onProviderEnabled(provider: String) {
            if (provider == LocationManager.GPS_PROVIDER) executor.execute { restart() }
        }
        override fun onProviderDisabled(provider: String) = Unit
    }

    private val statusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(s: GnssStatus) {
            val t = Clocks.elapsedNs()
            var used = 0
            for (i in 0 until s.satelliteCount) {
                if (s.usedInFix(i)) used++
                status.write(
                    Fmt.row(
                        t.toString(),
                        s.getSvid(i).toString(),
                        s.getConstellationType(i).toString(),
                        Fmt.opt(s.getCn0DbHz(i)),
                        if (s.hasCarrierFrequencyHz(i)) Fmt.opt(s.getCarrierFrequencyHz(i)) else "",
                        Fmt.bool(s.usedInFix(i)),
                        Fmt.opt(s.getElevationDegrees(i)),
                        Fmt.opt(s.getAzimuthDegrees(i)),
                        Fmt.bool(s.hasEphemerisData(i)),
                        Fmt.bool(s.hasAlmanacData(i)),
                        if (s.hasBasebandCn0DbHz(i)) Fmt.opt(s.getBasebandCn0DbHz(i)) else "",
                    ),
                )
            }
            satsUsed.set(used)
        }
    }

    private val nmeaListener = OnNmeaMessageListener { message, timestamp ->
        // The sentence contains commas, so it is written as one quoted CSV field.
        nmea.write(Fmt.row(Clocks.elapsedNs().toString(), timestamp.toString(), Fmt.field(message.trim())))
    }

    private val rawLogger = raw?.let { RawGnssLogger(it) }
    private val measurementCallback = object : GnssMeasurementsEvent.Callback() {
        override fun onGnssMeasurementsReceived(event: GnssMeasurementsEvent) {
            rawLogger?.log(event)
        }
    }

    private fun handleFix(l: Location) {
        val receipt = Clocks.elapsedNs()
        lastFixReceiptNs = receipt
        fixes.write(
            Fmt.row(
                l.elapsedRealtimeNanos.toString(),
                if (l.hasElapsedRealtimeUncertaintyNanos()) Fmt.opt(l.elapsedRealtimeUncertaintyNanos) else "",
                receipt.toString(),
                l.time.toString(),
                Fmt.fixed(l.latitude, 9),
                Fmt.fixed(l.longitude, 9),
                if (l.hasAltitude()) Fmt.fixed(l.altitude, 3) else "",
                if (Build.VERSION.SDK_INT >= 34 && l.hasMslAltitude()) Fmt.fixed(l.mslAltitudeMeters, 3) else "",
                if (l.hasAccuracy()) Fmt.opt(l.accuracy) else "",
                if (l.hasVerticalAccuracy()) Fmt.opt(l.verticalAccuracyMeters) else "",
                if (l.hasSpeed()) Fmt.opt(l.speed) else "",
                if (l.hasSpeedAccuracy()) Fmt.opt(l.speedAccuracyMetersPerSecond) else "",
                if (l.hasBearing()) Fmt.opt(l.bearing) else "",
                if (l.hasBearingAccuracy()) Fmt.opt(l.bearingAccuracyDegrees) else "",
                Fmt.bool(l.isMock),
                l.provider ?: "",
                satsUsed.get().let { if (it >= 0) it.toString() else "" },
            ),
        )
        onFix(l)
    }

    @SuppressLint("MissingPermission")
    fun start(fullTracking: Boolean) {
        if (started) return
        this.fullTracking = fullTracking
        val request = LocationRequest.Builder(0L)
            .setQuality(LocationRequest.QUALITY_HIGH_ACCURACY)
            .setMinUpdateIntervalMillis(0L)
            .setMaxUpdateDelayMillis(0L)
            .build()
        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, request, executor, locationListener)
        lm.registerGnssStatusCallback(executor, statusCallback)
        lm.addNmeaListener(executor, nmeaListener)
        if (rawLogger != null) {
            val req = GnssMeasurementRequest.Builder().setFullTracking(fullTracking).build()
            lm.registerGnssMeasurementsCallback(req, executor, measurementCallback)
        }
        started = true
    }

    /** Drops and re-creates all GNSS registrations (watchdog / provider re-enabled). */
    @Synchronized fun restart() {
        if (!started) return
        stop()
        start(fullTracking)
        restarts++
    }

    fun stop() {
        if (!started) return
        runCatching { lm.removeUpdates(locationListener) }
        runCatching { lm.unregisterGnssStatusCallback(statusCallback) }
        runCatching { lm.removeNmeaListener(nmeaListener) }
        if (rawLogger != null) runCatching { lm.unregisterGnssMeasurementsCallback(measurementCallback) }
        started = false
    }

    fun hardwareInfo(): Map<String, String> = buildMap {
        put("gnss_hardware_model_name", lm.gnssHardwareModelName ?: "")
        put("gnss_year_of_hardware", lm.gnssYearOfHardware.toString())
        put("gnss_capabilities", runCatching { lm.gnssCapabilities.toString() }.getOrDefault(""))
    }

    companion object {
        const val FIXES_HEADER =
            "t_ns,t_unc_ns,receipt_ns,utc_ms,lat_deg,lon_deg,alt_ellipsoid_m,alt_msl_m,h_acc_m,v_acc_m," +
                "speed_mps,speed_acc_mps,bearing_deg,bearing_acc_deg,is_mock,provider,n_sats_used"
        const val STATUS_HEADER =
            "receipt_ns,svid,constellation,cn0_dbhz,carrier_hz,used_in_fix,elevation_deg,azimuth_deg,has_ephemeris,has_almanac,baseband_cn0_dbhz"
        const val NMEA_HEADER = "receipt_ns,timestamp_ms,sentence"
    }
}
