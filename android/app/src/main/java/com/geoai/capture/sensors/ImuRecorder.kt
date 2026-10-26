package com.geoai.capture.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import com.geoai.capture.io.AsyncLineWriter
import org.json.JSONArray
import org.json.JSONObject

/**
 * Motion sensors on their own thread. SensorEvent.timestamp is on the elapsedRealtime clock, the
 * same as camera frames and GNSS fixes. Values are written with Float.toString (locale-independent,
 * may use E-notation).
 */
class ImuRecorder(context: Context, private val handler: Handler, private val out: AsyncLineWriter) {

    private val sm = context.getSystemService(SensorManager::class.java)

    private data class Channel(val type: Int, val name: String, val periodUs: Int)

    private val channels = listOf(
        Channel(Sensor.TYPE_ACCELEROMETER, "acc", 5_000),
        Channel(Sensor.TYPE_GYROSCOPE, "gyro", 5_000),
        Channel(Sensor.TYPE_GRAVITY, "gravity", 20_000),
        Channel(Sensor.TYPE_GAME_ROTATION_VECTOR, "game_rv", 20_000),
        Channel(Sensor.TYPE_MAGNETIC_FIELD, "mag", 20_000),
    )
    private val names = HashMap<Int, String>()

    /** Latest gravity reading in device axes (reaction to gravity: points up). */
    @Volatile var latestGravity: FloatArray? = null
        private set
    @Volatile var started = false
        private set

    private val listener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            val name = names[e.sensor.type] ?: return
            val v = e.values
            val w = if (e.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR && v.size >= 4) v[3].toString() else ""
            out.write("${e.timestamp},$name,${v[0]},${v.getOrElse(1) { 0f }},${v.getOrElse(2) { 0f }},$w,${e.accuracy}")
            if (e.sensor.type == Sensor.TYPE_GRAVITY) latestGravity = floatArrayOf(v[0], v[1], v[2])
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit
    }

    fun start() {
        if (started) return
        for (c in channels) {
            val s = sm.getDefaultSensor(c.type) ?: continue
            names[c.type] = c.name
            sm.registerListener(listener, s, c.periodUs, 0, handler)
        }
        started = true
    }

    fun stop() {
        if (!started) return
        sm.unregisterListener(listener)
        started = false
    }

    fun describe(): JSONArray = JSONArray().also { arr ->
        for (c in channels) {
            val s = sm.getDefaultSensor(c.type) ?: continue
            arr.put(
                JSONObject()
                    .put("name", c.name)
                    .put("android_type", c.type)
                    .put("sensor_name", s.name)
                    .put("vendor", s.vendor)
                    .put("requested_period_us", c.periodUs)
                    .put("min_delay_us", s.minDelay)
                    .put("max_range", s.maximumRange.toDouble())
                    .put("resolution", s.resolution.toDouble()),
            )
        }
    }

    companion object {
        const val HEADER = "t_ns,sensor,x,y,z,w,accuracy"
    }
}
