package com.geoai.capture.util

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/** Lifecycle breadcrumbs in logcat (tag GeoAI): `adb logcat -s GeoAI`. Never per-frame. */
object Diag {
    private const val TAG = "GeoAI"
    fun i(msg: String) {
        Log.i(TAG, "[${Thread.currentThread().name}] $msg")
    }

    fun e(msg: String, t: Throwable? = null) {
        Log.e(TAG, "[${Thread.currentThread().name}] $msg", t)
    }

    /** Error details for events.csv: class, message, top stack frames and the cause chain. */
    fun errorJson(t: Throwable): JSONObject {
        val o = JSONObject().put("error", t.toString())
        o.put("stack", JSONArray(t.stackTrace.take(12).map { it.toString() }))
        var cause = t.cause
        val causes = JSONArray()
        while (cause != null && causes.length() < 4) {
            causes.put(cause.toString())
            cause = cause.cause
        }
        if (causes.length() > 0) o.put("causes", causes)
        return o
    }
}
