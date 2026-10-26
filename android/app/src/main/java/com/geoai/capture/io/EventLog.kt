package com.geoai.capture.io

import com.geoai.capture.util.Clocks
import com.geoai.capture.util.Fmt
import org.json.JSONObject

/** events.csv: t_ns,type,value_json. Thread-safe (the writer queue is lock-free). */
class EventLog(private val writer: AsyncLineWriter) {
    fun log(type: String, value: JSONObject? = null, tNs: Long = Clocks.elapsedNs()) {
        writer.write(Fmt.row(tNs.toString(), type, Fmt.field(value?.toString() ?: "{}")))
    }

    companion object {
        const val HEADER = "t_ns,type,value_json"
    }
}
