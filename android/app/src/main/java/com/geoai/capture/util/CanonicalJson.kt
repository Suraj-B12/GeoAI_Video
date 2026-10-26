package com.geoai.capture.util

import org.json.JSONArray
import org.json.JSONObject

/**
 * Canonical JSON (sorted keys, no whitespace) used for the signed manifest payload.
 * Byte-identical to Python's json.dumps(obj, sort_keys=True, separators=(",", ":"),
 * ensure_ascii=False) as long as the payload holds only strings, integers, booleans, null,
 * objects and arrays, which is all the manifest payload contains.
 */
object CanonicalJson {
    fun encode(value: Any?): String = StringBuilder().also { append(it, value) }.toString()

    private fun append(sb: StringBuilder, v: Any?) {
        when (v) {
            null, JSONObject.NULL -> sb.append("null")
            is JSONObject -> {
                sb.append('{')
                v.keys().asSequence().toList().sorted().forEachIndexed { i, k ->
                    if (i > 0) sb.append(',')
                    quote(sb, k)
                    sb.append(':')
                    append(sb, v.get(k))
                }
                sb.append('}')
            }
            is JSONArray -> {
                sb.append('[')
                for (i in 0 until v.length()) {
                    if (i > 0) sb.append(',')
                    append(sb, v.get(i))
                }
                sb.append(']')
            }
            is String -> quote(sb, v)
            is Boolean -> sb.append(v.toString())
            is Int, is Long, is Short, is Byte -> sb.append(v.toString())
            is Float, is Double ->
                throw IllegalArgumentException("Floating-point values are not allowed in canonical payloads")
            else -> quote(sb, v.toString())
        }
    }

    private fun quote(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                '\b' -> sb.append("\\b")
                '\u000C' -> sb.append("\\f")
                else -> if (c < ' ') sb.append(String.format(java.util.Locale.ROOT, "\\u%04x", c.code)) else sb.append(c)
            }
        }
        sb.append('"')
    }
}
