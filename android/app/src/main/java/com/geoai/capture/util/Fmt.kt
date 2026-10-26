package com.geoai.capture.util

import java.util.Locale

/**
 * Locale-independent formatting for everything written to disk.
 *
 * Never call String.format without Locale.ROOT: on a phone set to a comma-decimal locale it
 * writes "12,97" and silently breaks every CSV reader on the PC.
 * Kotlin/Java toString() of Float/Double is locale-independent; it may use E-notation (1.0E-4),
 * which the session spec allows.
 */
object Fmt {
    fun fixed(v: Double, decimals: Int): String = String.format(Locale.ROOT, "%.${decimals}f", v)

    fun opt(v: Double?): String = if (v == null || v.isNaN() || v.isInfinite()) "" else v.toString()
    fun opt(v: Float?): String = if (v == null || v.isNaN() || v.isInfinite()) "" else v.toString()
    fun opt(v: Long?): String = v?.toString() ?: ""
    fun opt(v: Int?): String = v?.toString() ?: ""
    fun optFixed(v: Double?, decimals: Int): String =
        if (v == null || v.isNaN() || v.isInfinite()) "" else fixed(v, decimals)

    fun bool(v: Boolean): String = if (v) "1" else "0"

    /** One RFC 4180 field: quoted only when it contains a comma, quote, CR or LF. */
    fun field(s: String): String {
        if (s.none { it == ',' || it == '"' || it == '\n' || it == '\r' }) return s
        return "\"" + s.replace("\"", "\"\"") + "\""
    }

    /** Joins already-formatted numeric fields; use [field] for free text. */
    fun row(vararg fields: String): String = fields.joinToString(",")
}
