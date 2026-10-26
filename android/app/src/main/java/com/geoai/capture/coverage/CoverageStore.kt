package com.geoai.capture.coverage

import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.FileOutputStream

/**
 * Cumulative road coverage across sessions, next to the sessions folder:
 *   coverage/cells.bin      covered cells (big-endian Long keys, append-only)
 *   coverage/coverage.json  new metres per session; the total is their sum
 * A session is counted once. Sessions recorded before this existed, or cut short by a crash, are
 * counted from their gnss_fixes.csv by [backfill].
 */
class CoverageStore(dir: File) {
    private val dir = File(dir, "coverage")
    private val cellsFile = File(this.dir, "cells.bin")
    private val indexFile = File(this.dir, "coverage.json")

    fun loadCells(): HashSet<Long> {
        val set = HashSet<Long>()
        if (!cellsFile.isFile) return set
        DataInputStream(cellsFile.inputStream().buffered()).use { s ->
            try {
                while (true) set += s.readLong()
            } catch (_: EOFException) {
                // A torn last record (power loss mid-append) is simply dropped.
            }
        }
        return set
    }

    /** New metres per counted session id. */
    fun sessions(): Map<String, Double> {
        val o = runCatching { JSONObject(indexFile.readText()).getJSONObject("sessions") }.getOrNull() ?: return emptyMap()
        return o.keys().asSequence().associateWith { o.optDouble(it, 0.0) }
    }

    fun totalNewM(): Double = sessions().values.sum()

    /** Persists one session's result; cells first, so a crash between the two only re-counts nothing. */
    fun commit(sessionId: String, newM: Double, cells: List<Long>) {
        dir.mkdirs()
        if (cells.isNotEmpty()) {
            FileOutputStream(cellsFile, true).use { file ->
                // loadCells ignores a torn final Long. Remove those bytes before appending,
                // otherwise every subsequent cell is read at the wrong eight-byte boundary.
                val size = file.channel.size()
                file.channel.truncate(size - size % Long.SIZE_BYTES)
                DataOutputStream(file.buffered()).use { s ->
                    cells.forEach { s.writeLong(it) }
                    s.flush()
                    file.fd.sync()
                }
            }
        }
        val all = sessions().toMutableMap()
        all[sessionId] = newM
        val sess = JSONObject()
        all.toSortedMap().forEach { (k, v) -> sess.put(k, v) }
        val o = JSONObject()
            .put("version", 1)
            .put("cell_m", RoadCoverage.CELL_M)
            .put("total_new_m", all.values.sum())
            .put("sessions", sess)
        val tmp = File(dir, "coverage.json.tmp")
        tmp.writeText(o.toString(2))
        if (!tmp.renameTo(indexFile)) {
            indexFile.delete()
            tmp.renameTo(indexFile)
        }
    }

    /**
     * Counts every finished survey or test session in [sessionsDir] that is not counted yet, oldest first,
     * skipping [activeSessionId]. Returns the number of sessions added.
     */
    fun backfill(sessionsDir: File, activeSessionId: String?): Int {
        val done = sessions().keys
        val todo = (sessionsDir.listFiles() ?: emptyArray())
            .filter { it.isDirectory && it.name != activeSessionId && it.name !in done }
            .sortedBy { it.name }
        if (todo.isEmpty()) return 0
        val known = loadCells()
        var added = 0
        for (d in todo) {
            val meta = runCatching { JSONObject(File(d, "session.json").readText()) }.getOrNull() ?: continue
            if (meta.optString("session_type") !in COUNTED_TYPES) continue
            val fixes = File(d, "gnss_fixes.csv")
            if (!fixes.isFile) continue
            val rec = meta.optJSONObject("recording")
            val t0 = rec?.optLong("first_frame_t_ns", Long.MIN_VALUE) ?: Long.MIN_VALUE
            val t1 = rec?.optLong("last_frame_t_ns", Long.MAX_VALUE) ?: Long.MAX_VALUE
            val cov = RoadCoverage(known)
            readTrack(fixes, t0, t1).forEach { (lat, lon) -> cov.add(lat, lon) }
            val cells = cov.takeFresh()
            known += cells
            commit(meta.optString("session_id", d.name).ifBlank { d.name }, cov.newM, cells)
            added++
        }
        return added
    }

    companion object {
        val COUNTED_TYPES = setOf("survey", "test")

        /** The same fixes the live counter accepts: moving, ≤ 25 m accuracy, not mock, while recording. */
        fun accept(accM: Float?, speedMps: Float?, mock: Boolean): Boolean =
            accM != null && accM < 25f && speedMps != null && speedMps > 0.5f && !mock

        fun readTrack(fixes: File, t0: Long, t1: Long): List<Pair<Double, Double>> {
            val out = ArrayList<Pair<Double, Double>>()
            fixes.bufferedReader().useLines { lines ->
                val it = lines.iterator()
                if (!it.hasNext()) return@useLines
                val h = it.next().split(',')
                val iT = h.indexOf("t_ns")
                val iLat = h.indexOf("lat_deg")
                val iLon = h.indexOf("lon_deg")
                val iAcc = h.indexOf("h_acc_m")
                val iSpd = h.indexOf("speed_mps")
                val iMock = h.indexOf("is_mock")
                if (listOf(iT, iLat, iLon, iAcc, iSpd, iMock).any { i -> i < 0 }) return@useLines
                while (it.hasNext()) {
                    val c = it.next().split(',')
                    if (c.size < h.size) continue
                    val t = c[iT].toLongOrNull() ?: continue
                    if (t < t0 || t > t1) continue
                    if (!accept(c[iAcc].toFloatOrNull(), c[iSpd].toFloatOrNull(), c[iMock] == "1")) continue
                    val lat = c[iLat].toDoubleOrNull() ?: continue
                    val lon = c[iLon].toDoubleOrNull() ?: continue
                    out += lat to lon
                }
            }
            return out
        }
    }
}
