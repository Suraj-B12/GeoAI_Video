package com.geoai.capture.coverage

import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Road length driven for the first time ("non-repeating km").
 *
 * The ground is cut into [CELL_M] cells. Each GNSS step is split into pieces of at most [PIECE_M]; a piece
 * is new unless a cell within [MATCH_M] of it (about 3 sigma of the scatter between two passes of a phone
 * GNSS, plus a lane's offset) was covered before: by an earlier session, or earlier in this one more than [OWN_TRAIL_M] of path ago, so the
 * cells just behind the vehicle never count as a repeat. A road driven again in either direction counts once.
 *
 * Not thread-safe: the caller serialises [add], [breakTrack] and [takeFresh].
 */
class RoadCoverage(private val known: Set<Long>) {
    /** Cells first covered in this run → path distance (m) at which they were marked. */
    private val own = HashMap<Long, Double>()
    private val fresh = ArrayList<Long>()
    private var prevLat = Double.NaN
    private var prevLon = Double.NaN

    /** All accepted travel, metres. */
    var pathM = 0.0
        private set

    /** Travel over road not covered before, metres. */
    var newM = 0.0
        private set

    /** Adds the next accepted position; returns the new metres it added. */
    fun add(lat: Double, lon: Double): Double {
        if (prevLat.isNaN()) {
            prevLat = lat
            prevLon = lon
            return 0.0
        }
        val (dn, de) = offsetM(prevLat, prevLon, lat, lon)
        val d = sqrt(dn * dn + de * de)
        if (d > MAX_STEP_M) {
            // A jump or a long GNSS gap: the straight line between the fixes is not the road.
            prevLat = lat
            prevLon = lon
            return 0.0
        }
        var added = 0.0
        val n = ceil(d / PIECE_M).toInt().coerceAtLeast(1)
        val piece = d / n
        for (i in 0 until n) {
            val f = (i + 0.5) / n
            val pLat = prevLat + (lat - prevLat) * f
            val pLon = prevLon + (lon - prevLon) * f
            val at = pathM + piece * (i + 0.5)
            if (!coveredBefore(pLat, pLon, at)) added += piece
            val cell = cellOf(pLat, pLon)
            if (cell !in known && cell !in own) {
                own[cell] = at
                fresh += cell
            }
        }
        pathM += d
        newM += added
        prevLat = lat
        prevLon = lon
        return added
    }

    /** The next position starts a new track (after a GNSS outage or a pause in recording). */
    fun breakTrack() {
        prevLat = Double.NaN
        prevLon = Double.NaN
    }

    /** Cells first covered since the last call, for persisting. */
    fun takeFresh(): List<Long> = fresh.toList().also { fresh.clear() }

    private fun coveredBefore(lat: Double, lon: Double, at: Double): Boolean {
        val mLat = M_PER_DEG_LAT
        val mLon = M_PER_DEG_LAT * cos(Math.toRadians(lat))
        for (dy in -REACH..REACH) for (dx in -REACH..REACH) {
            if (dy * dy + dx * dx > REACH * REACH) continue
            val c = cellOf(lat + dy * CELL_M / mLat, lon + dx * CELL_M / mLon)
            if (c in known) return true
            val markedAt = own[c] ?: continue
            if (markedAt < at - OWN_TRAIL_M) return true
        }
        return false
    }

    companion object {
        const val CELL_M = 4.0
        const val MATCH_M = 12.0
        private const val REACH = (MATCH_M / CELL_M).toInt()
        const val PIECE_M = 4.0
        const val OWN_TRAIL_M = 50.0
        const val MAX_STEP_M = 100.0
        private const val M_PER_DEG_LAT = 111_320.0

        /** Cell key: row of [CELL_M] in latitude, column of [CELL_M] at that row's latitude. */
        fun cellOf(lat: Double, lon: Double): Long {
            val row = floor(lat * M_PER_DEG_LAT / CELL_M).toLong()
            val rowLat = (row + 0.5) * CELL_M / M_PER_DEG_LAT
            val col = floor(lon * M_PER_DEG_LAT * cos(Math.toRadians(rowLat)) / CELL_M).toLong()
            return (row shl 32) or (col and 0xffffffffL)
        }

        /** North and east offsets in metres (local flat-earth; exact enough below [MAX_STEP_M]). */
        fun offsetM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Pair<Double, Double> {
            val n = (lat2 - lat1) * M_PER_DEG_LAT
            val e = (lon2 - lon1) * M_PER_DEG_LAT * cos(Math.toRadians((lat1 + lat2) / 2))
            return n to e
        }
    }
}
