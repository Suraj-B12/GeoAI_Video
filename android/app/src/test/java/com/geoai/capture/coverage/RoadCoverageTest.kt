package com.geoai.capture.coverage

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.Random
import kotlin.math.cos

class RoadCoverageTest {
    @get:Rule val tmp = TemporaryFolder()

    private val lat0 = 12.9256
    private val lon0 = 77.5643
    private val mLat = 111_320.0
    private val mLon = 111_320.0 * cos(Math.toRadians(lat0))

    /** Position [n] metres north and [e] metres east of the origin. */
    private fun at(n: Double, e: Double): Pair<Double, Double> = lat0 + n / mLat to lon0 + e / mLon

    /**
     * A drive north along x = [e] from [from] to [to] metres, one fix every [step] metres. With [rnd], phone-GNSS-like
     * error: an offset per pass (sigma 3 m per axis; it drifts slowly, so it is near-constant over a pass) plus 0.5 m jitter.
     */
    private fun drive(cov: RoadCoverage, from: Double, to: Double, e: Double = 0.0, step: Double = 8.0, rnd: Random? = null) {
        val bn = (rnd?.nextGaussian() ?: 0.0) * 3.0
        val be = (rnd?.nextGaussian() ?: 0.0) * 3.0
        val n = (kotlin.math.abs(to - from) / step).toInt()
        for (i in 0..n) {
            val jn = (rnd?.nextGaussian() ?: 0.0) * 0.5
            val je = (rnd?.nextGaussian() ?: 0.0) * 0.5
            val (la, lo) = at(from + (to - from) * i / n + bn + jn, e + be + je)
            cov.add(la, lo)
        }
    }

    @Test fun straightNewRoadCountsInFull() {
        val cov = RoadCoverage(emptySet())
        drive(cov, 0.0, 1000.0)
        assertEquals(1000.0, cov.pathM, 1.0)
        assertEquals(1000.0, cov.newM, 1.0)
    }

    @Test fun sameRoadInLaterSessionCountsNothingEvenWithGnssNoise() {
        val first = RoadCoverage(emptySet())
        drive(first, 0.0, 1000.0, rnd = Random(1))
        val known = first.takeFresh().toHashSet()
        // Opposite direction, other lane (3.5 m across), independent 3 m GNSS noise.
        val second = RoadCoverage(known)
        drive(second, 1000.0, 0.0, e = 3.5, rnd = Random(2))
        assertTrue("path ${second.pathM}", second.pathM > 950)
        assertTrue("repeat counted as new: ${second.newM} m", second.newM < 30)
    }

    @Test fun outAndBackInOneSessionCountsOnce() {
        val cov = RoadCoverage(emptySet())
        drive(cov, 0.0, 500.0, rnd = Random(3))
        drive(cov, 500.0, 0.0, e = 3.5, rnd = Random(4))
        // 500 m new; the turnaround adds at most about the own-trail allowance.
        assertTrue("new ${cov.newM}", cov.newM in 490.0..500.0 + RoadCoverage.OWN_TRAIL_M + 10)
    }

    @Test fun parallelRoadFiftyMetresAwayIsNew() {
        val first = RoadCoverage(emptySet())
        drive(first, 0.0, 500.0)
        val second = RoadCoverage(first.takeFresh().toHashSet())
        drive(second, 0.0, 500.0, e = 50.0)
        assertEquals(500.0, second.newM, 5.0)
    }

    @Test fun partlyOverlappingRouteCountsOnlyTheNewPart() {
        val first = RoadCoverage(emptySet())
        drive(first, 0.0, 600.0, rnd = Random(5))
        val second = RoadCoverage(first.takeFresh().toHashSet())
        drive(second, 300.0, 1000.0, rnd = Random(6))
        assertEquals(400.0, second.newM, 25.0)
    }

    @Test fun standingStillAndJumpsAddNothing() {
        val cov = RoadCoverage(emptySet())
        val (la, lo) = at(0.0, 0.0)
        repeat(10) { cov.add(la, lo) }
        val (la2, lo2) = at(5000.0, 0.0)
        cov.add(la2, lo2)
        assertEquals(0.0, cov.pathM, 1e-9)
        cov.breakTrack()
        drive(cov, 5000.0, 5100.0)
        assertEquals(100.0, cov.newM, 1.0)
    }

    @Test fun storeCountsEachSessionOnceAndBackfillsFromCsv() {
        val root = tmp.newFolder("GeoAI")
        val sessions = File(root, "sessions").apply { mkdirs() }
        fun session(id: String, type: String, from: Double, to: Double) {
            val d = File(sessions, id).apply { mkdirs() }
            File(d, "session.json").writeText(
                """{"session_id":"$id","session_type":"$type","recording":{"first_frame_t_ns":0,"last_frame_t_ns":999000000000}}""",
            )
            val sb = StringBuilder("t_ns,t_unc_ns,receipt_ns,utc_ms,lat_deg,lon_deg,alt_ellipsoid_m,alt_msl_m,h_acc_m,v_acc_m," +
                "speed_mps,speed_acc_mps,bearing_deg,bearing_acc_deg,is_mock,provider,n_sats_used\n")
            val n = ((to - from) / 8).toInt()
            for (i in 0..n) {
                val (la, lo) = at(from + (to - from) * i / n, 0.0)
                sb.append("${i * 1_000_000_000L},,0,0,$la,$lo,,,4.0,,8.0,,,,0,gps,9\n")
            }
            File(d, "gnss_fixes.csv").writeText(sb.toString())
        }
        session("20260101T000000Z_A", "survey", 0.0, 800.0)
        session("20260102T000000Z_B", "test", 400.0, 1200.0)
        session("20260103T000000Z_C", "calibration_mount", 0.0, 3000.0)
        val store = CoverageStore(root)
        assertEquals(2, store.backfill(sessions, null)) // the calibration session is not road coverage
        assertEquals(0, store.backfill(sessions, null))
        val per = store.sessions()
        assertEquals(800.0, per.getValue("20260101T000000Z_A"), 10.0)
        assertEquals(400.0, per.getValue("20260102T000000Z_B"), 20.0)
        assertEquals(1200.0, store.totalNewM(), 25.0)
        // The cells survive a reload: driving the same road again adds nothing.
        val again = RoadCoverage(store.loadCells())
        drive(again, 0.0, 1200.0)
        assertTrue("new ${again.newM}", again.newM < 10)
    }

    @Test fun storeRepairsTornCellBeforeAppending() {
        val root = tmp.newFolder("torn")
        val store = CoverageStore(root)
        val first = 0x0123456789abcdefL
        val second = 0x1020304050607080L
        store.commit("first", 4.0, listOf(first))
        val cells = File(root, "coverage/cells.bin")
        cells.appendBytes(byteArrayOf(0x12, 0x34, 0x56))
        assertEquals(setOf(first), store.loadCells())
        store.commit("second", 4.0, listOf(second))
        assertEquals(setOf(first, second), store.loadCells())
        assertEquals(16L, cells.length())
        assertEquals(8.0, store.totalNewM(), 0.0)
    }

    /** The pilot's real S21 sessions, when present in the repo (`field/sessions`). Prints the figures. */
    @Test fun realSessions() {
        val dir = File("../../field/sessions")
        assumeTrue(dir.isDirectory)
        val root = tmp.newFolder("real")
        val sessions = File(root, "sessions").apply { mkdirs() }
        dir.listFiles()!!.filter { File(it, "gnss_fixes.csv").isFile }.forEach { s ->
            val d = File(sessions, s.name).apply { mkdirs() }
            File(s, "session.json").copyTo(File(d, "session.json"))
            File(s, "gnss_fixes.csv").copyTo(File(d, "gnss_fixes.csv"))
        }
        val store = CoverageStore(root)
        store.backfill(sessions, null)
        store.sessions().forEach { (id, m) -> println("coverage $id new_m=%.1f".format(m)) }
        println("coverage total_new_m=%.1f".format(store.totalNewM()))
        assertTrue(store.totalNewM() > 0)
    }
}
