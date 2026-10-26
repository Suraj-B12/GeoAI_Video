package com.geoai.capture.io

import java.io.File
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SessionRecoveryTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun prefersLastCompleteFrameOverLaterSensorTimestamp() {
        File(tmp.root, "frames.csv").writeText("frame,t_ns,iso\n0,100,50\n1,200,50\n2,999,50")
        File(tmp.root, "imu.csv").writeText("t_ns,x,y,z\n1000,1,2,3\n")
        assertEquals(200L, SessionStore.lastLoggedTimestamp(tmp.root))
    }

    @Test fun skipsMalformedRowsAndFallsBackToSensorLog() {
        File(tmp.root, "frames.csv").writeText("frame,t_ns,iso\n0,bad,50\n1,100\n")
        File(tmp.root, "imu.csv").writeText("t_ns,x,y,z\n1000,1,2,3\n2000,1,2,3\n")
        assertEquals(2000L, SessionStore.lastLoggedTimestamp(tmp.root))
    }

    @Test fun emptyOrHeaderOnlySessionHasNoTimestamp() {
        assertNull(SessionStore.lastLoggedTimestamp(tmp.root))
        File(tmp.root, "frames.csv").writeText("frame,t_ns,iso\n")
        File(tmp.root, "imu.csv").writeText("")
        assertNull(SessionStore.lastLoggedTimestamp(tmp.root))
    }

    @Test fun scansLongSensorLogAndIgnoresPartialFinalRecord() {
        File(tmp.root, "imu.csv").bufferedWriter().use { out ->
            out.write("t_ns,x,y,z\n")
            for (i in 1..200_000) out.write("${i * 5_000_000L},1,2,3\n")
            out.write("999999999999999,1,2,3")
        }
        assertEquals(1_000_000_000_000L, SessionStore.lastLoggedTimestamp(tmp.root))
    }
}
