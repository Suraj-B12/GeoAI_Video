package com.geoai.capture.io

import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class AsyncLineWriterTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test(timeout = 10_000) fun concurrentProducersKeepEveryCompleteLineAcrossWriters() {
        val io = IoScheduler()
        val writers = (0 until 4).map { AsyncLineWriter(File(tmp.root, "$it.csv"), "producer,sample", io) }
        val producers = Executors.newFixedThreadPool(8)
        try {
            val tasks = (0 until 8).map { producer ->
                producers.submit { for (sample in 0 until 500) writers[producer % 4].write("$producer,$sample") }
            }
            tasks.forEach { it.get(5, TimeUnit.SECONDS) }
        } finally {
            producers.shutdownNow()
            io.shutdown()
        }
        for ((index, writer) in writers.withIndex()) {
            val rows = writer.file.readLines()
            assertEquals("producer,sample", rows.first())
            val expected = (0 until 8).filter { it % 4 == index }
                .flatMap { producer -> (0 until 500).map { "$producer,$it" } }.toSet()
            assertEquals(expected, rows.drop(1).toSet())
            assertEquals(1001, rows.size)
            assertEquals(1001L, writer.linesWritten.get())
        }
    }

    @Test(timeout = 10_000) fun closeDrainsOnceAndRejectsLaterWrites() {
        val io = IoScheduler()
        val writer = AsyncLineWriter(File(tmp.root, "events.csv"), "event", io)
        try {
            writer.write("before close")
            writer.close()
            writer.write("after close")
            writer.close()
        } finally {
            io.shutdown()
        }
        assertEquals(listOf("event", "before close"), writer.file.readLines())
    }

    @Test(timeout = 10_000) fun oneFailedStreamDoesNotCancelEverySubsequentFlush() {
        val io = IoScheduler()
        val broken = AsyncLineWriter(File(tmp.root, "broken.csv"), null, io)
        val healthy = AsyncLineWriter(File(tmp.root, "healthy.csv"), null, io)
        try {
            // Simulate a storage-write failure without depending on phone hardware or disk space.
            broken.closeStream()
            broken.write("cannot flush")
            healthy.write("first")
            waitFor { healthy.linesWritten.get() == 1L }
            healthy.write("second")
            waitFor { healthy.linesWritten.get() == 2L }
            assertEquals(listOf("first", "second"), healthy.file.readLines())
        } finally {
            io.shutdown()
        }
    }

    private fun waitFor(done: () -> Boolean) {
        val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3)
        while (!done() && System.nanoTime() < end) Thread.sleep(10)
        assertTrue("scheduled flush stopped", done())
    }
}
