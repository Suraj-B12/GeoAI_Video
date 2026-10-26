package com.geoai.capture.io

import java.io.BufferedOutputStream
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * One shared I/O thread for all text logs of a session. Producers (camera, GNSS, IMU threads)
 * only enqueue strings, so disk latency can never stall a camera or sensor callback.
 * Every 250 ms queued lines are written and flushed to the OS (survives an app crash);
 * every [syncEveryMs] they are fsync'ed (survives a phone shutdown).
 */
class IoScheduler(private val syncEveryMs: Long = 5_000) {
    private val exec: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor { r ->
        Thread(r, "geoai-io").apply { priority = Thread.NORM_PRIORITY - 1 }
    }
    private val writers = mutableListOf<AsyncLineWriter>()
    private var lastSyncMs = System.currentTimeMillis()

    init {
        exec.scheduleWithFixedDelay({ drainAll() }, 250, 250, TimeUnit.MILLISECONDS)
    }

    @Synchronized internal fun register(w: AsyncLineWriter) { writers += w }

    private fun drainAll() {
        val now = System.currentTimeMillis()
        val sync = now - lastSyncMs >= syncEveryMs
        if (sync) lastSyncMs = now
        val snapshot = synchronized(this) { writers.toList() }
        for (w in snapshot) w.drain(sync)
    }

    /** Final drain + fsync + close, executed on the I/O thread; blocks the caller until done. */
    internal fun close(w: AsyncLineWriter) {
        exec.submit {
            w.drain(true)
            w.closeStream()
            synchronized(this) { writers -= w }
        }.get(10, TimeUnit.SECONDS)
    }

    fun shutdown() {
        val remaining = synchronized(this) { writers.toList() }
        remaining.forEach { runCatching { it.close() } }
        exec.shutdown()
        exec.awaitTermination(10, TimeUnit.SECONDS)
    }
}

class AsyncLineWriter(val file: File, header: String?, private val io: IoScheduler) : Closeable {
    private val queue = ConcurrentLinkedQueue<String>()
    private val fos = FileOutputStream(file, true)
    private val out = BufferedOutputStream(fos, 64 * 1024)
    @Volatile private var closed = false
    val linesWritten = AtomicLong()

    init {
        if (header != null && file.length() == 0L) queue.add(header)
        io.register(this)
    }

    fun write(line: String) {
        if (!closed) queue.add(line)
    }

    internal fun drain(sync: Boolean) {
        var n = 0L
        while (true) {
            val line = queue.poll() ?: break
            out.write(line.toByteArray(Charsets.UTF_8))
            out.write('\n'.code)
            n++
        }
        if (n > 0) {
            out.flush()
            linesWritten.addAndGet(n)
        }
        if (sync) runCatching { fos.fd.sync() }
    }

    internal fun closeStream() {
        runCatching { out.flush() }
        runCatching { fos.fd.sync() }
        runCatching { out.close() }
    }

    override fun close() {
        if (closed) return
        closed = true
        io.close(this)
    }
}
