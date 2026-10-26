package com.geoai.capture.io

import android.content.Context
import android.os.Environment
import android.os.StatFs
import com.geoai.capture.util.CanonicalJson
import com.geoai.capture.util.Clocks
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

/**
 * Session folders live in /sdcard/GeoAI/sessions when "All files access" is granted (survives
 * uninstall, reachable over USB with adb or MTP). Without it they fall back to the app-specific
 * folder /sdcard/Android/data/com.geoai.capture/files/GeoAI/sessions, which adb can still read
 * but which Android deletes on uninstall.
 */
class SessionStore(private val context: Context) {

    val usingSharedStorage: Boolean get() = Environment.isExternalStorageManager()

    val root: File
        get() = if (usingSharedStorage) File(Environment.getExternalStorageDirectory(), "GeoAI")
        else File(context.getExternalFilesDir(null), "GeoAI")

    val sessionsDir: File get() = File(root, "sessions").apply { mkdirs() }
    val reportsDir: File get() = File(root, "device_reports").apply { mkdirs() }

    fun freeBytes(): Long = runCatching { StatFs(sessionsDir.path).availableBytes }.getOrDefault(-1L)

    fun create(sessionId: String): File {
        val dir = File(sessionsDir, sessionId)
        check(dir.mkdirs() || dir.isDirectory) { "Cannot create $dir" }
        File(dir, "video").mkdirs()
        return dir
    }

    enum class State { ACTIVE, INCOMPLETE, FINALIZED, VERIFIED_ON_PC }

    data class Summary(
        val id: String,
        val dir: File,
        val state: State,
        val sessionType: String,
        val sizeBytes: Long,
        val durationS: Double?,
        val segments: Int,
    )

    fun list(activeSessionId: String?): List<Summary> =
        (sessionsDir.listFiles() ?: emptyArray()).filter { it.isDirectory }.sortedByDescending { it.name }.map { dir ->
            val session = readJson(File(dir, "session.json"))
            val manifest = File(dir, "manifest.json")
            val state = when {
                dir.name == activeSessionId -> State.ACTIVE
                !manifest.isFile -> State.INCOMPLETE
                receiptMatches(dir) -> State.VERIFIED_ON_PC
                else -> State.FINALIZED
            }
            val startNs = session?.optJSONObject("start")?.optLong("t_ns", -1L) ?: -1L
            val endNs = session?.optJSONObject("end")?.optLong("t_ns", -1L) ?: -1L
            Summary(
                id = dir.name,
                dir = dir,
                state = state,
                sessionType = session?.optString("session_type", "?") ?: "?",
                sizeBytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() },
                durationS = if (startNs > 0 && endNs > startNs) (endNs - startNs) / 1e9 else null,
                segments = File(dir, "video").listFiles()?.count { it.name.endsWith(".mp4") } ?: 0,
            )
        }

    /** pc_receipt.json is written by the PC pull tool after it verified every checksum. */
    fun receiptMatches(dir: File): Boolean {
        val receipt = readJson(File(dir, "pc_receipt.json")) ?: return false
        val manifest = File(dir, "manifest.json")
        if (!manifest.isFile) return false
        return receipt.optString("status") == "verified" &&
            receipt.optString("manifest_sha256").equals(sha256Hex(manifest), ignoreCase = true)
    }

    fun delete(dir: File): Boolean = dir.deleteRecursively()

    /** Finalises a session that has no manifest (e.g. the app or phone died mid-recording). */
    fun recover(dir: File, progress: (Float) -> Unit = {}) {
        val sessionFile = File(dir, "session.json")
        val session = readJson(sessionFile) ?: JSONObject().put("session_id", dir.name)
        if (!session.has("end") || session.optJSONObject("end")?.has("t_ns") != true) {
            val lastTs = lastLoggedTimestamp(dir)
            session.put(
                "end",
                JSONObject()
                    .put("t_ns", lastTs ?: JSONObject.NULL)
                    .put("wall_ms", Clocks.wallMs())
                    .put("reason", "recovered_after_interruption"),
            )
            session.put("recovered", true)
            writeJsonAtomic(sessionFile, session)
        }
        ManifestBuilder.write(dir, session.optString("session_id", dir.name), progress)
    }

    /** Last t_ns in frames.csv/imu.csv, so a recovered session still gets a plausible end time. */
    private fun lastLoggedTimestamp(dir: File): Long? {
        for (name in listOf("frames.csv", "imu.csv", "gnss_fixes.csv")) {
            val f = File(dir, name)
            if (!f.isFile || f.length() == 0L) continue
            val lines = f.readLines().filter { it.isNotBlank() }
            if (lines.size < 2) continue
            val header = lines.first().split(',')
            val idx = header.indexOf("t_ns")
            if (idx < 0) continue
            for (line in lines.asReversed()) {
                val v = line.split(',').getOrNull(idx)?.toLongOrNull()
                if (v != null) return v
            }
        }
        return null
    }

    companion object {
        fun readJson(f: File): JSONObject? = runCatching { JSONObject(f.readText()) }.getOrNull()

        /** Write to a temp file, fsync, then rename, so a crash never leaves half a JSON file. */
        fun writeJsonAtomic(target: File, json: JSONObject) {
            val tmp = File(target.parentFile, target.name + ".tmp")
            FileOutputStream(tmp).use { out ->
                out.write(json.toString(2).toByteArray(Charsets.UTF_8))
                out.flush()
                out.fd.sync()
            }
            check(tmp.renameTo(target)) { "rename failed for $target" }
        }

        fun sha256Hex(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return hex(md.digest())
        }

        private const val HEX = "0123456789abcdef"

        /** Locale-proof hex (String.format can emit non-ASCII digits in some locales). */
        fun hex(bytes: ByteArray): String {
            val sb = StringBuilder(bytes.size * 2)
            for (b in bytes) {
                val v = b.toInt() and 0xff
                sb.append(HEX[v ushr 4]).append(HEX[v and 0x0f])
            }
            return sb.toString()
        }
    }
}

object ManifestBuilder {
    const val SPEC_VERSION = "1.1"
    private val EXCLUDED = setOf("manifest.json", "pc_receipt.json")

    /**
     * Hashes every file under [dir] by re-reading it from storage (this also proves the data
     * is readable), signs the canonical payload and writes manifest.json atomically.
     */
    fun write(dir: File, sessionId: String, progress: (Float) -> Unit = {}) {
        val files = dir.walkTopDown()
            .filter { it.isFile && it.name !in EXCLUDED && !it.name.endsWith(".tmp") }
            .map { it to it.relativeTo(dir).invariantSeparatorsPath }
            .sortedBy { it.second }
            .toList()
        val total = files.sumOf { it.first.length() }.coerceAtLeast(1L)
        var done = 0L
        val entries = JSONArray()
        for ((file, rel) in files) {
            entries.put(
                JSONObject()
                    .put("path", rel)
                    .put("bytes", file.length())
                    .put("sha256", SessionStore.sha256Hex(file)),
            )
            done += file.length()
            progress(done.toFloat() / total)
        }
        val payload = JSONObject()
            .put("spec_version", SPEC_VERSION)
            .put("session_id", sessionId)
            .put("created_wall_ms", Clocks.wallMs())
            .put("files", entries)
        val manifest = JSONObject(payload.toString())
        try {
            val sig = Signer.sign(CanonicalJson.encode(payload).toByteArray(Charsets.UTF_8))
            manifest.put(
                "signature",
                JSONObject()
                    .put("alg", "ECDSA_P256_SHA256")
                    .put("payload", "canonical JSON of {spec_version, session_id, created_wall_ms, files}")
                    .put("public_key_x509_b64", sig.publicKeyX509B64)
                    .put("value_b64", sig.signatureB64),
            )
            manifest.put("attestation_chain_b64", JSONArray(sig.certificateChainB64))
        } catch (e: Exception) {
            manifest.put("signature", JSONObject.NULL)
            manifest.put("signature_error", e.toString())
        }
        SessionStore.writeJsonAtomic(File(dir, "manifest.json"), manifest)
    }
}
