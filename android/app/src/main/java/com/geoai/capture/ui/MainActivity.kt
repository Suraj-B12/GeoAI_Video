package com.geoai.capture.ui

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import com.geoai.capture.BuildConfig
import com.geoai.capture.CaptureApp
import com.geoai.capture.service.Phase
import com.geoai.capture.service.SurveyController
import com.geoai.capture.service.SurveyService
import com.geoai.capture.settings.SurveyParams
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as CaptureApp
        setContent {
            CaptureTheme {
                AppRoot(app = app, setKeepScreenOn = ::setKeepScreenOn)
            }
        }
        maybeDebugAutostart(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        maybeDebugAutostart(intent)
    }

    /** Debug builds: `--ez debug_dump_threads true` logs the app's own thread stacks (tag GeoAI-threads). */
    private fun dumpThreads() {
        for ((thread, stack) in Thread.getAllStackTraces()) {
            val name = thread.name
            if (!name.startsWith("geoai") && name != "main" && !name.startsWith("DefaultDispatcher")) continue
            val frames = stack.take(25).joinToString(separator = "\n") { "    at $it" }
            android.util.Log.i("GeoAI-threads", "--- $name state=${thread.state}\n$frames")
        }
    }

    private fun setKeepScreenOn(on: Boolean) {
        if (on) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    /**
     * Debug builds only, for automated tests on an emulator or a USB-connected phone:
     *   adb shell am start -n com.geoai.capture/.ui.MainActivity --ez debug_autostart true --ei debug_duration_s 20
     * Optional: --ei debug_min_free_mb 300 (emulators have little storage) and --es debug_session_type test.
     * Finalise interrupted sessions: --ez debug_recover true
     */
    private fun maybeDebugAutostart(intent: Intent?) {
        if (BuildConfig.DEBUG && intent?.getBooleanExtra("debug_dump_threads", false) == true) {
            intent.removeExtra("debug_dump_threads")
            dumpThreads()
            return
        }
        if (BuildConfig.DEBUG && intent?.getBooleanExtra("debug_recover", false) == true) {
            intent.removeExtra("debug_recover")
            val app = application as CaptureApp
            lifecycleScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                app.store.list(SurveyController.status.value.sessionId)
                    .filter { it.state == com.geoai.capture.io.SessionStore.State.INCOMPLETE }
                    .forEach { app.store.recover(it.dir) }
            }
            return
        }
        if (!BuildConfig.DEBUG || intent?.getBooleanExtra("debug_autostart", false) != true) return
        intent.removeExtra("debug_autostart")
        val seconds = intent.getIntExtra("debug_duration_s", 20).coerceIn(3, 3600)
        val type = intent.getStringExtra("debug_session_type") ?: SurveyParams.TYPE_TEST
        val params = SurveyParams(sessionType = type, operator = "debug-autostart", notes = "automated test")
        lifecycleScope.launch {
            if (SurveyController.status.value.busy) return@launch
            val minFree = if (intent.hasExtra("debug_min_free_mb")) intent.getIntExtra("debug_min_free_mb", 0) else null
            SurveyController.prepare(this@MainActivity, params.toJson().toString(), minFree)
            val ready = withTimeoutOrNull(30_000) {
                SurveyController.status.first { (it.phase == Phase.PREVIEW && it.locked) || it.phase == Phase.ERROR }
            }
            if (ready == null || ready.phase == Phase.ERROR) return@launch
            SurveyController.send(this@MainActivity, SurveyService.ACTION_START)
            withTimeoutOrNull(15_000) { SurveyController.status.first { it.phase == Phase.RECORDING } }
            delay(seconds * 1000L)
            SurveyController.send(this@MainActivity, SurveyService.ACTION_MARKER) {
                putExtra(SurveyService.EXTRA_MARKER_TYPE, "debug")
                putExtra(SurveyService.EXTRA_MARKER_NOTE, "end of automated run")
            }
            SurveyController.send(this@MainActivity, SurveyService.ACTION_STOP)
        }
    }
}
