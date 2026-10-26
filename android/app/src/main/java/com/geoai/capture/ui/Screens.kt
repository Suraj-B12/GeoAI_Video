package com.geoai.capture.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.PowerManager
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.geoai.capture.BuildConfig
import com.geoai.capture.CaptureApp
import com.geoai.capture.coverage.CoverageStore
import com.geoai.capture.devicecheck.DeviceCheck
import com.geoai.capture.io.SessionStore
import com.geoai.capture.service.Phase
import com.geoai.capture.service.SurveyController
import com.geoai.capture.service.SurveyService
import com.geoai.capture.service.SurveyStatus
import com.geoai.capture.settings.AppSettings
import com.geoai.capture.settings.Codec
import com.geoai.capture.settings.Resolution
import com.geoai.capture.settings.SurveyParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

enum class Screen { HOME, SETUP, SESSIONS, CHECK, SETTINGS }

private fun f(fmt: String, vararg a: Any?) = String.format(Locale.ROOT, fmt, *a)

data class PermissionState(
    val camera: Boolean,
    val location: Boolean,
    val notifications: Boolean,
    val allFiles: Boolean,
    val batteryExempt: Boolean,
) {
    val essentialOk: Boolean get() = camera && location

    companion object {
        private fun granted(c: Context, p: String) = ContextCompat.checkSelfPermission(c, p) == PackageManager.PERMISSION_GRANTED
        fun read(c: Context) = PermissionState(
            camera = granted(c, Manifest.permission.CAMERA),
            location = granted(c, Manifest.permission.ACCESS_FINE_LOCATION),
            notifications = Build.VERSION.SDK_INT < 33 || granted(c, Manifest.permission.POST_NOTIFICATIONS),
            allFiles = Environment.isExternalStorageManager(),
            batteryExempt = c.getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(c.packageName),
        )
    }
}

@Composable
fun AppRoot(app: CaptureApp, setKeepScreenOn: (Boolean) -> Unit) {
    val status by SurveyController.status.collectAsStateWithLifecycle()
    var screen by rememberSaveable { mutableStateOf(Screen.HOME) }
    var tick by remember { mutableIntStateOf(0) }
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { tick++ }
    val context = LocalContext.current
    val perms = remember(tick) { PermissionState.read(context) }
    val settings = remember(tick, screen, status.phase) { app.settingsRepo.load() }
    // System back on a sub-screen returns to Home instead of leaving the app.
    BackHandler(enabled = screen != Screen.HOME && !status.busy) { screen = Screen.HOME }
    LaunchedEffect(status.phase, settings.keepScreenOn) {
        setKeepScreenOn(settings.keepScreenOn && status.busy)
    }
    // safeDrawing = system bars + display cutout (the S21's punch-hole sits on one side in landscape) + keyboard.
    Scaffold(containerColor = Color.Black, contentWindowInsets = WindowInsets.safeDrawing) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            when {
                !perms.essentialOk -> PermissionsScreen(perms) { tick++ }
                status.busy || (status.phase == Phase.ERROR && status.error != null) -> LiveScreen(app, status, settings)
                else -> when (screen) {
                    Screen.HOME -> HomeScreen(app, perms, status) { screen = it }
                    Screen.SETUP -> SetupScreen(app) { screen = Screen.HOME }
                    Screen.SESSIONS -> SessionsScreen(app) { screen = Screen.HOME }
                    Screen.CHECK -> DeviceCheckScreen(app) { screen = Screen.HOME }
                    Screen.SETTINGS -> SettingsScreen(app) { screen = Screen.HOME }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ building blocks

@Composable
private fun Title(text: String, onBack: (() -> Unit)? = null) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)) {
        if (onBack != null) TextButton(onClick = onBack) { Text("‹ Back") }
        Text(text, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun Banner(text: String, color: Color, action: String? = null, onAction: () -> Unit = {}) {
    Card(colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.18f)), modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(text, color = color, modifier = Modifier.weight(1f))
            if (action != null) TextButton(onClick = onAction) { Text(action) }
        }
    }
}

@Composable
private fun Field(label: String, value: String, enabled: Boolean = true, number: Boolean = false, onChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        label = { Text(label) },
        singleLine = true,
        enabled = enabled,
        keyboardOptions = if (number) KeyboardOptions(keyboardType = KeyboardType.Decimal) else KeyboardOptions.Default,
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Choice(label: String, options: List<Pair<String, String>>, selected: String, enabled: Boolean = true, onSelect: (String) -> Unit) {
    Text(label, style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 8.dp))
    // Options wrap onto the next line on narrow screens (a 4-5 option row is wider than a phone in portrait).
    FlowRow(Modifier.fillMaxWidth()) {
        options.forEach { (value, text) ->
            // The whole row (circle + label) is one tap target.
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .selectable(selected = selected == value, enabled = enabled, role = Role.RadioButton) { onSelect(value) }
                    .padding(end = 12.dp, top = 4.dp, bottom = 4.dp),
            ) {
                RadioButton(selected = selected == value, onClick = null, enabled = enabled)
                Text(text)
            }
        }
    }
}

@Composable
private fun Toggle(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth().toggleable(value = checked, role = Role.Switch, onValueChange = onChange).padding(vertical = 4.dp),
    ) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = null)
    }
}

@Composable
private fun Stat(label: String, value: String, color: Color = Color.Unspecified) {
    Row(Modifier.fillMaxWidth().padding(vertical = 1.dp)) {
        Text(label, color = StatusColors.info, fontSize = 13.sp, modifier = Modifier.width(112.dp))
        Text(value, color = color, fontSize = 13.sp, fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f))
    }
}

private fun thermalName(s: Int) = when (s) {
    0 -> "none"; 1 -> "light"; 2 -> "MODERATE"; 3 -> "SEVERE"; 4 -> "CRITICAL"; 5 -> "EMERGENCY"; 6 -> "SHUTDOWN"; else -> "$s"
}

private fun shutter(ns: Long?) = if (ns == null || ns <= 0) "–" else "1/" + (1e9 / ns).toLong()

// ------------------------------------------------------------------ permissions

@Composable
private fun PermissionsScreen(perms: PermissionState, onChanged: () -> Unit) {
    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { onChanged() }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        Title("Permissions")
        Text("Survey recording needs the camera and precise location. Nothing leaves the phone: the app has no internet permission.")
        Spacer(Modifier.height(12.dp))
        Stat("Camera", if (perms.camera) "granted" else "missing", if (perms.camera) StatusColors.pass else StatusColors.fail)
        Stat("Precise location", if (perms.location) "granted" else "missing", if (perms.location) StatusColors.pass else StatusColors.fail)
        Stat("Notifications", if (perms.notifications) "granted" else "missing", if (perms.notifications) StatusColors.pass else StatusColors.warn)
        Spacer(Modifier.height(12.dp))
        Button(onClick = {
            val list = mutableListOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            if (Build.VERSION.SDK_INT >= 33) list += Manifest.permission.POST_NOTIFICATIONS
            launcher.launch(list.toTypedArray())
        }) { Text("Grant permissions") }
        TextButton(onClick = {
            context.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}")))
        }) { Text("Open app settings (if denied permanently)") }
    }
}

// ------------------------------------------------------------------ home

@Composable
private fun HomeScreen(app: CaptureApp, perms: PermissionState, status: SurveyStatus, go: (Screen) -> Unit) {
    val context = LocalContext.current
    val settings = remember { app.settingsRepo.load() }
    val free = remember { app.store.freeBytes() }
    val last = remember { app.settingsRepo.loadLastSurvey() }
    var roadKm by remember { mutableStateOf<Double?>(null) }
    LaunchedEffect(Unit) {
        // Counts sessions recorded before coverage existed (or cut short) so the total is right without starting one.
        roadKm = withContext(Dispatchers.IO) {
            runCatching {
                CoverageStore(app.store.root).run {
                    if (!status.busy) backfill(app.store.sessionsDir, status.sessionId)
                    totalNewM() / 1000.0
                }
            }.getOrNull()
        }
    }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        Title("GeoAI Capture")
        Text("Pavement survey recorder · v${BuildConfig.VERSION_NAME}", color = StatusColors.info)
        Spacer(Modifier.height(8.dp))
        Stat("Device ID", settings.deviceId)
        Stat("Sessions folder", app.store.sessionsDir.absolutePath)
        Stat("Free storage", if (free >= 0) f("%.1f GB", free / 1e9) else "?")
        Stat("Road covered", roadKm?.let { f("%.2f km, repeats not counted", it) } ?: "–", StatusColors.pass)
        if (!perms.allFiles) {
            Banner("All files access is off: sessions go to app storage, which Android deletes if the app is uninstalled.", StatusColors.fail, "Allow") {
                context.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${context.packageName}")))
            }
        }
        if (!perms.batteryExempt) {
            Banner("Battery optimisation is on: Android may stop long recordings. Also set Battery to 'Unrestricted' in app info.", StatusColors.warn, "Exempt") {
                context.startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:${context.packageName}")))
            }
        }
        if (!perms.notifications) Banner("Notifications are off: you will not get overheating or GNSS alerts.", StatusColors.warn)
        if (free in 0 until 20_000_000_000L) Banner("Less than 20 GB free. 4K uses about 27 GB per hour.", StatusColors.warn)
        status.message?.let { Banner(it, StatusColors.info) }
        Spacer(Modifier.height(12.dp))
        // One tap: the last survey's details, camera opens, focus/exposure lock and recording starts on a good GNSS fix.
        Button(
            onClick = { SurveyController.prepare(context, last.toJson().toString()) },
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
        ) { Text("Start survey", fontSize = 18.sp) }
        Text(
            listOf(
                last.sessionType,
                last.roadId.ifBlank { "no road ID" },
                last.roadName,
                last.nominalHeightM.takeIf { it.isNotBlank() }?.let { "mount $it m" },
                if (settings.autoRecord) "records automatically" else "press Record",
            ).filter { !it.isNullOrBlank() }.joinToString(" · "),
            color = StatusColors.info, fontSize = 13.sp,
        )
        Spacer(Modifier.height(8.dp))
        OutlinedButton(onClick = { go(Screen.SETUP) }, modifier = Modifier.fillMaxWidth()) { Text("Change survey details / calibration") }
        OutlinedButton(onClick = { go(Screen.SESSIONS) }, modifier = Modifier.fillMaxWidth()) { Text("Sessions") }
        OutlinedButton(onClick = { go(Screen.CHECK) }, modifier = Modifier.fillMaxWidth()) { Text("Device check") }
        OutlinedButton(onClick = { go(Screen.SETTINGS) }, modifier = Modifier.fillMaxWidth()) { Text("Settings") }
    }
}

// ------------------------------------------------------------------ setup

@Composable
private fun SetupScreen(app: CaptureApp, onBack: () -> Unit) {
    val context = LocalContext.current
    var p by remember { mutableStateOf(app.settingsRepo.loadLastSurvey()) }
    val calibration = p.sessionType == SurveyParams.TYPE_CAL_INTRINSICS || p.sessionType == SurveyParams.TYPE_CAL_MOUNT
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        Title("New session", onBack)
        Choice(
            "Session type",
            listOf(
                SurveyParams.TYPE_SURVEY to "Survey", SurveyParams.TYPE_TEST to "Test",
                SurveyParams.TYPE_CAL_INTRINSICS to "Lens calibration", SurveyParams.TYPE_CAL_MOUNT to "Mount calibration",
            ),
            p.sessionType,
        ) { p = p.copy(sessionType = it) }
        if (!calibration) {
            Field("Road ID", p.roadId) { p = p.copy(roadId = it) }
            Field("Road name", p.roadName) { p = p.copy(roadName = it) }
            Choice("Direction", listOf("increasing_chainage" to "Increasing chainage", "decreasing_chainage" to "Decreasing chainage"), p.direction) {
                p = p.copy(direction = it)
            }
            Field("Lane (e.g. L1 = leftmost)", p.lane) { p = p.copy(lane = it) }
            Field("Start chainage (km)", p.startChainageKm, number = true) { p = p.copy(startChainageKm = it) }
            Choice(
                "Weather / light",
                listOf("dry_sunny" to "Dry, sunny", "dry_overcast" to "Dry, overcast", "dry_shade" to "Dry, heavy shade", "wet" to "Wet", "other" to "Other"),
                p.weather,
            ) { p = p.copy(weather = it) }
        } else {
            Field("Target spec (board/mat, sizes, dictionary)", p.targetSpec) { p = p.copy(targetSpec = it) }
            Field("Target placement / measurements", p.targetPlacement) { p = p.copy(targetPlacement = it) }
            Text(
                if (p.sessionType == SurveyParams.TYPE_CAL_INTRINSICS)
                    "Lock focus on the road with the phone on its mount first, then move the board slowly through the whole frame at 1.5–3.5 m."
                else "Vehicle parked on flat ground; mat placed at the measured position; nothing moving for 10 s.",
                color = StatusColors.info,
            )
        }
        Field("Operator", p.operator) { p = p.copy(operator = it) }
        Field("Vehicle", p.vehicle) { p = p.copy(vehicle = it) }
        Field("Mount ID", p.mountId) { p = p.copy(mountId = it) }
        Choice("Camera faces", listOf("forward" to "Forward", "rearward" to "Rearward"), p.cameraFacing) { p = p.copy(cameraFacing = it) }
        Field("Nominal height above road (m)", p.nominalHeightM, number = true) { p = p.copy(nominalHeightM = it) }
        Field("Nominal pitch below horizontal (°)", p.nominalPitchDeg, number = true) { p = p.copy(nominalPitchDeg = it) }
        Field("Notes", p.notes) { p = p.copy(notes = it) }
        Spacer(Modifier.height(12.dp))
        Button(
            onClick = {
                app.settingsRepo.saveLastSurvey(p)
                SurveyController.prepare(context, p.toJson().toString())
            },
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
        ) { Text("Save and start", fontSize = 18.sp) }
    }
}

// ------------------------------------------------------------------ live (preview / recording)

@Composable
private fun LiveScreen(app: CaptureApp, s: SurveyStatus, settings: AppSettings) {
    val context = LocalContext.current
    val preview by SurveyController.preview.collectAsStateWithLifecycle()
    var confirmStop by remember { mutableStateOf(false) }
    var markerDialog by remember { mutableStateOf(false) }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val wide = maxWidth > maxHeight
        val previewBox: @Composable (Modifier) -> Unit = { m ->
            Box(m.aspectRatio(16f / 9f).background(Color(0xFF101010))) {
                preview?.let {
                    Image(it.asImageBitmap(), contentDescription = "camera preview (sensor orientation, as recorded)",
                        contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
                }
                Canvas(Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height
                    val line = Color.White.copy(alpha = 0.35f)
                    drawLine(line, Offset(w / 2, 0f), Offset(w / 2, h), 2f)
                    drawLine(line, Offset(0f, h / 2), Offset(w, h / 2), 2f)
                    drawRect(Color(0xFFFFB300).copy(alpha = 0.6f), Offset(w * 0.1f, h * 0.4f), androidx.compose.ui.geometry.Size(w * 0.8f, h * 0.6f), style = Stroke(3f))
                }
            }
        }
        val panel: @Composable (Modifier) -> Unit = { m -> StatusPanel(s, m) }
        // The controls never take more than this, and scroll inside it, so they always stay on screen.
        val controlsMax = maxHeight * 0.45f
        if (wide) {
            // Landscape (phone on its mount): the preview fits the height left above the controls,
            // instead of filling the width and pushing the buttons off the bottom of the screen.
            Row(Modifier.fillMaxSize().padding(8.dp)) {
                Column(Modifier.weight(1.3f).fillMaxHeight()) {
                    Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) { previewBox(Modifier) }
                    Controls(s, Modifier.heightIn(max = controlsMax), onStop = { confirmStop = true }, onMarker = { markerDialog = true })
                }
                Spacer(Modifier.width(8.dp))
                panel(Modifier.weight(1f).fillMaxHeight())
            }
        } else {
            Column(Modifier.fillMaxSize().padding(8.dp)) {
                previewBox(Modifier.fillMaxWidth())
                Controls(s, Modifier.heightIn(max = controlsMax), onStop = { confirmStop = true }, onMarker = { markerDialog = true })
                panel(Modifier.weight(1f))
            }
        }
    }
    if (confirmStop) {
        AlertDialog(
            onDismissRequest = { confirmStop = false },
            title = { Text("Stop recording?") },
            text = { Text("The session will be finalised and checksummed.") },
            confirmButton = {
                Button(onClick = { confirmStop = false; SurveyController.send(context, SurveyService.ACTION_STOP) },
                    colors = ButtonDefaults.buttonColors(containerColor = StatusColors.fail)) { Text("Stop") }
            },
            dismissButton = { TextButton(onClick = { confirmStop = false }) { Text("Keep recording") } },
        )
    }
    if (markerDialog) MarkerDialog(onDismiss = { markerDialog = false }) { type, note ->
        markerDialog = false
        SurveyController.send(context, SurveyService.ACTION_MARKER) {
            putExtra(SurveyService.EXTRA_MARKER_TYPE, type)
            putExtra(SurveyService.EXTRA_MARKER_NOTE, note)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Controls(s: SurveyStatus, modifier: Modifier, onStop: () -> Unit, onMarker: () -> Unit) {
    val context = LocalContext.current
    Column(modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(vertical = 8.dp)) {
        Text("${s.phase}  ${s.sessionId ?: ""}", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
        s.message?.let { Text(it, color = StatusColors.info) }
        s.error?.let { Text(it, color = StatusColors.fail) }
        when (s.phase) {
            Phase.PREPARING, Phase.LOCKING, Phase.STARTING, Phase.STOPPING -> LinearProgressIndicator(Modifier.fillMaxWidth().padding(vertical = 8.dp))
            Phase.FINALIZING -> LinearProgressIndicator(progress = { s.finalizeProgress ?: 0f }, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp))
            else -> Unit
        }
        if (!s.locationEnabled && (s.phase == Phase.PREVIEW || s.phase == Phase.RECORDING)) {
            Banner("Location is OFF. GNSS positions are not being recorded.", StatusColors.fail, "Turn on") {
                context.startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
            }
        }
        if (s.phase == Phase.PREVIEW && s.sessionType == SurveyParams.TYPE_SURVEY && !s.gnssReady) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Waiting for a GNSS fix of 10 m or better; recording then starts by itself.", color = StatusColors.warn, fontSize = 13.sp,
                    modifier = Modifier.weight(1f))
                TextButton(onClick = {
                    SurveyController.send(context, SurveyService.ACTION_START) { putExtra(SurveyService.EXTRA_FORCE_START, true) }
                }, enabled = s.locked) { Text("Record anyway") }
            }
        }
        // Buttons wrap onto a second line rather than running off the edge.
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth().padding(top = 4.dp)) {
            when (s.phase) {
                Phase.PREVIEW -> {
                    val needsGnss = s.sessionType == SurveyParams.TYPE_SURVEY && !s.gnssReady
                    Button(onClick = { SurveyController.send(context, SurveyService.ACTION_START) }, enabled = s.locked && !needsGnss,
                        colors = ButtonDefaults.buttonColors(containerColor = StatusColors.fail)) { Text("● Record") }
                    OutlinedButton(onClick = { SurveyController.send(context, SurveyService.ACTION_LOCK) }) { Text("Re-lock") }
                    TextButton(onClick = { SurveyController.send(context, SurveyService.ACTION_CANCEL) }) { Text("Cancel") }
                }
                Phase.RECORDING -> {
                    Button(onClick = onStop, colors = ButtonDefaults.buttonColors(containerColor = StatusColors.fail)) { Text("■ Stop") }
                    OutlinedButton(onClick = onMarker) { Text("Marker") }
                }
                Phase.ERROR -> Button(onClick = { SurveyController.clearError() }) { Text("Back") }
                else -> Unit
            }
        }
    }
}

@Composable
private fun StatusPanel(s: SurveyStatus, modifier: Modifier) {
    Column(modifier.verticalScroll(rememberScrollState())) {
        val orientColor = when (s.orientation) { "UPRIGHT" -> StatusColors.pass; "UNKNOWN" -> StatusColors.warn; else -> StatusColors.fail }
        Stat("Orientation", when (s.orientation) {
            "UPRIGHT" -> "upright ✓"
            "UPSIDE_DOWN" -> "UPSIDE DOWN: rotate phone 180°"
            "PORTRAIT" -> "PORTRAIT: mount in landscape"
            else -> "unknown"
        }, orientColor)
        // Mount check for segment mode: the frame must show road from near to near + 6 m (5 m segment + 1 m window).
        val lookingUp = (s.tiltDeg ?: 0.0) < 0
        Stat("Tilt", MountText.tilt(s.tiltDeg), if (lookingUp) StatusColors.fail else Color.Unspecified)
        Stat("Road in view", MountText.roadRange(s.roadNearM, s.roadFarM, s.tiltDeg != null),
            if (s.tiltDeg != null && s.roadNearM == null) StatusColors.fail else Color.Unspecified)
        Stat(MountText.segmentLabel, MountText.segment(s.roadNearM, s.segmentFits), when (s.segmentFits) {
            true -> StatusColors.pass
            false -> StatusColors.fail
            null -> Color.Unspecified
        })
        val gnssOk = (s.gnssAccM ?: 99f) <= 5f && (s.gnssFixAgeS ?: 99.0) < 2.0
        Stat("GNSS", s.gnssAccM?.let { f("±%.1f m, %s sats, age %.1fs", it, s.gnssSats?.toString() ?: "?", s.gnssFixAgeS ?: 0.0) } ?: "no fix",
            if (gnssOk) StatusColors.pass else StatusColors.warn)
        if (s.mock) Stat("Mock location", "DETECTED: data untrusted", StatusColors.fail)
        Stat("Speed", s.speedKmh?.let { f("%.0f km/h", it) } ?: "–", if ((s.speedKmh ?: 0.0) > 50) StatusColors.fail else if ((s.speedKmh ?: 0.0) > 40) StatusColors.warn else Color.Unspecified)
        Stat("Exposure", "${shutter(s.exposureNs)} s  ISO ${s.iso ?: "–"}${if (s.lowLight) "  LOW LIGHT" else ""}", if (s.lowLight) StatusColors.warn else Color.Unspecified)
        Stat("Mode", s.exposureMode, if (s.exposureMode == "AUTO_FALLBACK") StatusColors.fail else Color.Unspecified)
        Stat("Road luma", s.meanLuma?.let { f("%.0f / 255  clip %.1f%%", it, (s.highlightFrac ?: 0.0) * 100) } ?: "–")
        Stat("Focus", "${s.focusMode} ${s.focusDiopters?.let { f("%.2f dpt", it) } ?: ""}", if (s.focusMode.startsWith("LOCKED") || s.focusMode == "FIXED") StatusColors.pass else StatusColors.warn)
        if (s.focusNote.isNotBlank()) Stat("Focus set by", s.focusNote, if (s.focusNote.startsWith("af default")) StatusColors.warn else Color.Unspecified)
        Stat("Sharpness", s.sharpness?.let { f("%.0f", it) } ?: "–")
        HorizontalDivider(Modifier.padding(vertical = 4.dp))
        Stat("Recording", if (s.phase == Phase.RECORDING) f("%d:%02d  %.2f km", (s.recordingS / 60).toInt(), (s.recordingS % 60).toInt(), s.distanceKm) else "–")
        Stat("New road", f("%.2f km this session", s.roadNewKm), StatusColors.pass)
        Stat("Road covered", f("%.2f km in total", s.roadTotalKm), StatusColors.pass)
        Stat("Stream", if (s.recordWidth > 0) "${s.recordWidth}x${s.recordHeight} ${s.streams}" else s.streams)
        Stat("Frames", "${s.framesEncoded}  unmatched ${s.framesUnmatched}", if (s.framesUnmatched > 0) StatusColors.warn else Color.Unspecified)
        Stat("Camera drops", "${s.captureFailures} failed, ${s.buffersLost} lost", if (s.captureFailures + s.buffersLost > 0) StatusColors.warn else Color.Unspecified)
        Stat("Segment", s.segment.toString())
        HorizontalDivider(Modifier.padding(vertical = 4.dp))
        Stat("Thermal", "${thermalName(s.thermalStatus)} ${s.headroom?.let { f("headroom %.2f", it) } ?: ""}",
            if (s.thermalStatus >= 3) StatusColors.fail else if (s.thermalStatus == 2) StatusColors.warn else StatusColors.pass)
        Stat("Battery", "${s.batteryPct?.let { "$it%" } ?: "?"} ${s.batteryTempC?.let { f("%.1f °C", it) } ?: ""} ${if (s.charging) "charging" else ""}",
            if ((s.batteryTempC ?: 0f) >= 42f) StatusColors.warn else Color.Unspecified)
        Stat("Free storage", s.freeGb?.let { f("%.1f GB", it) } ?: "–")
        if (s.warnings.isNotEmpty()) {
            HorizontalDivider(Modifier.padding(vertical = 4.dp))
            s.warnings.forEach { Text("⚠ $it", color = StatusColors.warn, fontSize = 13.sp) }
        }
    }
}

@Composable
private fun MarkerDialog(onDismiss: () -> Unit, onSave: (String, String) -> Unit) {
    var type by remember { mutableStateOf("km_stone") }
    var note by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add marker") },
        text = {
            Column {
                Choice("Type", listOf("km_stone" to "km stone", "distress" to "Distress", "event" to "Event"), type) { type = it }
                Field(if (type == "km_stone") "Chainage on stone (km)" else "Note", note, number = type == "km_stone") { note = it }
            }
        },
        confirmButton = { Button(onClick = { onSave(type, note) }) { Text("Save") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

// ------------------------------------------------------------------ sessions

@Composable
private fun SessionsScreen(app: CaptureApp, onBack: () -> Unit) {
    val scope = rememberCoroutineScope()
    var refresh by remember { mutableIntStateOf(0) }
    var items by remember { mutableStateOf<List<SessionStore.Summary>?>(null) }
    var busy by remember { mutableStateOf<String?>(null) }
    var confirmDelete by remember { mutableStateOf<SessionStore.Summary?>(null) }
    LaunchedEffect(refresh) {
        items = withContext(Dispatchers.IO) { app.store.list(SurveyController.status.value.sessionId) }
    }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        Title("Sessions", onBack)
        Text("Copy to the PC over USB (USB debugging on). On the PC run:", color = StatusColors.info)
        Text("python tools/pull_sessions.py <folder>", fontFamily = FontFamily.Monospace, fontSize = 12.sp)
        Text("It copies, verifies every checksum and marks the session VERIFIED_ON_PC here. Delete only verified sessions.",
            color = StatusColors.info, fontSize = 13.sp)
        Text("Phone folder: ${app.store.sessionsDir.absolutePath}", fontFamily = FontFamily.Monospace, fontSize = 11.sp, color = StatusColors.info)
        busy?.let { Text(it, color = StatusColors.warn) }
        Spacer(Modifier.height(8.dp))
        val list = items
        if (list == null) LinearProgressIndicator(Modifier.fillMaxWidth())
        else if (list.isEmpty()) Text("No sessions yet.")
        else list.forEach { s ->
            Card(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Column(Modifier.padding(12.dp)) {
                    Text(s.id, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    val stateColor = when (s.state) {
                        SessionStore.State.VERIFIED_ON_PC -> StatusColors.pass
                        SessionStore.State.FINALIZED -> StatusColors.info
                        SessionStore.State.ACTIVE -> StatusColors.warn
                        SessionStore.State.INCOMPLETE -> StatusColors.fail
                    }
                    Text("${s.sessionType} · ${s.state} · " + f("%.2f GB", s.sizeBytes / 1e9) +
                        (s.durationS?.let { f(" · %.1f min", it / 60) } ?: "") + " · ${s.segments} segment(s)", color = stateColor, fontSize = 13.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (s.state == SessionStore.State.INCOMPLETE) {
                            OutlinedButton(onClick = {
                                scope.launch {
                                    busy = "Finalising ${s.id}…"
                                    withContext(Dispatchers.IO) { app.store.recover(s.dir) }
                                    busy = null
                                    refresh++
                                }
                            }) { Text("Finalise") }
                        }
                        if (s.state != SessionStore.State.ACTIVE) {
                            TextButton(onClick = { confirmDelete = s }) { Text("Delete", color = StatusColors.fail) }
                        }
                    }
                }
            }
        }
    }
    confirmDelete?.let { s ->
        val verified = s.state == SessionStore.State.VERIFIED_ON_PC
        AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text(if (verified) "Delete verified session?" else "Delete UNVERIFIED session?") },
            text = { Text(if (verified) "The PC has confirmed every checksum of this session." else "The PC has NOT confirmed a copy of this session. The data will be lost for good.") },
            confirmButton = {
                Button(onClick = {
                    confirmDelete = null
                    scope.launch {
                        withContext(Dispatchers.IO) { app.store.delete(s.dir) }
                        refresh++
                    }
                }, colors = ButtonDefaults.buttonColors(containerColor = StatusColors.fail)) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Keep") } },
        )
    }
}

// ------------------------------------------------------------------ device check

@Composable
private fun DeviceCheckScreen(app: CaptureApp, onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var withGnss by remember { mutableStateOf(true) }
    var running by remember { mutableStateOf(false) }
    var progress by remember { mutableStateOf("") }
    var report by remember { mutableStateOf<DeviceCheck.Report?>(null) }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        Title("Device check", onBack)
        Text("Checks this phone against the capture requirements and saves a JSON report next to the sessions.", color = StatusColors.info)
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.toggleable(value = withGnss, enabled = !running, role = Role.Checkbox) { withGnss = it },
        ) {
            Checkbox(checked = withGnss, onCheckedChange = null, enabled = !running)
            Text("Include 60 s GNSS test (go outdoors, open sky)")
        }
        Button(onClick = {
            running = true
            report = null
            scope.launch {
                report = DeviceCheck.run(context.applicationContext, app.store, app.settingsRepo.load(), if (withGnss) 60 else 0) { msg ->
                    scope.launch { progress = msg }
                }
                running = false
            }
        }, enabled = !running) { Text(if (running) "Running…" else "Run check") }
        if (running) {
            LinearProgressIndicator(Modifier.fillMaxWidth().padding(vertical = 8.dp))
            Text(progress, color = StatusColors.info)
        }
        report?.let { r ->
            r.file?.let { Text("Saved: ${it.absolutePath}", fontSize = 12.sp, color = StatusColors.info) }
            Spacer(Modifier.height(8.dp))
            r.items.forEach { item ->
                val color = when (item.level) {
                    DeviceCheck.Level.PASS -> StatusColors.pass
                    DeviceCheck.Level.WARN -> StatusColors.warn
                    DeviceCheck.Level.FAIL -> StatusColors.fail
                    DeviceCheck.Level.INFO -> StatusColors.info
                }
                Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                    Text(item.level.name, color = color, fontWeight = FontWeight.Bold, modifier = Modifier.width(52.dp), fontSize = 13.sp)
                    Column {
                        Text(item.name, fontSize = 14.sp)
                        Text(item.detail, color = StatusColors.info, fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ settings

@Composable
private fun SettingsScreen(app: CaptureApp, onBack: () -> Unit) {
    val initial = remember { app.settingsRepo.load() }
    var deviceId by remember { mutableStateOf(initial.deviceId) }
    var resolution by remember { mutableStateOf(initial.resolution.name) }
    var codec by remember { mutableStateOf(initial.codec.name) }
    var bitrate by remember { mutableStateOf(initial.bitrateMbps.toString()) }
    var blur by remember { mutableStateOf(initial.blurBudgetMm.toString()) }
    var maxExp by remember { mutableStateOf(initial.maxExposureDenominator.toString()) }
    var lowExp by remember { mutableStateOf(initial.lowLightExposureDenominator.toString()) }
    var maxIso by remember { mutableStateOf(initial.maxIso.toString()) }
    var luma by remember { mutableStateOf(initial.targetLuma.toString()) }
    var unknownSpeed by remember { mutableStateOf(initial.unknownSpeedKmh.toString()) }
    var segment by remember { mutableStateOf(initial.segmentMinutes.toString()) }
    var ois by remember { mutableStateOf(initial.oisEnabled) }
    var stepDown by remember { mutableStateOf(initial.thermalStepDown) }
    var raw by remember { mutableStateOf(initial.rawGnss) }
    var fullTracking by remember { mutableStateOf(initial.gnssFullTracking) }
    var screenOn by remember { mutableStateOf(initial.keepScreenOn) }
    var alerts by remember { mutableStateOf(initial.alerts) }
    var autoRecord by remember { mutableStateOf(initial.autoRecord) }
    var saved by remember { mutableStateOf(false) }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        Title("Settings", onBack)
        Text("Every setting is recorded in each session's session.json.", color = StatusColors.info)
        Field("Device ID (letters, digits, - and _)", deviceId) { deviceId = it }
        Choice("Resolution", Resolution.entries.map { it.name to it.label }, resolution) { resolution = it }
        Choice("Codec", Codec.entries.map { it.name to it.label }, codec) { codec = it }
        Field("Bitrate at 4K (Mbps; scaled by area for smaller sizes)", bitrate, number = true) { bitrate = it }
        Field("Blur budget (mm of ground smear per exposure)", blur, number = true) { blur = it }
        Field("Longest exposure in normal light (1/x s)", maxExp, number = true) { maxExp = it }
        Field("Longest exposure in low light (1/x s)", lowExp, number = true) { lowExp = it }
        Field("Maximum ISO", maxIso, number = true) { maxIso = it }
        Field("Target road luma (0–255)", luma, number = true) { luma = it }
        Field("Speed assumed without GNSS (km/h)", unknownSpeed, number = true) { unknownSpeed = it }
        Field("Segment length (minutes)", segment, number = true) { segment = it }
        Toggle("Optical stabilisation (OIS)", ois) { ois = it }
        Toggle("Drop to 1080p when the phone overheats", stepDown) { stepDown = it }
        Toggle("Log raw GNSS measurements (for PPK; more battery)", raw) { raw = it }
        Toggle("GNSS full tracking (with raw logging)", fullTracking) { fullTracking = it }
        Toggle("Keep screen on while the camera is active", screenOn) { screenOn = it }
        Toggle("Vibration + notification alerts", alerts) { alerts = it }
        Toggle("Start recording automatically (camera locked + GNSS fix)", autoRecord) { autoRecord = it }
        Spacer(Modifier.height(12.dp))
        Button(onClick = {
            val d = AppSettings(initial.deviceId)
            app.settingsRepo.save(
                AppSettings(
                    deviceId = deviceId,
                    resolution = Resolution.valueOf(resolution),
                    codec = Codec.valueOf(codec),
                    bitrateMbps = bitrate.toIntOrNull()?.coerceIn(8, 200) ?: d.bitrateMbps,
                    blurBudgetMm = blur.replace(',', '.').toDoubleOrNull()?.coerceIn(0.5, 20.0) ?: d.blurBudgetMm,
                    maxExposureDenominator = maxExp.toIntOrNull()?.coerceIn(60, 16000) ?: d.maxExposureDenominator,
                    lowLightExposureDenominator = lowExp.toIntOrNull()?.coerceIn(30, 16000) ?: d.lowLightExposureDenominator,
                    maxIso = maxIso.toIntOrNull()?.coerceIn(50, 12800) ?: d.maxIso,
                    targetLuma = luma.toIntOrNull()?.coerceIn(40, 200) ?: d.targetLuma,
                    unknownSpeedKmh = unknownSpeed.replace(',', '.').toDoubleOrNull()?.coerceIn(5.0, 120.0) ?: d.unknownSpeedKmh,
                    oisEnabled = ois,
                    segmentMinutes = segment.toIntOrNull()?.coerceIn(1, 30) ?: d.segmentMinutes,
                    thermalStepDown = stepDown,
                    rawGnss = raw,
                    gnssFullTracking = fullTracking,
                    keepScreenOn = screenOn,
                    alerts = alerts,
                    autoRecord = autoRecord,
                ),
            )
            saved = true
        }, modifier = Modifier.fillMaxWidth()) { Text("Save") }
        OutlinedButton(onClick = { app.settingsRepo.resetToDefaults(); onBack() }, modifier = Modifier.fillMaxWidth()) {
            Text("Reset to recommended settings")
        }
        if (saved) Text("Saved. Applies to the next session.", color = StatusColors.pass)
    }
}
