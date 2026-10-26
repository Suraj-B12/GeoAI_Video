package com.geoai.capture.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** High-contrast dark theme: readable outdoors, low screen power on OLED. */
private val Scheme = darkColorScheme(
    primary = Color(0xFFFFB300),
    onPrimary = Color(0xFF1A1300),
    secondary = Color(0xFF80CBC4),
    background = Color(0xFF000000),
    surface = Color(0xFF121212),
    surfaceVariant = Color(0xFF1E1E1E),
    onSurface = Color(0xFFF2F2F2),
    error = Color(0xFFFF5252),
)

object StatusColors {
    val pass = Color(0xFF66BB6A)
    val warn = Color(0xFFFFCA28)
    val fail = Color(0xFFFF5252)
    val info = Color(0xFF90A4AE)
}

@Composable
fun CaptureTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
