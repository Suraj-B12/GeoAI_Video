package com.geoai.capture

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import com.geoai.capture.io.SessionStore
import com.geoai.capture.service.SurveyService
import com.geoai.capture.settings.SettingsRepository

class CaptureApp : Application() {
    lateinit var settingsRepo: SettingsRepository
        private set
    lateinit var store: SessionStore
        private set

    override fun onCreate() {
        super.onCreate()
        settingsRepo = SettingsRepository(this)
        store = SessionStore(this)
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(SurveyService.CHANNEL_SURVEY, getString(R.string.notif_channel_survey), NotificationManager.IMPORTANCE_LOW),
        )
        nm.createNotificationChannel(
            NotificationChannel(SurveyService.CHANNEL_ALERTS, getString(R.string.notif_channel_alerts), NotificationManager.IMPORTANCE_HIGH)
                .apply { enableVibration(true) },
        )
    }
}
