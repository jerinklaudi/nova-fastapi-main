package com.example.nova_app

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.view.KeyEvent
import android.view.accessibility.AccessibilityEvent
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.Build
import android.content.Context
import android.util.Log

class VolumeAccessibilityService : AccessibilityService() {

    private var volumeUpCount = 0


    private var lastVolumeUpTime = 0L


    private val RESET_TIME_MS = 2000L // 2 seconds window

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // No-op
    }

    override fun onInterrupt() {
        // No-op
    }

    override fun onKeyEvent(event: KeyEvent): Boolean {
        val action = event.action
        val keyCode = event.keyCode
        val currentTime = System.currentTimeMillis()

        if (action == KeyEvent.ACTION_DOWN) {
            when (keyCode) {
                KeyEvent.KEYCODE_VOLUME_UP -> {
                    if (currentTime - lastVolumeUpTime > RESET_TIME_MS) {
                        volumeUpCount = 0
                    }
                    volumeUpCount++
                    lastVolumeUpTime = currentTime
                    Log.d("NOVA_ACCESSIBILITY", "Volume Up: $volumeUpCount")

                    if (volumeUpCount == 3) {
                        triggerAction("ACTIVATION_SCREEN")
                        volumeUpCount = 0
                    }
                    // Return false to allow normal volume change
                }
// Volume Down trigger removed as per user request
// Power Button trigger removed as per user request
            }
        }
        return super.onKeyEvent(event)
    }

    private fun triggerAction(mode: String) {
        Log.d("NOVA_ACCESSIBILITY", "Triggering mode: $mode")
        vibrate(mode)
        
        val intent = Intent(this, MainActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        intent.putExtra("ACTIVATION_MODE", mode)
        startActivity(intent)
    }

    private fun vibrate(mode: String) {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            when (mode) {
                "NAVIGATION_MODE" -> vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE))
                "RECOGNITION_MODE" -> vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 100, 100, 100), -1))
                "EMERGENCY_MODE" -> vibrator.vibrate(VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE))
            }
        } else {
            @Suppress("DEPRECATION")
            when (mode) {
                 "NAVIGATION_MODE" -> vibrator.vibrate(200)
                 "RECOGNITION_MODE" -> vibrator.vibrate(longArrayOf(0, 100, 100, 100), -1)
                 "EMERGENCY_MODE" -> vibrator.vibrate(1000)
            }
        }
    }
}
