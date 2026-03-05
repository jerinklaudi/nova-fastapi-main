package com.example.nova_app

import android.content.Intent
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.util.Log

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.nova_app/activation"
    private var methodChannel: MethodChannel? = null
    private var pendingActivationMode: String? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        methodChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
        
        methodChannel?.setMethodCallHandler { call, result ->
             if (call.method == "checkPendingActivation") {
                 result.success(pendingActivationMode)
                 pendingActivationMode = null
             } else {
                 result.notImplemented()
             }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent) {
        val mode = intent.getStringExtra("ACTIVATION_MODE")
        if (mode != null) {
            Log.d("NOVA_MAIN", "Received activation mode: $mode")
            pendingActivationMode = mode
            // If Flutter is already running, send it immediately
            methodChannel?.invokeMethod("onActivationTriggered", mode)
        }
    }
}
