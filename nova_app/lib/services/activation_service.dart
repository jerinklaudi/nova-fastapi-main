import 'dart:async';
import 'package:flutter/services.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'package:flutter_tts/flutter_tts.dart';

class ActivationService {
  static final ActivationService _instance = ActivationService._internal();
  static ActivationService get instance => _instance;

  // MethodChannel for Native Triggers
  final MethodChannel _channel = const MethodChannel('com.example.nova_app/activation');
  
  // Stream controller to broadcast triggers to UI
  final _triggerController = StreamController<String>.broadcast();
  Stream<String> get onTrigger => _triggerController.stream;

  // Speech to Text
  final SpeechToText _speech = SpeechToText();
  bool _isSpeechAvailable = false;
  bool _isListening = false;
  
  // TTS for feedback
  final FlutterTts _tts = FlutterTts();



  ActivationService._internal();

  /// Initialize all listeners
  Future<void> initialize() async {
    // 1. Setup MethodChannel Listener
    _channel.setMethodCallHandler(_handleNativeCall);
    


    // 3. Check for pending native triggers (if app launched by service)
    try {
      final String? pendingMode = await _channel.invokeMethod('checkPendingActivation');
      if (pendingMode != null) {
        _triggerController.add(pendingMode);
      }
    } catch (e) {
      print("Error checking pending activation: $e");
    }
  }

  /// Start Speech Listening (Foreground Only)
  Future<void> startVoiceListener() async {
    if (!_isSpeechAvailable) {
      _isSpeechAvailable = await _speech.initialize(
        onError: (e) => print("Speech Error: $e"),
        onStatus: (status) {
           if (status == 'notListening' && _isListening) {
              // Restart listening for continuous effect
              Future.delayed(const Duration(seconds: 1), () {
                  if (_isListening) _speech.listen(onResult: _onSpeechResult);
              });
           }
        }
      );
    }

    if (_isSpeechAvailable) {
      _isListening = true;
      _speech.listen(
        onResult: _onSpeechResult,
        listenFor: const Duration(seconds: 30),
        pauseFor: const Duration(seconds: 3),
        partialResults: true,
      );
    }
  }

  void stopVoiceListener() {
    _isListening = false;
    _speech.stop();
  }

  void _onSpeechResult(result) {
    String recognized = result.recognizedWords.toLowerCase();
    if (recognized.contains("start nova") || recognized.contains("activate nova")) {
      _triggerController.add("NAVIGATION_MODE");
      _speak("Voice command received. Starting Navigation.");
      // Stop listening briefly or reset?
      // Continuous means we keep listening, but maybe pause during action
    }
  }

  Future<dynamic> _handleNativeCall(MethodCall call) async {
    if (call.method == "onActivationTriggered") {
       final String mode = call.arguments;
       _triggerController.add(mode);
    }
  }
  
  Future<void> _speak(String text) async {
      await _tts.speak(text);
  }
}
