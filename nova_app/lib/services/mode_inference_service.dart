import 'dart:convert';
import 'dart:io';
import 'package:flutter_tts/flutter_tts.dart';
import '../models/detection_models.dart';
import '../services/api_service.dart';
import '../core/modes.dart';
import '../core/config.dart';

/// Handles mode-specific inference logic
class ModeInferenceService {
  final FlutterTts tts;

  // === TTS debounce state ===
  final Map<String, DateTime> _lastSpokenAt = {};
  static const int _ttsCooldownSeconds = 4;

  // === Recognition debounce ===
  String _lastRecognitionPhrase = '';
  DateTime _lastRecognitionSpeechTime = DateTime(2000);
  static const int _recognitionCooldownSeconds = 8;

  ModeInferenceService({required this.tts});

  /// Process frame based on current mode
  Future<ModeInferenceResult> processFrame(
    File imageFile,
    NovaMode mode,
  ) async {
    try {
      switch (mode) {
        case NovaMode.navigation:
          return await _processNavigation(imageFile);
        case NovaMode.reading:
          return await _processReading(imageFile);
        case NovaMode.recognition:
          return await _processRecognition(imageFile);
        case NovaMode.emergency:
          await tts.speak('Emergency mode is active. Inference paused.');
          return ModeInferenceResult.error(
              'Emergency mode does not run camera inference');
      }
    } catch (e) {
      await tts.speak('Error processing image: $e');
      return ModeInferenceResult.error(e.toString());
    }
  }

  /// Navigation mode: Detect obstacles, estimate depth, provide guidance
  Future<ModeInferenceResult> _processNavigation(File imageFile) async {
    try {
      debugPrint('[NAV][MODE] ► Processing navigation frame...');
      debugPrint('[NAV][MODE] ► Image path: ${imageFile.path}');
      debugPrint('[NAV][MODE] ► Image exists: ${await imageFile.exists()}');

      final result = await ApiService.getNavigationGuidance(imageFile);

      debugPrint('[NAV][MODE] ◄ API response received. Keys: ${result.keys.toList()}');

      NavigationGuidanceResponse response;
      try {
        response = NavigationGuidanceResponse.fromJson(result);
        debugPrint('[NAV][MODE] ✓ Parsed guidance: "${response.guidance.guidance}"');
        debugPrint('[NAV][MODE] ✓ Obstacles: ${response.guidance.obstacles.length}');
        debugPrint('[NAV][MODE] ✓ Safety warnings: ${response.guidance.safetyWarnings}');
        debugPrint('[NAV][MODE] ✓ debugFrame present: ${response.guidance.debugFrameBase64 != null}');
      } catch (parseErr, parseStack) {
        debugPrint('[NAV][MODE] ✗ JSON→Model parse FAILED: $parseErr');
        debugPrint('[NAV][MODE] STACK: $parseStack');
        rethrow;
      }

      // === TTS DEBOUNCE — 4-second per-command cooldown ===
      final guidanceText = response.guidance.guidance;
      final navCommand = _extractCommand(guidanceText);
      final isUrgent = navCommand == 'STOP' ||
          navCommand == 'CAUTION' ||
          navCommand == 'MOVE_LEFT' ||
          navCommand == 'MOVE_RIGHT';

      if (isUrgent && guidanceText.isNotEmpty) {
        final now = DateTime.now();
        final lastSpoken = _lastSpokenAt[navCommand];
        final cooldownPassed = lastSpoken == null ||
            now.difference(lastSpoken).inSeconds >= _ttsCooldownSeconds;

        if (cooldownPassed) {
          _lastSpokenAt[navCommand] = now;
          debugPrint('[NAV][MODE] ► TTS ($navCommand): "$guidanceText"');
          // === STOP OVERLAP — kill any in-progress speech before speaking ===
          await tts.stop();
          await tts.speak(guidanceText);
        } else {
          final remaining =
              _ttsCooldownSeconds - now.difference(lastSpoken).inSeconds;
          debugPrint(
              '[NAV][MODE] ⏱ TTS debounced ($navCommand): ${remaining}s remaining');
        }
      } else if (!isUrgent) {
        // Path clear — reset cooldown so next alert is heard immediately
        _lastSpokenAt.clear();
      }

      // safetyWarnings are logged for display only — NOT spoken (avoids double TTS)
      for (final warning in response.guidance.safetyWarnings) {
        debugPrint('[NAV][MODE] ► Warning (display only): $warning');
      }

      debugPrint('[NAV][MODE] ✓ Navigation mode inference complete.');

      // Decode the base64 heatmap frame
      List<int>? heatmapBytes;
      final b64 = response.guidance.debugFrameBase64;
      if (b64 != null && b64.isNotEmpty) {
        try {
          heatmapBytes = base64Decode(b64);
          debugPrint('[NAV][MODE] ✓ Heatmap decoded: ${heatmapBytes.length} bytes');
        } catch (_) {
          debugPrint('[NAV][MODE] ✗ Failed to decode heatmap base64');
        }
      }

      return ModeInferenceResult.success(
        detections: response.guidance.obstacles,
        guidance: guidanceText,
        inferenceTimeMs: response.inferenceTimeMs,
        heatmapBytes: heatmapBytes,
        navCommand: navCommand,
      );
    } catch (e, s) {
      debugPrint('[NAV][MODE] ✗ NAV ERROR: $e');
      debugPrint('[NAV][MODE] STACKTRACE:\n$s');
      await tts.speak('Navigation error. Retrying.');
      return ModeInferenceResult.error(e.toString());
    }
  }

  /// Extract a canonical command tag from guidance text for logic branching
  String _extractCommand(String guidance) {
    final g = guidance.toLowerCase();
    if (g.contains('stop')) return 'STOP';
    if (g.contains('slow down') || g.contains('caution')) return 'CAUTION';
    if (g.contains('move left')) return 'MOVE_LEFT';
    if (g.contains('move right')) return 'MOVE_RIGHT';
    return 'PATH_CLEAR';
  }

  /// Reading mode: Recognize text via OCR
  Future<ModeInferenceResult> _processReading(File imageFile) async {
    if (!NovaConfig.enableOCR) {
      print('[NOVA DEBUG] ModeInferenceService: Reading/OCR disabled by config');
      return ModeInferenceResult.error('Reading mode disabled');
    }

    try {
      final result = await ApiService.recognizeText(imageFile);
      final response = TextDetectionResponse.fromJson(result);

      if (response.textRegions.isEmpty) {
        await tts.speak('No text detected');
      } else {
        final allText = response.textRegions.map((r) => r.text).join(' ');
        await tts.speak('Text detected: $allText');
      }

      return ModeInferenceResult.success(
        textRegions: response.textRegions,
        inferenceTimeMs: response.inferenceTimeMs ?? 0.0,
      );
    } catch (e) {
      await tts.speak('Reading mode error');
      return ModeInferenceResult.error(e.toString());
    }
  }

  /// Recognition mode: Detect faces and objects (with TTS debounce)
  Future<ModeInferenceResult> _processRecognition(File imageFile) async {
    if (!NovaConfig.enableRecognitionMode) {
      print('[NOVA DEBUG] ModeInferenceService: Recognition mode disabled by config');
      return ModeInferenceResult.error('Recognition mode disabled');
    }

    try {
      // Get both face and object detections
      final faceResult = await ApiService.detectFaces(
        imageFile,
        recognizeFaces: true,
      );
      final objectResult = await ApiService.detectObjects(imageFile);

      final faceResponse = FaceDetectionResponse.fromJson(faceResult);
      final objectResponse = ObjectDetectionResponse.fromJson(objectResult);

      // Build a speech phrase from detections
      final parts = <String>[];

      if (faceResponse.faces.isNotEmpty) {
        for (final face in faceResponse.faces) {
          final id = face.personId;
          if (id != null && id.isNotEmpty && id != 'unknown') {
            parts.add('I can see $id');
          }
        }
        // Count unknowns
        final unknowns = faceResponse.faces
            .where((f) => f.personId == null || f.personId == 'unknown' || f.personId!.isEmpty)
            .length;
        if (unknowns > 0) {
          parts.add("$unknowns unknown face${unknowns > 1 ? 's' : ''}");
        }
      }

      if (objectResponse.detections.isNotEmpty) {
        parts.add("${objectResponse.detections.length} object${objectResponse.detections.length > 1 ? 's' : ''} detected");
      }

      // === 8-second recognition debounce ===
      final phrase = parts.isNotEmpty ? parts.join(', ') : '';
      if (phrase.isNotEmpty) {
        final now = DateTime.now();
        final isDifferent = phrase != _lastRecognitionPhrase;
        final cooldownPassed =
            now.difference(_lastRecognitionSpeechTime).inSeconds >= _recognitionCooldownSeconds;

        if (isDifferent || cooldownPassed) {
          _lastRecognitionPhrase = phrase;
          _lastRecognitionSpeechTime = now;
          await tts.stop();
          await tts.speak(phrase);
        }
      }
      // Do NOT speak "No objects or faces detected" — avoid spam

      return ModeInferenceResult.success(
        detections: objectResponse.detections,
        faces: faceResponse.faces,
        inferenceTimeMs: faceResponse.inferenceTimeMs ?? 0.0,
      );
    } catch (e) {
      // Debounced error speech
      final now = DateTime.now();
      if (now.difference(_lastRecognitionSpeechTime).inSeconds >= _recognitionCooldownSeconds) {
        _lastRecognitionSpeechTime = now;
        await tts.speak('Recognition mode error');
      }
      return ModeInferenceResult.error(e.toString());
    }
  }
}

/// Result container for inference operations
class ModeInferenceResult {
  final bool success;
  final String? error;
  final List<Detection> detections;
  final List<TextRegion> textRegions;
  final List<Face> faces;
  final String guidance;
  final double inferenceTimeMs;
  /// Raw JPEG bytes of the depth heatmap frame from the backend
  final List<int>? heatmapBytes;
  /// Canonical command tag: STOP / CAUTION / MOVE_LEFT / MOVE_RIGHT / PATH_CLEAR
  final String navCommand;

  ModeInferenceResult({
    required this.success,
    this.error,
    this.detections = const [],
    this.textRegions = const [],
    this.faces = const [],
    this.guidance = '',
    this.inferenceTimeMs = 0,
    this.heatmapBytes,
    this.navCommand = 'PATH_CLEAR',
  });

  factory ModeInferenceResult.success({
    List<Detection> detections = const [],
    List<TextRegion> textRegions = const [],
    List<Face> faces = const [],
    String guidance = '',
    double inferenceTimeMs = 0,
    List<int>? heatmapBytes,
    String navCommand = 'PATH_CLEAR',
  }) {
    return ModeInferenceResult(
      success: true,
      detections: detections,
      textRegions: textRegions,
      faces: faces,
      guidance: guidance,
      inferenceTimeMs: inferenceTimeMs,
      heatmapBytes: heatmapBytes,
      navCommand: navCommand,
    );
  }

  factory ModeInferenceResult.error(String errorMsg) {
    return ModeInferenceResult(
      success: false,
      error: errorMsg,
    );
  }
}

void debugPrint(String message) {
  print(message);
}
