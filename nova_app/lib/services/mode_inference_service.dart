import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../models/detection_models.dart';
import '../services/api_service.dart';
import '../core/modes.dart';
import '../core/config.dart';

/// Handles mode-specific inference logic
class ModeInferenceService {
  final FlutterTts tts;
  static const Set<String> _alertObjectLabels = {
    'car',
    'truck',
    'bus',
    'person',
    'motorcycle',
    'bicycle',
    'dog',
    'cat',
    'chair',
    'couch',
    'sofa',
    'bench',
    'dining table',
    'table',
    'bed',
    'potted plant',
  };

  // === Recognition debounce ===
  String _lastRecognitionPhrase = '';
  DateTime _lastRecognitionSpeechTime = DateTime(2000);
  static const int _recognitionCooldownSeconds = 8;
  static const int _minStableFramesForFaceRecognition = 2;
  static const int _minStableFramesForObjectRecognition = 1;
  static const double _minFaceConfidenceForRecognition = 0.35;
  static const double _minObjectConfidenceForRecognition = 0.25;
  final Map<String, int> _recognitionStreaks = {};

  // === Navigation false-positive suppression ===
  static const int _minStableFramesForNavigation = 2;
  static const double _navigationAlertMinConfidence = 0.35;
  static const double _navigationMinAreaRatio = 0.005;
  static const double _navigationPersonMinConfidence = 0.25;
  static const double _navigationPersonMinAreaRatio = 0.002;
  final Map<String, int> _navigationStreaks = {};
  static const Set<String> _navigationAlertLabels = {
    'person',
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'dog',
    'cat',
    'scooter',
    'chair',
    'couch',
    'sofa',
    'bench',
    'dining table',
    'table',
    'bed',
    'potted plant',
  };

  // === Navigation guidance debounce ===
  String _lastNavigationCommand = 'PATH_CLEAR';
  DateTime _lastNavigationSpeechTime = DateTime(2000);
  static const int _navigationCooldownSeconds = 12;

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

      final result = await ApiService.getNavigationGuidance(
        imageFile,
        objectConfidenceThreshold: NovaConfig.objectDetectionThreshold,
        textConfidenceThreshold: NovaConfig.ocrConfidenceThreshold,
        enableTextDetection: NovaConfig.enableOCR,
        enableDepthEstimation: NovaConfig.enableDepthEstimation,
        generateAudio: false,
      );

      debugPrint(
          '[NAV][MODE] ◄ API response received. Keys: ${result.keys.toList()}');

      NavigationGuidanceResponse response;
      try {
        response = NavigationGuidanceResponse.fromJson(result);
        debugPrint(
            '[NAV][MODE] ✓ Parsed guidance: "${response.guidance.guidance}"');
        debugPrint(
            '[NAV][MODE] ✓ Obstacles: ${response.guidance.obstacles.length}');
        debugPrint(
            '[NAV][MODE] ✓ Safety warnings: ${response.guidance.safetyWarnings}');
        debugPrint(
            '[NAV][MODE] ✓ debugFrame present: ${response.guidance.debugFrameBase64 != null}');
      } catch (parseErr, parseStack) {
        debugPrint('[NAV][MODE] ✗ JSON→Model parse FAILED: $parseErr');
        debugPrint('[NAV][MODE] STACK: $parseStack');
        rethrow;
      }

      final filteredObstacles =
          _filterStableNavigationObstacles(response.guidance.obstacles);

      // Build guidance from filtered obstacles to suppress random detections.
      final guidanceText = _buildNavigationGuidanceText(filteredObstacles);
      final navCommand =
          _deriveNavigationCommandFromObstacles(filteredObstacles);
      final commandSpeech = _navigationCommandSpeech(navCommand);
      final dangerDirection = _buildDangerDirectionPhrase(filteredObstacles);
      final warningSummary = response.guidance.safetyWarnings
          .where((warning) => warning.trim().isNotEmpty)
          .join('. ');
      final hasDanger = filteredObstacles.isNotEmpty ||
          warningSummary.toLowerCase().contains('danger') ||
          navCommand != 'PATH_CLEAR';

      String navigationSpeech = '';
      if (hasDanger) {
        navigationSpeech = _buildPriorityNavigationSpeech(
          guidanceText: guidanceText,
          dangerDirection: dangerDirection,
          commandSpeech: commandSpeech,
          warningSummary: warningSummary,
        );
      } else if (_lastNavigationCommand != 'PATH_CLEAR') {
        // Announce clear path only once when transitioning from danger to safe.
        navigationSpeech = 'Path clear.';
      }

      if (navigationSpeech.isNotEmpty) {
        final now = DateTime.now();
        final timeSinceLastSpeech =
            now.difference(_lastNavigationSpeechTime).inSeconds;
        final cooldownPassed =
          timeSinceLastSpeech >= _navigationCooldownSeconds;
        final commandChanged = navCommand != _lastNavigationCommand;
        // Speak commands on change/cooldown; "Path clear" only on transition.
        final canSpeak = hasDanger
            ? (commandChanged || cooldownPassed)
            : (_lastNavigationCommand != 'PATH_CLEAR');

        if (canSpeak && navigationSpeech.isNotEmpty) {
          _lastNavigationCommand = navCommand;
          _lastNavigationSpeechTime = now;
          final volume = _computeNavigationVolume(filteredObstacles);
          debugPrint('[NAV][MODE] ► TTS: "$navigationSpeech"');
          debugPrint('[NAV][MODE] ► TTS volume: ${volume.toStringAsFixed(2)}');
          await tts.stop();
          await tts.setVolume(volume);
          await tts.speak(navigationSpeech);
        } else if (navigationSpeech.isNotEmpty) {
          final remaining = _navigationCooldownSeconds - timeSinceLastSpeech;
          debugPrint(
              '[NAV][MODE] ⏱ Debounced (${remaining}s): "$navigationSpeech"');
        }
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
          debugPrint(
              '[NAV][MODE] ✓ Heatmap decoded: ${heatmapBytes.length} bytes');
        } catch (_) {
          debugPrint('[NAV][MODE] ✗ Failed to decode heatmap base64');
        }
      }

      return ModeInferenceResult.success(
        detections: filteredObstacles,
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

  String _normalizeNavigationSpeech(String text) {
    if (text.isEmpty) return text;
    var cleaned = text.trim();
    cleaned = cleaned.replaceAll(
        RegExp(r'\bi see\b', caseSensitive: false), 'Detected');
    cleaned = cleaned.replaceAll(RegExp(r'\s+'), ' ');
    return cleaned;
  }

  String _buildPriorityNavigationSpeech({
    required String guidanceText,
    required String dangerDirection,
    required String commandSpeech,
    required String warningSummary,
  }) {
    final normalizedGuidance = _normalizeNavigationSpeech(guidanceText);
    final normalizedDirection = _normalizeNavigationSpeech(dangerDirection);
    final normalizedCommand = _normalizeNavigationSpeech(commandSpeech);
    final normalizedWarnings = _normalizeNavigationSpeech(warningSummary);

    if (normalizedDirection.isNotEmpty) {
      if (normalizedCommand.isNotEmpty) {
        return '$normalizedDirection. $normalizedCommand';
      }
      return normalizedDirection;
    }

    if (normalizedWarnings.toLowerCase().contains('danger')) {
      return normalizedWarnings;
    }

    if (normalizedGuidance.isNotEmpty) {
      return normalizedGuidance;
    }

    if (normalizedCommand.isNotEmpty) {
      return normalizedCommand;
    }

    if (normalizedWarnings.isNotEmpty) {
      return normalizedWarnings;
    }

    return '';
  }

  String _navigationCommandSpeech(String navCommand) {
    switch (navCommand) {
      case 'MOVE_LEFT':
        return 'Move left';
      case 'MOVE_RIGHT':
        return 'Move right';
      case 'STOP':
        return 'Stop immediately';
      case 'CAUTION':
        return 'Proceed with caution';
      default:
        return '';
    }
  }

  double _computeNavigationVolume(List<Detection> obstacles) {
    if (obstacles.isEmpty) {
      return 0.45;
    }

    final alertDetections = obstacles.where((d) {
      final label = d.label.toLowerCase();
      if (_alertObjectLabels.contains(label)) return true;
      return label.contains('person') ||
          label.contains('car') ||
          label.contains('truck') ||
          label.contains('bus') ||
          label.contains('bike') ||
          label.contains('motorcycle');
    }).toList();

    final candidates = alertDetections.isNotEmpty ? alertDetections : obstacles;
    final validDistances = candidates
        .map((d) => d.distance)
        .whereType<double>()
        .where((d) => d > 0)
        .toList();

    final isAlert = alertDetections.isNotEmpty;
    if (validDistances.isEmpty) {
      return isAlert ? 0.70 : 0.40;
    }

    final closest = validDistances.reduce((a, b) => a < b ? a : b);
    if (isAlert) {
      if (closest <= 1.2) return 1.0;
      if (closest <= 2.5) return 0.85;
      if (closest <= 4.0) return 0.70;
      return 0.60;
    }

    if (closest <= 1.2) return 0.55;
    if (closest <= 2.5) return 0.45;
    if (closest <= 4.0) return 0.35;
    return 0.30;
  }

  /// Reading mode: Recognize text via OCR
  Future<ModeInferenceResult> _processReading(File imageFile) async {
    if (!NovaConfig.enableOCR) {
      print(
          '[NOVA DEBUG] ModeInferenceService: Reading/OCR disabled by config');
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
      print(
          '[NOVA DEBUG] ModeInferenceService: Recognition mode disabled by config');
      return ModeInferenceResult.error('Recognition mode disabled');
    }

    try {
      FaceDetectionResponse faceResponse = FaceDetectionResponse(
        faces: const [],
        inferenceTimeMs: 0.0,
      );
      ObjectDetectionResponse objectResponse = ObjectDetectionResponse(
        detections: const [],
        inferenceTimeMs: 0.0,
      );

      try {
        final faceResult = await ApiService.detectFaces(
          imageFile,
          confidenceThreshold: _minFaceConfidenceForRecognition,
          recognizeFaces: true,
        );
        faceResponse = FaceDetectionResponse.fromJson(faceResult);
      } catch (e) {
        debugPrint('[REC][MODE] Face recognition call failed: $e');
      }

      try {
        final objectResult = await ApiService.detectObjects(
          imageFile,
          confidenceThreshold: _minObjectConfidenceForRecognition,
        );
        objectResponse = ObjectDetectionResponse.fromJson(objectResult);
      } catch (e) {
        debugPrint('[REC][MODE] Object detection call failed: $e');
      }

      List<Map<String, dynamic>> registeredObjects = const [];
      try {
        registeredObjects = await ApiService.listRegisteredObjects();
      } catch (e) {
        debugPrint('[REC][MODE] Registered object list failed: $e');
      }

      final registeredLabelToName = <String, String>{};
      final registeredPriority = <String, bool>{};
      for (final obj in registeredObjects) {
        final label = _normalizeObjectLabel(
            (obj['target_label'] ?? '').toString().toLowerCase());
        final name = (obj['name'] ?? '').toString();
        if (label.isNotEmpty && name.isNotEmpty) {
          registeredLabelToName[label] = name;
          registeredPriority[label] = (obj['is_priority'] ?? false) == true;
        }
      }

      final recognizedObjectDetections = <Detection>[];
      final recognizedObjectRawLabel = <String, String>{};
      final seenRecognitionKeys = <String>{};

      for (final detection in objectResponse.detections) {
        final label = _normalizeObjectLabel(detection.label.toLowerCase());
        if (detection.confidence < _minObjectConfidenceForRecognition) {
          continue;
        }
        final registeredName = registeredLabelToName[label];
        if (registeredName != null) {
          final key = 'obj:$registeredName';
          final streak = (_recognitionStreaks[key] ?? 0) + 1;
          _recognitionStreaks[key] = streak;
          seenRecognitionKeys.add(key);

          if (streak < _minStableFramesForObjectRecognition) {
            continue;
          }

          recognizedObjectDetections.add(
            Detection(
              label: registeredName,
              confidence: detection.confidence,
              bbox: detection.bbox,
              distance: detection.distance,
            ),
          );
          recognizedObjectRawLabel[registeredName.toLowerCase()] = label;
        }
      }

      final stableFaceNames = <String>[];
      final stableRecognizedFaces = <Face>[];
      for (final face in faceResponse.faces) {
        final id = face.personId;
        if (id == null || id.isEmpty || id == 'unknown') {
          continue;
        }
        if (face.confidence < _minFaceConfidenceForRecognition) {
          continue;
        }
        final key = 'face:$id';
        final streak = (_recognitionStreaks[key] ?? 0) + 1;
        _recognitionStreaks[key] = streak;
        seenRecognitionKeys.add(key);
        if (streak >= _minStableFramesForFaceRecognition) {
          stableFaceNames.add(id);
          stableRecognizedFaces.add(
            Face(
              confidence: face.confidence,
              bbox: face.bbox,
              personId: id,
              embedding: face.embedding,
            ),
          );
        }
      }

      _decayRecognitionStreaks(seenRecognitionKeys);

      // Build a speech phrase from detections
      final parts = <String>[];

      if (stableFaceNames.isNotEmpty) {
        for (final id in stableFaceNames.toSet()) {
          parts.add('I can see $id');
        }
      }

      if (faceResponse.faces.isNotEmpty) {
        // Count unknowns
        final unknowns = faceResponse.faces
            .where((f) =>
                f.confidence >= _minFaceConfidenceForRecognition &&
                (f.personId == null ||
                    f.personId == 'unknown' ||
                    f.personId!.isEmpty))
            .length;
        if (unknowns > 0) {
          parts.add(unknowns == 1
              ? 'Unknown person found'
              : '$unknowns unknown people found');
        }
      }

      if (recognizedObjectDetections.isNotEmpty) {
        final priorityObjects = recognizedObjectDetections
            .where((d) {
              final rawLabel = recognizedObjectRawLabel[d.label.toLowerCase()];
              return rawLabel != null && registeredPriority[rawLabel] == true;
            })
            .map((d) => d.label)
            .toList();
        final normalObjects = recognizedObjectDetections
            .where((d) {
              final rawLabel = recognizedObjectRawLabel[d.label.toLowerCase()];
              return rawLabel == null || registeredPriority[rawLabel] != true;
            })
            .map((d) => d.label)
            .toList();

        if (priorityObjects.isNotEmpty) {
          parts.add('Priority object detected: ${priorityObjects.join(', ')}');
        }
        if (normalObjects.isNotEmpty) {
          parts.add('Detected ${normalObjects.join(', ')}');
        }
      }

      // === 8-second recognition debounce ===
      final phrase = parts.isNotEmpty ? parts.join(', ') : '';
      if (phrase.isNotEmpty) {
        final now = DateTime.now();
        final isDifferent = phrase != _lastRecognitionPhrase;
        final cooldownPassed =
            now.difference(_lastRecognitionSpeechTime).inSeconds >=
                _recognitionCooldownSeconds;

        if (isDifferent || cooldownPassed) {
          _lastRecognitionPhrase = phrase;
          _lastRecognitionSpeechTime = now;
          if (recognizedObjectDetections.isNotEmpty) {
            SystemSound.play(SystemSoundType.alert);
          }
          await tts.stop();
          await tts.speak(phrase);
        }
      }
      // Do NOT speak "No objects or faces detected" — avoid spam

      return ModeInferenceResult.success(
        detections: recognizedObjectDetections,
        faces: stableRecognizedFaces,
        inferenceTimeMs: faceResponse.inferenceTimeMs ?? 0.0,
      );
    } catch (e) {
      // Debounced error speech
      final now = DateTime.now();
      if (now.difference(_lastRecognitionSpeechTime).inSeconds >=
          _recognitionCooldownSeconds) {
        _lastRecognitionSpeechTime = now;
        await tts.speak('Recognition mode error');
      }
      return ModeInferenceResult.error(e.toString());
    }
  }

  List<Detection> _filterStableNavigationObstacles(List<Detection> raw) {
    final filtered = <Detection>[];
    final seenKeys = <String>{};

    for (final d in raw) {
      final label = d.label.toLowerCase();
      if (!_navigationAlertLabels.contains(label)) {
        continue;
      }

      final minConfidence =
          label == 'person' ? _navigationPersonMinConfidence : _navigationAlertMinConfidence;
      final minAreaRatio =
          label == 'person' ? _navigationPersonMinAreaRatio : _navigationMinAreaRatio;

      if (d.confidence < minConfidence) {
        continue;
      }

      final width = (d.bbox.right - d.bbox.left).clamp(0.0, 1.0);
      final height = (d.bbox.bottom - d.bbox.top).clamp(0.0, 1.0);
      final area = width * height;
      if (area < minAreaRatio) {
        continue;
      }

      final cx = (d.bbox.left + d.bbox.right) / 2.0;
      final zone = cx < 0.33 ? 'left' : (cx > 0.67 ? 'right' : 'center');
      final key = 'nav:$label:$zone';
      final streak = (_navigationStreaks[key] ?? 0) + 1;
      _navigationStreaks[key] = streak;
      seenKeys.add(key);

      if (streak >= _minStableFramesForNavigation) {
        filtered.add(d);
      }
    }

    _decayNavigationStreaks(seenKeys);
    return filtered;
  }

  String _buildDangerDirectionPhrase(List<Detection> obstacles) {
    if (obstacles.isEmpty) return '';

    Detection? best;
    double bestScore = double.negativeInfinity;
    for (final d in obstacles) {
      final distance = d.distance ?? 0.0;
      final width = (d.bbox.right - d.bbox.left).clamp(0.0, 1.0);
      final height = (d.bbox.bottom - d.bbox.top).clamp(0.0, 1.0);
      final area = width * height;
      final score = distance > 0 ? (1.0 / (distance + 0.01)) : area;
      if (score > bestScore) {
        bestScore = score;
        best = d;
      }
    }

    if (best == null) return '';
    final cx = (best.bbox.left + best.bbox.right) / 2.0;
    final side = cx < 0.40 ? 'left' : (cx > 0.60 ? 'right' : 'center');
    if (side == 'center') {
      return 'Danger ahead in center: ${best.label}';
    }
    return 'Danger on the $side: ${best.label}';
  }

  String _buildNavigationGuidanceText(List<Detection> obstacles) {
    if (obstacles.isEmpty) {
      return 'Clear path ahead.';
    }
    final phrase = _buildDangerDirectionPhrase(obstacles);
    return phrase.isEmpty ? 'Proceed with caution.' : phrase;
  }

  String _deriveNavigationCommandFromObstacles(List<Detection> obstacles) {
    if (obstacles.isEmpty) {
      return 'PATH_CLEAR';
    }

    Detection? best;
    double bestScore = double.negativeInfinity;
    for (final d in obstacles) {
      final distance = d.distance ?? 0.0;
      final width = (d.bbox.right - d.bbox.left).clamp(0.0, 1.0);
      final height = (d.bbox.bottom - d.bbox.top).clamp(0.0, 1.0);
      final area = width * height;
      final score = distance > 0 ? (1.0 / (distance + 0.01)) : area;
      if (score > bestScore) {
        bestScore = score;
        best = d;
      }
    }

    if (best == null) {
      return 'CAUTION';
    }

    final cx = (best.bbox.left + best.bbox.right) / 2.0;
    if (cx < 0.40) {
      return 'MOVE_RIGHT';
    }
    if (cx > 0.60) {
      return 'MOVE_LEFT';
    }
    return 'STOP';
  }

  void _decayRecognitionStreaks(Set<String> seenKeys) {
    final keys = _recognitionStreaks.keys.toList();
    for (final key in keys) {
      if (seenKeys.contains(key)) {
        continue;
      }
      final next = (_recognitionStreaks[key] ?? 0) - 1;
      if (next <= 0) {
        _recognitionStreaks.remove(key);
      } else {
        _recognitionStreaks[key] = next;
      }
    }
  }

  void _decayNavigationStreaks(Set<String> seenKeys) {
    final keys = _navigationStreaks.keys.toList();
    for (final key in keys) {
      if (seenKeys.contains(key)) {
        continue;
      }
      final next = (_navigationStreaks[key] ?? 0) - 1;
      if (next <= 0) {
        _navigationStreaks.remove(key);
      } else {
        _navigationStreaks[key] = next;
      }
    }
  }

  String _normalizeObjectLabel(String label) {
    final value = label.trim().toLowerCase();
    switch (value) {
      case 'tvremote':
      case 'remote control':
        return 'remote';
      case 'cell phone':
      case 'mobile phone':
        return 'phone';
      case 'water bottle':
        return 'bottle';
      default:
        return value;
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
