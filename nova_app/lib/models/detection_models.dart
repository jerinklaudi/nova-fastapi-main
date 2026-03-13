/// Models for parsing backend responses

class BoundingBox {
  final double left;
  final double top;
  final double right;
  final double bottom;

  BoundingBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    try {
      final left = (json['left'] as num?)?.toDouble() ?? 0.0;
      final top = (json['top'] as num?)?.toDouble() ?? 0.0;
      final right = (json['right'] as num?)?.toDouble() ?? 0.0;
      final bottom = (json['bottom'] as num?)?.toDouble() ?? 0.0;
      
      // Validate bounding box values
      if (left < 0 || top < 0 || right < 0 || bottom < 0) {
        print('[NOVA DEBUG] ⚠️  Invalid bbox values: left=$left, top=$top, right=$right, bottom=$bottom');
      }
      
      return BoundingBox(
        left: left,
        top: top,
        right: right,
        bottom: bottom,
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing BoundingBox: $e, json=$json');
      return BoundingBox(left: 0, top: 0, right: 0, bottom: 0);
    }
  }
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox;

  Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    try {
      final label = json['label'] as String? ?? 'unknown';
      final confidence = (json['confidence'] as num?)?.toDouble() ?? 0.0;
      
      // Validate confidence value
      if (confidence < 0 || confidence > 1) {
        print('[NOVA DEBUG] ⚠️  Invalid confidence value: $confidence (should be 0-1)');
      }
      
      return Detection(
        label: label,
        confidence: confidence.clamp(0.0, 1.0),
        bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>? ?? {}),
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing Detection: $e, json=$json');
      return Detection(
        label: 'error',
        confidence: 0.0,
        bbox: BoundingBox(left: 0, top: 0, right: 0, bottom: 0),
      );
    }
  }
}

class ObjectDetectionResponse {
  final List<Detection> detections;
  final double? inferenceTimeMs;

  ObjectDetectionResponse({
    required this.detections,
    this.inferenceTimeMs,
  });

  factory ObjectDetectionResponse.fromJson(Map<String, dynamic> json) {
    try {
      final detectionsList = <Detection>[];
      final rawDetections = json['detections'] as List<dynamic>? ?? [];
      
      for (final d in rawDetections) {
        try {
          detectionsList.add(Detection.fromJson(d as Map<String, dynamic>));
        } catch (e) {
          print('[NOVA DEBUG] ⚠️  Skipped malformed detection: $d, error: $e');
        }
      }

      if (detectionsList.isEmpty && rawDetections.isNotEmpty) {
        print('[NOVA DEBUG] ⚠️  Response had ${rawDetections.length} detections but all failed to parse');
      }

      return ObjectDetectionResponse(
        detections: detectionsList,
        inferenceTimeMs: (json['inference_time_ms'] as num?)?.toDouble(),
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing ObjectDetectionResponse: $e');
      print('[NOVA DEBUG] Raw JSON: ${json.toString().substring(0, 200)}...');
      return ObjectDetectionResponse(detections: [], inferenceTimeMs: null);
    }
  }
}

class Face {
  final double confidence;
  final BoundingBox bbox;
  final String? personId;
  final List<double>? embedding;

  Face({
    required this.confidence,
    required this.bbox,
    this.personId,
    this.embedding,
  });

  factory Face.fromJson(Map<String, dynamic> json) {
    try {
      final confidence = (json['confidence'] as num?)?.toDouble() ?? 0.0;
      
      // Validate confidence
      if (confidence < 0 || confidence > 1) {
        print('[NOVA DEBUG] ⚠️  Invalid face confidence: $confidence');
      }
      
      final embeddingList = <double>[];
      final rawEmbedding = json['embedding'] as List<dynamic>?;
      if (rawEmbedding != null) {
        try {
          embeddingList.addAll(rawEmbedding.map((e) => (e as num).toDouble()));
        } catch (e) {
          print('[NOVA DEBUG] ⚠️  Error parsing embedding: $e');
        }
      }

      return Face(
        confidence: confidence.clamp(0.0, 1.0),
        bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>? ?? {}),
        personId: json['person_id'] as String?,
        embedding: embeddingList.isNotEmpty ? embeddingList : null,
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing Face: $e, json=$json');
      return Face(
        confidence: 0.0,
        bbox: BoundingBox(left: 0, top: 0, right: 0, bottom: 0),
      );
    }
  }
}

class FaceDetectionResponse {
  final List<Face> faces;
  final double? inferenceTimeMs;
  final String? audioDescription;

  FaceDetectionResponse({
    required this.faces,
    this.inferenceTimeMs,
    this.audioDescription,
  });

  factory FaceDetectionResponse.fromJson(Map<String, dynamic> json) {
    try {
      final facesList = <Face>[];
      final rawFaces = json['faces'] as List<dynamic>? ?? [];
      
      for (final f in rawFaces) {
        try {
          facesList.add(Face.fromJson(f as Map<String, dynamic>));
        } catch (e) {
          print('[NOVA DEBUG] ⚠️  Skipped malformed face: $f, error: $e');
        }
      }

      if (facesList.isEmpty && rawFaces.isNotEmpty) {
        print('[NOVA DEBUG] ⚠️  Response had ${rawFaces.length} faces but all failed to parse');
      }

      return FaceDetectionResponse(
        faces: facesList,
        inferenceTimeMs: (json['inference_time_ms'] as num?)?.toDouble(),
        audioDescription: json['audio_description'] as String?,
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing FaceDetectionResponse: $e');
      return FaceDetectionResponse(faces: [], inferenceTimeMs: null);
    }
  }
}

class TextRegion {
  final String text;
  final double confidence;
  final BoundingBox bbox;

  TextRegion({
    required this.text,
    required this.confidence,
    required this.bbox,
  });

  factory TextRegion.fromJson(Map<String, dynamic> json) {
    try {
      final text = json['text'] as String? ?? '';
      final confidence = (json['confidence'] as num?)?.toDouble() ?? 0.0;
      
      // Validate confidence
      if (confidence < 0 || confidence > 1) {
        print('[NOVA DEBUG] ⚠️  Invalid text confidence: $confidence');
      }
      
      return TextRegion(
        text: text,
        confidence: confidence.clamp(0.0, 1.0),
        bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>? ?? {}),
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing TextRegion: $e, json=$json');
      return TextRegion(
        text: 'ERROR',
        confidence: 0.0,
        bbox: BoundingBox(left: 0, top: 0, right: 0, bottom: 0),
      );
    }
  }
}

class TextDetectionResponse {
  final List<TextRegion> textRegions;
  final double? inferenceTimeMs;

  TextDetectionResponse({
    required this.textRegions,
    this.inferenceTimeMs,
  });

  factory TextDetectionResponse.fromJson(Map<String, dynamic> json) {
    try {
      final regionsList = <TextRegion>[];
      final rawRegions = json['text_regions'] as List<dynamic>? ?? [];
      
      for (final r in rawRegions) {
        try {
          regionsList.add(TextRegion.fromJson(r as Map<String, dynamic>));
        } catch (e) {
          print('[NOVA DEBUG] ⚠️  Skipped malformed text region: $r, error: $e');
        }
      }

      if (regionsList.isEmpty && rawRegions.isNotEmpty) {
        print('[NOVA DEBUG] ⚠️  Response had ${rawRegions.length} text regions but all failed to parse');
      }

      return TextDetectionResponse(
        textRegions: regionsList,
        inferenceTimeMs: (json['inference_time_ms'] as num?)?.toDouble(),
      );
    } catch (e) {
      print('[NOVA DEBUG] ❌ Error parsing TextDetectionResponse: $e');
      return TextDetectionResponse(textRegions: [], inferenceTimeMs: null);
    }
  }
}

class NavigationGuidance {
  final List<Detection> obstacles;
  final List<TextRegion> textSigns;
  final String guidance;
  final List<String> safetyWarnings;
  final String? debugFrameBase64;

  NavigationGuidance({
    required this.obstacles,
    required this.textSigns,
    required this.guidance,
    required this.safetyWarnings,
    this.debugFrameBase64,
  });

  factory NavigationGuidance.fromJson(Map<String, dynamic> json) {
    final obstaclesList = (json['obstacles'] as List<dynamic>? ?? [])
        .map((o) => Detection.fromJson(o as Map<String, dynamic>))
        .toList();

    final textList = (json['text_signs'] as List<dynamic>? ?? [])
        .map((t) => TextRegion.fromJson(t as Map<String, dynamic>))
        .toList();

    return NavigationGuidance(
      obstacles: obstaclesList,
      textSigns: textList,
      guidance: json['guidance'] as String? ?? '',
      safetyWarnings: List<String>.from(json['safety_warnings'] as List? ?? []),
      debugFrameBase64: json['debug_frame_base64'] as String?,
    );
  }
}

class NavigationGuidanceResponse {
  final NavigationGuidance guidance;
  final double inferenceTimeMs;

  NavigationGuidanceResponse({
    required this.guidance,
    required this.inferenceTimeMs,
  });

  factory NavigationGuidanceResponse.fromJson(Map<String, dynamic> json) {
    return NavigationGuidanceResponse(
      guidance: NavigationGuidance.fromJson(json['guidance'] as Map<String, dynamic>),
      inferenceTimeMs: (json['inference_time_ms'] as num? ?? 0.0).toDouble(),
    );
  }
}
