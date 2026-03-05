import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart' as http_parser;
import '../core/config.dart';

/// HTTP client for communicating with FastAPI backend
class ApiService {
  static String get baseUrl => NovaConfig.backendUrl;
  static const Duration timeout = Duration(seconds: 60);

  /// Detect objects in image
  /// Returns JSON response with detection results
  static Future<Map<String, dynamic>> detectObjects(
    File imageFile, {
    double confidenceThreshold = 0.5,
  }) async {
    final uri = Uri.parse('$baseUrl/detect/objects')
        .replace(queryParameters: {
      'confidence_threshold': confidenceThreshold.toString(),
    });

    try {
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      final response = await request.send().timeout(timeout);
      
      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else {
        throw Exception('Failed to detect objects: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Object detection error: $e');
    }
  }

  /// Detect faces in image
  static Future<Map<String, dynamic>> detectFaces(
    File imageFile, {
    double confidenceThreshold = 0.5,
    bool recognizeFaces = false,
  }) async {
    final uri = Uri.parse('$baseUrl/detect/faces').replace(queryParameters: {
      'confidence_threshold': confidenceThreshold.toString(),
      'recognize_faces': recognizeFaces.toString(),
    });

    try {
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      final response = await request.send().timeout(timeout);

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else {
        throw Exception('Failed to detect faces: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Face detection error: $e');
    }
  }

  /// Recognize text in image (OCR)
  static Future<Map<String, dynamic>> recognizeText(File imageFile) async {
    final uri = Uri.parse('$baseUrl/detect/text');

    try {
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      final response = await request.send().timeout(timeout);

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else {
        throw Exception('Failed to recognize text: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('OCR error: $e');
    }
  }

  /// Estimate depth map
  static Future<Map<String, dynamic>> estimateDepth(File imageFile) async {
    final uri = Uri.parse('$baseUrl/detect/depth');

    try {
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      final response = await request.send().timeout(timeout);

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else {
        throw Exception('Failed to estimate depth: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Depth estimation error: $e');
    }
  }

  /// Get navigation guidance (combined detection + depth)
  static Future<Map<String, dynamic>> getNavigationGuidance(
    File imageFile,
  ) async {
    final uri = Uri.parse('$baseUrl/detect/navigation');

    try {
      // Determine content type from file extension
      String contentType = 'image/jpeg';
      if (imageFile.path.toLowerCase().endsWith('.png')) {
        contentType = 'image/png';
      }

      // ── STEP 2: Log request details ──────────────────────────────────
      final fileSize = await imageFile.length();
      debugPrint('[NAV][API] ► Full request URL: $uri');
      debugPrint('[NAV][API] ► Content-Type: $contentType');
      debugPrint('[NAV][API] ► File size: $fileSize bytes');

      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
          contentType: http_parser.MediaType.parse(contentType),
        ));

      final stopwatch = Stopwatch()..start();
      final response = await request.send().timeout(timeout);
      stopwatch.stop();

      // ── STEP 2: Log response timing ───────────────────────────────────
      debugPrint('[NAV][API] ◄ HTTP Status: ${response.statusCode}');
      debugPrint('[NAV][API] ◄ Response time: ${stopwatch.elapsedMilliseconds}ms');

      final responseBody = await response.stream.bytesToString();
      debugPrint('[NAV][API] ◄ Response body length: ${responseBody.length}');

      // ── STEP 3: Detect backend 500 ────────────────────────────────────
      if (response.statusCode != 200) {
        debugPrint('[NAV][API] ✗ BACKEND ERROR ${response.statusCode}:');
        debugPrint('[NAV][API] ✗ Body: $responseBody');
        throw Exception(
          'Navigation API returned ${response.statusCode}: $responseBody',
        );
      }

      // ── STEP 1: Log raw response ──────────────────────────────────────
      debugPrint('[NAV][API] ◄ Raw response (first 500 chars): '
          '${responseBody.length > 500 ? responseBody.substring(0, 500) : responseBody}');

      // ── STEP 1: Log JSON decode ────────────────────────────────────────
      debugPrint('[NAV][API] Attempting JSON decode...');
      Map<String, dynamic> decoded;
      try {
        decoded = jsonDecode(responseBody) as Map<String, dynamic>;
        debugPrint('[NAV][API] ✓ JSON decode success. Keys: ${decoded.keys.toList()}');
      } catch (jsonErr) {
        debugPrint('[NAV][API] ✗ JSON decode FAILED: $jsonErr');
        rethrow;
      }

      // ── STEP 4: Validate NavigationGuidanceResult fields ─────────────
      final guidance = decoded['guidance'];
      debugPrint('[NAV][API] guidance field present: ${guidance != null}');
      if (guidance is Map) {
        debugPrint('[NAV][API]   obstacles present: ${guidance['obstacles'] != null}');
        debugPrint('[NAV][API]   guidance string: "${guidance['guidance']}"');
        debugPrint('[NAV][API]   safety_warnings: ${guidance['safety_warnings']}');
        debugPrint('[NAV][API]   inference_time_ms: ${guidance['inference_time_ms']}');
      } else {
        debugPrint('[NAV][API] ✗ guidance field is null or not a Map — value=$guidance');
      }
      debugPrint('[NAV][API] inference_time_ms (top): ${decoded['inference_time_ms']}');

      return decoded;
    } catch (e, s) {
      // ── STEP 6: Full exception + stacktrace ───────────────────────────
      debugPrint('[NAV][API] ✗ EXCEPTION in getNavigationGuidance: $e');
      debugPrint('[NAV][API] STACKTRACE:\n$s');
      throw Exception('Navigation guidance error: $e');
    }
  }


  /// Health check
  static Future<bool> healthCheck() async {
    try {
      final response =
          await http.get(Uri.parse('$baseUrl/health/')).timeout(timeout);
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
