import 'package:flutter/material.dart';
import '../models/detection_models.dart';

/// Custom painter for drawing detection overlays on camera preview
class DetectionOverlayPainter extends CustomPainter {
  final List<Detection> detections;
  final List<TextRegion> textRegions;
  final List<Face> faces;
  final Size imageSize;
  final Size canvasSize;

  DetectionOverlayPainter({
    this.detections = const [],
    this.textRegions = const [],
    this.faces = const [],
    required this.imageSize,
    required this.canvasSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw faces: Green if recognized, Red if unknown
    for (final face in faces) {
      final bool isRecognized = face.personId != null && 
                                face.personId!.isNotEmpty && 
                                face.personId != 'unknown';
      final Color boxColor = isRecognized ? Colors.green : Colors.red;
      final String label = isRecognized ? face.personId! : 'Unknown person';

      _drawBoundingBox(
        canvas,
        face.bbox,
        boxColor,
        '$label (${(face.confidence * 100).toStringAsFixed(0)}%)',
      );
    }

    // Draw text regions with blue bounding boxes
    for (final textRegion in textRegions) {
      _drawBoundingBox(
        canvas,
        textRegion.bbox,
        Colors.blue,
        textRegion.text,
      );
    }

    // Draw object detections with yellow bounding boxes
    for (final detection in detections) {
      _drawBoundingBox(
        canvas,
        detection.bbox,
        Colors.yellow,
        '${detection.label} (${(detection.confidence * 100).toStringAsFixed(0)}%)',
      );
    }
  }

  /// Draw a single bounding box with label
  void _drawBoundingBox(
    Canvas canvas,
    BoundingBox bbox,
    Color color,
    String label,
  ) {
    // Scale bounding box coordinates to fit canvas size
    final left = bbox.left * canvasSize.width;
    final top = bbox.top * canvasSize.height;
    final right = bbox.right * canvasSize.width;
    final bottom = bbox.bottom * canvasSize.height;

    final rect = Rect.fromLTRB(left, top, right, bottom);

    // Draw border
    final paint = Paint()
      ..color = color
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    canvas.drawRect(rect, paint);

    // Draw label background
    final textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();

    final labelBg = Rect.fromLTWH(
      left,
      top - textPainter.height - 8,
      textPainter.width + 8,
      textPainter.height + 4,
    );

    final bgPaint = Paint()..color = color;
    canvas.drawRect(labelBg, bgPaint);

    // Draw label text
    textPainter.paint(
      canvas,
      Offset(left + 4, top - textPainter.height - 4),
    );
  }

  @override
  bool shouldRepaint(DetectionOverlayPainter oldDelegate) {
    return detections != oldDelegate.detections ||
        textRegions != oldDelegate.textRegions ||
        faces != oldDelegate.faces;
  }
}
