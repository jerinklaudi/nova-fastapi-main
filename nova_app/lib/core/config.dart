/// Configuration for backend connectivity and other app settings
class NovaConfig {
  /// Backend server configuration
  /// Single source of truth for backend connectivity.
  /// Change only these two values when your backend IP/port changes.
  static const String backendHost = '172.26.234.31';
  static const int backendPort = 8000;
  static String get backendUrl => 'http://$backendHost:$backendPort';

  /// Camera settings
  static const int cameraFrameRateMs = 500; // Process frame every 500ms

  /// Inference settings
  static const double objectDetectionThreshold =
      0.5; // Increased to reduce false positives
  static const double faceDetectionThreshold = 0.35;
  static const double ocrConfidenceThreshold = 0.3;
  static const bool autoProcessFrames =
      true; // Continuous auto-detection enabled

  /// Feature flags
  static const bool enableDepthEstimation = true;
  static const bool enableFaceRecognition = true;
  static const bool enableOCR = true; // Enable Reading mode
  static const bool enableRecognitionMode = true; // Enable Recognition mode

  /// UI settings
  static const bool showFPS = false;
  static const bool showInferenceTime = true;
}
