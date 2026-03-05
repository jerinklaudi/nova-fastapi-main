/// App settings model for NOVA
class AppSettings {
  // Voice & Audio Settings
  final double speechRate;
  final double volume;
  final bool enableVoiceFeedback;

  // Detection Settings
  final bool autoDetection;
  final int frameRateMs;
  final double confidenceThreshold;

  // Accessibility Settings
  final bool hapticFeedback;
  final bool highContrast;
  final bool largeText;

  // Advanced Settings
  final String backendHost;
  final int backendPort;
  final bool debugMode;

  const AppSettings({
    // Voice defaults
    this.speechRate = 0.45,
    this.volume = 1.0,
    this.enableVoiceFeedback = true,
    // Detection defaults
    this.autoDetection = true,
    this.frameRateMs = 500,
    this.confidenceThreshold = 0.5,
    // Accessibility defaults
    this.hapticFeedback = false,
    this.highContrast = false,
    this.largeText = false,
    // Advanced defaults
    this.backendHost = '10.135.136.76',
    this.backendPort = 8000,
    this.debugMode = false,
  });

  /// Create settings from JSON map
  factory AppSettings.fromJson(Map<String, dynamic> json) {
    return AppSettings(
      speechRate: json['speechRate'] ?? 0.45,
      volume: json['volume'] ?? 1.0,
      enableVoiceFeedback: json['enableVoiceFeedback'] ?? true,
      autoDetection: json['autoDetection'] ?? true,
      frameRateMs: json['frameRateMs'] ?? 500,
      confidenceThreshold: json['confidenceThreshold'] ?? 0.5,
      hapticFeedback: json['hapticFeedback'] ?? false,
      highContrast: json['highContrast'] ?? false,
      largeText: json['largeText'] ?? false,
      backendHost: json['backendHost'] ?? '10.135.136.76',
      backendPort: json['backendPort'] ?? 8000,
      debugMode: json['debugMode'] ?? false,
    );
  }

  /// Convert settings to JSON map
  Map<String, dynamic> toJson() {
    return {
      'speechRate': speechRate,
      'volume': volume,
      'enableVoiceFeedback': enableVoiceFeedback,
      'autoDetection': autoDetection,
      'frameRateMs': frameRateMs,
      'confidenceThreshold': confidenceThreshold,
      'hapticFeedback': hapticFeedback,
      'highContrast': highContrast,
      'largeText': largeText,
      'backendHost': backendHost,
      'backendPort': backendPort,
      'debugMode': debugMode,
    };
  }

  /// Create a copy with modified fields
  AppSettings copyWith({
    double? speechRate,
    double? volume,
    bool? enableVoiceFeedback,
    bool? autoDetection,
    int? frameRateMs,
    double? confidenceThreshold,
    bool? hapticFeedback,
    bool? highContrast,
    bool? largeText,
    String? backendHost,
    int? backendPort,
    bool? debugMode,
  }) {
    return AppSettings(
      speechRate: speechRate ?? this.speechRate,
      volume: volume ?? this.volume,
      enableVoiceFeedback: enableVoiceFeedback ?? this.enableVoiceFeedback,
      autoDetection: autoDetection ?? this.autoDetection,
      frameRateMs: frameRateMs ?? this.frameRateMs,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      hapticFeedback: hapticFeedback ?? this.hapticFeedback,
      highContrast: highContrast ?? this.highContrast,
      largeText: largeText ?? this.largeText,
      backendHost: backendHost ?? this.backendHost,
      backendPort: backendPort ?? this.backendPort,
      debugMode: debugMode ?? this.debugMode,
    );
  }
}
