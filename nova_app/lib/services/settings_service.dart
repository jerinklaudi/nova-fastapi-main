import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/app_settings.dart';

/// Service for managing app settings persistence
class SettingsService {
  static const String _settingsKey = 'nova_app_settings';
  static SettingsService? _instance;
  static AppSettings _currentSettings = const AppSettings();

  SettingsService._();

  /// Get singleton instance
  static SettingsService get instance {
    _instance ??= SettingsService._();
    return _instance!;
  }

  /// Get current settings
  AppSettings get settings => _currentSettings;

  /// Load settings from storage
  Future<AppSettings> loadSettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonString = prefs.getString(_settingsKey);
      
      if (jsonString != null) {
        final json = jsonDecode(jsonString) as Map<String, dynamic>;
        _currentSettings = AppSettings.fromJson(json);
        print('[Settings] Loaded settings from storage');
      } else {
        _currentSettings = const AppSettings();
        print('[Settings] No saved settings found, using defaults');
      }
    } catch (e) {
      print('[Settings] Error loading settings: $e');
      _currentSettings = const AppSettings();
    }
    
    return _currentSettings;
  }

  /// Save settings to storage
  Future<bool> saveSettings(AppSettings settings) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonString = jsonEncode(settings.toJson());
      final success = await prefs.setString(_settingsKey, jsonString);
      
      if (success) {
        _currentSettings = settings;
        print('[Settings] Settings saved successfully');
      }
      
      return success;
    } catch (e) {
      print('[Settings] Error saving settings: $e');
      return false;
    }
  }

  /// Reset settings to defaults
  Future<bool> resetToDefaults() async {
    return await saveSettings(const AppSettings());
  }

  /// Update specific setting
  Future<bool> updateSetting({
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
  }) async {
    final newSettings = _currentSettings.copyWith(
      speechRate: speechRate,
      volume: volume,
      enableVoiceFeedback: enableVoiceFeedback,
      autoDetection: autoDetection,
      frameRateMs: frameRateMs,
      confidenceThreshold: confidenceThreshold,
      hapticFeedback: hapticFeedback,
      highContrast: highContrast,
      largeText: largeText,
      backendHost: backendHost,
      backendPort: backendPort,
      debugMode: debugMode,
    );
    
    return await saveSettings(newSettings);
  }
}
