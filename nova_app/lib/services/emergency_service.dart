import 'dart:async';
import 'dart:convert';
import 'dart:ui';

import 'package:flutter_tts/flutter_tts.dart';
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:telephony/telephony.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:vibration/vibration.dart';

/// Phase 1 Emergency System Service (v2).
///
/// Triggered by long-pressing the emergency button.
/// Full pipeline:
///   1. Vibrate + TTS warning with 5s cancel window
///   2. GPS fetch (graceful degradation)
///   3. Silent SMS to ALL registered contacts
///   4. TTS confirmation + 60s cancelable wait
///   5. Direct automatic call to FIRST contact
class EmergencyService {
  // ─── Singleton ────────────────────────────────────────────────────────────
  EmergencyService._();
  static final EmergencyService instance = EmergencyService._();

  // ─── State ────────────────────────────────────────────────────────────────
  bool _isEmergencyActive = false;
  bool _isCancelled = false;

  /// Public read-only flag so navigation can check emergency state.
  bool get isActive => _isEmergencyActive;

  /// Callbacks for HomeScreen to pause/resume navigation stream.
  /// Set these once from HomeScreen.initState — never reassign mid-emergency.
  VoidCallback? onNavigationPause;
  VoidCallback? onNavigationResume;

  // ─── Internal helpers ─────────────────────────────────────────────────────
  final Telephony _telephony = Telephony.instance;
  final stt.SpeechToText _speech = stt.SpeechToText();

  static const String _contactsKey = 'emergency_contacts';

  // ─── Public API ───────────────────────────────────────────────────────────

  /// Entry point. Pass in the app's shared [FlutterTts] instance.
  Future<void> triggerEmergency(FlutterTts tts) async {
    print('[EMERGENCY] triggerEmergency() called. isActive=$_isEmergencyActive');
    if (_isEmergencyActive) {
      print('[EMERGENCY] ⚠️  Already active — ignoring re-trigger.');
      return;
    }
    _isEmergencyActive = true;
    _isCancelled = false;

    // Pause navigation stream immediately
    _pauseNavigation();

    try {
      // ── STEP 0: Read ALL contacts ─────────────────────────────────────────
      final List<String> allPhones = await _readAllContactPhones();
      if (allPhones.isEmpty) {
        print('[EMERGENCY] ❌ No emergency contact registered.');
        await _speak(tts, 'No emergency contact registered.');
        _resumeNavigation();
        _reset();
        return;
      }
      final String primaryPhone = allPhones.first;
      print('[EMERGENCY] ✓ ${allPhones.length} contact(s) loaded. Primary: $primaryPhone');

      // ── STEP 1: Vibrate + TTS ─────────────────────────────────────────────
      await _vibrateStrong();
      await _speak(
        tts,
        'Emergency mode activated. Sending alert in 5 seconds. Say cancel to stop.',
      );

      // ── STEP 2: 5-second cancel window ───────────────────────────────────
      await _startCancelListenerWithCountdown();
      if (_isCancelled) {
        print('[EMERGENCY] 🛑 Cancelled by user during initial countdown.');
        await _speak(tts, 'Emergency cancelled.');
        _resumeNavigation();
        _reset();
        return;
      }

      // ── STEP 3: GPS ───────────────────────────────────────────────────────
      final Position? position = await _fetchLocation();

      // ── STEP 4: Permission check + Silent SMS to ALL contacts ─────────────
      final bool smsGranted = await _requestSmsPermission();
      if (!smsGranted) {
        print('[EMERGENCY] ❌ SMS permission denied.');
        await _speak(tts, 'SMS permission required for emergency.');
        _resumeNavigation();
        _reset();
        return;
      }

      for (final phone in allPhones) {
        await _sendSilentSms(phone, position);
      }

      // ── STEP 5: TTS confirmation + 60-second cancelable wait ──────────────
      await _speak(
        tts,
        'Emergency message sent. Calling primary contact in 60 seconds. Say cancel to stop.',
      );

      // Start STT listener for cancel during the 60s window
      await _startCancelListener();

      for (int i = 0; i < 60; i++) {
        await Future.delayed(const Duration(seconds: 1));
        if (_isCancelled) {
          print('[EMERGENCY] 🛑 Cancelled by user during 60s countdown (at ${i + 1}s).');
          await _speech.stop();
          await _speak(tts, 'Emergency cancelled.');
          _resumeNavigation();
          _reset();
          return;
        }
      }
      await _speech.stop();

      // ── STEP 6: Direct automatic call to first contact ────────────────────
      if (!_isCancelled) {
        // Ensure stream is fully stopped before launching external call
        _pauseNavigation();
        await _makeDirectCall(primaryPhone);
      }
    } catch (e, st) {
      print('[EMERGENCY] ❌ Unhandled error: $e\n$st');
    } finally {
      _reset();
    }
  }

  // ─── Private helpers ──────────────────────────────────────────────────────

  void _reset() {
    print('[EMERGENCY] 🔄 _reset() called. Was active=$_isEmergencyActive, wasCancelled=$_isCancelled');
    _isEmergencyActive = false;
    _isCancelled = false;
  }

  void _pauseNavigation() {
    print('[EMERGENCY] ⏸️  Requesting navigation pause.');
    onNavigationPause?.call();
  }

  void _resumeNavigation() {
    print('[EMERGENCY] ▶️  Requesting navigation resume.');
    onNavigationResume?.call();
  }

  /// Reads ALL stored emergency contact phone numbers.
  Future<List<String>> _readAllContactPhones() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_contactsKey);
      if (raw == null) return [];

      final List<dynamic> list = jsonDecode(raw);
      if (list.isEmpty) return [];

      final phones = <String>[];
      for (final item in list) {
        final map = item as Map<String, dynamic>;
        final phone = map['phone'] as String?;
        if (phone != null && phone.trim().isNotEmpty) {
          phones.add(phone.trim());
        }
      }
      return phones;
    } catch (e) {
      print('[EMERGENCY] ❌ Failed to read contacts: $e');
      return [];
    }
  }

  /// Strong vibration pattern: 3× long pulses.
  Future<void> _vibrateStrong() async {
    try {
      if (await Vibration.hasVibrator() ?? false) {
        Vibration.vibrate(
          pattern: [0, 600, 200, 600, 200, 600],
          intensities: [0, 255, 0, 255, 0, 255],
        );
      }
    } catch (e) {
      print('[EMERGENCY] Vibration error (non-fatal): $e');
    }
  }

  /// Speak [text] and wait for completion.
  Future<void> _speak(FlutterTts tts, String text) async {
    try {
      await tts.awaitSpeakCompletion(true);
      await tts.speak(text);
    } catch (e) {
      print('[EMERGENCY] TTS error (non-fatal): $e');
    }
  }

  /// Listens for "cancel" keyword for 5 seconds (initial countdown).
  Future<void> _startCancelListenerWithCountdown() async {
    bool speechAvailable = false;
    try {
      speechAvailable = await _speech.initialize(
        onError: (e) => print('[EMERGENCY] STT error: $e'),
      );
    } catch (e) {
      print('[EMERGENCY] STT init failed (non-fatal): $e');
    }

    if (speechAvailable) {
      try {
        await _speech.listen(
          onResult: (result) {
            final words = result.recognizedWords.toLowerCase();
            print('[EMERGENCY] STT heard: "$words"');
            if (words.contains('cancel')) {
              _isCancelled = true;
              _speech.stop();
            }
          },
          listenFor: const Duration(seconds: 5),
          pauseFor: const Duration(seconds: 5),
          localeId: 'en_US',
          listenOptions: stt.SpeechListenOptions(cancelOnError: false),
        );
      } catch (e) {
        print('[EMERGENCY] STT listen error (non-fatal): $e');
      }
    }

    // Always wait the full 5 seconds regardless of STT availability
    await Future.delayed(const Duration(seconds: 5));
    await _speech.stop();
  }

  /// Starts STT listener for "cancel" (used during 60s window).
  /// Does NOT block — the caller loops and checks _isCancelled.
  Future<void> _startCancelListener() async {
    bool speechAvailable = false;
    try {
      speechAvailable = await _speech.initialize(
        onError: (e) => print('[EMERGENCY] STT error: $e'),
      );
    } catch (e) {
      print('[EMERGENCY] STT init failed for 60s window (non-fatal): $e');
    }

    if (speechAvailable) {
      try {
        await _speech.listen(
          onResult: (result) {
            final words = result.recognizedWords.toLowerCase();
            print('[EMERGENCY] STT heard (60s window): "$words"');
            if (words.contains('cancel')) {
              _isCancelled = true;
              _speech.stop();
            }
          },
          listenFor: const Duration(seconds: 60),
          pauseFor: const Duration(seconds: 60),
          localeId: 'en_US',
          listenOptions: stt.SpeechListenOptions(cancelOnError: false),
        );
      } catch (e) {
        print('[EMERGENCY] STT listen error for 60s window (non-fatal): $e');
      }
    }
  }

  /// Requests SMS send permission via permission_handler.
  Future<bool> _requestSmsPermission() async {
    try {
      final status = await Permission.sms.request();
      print('[EMERGENCY] SMS permission: $status');
      return status.isGranted;
    } catch (e) {
      print('[EMERGENCY] SMS permission error: $e');
      return false;
    }
  }

  /// Fetches GPS location. Returns null gracefully on any failure.
  Future<Position?> _fetchLocation() async {
    try {
      // 1. Check if location services are enabled
      final bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      print('[EMERGENCY] GPS service enabled: $serviceEnabled');
      if (!serviceEnabled) {
        print('[EMERGENCY] ❌ Location services are OFF — continuing without GPS.');
        return null;
      }

      // 2. Check permission
      LocationPermission perm = await Geolocator.checkPermission();
      print('[EMERGENCY] GPS permission (before request): $perm');

      if (perm == LocationPermission.denied) {
        print('[EMERGENCY] Requesting location permission…');
        perm = await Geolocator.requestPermission();
        print('[EMERGENCY] GPS permission (after request): $perm');
      }

      if (perm == LocationPermission.denied) {
        print('[EMERGENCY] ❌ Location permission denied — continuing without GPS.');
        return null;
      }
      if (perm == LocationPermission.deniedForever) {
        print('[EMERGENCY] ❌ Location permission permanently denied — continuing without GPS.');
        return null;
      }

      print('[EMERGENCY] ✓ Location permission granted: $perm');

      // 3. Small delay to let GPS hardware warm up after permission dialog
      print('[EMERGENCY] Waiting 2s for GPS hardware warm-up…');
      await Future.delayed(const Duration(seconds: 2));

      // 4. Get position (no timeLimit — let it complete naturally)
      print('[EMERGENCY] Requesting current position (accuracy: best)…');
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.best,
      );
      print('[EMERGENCY] ✓ GPS acquired: lat=${position.latitude}, lng=${position.longitude}, accuracy=${position.accuracy}m');
      return position;
    } catch (e) {
      print('[EMERGENCY] ❌ GPS error (non-fatal, continuing without location): $e');
      return null;
    }
  }

  /// Builds the SMS body.
  String _buildSmsBody(Position? position) {
    print('[EMERGENCY] Building SMS body. Has position: ${position != null}');
    final buffer = StringBuffer(
      'Emergency Alert from NOVA.\n'
      'I may be in danger. Please contact me immediately.',
    );
    if (position != null) {
      final lat = position.latitude.toStringAsFixed(6);
      final lng = position.longitude.toStringAsFixed(6);
      buffer.writeln('\n\nLive location:\nhttps://maps.google.com/?q=$lat,$lng');
      print('[EMERGENCY] ✓ Location link appended to SMS: $lat,$lng');
    } else {
      print('[EMERGENCY] ⚠️  No location available — SMS sent without coordinates.');
    }
    return buffer.toString();
  }

  /// Sends a fully silent background SMS using the [Telephony] package.
  Future<void> _sendSilentSms(String phone, Position? position) async {
    final body = _buildSmsBody(position);
    print('[EMERGENCY] 📤 Sending silent SMS to $phone…');

    try {
      final bool? permGranted = await _telephony.requestPhoneAndSmsPermissions;
      if (permGranted != true) {
        print('[EMERGENCY] ❌ Telephony permission denied for $phone.');
        return;
      }

      await _telephony.sendSms(
        to: phone,
        message: body,
        statusListener: (SendStatus status) {
          print('[EMERGENCY] SMS status for $phone: $status');
        },
        isMultipart: true,
      );
      print('[EMERGENCY] ✓ SMS sent to $phone.');
    } catch (e) {
      print('[EMERGENCY] ❌ SMS send error for $phone (continuing): $e');
    }
  }

  /// Makes a direct automatic phone call using ACTION_CALL intent.
  /// Falls back to tel: URI (dialer) if CALL_PHONE permission is denied.
  Future<void> _makeDirectCall(String phone) async {
    print('[EMERGENCY] 📞 Initiating direct call to $phone…');
    try {
      // Request CALL_PHONE permission
      final phonePermission = await Permission.phone.request();
      print('[EMERGENCY] Phone permission: $phonePermission');

      if (phonePermission.isGranted) {
        // Direct call via ACTION_CALL intent
        final Uri callUri = Uri.parse('tel:$phone');
        await launchUrl(
          callUri,
          mode: LaunchMode.externalApplication,
        );
        print('[EMERGENCY] ✓ Direct call launched.');
      } else {
        // Fallback: open dialer UI
        print('[EMERGENCY] ⚠️  CALL_PHONE denied — falling back to dialer.');
        final uri = Uri(scheme: 'tel', path: phone);
        if (await canLaunchUrl(uri)) {
          await launchUrl(uri);
          print('[EMERGENCY] ✓ Dialer fallback launched.');
        } else {
          print('[EMERGENCY] ❌ Cannot launch tel: URI.');
        }
      }
    } catch (e) {
      print('[EMERGENCY] ❌ Call error (non-fatal): $e');
    }
  }
}
