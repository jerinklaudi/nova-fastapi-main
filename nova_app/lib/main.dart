import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'screens/home_screen.dart';
import 'services/settings_service.dart';

import 'services/activation_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // ── STEP 6: Global Flutter error handler ─────────────────────────────
  FlutterError.onError = (FlutterErrorDetails details) {
    debugPrint('[FLUTTER GLOBAL ERROR] ${details.exception}');
    debugPrint('[FLUTTER GLOBAL ERROR] STACK:\n${details.stack}');
  };

  // Initialize settings early
  await SettingsService.instance.loadSettings();
  
  // Initialize Activation Service (Native Bridge)
  await ActivationService.instance.initialize();
  
  runApp(const NovaApp());
}

class NovaApp extends StatelessWidget {
  const NovaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NOVA',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.yellow),
        useMaterial3: true,
      ),
      home: const ActivationScreen(),
    );
  }
}

class ActivationScreen extends StatefulWidget {
  const ActivationScreen({super.key});

  @override
  State<ActivationScreen> createState() => _ActivationScreenState();
}

class _ActivationScreenState extends State<ActivationScreen> {
  final FlutterTts tts = FlutterTts();
  final SpeechToText _speechToText = SpeechToText();
  bool _isListening = false;
  String _statusMessage = "Initializing...";

  @override
  void initState() {
    super.initState();
    _initializeSpeech();
    speakWelcome();
  }

  @override
  void dispose() {
    _speechToText.stop();
    _speechToText.cancel();
    super.dispose();
  }

  Future<void> speakWelcome() async {
    await tts.setLanguage("en-US");
    await tts.setSpeechRate(0.5);
    await tts.speak("Welcome to NOVA. Say Activate, Start, or Nova to begin.");
  }

  Future<void> _initializeSpeech() async {
    // Request microphone permission
    final status = await Permission.microphone.request();

    if (!mounted) return;

    if (status.isGranted) {
      bool available = await _speechToText.initialize(
        onStatus: (status) {
          if (!mounted) return;
          if (status == 'notListening') {
            setState(() => _isListening = false);
            // Restart listening if not activated
            Future.delayed(const Duration(seconds: 1), () {
              if (mounted) _startListening();
            });
          } else if (status == 'listening') {
            setState(() => _isListening = true);
          }
        },
        onError: (error) {
          if (!mounted) return;
          setState(() {
            _statusMessage = "Voice activation unavailable";
            _isListening = false;
          });
        },
      );

      if (!mounted) return;

      if (available) {
        _startListening();
      } else {
        setState(() {
          _statusMessage = "Voice recognition not available";
        });
      }
    } else {
      if (!mounted) return;
      setState(() {
        _statusMessage = "Microphone permission denied. Use button to activate.";
      });
    }
  }

  void _startListening() async {
    if (!mounted || _isListening || !_speechToText.isAvailable) return;

    setState(() {
      _isListening = true;
      _statusMessage = "Listening for activation command...";
    });

    await _speechToText.listen(
      onResult: (result) {
        if (!mounted) return;
        final words = result.recognizedWords.toLowerCase();
        if (words.contains('activate') ||
            words.contains('start') ||
            words.contains('nova')) {
          _activateApp();
        }
      },
      listenMode: ListenMode.confirmation,
      cancelOnError: true,
      partialResults: true,
    );
  }

  void _activateApp() async {
    await _speechToText.stop();
    await tts.speak("Activating NOVA");

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const HomeScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Microphone icon with listening animation
            if (_isListening)
              Container(
                margin: const EdgeInsets.only(bottom: 30),
                child: Icon(
                  Icons.mic,
                  size: 80,
                  color: Colors.red.shade400,
                ),
              ),

            // Status message
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
              child: Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),

            const SizedBox(height: 20),

            // Activation button
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.yellow,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(
                  horizontal: 40,
                  vertical: 20,
                ),
                textStyle: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              onPressed: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (_) => const HomeScreen()),
                );
              },
              child: const Text("TAP TO ACTIVATE NOVA"),
            ),
          ],
        ),
      ),
    );
  }
}
