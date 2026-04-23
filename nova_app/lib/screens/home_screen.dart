import 'dart:io';
import 'dart:typed_data';
import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import '../core/modes.dart';
import '../core/config.dart';
import '../services/api_service.dart';
import '../services/mode_inference_service.dart';
import '../services/settings_service.dart';
import '../models/app_settings.dart';
import '../widgets/detection_overlay.dart';
import '../services/activation_service.dart';
import '../services/emergency_service.dart';
import 'settings_screen.dart';
import '../main.dart'; // For ActivationScreen

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  StreamSubscription<String>? _activationSub;
  bool _isDisposed = false;
  NovaMode _currentMode = NovaMode.navigation;
  final FlutterTts _tts = FlutterTts();
  late ModeInferenceService _inferenceService;

  // Settings
  final _settingsService = SettingsService.instance;
  AppSettings _settings = AppSettings();

  // Inference state
  bool _isProcessing = false;
  bool _inferenceInProgress = false; // Throttle lock for concurrent inferences
  ModeInferenceResult? _lastResult;
  NovaMode? _lastInferenceMode; // Track which mode last performed inference
  String _statusMessage = 'Ready';
  bool _backendConnected = false;
  int _frameCount = 0; // For frame-skip throttling in image stream
  DateTime _lastFrameProcessedAt = DateTime.fromMillisecondsSinceEpoch(0);
  bool _isNavigationActive =
      false; // Flag to prevent eager TTS/stream on startup
  Uint8List? _heatmapImage; // Latest heatmap frame from backend (decoded JPEG)
  String _navCommand =
      'PATH_CLEAR'; // Latest canonical command for banner color

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _inferenceService = ModeInferenceService(tts: _tts);

    // Register emergency ↔ navigation callbacks
    EmergencyService.instance.onNavigationPause = _pauseNavigationForEmergency;
    EmergencyService.instance.onNavigationResume =
        _resumeNavigationAfterEmergency;

    // Listen for Activations (Native or Voice or Shake)
    _activationSub = ActivationService.instance.onTrigger.listen((mode) {
      print("Activation Triggered: $mode");
      if (mode == "NAVIGATION_MODE") {
        _onModeChanged(NovaMode.navigation);
      } else if (mode == "RECOGNITION_MODE") {
        _onModeChanged(NovaMode.recognition);
      } else if (mode == "EMERGENCY_MODE") {
        _onModeChanged(NovaMode.emergency);
      } else if (mode == "ACTIVATION_SCREEN") {
        // Navigate back to Activation Screen
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (context) => const ActivationScreen()),
          (route) => false,
        );
      }
    });

    // Run async initialization in the correct order
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      // 1. Load settings first — must complete before TTS reads them
      await _loadSettings();
      // 2. Apply settings to TTS
      await _initializeTTS();
      // 3. Start camera
      await _initializeCamera();
      // 4. Backend health check for status indicator
      await _checkBackendConnection();
      // 5. Start Voice Listener for "Start NOVA"
      ActivationService.instance.startVoiceListener();
    } catch (e) {
      print('[NOVA DEBUG] ❌ Initialization failed: $e');
      _updateStatus('Initialization failed. Tap Retry.');
    }
  }

  Future<void> _loadSettings() async {
    _settings = await _settingsService.loadSettings();
    print(
        '[NOVA DEBUG] Settings loaded: speechRate=${_settings.speechRate}, autoDetection=${_settings.autoDetection}');
  }

  Future<void> _initializeTTS() async {
    await _tts.setSpeechRate(_settings.speechRate);
    await _tts.setVolume(_settings.volume);
  }

  Future<void> _initializeCamera() async {
    try {
      print('[NOVA DEBUG] Initializing camera...');
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        print('[NOVA DEBUG] ❌ No cameras available');
        _updateStatus('No cameras available');
        return;
      }
      print('[NOVA DEBUG] ✓ ${_cameras!.length} camera(s) found');

      _controller = CameraController(
        _cameras![0],
        ResolutionPreset.low,
        enableAudio: false,
      );

      await _controller!.initialize();
      print(
          '[NOVA DEBUG] ✓ Camera initialized (Resolution: ${_controller!.value.previewSize})');

      // Stop flash strictly
      await _controller!.setFlashMode(FlashMode.off);
      print('[NOVA DEBUG] ✓ Flash disabled');

      // Do NOT start image stream here to avoid startup lag.
      // It will be started later via _manageImageStream depending on mode.

      _updateStatus('Camera ready');

      // Post-frame callback to log exact render time
      WidgetsBinding.instance.addPostFrameCallback((_) {
        print('[NOVA DEBUG] 🚀 UI FIRST FRAME RENDERED');
      });

      if (mounted) setState(() {});

      // Auto-start stream when in navigation mode (no swipe needed at launch)
      if (_currentMode == NovaMode.navigation) {
        _isNavigationActive = true;
        _manageImageStream();
      }
    } catch (e) {
      print('[NOVA DEBUG] ❌ Camera initialization failed: $e');
      _updateStatus('Camera initialization failed: $e');
      await _tts.speak('Camera initialization failed');
    }
  }

  Future<void> _checkBackendConnection() async {
    try {
      print('[NOVA DEBUG] Checking backend connection...');
      final connected = await ApiService.healthCheck();
      setState(() {
        _backendConnected = connected;
      });
      if (connected) {
        print('[NOVA DEBUG] ✓ Backend connected successfully');
        await _tts.speak('Backend connected');
      } else {
        print('[NOVA DEBUG] ❌ Backend health check failed');
        await _tts.speak('Backend connection failed. Is the server running?');
      }
    } catch (e) {
      print('[NOVA DEBUG] ❌ Backend connection exception: $e');
      setState(() => _backendConnected = false);
    }
  }

  /// Called for every frame from the image stream.
  /// Processes every 5th frame; skips if any inference is already running.
  /// Stream is NEVER stopped here — zero camera session restarts.
  void _onFrameAvailable(CameraImage cameraImage) {
    if (!_isNavigationActive)
      return; // hard lock — skip if navigation hasn't been started
    if (EmergencyService.instance.isActive)
      return; // hard lock — skip during emergency

    final now = DateTime.now();
    if (now.difference(_lastFrameProcessedAt).inMilliseconds < 900) return;

    _frameCount++;
    if (_frameCount % 12 != 0)
      return; // process fewer frames to keep preview smooth
    if (_inferenceInProgress) return; // hard lock — skip if busy

    _lastFrameProcessedAt = now;
    _inferenceInProgress = true; // claim slot immediately (sync)
    if (mounted) setState(() => _isProcessing = true);
    _runInferenceOnStreamFrame(cameraImage);
  }

  /// Encodes [cameraImage] to JPEG in-memory (no stream stop/restart),
  /// writes once to a temp file, then runs mode inference.
  Future<void> _runInferenceOnStreamFrame(CameraImage cameraImage) async {
    try {
      // Bail immediately if emergency activated mid-processing
      if (EmergencyService.instance.isActive) {
        print('[NOVA DEBUG] ⛔ Inference aborted — emergency active.');
        _inferenceInProgress = false;
        if (mounted) setState(() => _isProcessing = false);
        return;
      }

      final File imageFile = await _cameraImageToFile(cameraImage);
      print(
          '[NOVA DEBUG] === Stream frame → ${imageFile.path} (${imageFile.lengthSync()} bytes)');

      // Check again before expensive API call
      if (EmergencyService.instance.isActive) {
        print(
            '[NOVA DEBUG] ⛔ Inference aborted before API call — emergency active.');
        _inferenceInProgress = false;
        if (mounted) setState(() => _isProcessing = false);
        return;
      }

      final result = await _runModeInference(imageFile);

      if (mounted) {
        final heatBytes = result.heatmapBytes != null
            ? Uint8List.fromList(result.heatmapBytes!)
            : null;
        setState(() {
          _lastResult = result;
          _lastInferenceMode = _currentMode;
          _isProcessing = false;
          _inferenceInProgress = false;
          if (heatBytes != null) _heatmapImage = heatBytes;
          _navCommand = result.navCommand;
        });
      } else {
        _inferenceInProgress = false;
      }
      print('[NOVA DEBUG] === Pipeline complete ===');
    } catch (e) {
      print('[NOVA DEBUG] ❌ Stream inference error: $e');
      _inferenceInProgress = false;
      if (mounted) setState(() => _isProcessing = false);
    }
    // NOTE: No stream stop/restart — stream keeps running throughout
  }

  /// Converts a [CameraImage] (YUV420 on Android) to a JPEG [File]
  /// without interrupting the image stream.
  Future<File> _cameraImageToFile(CameraImage cameraImage) async {
    final Uint8List jpegBytes = _convertCameraImageToJpeg(cameraImage);
    final dir = await getTemporaryDirectory();
    final path =
        '${dir.path}/nova_frame_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final file = File(path);
    await file.writeAsBytes(jpegBytes, flush: true);
    return file;
  }

  /// Encodes a [CameraImage] to JPEG bytes using the `image` package.
  /// Handles Android YUV420 and iOS BGRA8888 formats.
  Uint8List _convertCameraImageToJpeg(CameraImage cameraImage) {
    img.Image? frame;

    final format = cameraImage.format.group;
    if (format == ImageFormatGroup.yuv420) {
      // Android path: keep full frame resolution for more reliable object detection.
      final int srcWidth = cameraImage.width;
      final int srcHeight = cameraImage.height;
      final int outWidth = srcWidth;
      final int outHeight = srcHeight;

      final Uint8List yPlane = cameraImage.planes[0].bytes;
      final int yRowStride = cameraImage.planes[0].bytesPerRow;

      if (cameraImage.planes.length >= 3) {
        final Uint8List uPlane = cameraImage.planes[1].bytes;
        final Uint8List vPlane = cameraImage.planes[2].bytes;
        final int uvRowStride = cameraImage.planes[1].bytesPerRow;
        final int uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1;

        frame = img.Image(width: outWidth, height: outHeight);
        for (int oy = 0; oy < outHeight; oy++) {
          final int sy = oy;
          for (int ox = 0; ox < outWidth; ox++) {
            final int sx = ox;

            final int yIdx = sy * yRowStride + sx;
            final int uvIdx =
                (sy ~/ 2) * uvRowStride + (sx ~/ 2) * uvPixelStride;

            if (yIdx < 0 ||
                yIdx >= yPlane.length ||
                uvIdx < 0 ||
                uvIdx >= uPlane.length ||
                uvIdx >= vPlane.length) {
              continue;
            }

            final int yVal = yPlane[yIdx];
            final int uVal = uPlane[uvIdx];
            final int vVal = vPlane[uvIdx];

            final int c = yVal - 16;
            final int d = uVal - 128;
            final int e = vVal - 128;

            final int r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255);
            final int g =
                ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255);
            final int b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255);

            frame.setPixelRgb(ox, oy, r, g, b);
          }
        }
      } else {
        // Rare fallback: grayscale if U/V planes are unavailable.
        frame = img.Image(width: outWidth, height: outHeight);
        for (int oy = 0; oy < outHeight; oy++) {
          final int sy = oy;
          for (int ox = 0; ox < outWidth; ox++) {
            final int sx = ox;
            final int yIdx = sy * yRowStride + sx;
            if (yIdx < 0 || yIdx >= yPlane.length) continue;
            final int luma = yPlane[yIdx];
            frame.setPixelRgb(ox, oy, luma, luma, luma);
          }
        }
      }
    } else if (format == ImageFormatGroup.bgra8888) {
      // iOS: BGRA8888
      frame = img.Image.fromBytes(
        width: cameraImage.width,
        height: cameraImage.height,
        bytes: cameraImage.planes[0].bytes.buffer,
        order: img.ChannelOrder.bgra,
      );
    } else {
      // Fallback: treat first plane as grayscale
      frame = img.Image.fromBytes(
        width: cameraImage.width,
        height: cameraImage.height,
        bytes: cameraImage.planes[0].bytes.buffer,
        numChannels: 1,
      );
    }

    return Uint8List.fromList(img.encodeJpg(frame, quality: 85));
  }

  /// Routes [imageFile] to the correct mode-specific inference method.
  Future<ModeInferenceResult> _runModeInference(File imageFile) async {
    switch (_currentMode) {
      case NovaMode.navigation:
        return _captureNavigation(imageFile);
      case NovaMode.reading:
        return _captureReading(imageFile);
      case NovaMode.recognition:
        return _captureRecognition(imageFile);
      case NovaMode.emergency:
        return ModeInferenceResult.success(guidance: 'Emergency Mode Active');
    }
  }

  Future<void> _announceMode() async {
    if (!_isNavigationActive) return;
    await _tts.speak(_currentMode.label);
  }

  void _updateStatus(String message) {
    setState(() => _statusMessage = message);
  }

  void _onModeChanged(NovaMode mode) {
    if (EmergencyService.instance.isActive) {
      print('[NOVA DEBUG] ⛔ Mode change blocked — emergency active.');
      return;
    }
    if (_currentMode == mode && _isNavigationActive) return;
    setState(() {
      _currentMode = mode;
      _isNavigationActive = true;
      // Clear stale detection data
      _lastResult = null;
    });
    print('[NOVA DEBUG] Mode switched via activation: ${mode.label}');
    _announceMode();
    _manageImageStream();
  }

  Future<void> _manageImageStream() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    final bool shouldStream = (_currentMode == NovaMode.navigation ||
            _currentMode == NovaMode.recognition ||
            _currentMode == NovaMode.reading) &&
        _settings.autoDetection &&
        _isNavigationActive &&
        !EmergencyService.instance.isActive;

    if (shouldStream) {
      if (!_controller!.value.isStreamingImages) {
        // Delay stream start by 300ms to avoid UI freeze during mode switch
        print('[NOVA DEBUG] Delaying stream start for 300ms...');
        await Future.delayed(const Duration(milliseconds: 300));
        if (!_isDisposed &&
            mounted &&
            _controller!.value.isInitialized &&
            !_controller!.value.isStreamingImages &&
            (_currentMode == NovaMode.navigation ||
                _currentMode == NovaMode.recognition ||
                _currentMode == NovaMode.reading) &&
            _isNavigationActive &&
            !EmergencyService.instance.isActive) {
          try {
            await _controller!.startImageStream(_onFrameAvailable);
            print(
                '[NOVA DEBUG] ✓ Image stream started for ${_currentMode.label} Mode');
          } catch (e) {
            print('[NOVA DEBUG] ❌ Failed to start image stream: $e');
          }
        }
      }
    } else {
      if (_controller!.value.isStreamingImages) {
        try {
          await _controller!.stopImageStream();
          print(
              '[NOVA DEBUG] ✓ Image stream stopped (mode: ${_currentMode.label})');
        } catch (e) {
          print('[NOVA DEBUG] ❌ Failed to stop image stream: $e');
        }
      }
    }
  }

  void _handleModeSwitch(bool isLeftSwipe) {
    if (EmergencyService.instance.isActive) {
      print('[NOVA DEBUG] ⛔ Swipe mode switch blocked — emergency active.');
      return;
    }
    setState(() {
      _isNavigationActive = true;

      // Build list of allowed modes according to feature flags
      final allowedModes = <NovaMode>[NovaMode.navigation];
      if (NovaConfig.enableOCR) allowedModes.add(NovaMode.reading);
      if (NovaConfig.enableRecognitionMode)
        allowedModes.add(NovaMode.recognition);

      // Ensure current mode exists in allowedModes; if not, reset to first
      int idx = allowedModes.indexOf(_currentMode);
      if (idx == -1) {
        _currentMode = allowedModes.first;
      } else {
        idx = isLeftSwipe
            ? (idx + 1) % allowedModes.length
            : (idx - 1 + allowedModes.length) % allowedModes.length;
        _currentMode = allowedModes[idx];
      }

      // Clear stale detection data when mode changes
      if (_lastInferenceMode != _currentMode) {
        _lastResult = null;
        print('[NOVA DEBUG] Cleared stale detection data from previous mode');
      }
    });
    print('[NOVA DEBUG] Mode switched to: ${_currentMode.label}');
    _announceMode();
    _manageImageStream();
  }

  /// Manual capture button — takes a snapshot without touching the stream.
  /// Stream keeps running; snapshot is grabbed via the same JPEG conversion path.
  Future<void> _captureAndProcess() async {
    if (EmergencyService.instance.isActive) {
      print('[NOVA DEBUG] ⛔ Manual capture blocked — emergency active.');
      return;
    }
    if (_inferenceInProgress || _isProcessing) {
      print('[NOVA DEBUG] ⏭️  Manual capture skipped: inference in progress');
      return;
    }
    _inferenceInProgress = true;
    if (mounted) setState(() => _isProcessing = true);
    print('[NOVA DEBUG] === Starting manual inference pipeline ===');

    try {
      // Stop stream only for the snapshot, restart before sending to backend
      // so the camera preview stays live during network latency.
      final bool wasStreaming = _controller!.value.isStreamingImages;
      if (wasStreaming) await _controller!.stopImageStream();
      final xFile = await _controller!.takePicture();
      final imageFile = File(xFile.path);
      if (wasStreaming && mounted && _controller!.value.isInitialized) {
        await _controller!.startImageStream(_onFrameAvailable);
      }
      print(
          '[NOVA DEBUG] Manual capture: ${imageFile.path} (${imageFile.lengthSync()} bytes)');

      final result = await _runModeInference(imageFile);

      if (mounted) {
        setState(() {
          _lastResult = result;
          _lastInferenceMode = _currentMode;
          _isProcessing = false;
          _inferenceInProgress = false;
        });
      } else {
        _inferenceInProgress = false;
      }
      print('[NOVA DEBUG] === Manual pipeline complete ===');
    } catch (e) {
      print('[NOVA DEBUG] ❌ Exception in manual capture: $e');
      _updateStatus('Capture error: $e');
      _inferenceInProgress = false;
      if (mounted) setState(() => _isProcessing = false);
      // Ensure stream is restored on error
      if (mounted &&
          _settings.autoDetection &&
          _controller != null &&
          _controller!.value.isInitialized &&
          !_controller!.value.isStreamingImages) {
        await _controller!.startImageStream(_onFrameAvailable);
      }
    }
  }

  /// Navigation Mode: Obstacle detection + depth estimation + guidance
  Future<ModeInferenceResult> _captureNavigation(File imageFile) async {
    try {
      print(
          '[NOVA DEBUG] Mode: NAVIGATION | Calling getNavigationGuidance API...');
      final startTime = DateTime.now();

      final result =
          await _inferenceService.processFrame(imageFile, NovaMode.navigation);

      final elapsed = DateTime.now().difference(startTime).inMilliseconds;

      if (!result.success) {
        print('[NOVA DEBUG] ⚠️  Navigation API returned success=false');
        print('[NOVA DEBUG] Error: ${result.error}');
      } else {
        print('[NOVA DEBUG] ✓ Navigation inference successful (${elapsed}ms)');
        print('[NOVA DEBUG] - Guidance: "${result.guidance}"');
        print('[NOVA DEBUG] - Obstacles detected: ${result.detections.length}');

        // Validate parsed data
        if (result.detections.isEmpty) {
          print('[NOVA DEBUG] ℹ️  No obstacles detected in scene');
        } else {
          for (int i = 0; i < result.detections.length; i++) {
            final det = result.detections[i];
            print(
                '[NOVA DEBUG] Detection $i: ${det.label} (${(det.confidence * 100).toStringAsFixed(1)}%) - bbox(l:${det.bbox.left.toStringAsFixed(0)}, t:${det.bbox.top.toStringAsFixed(0)}, r:${det.bbox.right.toStringAsFixed(0)}, b:${det.bbox.bottom.toStringAsFixed(0)})');
          }
        }
      }
      return result;
    } catch (e) {
      print('[NOVA DEBUG] ❌ Navigation mode exception: $e');
      return ModeInferenceResult.error('Navigation error: $e');
    }
  }

  /// Reading Mode: OCR text extraction
  Future<ModeInferenceResult> _captureReading(File imageFile) async {
    if (!NovaConfig.enableOCR) {
      print(
          '[NOVA DEBUG] READING mode is disabled via configuration — skipping inference.');
      return ModeInferenceResult.error('Reading mode disabled');
    }

    try {
      print('[NOVA DEBUG] Mode: READING | Calling recognizeText API...');
      final startTime = DateTime.now();

      final result =
          await _inferenceService.processFrame(imageFile, NovaMode.reading);

      final elapsed = DateTime.now().difference(startTime).inMilliseconds;

      if (!result.success) {
        print('[NOVA DEBUG] ⚠️  Reading API returned success=false');
        print('[NOVA DEBUG] Error: ${result.error}');
      } else {
        print('[NOVA DEBUG] ✓ OCR inference successful (${elapsed}ms)');
        print(
            '[NOVA DEBUG] - Text regions extracted: ${result.textRegions.length}');

        // Validate parsed data
        if (result.textRegions.isEmpty) {
          print('[NOVA DEBUG] ℹ️  No text regions detected');
        } else {
          for (int i = 0; i < result.textRegions.length; i++) {
            final text = result.textRegions[i];
            print(
                '[NOVA DEBUG] Text $i: "${text.text}" (conf:${(text.confidence * 100).toStringAsFixed(1)}%) - bbox(l:${text.bbox.left.toStringAsFixed(0)}, t:${text.bbox.top.toStringAsFixed(0)}, r:${text.bbox.right.toStringAsFixed(0)}, b:${text.bbox.bottom.toStringAsFixed(0)})');
          }
        }
      }
      return result;
    } catch (e) {
      print('[NOVA DEBUG] ❌ Reading mode exception: $e');
      return ModeInferenceResult.error('Reading error: $e');
    }
  }

  /// Recognition Mode: Face + Object detection
  Future<ModeInferenceResult> _captureRecognition(File imageFile) async {
    if (!NovaConfig.enableRecognitionMode) {
      print(
          '[NOVA DEBUG] RECOGNITION mode is disabled via configuration — skipping inference.');
      return ModeInferenceResult.error('Recognition mode disabled');
    }

    try {
      print(
          '[NOVA DEBUG] Mode: RECOGNITION | Calling detectFaces + detectObjects APIs...');
      final startTime = DateTime.now();

      final result =
          await _inferenceService.processFrame(imageFile, NovaMode.recognition);

      final elapsed = DateTime.now().difference(startTime).inMilliseconds;

      if (!result.success) {
        print('[NOVA DEBUG] ⚠️  Recognition API returned success=false');
        print('[NOVA DEBUG] Error: ${result.error}');
      } else {
        print('[NOVA DEBUG] ✓ Recognition inference successful (${elapsed}ms)');
        print('[NOVA DEBUG] - Faces detected: ${result.faces.length}');
        print('[NOVA DEBUG] - Objects detected: ${result.detections.length}');
        print('[NOVA DEBUG] - Inference time: ${result.inferenceTimeMs}ms');

        // Validate parsed data
        if (result.faces.isEmpty && result.detections.isEmpty) {
          print('[NOVA DEBUG] ℹ️  No faces or objects detected');
        } else {
          if (result.faces.isNotEmpty) {
            for (int i = 0; i < result.faces.length; i++) {
              final face = result.faces[i];
              print(
                  '[NOVA DEBUG] Face $i: ID=${face.personId ?? "unknown"} (conf:${(face.confidence * 100).toStringAsFixed(1)}%) - bbox(l:${face.bbox.left.toStringAsFixed(0)}, t:${face.bbox.top.toStringAsFixed(0)}, r:${face.bbox.right.toStringAsFixed(0)}, b:${face.bbox.bottom.toStringAsFixed(0)})');
            }
          }
          if (result.detections.isNotEmpty) {
            for (int i = 0; i < result.detections.length; i++) {
              final det = result.detections[i];
              print(
                  '[NOVA DEBUG] Object $i: ${det.label} (conf:${(det.confidence * 100).toStringAsFixed(1)}%) - bbox(l:${det.bbox.left.toStringAsFixed(0)}, t:${det.bbox.top.toStringAsFixed(0)}, r:${det.bbox.right.toStringAsFixed(0)}, b:${det.bbox.bottom.toStringAsFixed(0)})');
            }
          }
        }
      }
      return result;
    } catch (e) {
      print('[NOVA DEBUG] ❌ Recognition mode exception: $e');
      return ModeInferenceResult.error('Recognition error: $e');
    }
  }

  /// Build detection overlay based on current mode and results
  Widget _buildDetectionOverlay() {
    if (_lastResult == null || !_lastResult!.success) {
      return const SizedBox.expand();
    }

    return CustomPaint(
      painter: DetectionOverlayPainter(
        detections: _lastResult!.detections,
        textRegions: _lastResult!.textRegions,
        faces: _lastResult!.faces,
        imageSize: const Size(1280, 720),
        canvasSize: MediaQuery.of(context).size,
      ),
      child: const SizedBox.expand(),
    );
  }

  /// Build the depth heatmap overlay (blended JPEG from backend)
  Widget _buildHeatmapOverlay() {
    if (_heatmapImage == null || _currentMode != NovaMode.navigation) {
      return const SizedBox.shrink();
    }
    return Positioned.fill(
      child: Opacity(
        opacity: 0.55,
        child: Image.memory(
          _heatmapImage!,
          fit: BoxFit.cover,
          gaplessPlayback: true, // prevents flicker between frames
        ),
      ),
    );
  }

  /// Color-coded navigation command banner
  Color _commandBannerColor() {
    switch (_navCommand) {
      case 'STOP':
        return Colors.red.shade700;
      case 'CAUTION':
        return Colors.orange.shade700;
      case 'MOVE_LEFT':
        return Colors.deepOrange;
      case 'MOVE_RIGHT':
        return Colors.deepOrange;
      default:
        return Colors.green.shade700;
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    WidgetsBinding.instance.removeObserver(this);
    // Unregister emergency callbacks to avoid leaks
    EmergencyService.instance.onNavigationPause = null;
    EmergencyService.instance.onNavigationResume = null;
    _activationSub?.cancel();
    ActivationService.instance.stopVoiceListener();
    final controller = _controller;
    _controller = null;
    if (controller != null) {
      unawaited(_releaseCameraController(controller));
    }
    super.dispose();
  }

  Future<void> _releaseCameraController(CameraController controller) async {
    try {
      if (controller.value.isInitialized &&
          controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (e) {
      print('[NOVA DEBUG] stopImageStream during release failed: $e');
    }
    try {
      await controller.dispose();
    } catch (e) {
      print('[NOVA DEBUG] controller dispose failed: $e');
    }
  }

  // ─── Emergency ↔ Navigation lifecycle ──────────────────────────────────────

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    print('[NOVA DEBUG] AppLifecycleState changed: $state');
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      final controller = _controller;
      _controller = null;
      _isNavigationActive = false;
      _inferenceInProgress = false;
      _isProcessing = false;
      if (controller != null) {
        unawaited(_releaseCameraController(controller));
      }
      return;
    }

    if (state == AppLifecycleState.resumed) {
      // App came back from phone call or other external activity
      if (!EmergencyService.instance.isActive) {
        // Emergency is over (call completed or was cancelled).
        // Re-initialize camera if needed.
        print('[NOVA DEBUG] Resumed — reinitializing camera.');
        _reinitializeCameraAfterEmergency();
      } else {
        print(
            '[NOVA DEBUG] Resumed but emergency still active — staying paused.');
      }
    }
  }

  /// Called by EmergencyService.onNavigationPause callback.
  void _pauseNavigationForEmergency() {
    print('[NOVA DEBUG] ⏸️  Pausing navigation for emergency.');
    _isNavigationActive = false;
    _inferenceInProgress = false;
    _isProcessing = false;
    // Stop image stream safely
    if (_controller != null &&
        _controller!.value.isInitialized &&
        _controller!.value.isStreamingImages) {
      unawaited(() async {
        try {
          await _controller!.stopImageStream();
          print('[NOVA DEBUG] ✓ Image stream stopped for emergency.');
        } catch (e) {
          print('[NOVA DEBUG] ❌ Error stopping stream for emergency: $e');
        }
      }());
    }
    if (mounted) setState(() {});
  }

  /// Called by EmergencyService.onNavigationResume callback.
  void _resumeNavigationAfterEmergency() {
    print('[NOVA DEBUG] ▶️  Resuming navigation after emergency cancelled.');
    _isNavigationActive = true;
    _manageImageStream();
    if (mounted) setState(() {});
  }

  /// Re-initializes the camera after returning from an external call.
  Future<void> _reinitializeCameraAfterEmergency() async {
    try {
      // Dispose old controller if it exists
      if (_controller != null) {
        if (_controller!.value.isStreamingImages) {
          await _controller!.stopImageStream();
        }
        await _controller!.dispose();
        _controller = null;
      }
      // Re-init camera fresh
      await _initializeCamera();
    } catch (e) {
      print('[NOVA DEBUG] ❌ Failed to reinitialize camera after emergency: $e');
    }
  }

  Future<void> _pauseCameraSessionForNavigation() async {
    final controller = _controller;
    if (controller == null) return;
    try {
      if (controller.value.isInitialized && controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (e) {
      print('[NOVA DEBUG] pause camera stopImageStream failed: $e');
    }
    try {
      await controller.dispose();
    } catch (e) {
      print('[NOVA DEBUG] pause camera dispose failed: $e');
    }
    _controller = null;
  }

  Future<void> _resumeCameraSessionAfterSettings() async {
    try {
      _inferenceInProgress = false;
      _isProcessing = false;
      if (_controller == null || !_controller!.value.isInitialized) {
        await _initializeCamera();
      }
      await _manageImageStream();
      if (mounted) setState(() {});
    } catch (e) {
      print('[NOVA DEBUG] resume camera after settings failed: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(color: Colors.yellow),
              const SizedBox(height: 20),
              Text(
                _statusMessage,
                style: const TextStyle(
                  color: Colors.yellow,
                  fontSize: 16,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _initialize,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        onHorizontalDragEnd: (details) {
          if (details.primaryVelocity == null) return;
          _handleModeSwitch(details.primaryVelocity! < 0);
        },
        child: Stack(
          children: [
            // Camera preview — fills full screen (cover fit)
            SizedBox.expand(
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _controller!.value.previewSize?.height ?? 1,
                  height: _controller!.value.previewSize?.width ?? 1,
                  child: CameraPreview(_controller!),
                ),
              ),
            ),

            // Depth heatmap overlay (navigation mode only)
            _buildHeatmapOverlay(),

            // Detection overlays (bounding boxes)
            _buildDetectionOverlay(),

            // Settings icon (top-left)
            Positioned(
              top: 40,
              left: 20,
              child: GestureDetector(
                onTap: () async {
                  await _pauseCameraSessionForNavigation();
                  try {
                    await Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const SettingsScreen()),
                    );
                  } finally {
                    // Reload settings and re-acquire camera after returning
                    await _loadSettings();
                    await _initializeTTS();
                    await _resumeCameraSessionAfterSettings();
                  }
                },
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.7),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(
                    Icons.settings,
                    color: Colors.yellow,
                    size: 28,
                  ),
                ),
              ),
            ),

            // Mode indicator (top-center)
            Positioned(
              top: 40,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.all(12),
                  color: Colors.black.withOpacity(0.7),
                  child: Text(
                    _currentMode.label.toUpperCase(),
                    style: const TextStyle(
                      color: Colors.yellow,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),

            // Status indicator (top-right)
            Positioned(
              top: 40,
              right: 20,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                color: _backendConnected ? Colors.green : Colors.red,
                child: Text(
                  _backendConnected ? 'ONLINE' : 'OFFLINE',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),

            // Processing indicator
            if (_isProcessing)
              Positioned(
                bottom: 100,
                left: 0,
                right: 0,
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    color: Colors.yellow.withOpacity(0.8),
                    child: const Text(
                      'PROCESSING...',
                      style: TextStyle(
                        color: Colors.black,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),

            // Navigation guidance banner
            if (_currentMode == NovaMode.navigation &&
                _lastResult != null &&
                _lastResult!.success &&
                _lastResult!.guidance.isNotEmpty)
              Positioned(
                bottom: 110,
                left: 16,
                right: 16,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                  decoration: BoxDecoration(
                    color: _commandBannerColor().withOpacity(0.85),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    _lastResult!.guidance,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
              ),

            // Bottom control bar — centered emergency only
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: SafeArea(
                top: false,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 20, top: 12),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Center(
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onLongPress: () {
                            print(
                                '[NOVA DEBUG] 🚨 Emergency button LONG-PRESSED. isActive=${EmergencyService.instance.isActive}');
                            Future.microtask(() => EmergencyService.instance
                                .triggerEmergency(_tts));
                          },
                          child: Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.red.withOpacity(0.7),
                              borderRadius: BorderRadius.circular(50),
                            ),
                            child: const Icon(
                              Icons.emergency,
                              color: Colors.white,
                              size: 30,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
