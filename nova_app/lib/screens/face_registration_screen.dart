import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

/// Camera-based face registration screen.
/// Opened from SettingsScreen with the person's name already provided.
/// Automatically captures [_targetFrames] frames at [_captureIntervalMs] ms intervals.
class FaceRegistrationScreen extends StatefulWidget {
  final String personName;

  const FaceRegistrationScreen({super.key, required this.personName});

  @override
  State<FaceRegistrationScreen> createState() => _FaceRegistrationScreenState();
}

class _FaceRegistrationScreenState extends State<FaceRegistrationScreen> {
  static const int _targetFrames = 20;
  static const int _minAccepted = 5;
  static const int _captureIntervalMs =
      800; // increased from 400ms to reduce backend load

  CameraController? _cameraController;
  bool _cameraReady = false;
  bool _isCapturing = false;
  bool _isSaving = false;
  bool _isDone = false;

  int _framesSent = 0;
  int _framesAccepted = 0;
  String _status = 'Initializing camera...';

  Timer? _captureTimer;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _status = 'No cameras found.');
        return;
      }

      // Auto-select front camera
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();

      if (mounted) {
        setState(() {
          _cameraReady = true;
          _status = 'Starting capture...';
        });

        // Auto-start capture immediately
        _startCapture();
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Camera error: $e');
      }
    }
  }

  void _startCapture() {
    if (!_cameraReady || _isCapturing) return;
    setState(() {
      _isCapturing = true;
      _framesSent = 0;
      _framesAccepted = 0;
      _status = 'Capturing... Please look at the camera.';
    });

    _captureTimer = Timer.periodic(
      const Duration(milliseconds: _captureIntervalMs),
      (timer) async {
        if (_framesSent >= _targetFrames) {
          timer.cancel();
          await _finishCapture();
          return;
        }
        await _captureOneFrame();
      },
    );
  }

  Future<void> _captureOneFrame() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized)
      return;

    // Throttle to prevent multiple active takePicture calls
    if (_cameraController!.value.isTakingPicture) return;

    try {
      final XFile xfile = await _cameraController!.takePicture();
      final File imageFile = File(xfile.path);
      _framesSent++;

      final result = await ApiService.registerFaceFrame(
        name: widget.personName,
        imageFile: imageFile,
      );

      debugPrint('[FACE REG] Frame API response: $result');

      if (result['accepted'] == true) {
        _framesAccepted++;
        if (mounted) {
          setState(() {
            _status =
                'Captured $_framesAccepted / $_targetFrames valid frames...';
          });
        }
      } else {
        // Show rejection reason so the user can adjust
        final reason = result['message'] as String? ?? 'Frame rejected';
        debugPrint('[FACE REG] Frame rejected: $reason');
        if (mounted) {
          setState(() {
            _status = '$reason';
          });
        }
      }
    } catch (e) {
      debugPrint('[FACE REG] Error capturing frame: $e');
      if (mounted) {
        setState(() => _status = 'Network error — retrying...');
      }
    }
  }

  Future<void> _finishCapture() async {
    if (_framesAccepted < _minAccepted) {
      try {
        await ApiService.cancelRegistration(widget.personName);
      } catch (_) {
        // Best effort cleanup; ignore cancel failures.
      }
      if (mounted) {
        setState(() {
          _isCapturing = false;
          _status =
              'Only $_framesAccepted clear frames captured (need $_minAccepted).\n'
              'Registration failed. Please close and try again in better lighting.';
        });
      }
      return;
    }

    setState(() {
      _isCapturing = false;
      _isSaving = true;
      _status = 'Saving registration...';
    });

    try {
      final result = await ApiService.saveRegistration(widget.personName);
      if (mounted) {
        setState(() {
          _isSaving = false;
          _isDone = true;
          _status = result['message'] as String? ??
              '${widget.personName} registered successfully!';
        });

        // Auto-close after 2.5s on success
        Future.delayed(const Duration(milliseconds: 2500), () {
          if (mounted) Navigator.of(context).pop(true);
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isSaving = false;
          _status = 'Failed to save: $e';
        });
      }
    }
  }

  Future<void> _cancel() async {
    _captureTimer?.cancel();
    if (_framesSent > 0 && !_isSaving && !_isDone) {
      try {
        await ApiService.cancelRegistration(widget.personName);
      } catch (_) {
        // Best effort cleanup; ignore cancel failures.
      }
    }
    if (mounted) Navigator.of(context).pop(false);
  }

  @override
  void dispose() {
    _captureTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('Register: ${widget.personName}'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.yellow,
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: _cancel,
        ),
      ),
      body: Column(
        children: [
          // Camera preview
          Expanded(
            child: _cameraReady
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: CameraPreview(_cameraController!),
                      ),
                    ),
                  )
                : const Center(
                    child: CircularProgressIndicator(color: Colors.yellow),
                  ),
          ),

          // Progress bar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 24),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: _targetFrames > 0
                    ? (_framesAccepted / _targetFrames).clamp(0.0, 1.0)
                    : 0,
                backgroundColor: Colors.grey[900],
                color: _isDone
                    ? Colors.green
                    : (_framesAccepted >= _minAccepted
                        ? Colors.yellow
                        : Colors.orange),
                minHeight: 12,
              ),
            ),
          ),

          // Status text
          Padding(
            padding: const EdgeInsets.only(left: 24, right: 24, bottom: 48),
            child: Text(
              _status,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: _isDone ? Colors.green : Colors.yellow,
                fontSize: 16,
                fontWeight: _isDone ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
