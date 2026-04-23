import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

class ObjectData {
  final String id;
  final String name;
  final String targetLabel;
  final bool isPriority;
  final int samplesUsed;

  ObjectData({
    required this.id,
    required this.name,
    required this.targetLabel,
    required this.isPriority,
    required this.samplesUsed,
  });

  factory ObjectData.fromJson(Map<String, dynamic> json) => ObjectData(
        id: (json['id'] ?? '').toString(),
        name: (json['name'] ?? '').toString(),
        targetLabel: (json['target_label'] ?? '').toString(),
        isPriority: (json['is_priority'] ?? false) == true,
        samplesUsed: (json['samples_used'] as num?)?.toInt() ?? 0,
      );
}

class ObjectRegistrationScreen extends StatefulWidget {
  const ObjectRegistrationScreen({super.key});

  @override
  State<ObjectRegistrationScreen> createState() =>
      _ObjectRegistrationScreenState();
}

class _ObjectRegistrationScreenState extends State<ObjectRegistrationScreen> {
  List<ObjectData> _objects = [];
  List<String> _validLabels = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _loading = true);
    await Future.wait([
      _loadObjects(),
      _loadValidLabels(),
    ]);
    if (!mounted) return;
    setState(() => _loading = false);
  }

  Future<void> _loadValidLabels() async {
    try {
      final labels = await ApiService.listObjectLabels();
      if (!mounted) return;
      setState(() => _validLabels = labels);
    } catch (_) {
      // Keep UI usable even if labels endpoint fails.
    }
  }

  Future<void> _loadObjects() async {
    try {
      final objects = await ApiService.listRegisteredObjects();
      if (!mounted) return;
      setState(() {
        _objects = objects.map((e) => ObjectData.fromJson(e)).toList();
      });
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Failed to load objects from backend'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _startNewRegistration() async {
    final result = await _showObjectNameDialog();
    if (result == null) return;

    if (!mounted) return;
    final success = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (_) => ObjectRegistrationCaptureScreen(
          objectName: result['name']!,
          targetLabel: result['label']!,
          isPriority: result['priority']!,
        ),
      ),
    );

    if (success == true) {
      await _loadObjects();
    }
  }

  Future<Map<String, dynamic>?> _showObjectNameDialog() async {
    final nameController = TextEditingController();
    final labelController = TextEditingController();
    bool isPriority = false;

    return showDialog<Map<String, dynamic>>(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          backgroundColor: Colors.grey[900],
          title: const Text(
            'Register Object',
            style: TextStyle(color: Colors.yellow),
          ),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    labelText: 'Custom Object Name',
                    labelStyle: TextStyle(color: Colors.grey),
                    hintText: 'e.g., My Keys, Wallet, Medicine Box',
                    hintStyle: TextStyle(color: Colors.grey),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.grey),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.yellow),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Autocomplete<String>(
                  optionsBuilder: (TextEditingValue textEditingValue) {
                    final query = textEditingValue.text.trim().toLowerCase();
                    final source = _validLabels;
                    if (query.isEmpty) {
                      return source.take(12);
                    }
                    return source
                        .where((label) => label.toLowerCase().contains(query))
                        .take(12);
                  },
                  onSelected: (String selection) {
                    labelController.text = selection;
                  },
                  fieldViewBuilder: (
                    context,
                    textEditingController,
                    focusNode,
                    onFieldSubmitted,
                  ) {
                    textEditingController.text = labelController.text;
                    return TextField(
                      controller: textEditingController,
                      focusNode: focusNode,
                      style: const TextStyle(color: Colors.white),
                      onChanged: (value) => labelController.text = value,
                      decoration: InputDecoration(
                        labelText: 'Target YOLO Label',
                        labelStyle: const TextStyle(color: Colors.grey),
                        hintText: _validLabels.isEmpty
                            ? 'Loading labels...'
                            : 'e.g., bottle, cup, cell phone, book',
                        hintStyle: const TextStyle(color: Colors.grey),
                        enabledBorder: const OutlineInputBorder(
                          borderSide: BorderSide(color: Colors.grey),
                        ),
                        focusedBorder: const OutlineInputBorder(
                          borderSide: BorderSide(color: Colors.yellow),
                        ),
                        suffixIcon: _validLabels.isEmpty
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: Padding(
                                  padding: EdgeInsets.all(12),
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.yellow,
                                  ),
                                ),
                              )
                            : const Icon(Icons.arrow_drop_down,
                                color: Colors.grey),
                      ),
                    );
                  },
                  optionsViewBuilder: (context, onSelected, options) {
                    return Align(
                      alignment: Alignment.topLeft,
                      child: Material(
                        color: Colors.grey[900],
                        elevation: 4,
                        child: SizedBox(
                          width: MediaQuery.of(context).size.width * 0.72,
                          child: ListView.builder(
                            padding: EdgeInsets.zero,
                            shrinkWrap: true,
                            itemCount: options.length,
                            itemBuilder: (context, index) {
                              final option = options.elementAt(index);
                              return ListTile(
                                dense: true,
                                title: Text(
                                  option,
                                  style: const TextStyle(color: Colors.white),
                                ),
                                onTap: () => onSelected(option),
                              );
                            },
                          ),
                        ),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 12),
                CheckboxListTile(
                  title: const Text(
                    'Priority Alert',
                    style: TextStyle(color: Colors.white),
                  ),
                  subtitle: const Text(
                    'Get immediate alerts when detected',
                    style: TextStyle(color: Colors.grey, fontSize: 12),
                  ),
                  value: isPriority,
                  activeColor: Colors.yellow,
                  onChanged: (value) {
                    setDialogState(() => isPriority = value ?? false);
                  },
                  contentPadding: EdgeInsets.zero,
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
            ),
            TextButton(
              onPressed: () {
                final name = nameController.text.trim();
                final label = labelController.text.trim().toLowerCase();
                if (name.isEmpty || label.isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Please enter both name and target label'),
                      backgroundColor: Colors.red,
                    ),
                  );
                  return;
                }
                if (_validLabels.isNotEmpty && !_validLabels.contains(label)) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Please choose a valid YOLO label'),
                      backgroundColor: Colors.red,
                    ),
                  );
                  return;
                }
                Navigator.pop(context, {
                  'name': name,
                  'label': label,
                  'priority': isPriority,
                });
              },
              child: const Text('Start Capture', style: TextStyle(color: Colors.yellow)),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _deleteObject(ObjectData object) async {
    try {
      await ApiService.deleteRegisteredObject(object.id);
      await _loadObjects();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Delete failed: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text(
          'Object Registration',
          style: TextStyle(color: Colors.yellow),
        ),
        iconTheme: const IconThemeData(color: Colors.yellow),
        actions: [
          IconButton(
            onPressed: _loadData,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Colors.yellow))
          : _objects.isEmpty
              ? const Center(
                  child: Text(
                    'No objects registered.\nTap + to register an object.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey, fontSize: 16),
                  ),
                )
              : GridView.builder(
                  padding: const EdgeInsets.all(16),
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    crossAxisSpacing: 16,
                    mainAxisSpacing: 16,
                    childAspectRatio: 0.78,
                  ),
                  itemCount: _objects.length,
                  itemBuilder: (context, index) {
                    final object = _objects[index];
                    return Card(
                      color: Colors.grey[900],
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Expanded(
                            child: Stack(
                              children: [
                                Container(
                                  color: Colors.black,
                                  width: double.infinity,
                                  alignment: Alignment.center,
                                  child: const Icon(
                                    Icons.category,
                                    color: Colors.yellow,
                                    size: 48,
                                  ),
                                ),
                                if (object.isPriority)
                                  const Positioned(
                                    top: 8,
                                    right: 8,
                                    child: Icon(
                                      Icons.priority_high,
                                      color: Colors.red,
                                      size: 24,
                                    ),
                                  ),
                              ],
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.all(8),
                            child: Column(
                              children: [
                                Text(
                                  object.name,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                                Text(
                                  object.targetLabel,
                                  style: const TextStyle(
                                    color: Colors.grey,
                                    fontSize: 12,
                                  ),
                                ),
                                if (object.samplesUsed > 0)
                                  Text(
                                    '${object.samplesUsed} samples',
                                    style: const TextStyle(
                                      color: Colors.yellow,
                                      fontSize: 11,
                                    ),
                                  ),
                                if (object.isPriority)
                                  const Text(
                                    'Priority',
                                    style: TextStyle(
                                      color: Colors.red,
                                      fontSize: 10,
                                    ),
                                  ),
                                const SizedBox(height: 4),
                                IconButton(
                                  icon: const Icon(Icons.delete,
                                      color: Colors.red),
                                  onPressed: () => _deleteObject(object),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
      floatingActionButton: FloatingActionButton(
        onPressed: _startNewRegistration,
        backgroundColor: Colors.yellow,
        child: const Icon(Icons.camera_alt, color: Colors.black),
      ),
    );
  }
}

class ObjectRegistrationCaptureScreen extends StatefulWidget {
  final String objectName;
  final String targetLabel;
  final bool isPriority;

  const ObjectRegistrationCaptureScreen({
    super.key,
    required this.objectName,
    required this.targetLabel,
    required this.isPriority,
  });

  @override
  State<ObjectRegistrationCaptureScreen> createState() =>
      _ObjectRegistrationCaptureScreenState();
}

class _ObjectRegistrationCaptureScreenState
    extends State<ObjectRegistrationCaptureScreen> {
  static const int _targetFrames = 20;
  static const int _minAccepted = 5;
  static const int _captureIntervalMs = 800;

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

      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
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
      _status = 'Capturing... Keep the object in frame.';
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
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }
    if (_cameraController!.value.isTakingPicture) return;

    try {
      final XFile xfile = await _cameraController!.takePicture();
      final File imageFile = File(xfile.path);
      _framesSent++;

      final result = await ApiService.registerObjectFrame(
        name: widget.objectName,
        targetLabel: widget.targetLabel,
        imageFile: imageFile,
      );

      if (result['accepted'] == true) {
        _framesAccepted++;
        if (mounted) {
          setState(() {
            _status = 'Captured $_framesAccepted / $_targetFrames valid frames...';
          });
        }
      } else {
        final reason = result['message'] as String? ?? 'Frame rejected';
        if (mounted) {
          setState(() {
            _status = reason;
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Network error — retrying...');
      }
    }
  }

  Future<void> _finishCapture() async {
    if (_framesAccepted < _minAccepted) {
      try {
        await ApiService.cancelObjectRegistration(widget.objectName);
      } catch (_) {}
      if (mounted) {
        setState(() {
          _isCapturing = false;
          _status =
              'Only $_framesAccepted clear frames captured (need $_minAccepted).\nRegistration failed. Please try again with the object centered and well lit.';
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
      final result = await ApiService.saveObjectRegistration(
        name: widget.objectName,
        targetLabel: widget.targetLabel,
        isPriority: widget.isPriority,
      );
      if (mounted) {
        setState(() {
          _isSaving = false;
          _isDone = true;
          _status = result['message'] as String? ??
              '${widget.objectName} registered successfully!';
        });
        Future.delayed(const Duration(milliseconds: 2000), () {
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
        await ApiService.cancelObjectRegistration(widget.objectName);
      } catch (_) {}
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
        title: Text('Register: ${widget.objectName}'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.yellow,
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: _cancel,
        ),
      ),
      body: Column(
        children: [
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
