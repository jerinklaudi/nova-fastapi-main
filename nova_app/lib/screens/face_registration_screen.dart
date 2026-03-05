import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class FaceData {
  final String id;
  final String name;
  final String imagePath;

  FaceData({
    required this.id,
    required this.name,
    required this.imagePath,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'imagePath': imagePath,
      };

  factory FaceData.fromJson(Map<String, dynamic> json) => FaceData(
        id: json['id'],
        name: json['name'],
        imagePath: json['imagePath'],
      );
}

class FaceRegistrationScreen extends StatefulWidget {
  const FaceRegistrationScreen({super.key});

  @override
  State<FaceRegistrationScreen> createState() => _FaceRegistrationScreenState();
}

class _FaceRegistrationScreenState extends State<FaceRegistrationScreen> {
  List<FaceData> _faces = [];
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadFaces();
  }

  Future<void> _loadFaces() async {
    final prefs = await SharedPreferences.getInstance();
    final facesJson = prefs.getString('registered_faces');
    if (facesJson != null) {
      final List<dynamic> decoded = jsonDecode(facesJson);
      setState(() {
        _faces = decoded.map((e) => FaceData.fromJson(e)).toList();
      });
    }
  }

  Future<void> _saveFaces() async {
    final prefs = await SharedPreferences.getInstance();
    final facesJson = jsonEncode(_faces.map((e) => e.toJson()).toList());
    await prefs.setString('registered_faces', facesJson);
  }

  Future<void> _captureFace() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
      preferredCameraDevice: CameraDevice.front,
    );

    if (image != null && mounted) {
      _showNameDialog(image.path);
    }
  }

  void _showNameDialog(String imagePath) {
    final nameController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey[900],
        title: const Text(
          'Register Face',
          style: TextStyle(color: Colors.yellow),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.file(
                File(imagePath),
                height: 200,
                width: 200,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: nameController,
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                labelText: 'Name',
                labelStyle: TextStyle(color: Colors.grey),
                hintText: 'e.g., Mom, Dad, Friend',
                hintStyle: TextStyle(color: Colors.grey),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.grey),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.yellow),
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () {
              if (nameController.text.isNotEmpty) {
                setState(() {
                  _faces.add(FaceData(
                    id: DateTime.now().toString(),
                    name: nameController.text,
                    imagePath: imagePath,
                  ));
                });
                _saveFaces();
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('${nameController.text} registered!'),
                    backgroundColor: Colors.green,
                  ),
                );
              }
            },
            child: const Text('Save', style: TextStyle(color: Colors.yellow)),
          ),
        ],
      ),
    );
  }

  void _deleteFace(FaceData face) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey[900],
        title: const Text('Delete Face', style: TextStyle(color: Colors.yellow)),
        content: Text(
          'Delete ${face.name}?',
          style: const TextStyle(color: Colors.white),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () {
              setState(() => _faces.remove(face));
              _saveFaces();
              // Delete image file
              try {
                File(face.imagePath).deleteSync();
              } catch (e) {
                print('Error deleting image: $e');
              }
              Navigator.pop(context);
            },
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text(
          'Face Registration',
          style: TextStyle(color: Colors.yellow),
        ),
        iconTheme: const IconThemeData(color: Colors.yellow),
      ),
      body: _faces.isEmpty
          ? const Center(
              child: Text(
                'No faces registered.\nTap + to register a face.',
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
                childAspectRatio: 0.75,
              ),
              itemCount: _faces.length,
              itemBuilder: (context, index) {
                final face = _faces[index];
                return Card(
                  color: Colors.grey[900],
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Expanded(
                        child: ClipRRect(
                          borderRadius: const BorderRadius.vertical(
                            top: Radius.circular(4),
                          ),
                          child: Image.file(
                            File(face.imagePath),
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(8),
                        child: Column(
                          children: [
                            Text(
                              face.name,
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 4),
                            IconButton(
                              icon: const Icon(Icons.delete, color: Colors.red),
                              onPressed: () => _deleteFace(face),
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
        onPressed: _captureFace,
        backgroundColor: Colors.yellow,
        child: const Icon(Icons.camera_alt, color: Colors.black),
      ),
    );
  }
}
