import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class ObjectData {
  final String id;
  final String name;
  final String imagePath;
  final bool isPriority;

  ObjectData({
    required this.id,
    required this.name,
    required this.imagePath,
    this.isPriority = false,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'imagePath': imagePath,
        'isPriority': isPriority,
      };

  factory ObjectData.fromJson(Map<String, dynamic> json) => ObjectData(
        id: json['id'],
        name: json['name'],
        imagePath: json['imagePath'],
        isPriority: json['isPriority'] ?? false,
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
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadObjects();
  }

  Future<void> _loadObjects() async {
    final prefs = await SharedPreferences.getInstance();
    final objectsJson = prefs.getString('registered_objects');
    if (objectsJson != null) {
      final List<dynamic> decoded = jsonDecode(objectsJson);
      setState(() {
        _objects = decoded.map((e) => ObjectData.fromJson(e)).toList();
      });
    }
  }

  Future<void> _saveObjects() async {
    final prefs = await SharedPreferences.getInstance();
    final objectsJson = jsonEncode(_objects.map((e) => e.toJson()).toList());
    await prefs.setString('registered_objects', objectsJson);
  }

  Future<void> _captureObject() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
    );

    if (image != null && mounted) {
      _showNameDialog(image.path);
    }
  }

  void _showNameDialog(String imagePath) {
    final nameController = TextEditingController();
    bool isPriority = false;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          backgroundColor: Colors.grey[900],
          title: const Text(
            'Register Object',
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
                  labelText: 'Object Name',
                  labelStyle: TextStyle(color: Colors.grey),
                  hintText: 'e.g., Keys, Wallet, Medicine',
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
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
            ),
            TextButton(
              onPressed: () {
                if (nameController.text.isNotEmpty) {
                  setState(() {
                    _objects.add(ObjectData(
                      id: DateTime.now().toString(),
                      name: nameController.text,
                      imagePath: imagePath,
                      isPriority: isPriority,
                    ));
                  });
                  _saveObjects();
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
      ),
    );
  }

  void _deleteObject(ObjectData object) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey[900],
        title:
            const Text('Delete Object', style: TextStyle(color: Colors.yellow)),
        content: Text(
          'Delete ${object.name}?',
          style: const TextStyle(color: Colors.white),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () {
              setState(() => _objects.remove(object));
              _saveObjects();
              // Delete image file
              try {
                File(object.imagePath).deleteSync();
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
          'Object Registration',
          style: TextStyle(color: Colors.yellow),
        ),
        iconTheme: const IconThemeData(color: Colors.yellow),
      ),
      body: _objects.isEmpty
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
                childAspectRatio: 0.75,
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
                            ClipRRect(
                              borderRadius: const BorderRadius.vertical(
                                top: Radius.circular(4),
                              ),
                              child: Image.file(
                                File(object.imagePath),
                                fit: BoxFit.cover,
                                width: double.infinity,
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
                              icon: const Icon(Icons.delete, color: Colors.red),
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
        onPressed: _captureObject,
        backgroundColor: Colors.yellow,
        child: const Icon(Icons.camera_alt, color: Colors.black),
      ),
    );
  }
}
