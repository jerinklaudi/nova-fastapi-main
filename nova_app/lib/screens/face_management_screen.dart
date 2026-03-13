import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'face_registration_screen.dart';

class FaceManagementScreen extends StatefulWidget {
  const FaceManagementScreen({super.key});

  @override
  State<FaceManagementScreen> createState() => _FaceManagementScreenState();
}

class _FaceManagementScreenState extends State<FaceManagementScreen> {
  List<Map<String, dynamic>> _faces = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadFaces();
  }

  Future<void> _loadFaces() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final faces = await ApiService.listRegisteredFaces();
      if (mounted) {
        setState(() {
          _faces = faces;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Failed to load faces:\n$e';
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _deleteFace(String name) async {
    final bool confirm = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            backgroundColor: Colors.grey[900],
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            title: const Text('Delete Face', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold)),
            content: Text('Are you sure you want to delete registration for "$name"?', style: const TextStyle(color: Colors.white70)),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Cancel', style: TextStyle(color: Colors.grey, fontSize: 16)),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(ctx, true),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red.shade800,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                child: const Text('Delete', style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ],
          ),
        ) ??
        false;

    if (!confirm) return;

    try {
      await ApiService.deleteRegisteredFace(name);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Deleted $name'), 
            backgroundColor: Colors.green.shade700,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
        );
        _loadFaces(); // Reload list
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error deleting: $e'), 
            backgroundColor: Colors.red.shade800,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
        );
      }
    }
  }

  Future<void> _addFace() async {
    final name = await _showNameDialog(context);
    if (name == null || name.trim().isEmpty) return;
    
    if (!mounted) return;
    
    final success = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (_) => FaceRegistrationScreen(personName: name.trim()),
      ),
    );
    
    if (success == true && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${name.trim()} successfully registered!'),
          backgroundColor: Colors.green.shade700,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
      );
      _loadFaces();
    }
  }

  Future<String?> _showNameDialog(BuildContext context) {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Add New Person', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold)),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: Colors.white),
          textCapitalization: TextCapitalization.words,
          decoration: InputDecoration(
            hintText: 'e.g. Alice, Bob',
            hintStyle: const TextStyle(color: Colors.white30),
            filled: true,
            fillColor: Colors.black26,
            enabledBorder: OutlineInputBorder(
              borderSide: BorderSide.none,
              borderRadius: BorderRadius.circular(8)
            ),
            focusedBorder: OutlineInputBorder(
              borderSide: const BorderSide(color: Colors.yellow, width: 2),
              borderRadius: BorderRadius.circular(8)
            ),
          ),
          onSubmitted: (v) => Navigator.pop(ctx, v),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey, fontSize: 16)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.yellow,
              foregroundColor: Colors.black,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            ),
            child: const Text('Next', style: TextStyle(fontWeight: FontWeight.bold)),
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
        title: const Text('Manage Faces', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.yellow),
        elevation: 0,
        actions: [
          if (!_isLoading && _faces.isNotEmpty)
            Center(
              child: Padding(
                padding: const EdgeInsets.only(right: 16.0),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.white10,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '${_faces.length} Registered',
                    style: const TextStyle(color: Colors.white70, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadFaces,
        color: Colors.yellow,
        backgroundColor: Colors.grey[900],
        child: _buildBody(),
      ),
      floatingActionButton: _isLoading || _error != null ? null : FloatingActionButton.extended(
        onPressed: _addFace,
        backgroundColor: Colors.yellow,
        foregroundColor: Colors.black,
        icon: const Icon(Icons.person_add),
        label: const Text('Add Person', style: TextStyle(fontWeight: FontWeight.bold)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator(color: Colors.yellow));
    }
    if (_error != null) {
      return ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(24.0),
        children: [
          const SizedBox(height: 100),
          const Icon(Icons.cloud_off, color: Colors.redAccent, size: 80),
          const SizedBox(height: 24),
          Text(
            'Connection Drop', 
            textAlign: TextAlign.center, 
            style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)
          ),
          const SizedBox(height: 12),
          Text(
            _error!, 
            textAlign: TextAlign.center, 
            style: const TextStyle(color: Colors.white60, fontSize: 16)
          ),
          const SizedBox(height: 32),
          Center(
            child: ElevatedButton.icon(
              onPressed: _loadFaces,
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again', style: TextStyle(fontWeight: FontWeight.bold)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.yellow, 
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ],
      );
    }
    if (_faces.isEmpty) {
      return ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(24.0),
        children: [
          const SizedBox(height: 100),
          Container(
            padding: const EdgeInsets.all(32),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.grey[900],
            ),
            child: const Icon(Icons.no_accounts, color: Colors.white24, size: 80),
          ),
          const SizedBox(height: 32),
          const Text(
            'No Faces Yet',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          const Text(
            'Add people to NOVA so they can be recognized by name during navigation and recognition modes.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white60, fontSize: 16, height: 1.5),
          ),
          const SizedBox(height: 48),
          Center(
            child: ElevatedButton.icon(
              onPressed: _addFace,
              icon: const Icon(Icons.person_add),
              label: const Text('Register First Face', style: TextStyle(fontWeight: FontWeight.bold)),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.yellow, 
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ],
      );
    }

    return ListView.builder(
      physics: const AlwaysScrollableScrollPhysics(),
      itemCount: _faces.length,
      padding: const EdgeInsets.only(top: 8, bottom: 100, left: 12, right: 12),
      itemBuilder: (context, index) {
        final face = _faces[index];
        final name = face['name'] as String? ?? 'Unknown';
        final sampleCount = face['sample_count'] as int? ?? 0;
        final initial = name.isNotEmpty ? name[0].toUpperCase() : '?';

        return Card(
          color: const Color(0xff1e1e1e),
          elevation: 4,
          shadowColor: Colors.black54,
          margin: const EdgeInsets.symmetric(vertical: 8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: const BorderSide(color: Colors.white10, width: 1),
          ),
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: ListTile(
              contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              leading: Container(
                width: 50,
                height: 50,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFFFFD700), Color(0xFFFFA500)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(color: Colors.yellow.withOpacity(0.2), blurRadius: 8, spreadRadius: 2),
                  ],
                ),
                child: Center(
                  child: Text(
                    initial,
                    style: const TextStyle(color: Colors.black, fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              title: Text(name, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
              subtitle: Padding(
                padding: const EdgeInsets.only(top: 6.0),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.white10,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.camera_alt, color: Colors.white54, size: 12),
                          const SizedBox(width: 4),
                          Text('$sampleCount samples', style: const TextStyle(color: Colors.white70, fontSize: 12)),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              trailing: IconButton(
                icon: const Icon(Icons.delete_outline, color: Colors.white54, size: 28),
                splashRadius: 24,
                onPressed: () => _deleteFace(name),
              ),
            ),
          ),
        );
      },
    );
  }
}
