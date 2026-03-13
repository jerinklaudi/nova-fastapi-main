import 'package:flutter/material.dart';
import '../models/app_settings.dart';
import '../services/settings_service.dart';
import 'emergency_contacts_screen.dart';
import 'face_registration_screen.dart';
import 'face_management_screen.dart';
import 'object_registration_screen.dart';
import 'about_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _settingsService = SettingsService.instance;
  late AppSettings _settings;
  final _hostController = TextEditingController();
  final _portController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _settings = _settingsService.settings;
    _hostController.text = _settings.backendHost;
    _portController.text = _settings.backendPort.toString();
  }

  @override
  void dispose() {
    _hostController.dispose();
    _portController.dispose();
    super.dispose();
  }

  Future<void> _updateSettings(AppSettings newSettings) async {
    await _settingsService.saveSettings(newSettings);
    setState(() => _settings = newSettings);
  }

  Future<void> _resetToDefaults() async {
    await _settingsService.resetToDefaults();
    setState(() {
      _settings = const AppSettings();
      _hostController.text = _settings.backendHost;
      _portController.text = _settings.backendPort.toString();
    });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Settings reset to defaults'),
          backgroundColor: Colors.green,
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
          'Settings',
          style: TextStyle(color: Colors.yellow),
        ),
        iconTheme: const IconThemeData(color: Colors.yellow),
        actions: [
          TextButton(
            onPressed: _resetToDefaults,
            child: const Text(
              'Reset',
              style: TextStyle(color: Colors.red),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Voice & Audio Settings
          _buildSectionHeader('🔊 VOICE & AUDIO'),
          _buildSlider(
            label: 'Speech Rate',
            value: _settings.speechRate,
            min: 0.3,
            max: 1.0,
            divisions: 14,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(speechRate: value),
            ),
          ),
          _buildSlider(
            label: 'Volume',
            value: _settings.volume,
            min: 0.0,
            max: 1.0,
            divisions: 10,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(volume: value),
            ),
          ),
          _buildSwitch(
            label: 'Voice Feedback',
            value: _settings.enableVoiceFeedback,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(enableVoiceFeedback: value),
            ),
          ),

          const SizedBox(height: 24),

          const SizedBox(height: 24),

          // Detection Settings
          _buildSectionHeader('🎯 DETECTION'),
          _buildSwitch(
            label: 'Auto-Detection',
            value: _settings.autoDetection,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(autoDetection: value),
            ),
          ),

  _buildSlider(
            label: 'Frame Rate (${_settings.frameRateMs}ms)',
            value: _settings.frameRateMs.toDouble(),
            min: 300,
            max: 2000,
            divisions: 17,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(frameRateMs: value.toInt()),
            ),
          ),
          _buildSlider(
            label: 'Confidence Threshold',
            value: _settings.confidenceThreshold,
            min: 0.3,
            max: 0.9,
            divisions: 12,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(confidenceThreshold: value),
            ),
          ),

          const SizedBox(height: 24),

          // Accessibility Settings
          _buildSectionHeader('♿ ACCESSIBILITY'),
          _buildSwitch(
            label: 'Haptic Feedback',
            value: _settings.hapticFeedback,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(hapticFeedback: value),
            ),
          ),
          _buildSwitch(
            label: 'High Contrast Mode',
            value: _settings.highContrast,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(highContrast: value),
            ),
          ),
          _buildSwitch(
            label: 'Large Text',
            value: _settings.largeText,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(largeText: value),
            ),
          ),

          const SizedBox(height: 24),

          // Advanced Settings
          _buildSectionHeader('⚙️ ADVANCED'),
          _buildTextField(
            label: 'Backend Host',
            controller: _hostController,
            onSubmitted: (value) => _updateSettings(
              _settings.copyWith(backendHost: value),
            ),
          ),
          _buildTextField(
            label: 'Backend Port',
            controller: _portController,
            keyboardType: TextInputType.number,
            onSubmitted: (value) => _updateSettings(
              _settings.copyWith(backendPort: int.tryParse(value) ?? 8000),
            ),
          ),
          _buildSwitch(
            label: 'Debug Mode',
            value: _settings.debugMode,
            onChanged: (value) => _updateSettings(
              _settings.copyWith(debugMode: value),
            ),
          ),

          const SizedBox(height: 32),

          // Reset button
          ElevatedButton.icon(
            onPressed: _resetToDefaults,
            icon: const Icon(Icons.restore),
            label: const Text('Reset to Defaults'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.all(16),
            ),
          ),

          const SizedBox(height: 32),
          const Divider(color: Colors.grey),
          const SizedBox(height: 16),

          // Additional Features Section
          _buildSectionHeader('📱 ADDITIONAL FEATURES'),

          // Emergency Contacts
          _buildNavigationTile(
            icon: Icons.emergency,
            title: 'Emergency Contacts',
            subtitle: 'Manage emergency contact numbers',
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => const EmergencyContactsScreen(),
                ),
              );
            },
          ),

          const SizedBox(height: 12),

          // Face Registration
          _buildNavigationTile(
            icon: Icons.face_retouching_natural,
            title: 'Register a Face',
            subtitle: 'Add a new person to recognition',
            onTap: () async {
              final name = await _showNameDialog(context);
              if (name == null || name.trim().isEmpty) return;
              
              if (context.mounted) {
                final success = await Navigator.push<bool>(
                  context,
                  MaterialPageRoute(
                    builder: (_) => FaceRegistrationScreen(personName: name.trim()),
                  ),
                );
                
                if (success == true && context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('${name.trim()} successfully registered!'),
                      backgroundColor: Colors.green,
                    ),
                  );
                }
              }
            },
          ),
          
          const SizedBox(height: 12),
          
          // Face Management
          _buildNavigationTile(
            icon: Icons.manage_accounts,
            title: 'Manage Registered Faces',
            subtitle: 'View or delete registered people',
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => const FaceManagementScreen(),
                ),
              );
            },
          ),

          const SizedBox(height: 12),

          // Object Registration
          _buildNavigationTile(
            icon: Icons.category,
            title: 'Object Registration',
            subtitle: 'Register custom objects to detect',
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => const ObjectRegistrationScreen(),
                ),
              );
            },
          ),

          const SizedBox(height: 12),

          // About
          _buildNavigationTile(
            icon: Icons.info,
            title: 'About NOVA',
            subtitle: 'App information and credits',
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => const AboutScreen(),
                ),
              );
            },
          ),

          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12, top: 8),
      child: Text(
        title,
        style: const TextStyle(
          color: Colors.yellow,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildSlider({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.white, fontSize: 16),
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          activeColor: Colors.yellow,
          inactiveColor: Colors.grey,
          label: value.toStringAsFixed(2),
          onChanged: onChanged,
        ),
      ],
    );
  }

  Widget _buildSwitch({
    required String label,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return SwitchListTile(
      title: Text(
        label,
        style: const TextStyle(color: Colors.white, fontSize: 16),
      ),
      value: value,
      activeColor: Colors.yellow,
      onChanged: onChanged,
      contentPadding: EdgeInsets.zero,
    );
  }

  Widget _buildTextField({
    required String label,
    required TextEditingController controller,
    TextInputType? keyboardType,
    required ValueChanged<String> onSubmitted,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: TextField(
        controller: controller,
        keyboardType: keyboardType,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: const TextStyle(color: Colors.grey),
          enabledBorder: const OutlineInputBorder(
            borderSide: BorderSide(color: Colors.grey),
          ),
          focusedBorder: const OutlineInputBorder(
            borderSide: BorderSide(color: Colors.yellow),
          ),
        ),
        onSubmitted: onSubmitted,
      ),
    );
  }

  Widget _buildNavigationTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Card(
      color: Colors.grey[900],
      child: ListTile(
        leading: Icon(icon, color: Colors.yellow, size: 28),
        title: Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        subtitle: Text(
          subtitle,
          style: const TextStyle(color: Colors.grey, fontSize: 12),
        ),
        trailing: const Icon(Icons.arrow_forward_ios, color: Colors.grey, size: 16),
        onTap: onTap,
      ),
    );
  }

  Future<String?> _showNameDialog(BuildContext context) {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.grey[900],
        title: const Text('Enter Person\'s Name', style: TextStyle(color: Colors.yellow)),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: Colors.white),
          textCapitalization: TextCapitalization.words,
          decoration: const InputDecoration(
            hintText: 'e.g. Alice, Bob',
            hintStyle: TextStyle(color: Colors.grey),
            enabledBorder: OutlineInputBorder(borderSide: BorderSide(color: Colors.grey)),
            focusedBorder: OutlineInputBorder(borderSide: BorderSide(color: Colors.yellow)),
          ),
          onSubmitted: (v) => Navigator.pop(ctx, v),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            child: const Text('Next', style: TextStyle(color: Colors.yellow)),
          ),
        ],
      ),
    );
  }
}
