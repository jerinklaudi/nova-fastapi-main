import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text(
          'About NOVA',
          style: TextStyle(color: Colors.yellow),
        ),
        iconTheme: const IconThemeData(color: Colors.yellow),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // App Logo/Icon
            Center(
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: Colors.yellow,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: const Icon(
                  Icons.visibility,
                  size: 80,
                  color: Colors.black,
                ),
              ),
            ),

            const SizedBox(height: 24),

            // App Name and Version
            const Center(
              child: Text(
                'NOVA',
                style: TextStyle(
                  color: Colors.yellow,
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const Center(
              child: Text(
                'Navigation and Object Vision Assistant',
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 14,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 8),
            const Center(
              child: Text(
                'Version 1.0.0',
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 12,
                ),
              ),
            ),

            const SizedBox(height: 32),

            // Description
            _buildSection(
              'About',
              'NOVA is an AI-powered assistive technology app designed to help visually impaired users navigate their environment, read text, and recognize faces and objects through real-time camera analysis and voice feedback.',
            ),

            const SizedBox(height: 24),

            // Features
            _buildSection(
              'Features',
              '• Navigation Mode: Real-time obstacle detection and guidance\n'
                  '• Reading Mode: OCR text extraction and reading\n'
                  '• Recognition Mode: Face and object detection\n'
                  '• Voice Activation: Hands-free app control\n'
                  '• Emergency Contacts: Quick access to emergency numbers\n'
                  '• Customizable Settings: Personalize your experience',
            ),

            const SizedBox(height: 24),

            // Developer Info
            _buildSection(
              'Developed By',
              'SRM Institute of Science and Technology\n'
                  'Major Project - Semester 8\n'
                  '2026',
            ),

            const SizedBox(height: 24),

            // Technology Stack
            _buildSection(
              'Technology',
              '• Flutter - Cross-platform mobile framework\n'
                  '• FastAPI - Backend server\n'
                  '• YOLOv5 - Object detection\n'
                  '• MiDaS - Depth estimation\n'
                  '• TensorFlow Lite - On-device ML',
            ),

            const SizedBox(height: 24),

            // Privacy
            _buildSection(
              'Privacy',
              'NOVA processes all camera data locally on your device and backend server. No data is sent to third parties. Face and object registration data is stored locally on your device.',
            ),

            const SizedBox(height: 24),

            // Contact
            _buildSection(
              'Support',
              'For support or feedback, please contact:\nsupport@nova-app.com',
            ),

            const SizedBox(height: 32),

            // Copyright
            const Center(
              child: Text(
                '© 2026 NOVA. All rights reserved.',
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 12,
                ),
              ),
            ),

            const SizedBox(height: 16),

            // License Button
            Center(
              child: TextButton(
                onPressed: () {
                  showLicensePage(
                    context: context,
                    applicationName: 'NOVA',
                    applicationVersion: '1.0.0',
                    applicationIcon: Container(
                      width: 60,
                      height: 60,
                      decoration: BoxDecoration(
                        color: Colors.yellow,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.visibility,
                        size: 40,
                        color: Colors.black,
                      ),
                    ),
                  );
                },
                child: const Text(
                  'View Licenses',
                  style: TextStyle(color: Colors.yellow),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSection(String title, String content) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            color: Colors.yellow,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          content,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            height: 1.5,
          ),
        ),
      ],
    );
  }
}
