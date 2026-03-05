# AI Vision Assistant

A comprehensive AI-powered vision assistant that provides object detection, face detection, and face recognition capabilities with audio feedback for visually impaired users.

## Features

- **Object Detection**: Detect and identify objects in images using YOLO model
- **Face Detection**: Detect faces in images with high accuracy
- **Face Recognition**: Recognize known faces and identify people
- **Audio Feedback**: Generate spoken descriptions of detected objects and faces
- **RESTful API**: Easy-to-use API endpoints for integration
- **Real-time Processing**: Optimized for fast inference on various hardware

## Architecture

The project follows a modular architecture with clear separation of concerns:

```
backend/
├── app/           # FastAPI application
│   ├── api/       # API endpoints
│   ├── core/      # Configuration
│   ├── models/    # ML model wrappers
│   └── main.py    # Application entry point
└── requirements.txt

nova_app/          # Flutter mobile application
├── lib/
│   ├── core/      # App configuration
│   ├── screens/   # UI screens
│   └── services/  # API services
└── pubspec.yaml
```

## Installation

### Backend Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd nova-fastapi-main/backend
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the backend directory:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Download models:**
   - Download YOLO model (e.g., yolov5s.tflite) to `models/` directory
   - Download face detection model (e.g., face_model.onnx) to `models/` directory
   - Download MiDaS depth estimation model to `models/midas_v3_small.tflite`
   - PaddleOCR models will be downloaded automatically on first use

6. **Run the server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

    ```

### Local Network Setup (Mobile Access)

To access the backend from the Flutter app on a physical device, both devices must be on the same Wi-Fi network.

1.  **Find your PC's Local IP Address:**
    - Open **Command Prompt** or **PowerShell**.
    - Run the command: `ipconfig`
    - Look for **IPv4 Address** under your active network adapter (e.g., `192.168.1.15`).

2.  **Configure the Flutter App:**
    - Open [nova_app/lib/core/config.dart](file:///c:/Users/jerin/Desktop/Jerin/SRM/Semesters/Semester%208/Major%20Project/NOVA/nova-fastapi-main/nova_app/lib/core/config.dart).
    - Update `backendHost` with your PC's IP address:
      ```dart
      static const String backendHost = '192.168.1.15'; // Replace with your IP
      ```

3.  **Run the Backend:**
    Ensure the backend is running on `0.0.0.0` to accept external connections:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

### Flutter App Setup

1. **Navigate to nova_app:**

   ```bash
   cd ../nova_app
   ```

2. **Install dependencies:**

   ```bash
   flutter pub get
   ```

3. **Run the app:**
   ```bash
   flutter run
   ```

## API Endpoints

### Health Check

- `GET /health/` - Health check endpoint
- `GET /health/status` - Detailed system status

### Object Detection

- `POST /inference/objects` - Detect objects in an image
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `confidence_threshold`: Minimum confidence (0.0-1.0)
    - `return_audio`: Generate audio feedback (boolean)

### Face Detection

- `POST /inference/faces` - Detect faces in an image
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `confidence_threshold`: Minimum confidence (0.0-1.0)
    - `recognize_faces`: Perform face recognition (boolean)
    - `return_audio`: Generate audio feedback (boolean)

### SFace Detection

- `POST /inference/faces` - Detect faces using SFace model (alternative to standard face detection)
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `confidence_threshold`: Minimum confidence (0.0-1.0)
    - `recognize_faces`: Perform face recognition (boolean)
    - `return_audio`: Generate audio feedback (boolean)

### Combined Detection

- `POST /inference/all` - Perform both object and face detection
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `object_confidence`: Object detection threshold
    - `face_confidence`: Face detection threshold
    - `recognize_faces`: Perform face recognition (boolean)
    - `return_audio`: Generate audio feedback (boolean)

### Depth Estimation

- `POST /inference/depth` - Estimate depth from an image using MiDaS
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `return_audio`: Generate audio feedback (boolean)

### Text Detection

- `POST /inference/text` - Detect text in an image using PaddleOCR
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `confidence_threshold`: Minimum confidence (0.0-1.0)
    - `return_audio`: Generate audio feedback (boolean)

### Navigation Guidance

- `POST /inference/navigation` - Generate navigation guidance using YOLO + MiDaS + OCR fusion
  - Parameters:
    - `file`: Image file (JPEG/PNG)
    - `object_confidence`: Object detection threshold
    - `text_confidence`: Text detection threshold
    - `return_audio`: Generate audio feedback (boolean)
- `GET /inference/navigation/models` - Get status of navigation models

### Model Information

- `GET /inference/models/info` - Get loaded model information

## Configuration

The application uses environment variables for configuration. Key settings include:

- `YOLO_MODEL_PATH`: Path to YOLO model file
- `FACE_MODEL_PATH`: Path to face detection/recognition model
- `MAX_IMAGE_SIZE`: Maximum allowed image size in bytes
- `ALLOWED_IMAGE_TYPES`: Comma-separated list of allowed image types
- `CONFIDENCE_THRESHOLD`: Default confidence threshold
- `IOU_THRESHOLD`: IoU threshold for NMS

## Models

### Required Models

1. **YOLO Model**: For object detection
   - Format: TensorFlow Lite (.tflite)
   - Input: 640x640 RGB images
   - Output: Bounding boxes and class probabilities

2. **Face Detection Model**: For face detection
   - Format: ONNX (.onnx)
   - Input: 112x112 RGB images
   - Output: Face bounding boxes and confidence scores

3. **Face Recognition Model**: For face recognition
   - Format: ONNX (.onnx)
   - Input: 112x112 RGB images
   - Output: Face embeddings

4. **MiDaS Model**: For depth estimation
   - Format: TensorFlow Lite (.tflite)
   - Input: Variable size RGB images
   - Output: Depth map

5. **PaddleOCR Models**: For text detection and recognition
   - Format: PaddlePaddle models
   - Input: Variable size RGB images
   - Output: Text bounding boxes and recognized text

### Model Download

Models can be downloaded from:

- YOLO: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Face Detection: [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- Face Recognition: [ArcFace](https://github.com/deepinsight/insightface)
- MiDaS: [MiDaS](https://github.com/isl-org/MiDaS)
- PaddleOCR: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## Testing

Run the test suite:

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd ../frontend
npm test
```

## Audio Feedback

The system supports audio feedback on multiple platforms:

- **Windows**: Uses PowerShell and SAPI
- **macOS**: Uses built-in `say` command
- **Linux**: Uses `espeak` or `festival`

Install required packages:

```bash
# Ubuntu/Debian
sudo apt install espeak festival

# macOS
# Built-in `say` command available

# Windows
# PowerShell and SAPI available by default
```

## Performance Optimization

### Hardware Acceleration

- **GPU**: Install CUDA-enabled TensorFlow for GPU acceleration
- **Edge TPU**: Use TensorFlow Lite with Edge TPU support
- **CPU Optimization**: Use optimized TensorFlow builds

### Model Optimization

- Use quantized models for faster inference
- Implement model caching for reduced load times
- Use batch processing for multiple images

## Security

- Input validation for all file uploads
- Size limits to prevent DoS attacks
- Content type validation
- Secure API endpoints with authentication (optional)

## Deployment

### Docker

```bash
# Build backend
docker build -t ai-vision-backend .

# Run backend
docker run -p 8000:8000 ai-vision-backend

# Build frontend
docker build -t ai-vision-frontend .

# Run frontend
docker run -p 3000:3000 ai-vision-frontend
```

### Production

- Use a production WSGI server (e.g., Gunicorn)
- Set up reverse proxy (e.g., Nginx)
- Configure SSL/TLS certificates
- Implement monitoring and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Join our Discord community
- Email: support@ai-vision-assistant.com

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
