import os
from typing import List

class Settings:
    PROJECT_NAME: str = "AI Vision Assistant"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model paths - using actual available models
    YOLO_MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "models",
        "yolov5s-fp16.tflite"
    )
    FACE_MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "models", "yolov8n-face.onnx"
    )
    SFACE_MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "models", "sface.onnx"
    )
    LANDMARK_MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "models", "github_landmark.onnx"
    )
    MIDAS_MODEL_PATH: str = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        )
    ),
    "models",
    "midas_v21_small_256.pt"
)

    PADDLE_OCR_MODEL_PATH: str = "paddleocr_default"  # Will use default PaddleOCR models
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    
    # Navigation guidance settings
    DEPTH_DISTANCE_SCALE: float = 10.0
    DEPTH_CLOSE_THRESHOLD: float = 0.8
    DEPTH_GRADIENT_THRESHOLD: float = 0.5
    
    # Image processing
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]

settings = Settings()