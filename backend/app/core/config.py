import os
from typing import List


class Settings:
    PROJECT_NAME: str = "AI Vision Assistant"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Base project directory
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        )
    )

    # Model paths
    YOLO_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "yolov8n.pt")
    FACE_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "yolov8n-face.onnx")
    SFACE_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "sface.onnx")
    LANDMARK_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "github_landmark.onnx")

    MIDAS_MODEL_PATH: str = os.path.join(
        BASE_DIR,
        "models",
        "midas_v21_small_256.pt"
    )

    PADDLE_OCR_MODEL_PATH: str = "paddleocr_default"

    # Detection thresholds
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45

    # Navigation guidance settings
    DEPTH_DISTANCE_SCALE: float = 10.0
    DEPTH_CLOSE_THRESHOLD: float = 0.8
    DEPTH_GRADIENT_THRESHOLD: float = 0.5

    # Image processing
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]


settings = Settings()