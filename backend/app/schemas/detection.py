from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class DetectionResult(BaseModel):
    """Object detection result."""
    label: str = Field(..., description="Detected object class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: 'BoundingBox' = Field(..., description="Bounding box coordinates")
    distance: Optional[float] = Field(None, description="Estimated distance in meters")

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    left: float = Field(..., ge=0.0, le=1.0, description="Left coordinate (normalized)")
    top: float = Field(..., ge=0.0, le=1.0, description="Top coordinate (normalized)")
    right: float = Field(..., ge=0.0, le=1.0, description="Right coordinate (normalized)")
    bottom: float = Field(..., ge=0.0, le=1.0, description="Bottom coordinate (normalized)")

class FaceDetectionResult(BaseModel):
    """Face detection result."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Face bounding box")
    embedding: Optional[List[float]] = Field(None, description="Face embedding vector")
    person_id: Optional[str] = Field(None, description="Recognized person ID")

class TextDetectionResult(BaseModel):
    """Text detection result."""
    text: str = Field("", description="Detected text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Text bounding box")

class DepthEstimationResult(BaseModel):
    """Depth estimation result."""
    depth_map: Optional[List[List[float]]] = Field(None, description="Normalized depth map")
    min_depth: float = Field(..., description="Minimum depth value")
    max_depth: float = Field(..., description="Maximum depth value")
    mean_depth: float = Field(..., description="Mean depth value")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

class NavigationGuidanceResult(BaseModel):
    """Navigation guidance result."""
    obstacles: List[DetectionResult] = Field([], description="Detected obstacles")
    text_signs: List[TextDetectionResult] = Field([], description="Detected text signs")
    depth_analysis: Optional[DepthEstimationResult] = Field(None, description="Depth analysis")
    guidance: str = Field("", description="Navigation guidance text")
    safety_warnings: List[str] = Field([], description="Safety warnings")
    inference_time_ms: float = Field(..., description="Total inference time")
    debug_frame_base64: Optional[str] = Field(None, description="Base64 encoded debug frame")

class ObjectDetectionResponse(BaseModel):
    """Object detection API response."""
    detections: List[DetectionResult] = Field([], description="List of detected objects")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")

class FaceDetectionResponse(BaseModel):
    """Face detection API response."""
    faces: List[FaceDetectionResult] = Field([], description="List of detected faces")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")
    audio_description: Optional[str] = Field(None, description="Audio feedback description")
    audio_file: Optional[str] = Field(None, description="Path to audio file")

class TextDetectionResponse(BaseModel):
    """Text detection API response."""
    text_regions: List[TextDetectionResult] = Field([], description="List of detected text regions")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")

class DepthEstimationResponse(BaseModel):
    """Depth estimation API response."""
    min_depth: float = Field(0.0, description="Minimum depth value")
    max_depth: float = Field(0.0, description="Maximum depth value")
    mean_depth: float = Field(0.0, description="Mean depth value")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")

class NavigationGuidanceResponse(BaseModel):
    """Navigation guidance API response."""
    guidance: NavigationGuidanceResult = Field(..., description="Navigation guidance result")
    inference_time_ms: float = Field(..., description="Total inference time")

class HealthResponse(BaseModel):
    """Health check API response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field([], description="List of loaded models")

class ModelInfoResponse(BaseModel):
    """Model information API response."""
    yolo_detector: Optional[Dict[str, Any]] = Field(None, description="YOLO model information")
    face_detector: Optional[Dict[str, Any]] = Field(None, description="Face detection model information")
    face_recognizer: Optional[Dict[str, Any]] = Field(None, description="Face recognition model information")
    midas_depth: Optional[Dict[str, Any]] = Field(None, description="MiDaS depth model information")
    paddle_ocr: Optional[Dict[str, Any]] = Field(None, description="PaddleOCR model information")
    status: str = Field(..., description="Overall model loading status")

# Update forward references
DetectionResult.update_forward_refs()
FaceDetectionResult.update_forward_refs()
TextDetectionResult.update_forward_refs()
DepthEstimationResult.update_forward_refs()
NavigationGuidanceResult.update_forward_refs()
ObjectDetectionResponse.update_forward_refs()
FaceDetectionResponse.update_forward_refs()
TextDetectionResponse.update_forward_refs()
DepthEstimationResponse.update_forward_refs()
NavigationGuidanceResponse.update_forward_refs()