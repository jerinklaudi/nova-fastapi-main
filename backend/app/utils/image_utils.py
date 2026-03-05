import io
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
from fastapi import HTTPException
from app.core.logging import get_logger

logger = get_logger(__name__)

def validate_image_content_type(content_type: str) -> bool:
    """Validate if the content type is supported."""
    return content_type.lower() in ["image/jpeg", "image/jpg", "image/png"]

def validate_image_size(size: int, max_size: int = 10 * 1024 * 1024) -> bool:
    """Validate if the image size is within limits."""
    return size <= max_size

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load image from bytes."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def convert_to_opencv_format(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to numpy array (RGB)
    image_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_bgr

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image for model inference."""
    # Convert to float32 and normalize to [0, 1]
    image_float = image.astype(np.float32) / 255.0
    return image_float

def preprocess_image_for_yolo(image: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess image for YOLO model."""
    try:
        # Convert to OpenCV format
        image_cv = convert_to_opencv_format(image)
        
        # Resize to target size
        resized_image = resize_image(image_cv, target_size)
        
        # Normalize
        normalized_image = normalize_image(resized_image)
        
        # Add batch dimension
        processed_image = np.expand_dims(normalized_image, axis=0)
        
        logger.info(f"Image preprocessed for YOLO: input shape {image.size}, output shape {processed_image.shape}")
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Failed to preprocess image for YOLO: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

def preprocess_image_for_midas(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess image for MiDaS depth estimation."""
    try:
        # Convert to OpenCV format
        image_cv = convert_to_opencv_format(image)
        
        # Resize to target size (MiDaS v2.1 small expects 256x256)
        resized_image = resize_image(image_cv, target_size)
        
        # Normalize
        normalized_image = normalize_image(resized_image)
        
        logger.info(f"Image preprocessed for MiDaS: input shape {image.size}, output shape {normalized_image.shape}")
        
        return normalized_image
        
    except Exception as e:
        logger.error(f"Failed to preprocess image for MiDaS: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

def preprocess_image_for_ocr(image: Image.Image, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Preprocess image for OCR detection."""
    try:
        # Convert to OpenCV format
        image_cv = convert_to_opencv_format(image)
        
        # Resize if target size specified
        if target_size:
            resized_image = resize_image(image_cv, target_size)
        else:
            resized_image = image_cv
        
        logger.info(f"Image preprocessed for OCR: input shape {image.size}, output shape {resized_image.shape}")
        
        return resized_image
        
    except Exception as e:
        logger.error(f"Failed to preprocess image for OCR: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

def preprocess_image_for_face(image: Image.Image, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """Preprocess image for face detection/recognition."""
    try:
        # Convert to OpenCV format
        image_cv = convert_to_opencv_format(image)
        
        # Resize to target size
        resized_image = resize_image(image_cv, target_size)
        
        # Normalize
        normalized_image = normalize_image(resized_image)
        
        # Convert to CHW format (channels first) for ONNX models
        # OpenCV gives HWC, we need CHW
        chw_image = np.transpose(normalized_image, (2, 0, 1))
        
        # Add batch dimension
        processed_image = np.expand_dims(chw_image, axis=0)
        
        logger.info(f"Image preprocessed for face model: input shape {image.size}, output shape {processed_image.shape}")
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Failed to preprocess image for face model: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

def postprocess_image_for_response(image: np.ndarray) -> bytes:
    """Convert processed image back to bytes for response."""
    try:
        # Convert back to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to postprocess image: {str(e)}")
        raise HTTPException(status_code=500, detail="Image postprocessing failed")

def draw_bounding_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on image."""
    try:
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(image, label_text, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to draw bounding boxes: {str(e)}")
        return image