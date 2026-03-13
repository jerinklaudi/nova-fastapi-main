from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from fastapi import HTTPException
from app.core.logging import get_logger
from app.core.config import settings
from app.utils.image_utils import (
    validate_image_content_type, validate_image_size, load_image_from_bytes,
    preprocess_image_for_yolo, preprocess_image_for_face
)

logger = get_logger(__name__)

class ImagePreprocessor:
    """Service for image preprocessing."""
    
    @staticmethod
    def validate_and_load_image(image_bytes: bytes, content_type: str, max_size: Optional[int] = None) -> Image.Image:
        """Validate and load image from bytes."""
        try:
            normalized_content_type = (content_type or "").lower()
            logger.info(f"Validating image: content_type={content_type}, size={len(image_bytes)} bytes")
            
            # Validate content type
            logger.debug(f"Checking content type against allowed types: {settings.ALLOWED_IMAGE_TYPES}")
            if not validate_image_content_type(normalized_content_type):
                if normalized_content_type == "application/octet-stream":
                    logger.warning(
                        "Received generic octet-stream upload; attempting image decode based on file bytes"
                    )
                else:
                    logger.error(f"Invalid content type: {content_type}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported image type. Supported types: {settings.ALLOWED_IMAGE_TYPES}"
                    )
            logger.debug("✓ Content type validated")
            
            # Validate image size
            max_size = max_size or settings.MAX_IMAGE_SIZE
            logger.debug(f"Checking image size: {len(image_bytes)} bytes (max: {max_size} bytes)")
            if not validate_image_size(len(image_bytes), max_size):
                logger.error(f"Image too large: {len(image_bytes)} bytes > {max_size} bytes")
                raise HTTPException(
                    status_code=413, 
                    detail=f"Image too large. Maximum size: {max_size // (1024*1024)}MB"
                )
            logger.debug("✓ Image size validated")
            
            # Load image
            logger.debug("Loading image from bytes...")
            image = load_image_from_bytes(image_bytes)
            
            logger.info(f"✓ Image validated and loaded successfully: size={image.size}, mode={image.mode}")
            
            return image
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"✗ Image validation/loading failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
    
    @staticmethod
    def preprocess_for_yolo(image: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess image for YOLO model."""
        try:
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # YOLOv5 expects RGB format, so convert back
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image_resized = cv2.resize(image_rgb, target_size)
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            processed_image = np.expand_dims(image_normalized, axis=0)
            
            logger.info(f"Image preprocessed for YOLO: {processed_image.shape}")
            return processed_image
        except Exception as e:
            logger.error(f"YOLO preprocessing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Image preprocessing failed")
    
    @staticmethod
    def preprocess_for_face(image: Image.Image, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """Preprocess image for face detection/recognition."""
        try:
            processed_image = preprocess_image_for_face(image, target_size)
            logger.info(f"Image preprocessed for face model: {processed_image.shape}")
            return processed_image
        except Exception as e:
            logger.error(f"Face preprocessing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Image preprocessing failed")
    
    @staticmethod
    def resize_image(image: Image.Image, max_dimension: int = 1920) -> Image.Image:
        """Resize image to fit within max dimension while maintaining aspect ratio."""
        try:
            width, height = image.size
            
            if max(width, height) <= max_dimension:
                return image
            
            # Calculate scaling factor
            scale = max_dimension / max(width, height)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            logger.info(f"Image resized: {image.size} -> {resized_image.size}")
            
            return resized_image
            
        except Exception as e:
            logger.error(f"Image resizing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Image resizing failed")
    
    @staticmethod
    def validate_image_dimensions(image: Image.Image, min_width: int = 32, min_height: int = 32) -> bool:
        """Validate minimum image dimensions."""
        width, height = image.size
        return width >= min_width and height >= min_height