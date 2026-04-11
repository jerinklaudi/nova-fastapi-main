import os
import numpy as np
import cv2
from typing import List, Tuple, Optional
import time
import re
from app.core.logging import get_logger, log_model_loading, log_inference
from app.core.config import settings
from app.schemas.detection import TextDetectionResult, BoundingBox

logger = get_logger(__name__)

# Runtime flags for PaddleOCR (must be set BEFORE importing paddleocr)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_new_executor", "0")

# Select backend - PaddleOCR or EasyOCR
OCR_BACKEND = os.environ.get("NOVA_OCR_BACKEND", "easyocr").strip().lower()
PaddleOCR = None
easyocr = None

# Try to import backends gracefully
try:
    import easyocr
    OCR_BACKEND = "easyocr"
    logger.info("EasyOCR found - using as backend")
except ImportError:
    logger.warning("EasyOCR not available, trying PaddleOCR")
    try:
        from paddleocr import PaddleOCR
        OCR_BACKEND = "paddleocr"
        logger.info("PaddleOCR found - using as backend")
    except ImportError:
        logger.error("Neither EasyOCR nor PaddleOCR available - OCR will be disabled")
        OCR_BACKEND = None
        PaddleOCR = None
        easyocr = None


class PaddleOCRDetector:
    """PaddleOCR/EasyOCR text detection and recognition - Offline first."""
    
    def __init__(self, lang: str = "en", model_path: Optional[str] = None):
        self.model_path = model_path or settings.PADDLE_OCR_MODEL_PATH
        self.lang = lang
        self.reader = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the OCR model."""
        try:
            if OCR_BACKEND is None:
                logger.warning("OCR models not available - OCR is disabled")
                self.reader = None
                return
            
            logger.info(f"NOVA OCR Module - Initializing ({OCR_BACKEND})")
            logger.info(f"Language: {self.lang}")
            logger.info(f"Device: CPU")
            logger.info("Offline after first successful model download")
            
            if OCR_BACKEND == "paddleocr":
                # PaddleOCR with angle classification and CPU-only
                self.reader = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False
                )
                backend_msg = "PaddleOCR"
            else:
                # EasyOCR with CPU-only
                self.reader = easyocr.Reader([self.lang], gpu=False)
                backend_msg = "EasyOCR"
            
            logger.info(f"OK: Backend: {backend_msg}")
            log_model_loading("PaddleOCR", True)
            
        except Exception as e:
            log_model_loading("PaddleOCR", False, str(e))
            logger.error(f"Failed to load OCR model: {str(e)}")
            self.reader = None


class PaddleOCRDetector:
    """PaddleOCR/EasyOCR text detection and recognition - Offline first."""
    
    def __init__(self, lang: str = "en", model_path: Optional[str] = None):
        self.model_path = model_path or settings.PADDLE_OCR_MODEL_PATH
        self.lang = lang
        self.reader = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the OCR model."""
        try:
            logger.info(f"NOVA OCR Module - Initializing ({OCR_BACKEND})")
            logger.info(f"Language: {self.lang}")
            logger.info(f"Device: CPU")
            logger.info("Offline after first successful model download")
            
            if OCR_BACKEND == "paddleocr":
                # PaddleOCR with angle classification and CPU-only
                self.reader = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False
                )
                backend_msg = "PaddleOCR"
            else:
                # EasyOCR with CPU-only
                self.reader = easyocr.Reader([self.lang], gpu=False)
                backend_msg = "EasyOCR"
            
            logger.info(f"OK: Backend: {backend_msg}")
            log_model_loading("PaddleOCR", True)
            
        except Exception as e:
            log_model_loading("PaddleOCR", False, str(e))
            logger.error(f"Failed to load OCR model: {str(e)}")
            self.reader = None
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleanup for TTS safety."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9.,;:'\"!? ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _remove_gibberish_words(self, text: str, threshold: float = 2.5) -> str:
        """Remove gibberish words using word frequency."""
        try:
            from wordfreq import zipf_frequency
            words = text.split()
            clean_words = [
                w for w in words if zipf_frequency(w, "en") >= threshold
            ]
            return " ".join(clean_words)
        except ImportError:
            # Fallback if wordfreq is not available
            logger.debug("wordfreq not available, skipping gibberish removal")
            return text
        except Exception as e:
            logger.warning(f"Gibberish removal failed: {str(e)}")
            return text
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text for better TTS output."""
        if not text:
            return ""
        text = self._basic_cleanup(text)
        text = self._remove_gibberish_words(text)
        return text
    
    def recognize_text(self, image: np.ndarray) -> List[TextDetectionResult]:
        """Detect and recognize text in the input image."""
        start_time = time.perf_counter()
        
        try:
            if self.reader is None:
                logger.warning("OCR model not loaded or not available")
                return []
            
            # Store original shape for postprocessing
            original_shape = image.shape
            
            # Ensure BGR format for compatibility
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already in BGR, keep as is
                img = image
            else:
                img = image
            
            # Run OCR
            if OCR_BACKEND == "paddleocr":
                try:
                    raw_results = self.reader.ocr(img, cls=True)
                except TypeError:
                    raw_results = self.reader.ocr(img)
                except AttributeError:
                    raw_results = self.reader.predict(img)
            elif OCR_BACKEND == "easyocr":
                # EasyOCR returns: [(bbox, text, conf), ...]
                raw_results = self.reader.readtext(img)
            else:
                logger.warning("OCR backend not available")
                return []
            
            inference_time = time.perf_counter() - start_time
            
            text_blocks = []
            
            # Normalize raw_results
            lines = []
            if raw_results:
                if OCR_BACKEND == "paddleocr":
                    if isinstance(raw_results, list) and len(raw_results) == 1 and isinstance(raw_results[0], list):
                        lines = raw_results[0]
                    elif isinstance(raw_results, list):
                        lines = raw_results
                else:
                    lines = raw_results
            
            # Process detections
            if lines:
                for line in lines:
                    if OCR_BACKEND == "paddleocr":
                        if line is None or len(line) < 2:
                            continue
                        bbox = line[0]
                        text = line[1][0]
                        confidence = float(line[1][1])
                    else:
                        if line is None or len(line) < 3:
                            continue
                        bbox = line[0]
                        text = line[1]
                        confidence = float(line[2])
                    
                    # Convert bbox to normalized coordinates
                    pts = np.array(bbox, np.float32)
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    
                    x_min, x_max = float(x_coords.min()), float(x_coords.max())
                    y_min, y_max = float(y_coords.min()), float(y_coords.max())
                    
                    height, width = original_shape[:2]
                    
                    bbox_obj = BoundingBox(
                        left=x_min / width,
                        top=y_min / height,
                        right=x_max / width,
                        bottom=y_max / height
                    )
                    
                    # Clean text for TTS
                    clean_text = self._clean_ocr_text(text)
                    
                    text_result = TextDetectionResult(
                        text=clean_text,
                        confidence=confidence,
                        bbox=bbox_obj
                    )
                    
                    text_blocks.append(text_result)
            
            # Log inference details
            log_inference("PaddleOCR", inference_time, image.shape, len(text_blocks))
            
            logger.info(f"Text detection completed: {len(text_blocks)} text regions found in {inference_time:.3f}s")
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"Text detection failed: {str(e)}")
            raise
    
    def detect_text(self, image: np.ndarray) -> List[TextDetectionResult]:
        """Alias for recognize_text for compatibility."""
        return self.recognize_text(image)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "model_type": f"OCR ({OCR_BACKEND.upper()})",
            "language": self.lang,
            "capabilities": ["detection", "recognition", "text_cleaning"]
        }
