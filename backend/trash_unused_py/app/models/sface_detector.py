import numpy as np
import cv2
from typing import List, Tuple ,Optional

import time
from app.core.logging import get_logger, log_model_loading, log_inference
from app.core.config import settings
from app.schemas.detection import FaceDetectionResult, BoundingBox

logger = get_logger(__name__)

class SFaceDetector:
    """SFace face detection and recognition model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.SFACE_MODEL_PATH
        self.detector = None
        self.recognizer = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load SFace detection and recognition models."""
        try:
            # Load SFace detector
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),  # Default input size
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000
            )
            
            # Load SFace recognizer
            self.recognizer = cv2.FaceRecognizerSF.create(
                self.model_path.replace('.onnx', '_recognizer.onnx'),
                ""
            )
            
            log_model_loading("SFace", True)
            logger.info(f"SFace models loaded successfully from {self.model_path}")
            
        except Exception as e:
            log_model_loading("SFace", False, str(e))
            logger.error(f"Failed to load SFace models: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SFace."""
        # SFace expects BGR format
        if image.ndim == 3 and image.shape[-1] == 3:
            # Convert RGB to BGR if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def _postprocess_detections(self, faces: np.ndarray, image_shape: Tuple[int, int]) -> List[FaceDetectionResult]:
        """Postprocess SFace detections."""
        results = []
        
        if faces is None or faces[1] is None:
            return results
        
        # SFace returns faces in format: [num_faces, 15] where 15 includes:
        # [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y, 
        # nose_tip_x, nose_tip_y, mouth_right_x, mouth_right_y, 
        # mouth_left_x, mouth_left_y, confidence]
        
        num_faces = faces[1][0][0]
        
        for i in range(int(num_faces)):
            face_data = faces[1][0][i+1]
            
            # Extract bounding box
            x, y, w, h = face_data[0:4]
            confidence = face_data[14]
            
            # Convert to normalized coordinates
            height, width = image_shape[:2]
            
            bbox = BoundingBox(
                left=float(x / width),
                top=float(y / height),
                right=float((x + w) / width),
                bottom=float((y + h) / height)
            )
            
            # Create face detection result
            face_result = FaceDetectionResult(
                confidence=float(confidence),
                bbox=bbox,
                embedding=None,
                person_id=None
            )
            
            results.append(face_result)
        
        return results
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces in the input image."""
        start_time = time.time()
        
        try:
            # Store original shape for postprocessing
            original_shape = image.shape
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Set input size for detector
            self.detector.setInputSize((processed_image.shape[1], processed_image.shape[0]))
            
            # Run face detection
            faces = self.detector.detect(processed_image)
            
            inference_time = time.time() - start_time
            
            # Postprocess results
            face_results = self._postprocess_detections(faces, original_shape)
            
            # Log inference details
            log_inference("SFace Detection", inference_time, processed_image.shape, len(face_results))
            
            logger.info(f"Face detection completed: {len(face_results)} faces found in {inference_time:.3f}s")
            
            return face_results
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise
    
    def extract_embedding(self, image: np.ndarray, face_result: FaceDetectionResult) -> Optional[np.ndarray]:
        """Extract face embedding for recognition."""
        try:
            if self.recognizer is None:
                return None
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Convert normalized bbox to pixel coordinates
            height, width = image.shape[:2]
            x = int(face_result.bbox.left * width)
            y = int(face_result.bbox.top * height)
            w = int((face_result.bbox.right - face_result.bbox.left) * width)
            h = int((face_result.bbox.bottom - face_result.bbox.top) * height)
            
            # Extract face region
            face_roi = processed_image[y:y+h, x:x+w]
            
            # Extract embedding
            embedding = self.recognizer.feature(face_roi)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            return None
    
    def recognize_face(self, image: np.ndarray, face_result: FaceDetectionResult) -> Optional[str]:
        """Recognize face and return person ID."""
        try:
            if self.recognizer is None:
                return None
            
            # Extract embedding
            embedding = self.extract_embedding(image, face_result)
            if embedding is None:
                return None
            
            # For now, return a placeholder ID
            # In a real implementation, you would compare against a database
            # of known face embeddings
            return f"person_{hash(embedding.tobytes()) % 1000:03d}"
            
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            return None
    
    def detect_and_recognize(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces and perform recognition."""
        start_time = time.time()
        
        try:
            # Detect faces
            face_results = self.detect_faces(image)
            
            # Perform recognition for each face
            for face_result in face_results:
                embedding = self.extract_embedding(image, face_result)
                person_id = self.recognize_face(image, face_result)
                
                if embedding is not None:
                    face_result.embedding = embedding.tolist()
                if person_id is not None:
                    face_result.person_id = person_id
            
            total_time = time.time() - start_time
            
            logger.info(f"Face detection and recognition completed: {len(face_results)} faces in {total_time:.3f}s")
            
            return face_results
            
        except Exception as e:
            logger.error(f"Face detection and recognition failed: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "model_type": "SFace Detection + Recognition",
            "capabilities": ["detection", "recognition", "embedding_extraction"],
            "input_format": "BGR images",
            "output_format": "Bounding boxes + embeddings"
        }