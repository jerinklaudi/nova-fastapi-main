import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict
import time
from app.core.logging import get_logger, log_model_loading, log_inference, log_detections
from app.core.config import settings
from app.schemas.detection import FaceDetectionResult, BoundingBox

logger = get_logger(__name__)

class FaceDetector:
    """Face detection model wrapper using ONNX Runtime."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.FACE_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_names = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ONNX model."""
        try:
            self.session = ort.InferenceSession(self.model_path)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            log_model_loading("Face Detection", True)
            logger.info(f"Face detection model loaded successfully from {self.model_path}")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output names: {self.output_names}")
            
        except Exception as e:
            log_model_loading("Face Detection", False, str(e))
            logger.error(f"Failed to load face detection model: {str(e)}")
            raise
    
    def _preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for face detection."""
        # Image should already be preprocessed by image_utils
        # Just ensure correct format
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
    def _postprocess_output(self, outputs: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess model outputs to extract face detections."""
        # This is a simplified version - actual implementation depends on model output format
        # Common face detection models output: [boxes, scores, landmarks]
        
        if len(outputs) < 2:
            return []
        
        boxes = outputs[0]  # Shape: (N, 4) - [x1, y1, x2, y2]
        scores = outputs[1]  # Shape: (N,) - confidence scores
        
        # Filter by confidence threshold
        high_confidence_mask = scores >= settings.CONFIDENCE_THRESHOLD
        boxes = boxes[high_confidence_mask]
        scores = scores[high_confidence_mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert coordinates to normalized format [0, 1]
        height, width = image_shape
        
        detections = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Convert to normalized coordinates
            norm_x1 = x1 / width
            norm_y1 = y1 / height
            norm_x2 = x2 / width
            norm_y2 = y2 / height
            
            detections.append({
                'bbox': [norm_x1, norm_y1, norm_x2, norm_y2],
                'confidence': float(score)
            })
        
        return detections
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in the input image."""
        start_time = time.time()
        
        try:
            # Preprocess input
            input_data = self._preprocess_input(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            inference_time = time.time() - start_time
            
            # Log inference details
            log_inference("Face Detection", inference_time, input_data.shape, [out.shape for out in outputs])
            
            # Postprocess outputs
            detections = self._postprocess_output(outputs, image.shape[2:4])
            
            # Log detection results
            log_detections("Face Detection", len(detections), detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_names": self.output_names,
            "model_type": "ONNX"
        }

class FaceRecognizer:
    """Face recognition model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.FACE_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.embeddings_db = {}  # Simple in-memory database
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the face recognition model."""
        try:
            self.session = ort.InferenceSession(self.model_path)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            log_model_loading("Face Recognition", True)
            logger.info(f"Face recognition model loaded successfully from {self.model_path}")
            
        except Exception as e:
            log_model_loading("Face Recognition", False, str(e))
            logger.error(f"Failed to load face recognition model: {str(e)}")
            raise
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for recognition."""
        # Face image should already be preprocessed
        # Just ensure correct format
        if face_image.ndim == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        return face_image.astype(np.float32)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2.T)
        
        return float(similarity[0, 0]) if similarity.ndim > 1 else float(similarity)
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding from face image."""
        try:
            # Preprocess face
            input_data = self._preprocess_face(face_image)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
            
            # Get embedding
            embedding = outputs[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            raise
    
    def register_face(self, person_id: str, face_image: np.ndarray) -> bool:
        """Register a face with a person ID."""
        try:
            embedding = self.extract_embedding(face_image)
            self.embeddings_db[person_id] = embedding
            logger.info(f"Face registered for person: {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Face registration failed: {str(e)}")
            return False
    
    def recognize_face(self, face_image: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        """Recognize a face and return the person ID."""
        try:
            query_embedding = self.extract_embedding(face_image)
            
            best_match = None
            best_similarity = 0.0
            
            for person_id, stored_embedding in self.embeddings_db.items():
                similarity = self._calculate_similarity(query_embedding, stored_embedding)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = person_id
            
            if best_match:
                logger.info(f"Face recognized as: {best_match} (similarity: {best_similarity:.3f})")
            
            return best_match
            
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            return None
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "registered_faces": len(self.embeddings_db),
            "model_type": "ONNX"
        }