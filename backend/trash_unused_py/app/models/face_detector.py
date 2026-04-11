import numpy as np
import onnxruntime as ort
import cv2
import time
from typing import Optional

from app.core.logging import get_logger, log_model_loading, log_inference, log_detections
from app.core.config import settings

logger = get_logger(__name__)


class FaceDetector:
    """Face detection model wrapper using YOLOv8-face ONNX."""

    def __init__(self, model_path=None):
        self.model_path = model_path or settings.FACE_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_names = None
        self._load_model()

    def _load_model(self):
        try:

            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            log_model_loading("Face Detection", True)
            logger.info(f"Face detection model loaded from {self.model_path}")

        except Exception as e:

            log_model_loading("Face Detection", False, str(e))
            logger.error(f"Failed to load face detection model: {str(e)}")
            raise

    def _preprocess_input(self, image):

        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise ValueError("Input image is not a numpy array")

        if image.size == 0:
            raise ValueError("Input image is empty")

        # ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # resize for YOLOv8
        img = cv2.resize(image, (640, 640))

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        # normalize
        img = img.astype(np.float32) / 255.0

        # add batch
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess_output(self, outputs, image_shape):

        detections = []

        preds = outputs[0][0]

        height, width = image_shape

        for det in preds:

            confidence = float(det[4])

            if confidence < settings.CONFIDENCE_THRESHOLD:
                continue

            x_center, y_center, w, h = det[:4]

            x1 = (x_center - w / 2) * width
            y1 = (y_center - h / 2) * height
            x2 = (x_center + w / 2) * width
            y2 = (y_center + h / 2) * height

            detections.append({
                "bbox": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height
                ],
                "confidence": confidence
            })

        return detections

    def detect_faces(self, image):

        start_time = time.time()

        try:

            input_data = self._preprocess_input(image)

            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_data}
            )

            inference_time = time.time() - start_time

            log_inference(
                "Face Detection",
                inference_time,
                input_data.shape,
                [o.shape for o in outputs]
            )

            detections = self._postprocess_output(
                outputs,
                image.shape[:2]
            )

            log_detections("Face Detection", len(detections), detections)

            return detections

        except Exception as e:

            logger.error(f"Face detection failed: {str(e)}")
            raise

    def get_model_info(self):

        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_names": self.output_names,
            "model_type": "ONNX"
        }


class FaceRecognizer:
    """Face recognition using SFace ONNX."""

    def __init__(self, model_path=None):

        self.model_path = model_path or settings.SFACE_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.embeddings_db = {}

        self._load_model()

    def _load_model(self):

        try:

            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            log_model_loading("Face Recognition", True)
            logger.info(f"Face recognition model loaded from {self.model_path}")

        except Exception as e:

            log_model_loading("Face Recognition", False, str(e))
            logger.error(f"Failed to load face recognition model: {str(e)}")
            raise

    def _preprocess_face(self, face_image):

        if face_image is None:
            raise ValueError("Face image is None")

        # resize for recognition model
        face = cv2.resize(face_image, (112, 112))

        # HWC -> CHW
        face = face.transpose(2, 0, 1)

        face = face.astype(np.float32) / 255.0

        face = np.expand_dims(face, axis=0)

        return face

    def extract_embedding(self, face_image):

        input_data = self._preprocess_face(face_image)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        embedding = outputs[0]

        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def register_face(self, person_id, face_image):

        try:

            embedding = self.extract_embedding(face_image)

            self.embeddings_db[person_id] = embedding

            logger.info(f"Registered face for {person_id}")

            return True

        except Exception as e:

            logger.error(f"Face registration failed: {str(e)}")

            return False

    def recognize_face(self, face_image, threshold=0.6):

        try:

            query_embedding = self.extract_embedding(face_image)

            best_match = None
            best_similarity = 0

            for person_id, stored_embedding in self.embeddings_db.items():

                similarity = float(np.dot(query_embedding, stored_embedding.T))

                if similarity > best_similarity and similarity >= threshold:

                    best_similarity = similarity
                    best_match = person_id

            return best_match

        except Exception as e:

            logger.error(f"Face recognition failed: {str(e)}")

            return None