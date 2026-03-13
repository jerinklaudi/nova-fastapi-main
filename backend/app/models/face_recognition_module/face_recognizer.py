"""
face_recognizer.py
──────────────────
SFace ONNX face recognizer.

Ported directly from the project's uploaded standalone face_recognizer.py
(v3 / the version attached in this session), adapted as a backend module.

Requires:  backend/models/sface.onnx
Optional:  backend/models/github_landmark.onnx  (106-pt landmark refinement)

Public API
──────────
recognizer = FaceRecognizer(model_path, threshold, landmark_model_path)
embedding  = recognizer.get_embedding(face_crop, frame, coarse_box, coarse_kps)
  → np.ndarray (128,) normalised, or None if crop fails quality gate

is_crop_usable(crop) → bool
    Exported so registration endpoint can use the same quality gate.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings
from .embedding_db import EmbeddingDatabase

logger = logging.getLogger(__name__)

# ── Default model paths ───────────────────────────────────────────────────────
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR     = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "models"))
_DEFAULT_SFACE  = os.path.join(_MODELS_DIR, "sface.onnx")
_DEFAULT_LMARK  = os.path.join(_MODELS_DIR, "github_landmark.onnx")

# ── Thresholds ────────────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.60   # minimum cosine similarity to call it a match
MIN_LAPLACIAN_VAR     = 10.0   # below this → blurry / blank crop  (was 15.0)
MIN_MEAN_INTENSITY    = 2.0    # below this → nearly-black crop     (was 5.0)

# ── ArcFace canonical 5-point targets (112 × 112) ────────────────────────────
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark helpers (verbatim from standalone)
# ─────────────────────────────────────────────────────────────────────────────

def extract_5pts_from_106(lm_106: np.ndarray) -> np.ndarray:
    """Extract 5-point ArcFace landmarks from a 106-point landmark array."""
    return np.array([
        lm_106[list(range(66, 74))].mean(axis=0),   # left eye
        lm_106[list(range(75, 83))].mean(axis=0),   # right eye
        lm_106[[49]].mean(axis=0),                  # nose tip
        lm_106[[84]].mean(axis=0),                  # left mouth corner
        lm_106[[90]].mean(axis=0),                  # right mouth corner
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Crop quality gate (exported — used by registration endpoint too)
# ─────────────────────────────────────────────────────────────────────────────

def is_crop_usable(crop: np.ndarray) -> bool:
    """
    Return True if the crop looks like a real face (not blank / misaligned).
    This is the quality gate that prevents collapsed embeddings.
    """
    if crop is None or crop.size == 0:
        logger.debug("QualityGate: REJECTED — empty crop")
        return False
    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_int = float(gray.mean())
    ok       = (lap_var >= MIN_LAPLACIAN_VAR) and (mean_int >= MIN_MEAN_INTENSITY)
    logger.debug(
        "QualityGate: %s | laplacian=%.1f | mean_px=%.1f",
        "OK" if ok else "REJECTED", lap_var, mean_int,
    )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# FaceRecognizer
# ─────────────────────────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    SFace ONNX face recognizer.

    • Optionally refines alignment with a 106-pt landmark model.
    • Applies a crop quality gate before embedding.
    • Returns None (instead of a garbage embedding) when the crop is unusable.
    """

    LANDMARK_INPUT_SIZE = (1920, 1920)

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = RECOGNITION_THRESHOLD,
        landmark_model_path: Optional[str] = None,
    ) -> None:
        model_path = model_path or settings.SFACE_MODEL_PATH
        landmark_model_path = landmark_model_path or settings.LANDMARK_MODEL_PATH
        self.threshold = threshold
        self.db = EmbeddingDatabase()
        # In-memory buffer for frame-by-frame registration.
        # Maps person name → list of accepted 128-d embeddings.
        # Populated by accumulate_registration_frame(), consumed by flush_registration_buffer().
        self._reg_buffer: dict = {}

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SFace ONNX model not found: '{model_path}'. "
                "Place sface.onnx in backend/models/."
            )

        providers = ["CPUExecutionProvider"]

        # SFace session
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        inp              = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape            = inp.shape
        self._input_h    = shape[2] if isinstance(shape[2], int) else 112
        self._input_w    = shape[3] if isinstance(shape[3], int) else 112
        logger.info(
            "FaceRecognizer: loaded '%s' | input=%d×%d | threshold=%.2f",
            model_path, self._input_w, self._input_h, threshold,
        )

        # Optional 106-pt landmark session
        self._lm_session:    Optional[ort.InferenceSession] = None
        self._lm_input_name: Optional[str]                  = None

        lm_path = landmark_model_path or _DEFAULT_LMARK
        if lm_path and os.path.exists(lm_path):
            try:
                self._lm_session    = ort.InferenceSession(lm_path, providers=providers)
                self._lm_input_name = self._lm_session.get_inputs()[0].name
                logger.info("FaceRecognizer: landmark model loaded from '%s'", lm_path)
            except Exception as exc:
                logger.warning("FaceRecognizer: landmark model load failed — %s", exc)
        else:
            logger.info(
                "FaceRecognizer: no landmark model at '%s' — synthetic fallback active.",
                lm_path,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_landmark(
        self, frame: np.ndarray, coarse_box: List[int]
    ) -> Optional[np.ndarray]:
        """
        Run github_landmark.onnx on the face ROI.
        Returns the pre-aligned 224×224 BGR crop (align_imgs output, index 3)
        or None if no face detected / model fails.
        This is the CORRECT way to use this model — it performs its own
        internal affine alignment, so we use the result directly.
        """
        if self._lm_session is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = coarse_box[:4]

        # Generous 25% padding so the model has full face context
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)

        roi = frame[ry1:ry2, rx1:rx2]   # BGR uint8, variable size
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None

        try:
            outputs = self._lm_session.run(None, {self._lm_input_name: roi})
        except Exception as exc:
            logger.warning("Landmark inference error: %s", exc)
            return None

        # Output layout: [scores(0), bboxes(1), kpss(2), align_imgs(3), lmks(4), M(5)]
        # align_imgs: (N, 224, 224, 3) uint8 - already perfectly aligned face crops
        align_imgs = np.array(outputs[3], dtype=np.uint8)
        if align_imgs.ndim != 4 or align_imgs.shape[0] == 0:
            logger.debug("Landmark model: no align_imgs output (0 detections in ROI).")
            return None

        # Pick the detection with highest score
        scores = np.array(outputs[0], dtype=np.float32)
        best_idx = int(np.argmax(scores)) if scores.size > 0 else 0
        aligned_224 = align_imgs[best_idx]   # (224, 224, 3) BGR uint8

        # Basic sanity: reject blank (all-dark or all-white) crops
        mean_pix = float(aligned_224.mean())
        if mean_pix < 15.0 or mean_pix > 245.0:
            logger.debug("Landmark align_imgs rejected: mean_pix=%.1f", mean_pix)
            return None

        # Resize to 112×112 for SFace
        aligned_112 = cv2.resize(aligned_224, (112, 112), interpolation=cv2.INTER_LINEAR)
        return aligned_112

    def _preprocess_sface(self, face_crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_crop, (self._input_w, self._input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_embedding(
        self,
        face_crop: np.ndarray,
        frame: Optional[np.ndarray] = None,
        coarse_box: Optional[List[int]] = None,
        coarse_kps: Optional[List] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute a normalised 128-D SFace embedding.

        Alignment priority:
          1. github_landmark.onnx align_imgs output (perfectly aligned 112×112 crop)
          2. Coarse face_crop from YOLOv8-face (fallback only)

        Returns np.ndarray (128,) or None if crop fails quality gate.
        """
        aligned = face_crop

        # Try refined alignment using the landmark model's built-in aligned output
        if self._lm_session is not None and frame is not None and coarse_box is not None:
            lm_aligned = self._run_landmark(frame, coarse_box)
            if lm_aligned is not None:
                aligned = lm_aligned
                logger.debug("FaceRecognizer: using landmark-aligned crop (✓)")
            else:
                logger.debug("FaceRecognizer: landmark failed — falling back to coarse crop")

        # Quality gate — reject blank / misaligned crops
        if not is_crop_usable(aligned):
            logger.debug("FaceRecognizer: crop failed quality gate")
            return None

        # SFace forward pass
        blob    = self._preprocess_sface(aligned)
        outputs = self._session.run(None, {self._input_name: blob})
        emb     = outputs[0].flatten().astype(np.float32).copy()
        norm    = np.linalg.norm(emb)
        if norm > 1e-9:
            emb /= norm
        return emb

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Wrapper for get_embedding that simplifies the API for registration.
        Assumes the face_image is already a reasonably centered crop.
        """
        if not is_crop_usable(face_image):
            return None
        return self.get_embedding(face_image)

    def recognize_face(
        self, 
        face_image: np.ndarray, 
        threshold: float = 0.5,
        frame: Optional[np.ndarray] = None,
        coarse_box: Optional[List[int]] = None,
        coarse_kps: Optional[List[List[float]]] = None
    ) -> Optional[str]:
        """
        Extract embedding and compare against stored embeddings.
        If frame/kps are provided, performs refined alignment for better accuracy.
        """
        # Try to get high-quality aligned embedding first
        if frame is not None and coarse_kps is not None:
            query_emb = self.get_embedding(face_image, frame, coarse_box, coarse_kps)
        else:
            query_emb = self.extract_embedding(face_image)
            
        if query_emb is None or not self.db:
            return None

        # Ensure query embedding is normalized
        norm = np.linalg.norm(query_emb)
        if norm > 1e-9:
            query_emb = query_emb / norm

        best_id = None
        best_sim = -1.0

        # Iterate identities in database
        for person_id in self.db.list_identities():
            stored_emb = self.db._data[person_id]["embedding"]
            
            # Ensure stored embedding is normalized
            s_norm = np.linalg.norm(stored_emb)
            if s_norm > 1e-9:
                stored_emb = stored_emb / s_norm

            similarity = float(np.dot(query_emb, stored_emb.T))
            
            logger.debug("Similarity with '%s' = %.4f", person_id, similarity)

            if similarity > best_sim:
                best_sim = similarity
                best_id = person_id

        if best_sim >= threshold:
            logger.info("Face Match: '%s' | similarity=%.3f", best_id, best_sim)
            return best_id
        
        return None

    def get_model_info(self) -> dict:
        return {
            "model_type":     "SFace (ONNX)",
            "input_size":     f"{self._input_w}×{self._input_h}",
            "threshold":      self.threshold,
            "landmark_model": self._lm_session is not None,
        }

    def accumulate_registration_frame(self, name: str, embedding: np.ndarray) -> None:
        """
        Buffer one quality-gated embedding for *name* during a registration session.
        Called once per accepted camera frame from the registration endpoint.
        """
        if name not in self._reg_buffer:
            self._reg_buffer[name] = []
        self._reg_buffer[name].append(embedding.astype(np.float32).copy())

    def flush_registration_buffer(self, name: str) -> Optional[np.ndarray]:
        """
        Average all buffered embeddings for *name*, normalise to unit length,
        clear the buffer, and return the final embedding.

        Returns None if the buffer is empty (no quality frames were accepted).
        Called once at the end of a registration session to produce the final embedding.
        """
        frames = self._reg_buffer.pop(name, [])
        if not frames:
            return None
        stacked = np.stack(frames, axis=0)   # (N, 128)
        avg = stacked.mean(axis=0)           # (128,)
        norm = np.linalg.norm(avg)
        if norm > 1e-9:
            avg = avg / norm
        return avg.astype(np.float32)

    def clear_registration_buffer(self, name: str) -> None:
        """
        Discard all buffered frames for *name* without saving.
        Called when the user cancels the registration flow.
        """
        self._reg_buffer.pop(name, None)

    def get_registration_buffer_size(self, name: str) -> int:
        """Return how many accepted frames have been buffered for *name*."""
        return len(self._reg_buffer.get(name, []))
