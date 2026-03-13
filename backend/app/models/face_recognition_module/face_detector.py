"""
face_detector.py
────────────────
YOLOv8n-face ONNX detector.

Ported directly from the project's uploaded standalone face_detector.py
(v3 / the version attached in this session), adapted as a backend module.

Requires:  backend/models/yolov8n-face.onnx

Public API
──────────
detector = FaceDetector(model_path, conf_threshold, nms_threshold)
boxes, kps_list, scores = detector.detect(bgr_frame)

align_face(frame, landmarks, output_size=112) → np.ndarray
    ArcFace-aligned face crop ready for SFace.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Default model path ────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "models", "yolov8n-face.onnx")
)

# ── ArcFace canonical 5-point targets (112 × 112) ─────────────────────────────
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

_MIN_AREA = 1400   # discard boxes smaller than this (pixels²)


# ── align_face (standalone helper, imported by face_recognizer and inference) ──

def align_face(
    frame: np.ndarray,
    landmarks: list,
    output_size: int = 112,
) -> np.ndarray:
    """
    Warp *frame* so that the 5 given landmarks land on ArcFace canonical
    positions.  Returns uint8 BGR (output_size × output_size × 3).
    Returns a zeros image if landmarks are degenerate.
    """
    src_pts = np.array(landmarks, dtype=np.float32).reshape(5, 2)
    
    # Check for totally zeroed landmarks
    if np.all(src_pts == 0):
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    dst_pts   = _ARCFACE_DST * (output_size / 112.0)
    transform = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)[0]

    if transform is None:
        h, w = frame.shape[:2]
        x1 = max(0, int(src_pts[:, 0].min()))
        y1 = max(0, int(src_pts[:, 1].min()))
        x2 = min(w, int(src_pts[:, 0].max()))
        y2 = min(h, int(src_pts[:, 1].max()))
        
        # Ensure box has width/height before slicing
        if x2 <= x1 or y2 <= y1:
            return np.zeros((output_size, output_size, 3), dtype=np.uint8)
            
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((output_size, output_size, 3), dtype=np.uint8)
        return cv2.resize(crop, (output_size, output_size))

    return cv2.warpAffine(
        frame, transform, (output_size, output_size), flags=cv2.INTER_LINEAR
    )


# ── FaceDetector ──────────────────────────────────────────────────────────────

class FaceDetector:
    """
    YOLOv8n-face ONNX detector.

    Returns bounding boxes (x1,y1,x2,y2) and 5-pt facial landmarks in the
    same format as the standalone detector so all downstream code works
    without changes.

    Bug-fixes carried over from the standalone version:
      • Synthesises anatomically-plausible landmarks when the model output
        has no landmark columns (prevents align_face from getting all-zeros).
      • Shifts boxes slightly downward to correct YOLOv8-face upward bias.
      • Temporal EMA smoothing on box coordinates.
    """

    SHIFT_DOWN = 0.08   # fraction of box-height to shift down

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        nms_threshold: float  = 0.45,
    ) -> None:
        model_path = model_path or settings.FACE_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold
        self.input_h        = 640
        self.input_w        = 640
        self._prev_boxes: list = []

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLOv8n-face model not found: '{model_path}'. "
                "Place yolov8n-face.onnx in backend/models/."
            )

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "FaceDetector: loaded '%s' | conf=%.2f nms=%.2f",
            model_path, conf_threshold, nms_threshold,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers (verbatim from standalone)
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple]:
        orig_h, orig_w = frame.shape[:2]
        scale = 640.0 / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded  = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = (
            cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            .transpose(2, 0, 1)[np.newaxis]
            .astype(np.float32)
            / 255.0
        )
        return blob, scale, (orig_h, orig_w)

    @staticmethod
    def _synthesise_landmarks(boxes: np.ndarray) -> np.ndarray:
        """Generate 5-pt landmarks from bounding boxes when model gives none."""
        n   = len(boxes)
        kps = np.zeros((n, 5, 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w, h = float(x2 - x1), float(y2 - y1)
            kps[i] = [
                [x1 + 0.30 * w, y1 + 0.35 * h],   # left eye
                [x1 + 0.70 * w, y1 + 0.35 * h],   # right eye
                [x1 + 0.50 * w, y1 + 0.55 * h],   # nose tip
                [x1 + 0.35 * w, y1 + 0.75 * h],   # left mouth
                [x1 + 0.65 * w, y1 + 0.75 * h],   # right mouth
            ]
        return kps

    def _postprocess(
        self, outputs, scale: float, orig_hw: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        orig_h, orig_w = orig_hw
        empty_b = np.empty((0, 4), dtype=np.int32)
        empty_k = np.empty((0, 5, 2), dtype=np.float32)

        if not outputs or len(outputs) == 0:
            return empty_b, empty_k, []

        # YOLOv8-face models often output (1, dims, anchors) e.g. (1, 80, 8400)
        # or (1, anchors, dims) e.g. (1, 8400, 80)
        output = outputs[0]
        if output.ndim == 3:
            output = output[0]
        
        # Transpose if dims < anchors (standard YOLOv8 format)
        pred = output if output.shape[0] > output.shape[1] else output.T
        feat_dim = pred.shape[1]

        if feat_dim < 5:
            logger.warning(f"FaceDetector: unexpected feature dimension {feat_dim}")
            return empty_b, empty_k, []

        # Column indices: 0:4 is box, 4 is confidence
        confs = pred[:, 4]
        mask  = confs >= self.conf_threshold
        if not mask.any():
            return empty_b, empty_k, []

        pred, confs = pred[mask], confs[mask]
        n = len(pred)

        # Decode boxes from cx,cy,w,h to pixel boundaries
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes_640 = np.column_stack(
            (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        )

        # Detect landmarks if present (starting at index 5)
        # Face models have landmarks, generic YOLOv8 models usually don't.
        kps_cols  = pred[:, 5:] if feat_dim > 5 else np.empty((n, 0))
        kp_count  = kps_cols.shape[1]
        has_real  = False

        if kp_count >= 15:
            # Format: [x,y,conf] x 5
            kps_raw, has_real = kps_cols[:, :15].reshape(n, 5, 3)[:, :, :2], True
        elif kp_count >= 10:
            # Format: [x,y] x 5
            kps_raw, has_real = kps_cols[:, :10].reshape(n, 5, 2), True
        else:
            kps_raw = np.zeros((n, 5, 2), dtype=np.float32)
            logger.debug("FaceDetector: no landmark columns — will synthesise from boxes")

        indices = cv2.dnn.NMSBoxes(
            boxes_640.tolist(), confs.tolist(),
            self.conf_threshold, self.nms_threshold,
        )
        if len(indices) == 0:
            return empty_b, empty_k, []

        indices   = np.array(indices).flatten()
        boxes_640 = boxes_640[indices]
        kps_raw   = kps_raw[indices]
        scores    = confs[indices].tolist()

        inv        = 1.0 / scale
        boxes_orig = (boxes_640 * inv).astype(np.int32)
        kps_orig   = (kps_raw  * inv).astype(np.float32)

        if not has_real:
            kps_orig = self._synthesise_landmarks(boxes_orig)

        return boxes_orig, kps_orig, scores

    @staticmethod
    def _shift_boxes(
        boxes: np.ndarray, orig_h: int, orig_w: int, shift_down: float
    ) -> np.ndarray:
        b      = boxes.copy().astype(np.float32)
        offset = (b[:, 3] - b[:, 1]) * shift_down
        b[:, 1] += offset
        b[:, 3] += offset
        b[:, 0] = np.clip(b[:, 0], 0, orig_w)
        b[:, 1] = np.clip(b[:, 1], 0, orig_h)
        b[:, 2] = np.clip(b[:, 2], 0, orig_w)
        b[:, 3] = np.clip(b[:, 3], 0, orig_h)
        return b.astype(np.int32)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[List[int]], List[List[List[float]]], List[float]]:
        """
        Main detection logic returning pixel coordinates and raw landmarks.
        """
        try:
            blob, scale, (orig_h, orig_w) = self._preprocess(frame)
            
            # Run inference
            outputs = self._session.run(None, {self._input_name: blob})
            logger.debug("FaceDetector: output shapes: %s", [o.shape for o in outputs])
            
            boxes, kps, scores = self._postprocess(outputs, scale, (orig_h, orig_w))

            if len(boxes) == 0:
                self._prev_boxes = []
                return [], [], []

            boxes = self._shift_boxes(boxes, orig_h, orig_w, self.SHIFT_DOWN)

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep  = areas > _MIN_AREA
            boxes  = boxes[keep]
            kps    = kps[keep]
            scores = [s for s, k in zip(scores, keep) if k]

            final_boxes = boxes.tolist()
            final_kps   = kps.tolist()

            # Temporal EMA smoothing
            if self._prev_boxes and len(final_boxes) == len(self._prev_boxes):
                alpha = 0.6
                final_boxes = [
                    [int(alpha * c + (1 - alpha) * p) for c, p in zip(cur, prv)]
                    for cur, prv in zip(final_boxes, self._prev_boxes)
                ]

            self._prev_boxes = final_boxes
            return final_boxes, final_kps, scores
            
        except Exception as e:
            logger.error("FaceDetector.detect error: %s", e, exc_info=True)
            raise

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Compatibility wrapper for existing API endpoints.
        Internally calls modular detect() and formats results as 
        normalized dictionaries with 'bbox' and 'confidence'.
        Now also includes 'kps' and 'raw_box' for high-quality recognition.
        """
        boxes, kps, scores = self.detect(image)
        h, w = image.shape[:2]
        
        results = []
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            kp = kps[i]
            
            results.append({
                'bbox': [
                    float(box[0]) / w,
                    float(box[1]) / h,
                    float(box[2]) / w,
                    float(box[3]) / h
                ],
                'confidence': float(score),
                'kps': kp,        # Raw pixel coordinates for alignment
                'raw_box': box    # Raw pixel coordinates [x1, y1, x2, y2]
            })
        return results

    def get_model_info(self) -> dict:
        return {
            "model_type":      "YOLOv8n-face (ONNX)",
            "conf_threshold":  self.conf_threshold,
            "nms_threshold":   self.nms_threshold,
            "input_size":      f"{self.input_w}×{self.input_h}",
        }
