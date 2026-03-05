import numpy as np
import cv2
import torch
import collections
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.core.logging import get_logger, log_inference
from app.core.config import settings
from app.models.paddle_ocr import PaddleOCRDetector
from app.schemas.detection import (
    DetectionResult, TextDetectionResult, DepthEstimationResult,
    NavigationGuidanceResult, BoundingBox
)
from app.services.preprocessing import ImagePreprocessor
from app.services.postprocessing import DetectionPostprocessor

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
    
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
OCR_ENABLED = (
    os.environ.get("NOVA_OCR_ENABLED", "0").strip().lower()
    not in ("0", "false", "no")
)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8n_float16.tflite"
MIDAS_MODEL_PATH = BASE_DIR / "models" / "midas_v21_small_256.pt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# === RAISED CONF TO 0.55 + CENTER PROXIMITY FILTER + TTS DEBOUNCE ===
CONF_THRESHOLD  = 0.55  # raised from 0.25 — filter weak detections
NMS_THRESHOLD   = 0.45
MIN_AREA_RATIO  = 0.01
SMOOTHING_BUFFER = 5
DEPTH_SCALE_METERS = 4.0  # normalized depth 0–1 → approx 0–4 metres

# TTS debounce: do not repeat the same STOP/CAUTION command within this window
TTS_COOLDOWN_SECONDS = 4.0

# === CENTER PROXIMITY THRESHOLDS (MiDaS inverted: 0=far, 1=close) ===
# Tightened: center mean must be >= 0.38 (≈ arm's reach) to fire STOP/CAUTION
CENTER_DEPTH_GATE = 0.38   # mean closeness threshold for STOP / CAUTION
CENTER_MIN_GATE   = 0.33   # OR minimum closeness threshold

# === SIDE ZONE GATE ===
# MOVE_LEFT / MOVE_RIGHT only fire if the side object depth mean > this value (very close)
SIDE_DEPTH_GATE   = 0.30

# === CAUTIONARY CLASS SPLIT ===
# Only these classes contribute to risk calculation and voice commands.
# All other COCO classes (furniture, food, electronics) are visible in the
# heatmap / bounding boxes but IGNORED for guidance.
CAUTIONARY_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "horse", "cow", "elephant", "bear",
    "skateboard", "traffic light", "stop sign", "fire hydrant",
}

# Everything NOT in CAUTIONARY_CLASSES is ignored for guidance
# (chair, couch, potted plant, bed, dining table, tv, laptop, cell phone,
#  bottle, cup, bowl, book, clock, vase, refrigerator, toilet, sink, etc.)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ---------------------------------------------------------------------------
# Guidance text mapping
# ---------------------------------------------------------------------------
_COMMAND_GUIDANCE: Dict[str, str] = {
    "STOP":         "Stop immediately. Obstacle directly ahead.",
    "CAUTION":      "Caution. Slow down and proceed carefully.",
    "MOVE_LEFT":    "Move left. Obstacle on the right side.",
    "MOVE_RIGHT":   "Move right. Obstacle on the left side.",
    "PATH_CLEAR":   "Path clear. Proceed forward safely.",
    "INITIALIZING": "Initializing navigation sensors.",
}

_COMMAND_WARNINGS: Dict[str, List[str]] = {
    "STOP":         ["Immediate obstacle detected. Do not proceed."],
    "CAUTION":      ["Obstacle detected in path. Proceed with caution."],
    "MOVE_LEFT":    ["Obstacle on right — steer left."],
    "MOVE_RIGHT":   ["Obstacle on left — steer right."],
    "PATH_CLEAR":   [],
    "INITIALIZING": ["Navigation system initializing."],
}


# ---------------------------------------------------------------------------
# AssistiveNavigator  (inference-only — no UI / webcam / run() method)
# ---------------------------------------------------------------------------

class AssistiveNavigator:
    """
    Encapsulates YOLO + MiDaS inference and navigation decision logic.
    Designed to be instantiated once and reused across requests.
    """

    def __init__(self) -> None:
        self.yolo_interpreter   = None
        self.yolo_input_details  = None
        self.yolo_output_details = None
        self.midas_model        = None
        self.midas_transform    = None

        self.nav_buffer = collections.deque(maxlen=SMOOTHING_BUFFER)
        self.device     = torch.device("cpu")

        # Navigation geometry thresholds
        self.lower_fraction    = 0.52
        self.near_percentile   = 78
        # === TIGHTENED THRESHOLDS — require stronger signal to fire ===
        self.stop_threshold    = 0.70   # was 0.65
        self.caution_threshold = 0.60   # was 0.40 — stops chair/bottle triggering CAUTION
        self.side_bias         = 0.22
        self.ema_alpha         = 0.65

        # High-risk COCO classes — extra risk boost for these within CAUTIONARY set
        self.high_risk_classes = {
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "dog", "cat", "horse", "cow", "traffic light", "stop sign",
            "fire hydrant", "skateboard",
        }

        # EMA state
        self.prev_left_risk   = 0.0
        self.prev_center_risk = 0.0
        self.prev_right_risk  = 0.0

        # TTS debounce state
        self._last_tts_command: str  = ""
        self._last_tts_time: float   = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_yolo_model(self) -> None:
        logger.info(f"Loading YOLO TFLite model from: {YOLO_MODEL_PATH}")
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")
        
        interpreter = tflite.Interpreter(model_path=str(YOLO_MODEL_PATH))
        interpreter.allocate_tensors()
        
        # Let TFLite use the model's original tensor size to avoid CONCAT node errors
            
        # Guarantee allocation happens, whether resize succeeded or failed
        interpreter.allocate_tensors()
        self.yolo_interpreter   = interpreter
        self.yolo_input_details  = interpreter.get_input_details()
        self.yolo_output_details = interpreter.get_output_details()
        logger.info("YOLO TFLite model loaded successfully.")
        logger.debug(f"YOLO input  dtype: {self.yolo_input_details[0]['dtype']} "
                     f"shape: {self.yolo_input_details[0]['shape']}")
        logger.debug(f"YOLO output dtype: {self.yolo_output_details[0]['dtype']} "
                     f"shape: {self.yolo_output_details[0]['shape']}")

    def load_midas_model(self) -> None:
        logger.info(f"Loading MiDaS model from: {MIDAS_MODEL_PATH}")
        if not MIDAS_MODEL_PATH.exists():
            raise FileNotFoundError(f"MiDaS model not found at: {MIDAS_MODEL_PATH}")
        self.midas_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        state_dict = torch.load(str(MIDAS_MODEL_PATH), map_location=self.device)
        self.midas_model.load_state_dict(state_dict)
        self.midas_model.to(self.device)
        self.midas_model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.midas_transform = transforms.small_transform
        logger.info("MiDaS model loaded successfully.")
        print("[DEBUG] MiDaS model loaded successfully")

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

    def run_yolo(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO inference and return raw output tensor."""
        t0 = time.time()
        
        # Pull actual allocated dimensions (e.g., 416 or 640)
        input_shape = self.yolo_input_details[0]['shape']
        input_size_h, input_size_w = input_shape[1], input_shape[2]
        
        orig_h, orig_w = frame.shape[:2]
        
        # Letterbox resize
        scale = min(input_size_w / orig_w, input_size_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_w = int((input_size_w - new_w) / 2)
        pad_h = int((input_size_h - new_h) / 2)
        
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = np.full((input_size_h, input_size_w, 3), 114, dtype=np.uint8)
        img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
        
        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm  = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        input_detail = self.yolo_input_details[0]
        if input_detail["dtype"] == np.uint8:
            scale, zp = input_detail["quantization"]
            img_batch  = (img_batch / scale + zp).astype(np.uint8)
        elif input_detail["dtype"] == np.int8:
            scale, zp = input_detail["quantization"]
            img_batch  = (img_batch / scale + zp).astype(np.int8)

        self.yolo_interpreter.set_tensor(input_detail["index"], img_batch)
        self.yolo_interpreter.invoke()

        output_detail = self.yolo_output_details[0]
        output = self.yolo_interpreter.get_tensor(output_detail["index"])
        if output_detail["dtype"] in (np.uint8, np.int8):
            scale, zp = output_detail["quantization"]
            output = (output.astype(np.float32) - zp) * scale

        yolo_ms = (time.time() - t0) * 1000
        logger.info(
            f"[PERF][YOLO] inference={yolo_ms:.1f}ms "
            f"output_shape={output.shape}"
        )
        return output

    def decode_yolo_output(
        self, output: np.ndarray, frame_shape: tuple
    ) -> List[Dict]:
        """Decode raw YOLOv8 output into a list of detection dicts."""
        orig_h, orig_w = frame_shape[:2]
        predictions    = output[0]  # [N, 6] format from new TFLite export
        frame_area     = orig_w * orig_h

        boxes, confidences, class_ids = [], [], []

        # Validate we got 6 columns [x1, y1, x2, y2, score, class]
        if predictions.shape[-1] != 6:
            logger.error(f"[YOLO] Expected 6 columns, got {predictions.shape[-1]}")
            return []

        scores = predictions[:, 4]
        mask = scores > CONF_THRESHOLD
        filtered = predictions[mask]

        input_shape = self.yolo_input_details[0]['shape']
        input_size_w = input_shape[2]
        input_size_h = input_shape[1]

        for row in filtered:
            x1_raw, y1_raw, x2_raw, y2_raw = row[0], row[1], row[2], row[3]
            score  = float(row[4])
            cls_id = int(row[5])

            if x2_raw <= 1.5:
                # Normalized
                x1 = int(np.clip(x1_raw * orig_w, 0, orig_w))
                y1 = int(np.clip(y1_raw * orig_h, 0, orig_h))
                x2 = int(np.clip(x2_raw * orig_w, 0, orig_w))
                y2 = int(np.clip(y2_raw * orig_h, 0, orig_h))
            else:
                # Pixel space
                x1 = int(np.clip(x1_raw * orig_w / input_size_w, 0, orig_w))
                y1 = int(np.clip(y1_raw * orig_h / input_size_h, 0, orig_h))
                x2 = int(np.clip(x2_raw * orig_w / input_size_w, 0, orig_w))
                y2 = int(np.clip(y2_raw * orig_h / input_size_h, 0, orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1

            boxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(score))
            class_ids.append(cls_id)

        print(f"[DEBUG] Raw YOLO confidences (passed conf filter): {confidences}")

        if not boxes:
            print(f"[DEBUG] No boxes passed confidence threshold {CONF_THRESHOLD}")
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if indices is None or len(indices) == 0:
            print(f"[DEBUG] NMS returned 0 boxes")
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        else:
            indices = list(indices)

        detections = []
        for i in indices:
            x1, y1, bw, bh = boxes[i]
            x2 = min(orig_w - 1, x1 + bw)
            y2 = min(orig_h - 1, y1 + bh)
            x1 = max(0, x1)
            y1 = max(0, y1)
            cname = (
                COCO_CLASSES[class_ids[i]]
                if class_ids[i] < len(COCO_CLASSES)
                else "unknown"
            )
            detections.append(
                {
                    "box":        (x1, y1, x2, y2),
                    "confidence": confidences[i],
                    "class_id":   class_ids[i],
                    "class_name": cname,
                }
            )

        print(f"[DEBUG] Final detections after NMS: {len(detections)} — "
              f"{[d['class_name'] for d in detections]}")
        logger.debug(f"[YOLO] Final detections after NMS: {len(detections)}")
        return detections


    def run_midas(self, frame: np.ndarray) -> np.ndarray:
        """Run MiDaS depth estimation and return normalised depth map (same size as frame)."""
        t0 = time.time()
        orig_h, orig_w = frame.shape[:2]

        print("[DEBUG] Running MiDaS inference...")

        # Resize to 256x256 using letterbox to maintain aspect ratio
        input_size = 256
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_w_int = int((input_size - new_w) / 2)
        pad_h_int = int((input_size - new_h) / 2)
        
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        img[pad_h_int:pad_h_int+new_h, pad_w_int:pad_w_int+new_w] = img_resized
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_rgb = (img_rgb - mean) / std
        img_rgb = np.transpose(img_rgb, (2, 0, 1))

        input_batch = (
            torch.from_numpy(img_rgb).unsqueeze(0).float().to(self.device)
        )

        with torch.no_grad():
            prediction = self.midas_model(input_batch)
            
            # Crop padding out of prediction before spatial interpolation
            pred_cropped = prediction[:, pad_h_int : pad_h_int+new_h, pad_w_int : pad_w_int+new_w]
            
            # Interpolate cropped section back to original resolution
            prediction = torch.nn.functional.interpolate(
                pred_cropped.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        depth_range = d_max - d_min

        # Robust normalization with 1e-6 guard
        depth = (depth - d_min) / (depth_range + 1e-6)

        midas_ms = (time.time() - t0) * 1000

        print(f"[DEBUG] Depth shape: {depth.shape}")
        print(f"[DEBUG] Depth min: {depth.min():.4f}, max: {depth.max():.4f}, mean: {depth.mean():.4f}")

        if depth_range < 1e-4:
            print("[ERROR] Invalid depth map — depth_range nearly zero. MiDaS may be failing.")
            logger.warning(f"[MiDaS] DEPTH MAP FLAT — NOT WORKING (raw range={depth_range:.8f})")
        if depth.max() == 0.0:
            print("[ERROR] Invalid depth map — all zeros after normalization.")
            logger.warning("[MiDaS] depth_map is all zeros — depth estimation may have failed.")

        logger.info(
            f"[PERF][MiDaS] inference={midas_ms:.1f}ms "
            f"depth_shape={depth.shape} "
            f"min={depth.min():.4f} max={depth.max():.4f} mean={depth.mean():.4f}"
        )

        return depth

    # ------------------------------------------------------------------
    # Navigation logic
    # ------------------------------------------------------------------

    def _max_free_run(self, mask: np.ndarray) -> int:
        """Return the maximum contiguous horizontal free-space run in mask."""
        max_run = 0
        for row in mask:
            current = 0
            for val in row:
                if val:
                    current += 1
                    if current > max_run:
                        max_run = current
                else:
                    current = 0
        return max_run

    def compute_navigation(
        self,
        detections: List[Dict],
        depth_map: Optional[np.ndarray],
        frame_shape: tuple,
    ) -> Dict:
        """
        Zone-based navigation decision.
        Primary signal:  YOLO detections + depth-based distance per zone.
        Secondary signal: depth-map free-space check for path-clear veto.
        Returns a result dict with keys: command, risks, detections (annotated),
        suggested_direction.
        """
        orig_h, orig_w = frame_shape[:2]
        frame_area     = float(orig_w * orig_h)

        # ── Zone boundaries (pixel x) ─────────────────────────────────────
        zone_x1 = orig_w / 3.0       # LEFT  : 0 .. zone_x1
        zone_x2 = 2.0 * orig_w / 3.0 # CENTER: zone_x1 .. zone_x2
                                      # RIGHT : zone_x2 .. orig_w

        # ── Risk accumulators per zone ────────────────────────────────────
        risk_left   = 0.0
        risk_center = 0.0
        risk_right  = 0.0

        # ── Representative detections per zone (closest obstacle) ─────────
        closest: Dict[str, Optional[Dict]] = {"LEFT": None, "CENTER": None, "RIGHT": None}

        if not detections and depth_map is None:
            return {
                "command": "INITIALIZING",
                "risks": {"left": 0.0, "center": 0.0, "right": 0.0},
                "suggested_direction": None,
                "detections": [],
            }

        # ── PART 2+3+4: Zone assignment + distance + risk per detection ────
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            box_area   = float((x2 - x1) * (y2 - y1))
            area_ratio = box_area / frame_area

            # Zone from bbox center X
            cx = (x1 + x2) / 2.0
            if cx < zone_x1:
                zone = "LEFT"
            elif cx < zone_x2:
                zone = "CENTER"
            else:
                zone = "RIGHT"
            det["zone"] = zone

            # Distance from depth map ROI
            det["distance_m"] = None
            det["depth_norm"]  = None   # raw normalised depth (0=far, 1=close)
            if depth_map is not None:
                dy1 = max(0, min(y1, depth_map.shape[0] - 1))
                dy2 = max(0, min(y2, depth_map.shape[0]))
                dx1 = max(0, min(x1, depth_map.shape[1] - 1))
                dx2 = max(0, min(x2, depth_map.shape[1]))
                roi = depth_map[dy1:dy2, dx1:dx2]
                if roi.size > 0:
                    median_norm = float(np.median(roi))
                    det["distance_m"] = round(median_norm * DEPTH_SCALE_METERS, 2)
                    # Inverted: MiDaS 0=close → we want 1=close for the gate
                    det["depth_norm"] = round(1.0 - median_norm, 4)

            # === CAUTIONARY CLASS FILTER ===
            # Non-cautionary objects (chair, bottle, cell phone …) appear in the
            # heatmap and bounding boxes but DO NOT contribute to risk or voice.
            if det["class_name"] not in CAUTIONARY_CLASSES:
                logger.info(
                    f"[NAV][IGNORED] {det['class_name']} zone={zone} "
                    f"— not in CAUTIONARY_CLASSES, skipping risk accumulation"
                )
                continue  # skip risk accumulation; detection still drawn on frame

            # Risk from distance + area
            dist_m = det["distance_m"]
            if dist_m is not None:
                if dist_m < 1.5:
                    dist_risk = 1.0   # critical
                elif dist_m < 3.0:
                    dist_risk = 0.6   # medium
                else:
                    dist_risk = 0.2   # low
            else:
                # No depth — fall back to area-based risk estimate
                dist_risk = min(1.0, area_ratio * 8.0)

            class_boost = 1.5 if det["class_name"] in self.high_risk_classes else 1.0
            det_risk = min(1.0, (dist_risk * 0.70 + area_ratio * 0.30) * class_boost)

            dist_str = f"{dist_m}m" if dist_m is not None else "no-depth"
            logger.info(
                f"[NAV][DET] {det['class_name']} | zone={zone} "
                f"conf={det['confidence']:.2f} dist={dist_str} "
                f"area_ratio={area_ratio:.3f} det_risk={det_risk:.3f}"
            )

            # Accumulate zone risk (keep highest per zone)
            if zone == "LEFT":
                risk_left = max(risk_left, det_risk)
                if closest["LEFT"] is None or (
                    dist_m is not None
                    and (closest["LEFT"].get("distance_m") is None
                         or dist_m < closest["LEFT"]["distance_m"])
                ):
                    closest["LEFT"] = det
            elif zone == "CENTER":
                risk_center = max(risk_center, det_risk)
                if closest["CENTER"] is None or (
                    dist_m is not None
                    and (closest["CENTER"].get("distance_m") is None
                         or dist_m < closest["CENTER"]["distance_m"])
                ):
                    closest["CENTER"] = det
            else:
                risk_right = max(risk_right, det_risk)
                if closest["RIGHT"] is None or (
                    dist_m is not None
                    and (closest["RIGHT"].get("distance_m") is None
                         or dist_m < closest["RIGHT"]["distance_m"])
                ):
                    closest["RIGHT"] = det

        # ── Depth-only fallback when no YOLO detections ───────────────────
        debug_depth = "N/A"
        if depth_map is not None:
            lower_start = int(orig_h * 0.52)
            lh = depth_map[lower_start:, :]
            z0, z1 = lh.shape[1] // 3, (2 * lh.shape[1]) // 3
            mean_l = float(lh[:, :z0].mean())
            mean_c = float(lh[:, z0:z1].mean())
            mean_r = float(lh[:, z1:].mean())
            debug_depth = f"L:{mean_l:.3f} C:{mean_c:.3f} R:{mean_r:.3f}"

        if not detections and depth_map is not None:
            # 0=close, 1=far -> Invert so small depth = high risk
            risk_left   = max(0.0, min(1.0, (1.0 - mean_l) * 0.8))
            risk_center = max(0.0, min(1.0, (1.0 - mean_c) * 0.8))
            risk_right  = max(0.0, min(1.0, (1.0 - mean_r) * 0.8))
            logger.info(f"[NAV][DEPTH-ONLY] lower-half means L={mean_l:.3f} C={mean_c:.3f} R={mean_r:.3f}")
            print(f"[DEBUG MiDaS] means -> L={mean_l:.3f} C={mean_c:.3f} R={mean_r:.3f} | risks -> L={risk_left:.2f} C={risk_center:.2f} R={risk_right:.2f}")

        # ── EMA smoothing ─────────────────────────────────────────────────
        a = self.ema_alpha
        left_risk   = a * self.prev_left_risk   + (1.0 - a) * risk_left
        center_risk = a * self.prev_center_risk + (1.0 - a) * risk_center
        right_risk  = a * self.prev_right_risk  + (1.0 - a) * risk_right

        self.prev_left_risk   = left_risk
        self.prev_center_risk = center_risk
        self.prev_right_risk  = right_risk

        logger.info(
            f"[NAV] risks (EMA) — L={left_risk:.3f} C={center_risk:.3f} R={right_risk:.3f}"
        )

        # ── Compute center zone depth mean (for proximity gate) ──────────
        center_depth_mean = 1.0  # default = far (safe) — 0=far, 1=close (inverted MiDaS)
        center_depth_min  = 1.0
        if depth_map is not None:
            lower_start = int(orig_h * 0.52)
            lh = depth_map[lower_start:, :]
            z0, z1 = lh.shape[1] // 3, (2 * lh.shape[1]) // 3
            center_slice = lh[:, z0:z1]
            if center_slice.size > 0:
                # MiDaS: 0=close, 1=far → invert so 1=close
                center_depth_mean = float(1.0 - center_slice.mean())
                center_depth_min  = float(1.0 - center_slice.min())

        # ── PART 5: Decision logic ─────────────────────────────────────────
        suggested_direction = None
        # === CENTER PROXIMITY FILTER ===
        # STOP / CAUTION only trigger when the center zone is physically close.
        # Uses module-level constants: CENTER_DEPTH_GATE / CENTER_MIN_GATE
        center_is_close   = (
            center_depth_mean >= CENTER_DEPTH_GATE
            or center_depth_min >= CENTER_MIN_GATE
        )
        logger.info(
            f"[NAV] center_depth_mean={center_depth_mean:.3f} "
            f"center_depth_min={center_depth_min:.3f} center_is_close={center_is_close}"
        )

        # === SIDE ZONE DEPTH GATE ===
        # MOVE commands only fire when the closest side object
        # is genuinely close (depth_norm > SIDE_DEPTH_GATE = 0.30).
        left_close  = (
            closest["LEFT"]  is not None
            and (closest["LEFT"].get("depth_norm")  or 0.0) > SIDE_DEPTH_GATE
        )
        right_close = (
            closest["RIGHT"] is not None
            and (closest["RIGHT"].get("depth_norm") or 0.0) > SIDE_DEPTH_GATE
        )

        if center_risk >= self.stop_threshold and center_is_close:
            command = "STOP"
        elif (
            left_risk  >= self.stop_threshold
            and right_risk >= self.stop_threshold
            and center_is_close
        ):
            command = "STOP"
        elif center_risk >= self.caution_threshold and center_is_close:
            command = "CAUTION"
            suggested_direction = "LEFT" if left_risk <= right_risk else "RIGHT"
        elif left_risk >= self.stop_threshold and left_close:
            command = "MOVE_RIGHT"
        elif right_risk >= self.stop_threshold and right_close:
            command = "MOVE_LEFT"
        elif left_risk >= self.caution_threshold and left_close:
            command = "MOVE_RIGHT"
        elif right_risk >= self.caution_threshold and right_close:
            command = "MOVE_LEFT"
        else:
            command = "PATH_CLEAR"

        logger.info(
            f"[NAV] cmd={command} center_depth_mean={center_depth_mean:.3f} "
            f"center_is_close={center_is_close} suggested_dir={suggested_direction}"
        )

        return {
            "command":            command,
            "risks":              {
                "left":   round(left_risk,   4),
                "center": round(center_risk, 4),
                "right":  round(right_risk,  4),
            },
            "suggested_direction": suggested_direction,
            "closest":            closest,
            "debug_depth":        debug_depth if 'debug_depth' in locals() else "N/A",
            "center_depth_mean":  round(center_depth_mean, 4),
            "center_is_close":    center_is_close,   # passed to TTS gate
        }

    def apply_temporal_smoothing(self, command: str) -> str:
        """Majority-vote smoothing over a sliding buffer of recent commands."""
        self.nav_buffer.append(command)
        counter  = collections.Counter(self.nav_buffer)
        smoothed = counter.most_common(1)[0][0]
        logger.info(
            f"[SMOOTH] raw='{command}' smoothed='{smoothed}'"
        )
        return smoothed

    def should_speak(
        self,
        command: str,
        center_is_close: bool,
    ) -> bool:
        """
        === CENTER PROXIMITY FILTER + TTS DEBOUNCE ===
        Returns True only when the TTS layer should actually speak.

        Rules:
          1. Only STOP / CAUTION commands are gated — other commands
             (MOVE_LEFT, MOVE_RIGHT, PATH_CLEAR) pass through freely.
          2. For STOP / CAUTION: the center zone must be close
             (center_depth_mean >= CENTER_DEPTH_GATE OR
              center_depth_min  >= CENTER_MIN_GATE).
          3. 4-second cooldown: the same command is not repeated within
             TTS_COOLDOWN_SECONDS even if the proximity test passes.
        """
        if command not in ("STOP", "CAUTION"):
            return True  # non-alarm commands always pass

        # Gate 1: center must be physically close
        if not center_is_close:
            logger.info(
                f"[TTS][SUPPRESSED] cmd={command} — center not close enough "
                f"(CENTER_DEPTH_GATE={CENTER_DEPTH_GATE} CENTER_MIN_GATE={CENTER_MIN_GATE})"
            )
            return False

        # Gate 2: 4-second debounce on same command
        now = time.time()
        if (
            command == self._last_tts_command
            and (now - self._last_tts_time) < TTS_COOLDOWN_SECONDS
        ):
            remaining = TTS_COOLDOWN_SECONDS - (now - self._last_tts_time)
            logger.info(
                f"[TTS][DEBOUNCED] cmd={command} — "
                f"cooldown {remaining:.1f}s remaining"
            )
            return False

        # All gates passed — allow speech and record timestamp
        self._last_tts_command = command
        self._last_tts_time    = now
        logger.info(f"[TTS][SPEAK] cmd={command} center_is_close={center_is_close}")
        return True


# ---------------------------------------------------------------------------
# NavigationGuidanceService  (FastAPI-compatible singleton wrapper)
# ---------------------------------------------------------------------------

class NavigationGuidanceService:
    """Navigation guidance service using AssistiveNavigator (YOLO + MiDaS) + optional OCR."""

    def __init__(self) -> None:
        logger.info("=" * 60)
        logger.info("INITIALIZING NAVIGATION GUIDANCE SERVICE")
        logger.info("=" * 60)

        self.navigator    = AssistiveNavigator()
        self.ocr_detector  = None
        self.preprocessor  = ImagePreprocessor()
        self.postprocessor = DetectionPostprocessor()

        self._load_models()

        logger.info("Navigation Guidance Service Initialization Complete:")
        logger.info(
            f"  - YOLO (TFLite):  "
            f"{'✓ Loaded' if self.navigator.yolo_interpreter else '✗ Not available'}"
        )
        logger.info(
            f"  - MiDaS (PyTorch): "
            f"{'✓ Loaded' if self.navigator.midas_model else '✗ Not available'}"
        )
        logger.info(
            f"  - OCR Detector:   "
            f"{'✓ Loaded' if self.ocr_detector else '✗ Not available'}"
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Model loading and Warm-up
    # ------------------------------------------------------------------

    def _warmup_models(self) -> None:
        """Run a dummy forward pass to compile graphs and allocate memory."""
        logger.info("Running model warm-up inferences...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            if self.navigator.yolo_interpreter is not None:
                logger.info("  -> Warming up YOLO...")
                self.navigator.run_yolo(dummy_frame)
            
            if self.navigator.midas_model is not None:
                logger.info("  -> Warming up MiDaS...")
                self.navigator.run_midas(dummy_frame)
                
            logger.info("✓ Warm-up complete")
        except Exception as e:
            logger.warning(f"⚠ Warm-up failed (non-fatal): {e}")

    def _load_models(self) -> None:
        # YOLO
        logger.info("Loading YOLO TFLite detector...")
        try:
            self.navigator.load_yolo_model()
            logger.info("✓ YOLO TFLite detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ YOLO detector not available: {e}")
            logger.warning("  Navigation will continue without object detection")

        # MiDaS
        logger.info("Loading MiDaS depth estimator...")
        try:
            self.navigator.load_midas_model()
            logger.info("✓ MiDaS depth estimator loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ MiDaS depth estimator not available: {e}")
            logger.warning("  Navigation will continue without depth estimation")

        # OCR (optional)
        if OCR_ENABLED:
            logger.info("Loading PaddleOCR detector for text recognition...")
            try:
                self.ocr_detector = PaddleOCRDetector()
                logger.info("✓ PaddleOCR detector loaded successfully")
            except Exception as e:
                logger.warning(f"⚠ PaddleOCR detector not available: {e}")
                logger.warning("  Navigation will continue without text recognition")
        else:
            logger.info("OCR disabled via environment variable (NOVA_OCR_ENABLED=0)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_detections_to_schema(
        raw_detections: List[Dict], frame_shape: tuple
    ) -> List[DetectionResult]:
        """Convert raw detection dicts from AssistiveNavigator to DetectionResult schema."""
        orig_h, orig_w = frame_shape[:2]
        results = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["box"]
            bbox = BoundingBox(
                left   = x1 / orig_w,
                top    = y1 / orig_h,
                right  = x2 / orig_w,
                bottom = y2 / orig_h,
            )
            result = DetectionResult(
                label      = det["class_name"],
                confidence = round(det["confidence"], 4),
                bbox       = bbox,
            )
            results.append(result)
        return results

    @staticmethod
    def _build_depth_result(
        depth_map: Optional[np.ndarray],
        inference_time_ms: float = 0.0,
    ) -> Optional[DepthEstimationResult]:
        """Wrap a numpy depth map into the DepthEstimationResult schema with all required fields."""
        if depth_map is None:
            return None
        d_min  = float(depth_map.min())
        d_max  = float(depth_map.max())
        d_mean = float(depth_map.mean())
        print(f"[DEBUG] depth_analysis: min={d_min:.4f} max={d_max:.4f} mean={d_mean:.4f}")
        return DepthEstimationResult(
            min_depth        = d_min,
            max_depth        = d_max,
            mean_depth       = d_mean,
            inference_time_ms = inference_time_ms,
        )

    @staticmethod
    def _guidance_from_command(
        command: str,
        suggested_direction: Optional[str],
        detections: Optional[List[Dict]] = None,
        nav_result: Optional[Dict] = None,
    ) -> str:
        """
        Generate human-readable guidance WITHOUT numeric distances.
        Proximity is expressed as a natural phrase so TTS is concise.
        Numeric values are still stored in depth_analysis for visual display.
        """
        closest = (nav_result or {}).get("closest", {})

        # === NO DISTANCE NUMBERS IN TTS ===
        # Map normalised depth (0=far,1=close via inversion) to a short phrase.
        # dist_m here is the scaled value from DEPTH_SCALE_METERS (0‒4 m).
        def _proximity_phrase(dist_m: Optional[float]) -> str:
            if dist_m is None:
                return "nearby"
            if dist_m < 1.0:
                return "very close"
            if dist_m < 2.0:
                return "nearby"
            return "ahead"

        def _det_str(zone: str) -> str:
            """Return 'label [proximity]' — no numeric distance in spoken text."""
            det = (closest or {}).get(zone)
            if det is None:
                return "obstacle"
            label = det.get("class_name", "obstacle")
            phrase = _proximity_phrase(det.get("distance_m"))
            return f"{label} {phrase}"

        if command == "STOP":
            det_s = _det_str("CENTER") or _det_str("LEFT") or _det_str("RIGHT")
            return f"{det_s.capitalize()} directly ahead. Stop."

        elif command == "CAUTION":
            direction = f" Move {suggested_direction.lower()}." if suggested_direction else ""
            det_s = _det_str("CENTER")
            return f"{det_s.capitalize()} detected. Slow down.{direction}"

        elif command == "MOVE_RIGHT":
            det_s = _det_str("LEFT")
            return f"{det_s.capitalize()} on your left. Move right."

        elif command == "MOVE_LEFT":
            det_s = _det_str("RIGHT")
            return f"{det_s.capitalize()} on your right. Move left."

        elif command == "PATH_CLEAR":
            return "Path clear."

        else:
            return "Initializing navigation."

    @staticmethod
    def _warnings_from_command(command: str) -> List[str]:
        """Map a navigation command to safety warning strings."""
        return list(_COMMAND_WARNINGS.get(command, ["Proceed with caution."]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_navigation_guidance(
        self, image_data: bytes, image_type: str
    ) -> NavigationGuidanceResult:
        """Generate navigation guidance from an input image."""
        start_time = time.time()

        try:
            # 1. Load and validate image
            image = self.preprocessor.validate_and_load_image(image_data, image_type)

            # 2. Convert to BGR numpy array
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            logger.debug(f"Navigation pipeline — image shape: {image_np.shape}")

            # 3. YOLO inference
            raw_yolo_output = None
            raw_detections  = []
            t_yolo = time.time()
            if self.navigator.yolo_interpreter is not None:
                raw_yolo_output = self.navigator.run_yolo(image_np)
                raw_detections  = self.navigator.decode_yolo_output(
                    raw_yolo_output, image_np.shape
                )
            yolo_total_ms = (time.time() - t_yolo) * 1000
            logger.info(f"[PERF][YOLO] total (incl. decode)={yolo_total_ms:.1f}ms detections={len(raw_detections)}")

            # 4. MiDaS depth estimation
            depth_map = None
            t_midas = time.time()
            if self.navigator.midas_model is not None:
                depth_map = self.navigator.run_midas(image_np)
            midas_total_ms = (time.time() - t_midas) * 1000
            logger.info(f"[PERF][MiDaS] total={midas_total_ms:.1f}ms available={depth_map is not None}")

            # 5. OCR (optional)
            ocr_results = []
            if self.ocr_detector and OCR_ENABLED:
                logger.debug("Running OCR detector")
                ocr_results = self.ocr_detector.detect_text(image_np)
                ocr_results = self.postprocessor.filter_text_detections(
                    ocr_results, confidence_threshold=0.3
                )
                logger.debug(f"OCR text regions: {len(ocr_results)}")
            else:
                logger.debug(
                    f"OCR skipped "
                    f"(present={self.ocr_detector is not None}, enabled={OCR_ENABLED})"
                )

            # 6. Navigation decision
            nav_result = self.navigator.compute_navigation(
                raw_detections, depth_map, image_np.shape
            )
            raw_command   = nav_result["command"]
            # ── Issue 5: guidance MUST use final smoothed command ─────────
            final_command = self.navigator.apply_temporal_smoothing(raw_command)
            suggested_dir = nav_result.get("suggested_direction")

            # === TTS DEBOUNCE — center-zone + proximity + 4-second cooldown ===
            center_is_close = nav_result.get("center_is_close", False)
            tts_allowed     = self.navigator.should_speak(final_command, center_is_close)
            logger.info(
                f"[TTS] final_command={final_command} "
                f"center_is_close={center_is_close} tts_allowed={tts_allowed}"
            )

            logger.info(
                f"[NAV] raw_command={raw_command} "
                f"final_command(smoothed)={final_command} "
                f"risks={nav_result['risks']}"
            )

            # 7. Build schema-compatible outputs
            schema_detections = self._raw_detections_to_schema(
                raw_detections, image_np.shape
            )
            depth_result   = self._build_depth_result(depth_map, midas_total_ms)
            # Use nav_result["closest"] for zone-aware guidance
            guidance_text  = self._guidance_from_command(
                final_command, suggested_dir,
                nav_result=nav_result,
            )
            safety_warnings = self._warnings_from_command(final_command)

            logger.info(f"[NAV] guidance_text='{guidance_text}'")
            total_time = time.time() - start_time
            logger.info(
                f"[PERF] ═══ PIPELINE SUMMARY ═══ "
                f"YOLO={yolo_total_ms:.0f}ms "
                f"MiDaS={midas_total_ms:.0f}ms "
                f"TOTAL={total_time * 1000:.0f}ms "
                f"cmd={final_command}"
            )
            log_inference(
                "Navigation Guidance",
                total_time,
                image_np.shape,
                len(schema_detections),
            )

            # 8. Generate Visual Debug Frame (Heatmap + Overlay)
            debug_b64 = None
            if depth_map is not None:
                import base64
                
                # Create colormap from normalized depth (0..1)
                depth_u8 = (depth_map * 255).astype(np.uint8)
                # Invert so far away (1.0) is blue/cold and close (0.0) is red/hot
                colored_depth = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_JET)
                
                # Blend with original image
                blended = cv2.addWeighted(image_np, 0.5, colored_depth, 0.5, 0)
                
                # Draw YOLO boxes
                for det in raw_detections:
                    x1, y1, x2, y2 = det["box"]
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(blended, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw Command text
                cv2.putText(blended, final_command, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Encode to JPEG base64
                ret, buffer = cv2.imencode('.jpg', blended)
                if ret:
                    debug_b64 = base64.b64encode(buffer).decode('utf-8')

            # 9. Construct and return result
            # safety_warnings is suppressed (empty) when TTS debounce blocks speech,
            # so the Flutter TTS layer simply checks len(safety_warnings) > 0 to decide
            # whether to speak. guidance_text is always populated for display.
            tts_warnings = safety_warnings if tts_allowed else []
            result = NavigationGuidanceResult(
                obstacles       = schema_detections,
                text_signs      = ocr_results,
                depth_analysis  = depth_result,
                guidance        = guidance_text,
                safety_warnings = tts_warnings,
                inference_time_ms = round(total_time * 1000, 2),
                debug_frame_base64 = debug_b64,
            )

            logger.debug(
                f"Navigation summary: obstacles={len(result.obstacles)} "
                f"signs={len(result.text_signs)} warnings={len(result.safety_warnings)}"
            )
            logger.info(f"Navigation guidance completed: {result.inference_time_ms}ms")

            return result

        except Exception as e:
            logger.error(f"Navigation guidance failed: {e}")
            raise

    def get_model_status(self) -> Dict[str, Any]:
        """Return status of all loaded models."""
        yolo_ok  = self.navigator.yolo_interpreter is not None
        midas_ok = self.navigator.midas_model is not None
        ocr_ok   = self.ocr_detector is not None

        status = {
            "yolo_detector":   yolo_ok,
            "depth_estimator": midas_ok,
            "ocr_detector":    ocr_ok,
            "preprocessor":    True,
            "postprocessor":   True,
        }

        model_info: Dict[str, Any] = {}
        if yolo_ok:
            model_info["yolo"] = {
                "model_path": str(YOLO_MODEL_PATH),
                "input_shape": self.navigator.yolo_input_details[0]["shape"].tolist(),
            }
        if midas_ok:
            model_info["depth"] = {
                "model_path": str(MIDAS_MODEL_PATH),
                "device":     str(self.navigator.device),
            }
        if ocr_ok:
            model_info["ocr"] = self.ocr_detector.get_model_info()

        return {
            "status":           status,
            "model_info":       model_info,
            "all_models_loaded": all(status.values()),
        }