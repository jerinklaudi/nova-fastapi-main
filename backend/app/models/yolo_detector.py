
import numpy as np
from typing import List, Optional
import time
import os
import logging
import cv2

# Try importing TFLite with compatibility fallbacks.
Interpreter = None

try:
    import tensorflow.lite as tflite
    Interpreter = getattr(tflite, "Interpreter", None)
except Exception:
    Interpreter = None

if Interpreter is None:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except Exception:
        try:
            import tensorflow as tf
            lite_module = getattr(tf, "lite", None)
            if lite_module is not None:
                Interpreter = getattr(lite_module, "Interpreter", None)
        except Exception:
            Interpreter = None

from app.core.config import settings
from app.schemas.detection import DetectionResult, BoundingBox

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLO object detection model wrapper using TFLite."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.output_details = None
        self.input_details = None
        self.interpreter = None
        self.tflite_loaded = False
        self.ultra_model = None
        self.ultra_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "models",
            "yolov8n.pt",
        )
        self.model_loaded = False
        self.labels = self._get_coco_labels()
        
        # NMS settings
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        
        self._load_model()
    
    def _get_coco_labels(self) -> List[str]:
        """Get COCO dataset labels."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def _load_model(self) -> None:
        """Load the YOLO TFLite model."""
        logger.info("NOVA Object Detection Module - Initializing...")

        # 1) Try TFLite first
        if Interpreter is None:
            logger.warning("TFLite Interpreter unavailable; will try Ultralytics fallback.")
        else:
            if not os.path.exists(self.model_path):
                possible_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "models",
                    "yolov5s-fp16.tflite",
                )
                if os.path.exists(possible_path):
                    logger.info(f"Found TFLite model at fallback path: {possible_path}")
                    self.model_path = possible_path

            if os.path.exists(self.model_path):
                try:
                    logger.info(f"Loading TFLite model: {self.model_path}")
                    self.interpreter = Interpreter(model_path=self.model_path)
                    self.interpreter.allocate_tensors()

                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.tflite_loaded = True
                    logger.info("✓ TFLite model loaded successfully")
                    logger.info(f"  Input: {self.input_details[0]['shape']}")
                    logger.info(f"  Output: {self.output_details[0]['shape']}")
                except Exception as e:
                    logger.warning(f"TFLite model load failed: {str(e)}")
            else:
                logger.warning(f"TFLite model not found at: {self.model_path}")

        # 2) Ultralytics YOLOv8 fallback
        if os.path.exists(self.ultra_model_path):
            try:
                from ultralytics import YOLO

                self.ultra_model = YOLO(self.ultra_model_path)
                logger.info(f"✓ Ultralytics fallback loaded: {self.ultra_model_path}")
            except Exception as e:
                logger.warning(f"Ultralytics fallback unavailable: {e}")

        self.model_loaded = self.tflite_loaded or (self.ultra_model is not None)
        if not self.model_loaded:
            logger.error("No object detector backend available.")

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Perform object detection using TFLite."""
        start_time = time.perf_counter()
        
        if not self.model_loaded:
            return []
            
        try:
            tflite_detections: List[DetectionResult] = []

            if self.tflite_loaded:
            # 1. Preprocess
                input_shape = self.input_details[0]['shape']
                target_h, target_w = input_shape[1], input_shape[2]

                # Accept both HWC images and already-batched NHWC tensors.
                if image.ndim == 4 and image.shape[0] == 1:
                    image = image[0]

                if image.dtype in (np.float16, np.float32, np.float64):
                    if image.max() <= 1.0:
                        image = (image * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)

                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
                # Resize
                # OpenCV expects (w, h)
                img_resized = cv2.resize(image, (target_w, target_h))
            
                # Convert BGR to RGB for model input
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
                # Normalize and add batch dimension
                input_data = img_rgb.astype(np.float32) / 255.0
                input_data = np.expand_dims(input_data, axis=0)
            
                # 2. Inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                logger.info(f"YOLO TFLite output shape: {getattr(output_data, 'shape', None)}")

                # Dequantize if output tensor is quantized.
                quant = self.output_details[0].get('quantization', (0.0, 0))
                if quant and len(quant) == 2:
                    scale, zero_point = quant
                    if scale not in (0, 0.0):
                        output_data = (output_data.astype(np.float32) - float(zero_point)) * float(scale)
            
                # post inference timing
                inference_time = time.perf_counter() - start_time
            
                # 3. Post-process
                # Common layouts are [1, N, 85] and [1, 85, N].
                pred = output_data[0] if output_data.ndim == 3 else output_data
                if pred.ndim != 2:
                    logger.warning(f"Unexpected YOLO output shape: {output_data.shape}")
                    pred = np.empty((0, 0), dtype=np.float32)
                elif pred.shape[0] < pred.shape[1]:
                    pred = pred.T

                if pred.shape[1] < 5:
                    logger.warning(f"Unexpected YOLO output feature dimension: {pred.shape}")
                    pred = np.empty((0, 0), dtype=np.float32)
            
                if pred.size > 0:
                    # Split into boxes and scores
                    boxes = pred[:, :4]  # xywh

                    # Scores = obj_conf * class_conf for standard YOLO heads.
                    if pred.shape[1] > 5:
                        class_scores = pred[:, 5:]
                        scores = pred[:, 4:5] * class_scores
                        class_ids = np.argmax(scores, axis=1)
                        max_scores = np.max(scores, axis=1)
                    else:
                        class_ids = np.zeros((pred.shape[0],), dtype=np.int32)
                        max_scores = pred[:, 4]
            
                    mask = max_scores >= self.conf_thres
                    logger.info(
                        "YOLO TFLite candidates: total=%d, above_conf(%.2f)=%d",
                        pred.shape[0],
                        self.conf_thres,
                        int(np.sum(mask)),
                    )

                    filtered_boxes = boxes[mask]
                    filtered_scores = max_scores[mask]
                    filtered_class_ids = class_ids[mask]
            
                    if len(filtered_boxes) > 0:
                        # Normalize boxes when model returns absolute coordinates.
                        if np.max(filtered_boxes) > 2.0:
                            filtered_boxes = filtered_boxes.copy().astype(np.float32)
                            filtered_boxes[:, 0] /= float(target_w)
                            filtered_boxes[:, 2] /= float(target_w)
                            filtered_boxes[:, 1] /= float(target_h)
                            filtered_boxes[:, 3] /= float(target_h)
                
                        # Convert to xyxy for NMS
                        boxes_xyxy = self.xywh2xyxy(filtered_boxes)

                        # NMS using OpenCV to avoid requiring torchvision at runtime.
                        nms_boxes = []
                        for box in boxes_xyxy:
                            x1, y1, x2, y2 = box
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            nms_boxes.append([float(x1), float(y1), float(w), float(h)])
                        indices = cv2.dnn.NMSBoxes(
                            nms_boxes,
                            filtered_scores.tolist(),
                            self.conf_thres,
                            self.iou_thres,
                        )
                        kept = int(np.array(indices).reshape(-1).size) if len(indices) > 0 else 0
                        logger.info(
                            "YOLO TFLite NMS kept=%d (iou=%.2f)",
                            kept,
                            self.iou_thres,
                        )
                        if len(indices) > 0:
                            indices = np.array(indices).reshape(-1)

                            # Create DetectionResult objects
                            for idx in indices:
                                idx = int(idx)
                                box = boxes_xyxy[idx]  # [x1, y1, x2, y2] normalized
                                score = float(filtered_scores[idx])
                                cls_id = int(filtered_class_ids[idx])
                                label = self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id)

                                # Clip coordinates to 0-1
                                x1 = max(0.0, min(1.0, float(box[0])))
                                y1 = max(0.0, min(1.0, float(box[1])))
                                x2 = max(0.0, min(1.0, float(box[2])))
                                y2 = max(0.0, min(1.0, float(box[3])))

                                tflite_detections.append(
                                    DetectionResult(
                                        label=label,
                                        confidence=score,
                                        bbox=BoundingBox(
                                            left=x1,
                                            top=y1,
                                            right=x2,
                                            bottom=y2,
                                        ),
                                    )
                                )

                if tflite_detections:
                    logger.info(f"TFLite Detection: {len(tflite_detections)} objects in {inference_time:.3f}s")
                    return tflite_detections
                logger.info("TFLite Detection produced 0 objects; trying Ultralytics fallback if available")

            # Fallback to Ultralytics if TFLite produced no detections.
            if self.ultra_model is not None:
                ultra_start = time.perf_counter()
                logger.info("Running Ultralytics fallback detection")
                results = self.ultra_model.predict(image, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
                detections: List[DetectionResult] = []
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and boxes.xyxy is not None:
                        h, w = image.shape[:2]
                        for b in boxes:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            cls_id = int(b.cls[0].item()) if b.cls is not None else 0
                            score = float(b.conf[0].item()) if b.conf is not None else 0.0
                            label = self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id)
                            detections.append(
                                DetectionResult(
                                    label=label,
                                    confidence=score,
                                    bbox=BoundingBox(
                                        left=max(0.0, min(1.0, x1 / w)),
                                        top=max(0.0, min(1.0, y1 / h)),
                                        right=max(0.0, min(1.0, x2 / w)),
                                        bottom=max(0.0, min(1.0, y2 / h)),
                                    ),
                                )
                            )

                logger.info(
                    f"Ultralytics fallback detection: {len(detections)} objects in "
                    f"{(time.perf_counter() - ultra_start):.3f}s"
                )
                return detections

            return []
            
        except Exception as e:
            logger.error(f"TFLite inference failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "YOLOv5s TFLite + YOLOv8 fallback",
            "model_size": "~14 MB",
            "device": "CPU",
            "num_classes": len(self.labels),
            "status": "loaded" if self.model_loaded else "fallback",
            "tflite_loaded": self.tflite_loaded,
            "ultralytics_loaded": self.ultra_model is not None,
            "confidence_threshold": self.conf_thres
        }
