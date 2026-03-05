
import numpy as np
import torch
import torchvision
from typing import List, Optional
import time
import os
import logging
import cv2

# Try importing TFLite
try:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
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
        logger.info("NOVA Object Detection Module (TFLite) - Initializing...")
        
        if Interpreter is None:
            logger.error("TFLite Interpreter not available. Please install tensorflow or tflite-runtime.")
            self.model_loaded = False
            return

        if not os.path.exists(self.model_path):
            logger.error(f"TFLite model not found at: {self.model_path}")
            # Try finding it in sibling directory if running from backend
            possible_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "yolov5s-fp16.tflite")
            if os.path.exists(possible_path):
                 logger.info(f"Found model at fallback path: {possible_path}")
                 self.model_path = possible_path
            else:
                 logger.warning("Object detection will return simulated results.")
                 self.model_loaded = False
                 return

        try:
            logger.info(f"Loading TFLite model: {self.model_path}")
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info("✓ TFLite model loaded successfully")
            logger.info(f"  Input: {self.input_details[0]['shape']}")
            logger.info(f"  Output: {self.output_details[0]['shape']}")
            
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {str(e)}")
            self.model_loaded = False

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
            # 1. Preprocess
            input_shape = self.input_details[0]['shape']
            target_h, target_w = input_shape[1], input_shape[2]
            
            # Resize
            # OpenCV expects (w, h)
            img_resized = cv2.resize(image, (target_w, target_h))
            
            # Convert BGR to RGB (if input is BGR)
            # Assuming input 'image' is BGR (standard OpenCV)
            # But duplicate validation in inference.py/preprocessing usually ensures RGB? 
            # Looking at previous code: "cvtColor(image, cv2.COLOR_BGR2RGB)" was used.
            # We'll assume input is BGR/RGB based on channel count, but safely convert to RGB.
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize and add batch dimension
            input_data = img_rgb.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            
            # 2. Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # post inference timing
            inference_time = time.perf_counter() - start_time
            
            # 3. Post-process
            # output_data shape: [1, 25200, 85]
            pred = output_data[0]
            
            # Split into boxes and scores
            boxes = pred[:, :4]  # xywh (normalized 0-1)
            # Scores = obj_conf * class_conf
            scores = pred[:, 4:5] * pred[:, 5:]
            
            # Filter by confidence
            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)
            
            mask = max_scores >= self.conf_thres
            
            filtered_boxes = boxes[mask]
            filtered_scores = max_scores[mask]
            filtered_class_ids = class_ids[mask]
            
            if len(filtered_boxes) == 0:
                logger.debug(f"No detections after thresholding ({inference_time:.3f}s)")
                return []
                
            # Convert to xyxy for NMS
            boxes_xyxy = self.xywh2xyxy(filtered_boxes)
            
            # NMS using torchvision
            # Convert to tensors
            boxes_tensor = torch.from_numpy(boxes_xyxy)
            scores_tensor = torch.from_numpy(filtered_scores)
            
            indices = torchvision.ops.nms(boxes_tensor, scores_tensor, self.iou_thres)
            
            # Create DetectionResult objects
            detections = []
            for idx in indices:
                idx = int(idx)
                box = boxes_xyxy[idx] # [x1, y1, x2, y2] normalized
                score = float(filtered_scores[idx])
                cls_id = int(filtered_class_ids[idx])
                label = self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id)
                
                # Clip coordinates to 0-1
                x1 = max(0.0, min(1.0, float(box[0])))
                y1 = max(0.0, min(1.0, float(box[1])))
                x2 = max(0.0, min(1.0, float(box[2])))
                y2 = max(0.0, min(1.0, float(box[3])))
                
                result = DetectionResult(
                    label=label,
                    confidence=score,
                    bbox=BoundingBox(
                        left=x1,
                        top=y1,
                        right=x2,
                        bottom=y2
                    )
                )
                detections.append(result)
                
            logger.info(f"TFLite Detection: {len(detections)} objects in {inference_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"TFLite inference failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "YOLOv5s TFLite",
            "model_size": "~14 MB",
            "device": "CPU (TFLite)",
            "num_classes": len(self.labels),
            "status": "loaded" if self.model_loaded else "fallback",
            "confidence_threshold": self.conf_thres
        }
