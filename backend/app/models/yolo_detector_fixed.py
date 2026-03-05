import numpy as np
import torch
from typing import List, Optional
import time
import os
import logging
import cv2
from datetime import datetime

from app.core.config import settings
from app.schemas.detection import DetectionResult, BoundingBox
from app.models.midas_depth import MiDaSDepthEstimator

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLO object detection model wrapper - with fallback for offline mode."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.model = None
        self.device = torch.device('cpu')
        self.labels = self._get_coco_labels()
        self.model_loaded = False
        
        # Initialize depth estimator for visualization
        self.depth_estimator = MiDaSDepthEstimator()
        
        # Create debug directory
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug_visualizations")
        os.makedirs(self.debug_dir, exist_ok=True)
        
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
        """Load the YOLO model with robust fallback."""
        logger.info("NOVA Object Detection Module - Initializing...")
        logger.info(f"Device: {self.device} (CPU-only, offline-capable)")
        
        # Try PyTorch Hub
        logger.info("Attempting to load YOLOv5n from PyTorch Hub...")
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'yolov5n',
                    pretrained=True,
                    force_reload=False,
                    device=self.device,
                    verbose=False
                )
                # Set very low confidence to capture everything for logging
                self.model.conf = 0.05 
                logger.info("✓ YOLOv5n loaded from PyTorch Hub")
                self.model_loaded = True
                return
        except Exception as e:
            logger.warning(f"PyTorch Hub loading failed: {str(e)[:100]}")
            logger.warning("Attempting fallback...")
        
        # Try loading from cache
        cache_dir = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        cached_model_path = os.path.join(cache_dir, "yolov5n.pt")
        
        if os.path.exists(cached_model_path):
            logger.info(f"Found cached model. Loading from cache...")
            try:
                self.model = torch.load(cached_model_path, map_location=self.device)
                self.model.conf = 0.05
                logger.info("✓ Model loaded from cache")
                self.model_loaded = True
                return
            except Exception as e:
                logger.warning(f"Cache loading failed: {str(e)}")
        
        # Check for local TFLite model
        if os.path.exists(self.model_path):
            logger.info(f"TFLite model available but not supported - using mock detections")
            logger.warning("The YOLO model could not be loaded. Object detection will return simulated results.")
            self.model = None
            self.model_loaded = False
            return
        
        logger.error("No YOLO model available. Object detection will return empty results.")
        self.model = None
        self.model_loaded = False
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Perform object detection on the input image."""
        start_time = time.perf_counter()
        
        try:
            if self.model is None:
                logger.debug("Model not loaded, returning empty detections")
                return []
            
            # Convert to PIL Image
            from PIL import Image
            import cv2
            
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # OpenCV uses BGR, convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Run inference
            results = self.model(pil_image) if self.model_loaded else None
            
            if results is None:
                return []
            
            inference_time = time.perf_counter() - start_time
            
            # Parse results
            detections = []
            all_candidates = [] # For visualization
            
            try:
                df = results.pandas().xyxy[0]  # Get detections as DataFrame
                
                # Log all candidates
                if not df.empty:
                    logger.info(f"--- Detection Candidates (Threshold: 0.05) ---")
                    for _, row in df.iterrows():
                        logger.info(f"Candidate: {row['name']} ({row['confidence']:.4f})")
                        
                        # Store for visualization
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        all_candidates.append({
                            "bbox": (x1, y1, x2, y2),
                            "label": row['name'],
                            "confidence": row['confidence']
                        })
                else:
                    logger.info("No candidates found even at low threshold.")

                for _, row in df.iterrows():
                    # Filter by configured threshold
                    if row['confidence'] < settings.CONFIDENCE_THRESHOLD:
                        continue
                        
                    # Normalize bbox coordinates
                    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                    height, width = image.shape[:2]
                    
                    detection = DetectionResult(
                        label=str(row['name']),
                        confidence=float(row['confidence']),
                        bbox=BoundingBox(
                            left=float(x1 / width),
                            top=float(y1 / height),
                            right=float(x2 / width),
                            bottom=float(y2 / height)
                        )
                    )
                    detections.append(detection)
                
                logger.debug(f"Detection completed in {inference_time:.3f}s: {len(detections)} valid objects")
                
                # --- VISUALIZATION BLOCK ---
                try:
                    # 1. Generate Depth Heatmap
                    depth_result = self.depth_estimator.estimate_depth(image)
                    if depth_result:
                        depth_map = np.array(depth_result.depth_map)
                        # Normalize to 0-255 uint8
                        depth_norm = (depth_map * 255).astype(np.uint8)
                        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    else:
                        depth_colormap = np.zeros_like(image)
                        cv2.putText(depth_colormap, "Depth Failed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # 2. Draw Bounding Boxes on Original Image
                    viz_image = image.copy()
                    for cand in all_candidates:
                        x1, y1, x2, y2 = cand['bbox']
                        conf = cand['confidence']
                        label = cand['label']
                        
                        # Color coding: Green for valid, Red for filtered
                        color = (0, 255, 0) if conf >= settings.CONFIDENCE_THRESHOLD else (0, 0, 255)
                        
                        cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(viz_image, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 3. Combine Images (Side-by-Side)
                    # Resize depth map to match image if needed (should match, but safety first)
                    if depth_colormap.shape != viz_image.shape:
                         depth_colormap = cv2.resize(depth_colormap, (viz_image.shape[1], viz_image.shape[0]))
                         
                    combined_viz = np.hstack((viz_image, depth_colormap))
                    
                    # 4. Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"detection_{timestamp}.jpg"
                    filepath = os.path.join(self.debug_dir, filename)
                    cv2.imwrite(filepath, combined_viz)
                    logger.info(f"Saved debug visualization to: {filepath}")
                    
                except Exception as viz_err:
                    logger.error(f"Visualization failed: {str(viz_err)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                # ---------------------------

            except Exception as e:
                logger.error(f"Failed to parse YOLO results: {str(e)}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "YOLOv5n" if self.model_loaded else "Mock (Model not loaded)",
            "model_size": "~14 MB",
            "device": str(self.device),
            "num_classes": len(self.labels),
            "status": "loaded" if self.model_loaded else "fallback",
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "logging_threshold": 0.05,
            "visualization": "Enabled (BBox + Depth)"
        }

