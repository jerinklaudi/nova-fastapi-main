import numpy as np
import cv2
from typing import List, Optional, Dict, Any
import time
import os
import base64
from app.core.logging import get_logger, log_inference
from app.core.config import settings
from app.models.yolo_detector import YOLODetector
from app.models.midas_depth import MiDaSDepthEstimator
from app.models.paddle_ocr import PaddleOCRDetector
from app.schemas.detection import (
    DetectionResult, TextDetectionResult, DepthEstimationResult, 
    NavigationGuidanceResult, BoundingBox
)
from app.services.preprocessing import ImagePreprocessor
from app.services.postprocessing import DetectionPostprocessor


logger = get_logger(__name__)

# Disable OCR in navigation by default while focusing on navigation development.
# Set environment var NOVA_OCR_ENABLED=1 to re-enable OCR for navigation.
OCR_ENABLED = os.environ.get("NOVA_OCR_ENABLED", "0").strip().lower() not in ("0", "false", "no")

class NavigationGuidanceService:
    """Navigation guidance service using YOLO + MiDaS + OCR fusion."""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("INITIALIZING NAVIGATION GUIDANCE SERVICE")
        logger.info("=" * 60)
        
        self.yolo_detector = None
        self.depth_estimator = None
        self.ocr_detector = None
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = DetectionPostprocessor()

        self._load_models()
        
        # Log final status
        logger.info("Navigation Guidance Service Initialization Complete:")
        logger.info(f"  - YOLO Detector: {'✓ Loaded' if self.yolo_detector else '✗ Not available'}")
        logger.info(f"  - Depth Estimator: {'✓ Loaded' if self.depth_estimator else '✗ Not available'}")
        logger.info(f"  - OCR Detector: {'✓ Loaded' if self.ocr_detector else '✗ Not available'}")
        logger.info("=" * 60)
    
    def _load_models(self) -> None:
        """Load all required models."""
        # Load YOLO detector
        logger.info("Loading YOLO detector for obstacle detection...")
        try:
            self.yolo_detector = YOLODetector()
            logger.info("✓ YOLO detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ YOLO detector not available: {str(e)}")
            logger.warning("  Navigation will continue without object detection")
        
        # Load MiDaS depth estimator
        logger.info("Loading MiDaS depth estimator...")
        try:
            self.depth_estimator = MiDaSDepthEstimator()
            logger.info("✓ MiDaS depth estimator loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ MiDaS depth estimator not available: {str(e)}")
            logger.warning("  Navigation will continue without depth estimation")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Load OCR detector
        if OCR_ENABLED:
            logger.info("Loading PaddleOCR detector for text recognition...")
            try:
                self.ocr_detector = PaddleOCRDetector()
                logger.info("✓ PaddleOCR detector loaded successfully")
            except Exception as e:
                logger.warning(f"⚠ PaddleOCR detector not available: {str(e)}")
                logger.warning("  Navigation will continue without text recognition")
        else:
            logger.info("OCR disabled via environment variable (NOVA_OCR_ENABLED=0)")
    
    def _calculate_distance_from_depth(self, depth_map: np.ndarray, bbox: BoundingBox) -> float:
        """Calculate distance to object from depth map and bounding box."""
        if depth_map is None or bbox is None:
            return 0.0
        
        # Convert normalized bbox to pixel coordinates
        height, width = depth_map.shape
        x_min = int(bbox.left * width)
        y_min = int(bbox.top * height)
        x_max = int(bbox.right * width)
        y_max = int(bbox.bottom * height)
        
        # Get depth values within the bounding box
        bbox_depth = depth_map[y_min:y_max, x_min:x_max]
        
        if bbox_depth.size == 0:
            return 0.0
        
        # Calculate mean depth within the bounding box
        mean_depth = np.mean(bbox_depth)
        
        # Convert depth to distance (this is a simplified conversion)
        # In a real implementation, you would need camera calibration
        distance = mean_depth * 10.0  # Simplified scaling factor
        
        return distance
    
    def _analyze_depth_safety(self, depth_map: np.ndarray, threshold_distance: float = 2.0) -> List[str]:
        """Analyze depth map for safety warnings."""
        warnings = []
        
        if depth_map is None:
            return warnings
        
        # Find areas that are too close (high depth values)
        close_threshold = 0.8  # Normalized depth threshold
        close_areas = depth_map > close_threshold

        # Avoid constant false positives: require a meaningful area to be close.
        close_ratio = float(np.mean(close_areas))
        if close_ratio > 0.08:
            warnings.append("Obstacle detected nearby")
        
        # Analyze depth gradients for steps or drop-offs
        try:
            depth_gradient = np.gradient(depth_map)
            gradient_magnitude = np.sqrt(depth_gradient[0]**2 + depth_gradient[1]**2)
            
            # High gradients might indicate steps or edges
            if np.max(gradient_magnitude) > 0.5:
                warnings.append("Potential step or elevation change detected")
        except Exception as e:
            logger.warning(f"Depth gradient analysis failed: {str(e)}")
        
        return warnings

    def _build_debug_heatmap_base64(self, depth_map: np.ndarray) -> Optional[str]:
        """Convert normalized depth map to a compact JPEG heatmap for frontend overlay."""
        try:
            if depth_map is None or depth_map.size == 0:
                return None

            clipped = np.clip(depth_map, 0.0, 1.0)
            depth_u8 = (clipped * 255.0).astype(np.uint8)
            heatmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            ok, encoded = cv2.imencode('.jpg', heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return None
            return base64.b64encode(encoded.tobytes()).decode('ascii')
        except Exception as e:
            logger.debug(f"Heatmap encoding failed: {str(e)}")
            return None
    
    def _analyze_zone_based_direction(self, alert_obstacles: List[DetectionResult], 
                                     min_alert_distance: float = 2.0) -> Optional[str]:
        """Analyze obstacle positions across 3 screen zones and recommend movement direction.
        
        Screen zones:
        - LEFT: 0.0 to 0.33
        - CENTER: 0.33 to 0.67
        - RIGHT: 0.67 to 1.0
        
        Direction logic:
        - If near obstacle (distance < min_alert_distance) on LEFT → "Move right"
        - If near obstacle on RIGHT → "Move left"
        - If centered → "Proceed carefully"
        
        Args:
            alert_obstacles: List of high-priority detections (car, person, etc.)
            min_alert_distance: Minimum distance (meters) to trigger directional alerts
        
        Returns:
            Direction string or None if no directional alert needed
        """
        if not alert_obstacles:
            return None
        
        # Filter obstacles that are close enough to warrant directional guidance.
        # Depth can be noisy/missing, so we fall back to on-screen bbox size.
        close_obstacles = []
        for obs in alert_obstacles:
            distance = float(getattr(obs, 'distance', 0.0) or 0.0)
            bbox = obs.bbox
            width = max(0.0, float(bbox.right) - float(bbox.left))
            height = max(0.0, float(bbox.bottom) - float(bbox.top))
            area_ratio = width * height

            # Primary trigger: valid depth says object is close.
            if distance > 0 and distance < min_alert_distance:
                close_obstacles.append(obs)
                continue

            # Fallback trigger: if depth missing/unreliable but object is visually large,
            # treat it as near enough for directional guidance.
            if (distance <= 0 or distance > (min_alert_distance + 1.5)) and area_ratio >= 0.10:
                close_obstacles.append(obs)
        
        if not close_obstacles:
            logger.debug(f"No obstacles within {min_alert_distance}m for directional guidance")
            return None
        
        logger.info(
            f"Zone analysis: {len(close_obstacles)} close obstacles within {min_alert_distance}m "
            f"(with bbox fallback)"
        )
        
        # Categorize close obstacles by horizontal zone
        left_zone_obstacles = []    # x < 0.33
        center_zone_obstacles = []  # 0.33 <= x < 0.67
        right_zone_obstacles = []   # x >= 0.67
        
        for obs in close_obstacles:
            bbox = obs.bbox
            center_x = (bbox.left + bbox.right) / 2.0
            distance = float(getattr(obs, 'distance', 0.0) or 0.0)
            
            if center_x < 0.33:
                left_zone_obstacles.append((obs.label, distance, center_x))
                logger.debug(f"  Left zone: {obs.label} at {distance:.1f}m (x={center_x:.2f})")
            elif center_x >= 0.67:
                right_zone_obstacles.append((obs.label, distance, center_x))
                logger.debug(f"  Right zone: {obs.label} at {distance:.1f}m (x={center_x:.2f})")
            else:
                center_zone_obstacles.append((obs.label, distance, center_x))
                logger.debug(f"  Center zone: {obs.label} at {distance:.1f}m (x={center_x:.2f})")
        
        # Determine direction based on zone occupancy
        if left_zone_obstacles and not right_zone_obstacles:
            # Obstacle on left → move right
            closest_left = min(left_zone_obstacles, key=lambda x: x[1])
            direction = "Move right"
            logger.info(f"Zone guidance: {direction} (obstacle on left: {closest_left[0]} at {closest_left[1]:.1f}m)")
            return direction
        elif right_zone_obstacles and not left_zone_obstacles:
            # Obstacle on right → move left
            closest_right = min(right_zone_obstacles, key=lambda x: x[1])
            direction = "Move left"
            logger.info(f"Zone guidance: {direction} (obstacle on right: {closest_right[0]} at {closest_right[1]:.1f}m)")
            return direction
        elif left_zone_obstacles and right_zone_obstacles:
            # Obstacles on both sides → center is safer or blocked
            closest_left = min(left_zone_obstacles, key=lambda x: x[1])
            closest_right = min(right_zone_obstacles, key=lambda x: x[1])
            if closest_left[1] < closest_right[1]:
                # Left is closer → move right
                direction = "Move right"
                logger.info(f"Zone guidance: {direction} (left is closer)")
            else:
                # Right is closer → move left
                direction = "Move left"
                logger.info(f"Zone guidance: {direction} (right is closer)")
            return direction
        else:
            # Obstacles only in center zone
            logger.info("Zone guidance: obstacles centered, proceed carefully")
            return "Proceed carefully"
    
    def _generate_guidance_text(self, 
                              alert_obstacles: List[DetectionResult],
                              quiet_obstacles: List[DetectionResult],
                              text_signs: List[TextDetectionResult],
                              depth_warnings: List[str]) -> str:
        """Generate navigation guidance text with zone-based directional awareness.
        
        Alert objects (car, person, etc.) get immediate, emphatic warnings.
        Quiet objects (laptop, phone, etc.) get gentle announcements.
        Directional guidance is based on object zones: left/center/right with distance threshold.
        """
        guidance_parts = []
        
        # Add HIGH-PRIORITY obstacle alerts with smart zone-based direction
        if alert_obstacles:
            obstacle_info = []
            for obs in alert_obstacles:
                distance = getattr(obs, 'distance', 0.0)
                if distance > 0:
                    obstacle_info.append(f"{obs.label} at {distance:.1f}m")
                else:
                    obstacle_info.append(obs.label)
            
            if obstacle_info:
                guidance_parts.append(f"Detected {', '.join(obstacle_info)} ahead")
            
            # Use zone-based analysis for directional guidance.
            # Trigger slightly earlier (2.8m) for practical usability.
            direction = self._analyze_zone_based_direction(alert_obstacles, min_alert_distance=2.8)
            if direction:
                guidance_parts.append(direction)
        
        # Add LOW-PRIORITY objects (quiet announcement)
        if quiet_obstacles:
            quiet_info = []
            for obs in quiet_obstacles:
                quiet_info.append(obs.label)
            
            if quiet_info:
                guidance_parts.append(f"Nearby: {', '.join(set(quiet_info))}")
        
        # Add text sign information
        if text_signs:
            sign_texts = [sign.text for sign in text_signs if sign.text.strip()]
            if sign_texts:
                guidance_parts.append(f"Signs: {', '.join(sign_texts)}")
        
        # Add safety warnings
        if depth_warnings:
            guidance_parts.extend(depth_warnings)
        
        # Generate final guidance
        if guidance_parts:
            return ". ".join(guidance_parts) + "."
        else:
            return "Clear path ahead."
    
    def _enhance_detections_with_depth(self, 
                                     detections: List[DetectionResult], 
                                     depth_map: np.ndarray) -> List[DetectionResult]:
        """Enhance object detections with depth information."""
        enhanced_detections = []
        
        for detection in detections:
            # Calculate distance for each detection
            distance = self._calculate_distance_from_depth(depth_map, detection.bbox)
            
            # Pydantic models are immutable-ish; update through model_copy.
            enhanced_detection = detection.model_copy(update={"distance": distance})
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def get_navigation_guidance(
        self,
        image_data: bytes,
        image_type: str,
        object_confidence: float = 0.25,
        text_confidence: float = 0.3,
    ) -> NavigationGuidanceResult:
        """Generate navigation guidance from input image."""
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self.preprocessor.validate_and_load_image(image_data, image_type)
            
            # Convert PIL Image to numpy array for processing
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            logger.debug(f"Navigation pipeline - input image shape: {image_np.shape}")

            # Run all models in parallel as much as possible
            yolo_detections = []
            depth_result = None
            ocr_results = []

            # Object detection with YOLO
            if self.yolo_detector:
                yolo_detections = self.yolo_detector.detect(image_np)
            logger.debug(f"YOLO detections count: {len(yolo_detections)}")
            logger.info(f"Navigation YOLO detections (pre-filter): {len(yolo_detections)}")

            # Depth estimation with MiDaS
            if self.depth_estimator:
                depth_result = self.depth_estimator.estimate_depth(image_np)
            logger.debug(f"Depth estimator available: {depth_result is not None}")

            # Text detection with OCR (skipped if OCR disabled)
            if self.ocr_detector and OCR_ENABLED:
                logger.debug("Running OCR detector for navigation guidance")
                ocr_results = self.ocr_detector.detect_text(image_np)
                logger.debug(f"OCR returned {len(ocr_results)} text regions")
            else:
                logger.debug(f"OCR skipped (ocr_detector present: {self.ocr_detector is not None}, OCR_ENABLED: {OCR_ENABLED})")

            # Postprocess results
            if self.postprocessor:
                pre_filter_count = len(yolo_detections)
                yolo_detections = self.postprocessor.filter_detections(
                    yolo_detections,
                    confidence_threshold=object_confidence,
                )
                ocr_results = self.postprocessor.filter_text_detections(
                    ocr_results,
                    confidence_threshold=text_confidence,
                )
                logger.info(
                    "Navigation YOLO detections after filter %.2f: %d -> %d",
                    object_confidence,
                    pre_filter_count,
                    len(yolo_detections),
                )
                if yolo_detections:
                    sample = ", ".join(
                        f"{d.label}:{d.confidence:.2f}" for d in yolo_detections[:5]
                    )
                    logger.info(f"Navigation YOLO sample detections: {sample}")
            
            # Categorize detections by priority (alert vs quiet)
            alert_detections, quiet_detections = self.postprocessor.categorize_detections_by_priority(yolo_detections)
            logger.info(f"Detection categorization: {len(alert_detections)} alert, {len(quiet_detections)} quiet")

            logger.debug(f"Postprocessed -> detections: {len(yolo_detections)}, text_signs: {len(ocr_results)}")
            
            # Enhance detections with depth information
            enhanced_alert_detections = []
            enhanced_quiet_detections = []
            depth_map_data = getattr(depth_result, "depth_map", None) if depth_result else None
            if depth_map_data is not None:
                depth_map = np.array(depth_map_data)
                enhanced_alert_detections = self._enhance_detections_with_depth(alert_detections, depth_map) if alert_detections else []
                enhanced_quiet_detections = self._enhance_detections_with_depth(quiet_detections, depth_map) if quiet_detections else []
            else:
                enhanced_alert_detections = alert_detections
                enhanced_quiet_detections = quiet_detections

            logger.debug(f"Enhanced detections: {len(enhanced_alert_detections)} alert + {len(enhanced_quiet_detections)} quiet")
            
            # All detections combined for front-end display
            all_enhanced_detections = enhanced_alert_detections + enhanced_quiet_detections
            
            # Analyze depth for safety warnings
            depth_warnings = []
            debug_frame_base64 = None
            if depth_map_data is not None:
                depth_map = np.array(depth_map_data)
                depth_warnings = self._analyze_depth_safety(depth_map)
                debug_frame_base64 = self._build_debug_heatmap_base64(depth_map)
            
            # Generate guidance text (pass both alert and quiet lists for differentiated speech)
            guidance_text = self._generate_guidance_text(
                enhanced_alert_detections, 
                enhanced_quiet_detections,
                ocr_results, 
                depth_warnings
            )
            
            total_time = time.time() - start_time
            
            # Log inference details
            log_inference("Navigation Guidance", total_time, image_np.shape, len(all_enhanced_detections))
            
            # Create result
            result = NavigationGuidanceResult(
                obstacles=all_enhanced_detections,
                text_signs=ocr_results,
                depth_analysis=depth_result,
                guidance=guidance_text,
                safety_warnings=depth_warnings,
                inference_time_ms=round(total_time * 1000, 2),
                debug_frame_base64=debug_frame_base64,
            )
            
            logger.debug(f"Navigation guidance summary: obstacles={len(result.obstacles)}, signs={len(result.text_signs)}, warnings={len(result.safety_warnings)}")
            logger.info(f"Navigation guidance completed: {result.inference_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Navigation guidance failed: {str(e)}")
            raise

    def _warmup_models(self) -> Dict[str, bool]:
        """Run a lightweight warmup pass for available models.

        This is intentionally best-effort and should never crash startup.
        """
        warmup_status = {
            "yolo": False,
            "depth": False,
            "ocr": False,
        }

        try:
            # Small synthetic frame to compile kernels/graphs and prime caches.
            warmup_image = np.zeros((256, 256, 3), dtype=np.uint8)

            if self.yolo_detector is not None:
                try:
                    self.yolo_detector.detect(warmup_image)
                    warmup_status["yolo"] = True
                except Exception as e:
                    logger.warning(f"YOLO warmup failed: {str(e)}")

            if self.depth_estimator is not None:
                try:
                    self.depth_estimator.estimate_depth(warmup_image)
                    warmup_status["depth"] = True
                except Exception as e:
                    logger.warning(f"Depth warmup failed: {str(e)}")

            if self.ocr_detector is not None and OCR_ENABLED:
                try:
                    self.ocr_detector.detect_text(warmup_image)
                    warmup_status["ocr"] = True
                except Exception as e:
                    logger.warning(f"OCR warmup failed: {str(e)}")

            logger.info(
                "Warmup complete - YOLO: %s, Depth: %s, OCR: %s",
                warmup_status["yolo"],
                warmup_status["depth"],
                warmup_status["ocr"],
            )
        except Exception as e:
            logger.warning(f"Navigation warmup encountered an unexpected error: {str(e)}")

        return warmup_status
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models used in navigation guidance."""
        status = {
            "yolo_detector": self.yolo_detector is not None,
            "depth_estimator": self.depth_estimator is not None,
            "ocr_detector": self.ocr_detector is not None,
            "preprocessor": True,
            "postprocessor": True
        }
        
        # Get model info if available
        model_info = {}
        
        if self.yolo_detector:
            model_info["yolo"] = self.yolo_detector.get_model_info()
        
        if self.depth_estimator:
            model_info["depth"] = self.depth_estimator.get_model_info()
        
        if self.ocr_detector:
            model_info["ocr"] = self.ocr_detector.get_model_info()
        
        return {
            "status": status,
            "model_info": model_info,
            "all_models_loaded": all(status.values())
        }