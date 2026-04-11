from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from app.core.logging import get_logger
from app.schemas.detection import DetectionResult, FaceDetectionResult, TextDetectionResult, BoundingBox


logger = get_logger(__name__)

class DetectionPostprocessor:
    """Service for postprocessing detection results."""
    
    @staticmethod
    def filter_detections_by_confidence(detections: List[DetectionResult], min_confidence: float) -> List[DetectionResult]:
        """Filter detections by minimum confidence threshold."""
        filtered_detections = [
            detection for detection in detections 
            if detection.confidence >= min_confidence
        ]
        
        logger.info(f"Filtered detections: {len(detections)} -> {len(filtered_detections)} "
                   f"(threshold: {min_confidence})")
        
        return filtered_detections
    
    @staticmethod
    def filter_text_detections(text_detections: List[TextDetectionResult], confidence_threshold: float) -> List[TextDetectionResult]:
        """Filter text detections by minimum confidence threshold."""
        filtered_detections = [
            detection for detection in text_detections 
            if detection.confidence >= confidence_threshold
        ]
        
        logger.info(f"Filtered text detections: {len(text_detections)} -> {len(filtered_detections)} "
                   f"(threshold: {confidence_threshold})")
        
        return filtered_detections
    
    @staticmethod
    def filter_detections(detections: List[DetectionResult], confidence_threshold: float) -> List[DetectionResult]:
        """Alias for filter_detections_by_confidence for compatibility."""
        return DetectionPostprocessor.filter_detections_by_confidence(detections, confidence_threshold)
    
    @staticmethod
    def categorize_detections_by_priority(detections: List[DetectionResult]) -> Tuple[List[DetectionResult], List[DetectionResult]]:
        """Categorize detections into high-priority (alert) and low-priority (quiet) objects.
        
        High-priority (alert): car, truck, bus, person, motorcycle, bicycle, dog, cat
        Low-priority (quiet): laptop, phone, backpack, bottle, cup, handbag, etc.
        
        Returns:
            Tuple of (alert_detections, quiet_detections)
        """
        # Objects that warrant immediate alerts (obstacles/hazards/people)
        alert_objects = {
            'car', 'truck', 'bus', 'person', 'motorcycle', 'bicycle', 'scooter',
            'dog', 'cat', 'bear', 'elephant', 'horse',  # Animals
            'stop sign', 'traffic light', 'fire hydrant',  # Street hazards
        }
        
        # Objects that should be announced quietly (small/harmless items)
        quiet_objects = {
            'laptop', 'phone', 'tablet', 'computer', 'keyboard', 'mouse',
            'backpack', 'suitcase', 'handbag', 'bag', 'purse',
            'bottle', 'cup', 'glass', 'chair', 'table', 'desk',
            'book', 'pen', 'watch', 'shoe', 'hat', 'umbrella',
        }
        
        alert_detections = []
        quiet_detections = []
        
        for detection in detections:
            label_lower = detection.label.lower()
            
            # Check if exact match in alert set
            if label_lower in alert_objects:
                alert_detections.append(detection)
                logger.debug(f"Alert object: {detection.label}")
            # Check if exact match in quiet set
            elif label_lower in quiet_objects:
                quiet_detections.append(detection)
                logger.debug(f"Quiet object: {detection.label}")
            # Default behavior: if label contains common alert keywords, alert
            elif any(keyword in label_lower for keyword in ['person', 'car', 'truck', 'bus', 'bike', 'motorcycle']):
                alert_detections.append(detection)
                logger.debug(f"Alert object (keyword match): {detection.label}")
            # Otherwise treat as quiet
            else:
                quiet_detections.append(detection)
                logger.debug(f"Quiet object (default): {detection.label}")
        
        logger.info(f"Object categorization: {len(alert_detections)} alert + {len(quiet_detections)} quiet = {len(detections)} total")
        return alert_detections, quiet_detections
    
    @staticmethod
    def normalize_bounding_boxes(detections: List[DetectionResult], image_width: int, image_height: int) -> List[DetectionResult]:
        """Normalize bounding box coordinates to [0, 1] range."""
        normalized_detections = []
        
        for detection in detections:
            bbox = detection.bbox
            
            # Convert from absolute to normalized coordinates
            normalized_bbox = BoundingBox(
                left=bbox.left / image_width,
                top=bbox.top / image_height,
                right=bbox.right / image_width,
                bottom=bbox.bottom / image_height
            )
            
            normalized_detection = DetectionResult(
                label=detection.label,
                confidence=detection.confidence,
                bbox=normalized_bbox
            )
            
            normalized_detections.append(normalized_detection)
        
        logger.info(f"Normalized bounding boxes for {len(normalized_detections)} detections")
        
        return normalized_detections
    
    @staticmethod
    def clip_bounding_boxes(detections: List[DetectionResult]) -> List[DetectionResult]:
        """Clip bounding box coordinates to [0, 1] range."""
        clipped_detections = []
        
        for detection in detections:
            bbox = detection.bbox
            
            # Clip coordinates to [0, 1] range
            clipped_bbox = BoundingBox(
                left=max(0.0, min(1.0, bbox.left)),
                top=max(0.0, min(1.0, bbox.top)),
                right=max(0.0, min(1.0, bbox.right)),
                bottom=max(0.0, min(1.0, bbox.bottom))
            )
            
            clipped_detection = DetectionResult(
                label=detection.label,
                confidence=detection.confidence,
                bbox=clipped_bbox
            )
            
            clipped_detections.append(clipped_detection)
        
        logger.info(f"Clipped bounding boxes for {len(clipped_detections)} detections")
        
        return clipped_detections
    
    @staticmethod
    def sort_detections_by_confidence(detections: List[DetectionResult]) -> List[DetectionResult]:
        """Sort detections by confidence in descending order."""
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Sorted {len(sorted_detections)} detections by confidence")
        
        return sorted_detections
    
    @staticmethod
    def merge_detection_results(object_detections: List[DetectionResult], 
                              face_detections: List[FaceDetectionResult]) -> Dict[str, Any]:
        """Merge object and face detection results."""
        result = {
            "objects": [d.dict() for d in object_detections],
            "faces": [d.dict() for d in face_detections],
            "total_detections": len(object_detections) + len(face_detections)
        }
        
        logger.info(f"Merged detection results: {len(object_detections)} objects, "
                   f"{len(face_detections)} faces")
        
        return result
    
    @staticmethod
    def format_detection_response(detections: List[DetectionResult], 
                                inference_time_ms: Optional[float] = None) -> Dict[str, Any]:
        """Format detection results for API response."""
        response = {
            "detections": [detection.dict() for detection in detections],
            "inference_time_ms": inference_time_ms
        }
        
        logger.info(f"Formatted detection response: {len(detections)} detections")
        
        return response
    
    @staticmethod
    def calculate_detection_statistics(detections: List[DetectionResult]) -> Dict[str, Any]:
        """Calculate statistics about detections."""
        if not detections:
            return {
                "total_detections": 0,
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "unique_labels": []
            }
        
        confidences = [d.confidence for d in detections]
        labels = [d.label for d in detections]
        
        stats = {
            "total_detections": len(detections),
            "average_confidence": round(sum(confidences) / len(confidences), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "unique_labels": list(set(labels)),
            "label_counts": {label: labels.count(label) for label in set(labels)}
        }
        
        logger.info(f"Detection statistics: {stats}")
        
        return stats
    
    @staticmethod
    def validate_bounding_boxes(detections: List[DetectionResult]) -> List[DetectionResult]:
        """Validate and fix bounding box coordinates."""
        validated_detections = []
        
        for detection in detections:
            bbox = detection.bbox
            
            # Ensure left < right and top < bottom
            left = min(bbox.left, bbox.right)
            right = max(bbox.left, bbox.right)
            top = min(bbox.top, bbox.bottom)
            bottom = max(bbox.top, bbox.bottom)
            
            # Ensure coordinates are within [0, 1] range
            left = max(0.0, min(1.0, left))
            right = max(0.0, min(1.0, right))
            top = max(0.0, min(1.0, top))
            bottom = max(0.0, min(1.0, bottom))
            
            validated_bbox = BoundingBox(
                left=left,
                top=top,
                right=right,
                bottom=bottom
            )
            
            validated_detection = DetectionResult(
                label=detection.label,
                confidence=detection.confidence,
                bbox=validated_bbox
            )
            
            validated_detections.append(validated_detection)
        
        logger.info(f"Validated bounding boxes for {len(validated_detections)} detections")
        
        return validated_detections