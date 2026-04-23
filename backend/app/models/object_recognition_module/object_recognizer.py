"""
Object Recognizer Module
Matches detected YOLO objects against registered object database.
Phase 1: Class-level matching (label-based personalization).
"""

import logging
from typing import Dict, List, Optional, Tuple
from app.models.object_recognition_module.object_db import ObjectDatabase

logger = logging.getLogger(__name__)


class ObjectRecognizer:
    """
    Matches YOLO detections against registered objects.
    Returns only recognized objects (those registered by the user).
    """

    def __init__(self):
        self.db = ObjectDatabase()
        logger.info("ObjectRecognizer initialized")

    def recognize_objects(
        self,
        yolo_detections: List[Dict],
        registered_only: bool = True,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Match YOLO detections against registered objects.

        Args:
            yolo_detections: List of detection dicts with 'label', 'confidence', etc.
            registered_only: If True, return only registered matches. If False, return all.
            confidence_threshold: Minimum confidence to consider a detection.

        Returns:
            (recognized_objects, unrecognized_objects): Two lists of detections.
            - recognized_objects: detections matching registered objects (with user name)
            - unrecognized_objects: detections not matching any registered object
        """
        recognized = []
        unrecognized = []

        all_registered = self.db.get_all_objects()
        registered_labels = {obj.get("target_label", "").lower(): obj for obj in all_registered}

        for detection in yolo_detections:
            conf = detection.get("confidence", 0.0)
            if conf < confidence_threshold:
                continue

            label = detection.get("label", "").lower()

            # Check if this label matches any registered object
            if label in registered_labels:
                registered_obj = registered_labels[label]
                enriched_detection = {
                    **detection,
                    "registered": True,
                    "registered_name": registered_obj.get("name", label),
                    "registered_id": registered_obj.get("id"),
                    "is_priority": registered_obj.get("is_priority", False),
                }
                recognized.append(enriched_detection)
            else:
                enriched_detection = {
                    **detection,
                    "registered": False,
                }
                unrecognized.append(enriched_detection)

        logger.info(
            f"ObjectRecognizer: {len(recognized)} recognized, {len(unrecognized)} unrecognized "
            f"(from {len(yolo_detections)} total detections)"
        )

        if registered_only:
            return recognized, []
        else:
            return recognized, unrecognized

    def get_announcement_text(self, detections: List[Dict]) -> str:
        """
        Build a natural speech announcement for recognized objects.

        Example:
          Input: [{'registered_name': 'my keys', 'is_priority': False}, ...]
          Output: "Detected your keys"
        """
        if not detections:
            return ""

        # Separate priority and normal detections
        priority = [d for d in detections if d.get("is_priority", False)]
        normal = [d for d in detections if not d.get("is_priority", False)]

        parts = []

        # Priority objects get urgent announcements
        if priority:
            priority_names = [d.get("registered_name", d.get("label", "object")) for d in priority]
            if len(priority_names) == 1:
                parts.append(f"PRIORITY: {priority_names[0]}")
            else:
                parts.append(f"PRIORITY OBJECTS: {', '.join(priority_names)}")

        # Normal objects get standard announcement
        if normal:
            normal_names = [d.get("registered_name", d.get("label", "object")) for d in normal]
            if len(normal_names) == 1:
                parts.append(f"Detected {normal_names[0]}")
            else:
                parts.append(f"Detected {', '.join(normal_names)}")

        return ". ".join(parts)

    def reload_db(self):
        """Reload the object database from disk."""
        self.db.reload()
        logger.info("ObjectRecognizer database reloaded")
