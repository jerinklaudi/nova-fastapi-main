"""
Object Database Module
Manages persistent storage and retrieval of registered objects.
Similar to EmbeddingDatabase but for objects instead of faces.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ObjectDatabase:
    """
    Persistent store for registered objects.
    Each object record contains:
      - id: unique identifier (timestamp-based or UUID)
      - name: user-friendly name ("my keys", "medicine box")
      - target_label: YOLO class name ("bottle", "cup", "keys", etc.)
      - is_priority: bool for priority alerts
      - created_at: timestamp
      - updated_at: timestamp
    """

    def __init__(self, db_path: Optional[str] = None):
        # Canonical path: backend/database/objects.json
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "database",
            "objects.json"
        )
        # Legacy path used by older builds: backend/app/database/objects.json
        self.legacy_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "database",
            "objects.json"
        )
        self.data = {"objects": []}
        self.reload()

    def reload(self):
        """Load objects.json from disk."""
        canonical_exists = os.path.exists(self.db_path)
        legacy_exists = os.path.exists(self.legacy_db_path)

        # Default to canonical path.
        load_path = self.db_path

        # If canonical is missing, but legacy exists, load legacy then migrate.
        if not canonical_exists and legacy_exists:
            load_path = self.legacy_db_path

        if os.path.exists(load_path):
            try:
                with open(load_path, 'r') as f:
                    self.data = json.load(f)
                if "objects" not in self.data:
                    self.data["objects"] = []

                # If canonical exists but is empty and legacy has data, migrate legacy data.
                if (
                    load_path == self.db_path
                    and legacy_exists
                    and len(self.data.get("objects", [])) == 0
                ):
                    try:
                        with open(self.legacy_db_path, 'r') as f:
                            legacy_data = json.load(f)
                        legacy_objects = legacy_data.get("objects", []) if isinstance(legacy_data, dict) else []
                        if legacy_objects:
                            self.data = {"objects": legacy_objects}
                            logger.info(
                                "Canonical object DB is empty; migrated %d objects from legacy path %s",
                                len(legacy_objects),
                                self.legacy_db_path,
                            )
                            self.save()
                    except Exception as migrate_error:
                        logger.warning(f"Legacy object DB migration skipped: {migrate_error}")

                # Persist back to canonical path if we loaded from legacy.
                if load_path != self.db_path:
                    logger.info(
                        "Migrating object DB from legacy path %s to %s",
                        self.legacy_db_path,
                        self.db_path,
                    )
                    self.save()

                logger.info(f"Loaded {len(self.data['objects'])} registered objects from {load_path}")
            except Exception as e:
                logger.error(f"Failed to reload object database: {e}")
                self.data = {"objects": []}
        else:
            logger.warning(f"Object database file not found at {self.db_path}, starting empty")
            self.data = {"objects": []}

    def save(self) -> bool:
        """Write objects.json to disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Saved {len(self.data['objects'])} objects to {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save object database: {e}")
            return False

    def add_object(
        self,
        obj_id: str,
        name: str,
        target_label: str,
        is_priority: bool = False,
        samples_used: int = 0,
    ) -> bool:
        """Register a new object."""
        if any(o["id"] == obj_id for o in self.data["objects"]):
            logger.warning(f"Object with id {obj_id} already exists")
            return False

        from datetime import datetime
        obj_record = {
            "id": obj_id,
            "name": name,
            "target_label": target_label,
            "is_priority": is_priority,
            "samples_used": samples_used,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.data["objects"].append(obj_record)
        logger.info(f"Added object: {name} (label: {target_label})")
        return True

    def delete_object(self, obj_id: str) -> bool:
        """Remove an object by id."""
        original_count = len(self.data["objects"])
        self.data["objects"] = [o for o in self.data["objects"] if o["id"] != obj_id]
        if len(self.data["objects"]) < original_count:
            logger.info(f"Deleted object with id {obj_id}")
            return True
        logger.warning(f"Object with id {obj_id} not found")
        return False

    def get_all_objects(self) -> List[Dict]:
        """Retrieve all registered objects."""
        return self.data.get("objects", [])

    def get_objects_by_label(self, label: str) -> List[Dict]:
        """Get all registered objects matching a YOLO class label."""
        return [o for o in self.data.get("objects", []) if o.get("target_label", "").lower() == label.lower()]

    def get_priority_objects(self) -> List[Dict]:
        """Retrieve only priority-flagged objects."""
        return [o for o in self.data.get("objects", []) if o.get("is_priority", False)]

    def get_object_by_id(self, obj_id: str) -> Optional[Dict]:
        """Retrieve a single object by id."""
        for o in self.data.get("objects", []):
            if o.get("id") == obj_id:
                return o
        return None

    def get_object_by_name(self, name: str) -> Optional[Dict]:
        """Retrieve a single object by name."""
        for o in self.data.get("objects", []):
            if o.get("name", "").lower() == name.lower():
                return o
        return None

    def update_priority(self, obj_id: str, is_priority: bool) -> bool:
        """Update priority flag for an object."""
        for o in self.data.get("objects", []):
            if o.get("id") == obj_id:
                o["is_priority"] = is_priority
                from datetime import datetime
                o["updated_at"] = datetime.now().isoformat()
                logger.info(f"Updated priority for object {obj_id}: {is_priority}")
                return True
        logger.warning(f"Object with id {obj_id} not found")
        return False
