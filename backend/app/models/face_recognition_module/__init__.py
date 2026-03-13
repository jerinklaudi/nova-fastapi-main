"""
face_recognition_module
───────────────────────
Face detection + recognition sub-package for the NOVA backend.

Used by:
  • backend/app/api/inference.py          — recognition mode inference
  • backend/app/api/face_registration.py  — Flutter settings registration
"""

from .embedding_db    import EmbeddingDatabase
from .face_detector   import FaceDetector, align_face
from .face_recognizer import FaceRecognizer, is_crop_usable

__all__ = [
    "EmbeddingDatabase",
    "FaceDetector",
    "FaceRecognizer",
    "align_face",
    "is_crop_usable",
]
