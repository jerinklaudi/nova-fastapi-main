"""
backend/app/api/face_registration.py
──────────────────────────────────────
FastAPI router for face registration from the Flutter Settings screen.

Endpoints (all under prefix /faces, registered in main.py):
  POST   /faces/register/frame    — send one camera frame, get back accept/reject (always HTTP 200)
  POST   /faces/register/save     — average buffer, write to faces.json, go live
  POST   /faces/register/cancel   — discard buffer
  GET    /faces/list              — list all registered identities
  DELETE /faces/{name}            — remove an identity

Design
──────
Registration is frame-by-frame: Flutter sends frames one at a time.
Each frame is processed through the full detection + landmark + SFace pipeline.
Quality-gated embeddings are buffered in the FaceRecognizer singleton's _reg_buffer.
When Flutter calls /save, the buffer is averaged → saved to faces.json.
The inference.py singleton (face_recognizer) is reused so the _reg_buffer
is shared in the same process — no server restart needed.

IMPORTANT: /register/frame always returns HTTP 200.
  • accepted=true  → face found and quality gate passed
  • accepted=false → no face OR quality gate failed; Flutter shows the message
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.logging import get_logger
from app.models.face_recognition_module.embedding_db import EmbeddingDatabase
from app.models.face_recognition_module.face_detector import FaceDetector, align_face
from app.models.face_recognition_module.face_recognizer import FaceRecognizer, is_crop_usable
from app.services.preprocessing import ImagePreprocessor

logger = get_logger(__name__)
router = APIRouter()

# Serialises concurrent writes to faces.json
_DB_LOCK = threading.Lock()

# Minimum quality frames required before /save is allowed
_MIN_ACCEPTED_FRAMES = 5


# ── Pydantic response models ──────────────────────────────────────────────────

class FrameResponse(BaseModel):
    success: bool
    name: str
    face_detected: bool
    debug_frame: Optional[str] = None
    accepted: bool
    buffer_size: int
    message: str


class SaveResponse(BaseModel):
    success: bool
    name: str
    samples_used: int
    message: str


class CancelResponse(BaseModel):
    cancelled: bool
    name: str


class IdentityInfo(BaseModel):
    name: str
    sample_count: int


class ListResponse(BaseModel):
    identities: List[IdentityInfo]
    total: int


class DeleteResponse(BaseModel):
    success: bool
    name: str
    message: str


# ── Shared singleton access ───────────────────────────────────────────────────

def _get_shared_recognizer() -> FaceRecognizer:
    """
    Returns the FaceRecognizer singleton from inference.py so that the
    _reg_buffer lives in the same instance that serves recognition requests.
    """
    import app.api.inference as _inf
    if _inf.face_recognizer is not None and _inf.face_recognizer is not False:
        return _inf.face_recognizer
    return _inf.get_face_recognizer()


def _get_shared_detector() -> FaceDetector:
    """Returns the FaceDetector singleton from inference.py."""
    import app.api.inference as _inf
    if _inf.face_detector is not None and _inf.face_detector is not False:
        return _inf.face_detector
    return _inf.get_face_detector()


def _reload_inference_db() -> None:
    """
    Reload the EmbeddingDatabase on the FaceRecognizer singleton so the next
    recognition request sees the newly saved identity without a server restart.
    """
    try:
        rec = _get_shared_recognizer()
        rec.db.reload()
        logger.info("FaceRecognizer.db reloaded — new identity is live immediately")
    except Exception as exc:
        logger.warning("Could not reload embedding DB: %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register/frame", response_model=FrameResponse)
async def register_frame(
    name: str = Form(..., description="Person's name"),
    image: UploadFile = File(..., description="One camera frame (JPEG)"),
):
    """
    Process one camera frame for registration.

    Flutter calls this once per captured frame (target: 20 frames).
    ALWAYS returns HTTP 200 — accepted=false if no face found or quality check failed.
    Returns {accepted: bool, buffer_size: int, name: str, message: str} so Flutter
    can show progress and rejection reasons.

    Pipeline per frame:
      decode → BGR → FaceDetector.detect() → align_face() → FaceRecognizer.get_embedding()
      (get_embedding internally runs github_landmark.onnx for refined alignment,
       then the quality gate, then SFace ONNX)
      If embedding is not None → accumulate_registration_frame(name, embedding)
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    # Decode image
    image_bytes = await image.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image decode failed: {exc}")

    recognizer = _get_shared_recognizer()
    buf_size = recognizer.get_registration_buffer_size(name)

    # Detect faces — return 200 with accepted=false if none found
    try:
        detector = _get_shared_detector()
        boxes, kps_list, scores = detector.detect(frame)
        logger.info(f"[TRACE] register_frame: detector found {len(boxes)} faces. Scores: {scores}")
    except Exception as exc:
        logger.error("register_frame: Face detection error: %s", exc, exc_info=True)
        return FrameResponse(
            success=False,
            name=name,
            face_detected=False,
            accepted=False,
            buffer_size=buf_size,
            message=f"Detection error: {exc}",
        )

    if not boxes:
        logger.debug("register_frame: No faces found by FaceDetector for '%s'", name)
        return FrameResponse(
            success=True,
            name=name,
            face_detected=False,
            accepted=False,
            buffer_size=buf_size,
            message="No face detected — keep your face in frame",
        )

    # Use highest-confidence face
    best_idx = int(np.argmax(scores))
    box = boxes[best_idx]
    kps = kps_list[best_idx]
    logger.info(f"[TRACE] register_frame: Selected best face {best_idx} with score {scores[best_idx]} at box {box}")

    # Draw bounding box for debug visualization on the *original* frame
    x1, y1, x2, y2 = map(int, box)
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Encode debug frame to base64
    _, buffer = cv2.imencode('.jpg', debug_frame)
    debug_base64 = base64.b64encode(buffer).decode('utf-8')

    # Align the face (output_size=112 for ArcFace scale)
    aligned = align_face(frame, kps, output_size=112)
    
    # Manually compute quality metrics identical to face_recognizer.py for trace logging
    try:
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        laplace_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_intensity = gray.mean()
        logger.info(f"[TRACE] register_frame: Quality metrics -> Laplacian Var: {laplace_var:.2f}, Mean Intensity: {mean_intensity:.2f}")
    except Exception as e:
        logger.warning(f"[TRACE] register_frame: Could not compute trace quality metrics: {e}")

    if not is_crop_usable(aligned):
        logger.debug("register_frame: Crop failed quality gate for '%s'", name)
        return FrameResponse(
            success=True,
            name=name,
            face_detected=True,
            debug_frame=debug_base64,
            accepted=False,
            buffer_size=buf_size,
            message="Face blurry or too dark — hold steady",
        )

    # get_embedding runs refined landmark alignment (github_landmark.onnx) + SFace
    try:
        embedding = recognizer.get_embedding(
            aligned,
            frame=frame,
            coarse_box=box,
            coarse_kps=kps,
        )
        logger.info(f"[TRACE] register_frame: get_embedding completed -> {'SUCCESS' if embedding is not None else 'NONE RETURNED'}")
    except Exception as exc:
        logger.error("register_frame: Exception during get_embedding: %s", exc, exc_info=True)
        return FrameResponse(
            success=False,
            name=name,
            face_detected=True,
            debug_frame=debug_base64,
            accepted=False,
            buffer_size=buf_size,
            message=f"Embedding error: {exc}",
        )

    if embedding is None:
        logger.debug("register_frame: get_embedding returned None (quality gate) for '%s'", name)
        return FrameResponse(
            success=True,
            name=name,
            face_detected=True,
            debug_frame=debug_base64,
            accepted=False,
            buffer_size=buf_size,
            message="Face crop failed quality checks — try better lighting",
        )

    recognizer.accumulate_registration_frame(name, embedding)
    buf_size = recognizer.get_registration_buffer_size(name)
    logger.info("Registration frame for '%s': ACCEPTED (buffer=%d)", name, buf_size)

    return FrameResponse(
        success=True,
        name=name,
        face_detected=True,
        debug_frame=debug_base64,
        accepted=True,
        buffer_size=buf_size,
        message="Frame accepted",
    )


@router.post("/register/save", response_model=SaveResponse)
async def save_registration(name: str = Form(...)):
    """
    Finalise registration for *name*.

    Averages all buffered embeddings, normalises the result, writes to
    faces.json, and reloads the live DB so the new identity is immediately
    active without restarting the server.

    Requires at least _MIN_ACCEPTED_FRAMES accepted frames.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    recognizer = _get_shared_recognizer()
    buf_size = recognizer.get_registration_buffer_size(name)

    if buf_size < _MIN_ACCEPTED_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {buf_size} quality frame(s) captured for '{name}' "
                f"(minimum {_MIN_ACCEPTED_FRAMES} required). "
                "Retry with better lighting and a front-facing pose."
            ),
        )

    # Average all buffered embeddings
    final_embedding = recognizer.flush_registration_buffer(name)
    if final_embedding is None:
        raise HTTPException(status_code=500, detail="Buffer flush returned None unexpectedly")

    # Write to faces.json under lock
    with _DB_LOCK:
        db = EmbeddingDatabase()
        db.add_embedding(name, final_embedding)
        saved = db.save()

    if not saved:
        raise HTTPException(
            status_code=500,
            detail=(
                "faces.json save was aborted by the collapse-check "
                "(two identities look too similar). Try re-registering."
            ),
        )

    _reload_inference_db()
    logger.info("Registration complete: '%s' saved with %d frames", name, buf_size)

    return SaveResponse(
        success=True,
        name=name,
        samples_used=buf_size,
        message=f"'{name}' registered successfully!",
    )


@router.post("/register/cancel", response_model=CancelResponse)
async def cancel_registration(name: str = Form(...)):
    """Discard buffered frames for *name* (user cancelled the registration flow)."""
    name = name.strip()
    try:
        recognizer = _get_shared_recognizer()
        recognizer.clear_registration_buffer(name)
    except Exception:
        pass  # best-effort
    logger.info("Registration cancelled for '%s'", name)
    return CancelResponse(cancelled=True, name=name)


@router.get("/list", response_model=ListResponse)
async def list_faces():
    """Return all registered identities with their sample counts."""
    db = EmbeddingDatabase()
    identities = [
        IdentityInfo(name=n, sample_count=db.get_sample_count(n))
        for n in db.list_identities()
    ]
    return ListResponse(identities=identities, total=len(identities))


@router.delete("/{name}", response_model=DeleteResponse)
async def delete_face(name: str):
    """Remove an identity from the database."""
    name = name.strip()
    with _DB_LOCK:
        db = EmbeddingDatabase()
        if not db.remove(name):
            raise HTTPException(status_code=404, detail=f"'{name}' not found")
        db.save(force=True)

    _reload_inference_db()
    logger.info("Deleted identity '%s'", name)
    return DeleteResponse(success=True, name=name, message=f"'{name}' deleted")
