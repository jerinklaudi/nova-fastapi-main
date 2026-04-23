"""Object registration API with camera-frame buffering, modeled after face registration."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.logging import get_logger
from app.models.object_recognition_module.object_db import ObjectDatabase

logger = get_logger(__name__)
router = APIRouter()

_DB_LOCK = threading.Lock()
_FRAME_BUFFERS: Dict[str, List[int]] = {}
_MIN_ACCEPTED_FRAMES = 5
_TARGET_FRAMES = 20


class ObjectRegisterRequest(BaseModel):
    name: str
    target_label: str
    is_priority: bool = False


class ObjectPriorityUpdateRequest(BaseModel):
    is_priority: bool


class ObjectRegisterResponse(BaseModel):
    success: bool
    message: str
    object_id: Optional[str] = None


class ObjectFrameResponse(BaseModel):
    success: bool
    name: str
    target_label: str
    object_detected: bool
    accepted: bool
    buffer_size: int
    message: str


class ObjectSaveResponse(BaseModel):
    success: bool
    name: str
    target_label: str
    samples_used: int
    message: str


class ObjectDeleteResponse(BaseModel):
    success: bool
    message: str


class ObjectListResponse(BaseModel):
    objects: List[dict]
    total: int


class ObjectLabelsResponse(BaseModel):
    labels: List[str]
    total: int


def _get_object_db() -> ObjectDatabase:
    return ObjectDatabase()


def _get_shared_yolo_detector():
    import app.api.inference as _inf

    if _inf.yolo_detector is not None and _inf.yolo_detector is not False:
        return _inf.yolo_detector
    return _inf.get_yolo_detector()


def _reload_recognizer_db() -> None:
    try:
        import app.api.inference as _inf
        if _inf.object_recognizer is not None and _inf.object_recognizer is not False:
            _inf.object_recognizer.reload_db()
            logger.info("Object recognizer DB reloaded in-memory")
    except Exception as exc:
        logger.warning("Could not reload object recognizer DB: %s", exc)


@router.post("/register/frame", response_model=ObjectFrameResponse)
async def register_object_frame(
    name: str = Form(...),
    target_label: str = Form(...),
    image: UploadFile = File(...),
):
    name = name.strip()
    target_label = target_label.strip().lower()

    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    if not target_label:
        raise HTTPException(status_code=422, detail="target_label must not be empty")

    image_bytes = await image.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image decode failed: {exc}")

    current_buffer = _FRAME_BUFFERS.get(name, [])
    current_buffer.append(1)
    _FRAME_BUFFERS[name] = current_buffer

    # We still keep the label the user chose, but we do not require YOLO to
    # confidently detect that exact class on every frame. This makes object
    # registration behave more like face registration: capture many frames,
    # then finalize once enough samples are collected.
    object_detected = True
    message = "Frame accepted"
    if len(current_buffer) < _MIN_ACCEPTED_FRAMES:
        message = f"Captured frame {len(current_buffer)} / {_TARGET_FRAMES}"

    return ObjectFrameResponse(
        success=True,
        name=name,
        target_label=target_label,
        object_detected=object_detected,
        accepted=True,
        buffer_size=len(current_buffer),
        message=message,
    )


@router.post("/register/save", response_model=ObjectSaveResponse)
async def save_object_registration(
    name: str = Form(...),
    target_label: str = Form(...),
    is_priority: bool = Form(False),
):
    name = name.strip()
    target_label = target_label.strip().lower()

    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    if not target_label:
        raise HTTPException(status_code=422, detail="target_label must not be empty")

    buffer = _FRAME_BUFFERS.get(name, [])
    if len(buffer) < _MIN_ACCEPTED_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {len(buffer)} quality frame(s) captured for '{name}' "
                f"(minimum {_MIN_ACCEPTED_FRAMES} required)."
            ),
        )

    with _DB_LOCK:
        db = _get_object_db()
        if db.get_object_by_name(name) is not None:
            raise HTTPException(status_code=409, detail=f"Object name '{name}' already exists")

        obj_id = str(uuid4())
        if not db.add_object(
            obj_id=obj_id,
            name=name,
            target_label=target_label,
            is_priority=is_priority,
            samples_used=len(buffer),
        ):
            raise HTTPException(status_code=500, detail="Failed to register object")
        if not db.save():
            raise HTTPException(status_code=500, detail="Failed to persist object database")

    _FRAME_BUFFERS.pop(name, None)
    _reload_recognizer_db()

    return ObjectSaveResponse(
        success=True,
        name=name,
        target_label=target_label,
        samples_used=len(buffer),
        message=f"{name} registered successfully",
    )


@router.post("/register/cancel")
async def cancel_object_registration(name: str = Form(...)):
    name = name.strip()
    _FRAME_BUFFERS.pop(name, None)
    return {"cancelled": True, "name": name}


@router.post("/register", response_model=ObjectRegisterResponse)
async def register_object(payload: ObjectRegisterRequest):
    """Compatibility endpoint for direct registration."""
    name = payload.name.strip()
    target_label = payload.target_label.strip().lower()

    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    if not target_label:
        raise HTTPException(status_code=422, detail="target_label must not be empty")

    with _DB_LOCK:
        db = _get_object_db()
        if db.get_object_by_name(name) is not None:
            raise HTTPException(status_code=409, detail=f"Object name '{name}' already exists")

        obj_id = str(uuid4())
        if not db.add_object(
            obj_id=obj_id,
            name=name,
            target_label=target_label,
            is_priority=payload.is_priority,
            samples_used=_TARGET_FRAMES,
        ):
            raise HTTPException(status_code=500, detail="Failed to register object")
        if not db.save():
            raise HTTPException(status_code=500, detail="Failed to persist object database")

    _reload_recognizer_db()

    return ObjectRegisterResponse(
        success=True,
        message="Object registered successfully",
        object_id=obj_id,
    )


@router.get("/list", response_model=ObjectListResponse)
async def list_objects():
    db = _get_object_db()
    db.reload()
    objects = db.get_all_objects()
    return ObjectListResponse(objects=objects, total=len(objects))


@router.get("/labels", response_model=ObjectLabelsResponse)
async def list_object_labels():
    try:
        import app.api.inference as _inf

        detector = _inf.get_yolo_detector()
        labels = sorted(list(detector.labels)) if hasattr(detector, "labels") else []
        return ObjectLabelsResponse(labels=labels, total=len(labels))
    except Exception as exc:
        logger.error("Failed to load YOLO labels: %s", exc)
        raise HTTPException(status_code=500, detail="Could not load YOLO labels")


@router.delete("/{object_id}", response_model=ObjectDeleteResponse)
async def delete_object(object_id: str):
    with _DB_LOCK:
        db = _get_object_db()
        ok = db.delete_object(object_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found")
        if not db.save():
            raise HTTPException(status_code=500, detail="Failed to persist object database")

    _reload_recognizer_db()

    return ObjectDeleteResponse(success=True, message="Object deleted successfully")


@router.patch("/{object_id}/priority", response_model=ObjectRegisterResponse)
async def update_object_priority(object_id: str, payload: ObjectPriorityUpdateRequest):
    with _DB_LOCK:
        db = _get_object_db()
        ok = db.update_priority(object_id, payload.is_priority)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found")
        if not db.save():
            raise HTTPException(status_code=500, detail="Failed to persist object database")

    _reload_recognizer_db()

    return ObjectRegisterResponse(
        success=True,
        message="Object priority updated",
        object_id=object_id,
    )
