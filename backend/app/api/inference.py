from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
import logging
import numpy as np
from app.core.logging import setup_logging, get_logger
from app.core.config import settings
from app.models.yolo_detector import YOLODetector
from app.models.face_recognition_module.face_detector import FaceDetector
from app.models.face_recognition_module.face_recognizer import FaceRecognizer
from app.models.object_recognition_module.object_recognizer import ObjectRecognizer
from app.models.midas_depth import MiDaSDepthEstimator
from app.models.paddle_ocr import PaddleOCRDetector
from app.services.preprocessing import ImagePreprocessor
from app.services.postprocessing import DetectionPostprocessor

from app.services.audio_feedback import AudioFeedbackService
from app.services.navigation_guidance import NavigationGuidanceService
from app.schemas.detection import (
    ObjectDetectionResponse, FaceDetectionResponse, 
    TextDetectionResponse, DepthEstimationResponse, NavigationGuidanceResponse
)

_RECOGNITION_THRESHOLD = 0.50  # minimum cosine similarity to call it a match

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Initialize models (will be loaded on first request)
yolo_detector = None
face_detector = None
face_recognizer = None
object_recognizer = None

router = APIRouter()


def _score_text_results(text_results):
    """Score OCR result quality; higher is better for readable speech output."""
    if not text_results:
        return 0.0

    total_conf = 0.0
    total_chars = 0
    alpha_chars = 0
    alnum_regions = 0

    for result in text_results:
        text = (result.text or "").strip()
        if not text:
            continue

        total_conf += float(result.confidence)
        total_chars += len(text)
        alpha_chars += sum(1 for ch in text if ch.isalpha())
        if any(ch.isalnum() for ch in text):
            alnum_regions += 1

    if alnum_regions == 0:
        return 0.0

    avg_conf = total_conf / alnum_regions
    # Favor readable, alphabetic content over tiny numeric fragments.
    return (alpha_chars * 2.5) + total_chars + (avg_conf * 5.0) + (alnum_regions * 1.5)


def _best_oriented_ocr_results(ocr_detector, image_np):
    """Run OCR on 0/90/180/270 orientations and keep the best-scoring result."""
    candidates = [
        (0, image_np),
        (90, np.rot90(image_np, 1)),
        (180, np.rot90(image_np, 2)),
        (270, np.rot90(image_np, 3)),
    ]

    best_score = -1.0
    best_angle = 0
    best_results = []

    for angle, candidate in candidates:
        try:
            results = ocr_detector.detect_text(candidate)
        except Exception as e:
            logger.warning(f"OCR failed for rotation {angle}°: {str(e)}")
            continue

        score = _score_text_results(results)
        logger.info(
            f"OCR rotation {angle}°: regions={len(results)} score={score:.2f}"
        )
        if score > best_score:
            best_score = score
            best_angle = angle
            best_results = results

    logger.info(
        f"Selected OCR rotation {best_angle}° with score={best_score:.2f} and {len(best_results)} regions"
    )
    return best_results

def get_yolo_detector():
    """Get YOLO detector instance, loading it if necessary."""
    global yolo_detector
    if yolo_detector is None:
        try:
            yolo_detector = YOLODetector()
            logger.info("YOLO detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {str(e)}")
            raise HTTPException(status_code=500, detail="YOLO model not available")
    return yolo_detector

def get_face_detector():
    """Get face detector instance, loading it if necessary."""
    global face_detector
    if face_detector is None:
        try:
            face_detector = FaceDetector()
            logger.info("Face detector initialized (YOLOv8n-face ONNX)")
        except Exception as e:
            logger.error(f"Face detector initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Face detection model not available: {e}")
    return face_detector

def get_face_recognizer():
    """Get face recognizer instance, loading it if necessary."""
    global face_recognizer
    if face_recognizer is None:
        try:
            face_recognizer = FaceRecognizer()
            logger.info("Face recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize face recognizer: {str(e)}")
            raise HTTPException(status_code=500, detail="Face recognition model not available")
    return face_recognizer

def get_object_recognizer():
    """Get object recognizer instance, loading it if necessary."""
    global object_recognizer
    if object_recognizer is None:
        try:
            object_recognizer = ObjectRecognizer()
            logger.info("Object recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize object recognizer: {str(e)}")
            raise HTTPException(status_code=500, detail="Object recognizer not available")
    return object_recognizer

@router.post("/objects", response_model=ObjectDetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    return_audio: bool = Query(default=False, description="Generate audio feedback")
):
    """
    Detect objects in an uploaded image using YOLO model.
    
    Args:
        file: Image file (JPEG/PNG)
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        return_audio: Whether to generate audio description
    
    Returns:
        ObjectDetectionResponse with detection results
    """
    start_time = time.time()
    
    try:
        # Validate and load image
        image_bytes = await file.read()
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        
        # Get detector and run inference
        detector = get_yolo_detector()
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            import cv2
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detections = detector.detect(image_np)
        
        # Filter by confidence threshold
        filtered_detections = DetectionPostprocessor.filter_detections_by_confidence(
            detections, confidence_threshold
        )
        
        # Format response
        inference_time = time.time() - start_time
        response = ObjectDetectionResponse(
            detections=filtered_detections,
            inference_time_ms=round(inference_time * 1000, 2)
        )
        
        # Generate audio feedback if requested
        if return_audio:
            try:
                description = AudioFeedbackService.generate_object_description(
                    [d.dict() for d in filtered_detections]
                )
                audio_path = AudioFeedbackService.text_to_speech(description)
                if audio_path:
                    response.audio_description = description
                    response.audio_file = audio_path
            except Exception as e:
                logger.warning(f"Audio feedback generation failed: {str(e)}")
        
        logger.info(f"Object detection completed: {len(filtered_detections)} objects in {inference_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Object detection failed")

@router.post("/detect")
async def detect(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Simple object detection endpoint - detects objects using YOLO.
    Returns JSON with detections list.
    
    Args:
        file: Image file (JPEG/PNG)
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
    
    Returns:
        JSON with detection results
    """
    start_time = time.time()
    
    try:
        # Validate and load image
        image_bytes = await file.read()
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        
        # Get detector and run inference
        detector = get_yolo_detector()
        detections = detector.detect(image)
        
        # Filter by confidence threshold
        filtered = [d for d in detections if d.confidence >= confidence_threshold]
        
        # Format response
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": {
                        "left": d.bbox.left,
                        "top": d.bbox.top,
                        "right": d.bbox.right,
                        "bottom": d.bbox.bottom
                    }
                }
                for d in filtered
            ],
            "inference_time_ms": round(inference_time * 1000, 2),
            "num_detections": len(filtered)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple object detection failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "detections": []
        }

@router.post("/faces", response_model=FaceDetectionResponse)
async def detect_faces(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    recognize_faces: bool = Query(default=False, description="Perform face recognition"),
    return_audio: bool = Query(default=False, description="Generate audio feedback")
):
    """
    Detect faces in an uploaded image and optionally perform recognition.
    
    Args:
        file: Image file (JPEG/PNG)
        confidence_threshold: Minimum confidence for face detections (0.0-1.0)
        recognize_faces: Whether to perform face recognition
        return_audio: Whether to generate audio description
    
    Returns:
        FaceDetectionResponse with face detection and recognition results
    """
    start_time = time.time()
    
    try:
        import cv2
        import numpy as np

        # Decode image bytes → BGR numpy array (needed by FaceDetector)
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=422, detail="Could not decode image")

        h, w = frame.shape[:2]

        # Run YOLOv8n-face detector: returns (boxes, kps_list, scores) in pixel coords
        detector = get_face_detector()
        boxes, kps_list, scores = detector.detect(frame)

        # Lazily load recognizer only if needed
        recognizer = get_face_recognizer() if recognize_faces else None

        faces = []
        for box, kps, score in zip(boxes, kps_list, scores):
            if float(score) < confidence_threshold:
                continue

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            face_result = {
                'confidence': float(score),
                'bbox': {
                    'left':   max(0.0, float(x1) / w),
                    'top':    max(0.0, float(y1) / h),
                    'right':  min(1.0, float(x2) / w),
                    'bottom': min(1.0, float(y2) / h),
                },
                'embedding': None,
                'person_id': None,
            }

            if recognize_faces and recognizer is not None:
                try:
                    face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    kps_list_py = kps.tolist() if hasattr(kps, 'tolist') else list(kps)
                    person_id = recognizer.recognize_face(
                        face_crop,
                        threshold=_RECOGNITION_THRESHOLD,
                        frame=frame,
                        coarse_box=[x1, y1, x2, y2],
                        coarse_kps=kps_list_py,
                    )
                    face_result['person_id'] = person_id
                except Exception as e:
                    logger.warning(f"Face recognition failed for one face: {e}")

            faces.append(face_result)

        inference_time = time.time() - start_time
        response = FaceDetectionResponse(
            faces=faces,
            inference_time_ms=round(inference_time * 1000, 2)
        )

        logger.info(f"Face detection completed: {len(faces)} faces in {inference_time:.3f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {e}")

@router.post("/all")
async def detect_all(
    file: UploadFile = File(..., description="Image file to analyze"),
    object_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Object detection confidence threshold"),
    face_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Face detection confidence threshold"),
    recognize_faces: bool = Query(default=False, description="Perform face recognition"),
    return_audio: bool = Query(default=False, description="Generate audio feedback")
):
    """
    Perform both object and face detection on an uploaded image.
    
    Args:
        file: Image file (JPEG/PNG)
        object_confidence: Minimum confidence for object detections
        face_confidence: Minimum confidence for face detections
        recognize_faces: Whether to perform face recognition
        return_audio: Whether to generate audio description
    
    Returns:
        Combined detection results
    """
    start_time = time.time()
    
    try:
        import cv2 as _cv2
        import numpy as _np

        image_bytes = await file.read()

        # Object detection path
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        object_processed = ImagePreprocessor.preprocess_for_yolo(image)
        object_detector = get_yolo_detector()
        object_detections = object_detector.detect(object_processed)
        filtered_objects = DetectionPostprocessor.filter_detections_by_confidence(
            object_detections, object_confidence
        )

        # Face detection path — decode BGR frame directly
        nparr = _np.frombuffer(image_bytes, _np.uint8)
        frame = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image for face detection")

        face_det = get_face_detector()
        boxes, kps_list_all, scores_all = face_det.detect(frame)
        h_f, w_f = frame.shape[:2]

        recognizer = get_face_recognizer() if recognize_faces else None

        faces = []
        for box, kps, score in zip(boxes, kps_list_all, scores_all):
            if float(score) < face_confidence:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_result = {
                'confidence': float(score),
                'bbox': {
                    'left':   max(0.0, float(x1) / w_f),
                    'top':    max(0.0, float(y1) / h_f),
                    'right':  min(1.0, float(x2) / w_f),
                    'bottom': min(1.0, float(y2) / h_f),
                },
                'embedding': None,
                'person_id': None,
            }
            if recognize_faces and recognizer is not None:
                try:
                    face_crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                    person_id = recognizer.recognize_face(
                        face_crop,
                        threshold=_RECOGNITION_THRESHOLD,
                        frame=frame,
                        coarse_box=[x1, y1, x2, y2],
                        coarse_kps=kps.tolist() if hasattr(kps, 'tolist') else list(kps),
                    )
                    face_result['person_id'] = person_id
                except Exception as e:
                    logger.warning(f"Face recognition in /all failed: {e}")
            faces.append(face_result)
        
        # Generate combined results
        inference_time = time.time() - start_time
        
        result = {
            "objects": [d.dict() for d in filtered_objects],
            "faces": faces,
            "inference_time_ms": round(inference_time * 1000, 2),
            "total_detections": len(filtered_objects) + len(faces)
        }
        
        # Generate audio feedback if requested
        if return_audio:
            try:
                description = AudioFeedbackService.generate_combined_description(
                    [d.dict() for d in filtered_objects], faces
                )
                audio_path = AudioFeedbackService.text_to_speech(description)
                if audio_path:
                    result["audio_description"] = description
                    result["audio_file"] = audio_path
            except Exception as e:
                logger.warning(f"Audio feedback generation failed: {str(e)}")
        
        logger.info(f"Combined detection completed: {len(filtered_objects)} objects, "
                   f"{len(faces)} faces in {inference_time:.3f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Detection failed")

@router.post("/recognition-personalized")
async def detect_recognition_personalized(
    file: UploadFile = File(..., description="Image file to analyze"),
    object_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Object detection confidence threshold"),
    face_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Face detection confidence threshold"),
    recognize_faces: bool = Query(default=True, description="Perform face recognition"),
    registered_only: bool = Query(default=True, description="Return only registered objects")
):
    """
    Personalized recognition mode endpoint.

    Returns:
      - face recognition results (known + unknown faces)
      - only registered object matches (by label in Phase 1)
    """
    start_time = time.time()

    try:
        import cv2 as _cv2
        import numpy as _np

        image_bytes = await file.read()

        # Object detections
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        object_processed = ImagePreprocessor.preprocess_for_yolo(image)
        object_detector = get_yolo_detector()
        object_detections = object_detector.detect(object_processed)
        filtered_objects = DetectionPostprocessor.filter_detections_by_confidence(
            object_detections, object_confidence
        )

        object_detections_dict = [d.dict() for d in filtered_objects]
        obj_recognizer = get_object_recognizer()
        recognized_objects, unrecognized_objects = obj_recognizer.recognize_objects(
            object_detections_dict,
            registered_only=registered_only,
            confidence_threshold=object_confidence,
        )

        # Face detections + recognition
        nparr = _np.frombuffer(image_bytes, _np.uint8)
        frame = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image for face detection")

        face_det = get_face_detector()
        boxes, kps_list_all, scores_all = face_det.detect(frame)
        h_f, w_f = frame.shape[:2]

        recognizer = get_face_recognizer() if recognize_faces else None

        faces = []
        for box, kps, score in zip(boxes, kps_list_all, scores_all):
            if float(score) < face_confidence:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_result = {
                'confidence': float(score),
                'bbox': {
                    'left':   max(0.0, float(x1) / w_f),
                    'top':    max(0.0, float(y1) / h_f),
                    'right':  min(1.0, float(x2) / w_f),
                    'bottom': min(1.0, float(y2) / h_f),
                },
                'embedding': None,
                'person_id': None,
            }
            if recognize_faces and recognizer is not None:
                try:
                    face_crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                    person_id = recognizer.recognize_face(
                        face_crop,
                        threshold=_RECOGNITION_THRESHOLD,
                        frame=frame,
                        coarse_box=[x1, y1, x2, y2],
                        coarse_kps=kps.tolist() if hasattr(kps, 'tolist') else list(kps),
                    )
                    face_result['person_id'] = person_id
                except Exception as e:
                    logger.warning(f"Face recognition in /recognition-personalized failed: {e}")
            faces.append(face_result)

        inference_time = time.time() - start_time

        result = {
            "faces": faces,
            "recognized_objects": recognized_objects,
            "unrecognized_objects": unrecognized_objects,
            "registered_only": registered_only,
            "inference_time_ms": round(inference_time * 1000, 2),
            "total_detections": len(faces) + len(recognized_objects),
        }

        logger.info(
            f"Personalized recognition completed: {len(faces)} faces, "
            f"{len(recognized_objects)} registered objects in {inference_time:.3f}s"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Personalized recognition failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Personalized recognition failed")

@router.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    try:
        yolo_info = get_yolo_detector().get_model_info() if yolo_detector else None
        face_info = get_face_detector().get_model_info() if face_detector else None
        recognizer_info = get_face_recognizer().get_model_info() if face_recognizer else None
        
        return {
            "yolo_detector": yolo_info,
            "face_detector": face_info,
            "face_recognizer": recognizer_info,
            "status": "models_loaded" if all([yolo_info, face_info, recognizer_info]) else "models_not_loaded"
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.post("/depth", response_model=DepthEstimationResponse)
async def estimate_depth(
    file: UploadFile = File(..., description="Image file to analyze"),
    return_audio: bool = Query(default=False, description="Generate audio feedback")
):
    """
    Estimate depth from an uploaded image using MiDaS model.
    
    Args:
        file: Image file (JPEG/PNG)
        return_audio: Whether to generate audio description
    
    Returns:
        DepthEstimationResponse with depth map and analysis
    """
    start_time = time.time()
    
    try:
        # Validate and load image
        image_bytes = await file.read()
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        
        # Initialize depth estimator
        try:
            depth_estimator = MiDaSDepthEstimator()
        except Exception as e:
            logger.error(f"Failed to initialize MiDaS depth estimator: {str(e)}")
            raise HTTPException(status_code=500, detail="Depth estimation model not available")
        
        # Run depth estimation
        depth_result = depth_estimator.estimate_depth(np.array(image))
        
        # Format response
        inference_time = time.time() - start_time
        response = DepthEstimationResponse(
            depth_map=depth_result.depth_map,
            min_depth=depth_result.min_depth,
            max_depth=depth_result.max_depth,
            mean_depth=depth_result.mean_depth,
            inference_time_ms=round(inference_time * 1000, 2)
        )
        
        # Generate audio feedback if requested
        if return_audio:
            try:
                description = AudioFeedbackService.generate_depth_description(
                    depth_result.min_depth, depth_result.max_depth, depth_result.mean_depth
                )
                audio_path = AudioFeedbackService.text_to_speech(description)
                if audio_path:
                    response.audio_description = description
                    response.audio_file = audio_path
            except Exception as e:
                logger.warning(f"Audio feedback generation failed: {str(e)}")
        
        logger.info(f"Depth estimation completed: {inference_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Depth estimation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Depth estimation failed")

@router.post("/text", response_model=TextDetectionResponse)
async def detect_text(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    return_audio: bool = Query(default=False, description="Generate audio feedback")
):
    """
    Detect text in an uploaded image using PaddleOCR.
    
    Args:
        file: Image file (JPEG/PNG)
        confidence_threshold: Minimum confidence for text detections (0.0-1.0)
        return_audio: Whether to generate audio description
    
    Returns:
        TextDetectionResponse with detected text regions
    """
    start_time = time.time()
    
    try:
        # Validate and load image
        image_bytes = await file.read()
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        
        # Initialize OCR detector
        try:
            ocr_detector = PaddleOCRDetector()
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR detector: {str(e)}")
            raise HTTPException(status_code=500, detail="OCR model not available")
        
        # Run text detection with orientation search to handle rotated camera captures.
        image_np = np.array(image)
        text_results = _best_oriented_ocr_results(ocr_detector, image_np)
        
        # Filter by confidence threshold
        filtered_results = [result for result in text_results if result.confidence >= confidence_threshold]
        
        # Format response
        inference_time = time.time() - start_time
        response = TextDetectionResponse(
            text_regions=filtered_results,
            inference_time_ms=round(inference_time * 1000, 2)
        )
        
        # Generate audio feedback if requested
        if return_audio:
            try:
                description = AudioFeedbackService.generate_text_description(filtered_results)
                audio_path = AudioFeedbackService.text_to_speech(description)
                if audio_path:
                    response.audio_description = description
                    response.audio_file = audio_path
            except Exception as e:
                logger.warning(f"Audio feedback generation failed: {str(e)}")
        
        logger.info(f"Text detection completed: {len(filtered_results)} text regions in {inference_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Text detection failed")

@router.post("/navigation", response_model=NavigationGuidanceResponse)
async def get_navigation_guidance(
    file: UploadFile = File(..., description="Image file to analyze"),
    object_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Object detection confidence threshold"),
    text_confidence: float = Query(default=0.3, ge=0.0, le=1.0, description="Text detection confidence threshold"),
    return_audio: bool = Query(default=True, description="Generate audio feedback")
):
    """
    Generate navigation guidance using fusion of YOLO + MiDaS + OCR.
    
    Args:
        file: Image file (JPEG/PNG)
        object_confidence: Minimum confidence for object detections
        text_confidence: Minimum confidence for text detections
        return_audio: Whether to generate audio description
    
    Returns:
        NavigationGuidanceResponse with comprehensive navigation guidance
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("NAVIGATION GUIDANCE REQUEST RECEIVED")
    logger.info("=" * 60)
    
    # DEBUG: Log incoming file metadata
    logger.info(f"[DEBUG] File object type: {type(file)}")
    logger.info(f"[DEBUG] File.filename: {file.filename}")
    logger.info(f"[DEBUG] File.content_type: {file.content_type}")
    logger.info(f"[DEBUG] File size (if available): {file.size if hasattr(file, 'size') else 'N/A'}")
    
    try:
        # Log request details
        logger.info(f"File: {file.filename}")
        logger.info(f"Content-Type: {file.content_type}")
        logger.info(f"Object confidence threshold: {object_confidence}")
        logger.info(f"Text confidence threshold: {text_confidence}")
        logger.info(f"Return audio: {return_audio}")
        
        # Read file
        logger.info("Stage 1/5: Reading uploaded file...")
        image_bytes = await file.read()
        logger.info(f"✓ File read successfully: {len(image_bytes)} bytes")
        
        # Validate and load image
        logger.info("Stage 2/5: Validating and loading image...")
        try:
            image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
            logger.info(f"✓ Image validated: size={image.size}, mode={image.mode}")
        except HTTPException as e:
            logger.error(f"✗ Image validation failed: {e.detail}")
            raise
        except Exception as e:
            logger.error(f"✗ Image loading failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image loading failed: {str(e)}")
        
        # Initialize navigation guidance service
        logger.info("Stage 3/5: Initializing navigation guidance service...")
        try:
            guidance_service = NavigationGuidanceService()
            logger.info("✓ Navigation service initialized")
        except Exception as e:
            logger.error(f"✗ Service initialization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Navigation guidance service initialization failed: {str(e)}")
        
        # Run navigation guidance
        logger.info("Stage 4/5: Running navigation guidance pipeline...")
        try:
            guidance_result = guidance_service.get_navigation_guidance(
                image_bytes,
                file.content_type,
                object_confidence=object_confidence,
                text_confidence=text_confidence,
            )
            logger.info(f"✓ Navigation guidance completed")
            logger.info(f"  - Obstacles detected: {len(guidance_result.obstacles)}")
            logger.info(f"  - Text signs detected: {len(guidance_result.text_signs)}")
            logger.info(f"  - Safety warnings: {len(guidance_result.safety_warnings)}")
            logger.info(f"  - Guidance: {guidance_result.guidance}")
        except Exception as e:
            logger.error(f"✗ Navigation guidance pipeline failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Navigation guidance failed: {str(e)}")
        
        # Format response - wrap the result in the response model
        logger.info("Stage 5/5: Formatting response...")
        inference_time = time.time() - start_time
        response = NavigationGuidanceResponse(
            guidance=guidance_result,
            inference_time_ms=round(inference_time * 1000, 2)
        )
        logger.info(f"✓ Response formatted successfully")
        
        # Generate audio feedback if requested
        if return_audio:
            logger.info("Generating audio feedback...")
            try:
                description = guidance_result.guidance
                audio_path = AudioFeedbackService.text_to_speech(description)
                if audio_path:
                    response.guidance.audio_description = description
                    response.guidance.audio_file = audio_path
                    logger.info(f"✓ Audio feedback generated: {audio_path}")
            except Exception as e:
                logger.warning(f"⚠ Audio feedback generation failed: {str(e)}")
        
        logger.info(f"✓ NAVIGATION GUIDANCE COMPLETED: {inference_time:.3f}s")
        logger.info("=" * 60)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ NAVIGATION GUIDANCE FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=" * 60)
        raise HTTPException(status_code=500, detail=f"Navigation guidance failed: {str(e)}")

@router.get("/navigation/models")
async def get_navigation_models_status():
    """Get status of all models used in navigation guidance."""
    try:
        guidance_service = NavigationGuidanceService()
        status = guidance_service.get_model_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get navigation models status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model status")
