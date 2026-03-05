from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
import logging
import numpy as np
from app.core.logging import setup_logging, get_logger
from app.core.config import settings
from app.models.yolo_detector import YOLODetector
from app.models.face_detector import FaceDetector
from app.models.sface_detector import SFaceDetector
from app.models.midas_depth import MiDaSDepthEstimator
from app.models.paddle_ocr import PaddleOCRDetector
from app.services.preprocessing import ImagePreprocessor
from app.services.postprocessing import DetectionPostprocessor

from app.services.navigation_guidance import NavigationGuidanceService
import cv2
from app.schemas.detection import (
    ObjectDetectionResponse, FaceDetectionResponse, 
    TextDetectionResponse, DepthEstimationResponse, NavigationGuidanceResponse,
    NavigationGuidanceResult, DetectionResult, BoundingBox
)
class FaceRecognizer:
    def __init__(self, *args, **kwargs):
        pass

    def recognize(self, *args, **kwargs):
        return []

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Initialize models (will be loaded on first request)
yolo_detector = None
face_detector = None
sface_detector = None
face_recognizer = None
navigation_service = None

router = APIRouter()

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
            logger.info("Face detector initialized")
        except Exception as e:
            logger.warning(f"Face detector initialization failed: {str(e)}")
            logger.warning("Face detection will be skipped")
            face_detector = False  # Mark as unavailable
    
    if face_detector is False:
        raise HTTPException(status_code=501, detail="Face detection model not available - disabled for this session")
    return face_detector

def get_sface_detector():
    """Get SFace detector instance, loading it if necessary."""
    global sface_detector
    if sface_detector is None:
        try:
            sface_detector = SFaceDetector()
            logger.info("SFace detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SFace detector: {str(e)}")
            raise HTTPException(status_code=500, detail="SFace model not available")
    return sface_detector

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

def get_navigation_service():
    """Get NavigationGuidanceService instance (loaded once, reused per request)."""
    global navigation_service
    if navigation_service is None:
        try:
            navigation_service = NavigationGuidanceService()
            logger.info("NavigationGuidanceService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NavigationGuidanceService: {str(e)}")
            raise HTTPException(status_code=500, detail="Navigation models not available")
    return navigation_service

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
        
        # Preprocess image for YOLO
        processed_image = ImagePreprocessor.preprocess_for_yolo(image)
        
        # Get detector and run inference
        detector = get_yolo_detector()
        detections = detector.detect(processed_image)
        
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
        # Validate and load image
        image_bytes = await file.read()
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        
        # Preprocess image for face detection
        processed_image = ImagePreprocessor.preprocess_for_face(image)
        
        # Get detector and run inference
        detector = get_face_detector()
        face_detections = detector.detect_faces(processed_image)
        
        # Process face detections
        faces = []
        for detection in face_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Create face detection result
            face_result = {
                'confidence': confidence,
                'bbox': {
                    'left': bbox[0],
                    'top': bbox[1],
                    'right': bbox[2],
                    'bottom': bbox[3]
                },
                'embedding': None,
                'person_id': None
            }
            
            # Perform face recognition if requested and confidence is high enough
            if recognize_faces and confidence >= confidence_threshold:
                try:
                    recognizer = get_face_recognizer()
                    
                    # Extract face region for recognition
                    # This is a simplified version - in practice you'd extract the face region
                    embedding = recognizer.extract_embedding(processed_image)
                    person_id = recognizer.recognize_face(processed_image)
                    
                    face_result['embedding'] = embedding.tolist() if embedding is not None else None
                    face_result['person_id'] = person_id
                    
                except Exception as e:
                    logger.warning(f"Face recognition failed: {str(e)}")
            
            faces.append(face_result)
        
        # Format response
        inference_time = time.time() - start_time
        response = FaceDetectionResponse(
            faces=faces,
            inference_time_ms=round(inference_time * 1000, 2)
        )
        
        # Generate audio feedback if requested
        if return_audio:
            try:
                description = AudioFeedbackService.generate_face_description(faces)
                audio_path = AudioFeedbackService.text_to_speech(description)
                if audio_path:
                    response.audio_description = description
                    response.audio_file = audio_path
            except Exception as e:
                logger.warning(f"Audio feedback generation failed: {str(e)}")
        
        logger.info(f"Face detection completed: {len(faces)} faces in {inference_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Face detection failed")

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
        # Validate and load image
        image_bytes = await file.read()
        image = ImagePreprocessor.validate_and_load_image(image_bytes, file.content_type)
        
        # Perform object detection
        object_processed = ImagePreprocessor.preprocess_for_yolo(image)
        object_detector = get_yolo_detector()
        object_detections = object_detector.detect(object_processed)
        filtered_objects = DetectionPostprocessor.filter_detections_by_confidence(
            object_detections, object_confidence
        )
        
        # Perform face detection
        face_processed = ImagePreprocessor.preprocess_for_face(image)
        face_detector = get_face_detector()
        face_detections = face_detector.detect_faces(face_processed)
        
        # Process faces with optional recognition
        faces = []
        for detection in face_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            face_result = {
                'confidence': confidence,
                'bbox': {
                    'left': bbox[0],
                    'top': bbox[1],
                    'right': bbox[2],
                    'bottom': bbox[3]
                },
                'embedding': None,
                'person_id': None
            }
            
            if recognize_faces and confidence >= face_confidence:
                try:
                    recognizer = get_face_recognizer()
                    embedding = recognizer.extract_embedding(face_processed)
                    person_id = recognizer.recognize_face(face_processed)
                    
                    face_result['embedding'] = embedding.tolist() if embedding is not None else None
                    face_result['person_id'] = person_id
                    
                except Exception as e:
                    logger.warning(f"Face recognition failed: {str(e)}")
            
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
            from app.models.paddle_ocr import OCR_AVAILABLE
            if not OCR_AVAILABLE:
                raise HTTPException(status_code=501, detail="OCR service not installed")
            
            ocr_detector = PaddleOCRDetector()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR detector: {str(e)}")
            raise HTTPException(status_code=500, detail="OCR model not available")
        
        # Run text detection
        text_results = ocr_detector.detect_text(np.array(image))
        
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
    return_audio: bool = Query(default=False, description="Generate audio feedback")
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
            nav_service = get_navigation_service()
            logger.info("✓ Navigation service initialized")
        except Exception as e:
            logger.error(f"✗ Service initialization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Navigation guidance service initialization failed: {str(e)}")
        
        # Run navigation guidance
        logger.info("Stage 4/5: Running navigation guidance pipeline...")
        try:
            guidance_result = nav_service.get_navigation_guidance(image_bytes, file.content_type)
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
        logger.info("NAVIGATION RESPONSE SENT SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Inject dynamic fields before sending JSON
        out_data = response.dict()
        out_data["ready"] = True
        out_data["debug_depth"] = getattr(guidance_result, "debug_depth", "N/A")
        return JSONResponse(content=out_data)
        
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
