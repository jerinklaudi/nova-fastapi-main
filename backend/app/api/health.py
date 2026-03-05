from fastapi import APIRouter
from app.core.config import settings
from app.schemas.detection import HealthResponse

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=["YOLO", "Face Detection", "Face Recognition"]
    )

@router.get("/status")
async def status():
    """Detailed status endpoint."""
    return {
        "status": "running",
        "version": settings.VERSION,
        "project_name": settings.PROJECT_NAME,
        "api_version": settings.VERSION,
        "models": {
            "yolo": {
                "path": settings.YOLO_MODEL_PATH,
                "status": "loaded" if hasattr(settings, '_yolo_loaded') else "not loaded"
            },
            "face_detection": {
                "path": settings.FACE_MODEL_PATH,
                "status": "loaded" if hasattr(settings, '_face_loaded') else "not loaded"
            }
        }
    }