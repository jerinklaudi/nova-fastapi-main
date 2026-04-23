from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, inference
from app.api import face_registration
from app.api import object_registration
from app.core.config import settings
from app.services.navigation_guidance import NavigationGuidanceService

app = FastAPI(
    title="NOVA API",
    version=settings.VERSION,
    description="AI Vision Assistant Backend API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.on_event("startup")
async def startup_event():
    """Eagerly load models and run warm-up inference at server startup."""
    import os
    print("[NOVA API] Starting background model initialization & warm-up...")
    
    # Verify critical models exist
    critical_models = {
        "YOLO Object Detection": settings.YOLO_MODEL_PATH,
        "Face Detection (YOLOv8n-face)": settings.FACE_MODEL_PATH,
        "Face Recognition (SFace)": settings.SFACE_MODEL_PATH,
        "MiDaS Depth Estimation": settings.MIDAS_MODEL_PATH
    }
    
    for name, path in critical_models.items():
        if os.path.exists(path):
            print(f"[NOVA API] ✓ {name} model found at {path}")
        else:
            print(f"[NOVA API] ⚠️ WARNING: {name} model NOT found at {path}")
    
    try:
        service = NavigationGuidanceService()
        service._warmup_models()
        print("[NOVA API] Navigation models loaded and warmed up successfully.")
    except Exception as e:
        print(f"[NOVA API] Failed to warm up models during startup: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(inference.router, prefix="/detect", tags=["inference"])
app.include_router(face_registration.router,prefix="/faces",tags=["Face Registration"])
app.include_router(object_registration.router, prefix="/objects", tags=["Object Registration"])
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)