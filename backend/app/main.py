from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, inference
from app.core.config import settings
from app.api.inference import get_navigation_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Vision Assistant Backend API"
)

@app.on_event("startup")
async def startup_event():
    """Eagerly load models and run warm-up inference at server startup."""
    print("[NOVA API] Starting background model initialization & warm-up...")
    try:
        service = get_navigation_service()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)