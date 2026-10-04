from fastapi import APIRouter
from app.api.v1.endpoints import health, video, reconstruction, analysis, ai

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(video.router, prefix="/video", tags=["Video Processing"])
api_router.include_router(reconstruction.router, prefix="/reconstruction", tags=["3D Reconstruction"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["AI Analysis"])
api_router.include_router(ai.router, prefix="/ai", tags=["AI Detection"])
