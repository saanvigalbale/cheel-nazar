"""
Video processing router stub.
Will handle drone video upload, frame extraction, and telemetry parsing.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_videos():
    """Placeholder: List uploaded drone videos."""
    return {"status": "placeholder", "module": "video", "message": "Video processing pipeline ready for implementation."}
