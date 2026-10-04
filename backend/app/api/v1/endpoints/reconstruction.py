"""
3D Reconstruction & Photogrammetry router stub.
Will handle COLMAP SfM jobs, MVS dense reconstruction, and point cloud export.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_reconstruction_status():
    """Placeholder: Retrieve 3D reconstruction pipeline jobs."""
    return {"status": "placeholder", "module": "reconstruction", "message": "3D reconstruction pipeline ready for implementation."}
