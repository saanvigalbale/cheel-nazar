import re
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class VideoUploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str


class VideoListItem(BaseModel):
    job_id: Optional[str] = None
    filename: str
    size_bytes: int
    modified_time: float


class VideoListResponse(BaseModel):
    total: int
    videos: List[VideoListItem]


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a drone video",
    description="Accepts single-pass drone footage in .mp4, .mov, .avi, or .mkv formats up to 500MB.",
)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    """
    Upload a drone video file for photogrammetric reconstruction and AI analysis.
    
    Validates file extension and size, saves to `uploads/` with a unique job ID,
    and stages the file for subsequent processing stages.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if not file_ext or file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        allowed_list = ", ".join(sorted(settings.ALLOWED_VIDEO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_list}",
        )

    # Generate unique job ID and sanitized destination filename
    job_id = str(uuid.uuid4())
    raw_stem = Path(file.filename).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_stem).strip("_")
    if not safe_stem:
        safe_stem = "drone_video"
    stored_filename = f"{job_id}_{safe_stem}{file_ext}"

    # Ensure uploads directory exists
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = settings.UPLOADS_DIR / stored_filename

    # Stream file to disk while enforcing max file size limit
    chunk_size = 1024 * 1024  # 1 MB chunk
    bytes_written = 0

    try:
        with open(destination_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    destination_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed upload size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded video: {str(exc)}",
        ) from exc

    if bytes_written == 0:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty (0 bytes).",
        )

    # Modular hook:
    # Future pipeline will trigger:
    # video -> frame extraction -> reconstruction -> AI analysis
    # e.g., await queue_reconstruction_pipeline(job_id=job_id, video_path=destination_path)

    return VideoUploadResponse(
        job_id=job_id,
        filename=stored_filename,
        status="uploaded",
        message="Video uploaded successfully",
    )


@router.get(
    "/",
    response_model=VideoListResponse,
    summary="List uploaded videos",
    description="Retrieve all uploaded drone videos stored in the staging directory.",
)
async def list_videos() -> VideoListResponse:
    """List uploaded drone video files in the staging directory."""
    if not settings.UPLOADS_DIR.exists():
        return VideoListResponse(total=0, videos=[])

    items: List[VideoListItem] = []
    for p in settings.UPLOADS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in settings.ALLOWED_VIDEO_EXTENSIONS:
            # Extract job_id if prefix matches UUID format
            parts = p.name.split("_", 1)
            job_id = parts[0] if len(parts) > 1 and len(parts[0]) == 36 else None
            stat = p.stat()
            items.append(
                VideoListItem(
                    job_id=job_id,
                    filename=p.name,
                    size_bytes=stat.st_size,
                    modified_time=stat.st_mtime,
                )
            )

    items.sort(key=lambda x: x.modified_time, reverse=True)
    return VideoListResponse(total=len(items), videos=items)
