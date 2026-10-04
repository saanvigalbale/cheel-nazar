"""
AI Detection Endpoint — Phase 5 (Cheel Nazar)
==============================================

POST /api/v1/ai/detect

Accepts a single image upload, validates it, delegates inference to the
detector service, and returns structured JSON detections.

The endpoint is intentionally decoupled from the detector implementation.
Swapping MockDetector → YOLODetector in Phase 6 requires no changes here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.services.ai.detector import BoundingBox, Detection, detector

router = APIRouter()

# ---------------------------------------------------------------------------
# Supported image extensions for validation
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB guard for images


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class BoundingBoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBoxOut
    is_mock: bool


class DetectResponse(BaseModel):
    status: str
    model: str
    image_filename: str
    detections: List[DetectionOut]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detection_to_out(d: Detection) -> DetectionOut:
    return DetectionOut(
        class_name=d.class_name,
        confidence=round(d.confidence, 4),
        bbox=BoundingBoxOut(x1=d.bbox.x1, y1=d.bbox.y1, x2=d.bbox.x2, y2=d.bbox.y2),
        is_mock=d.is_mock,
    )


def _validate_extension(filename: str) -> str:
    """Return the lower-cased extension or raise HTTP 400."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{ext}'. Allowed formats: {allowed}",
        )
    return ext


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Run AI object detection on an image",
    description=(
        "Upload an image file and receive structured object detection results. "
        "⚠ Phase 5 uses a MockDetector — no real model inference is performed. "
        "Real YOLO weights will be integrated in Phase 6."
    ),
)
async def detect_objects(file: UploadFile = File(...)) -> DetectResponse:
    """
    AI Object Detection endpoint.

    Workflow
    --------
    1. Validate filename and extension.
    2. Read upload into memory; enforce size limit.
    3. Write to a secure temporary file.
    4. Delegate to `detector.detect(temp_path)`.
    5. Delete the temporary file.
    6. Return structured JSON.

    Error codes
    -----------
    400 — Missing filename / unsupported extension / empty file / unreadable image.
    413 — Image exceeds the 50 MB size guard.
    500 — Unexpected detector failure.
    """

    # ── 1. Filename & extension validation ──────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in the upload.",
        )

    ext = _validate_extension(file.filename)

    # ── 2. Read upload with size guard ──────────────────────────────────────
    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        ) from exc

    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty (0 bytes).",
        )

    if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the 50 MB limit ({len(raw_bytes) / 1_048_576:.1f} MB).",
        )

    # ── 3. Write to a temporary file ────────────────────────────────────────
    tmp_path: str | None = None
    try:
        # delete=False so we control cleanup explicitly (avoids Windows locking issues)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        # ── 4. Run detector ─────────────────────────────────────────────────
        try:
            detections = detector.detect(tmp_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image file could not be located after upload: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or unreadable image: {exc}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Detector failed unexpectedly: {exc}",
            ) from exc

    finally:
        # ── 5. Cleanup temporary file ────────────────────────────────────────
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # Best-effort; do not mask the original response

    # ── 6. Return structured response ────────────────────────────────────────
    return DetectResponse(
        status="success",
        model=detector.model_name,
        image_filename=file.filename,
        detections=[_detection_to_out(d) for d in detections],
    )
