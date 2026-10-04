"""
3D Reconstruction endpoints (Phase 8.6).

Serves the REAL reconstruction produced by the ``reconstruction`` package
(Phases 8.1-8.5) and its artifacts. There is no placeholder data here and no
COLMAP dependency.

The reconstruction is LOCAL and UP-TO-SCALE. Georeferencing is reported as
unavailable unless real GPS metadata exists, in which case
``georeferencing.status`` explains exactly what was searched.
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.services.reconstruction_service import (
    ARTIFACT_MEDIA_TYPES,
    PRIMARY_ARTIFACT,
    build_reconstruction_payload,
    ensure_reconstruction,
    resolve_artifact,
)

router = APIRouter()


@router.get(
    "/",
    summary="Reconstruction module capabilities",
    description=(
        "Lists the reconstruction stages available and their coordinate/scale "
        "status. Read-only; no per-job data."
    ),
)
async def get_reconstruction_status() -> Dict[str, Any]:
    """Describe the reconstruction module and how it is served."""
    return {
        "module": "reconstruction",
        "status": "operational",
        "stages": [
            {"phase": "8.1", "name": "two_view_sfm", "implemented": True},
            {"phase": "8.2", "name": "global_sfm", "implemented": True},
            {"phase": "8.3", "name": "dense_backprojection", "implemented": True},
            {"phase": "8.4", "name": "3d_export", "implemented": True},
            {"phase": "8.5", "name": "georeferencing", "implemented": True},
        ],
        "engines": {
            "sfm": "OpenCV SIFT + essential matrix + solvePnPRansac (no COLMAP)",
            "dense": "Depth Anything V2 relative depth, anchored to the sparse cloud",
        },
        "primary_artifact": PRIMARY_ARTIFACT,
        "servable_artifacts": sorted(ARTIFACT_MEDIA_TYPES),
        "coordinate_system": "local_up_to_scale",
        "scale_status": "UP_TO_SCALE_NO_METRIC_UNITS",
        "georeferencing": {
            "available_when": "real GPS/IMU metadata exists for the job",
            "otherwise": "status=unavailable with an evidence-based reason",
        },
        "limitations": [
            "No metres, distances or areas are implied by the reconstruction.",
            "No GPS coordinates are produced without real flight metadata.",
        ],
    }


@router.get(
    "/{job_id}",
    summary="Reconstruction result for a job",
    description=(
        "Point counts, artifact availability and download URLs, plus the "
        "honest georeferencing status. Always returns a payload: if the job has "
        "not been reconstructed, status is 'not_available'."
    ),
)
async def get_job_reconstruction(job_id: str) -> Dict[str, Any]:
    """Summarise the reconstruction for ``job_id``."""
    return {"job_id": job_id, **build_reconstruction_payload(job_id)}


@router.post(
    "/{job_id}/run",
    summary="Run reconstruction for a job",
    description=(
        "Runs the real reconstruction if no artifacts exist yet. This is "
        "explicitly opt-in because it takes minutes. It NEVER re-runs YOLO, "
        "segmentation, depth estimation or disaster analysis."
    ),
)
async def run_job_reconstruction(job_id: str) -> Dict[str, Any]:
    """Reconstruct ``job_id`` only if nothing usable is already on disk."""
    return {"job_id": job_id, **ensure_reconstruction(job_id, run_missing=True)}


@router.get(
    "/{job_id}/artifact/{artifact_name}",
    summary="Download a reconstruction artifact",
    description=(
        "Streams a reconstruction artifact (GLB/PLY/NPZ/JSON) for a 3D viewer. "
        "Only allow-listed artifact names inside the job's own reconstruction "
        "directory are served."
    ),
    response_class=FileResponse,
)
async def get_reconstruction_artifact(job_id: str, artifact_name: str) -> FileResponse:
    """Serve one reconstruction artifact as binary bytes."""
    resolved = resolve_artifact(job_id, artifact_name)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Artifact '{artifact_name}' is not available for job {job_id}. "
                f"Servable artifacts: {sorted(ARTIFACT_MEDIA_TYPES)}"
            ),
        )
    path, media_type = resolved
    return FileResponse(
        path,
        media_type=media_type,
        filename=artifact_name,
        headers={"Cache-Control": "public, max-age=3600"},
    )
