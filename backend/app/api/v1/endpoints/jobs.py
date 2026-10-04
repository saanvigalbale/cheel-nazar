import asyncio
import json
import sys
import uuid
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.core.config import settings
from app.services.frame_extractor import extract_frames

# Ensure project root is in sys.path so the ai package can be imported
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

from ai.yolo_detector import detect_objects
from ai.depth_estimator import estimate_depth_for_frames

router = APIRouter()

JOBS: Dict[str, Dict[str, Any]] = {}


def _find_video(job_id: str) -> Optional[Path]:
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id.")

    matches = list(settings.UPLOADS_DIR.glob(f"{job_id}_*"))
    return matches[0] if matches else None


def _results_path(job_id: str) -> Path:
    return settings.DATA_DIR / job_id / "results.json"


def _build_mock_results(job_id: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "model_url": None,
        "georeference": {
            "lat": 28.6139,
            "lon": 77.2090,
            "altitude_m": 120.0
        },
        "detections": [
            {
                "label": "building",
                "count": 14,
                "avg_confidence": 0.91
            },
            {
                "label": "vehicle",
                "count": 6,
                "avg_confidence": 0.84
            },
            {
                "label": "person",
                "count": 3,
                "avg_confidence": 0.72
            },
            {
                "label": "obstacle",
                "count": 5,
                "avg_confidence": 0.78
            }
        ],
        "measurements": {
            "area_sq_m": 5400.0,
            "max_building_height_m": 18.5,
            "scene_width_m": 92.0,
            "scene_length_m": 58.7
        },
        "damage_analysis": {
            "damaged_structures": 3,
            "severity": "moderate",
            "affected_area_pct": 12.4
        },
        "uncertainty": {
            "overall_confidence": 0.82,
            "low_confidence_regions": 4
        }
    }


async def _run_pipeline(job_id: str) -> None:
    job = JOBS[job_id]

    try:
        video_path = _find_video(job_id)

        if video_path is None:
            raise FileNotFoundError("Uploaded video not found.")

        # Step 1: Extract frames (Phase 4)
        job["status"] = "extracting_frames"
        job["progress"] = 10

        output_dir = settings.DATA_DIR / job_id / "frames"

        result = await asyncio.to_thread(
            extract_frames,
            str(video_path),
            str(output_dir),
            fps=2.0
        )

        job["progress"] = 25
        job["frames_extracted"] = result["saved_frames"]

        # Step 2: AI YOLO Object Detection (Phase 6.2)
        job["status"] = "detecting_objects"
        job["progress"] = 35

        analysis_dir = settings.DATA_DIR / job_id / "analysis"
        detections_file = analysis_dir / "detections.json"

        detections = await asyncio.to_thread(
            detect_objects,
            frames_dir=output_dir,
            output_file=detections_file,
            conf_threshold=0.25,
        )

        job["detections_count"] = len(detections)
        job["progress"] = 55

        # Step 3: AI Monocular Depth Estimation (Phase 6.4)
        job["status"] = "estimating_depth"
        job["progress"] = 65

        depth_dir = analysis_dir / "depth"

        depth_maps = await asyncio.to_thread(
            estimate_depth_for_frames,
            frames_dir=output_dir,
            output_dir=depth_dir,
        )

        job["depth_maps_count"] = len(depth_maps)
        job["progress"] = 80

        # Step 4: 3D Reconstruction (Phase 5 placeholder)
        job["status"] = "reconstructing"
        await asyncio.sleep(2)
        job["progress"] = 90

        # Step 5: Georeferencing (placeholder)
        job["status"] = "georeferencing"
        await asyncio.sleep(1)
        job["progress"] = 100

        out_path = _results_path(job_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(
            json.dumps(_build_mock_results(job_id), indent=2)
        )

        job["status"] = "done"

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)


@router.post(
    "/{job_id}/process",
    status_code=status.HTTP_202_ACCEPTED
)
async def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks
):
    if _find_video(job_id) is None:
        raise HTTPException(
            status_code=404,
            detail="No uploaded video for this job_id."
        )

    existing = JOBS.get(job_id)

    if existing and existing["status"] not in ("done", "failed"):
        raise HTTPException(
            status_code=409,
            detail="Job is already processing."
        )

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "error": None
    }

    background_tasks.add_task(
        _run_pipeline,
        job_id
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Processing started"
    }


@router.get("/{job_id}/status")
async def get_status(job_id: str):
    job = JOBS.get(job_id)

    if job:
        return {
            "job_id": job_id,
            **job
        }

    if _results_path(job_id).exists():
        return {
            "job_id": job_id,
            "status": "done",
            "progress": 100,
            "error": None
        }

    if _find_video(job_id) is not None:
        return {
            "job_id": job_id,
            "status": "uploaded",
            "progress": 0,
            "error": None
        }

    raise HTTPException(
        status_code=404,
        detail="Unknown job_id."
    )


@router.get("/{job_id}/results")
async def get_results(job_id: str):
    path = _results_path(job_id)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Results not ready yet."
        )

    return json.loads(path.read_text())


@router.get("/{job_id}/detections")
async def get_detections(job_id: str):
    path = settings.DATA_DIR / job_id / "analysis" / "detections.json"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Detections not ready yet."
        )

    return json.loads(path.read_text())