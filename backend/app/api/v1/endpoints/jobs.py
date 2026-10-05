import asyncio
import json
import sys
import uuid
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.core.config import settings
from app.services.frame_extractor import extract_frames
from app.services.reconstruction_service import ensure_reconstruction

# Ensure project root is in sys.path so the ai package can be imported
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

from ai.yolo_detector import detect_objects
from ai.scene_segmenter import segment_frames
from ai.depth_estimator import estimate_depth_for_frames
from ai.results_builder import build_results
from ai.disaster import ArtifactBundle
from ai.disaster import available as registered_hazards
from ai.disaster import get as get_hazard_analyzer

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


def _disaster_summary_path(job_id: str) -> Path:
    return settings.DATA_DIR / job_id / "analysis" / "disaster" / "disasters_summary.json"


def _analyze_hazards(job_id: str) -> Dict[str, Any]:
    """Run every registered disaster analyzer over the Phase 6A artifacts.

    Hazard-agnostic: analyzers are discovered from the ``ai.disaster`` registry,
    so no hazard-specific logic lives in this module. The analyzers only read
    artifacts already produced by the detection / segmentation / depth stages -
    no model inference happens here.
    """
    bundle = ArtifactBundle(job_id, str(settings.DATA_DIR))

    disasters: Dict[str, Any] = {}
    for hazard_name in registered_hazards():
        analyzer = get_hazard_analyzer(hazard_name)()
        disasters[hazard_name] = analyzer.analyze(bundle)

    return {"disasters": disasters}


# Phase 6A: `_build_mock_results()` was removed. Results are now aggregated from
# real model artifacts (YOLO detections, SegFormer coverage, relative depth) by
# `ai.results_builder.build_results()`.


async def _run_pipeline(job_id: str) -> None:
    job = JOBS[job_id]

    try:
        video_path = _find_video(job_id)

        if video_path is None:
            raise FileNotFoundError("Uploaded video not found.")

        analysis_dir = settings.DATA_DIR / job_id / "analysis"

        # Step 1: Extract frames
        job["status"] = "extracting_frames"
        job["progress"] = 10

        frames_dir = settings.DATA_DIR / job_id / "frames"

        result = await asyncio.to_thread(
            extract_frames,
            str(video_path),
            str(frames_dir),
            fps=2.0
        )

        job["progress"] = 25
        job["frames_extracted"] = result["saved_frames"]

        # Step 2: YOLO11n object detection
        job["status"] = "detecting_objects"
        job["progress"] = 35

        detections_file = analysis_dir / "detections.json"

        detections = await asyncio.to_thread(
            detect_objects,
            frames_dir=frames_dir,
            output_file=detections_file,
            conf_threshold=0.25,
        )

        job["detections_count"] = len(detections)
        job["progress"] = 55

        # Step 3: SegFormer-B0 ADE20K semantic segmentation
        job["status"] = "segmenting_scene"
        job["progress"] = 60

        segmentation_dir = analysis_dir / "segmentation"

        segmentation = await asyncio.to_thread(
            segment_frames,
            frames_dir=frames_dir,
            output_dir=segmentation_dir,
        )

        job["segmentation_frames_count"] = segmentation["frames_processed"]
        job["progress"] = 72

        # Step 4: Depth Anything V2 (relative depth only)
        job["status"] = "estimating_depth"
        job["progress"] = 78

        depth_dir = analysis_dir / "depth"

        depth_maps = await asyncio.to_thread(
            estimate_depth_for_frames,
            frames_dir=frames_dir,
            output_dir=depth_dir,
        )

        job["depth_maps_count"] = len(depth_maps)
        job["progress"] = 88

        # Step 5: Disaster intelligence (artifacts only - no model inference)
        job["status"] = "analyzing_hazards"
        job["progress"] = 91

        disaster_payload = await asyncio.to_thread(_analyze_hazards, job_id)

        disaster_path = _disaster_summary_path(job_id)
        disaster_path.parent.mkdir(parents=True, exist_ok=True)
        disaster_path.write_text(
            json.dumps(disaster_payload, indent=2),
            encoding="utf-8",
        )

        job["hazards_analyzed"] = sorted(disaster_payload["disasters"])

        # Step 6: 3D reconstruction (REAL, Phases 8.1-8.4).
        # The processing pipeline IS the explicit decision to reconstruct: it
        # runs reconstruction for THIS job when its artifacts are missing, and
        # reuses existing artifacts when they are already on disk (never
        # regenerates). It does NOT re-run YOLO, segmentation, depth estimation
        # or flood analysis. Status GET endpoints stay cheap and read-only.
        job["status"] = "reconstructing"
        job["progress"] = 94

        reconstruction = await asyncio.to_thread(
            ensure_reconstruction,
            job_id,
            settings.RECONSTRUCTION_AUTORUN,
        )
        job["reconstruction_status"] = reconstruction.get("status")

        # Step 7: Georeferencing (Phase 8.5, honest by construction).
        # Reports unavailable unless real GPS/IMU metadata exists for the job.
        job["status"] = "georeferencing"
        job["progress"] = 97

        georeferencing = reconstruction.get("georeferencing", {})
        job["georeferenced"] = bool(reconstruction.get("georeferenced", False))
        job["georeferencing_status"] = georeferencing.get("status")

        # Step 8: Aggregate real results from the generated artifacts.
        results = await asyncio.to_thread(build_results, job_id)

        out_path = _results_path(job_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        job["progress"] = 100
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