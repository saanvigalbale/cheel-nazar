"""Phase 8.6 - reconstruction service for the FastAPI backend.

Single source of truth for *what reconstruction artifacts exist for a job* and
*what contract the API exposes*. The heavy reconstruction itself already lives
in the ``reconstruction`` package (Phases 8.1-8.5); this module only discovers,
summarises, serves and (optionally) triggers it.

Design rules:
  * NEVER re-run YOLO / segmentation / depth / disaster analysis.
  * Reuse existing artifacts by default; a full reconstruction takes minutes,
    so it must be an explicit decision (``run_missing``), not a side effect of
    reading a status endpoint.
  * Report ``georeferenced: false`` and ``UP_TO_SCALE_NO_METRIC_UNITS`` unless
    real GPS data exists. Never surface a coordinate that was not measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings

# Artifact name -> media type. This allow-list is also the path-traversal guard:
# anything not listed here can never be served.
ARTIFACT_MEDIA_TYPES: Dict[str, str] = {
    "dense_points_global.glb": "model/gltf-binary",
    "dense_points_global.ply": "application/octet-stream",
    "dense_points_global.npz": "application/octet-stream",
    "sparse_points_global.ply": "application/octet-stream",
    "sparse_points_global.npz": "application/octet-stream",
    "cameras_global.json": "application/json",
    "trajectory.json": "application/json",
    "reconstruction_global.json": "application/json",
    "dense_reconstruction.json": "application/json",
}

#: The artifact a 3D viewer should load first.
PRIMARY_ARTIFACT = "dense_points_global.glb"

#: Route prefix the API mounts these artifacts under (used to build URLs).
ARTIFACT_URL_PREFIX = "/api/v1/reconstruction"

SCALE_STATUS_UP_TO_SCALE = "UP_TO_SCALE_NO_METRIC_UNITS"
COORDINATE_SYSTEM_LOCAL = "local_up_to_scale"


def _reconstruction_dir(job_id: str) -> Path:
    return settings.DATA_DIR / job_id / "reconstruction"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def artifact_url(job_id: str, name: str) -> str:
    """Browser-reachable URL for a served artifact (not a filesystem path)."""
    return f"{ARTIFACT_URL_PREFIX}/{job_id}/artifact/{name}"


def discover_artifacts(job_id: str) -> Dict[str, Dict[str, Any]]:
    """Report which reconstruction artifacts exist for ``job_id``."""
    rec_dir = _reconstruction_dir(job_id)
    found: Dict[str, Dict[str, Any]] = {}
    for name in ARTIFACT_MEDIA_TYPES:
        path = rec_dir / name
        exists = path.is_file()
        found[name] = {
            "available": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "media_type": ARTIFACT_MEDIA_TYPES[name],
            "url": artifact_url(job_id, name) if exists else None,
        }
    return found


def resolve_artifact(job_id: str, name: str) -> Optional[Tuple[Path, str]]:
    """Resolve a requested artifact to ``(path, media_type)``.

    Returns ``None`` for unknown names or missing files. Only allow-listed
    filenames inside the job's own reconstruction directory are reachable, so a
    crafted request cannot escape the job directory.
    """
    media_type = ARTIFACT_MEDIA_TYPES.get(name)
    if media_type is None:
        return None
    rec_dir = _reconstruction_dir(job_id).resolve()
    path = (rec_dir / name).resolve()
    if rec_dir not in path.parents or not path.is_file():
        return None
    return path, media_type


def _job_video_path(job_id: str) -> Optional[Path]:
    try:
        matches = sorted(settings.UPLOADS_DIR.glob(f"{job_id}_*"))
    except OSError:
        return None
    return matches[0] if matches else None


def build_reconstruction_payload(job_id: str) -> Dict[str, Any]:
    """Build the reconstruction section for ``job_id`` from real artifacts.

    Always returns a payload: when nothing has been reconstructed it reports
    ``status: "not_available"`` rather than 404-ing, so the UI can explain why.
    """
    import sys

    if str(settings.ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(settings.ROOT_DIR))

    from reconstruction.georeferencing import georeference_job

    rec_dir = _reconstruction_dir(job_id)
    artifacts = discover_artifacts(job_id)

    sparse = _load_json(rec_dir / "reconstruction_global.json")
    dense = _load_json(rec_dir / "dense_reconstruction.json")

    sparse_count = (sparse or {}).get("n_points")
    if sparse_count is None:
        sparse_count = (sparse or {}).get("sparse_point_count")
    dense_count = (dense or {}).get("n_dense_points")
    if dense_count is None:
        dense_count = (dense or {}).get("dense_point_count")

    has_any = any(entry["available"] for entry in artifacts.values())
    if not has_any:
        status = "not_available"
    elif dense_count is not None or artifacts[PRIMARY_ARTIFACT]["available"]:
        status = "success"
    else:
        status = "partial"

    georef = georeference_job(job_id, settings.DATA_DIR, _job_video_path(job_id))

    limitations = list(georef.limitations)
    for source in (dense, sparse):
        for item in (source or {}).get("limitations", []) or []:
            if item not in limitations:
                limitations.append(item)

    served = {
        name: entry["url"] for name, entry in artifacts.items() if entry["available"]
    }

    return {
        "status": status,
        "kind": "denser_point_cloud" if status == "success" else None,
        "phase": "8.1-8.5",
        "sparse_point_count": sparse_count,
        "dense_point_count": dense_count,
        "n_frames_registered": (sparse or {}).get("n_frames_registered"),
        "coordinate_system": COORDINATE_SYSTEM_LOCAL,
        "scale_status": SCALE_STATUS_UP_TO_SCALE,
        "georeferenced": bool(georef.georeferenced),
        "georeferencing": {
            "status": georef.status,
            "crs": georef.crs,
            "source": georef.source,
            "n_control_points": georef.n_control_points,
            "reason": georef.reason,
            "requirements": georef.requirements,
            "metadata_probe": georef.probe,
        },
        "artifacts": artifacts,
        "artifact": {
            "primary": PRIMARY_ARTIFACT,
            "url": served.get(PRIMARY_ARTIFACT),
            "media_type": ARTIFACT_MEDIA_TYPES[PRIMARY_ARTIFACT],
        },
        "served": served,
        "limitations": limitations,
        "not_measured": [
            "gps_coordinates",
            "latitude_longitude",
            "altitude",
            "metric_distance",
            "metric_area",
            "metric_building_dimensions",
            "terrain_area",
        ],
    }


def ensure_reconstruction(job_id: str, run_missing: bool = False) -> Dict[str, Any]:
    """Reuse an existing reconstruction, and only compute one if asked.

    ``run_missing=False`` (the default, and what the job pipeline uses) never
    starts expensive work: it reports what is already on disk.
    """
    payload = build_reconstruction_payload(job_id)

    if payload["status"] != "not_available" or not run_missing:
        return payload

    import sys

    if str(settings.ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(settings.ROOT_DIR))

    from reconstruction.dense_sfm import reconstruct_dense
    from reconstruction.global_sfm import reconstruct_global

    frames_dir = settings.DATA_DIR / job_id / "frames"
    if not frames_dir.is_dir():
        payload["run_note"] = "No frames available; reconstruction was not attempted."
        return payload

    if not (_reconstruction_dir(job_id) / "cameras_global.json").is_file():
        global_result = reconstruct_global(job_id, settings.DATA_DIR)
        if not global_result.ok:
            payload["status"] = "global_reconstruction_failed"
            payload["run_note"] = global_result.message
            return payload

    dense_result = reconstruct_dense(job_id, settings.DATA_DIR)
    payload = build_reconstruction_payload(job_id)
    if not dense_result.ok:
        payload["status"] = "dense_reconstruction_failed"
        payload["run_note"] = dense_result.message
    return payload