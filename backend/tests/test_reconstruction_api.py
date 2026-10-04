"""Phase 8.6 - backend integration tests for the real reconstruction.

Exercises the FastAPI surface with REAL artifacts produced by the
``reconstruction`` package (not hand-written fixtures), and asserts that the
API never claims GPS coordinates or metric scale.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.reconstruction_service import (
    ARTIFACT_MEDIA_TYPES,
    PRIMARY_ARTIFACT,
    build_reconstruction_payload,
    discover_artifacts,
    resolve_artifact,
)

client = TestClient(app)

JOB = "11111111-2222-3333-4444-555555555555"
BASE = "/api/v1/reconstruction"


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Redirect the service at an isolated data directory."""
    root = tmp_path / "data"
    (root / JOB / "frames").mkdir(parents=True)
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True)
    monkeypatch.setattr(settings, "DATA_DIR", root)
    monkeypatch.setattr(settings, "UPLOADS_DIR", uploads)
    return root


def _build_reconstruction(root: Path, n_points: int = 250) -> Path:
    """Write REAL artifacts using the production writers."""
    from reconstruction.dense_sfm import write_binary_ply, write_glb_points

    rec = root / JOB / "reconstruction"
    rec.mkdir(parents=True, exist_ok=True)

    points = np.random.default_rng(0).normal(size=(n_points, 3))
    write_glb_points(points, rec / "dense_points_global.glb")
    write_binary_ply(points, rec / "dense_points_global.ply")
    np.savez_compressed(rec / "dense_points_global.npz", points=points)
    (rec / "sparse_points_global.ply").write_text("ply\n", encoding="utf-8")
    (rec / "cameras_global.json").write_text(
        json.dumps({"frame_00000.jpg": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                        "translation": [0, 0, 0]}}),
        encoding="utf-8",
    )
    (rec / "reconstruction_global.json").write_text(
        json.dumps({"n_points": 10805, "n_frames_registered": 57,
                    "limitations": ["UP_TO_SCALE"]}),
        encoding="utf-8",
    )
    (rec / "dense_reconstruction.json").write_text(
        json.dumps({"n_dense_points": n_points, "dense_point_count": n_points,
                    "limitations": ["relative depth"]}),
        encoding="utf-8",
    )
    return rec


# ---------------------------------------------------------------------------
# 1. Artifact discovery
# ---------------------------------------------------------------------------

def test_discover_artifacts_finds_real_files(data_dir):
    rec = _build_reconstruction(data_dir)
    found = discover_artifacts(JOB)

    assert found[PRIMARY_ARTIFACT]["available"] is True
    assert found[PRIMARY_ARTIFACT]["size_bytes"] == (
        rec / "dense_points_global.glb"
    ).stat().st_size
    assert found[PRIMARY_ARTIFACT]["url"].endswith(f"/{JOB}/artifact/{PRIMARY_ARTIFACT}")
    assert found[PRIMARY_ARTIFACT]["media_type"] == "model/gltf-binary"


def test_discover_artifacts_marks_absent_files_unavailable(data_dir):
    _build_reconstruction(data_dir)
    found = discover_artifacts(JOB)

    assert found["trajectory.json"]["available"] is False
    assert found["trajectory.json"]["url"] is None
    assert found["trajectory.json"]["size_bytes"] is None


def test_discover_artifacts_on_a_job_with_no_reconstruction(data_dir):
    found = discover_artifacts(JOB)
    assert all(entry["available"] is False for entry in found.values())


# ---------------------------------------------------------------------------
# 2. Missing artifacts
# ---------------------------------------------------------------------------

def test_resolve_artifact_rejects_unknown_and_missing(data_dir):
    assert resolve_artifact(JOB, "not_a_thing.txt") is None
    assert resolve_artifact(JOB, "trajectory.json") is None      # allow-listed, absent


def test_resolve_artifact_blocks_path_traversal(data_dir):
    _build_reconstruction(data_dir)
    for attempt in ("../results.json", "../../config.py", "..%2F..%2Fresults.json"):
        assert resolve_artifact(JOB, attempt) is None


def test_api_returns_404_for_a_missing_artifact(data_dir):
    _build_reconstruction(data_dir)
    assert client.get(f"{BASE}/{JOB}/artifact/dense_points_global.glb").status_code == 200
    assert client.get(f"{BASE}/{JOB}/artifact/trajectory.json").status_code == 404


def test_api_returns_a_payload_for_an_unknown_job(data_dir):
    response = client.get(f"{BASE}/99999999-9999-9999-9999-999999999999")
    assert response.status_code == 200
# ---------------------------------------------------------------------------
# 3 + 8. Reconstruction status and successful metadata
# ---------------------------------------------------------------------------

def test_module_status_endpoint_describes_real_stages():
    payload = client.get(f"{BASE}/").json()
    assert payload["status"] == "operational"
    assert payload["scale_status"] == "UP_TO_SCALE_NO_METRIC_UNITS"
    assert payload["coordinate_system"] == "local_up_to_scale"
    assert PRIMARY_ARTIFACT in payload["servable_artifacts"]
    assert all(stage["implemented"] for stage in payload["stages"])


def test_reconstruction_status_is_success_with_real_artifacts(data_dir):
    _build_reconstruction(data_dir, n_points=321)
    payload = build_reconstruction_payload(JOB)

    assert payload["status"] == "success"
    assert payload["kind"] == "denser_point_cloud"
    assert payload["sparse_point_count"] == 10805
    assert payload["dense_point_count"] == 321
    assert payload["n_frames_registered"] == 57
    assert payload["artifact"]["primary"] == PRIMARY_ARTIFACT
    assert payload["artifact"]["url"].endswith(PRIMARY_ARTIFACT)


def test_reconstruction_status_is_not_available_without_artifacts(data_dir):
    payload = build_reconstruction_payload(JOB)

    assert payload["status"] == "not_available"
    assert payload["kind"] is None
    assert payload["sparse_point_count"] is None
    assert payload["dense_point_count"] is None
    assert payload["artifact"]["url"] is None


def test_reconstruction_status_is_partial_when_only_sparse_exists(data_dir):
    rec = data_dir / JOB / "reconstruction"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "sparse_points_global.ply").write_text("ply\n", encoding="utf-8")
    (rec / "reconstruction_global.json").write_text(
        json.dumps({"n_points": 100}), encoding="utf-8"
    )
    payload = build_reconstruction_payload(JOB)

    assert payload["status"] == "partial"
    assert payload["sparse_point_count"] == 100
    assert payload["dense_point_count"] is None


def test_api_exposes_reconstruction_for_the_job(data_dir):
    _build_reconstruction(data_dir, n_points=128)
    response = client.get(f"{BASE}/{JOB}")
    assert response.status_code == 200
    payload = response.json()

    assert payload["job_id"] == JOB
    assert payload["status"] == "success"
    assert payload["dense_point_count"] == 128
    assert payload["scale_status"] == "UP_TO_SCALE_NO_METRIC_UNITS"
    assert payload["georeferenced"] is False


# ---------------------------------------------------------------------------
# 4. GLB / PLY serving
# ---------------------------------------------------------------------------

def test_api_serves_a_valid_glb(data_dir):
    _build_reconstruction(data_dir)
    response = client.get(f"{BASE}/{JOB}/artifact/dense_points_global.glb")

    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"

    raw = response.content
    magic, version, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67            # 'glTF'
    assert version == 2
    assert total == len(raw)

    json_len, json_type = struct.unpack("<II", raw[12:20])
    assert json_type == 0x4E4F534A
    gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))
    assert gltf["meshes"][0]["primitives"][0]["mode"] == 0   # POINTS
    assert gltf["accessors"][0]["count"] == 250


def test_api_serves_a_valid_binary_ply(data_dir):
    _build_reconstruction(data_dir)
    response = client.get(f"{BASE}/{JOB}/artifact/dense_points_global.ply")

    assert response.status_code == 200
    assert response.content.startswith(b"ply\n")
    assert b"format binary_little_endian 1.0" in response.content
    assert b"element vertex 250" in response.content


def test_api_serves_json_artifacts(data_dir):
    _build_reconstruction(data_dir)
    response = client.get(f"{BASE}/{JOB}/artifact/cameras_global.json")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "frame_00000.jpg" in response.json()


def test_every_allow_listed_artifact_name_has_a_media_type():
    assert set(ARTIFACT_MEDIA_TYPES) >= {PRIMARY_ARTIFACT}
    for name, media in ARTIFACT_MEDIA_TYPES.items():
        assert media and name.endswith((".glb", ".ply", ".npz", ".json"))
# ---------------------------------------------------------------------------
# 6. Georeferencing is honest
# ---------------------------------------------------------------------------

def test_georeferencing_unavailable_state_is_exposed(data_dir):
    _build_reconstruction(data_dir)
    payload = build_reconstruction_payload(JOB)
    geo = payload["georeferencing"]

    assert geo["status"] == "unavailable"
    assert payload["georeferenced"] is False
    assert geo["crs"] is None
    assert geo["n_control_points"] == 0
    assert geo["reason"], "an unavailable state must explain itself"
    assert geo["metadata_probe"]["n_frames_with_gps_exif"] == 0
    assert geo["metadata_probe"]["first_frame_gps_fix"] is None
    assert geo["requirements"], "must state what would be needed"


def test_api_never_claims_gps_or_metric_scale(data_dir):
    _build_reconstruction(data_dir)
    payload = client.get(f"{BASE}/{JOB}").json()

    assert payload["georeferenced"] is False
    assert payload["scale_status"] == "UP_TO_SCALE_NO_METRIC_UNITS"
    assert payload["coordinate_system"] == "local_up_to_scale"
    for forbidden in ("gps_coordinates", "latitude_longitude", "altitude",
                      "metric_distance", "metric_area", "terrain_area"):
        assert forbidden in payload["not_measured"]


def test_georeferencing_metadata_probe_lists_what_was_searched(data_dir):
    _build_reconstruction(data_dir)
    probe = client.get(f"{BASE}/{JOB}").json()["georeferencing"]["metadata_probe"]

    assert "frame_exif_gps" in probe["sources_checked"]
    assert "sidecar_telemetry" in probe["sources_checked"]
    assert "video_container_metadata" in probe["sources_checked"]


# ---------------------------------------------------------------------------
# 5 + 7. Results contract, additive and preserving
# ---------------------------------------------------------------------------

def test_results_contract_gains_a_reconstruction_section(data_dir):
    from ai.results_builder import build_results

    _build_reconstruction(data_dir, n_points=99)
    results = build_results(JOB, data_dir)
    reconstruction = results["reconstruction"]

    assert reconstruction["status"] == "success"
    assert reconstruction["sparse_point_count"] == 10805
    assert reconstruction["dense_point_count"] == 99
    assert reconstruction["n_frames_registered"] == 57
    assert reconstruction["georeferenced"] is False
    assert reconstruction["georeferencing_status"] == "unavailable"
    assert reconstruction["artifact"]["glb_available"] is True
    assert reconstruction["artifact"]["primary"] == "dense_points_global.glb"
    assert reconstruction["artifact"]["endpoint"].endswith("/artifact")
    assert reconstruction["scale_status"] == "UP_TO_SCALE_NO_METRIC_UNITS"


def test_results_contract_preserves_every_existing_key(data_dir):
    from ai.results_builder import build_results

    _build_reconstruction(data_dir)
    results = build_results(JOB, data_dir)

    for key in ("job_id", "models", "categories", "relative_depth", "disasters",
                "disaster_modules_available", "disaster_modules_not_implemented",
                "not_measured", "model_url"):
        assert key in results, f"existing results key {key} was removed"
    assert set(results["categories"]) >= {
        "people", "vehicles", "structures", "roads",
        "vegetation", "terrain", "possible_water_extent",
    }
    assert results["model_url"] is None


def test_results_contract_reports_not_available_without_reconstruction(data_dir):
    from ai.results_builder import build_results

    results = build_results(JOB, data_dir)

    assert results["reconstruction"]["status"] == "not_available"
    assert results["reconstruction"]["sparse_point_count"] is None
    assert results["reconstruction"]["dense_point_count"] is None
    assert results["reconstruction"]["georeferenced"] is False


def test_results_json_never_contains_coordinate_keys(data_dir):
    from ai.results_builder import build_results

    _build_reconstruction(data_dir)
    results = build_results(JOB, data_dir)
    reconstruction = results["reconstruction"]

    for forbidden in ("lat", "lon", "latitude", "longitude", "latitudes"):
        assert forbidden not in reconstruction
    assert "georeferenced" in reconstruction
    assert reconstruction["georeferenced"] is False


# ---------------------------------------------------------------------------
# Pipeline reuse semantics
# ---------------------------------------------------------------------------

def test_ensure_reconstruction_reuses_existing_artifacts_without_recomputing(data_dir):
    from app.services.reconstruction_service import ensure_reconstruction

    _build_reconstruction(data_dir, n_points=77)
    before = (data_dir / JOB / "reconstruction" / "dense_points_global.glb").stat().st_mtime_ns

    payload = ensure_reconstruction(JOB, run_missing=False)
    after = (data_dir / JOB / "reconstruction" / "dense_points_global.glb").stat().st_mtime_ns

    assert payload["status"] == "success"
    assert payload["dense_point_count"] == 77
    assert before == after, "existing artifacts must not be regenerated"


def test_ensure_reconstruction_does_not_run_when_not_requested(data_dir):
    from app.services.reconstruction_service import ensure_reconstruction

    payload = ensure_reconstruction(JOB, run_missing=False)

    assert payload["status"] == "not_available"
    assert not (data_dir / JOB / "reconstruction" / "dense_points_global.glb").exists()