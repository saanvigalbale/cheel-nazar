"""Phase 8.2 tests for global (multi-frame) sparse SfM.

Synthetic scenes only - no models, no network, no real job data. These lock the
incremental registration, pose composition, degeneracy detection and artifact
contract, including honest failure reporting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruction.global_sfm import (  # noqa: E402
    GLOBAL_FAILURE_MESSAGES,
    STATUS_GLOBAL_NO_REGISTRATION,
    STATUS_GLOBAL_NO_SEED,
    DEFAULT_GLOBAL_PARAMETERS,
    GlobalReconstructionResult,
    compose_pose,
    planarity_report,
    reconstruct_global,
    register_camera,
    select_frame_indices,
)
from reconstruction.sfm import (  # noqa: E402
    STATUS_INSUFFICIENT_FRAMES,
    STATUS_NO_FRAMES,
    camera_matrix,
)

PATCH = 24


def _write_frames(frames_dir: Path, images) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)


def _render_view(points3d: np.ndarray, K: np.ndarray, w: int, h: int, seed: int) -> np.ndarray:
    """Render unique textured patches at the projections of ``points3d``."""
    rng = np.random.default_rng(seed)
    proj = points3d @ K.T                       # (N, 3)
    uv = proj[:, :2] / proj[:, 2:3]
    image = np.full((h, w), 20, np.uint8)
    half = PATCH // 2
    for (x, y) in uv:
        xi, yi = int(round(x)), int(round(y))
        if not (half < xi < w - half and half < yi < h - half):
            continue
        texture = rng.integers(90, 256, (PATCH, PATCH), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.7)
        image[yi - half : yi + half, xi - half : xi + half] = texture
    return cv2.GaussianBlur(image, (3, 3), 0.5)


def _synthetic_scene(seed: int = 0, w: int = 640, h: int = 480, n: int = 160):
    """Three views of a clearly NON-planar 3D point cloud (avoids degeneracy)."""
    rng = np.random.default_rng(seed)
    K = camera_matrix(w, h, 1.2)
    X = np.column_stack(
        [
            rng.uniform(-1.2, 1.2, n),
            rng.uniform(-0.9, 0.9, n),
            rng.uniform(7.0, 11.0, n),          # volumetric + fully visible
        ]
    )
    views = [_render_view(X, K, w, h, seed + 1)]
    views.append(_render_view(X - np.array([1.0, 0.0, 0.0]), K, w, h, seed + 1))
    views.append(_render_view(X - np.array([2.0, 0.0, 0.0]), K, w, h, seed + 1))
    return views, X


# ---------------------------------------------------------------------------
# Pose composition
# ---------------------------------------------------------------------------

def test_compose_pose_matches_direct_composition():
    """world->camera poses must compose so X_child = R_rel X_parent + t_rel."""
    rng = np.random.default_rng(0)

    def rand_pose():
        R, _ = cv2.Rodrigues(rng.normal(scale=0.2, size=3))
        return {"rotation": R, "translation": rng.normal(scale=0.5, size=(3, 1))}

    parent, child = rand_pose(), rand_pose()
    r_rel, t_rel = compose_pose(parent, child)

    world = rng.normal(scale=2.0, size=(20, 3))
    in_parent = world @ parent["rotation"].T + parent["translation"].T
    in_child = world @ child["rotation"].T + child["translation"].T
    via_compose = in_parent @ r_rel.T + t_rel.T

    assert np.allclose(via_compose, in_child, atol=1e-9)


def test_compose_pose_of_identity_is_identity():
    identity = {"rotation": np.eye(3), "translation": np.zeros((3, 1))}
    rotation, translation = compose_pose(identity, identity)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, np.zeros((3, 1)))


# ---------------------------------------------------------------------------
# Camera registration
# ---------------------------------------------------------------------------

def test_register_camera_recovers_known_pose():
    rng = np.random.default_rng(3)
    w, h = 640, 480
    K = camera_matrix(w, h, 1.2)
    rvec = np.array([0.02, -0.05, 0.01], dtype=np.float64)
    R_true, _ = cv2.Rodrigues(rvec)
    t_true = np.array([[0.3], [-0.1], [0.2]], dtype=np.float64)
    # Well-conditioned 3D cloud: absolute pose is unambiguous.
    X = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, 120),
            rng.uniform(-1.5, 1.5, 120),
            rng.uniform(7.0, 11.0, 120),
        ]
    )
    proj = X @ R_true.T + t_true.T              # camera-frame (N, 3)
    pixel = proj @ K.T                           # [fx*x + cx*z, fy*y + cy*z, z]
    uv = pixel[:, :2] / pixel[:, 2:3]             # (N, 2) pixel coords

    pose = register_camera(
        np.ascontiguousarray(X), np.ascontiguousarray(uv), K, DEFAULT_GLOBAL_PARAMETERS
    )
    assert pose is not None, "known pose must be recovered"
    assert np.allclose(pose["rotation"], R_true, atol=1e-4)
    assert np.allclose(pose["translation"], t_true, atol=1e-4)
    assert np.allclose(pose["center"], (-R_true.T @ t_true).ravel(), atol=1e-4)
    assert pose["median_reproj_error_px"] < 1e-2


def test_register_camera_rejects_too_few_correspondences():
    K = camera_matrix(64, 48, 1.2)
    X = np.zeros((5, 3))
    uv = np.zeros((5, 2))
    params = {**DEFAULT_GLOBAL_PARAMETERS, "min_pnp_points": 30}
    assert register_camera(X, uv, K, params) is None


def test_register_camera_rejects_outliers():
    """A pose fitted to mostly wrong 2D points must not pass validation."""
    rng = np.random.default_rng(7)
    K = camera_matrix(640, 480, 1.2)
    X = np.column_stack(
        [rng.uniform(-1, 1, 200), rng.uniform(-0.7, 0.7, 200), rng.uniform(5, 12, 200)]
    )
    R_true, _ = cv2.Rodrigues(np.array([0.0, 0.03, 0.0]))
    proj = X @ R_true.T                          # camera-frame (N, 3)
    pixel = proj @ K.T                           # [fx*x + cx*z, fy*y + cy*z, z]
    uv = pixel[:, :2] / pixel[:, 2:3]             # (N, 2) pixel coords
    # scramble 90 % of the correspondences
    uv = uv.copy()
    uv[:180] = rng.uniform(0, 640, size=(180, 2))
    pose = register_camera(
        np.ascontiguousarray(X), np.ascontiguousarray(uv), K, DEFAULT_GLOBAL_PARAMETERS
    )
    assert pose is None or pose["median_reproj_error_px"] <= 3.0


# ---------------------------------------------------------------------------
# Degeneracy detection + frame selection
# ---------------------------------------------------------------------------

def test_planarity_report_flags_a_plane():
    rng = np.random.default_rng(1)
    plane = np.column_stack(
        [rng.uniform(-5, 5, 300), rng.uniform(-5, 5, 300), np.zeros(300)]
    )
    report = planarity_report(plane)
    assert report["planar"] is True
    assert report["third_axis_ratio"] < 0.01


def test_planarity_report_accepts_a_3d_cloud():
    rng = np.random.default_rng(2)
    cloud = rng.uniform(-5, 5, size=(300, 3))
    report = planarity_report(cloud)
    assert report["planar"] is False
    assert report["third_axis_ratio"] > 0.2


def test_planarity_report_handles_degenerate_input():
    assert planarity_report(np.zeros((0, 3)))["planar"] is None
    assert planarity_report(np.zeros((2, 3)))["planar"] is None


def test_select_frame_indices():
    assert select_frame_indices(57, 1, 57) == list(range(57))
    assert select_frame_indices(57, 2, 57) == list(range(0, 57, 2))
    limited = select_frame_indices(57, 1, 5)
    assert len(limited) == 5
    assert limited == sorted(limited)
    assert limited[-1] <= 56


# ---------------------------------------------------------------------------
# Honest failure paths
# ---------------------------------------------------------------------------

def test_missing_frames_reports_no_frames(tmp_path):
    result = reconstruct_global("nope", tmp_path / "data", write_artifacts=False)
    assert result.status == STATUS_NO_FRAMES
    assert result.ok is False
    assert result.n_points == 0


def test_empty_frames_directory_reports_no_frames(tmp_path):
    data = tmp_path / "data"
    (data / "empty" / "frames").mkdir(parents=True)
    result = reconstruct_global("empty", data, write_artifacts=False)
    assert result.status == STATUS_NO_FRAMES


def test_single_frame_reports_insufficient_frames(tmp_path):
    data = tmp_path / "data"
    _write_frames(data / "one" / "frames", [np.full((120, 160), 30, np.uint8)])
    result = reconstruct_global("one", data, write_artifacts=False)
    assert result.status == STATUS_INSUFFICIENT_FRAMES
    assert result.n_points == 0


def test_unmatched_frames_report_no_seed(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(5)
    frames = [
        rng.integers(0, 256, (240, 320), dtype=np.uint8),
        rng.integers(0, 256, (240, 320), dtype=np.uint8),
    ]
    _write_frames(data / "noise" / "frames", frames)
    result = reconstruct_global("noise", data, write_artifacts=False)
    assert result.status == STATUS_GLOBAL_NO_SEED
    assert result.ok is False
    assert result.n_points == 0
    assert result.stats["seed_attempts"], "seed attempts must be recorded"


def test_failed_run_writes_manifest_but_no_global_cloud(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(11)
    frames = [
        rng.integers(0, 256, (240, 320), dtype=np.uint8),
        rng.integers(0, 256, (240, 320), dtype=np.uint8),
    ]
    _write_frames(data / "noise2" / "frames", frames)
    reconstruct_global("noise2", data, write_artifacts=True)

    out = data / "noise2" / "reconstruction"
    assert (out / "reconstruction_global.json").is_file()
    assert not (out / "sparse_points_global.ply").exists()

# ---------------------------------------------------------------------------
# Global alignment on a synthetic multi-view scene
# ---------------------------------------------------------------------------

def test_global_alignment_registers_multiple_cameras(tmp_path):
    data = tmp_path / "data"
    views, _ = _synthetic_scene(seed=0)
    _write_frames(data / "synth3" / "frames", views)

    result = reconstruct_global(
        "synth3",
        data,
        parameters={"registration_stride": 1, "seed_stride": 1},
        write_artifacts=True,
    )

    assert result.status == "success", result.message
    assert result.ok is True
    assert result.n_points > 20
    assert len(result.frames_registered) == 3, result.frames_registered
    assert set(result.cameras) == set(result.frames_registered)
    for name, cam in result.cameras.items():
        assert len(cam["rotation"]) == 3 and len(cam["rotation"][0]) == 3
        assert len(cam["center"]) == 3
        assert cam["method"] in (
            "seed_identity",
            "seed_recoverPose",
            "solvePnPRansac(EPNP)",
        )

    out = data / "synth3" / "reconstruction"
    manifest = json.loads((out / "reconstruction_global.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["scale_status"] == "UP_TO_SCALE_NO_METRIC_UNITS"
    assert "HEURISTIC_NOT_CALIBRATED" in manifest["intrinsics_status"]
    assert manifest["n_frames_registered"] == 3

    trajectory = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["georeferenced"] is False
    assert len(trajectory["frames"]) == 3

    cameras = json.loads((out / "cameras_global.json").read_text(encoding="utf-8"))
    assert len(cameras) == 3

    ply_lines = (out / "sparse_points_global.ply").read_text(encoding="utf-8").splitlines()
    vertex_line = next(line for line in ply_lines if line.startswith("element vertex"))
    assert int(vertex_line.split()[-1]) == result.n_points

    with np.load(out / "sparse_points_global.npz") as npz:
        assert npz["points"].shape == (result.n_points, 3)
        assert np.all(np.isfinite(npz["points"]))


def test_trajectory_is_in_one_world_frame(tmp_path):
    data = tmp_path / "data"
    views, _ = _synthetic_scene(seed=4)
    _write_frames(data / "traj" / "frames", views)

    result = reconstruct_global(
        "traj", data, parameters={"registration_stride": 1}, write_artifacts=False
    )
    if result.ok:
        # The seed camera must sit exactly at the world origin.
        first = result.frames_registered[0]
        assert np.allclose(result.cameras[first]["center"], np.zeros(3), atol=1e-6)
        centers = np.array([result.cameras[n]["center"] for n in result.frames_registered])
        assert np.all(np.isfinite(centers))
        # Distinct viewpoints must not all collapse onto the same point.
        assert np.unique(np.round(centers, 4), axis=0).shape[0] >= 2


