"""Phase 8.1 tests for the OpenCV-based SfM foundation.

These use tiny synthetic scenes written to pytest's ``tmp_path`` - no models,
no network, no large downloads. They prove the reconstruction maths actually
works on real image geometry (two-view synthetic scene) and that failure modes
report honest statuses instead of fabricating geometry.
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

from reconstruction.sfm import (  # noqa: E402
    FAILURE_STATUSES,
    STATUS_INSUFFICIENT_FRAMES,
    STATUS_NO_FRAMES,
    STATUS_SUCCESS,
    camera_matrix,
    reconstruct_job,
    reconstruct_pair,
)


def _frame_dict(name: str, image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    return {"name": name, "gray": image, "scale": 1.0, "width": w, "height": h}


def _write_frames(frames_dir: Path, images) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        cv2.imwrite(str(frames_dir / f"frame_{index:05d}.png"), image)


def _synthetic_two_view(seed: int = 0, w: int = 640, h: int = 480, n: int = 140):
    """Render two views of a synthetic 3D scene with a known baseline.

    Each 3D point is a UNIQUE textured patch pasted at its projected location
    in both views. Unique texture is required: SIFT is scale/contrast
    invariant, so identically-shaped blobs produce ambiguous descriptors and
    ratio/mutual filtering correctly discards them.
    """
    rng = np.random.default_rng(seed)
    K = camera_matrix(w, h, 1.2)
    baseline = 1.2

    X = np.zeros((n, 3))
    X[:, 0] = rng.uniform(-0.9, 0.9, n)
    X[:, 1] = rng.uniform(-0.6, 0.6, n)
    X[:, 2] = rng.uniform(5.0, 9.0, n)

    def project(points3d: np.ndarray) -> np.ndarray:
        proj = (K @ points3d.T).T
        return proj[:, :2] / proj[:, 2:3]

    p1 = project(X)
    p2 = project(X - np.array([baseline, 0.0, 0.0]))

    patch = 24
    half = patch // 2
    image_a = np.full((h, w), 20, np.uint8)   # flat background -> no stray features
    image_b = np.full((h, w), 20, np.uint8)

    for (a, b) in zip(p1, p2):
        x1, y1 = int(round(a[0])), int(round(a[1]))
        x2, y2 = int(round(b[0])), int(round(b[1]))
        if not (half < x1 < w - half and half < y1 < h - half):
            continue
        if not (half < x2 < w - half and half < y2 < h - half):
            continue
        texture = rng.integers(90, 256, (patch, patch), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.7)
        image_a[y1 - half : y1 + half, x1 - half : x1 + half] = texture
        image_b[y2 - half : y2 + half, x2 - half : x2 + half] = texture

    image_a = cv2.GaussianBlur(image_a, (3, 3), 0.5)
    image_b = cv2.GaussianBlur(image_b, (3, 3), 0.5)
    return image_a, image_b


def _noise_pair(seed: int = 3, w: int = 320, h: int = 240) -> tuple:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w), dtype=np.uint8)
    b = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return a, b


# ---------------------------------------------------------------------------
# Failure modes - must never fabricate geometry
# ---------------------------------------------------------------------------

def test_missing_frames_directory_reports_no_frames(tmp_path):
    result = reconstruct_job("nope", tmp_path / "data", write_artifacts=False)
    assert result.status == STATUS_NO_FRAMES
    assert result.ok is False
    assert result.n_points == 0
    assert result.status in FAILURE_STATUSES


def test_empty_frames_directory_reports_no_frames(tmp_path):
    data = tmp_path / "data"
    (data / "empty-job" / "frames").mkdir(parents=True)
    result = reconstruct_job("empty-job", data, write_artifacts=False)
    assert result.status == STATUS_NO_FRAMES
    assert result.n_points == 0


def test_single_frame_reports_insufficient_frames(tmp_path):
    data = tmp_path / "data"
    image = np.zeros((240, 320), np.uint8)
    _write_frames(data / "one-frame" / "frames", [image])
    result = reconstruct_job("one-frame", data, write_artifacts=False)
    assert result.status == STATUS_INSUFFICIENT_FRAMES
    assert result.ok is False
    assert result.n_points == 0


def test_unmatched_frames_report_failure_not_geometry(tmp_path):
    """Random-noise pairs share no structure -> honest failure, zero points."""
    data = tmp_path / "data"
    a, b = _noise_pair()
    _write_frames(data / "noise" / "frames", [a, b])
    result = reconstruct_job("noise", data, write_artifacts=False)
    assert result.ok is False
    assert result.status in FAILURE_STATUSES
    assert result.n_points == 0
    assert np.all(np.isfinite(result.points))          # empty array is finite-safe


def test_failed_job_writes_manifest_but_no_point_cloud(tmp_path):
    """A failure must not leave behind fabricated geometry artifacts."""
    data = tmp_path / "data"
    a, b = _noise_pair(seed=9)
    _write_frames(data / "noise2" / "frames", [a, b])
    reconstruct_job("noise2", data, write_artifacts=True)

    out = data / "noise2" / "reconstruction"
    assert (out / "reconstruction.json").is_file()
    assert not (out / "sparse_points.ply").exists()
    assert not (out / "sparse_points.npz").exists()

# ---------------------------------------------------------------------------
# Successful synthetic reconstruction - proves the maths is real
# ---------------------------------------------------------------------------

def test_synthetic_two_view_reconstruction_succeeds():
    image_a, image_b = _synthetic_two_view()
    info = reconstruct_pair(
        _frame_dict("frame_00000.png", image_a),
        _frame_dict("frame_00001.png", image_b),
    )

    assert info["n_keypoints_a"] > 20
    assert info["n_matches"] > 30, "synthetic scene must yield plenty of matches"
    assert info["status"] == STATUS_SUCCESS
    assert info["n_inliers"] > 15
    assert info["n_kept"] > 20

    points = info["points"]
    assert points.ndim == 2 and points.shape[1] == 3
    assert np.all(np.isfinite(points))
    # Camera-1 is the origin: reconstructed points must be in front of it.
    assert np.all(points[:, 2] > 0)
    # Real parallax for this baseline/depth ratio (baseline 1.0, depth 5-10).
    assert info["median_parallax_deg"] > 3.0


def test_synthetic_job_writes_expected_artifacts(tmp_path):
    data = tmp_path / "data"
    image_a, image_b = _synthetic_two_view(seed=5)
    _write_frames(data / "synth" / "frames", [image_a, image_b])

    result = reconstruct_job("synth", data, parameters={"pair_stride": 1}, write_artifacts=True)
    assert result.status == STATUS_SUCCESS
    assert result.n_points > 20

    out = data / "synth" / "reconstruction"
    assert (out / "reconstruction.json").is_file()
    assert (out / "sparse_points.ply").is_file()
    assert (out / "sparse_points.npz").is_file()
    assert (out / "cameras.json").is_file()

    manifest = json.loads((out / "reconstruction.json").read_text(encoding="utf-8"))
    for key in (
        "module",
        "phase",
        "status",
        "ok",
        "message",
        "n_points",
        "frames_used",
        "pairs",
        "stats",
        "coordinate_note",
        "not_implemented",
        "parameters",
    ):
        assert key in manifest, key
    assert manifest["status"] == STATUS_SUCCESS
    assert manifest["ok"] is True
    assert "UP-TO-SCALE" in manifest["coordinate_note"]
    assert "georeferencing_gps_wgs84" in manifest["not_implemented"]

    # PLY must declare exactly the number of reconstructed vertices.
    ply_lines = (out / "sparse_points.ply").read_text(encoding="utf-8").splitlines()
    vertex_line = next(line for line in ply_lines if line.startswith("element vertex"))
    assert int(vertex_line.split()[-1]) == result.n_points
    body = ply_lines[ply_lines.index("end_header") + 1 :]
    assert len(body) == result.n_points

    # NPZ must carry the points and their per-pair tags.
    with np.load(out / "sparse_points.npz") as data_npz:
        assert data_npz["points"].shape == (result.n_points, 3)
        assert data_npz["pair_index"].shape == (result.n_points,)

    cameras = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
    assert cameras, "camera report must not be empty on success"
    assert cameras[0]["status"] == STATUS_SUCCESS
    assert cameras[0]["rotation"] is not None
    assert cameras[0]["translation"] is not None


def test_camera_matrix_is_plausible():
    K = camera_matrix(1280, 720, focal_ratio=1.2)
    assert K.shape == (3, 3)
    assert abs(K[0, 0] - 1.2 * 1280) < 1e-6
    assert abs(K[1, 1] - 1.2 * 1280) < 1e-6
    assert abs(K[0, 2] - 640) < 1e-6
    assert abs(K[1, 2] - 360) < 1e-6
    assert K[2, 2] == 1.0


def test_select_pairs_respects_stride_and_limit():
    from reconstruction.sfm import _select_pairs

    pairs = _select_pairs(57, stride=3, max_pairs=4)
    assert len(pairs) == 4
    assert all(j - i == 3 for i, j in pairs)
    assert all(0 <= i < j < 57 for i, j in pairs)

    # Degenerate inputs still return usable pairs rather than crashing.
    assert _select_pairs(2, stride=5, max_pairs=4) == [(0, 1)]
    assert _select_pairs(1, stride=1, max_pairs=4) == []


def test_missing_frames_propagates_via_load_frames(tmp_path):
    from reconstruction.sfm import load_frames

    assert load_frames(tmp_path / "does-not-exist") == []
    (tmp_path / "blank").mkdir()
    assert load_frames(tmp_path / "blank") == []

