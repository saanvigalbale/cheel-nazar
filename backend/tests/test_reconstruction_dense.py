"""Tests for Phase 8.3/8.4 dense reconstruction and 3D export.

Every fixture is synthetic and *exactly* known, so back-projection and the
world/camera transform are asserted against ground truth rather than eyeballed.
"""

import json
import struct

import cv2
import numpy as np
import pytest

from reconstruction.dense_sfm import (
    DEFAULT_DENSE_PARAMETERS,
    DENSE_FAILURE_MESSAGES,
    STATUS_DENSE_NO_CAMERAS,
    STATUS_DENSE_NO_DEPTH,
    STATUS_DENSE_NO_SPARSE,
    STATUS_DENSE_SUCCESS,
    backproject_depth,
    camera_to_world,
    cloud_diagonal,
    count_view_support,
    depth_from_curve,
    disparity_percentile_map,
    fit_depth_curve,
    load_global_cameras,
    load_sparse_cloud,
    project_world_points,
    reconstruct_dense,
    sample_disparity,
    voxel_downsample,
    write_binary_ply,
    write_glb_points,
)
from reconstruction.sfm import camera_matrix

WIDTH, HEIGHT = 160, 120
DEPTH_LO, DEPTH_HI = 10.0, 30.0


def _params(**overrides):
    params = dict(DEFAULT_DENSE_PARAMETERS)
    params.update(overrides)
    return params


def _camera_pose(tx=0.0, ry=0.0):
    rotation, _ = cv2.Rodrigues(np.array([0.0, ry, 0.0]))
    return rotation.astype(np.float64), np.array([[tx], [0.0], [0.0]]).astype(np.float64)


def _project(cam):
    k = camera_matrix(WIDTH, HEIGHT, 1.2)
    return np.column_stack([
        k[0, 0] * cam[:, 0] / cam[:, 2] + WIDTH / 2,
        k[1, 1] * cam[:, 1] / cam[:, 2] + HEIGHT / 2,
    ])


def _camera_samples(rng, n):
    """``n`` camera-frame points, each on a DISTINCT pixel of the test frame.

    Built by inverse projection so the sample coordinates and the camera-frame
    geometry are exactly consistent: sampling a 3D volume instead would collide
    on pixels and let anchors read back another point's disparity.

    Depth varies SMOOTHLY across the frame, as a real monocular depth map does.
    Independent random depths per pixel would be pure noise, would not survive
    JPEG encoding, and would test the codec rather than the algorithm.
    """
    k = camera_matrix(WIDTH, HEIGHT, 1.2)
    index = rng.choice(HEIGHT * WIDTH, size=int(n), replace=False)
    v, u = np.divmod(index, WIDTH)
    un = (u - WIDTH / 2.0) / (WIDTH / 2.0)
    vn = (v - HEIGHT / 2.0) / (HEIGHT / 2.0)
    mid = 0.5 * (DEPTH_LO + DEPTH_HI)
    amp = 0.5 * (DEPTH_HI - DEPTH_LO) * 0.8
    z = np.clip(
        mid + amp * (0.55 * un + 0.35 * vn + 0.25 * un * vn)
        + rng.normal(0.0, 0.15, index.shape[0]),
        DEPTH_LO,
        DEPTH_HI,
    )
    x = (u - WIDTH / 2.0) * z / k[0, 0]
    y = (v - HEIGHT / 2.0) * z / k[1, 1]
    return np.column_stack([x, y, z]), np.column_stack([u, v])


def _anchors(rng, n=800, tx=0.0, ry=0.0):
    rotation, translation = _camera_pose(tx=tx, ry=ry)
    cam, uv = _camera_samples(rng, n)
    points = camera_to_world(cam, rotation, translation)
    return camera_matrix(WIDTH, HEIGHT, 1.2), points, cam, uv


def _disparity_from_depth(depth):
    """Relative disparity values for individual points: bright = near, 1..255.

    Never 0: 0 is reserved for unpainted background pixels.
    """
    span = DEPTH_HI - DEPTH_LO
    scaled = np.clip((DEPTH_HI - np.asarray(depth, np.float64)) / span, 0.0, 1.0)
    return np.clip(1 + np.round(scaled * 254.0), 1, 255).astype(np.uint8)


def _paint(values, uv, shape=(HEIGHT, WIDTH)):
    """Paint per-point values into a 2D image at the given pixel coordinates."""
    image = np.zeros(shape, np.uint8)
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    image[v[inside], u[inside]] = values[inside]
    return image


def _flat_curve():
    return {
        "knot_u": np.array([0.0, 1.0]),
        "knot_z": np.array([10.0, 30.0]),
        "depth_lo": 10.0,
        "depth_hi": 30.0,
    }


def _synthetic_job(tmp_path, job_id="synthetic", n_cams=3, seed=5):
    """A fake job with a KNOWN scene, known poses and consistent depth maps."""
    rng = np.random.default_rng(seed)
    job_dir = tmp_path / "data" / job_id
    (job_dir / "frames").mkdir(parents=True)
    (job_dir / "analysis" / "depth").mkdir(parents=True)
    (job_dir / "reconstruction").mkdir(parents=True)

    poses, names = [], []
    for i in range(n_cams):
        poses.append(_camera_pose(tx=0.6 * i, ry=0.01 * i))
        name = f"frame_{i:05d}.jpg"
        names.append(name)
        cv2.imwrite(
            str(job_dir / "frames" / name), np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        )

    # Depth maps are painted from true scene geometry so the disparity really is
    # a monotone function of camera depth. Every pixel is covered, like a real
    # monocular depth map, so anchors span the whole percentile axis.
    dense, _ = _camera_samples(rng, HEIGHT * WIDTH)
    for name, (rotation, translation) in zip(names, poses):
        cam = dense @ rotation.T + translation.T
        # Each camera's depth map must be painted where ITS OWN projection of the
        # scene lands, or it will disagree with that camera's pose.
        disparity = _paint(_disparity_from_depth(cam[:, 2]), _project(cam))
        cv2.imwrite(str(job_dir / "analysis" / "depth" / name), disparity)

    world = camera_to_world(dense[:500], *poses[0])
    np.savez_compressed(
        job_dir / "reconstruction" / "sparse_points_global.npz", points=world
    )
    cameras = {
        name: {
            "rotation": rotation.tolist(),
            "translation": translation.reshape(3).tolist(),
            "center": (-rotation.T @ translation).ravel().tolist(),
            "method": "synthetic",
        }
        for name, (rotation, translation) in zip(names, poses)
    }
    (job_dir / "reconstruction" / "cameras_global.json").write_text(
        json.dumps(cameras), encoding="utf-8"
    )
    return job_id, world


# ---------------------------------------------------------------------------
# Projection and world/camera transforms
# ---------------------------------------------------------------------------

def test_project_world_points_matches_pinhole_projection():
    k = camera_matrix(WIDTH, HEIGHT, 1.2)
    rotation, translation = _camera_pose(tx=1.0, ry=0.05)
    cam, _ = _camera_samples(np.random.default_rng(1), 50)
    points = camera_to_world(cam, rotation, translation)

    uv, z, inside = project_world_points(
        points, rotation, translation, k, WIDTH, HEIGHT
    )
    assert np.allclose(uv[:, 0], k[0, 0] * cam[:, 0] / cam[:, 2] + WIDTH / 2)
    assert np.allclose(uv[:, 1], k[1, 1] * cam[:, 1] / cam[:, 2] + HEIGHT / 2)
    assert np.allclose(z, cam[:, 2])
    assert inside.all()


def test_points_behind_camera_are_excluded():
    behind = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, -1.0]])
    _, z, inside = project_world_points(
        behind, np.eye(3), np.zeros((3, 1)), camera_matrix(WIDTH, HEIGHT, 1.2),
        WIDTH, HEIGHT,
    )
    assert np.all(z < 0)
    assert not inside.any(), "points behind the camera must never be usable"


def test_camera_to_world_inverts_the_world_to_camera_projection():
    """The full round trip world -> camera -> world must be lossless."""
    rotation, translation = _camera_pose(tx=2.0, ry=0.2)
    cam, _ = _camera_samples(np.random.default_rng(2), 200)
    world = camera_to_world(cam, rotation, translation)

    assert np.allclose(camera_to_world(cam, rotation, translation), world, atol=1e-9)
    assert np.allclose(world @ rotation.T + translation.T, cam, atol=1e-9)


def test_sample_disparity_marks_out_of_bounds():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    values, inside = sample_disparity(
        img, np.array([[0.0, 0.0], [3.0, 2.0], [-1.0, 0.0]])
    )
    assert inside.tolist() == [True, True, False]
    assert values[0] == 0 and values[1] == 11
    assert values[2] == 0.0, "out-of-bounds samples must not leak values"
# ---------------------------------------------------------------------------
# Relative depth -> anchored world depth
# ---------------------------------------------------------------------------

def test_disparity_percentile_map_is_a_pure_rank_transform():
    img = np.array([[10, 10, 30], [30, 5, 20]], dtype=np.uint8)
    pct = disparity_percentile_map(img)

    assert pct.shape == img.shape
    # values 5,10,10,20,30,30 -> average ranks 0, 1.5, 1.5, 3, 4.5, 4.5
    assert set(np.round(pct, 6).ravel().tolist()) == {0.0, 0.3, 0.6, 0.9}
    assert pct[0, 0] == pct[0, 1], "ties must share a percentile"
    assert pct[0, 0] < pct[0, 2]
    assert pct[1, 1] == 0.0 and pct[0, 2] == 0.9


def test_disparity_percentile_map_of_a_uniform_map_is_uniform():
    """A blank depth map must not masquerade as spanning the full range."""
    pct = disparity_percentile_map(np.full((8, 8), 77, np.uint8))
    assert np.allclose(pct, 0.5)


def test_disparity_percentile_map_is_scale_and_shift_invariant():
    """Per-frame min-max normalisation must not change the map."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (16, 16)).astype(np.float64)
    rescaled = img * 2.5 + 100.0          # strictly increasing, order preserved

    assert np.allclose(
        disparity_percentile_map(img), disparity_percentile_map(rescaled), atol=1e-9
    )


def test_fit_depth_curve_recovers_the_anchored_depth_range():
    _, _, cam, uv = _anchors(np.random.default_rng(3), n=8000)
    disparity = _paint(_disparity_from_depth(cam[:, 2]), uv)

    curve = fit_depth_curve(disparity, uv, cam[:, 2], _params())
    assert curve is not None
    assert curve["knot_z"].shape == curve["knot_u"].shape
    assert curve["median_relative_error"] < 0.1
    assert DEPTH_LO <= curve["depth_lo"] <= curve["depth_hi"] <= DEPTH_HI
    # Higher percentile == higher disparity == nearer: depth must not increase.
    assert np.all(np.diff(curve["knot_z"]) <= 1e-9), "curve must be monotone in depth"


def test_fit_depth_curve_rejects_a_frame_with_too_few_anchors():
    rng = np.random.default_rng(4)
    uv = np.column_stack([rng.uniform(0, WIDTH, 3), rng.uniform(0, HEIGHT, 3)])
    disparity = np.zeros((HEIGHT, WIDTH), np.uint8)

    assert fit_depth_curve(
        disparity, uv, rng.uniform(DEPTH_LO, DEPTH_HI, 3), _params()
    ) is None


def test_fit_depth_curve_rejects_geometry_the_depth_model_contradicts():
    """If anchors disagree with the disparity map, no curve may be invented."""
    rng = np.random.default_rng(6)
    disparity = np.full((HEIGHT, WIDTH), 200, np.uint8)
    uv = np.column_stack([rng.uniform(0, WIDTH, 400), rng.uniform(0, HEIGHT, 400)])
    z = rng.uniform(DEPTH_LO, DEPTH_HI, 400)      # uncorrelated with disparity

    assert fit_depth_curve(disparity, uv, z, _params()) is None


def test_depth_from_curve_rejects_depths_outside_the_anchored_range():
    """Prevents a fabricated far-field shell where the model was never anchored."""
    depth, valid = depth_from_curve(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([10.0, 20.0, 30.0]),
        12.0,
        28.0,
    )
    assert valid.tolist() == [False, True, False]
    assert depth[0] == 0.0 and depth[2] == 0.0
    assert depth[1] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------

def test_backproject_depth_recovers_true_camera_frame_points():
    k, _, cam, uv = _anchors(np.random.default_rng(7), n=14000, tx=1.5, ry=0.1)
    disparity = _paint(_disparity_from_depth(cam[:, 2]), uv)
    params = _params(pixel_stride=1)

    curve = fit_depth_curve(disparity, uv, cam[:, 2], params)
    assert curve is not None

    grid, valid = backproject_depth(disparity, curve, k, params)
    assert grid.shape == (HEIGHT, WIDTH, 3)

    # Compare by exact pixel correspondence so no ordering assumption is needed:
    # every painted pixel must land on the true camera-frame point behind it.
    ui = np.rint(uv[:, 0]).astype(int)
    vi = np.rint(uv[:, 1]).astype(int)
    error = np.linalg.norm(grid[vi, ui] - cam, axis=1)
    assert error.shape[0] > 1000

    # Bulk accuracy must be tight: the recovered geometry really is the geometry.
    assert float(np.median(error)) < 0.05, "median back-projection error"
    assert float(np.percentile(error, 90)) < 0.15, "p90 back-projection error"
    # The far tail is an inherent limit of 8-bit RELATIVE depth: a whole depth
    # band near the horizon collapses into one grey level, so it cannot be
    # recovered more finely than that quantisation step.
    assert float(np.percentile(error, 99)) < 1.0
    assert float(np.mean(error < 0.35)) > 0.95


def test_backproject_depth_marks_invalid_pixels_exactly_zero():
    """Nothing in the anchored depth range -> nothing may be back-projected."""
    k = camera_matrix(WIDTH, HEIGHT, 1.2)
    rng = np.random.default_rng(11)
    disparity = rng.integers(60, 200, (HEIGHT, WIDTH)).astype(np.uint8)
    curve = _flat_curve()
    curve["depth_lo"], curve["depth_hi"] = 40.0, 50.0   # unreachable

    grid, valid = backproject_depth(disparity, curve, k, _params())
    assert not valid.any()
    assert np.all(grid == 0.0), "invalid pixels must be exactly zero, never NaN"


def test_backproject_depth_respects_pixel_stride():
    k = camera_matrix(WIDTH, HEIGHT, 1.2)
    rng = np.random.default_rng(8)
    disparity = rng.integers(60, 200, (HEIGHT, WIDTH)).astype(np.uint8)
    grid, valid = backproject_depth(
        disparity, _flat_curve(), k, _params(pixel_stride=4)
    )
    assert grid.shape == (len(range(0, HEIGHT, 4)), len(range(0, WIDTH, 4)), 3)
    assert valid.sum() == grid.shape[0] * grid.shape[1]


def test_backproject_depth_survives_a_degenerate_disparity_map():
    grid, valid = backproject_depth(
        np.zeros((0, 0), np.uint8), _flat_curve(),
        camera_matrix(WIDTH, HEIGHT, 1.2), _params(),
    )
    assert grid.shape[-1] == 3
    assert not valid.any()
# ---------------------------------------------------------------------------
# Filtering / sampling
# ---------------------------------------------------------------------------

def test_count_view_support_counts_distinct_frames_not_raw_points():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],       # same voxel, same frame
        [0.2, 0.0, 0.0],       # same voxel, same frame
        [0.05, 0.0, 0.0],      # same voxel, different frame
        [50.0, 50.0, 50.0],    # isolated
    ])
    frames = np.array([0, 0, 0, 1, 2])

    support = count_view_support(points, frames, radius=1.0)
    assert support[:4].tolist() == [2, 2, 2, 2], "counts DISTINCT frames"
    assert support[4] == 1


def test_count_view_support_edges():
    assert count_view_support(np.zeros((0, 3)), np.zeros((0,)), 1.0).shape == (0,)
    single = np.array([[1.0, 2.0, 3.0]])
    assert count_view_support(single, np.array([4]), 1.0).tolist() == [1]


def test_voxel_downsample_keeps_one_point_per_voxel():
    points = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],     # same voxel
        [5.00, 5.00, 5.00],     # different voxel
    ])
    kept = voxel_downsample(points, 1.0)
    assert kept.shape == (2, 3)
    assert np.allclose(kept[0], points[0]), "keeps the first point deterministically"
    assert np.allclose(kept[1], points[2])


def test_voxel_downsample_caps_deterministically():
    rng = np.random.default_rng(9)
    points = rng.normal(scale=10, size=(500, 3))
    first = voxel_downsample(points, 0.5, max_points=50, seed=1)
    second = voxel_downsample(points, 0.5, max_points=50, seed=1)
    assert first.shape[0] <= 50
    assert np.allclose(first, second), "sampling must be reproducible"


def test_voxel_downsample_edge_cases():
    assert voxel_downsample(np.zeros((0, 3)), 1.0).shape == (0, 3)
    pts = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(voxel_downsample(pts, 0.0), pts), "voxel <= 0 is a no-op"


def test_cloud_diagonal():
    assert cloud_diagonal(np.zeros((0, 3))) == 0.0
    cube = np.array([[0.0, 0, 0], [3.0, 4.0, 0.0]])
    assert cloud_diagonal(cube) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Artifact loading and export formats
# ---------------------------------------------------------------------------

def test_load_sparse_cloud_handles_missing_and_malformed(tmp_path):
    assert load_sparse_cloud(tmp_path / "nope.npz").shape == (0, 3)

    good = tmp_path / "good.npz"
    np.savez_compressed(good, points=np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]]))
    assert load_sparse_cloud(good).shape == (1, 3), "non-finite rows dropped"

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, points=np.zeros((4, 2)))
    assert load_sparse_cloud(bad).shape == (0, 3)


def test_load_global_cameras_skips_unusable_entries(tmp_path):
    path = tmp_path / "cameras_global.json"
    path.write_text(json.dumps({
        "good.jpg": {"rotation": np.eye(3).tolist(), "translation": [1, 2, 3]},
        "no_translation.jpg": {"rotation": np.eye(3).tolist()},
        "nan.jpg": {
            "rotation": np.eye(3).tolist(),
            "translation": [float("nan"), 0, 0],
        },
    }), encoding="utf-8")

    cameras = load_global_cameras(path)
    assert set(cameras) == {"good.jpg"}
    assert cameras["good.jpg"]["rotation"].shape == (3, 3)


def test_load_global_cameras_missing_file(tmp_path):
    assert load_global_cameras(tmp_path / "absent.json") == {}


def test_write_binary_ply_roundtrip(tmp_path):
    points = np.array([[1.5, -2.5, 3.5], [0.0, 0.0, 0.0]])
    raw = write_binary_ply(points, tmp_path / "cloud.ply").read_bytes()

    header, _, payload = raw.partition(b"end_header\n")
    text = header.decode("ascii")
    assert text.startswith("ply")
    assert "format binary_little_endian 1.0" in text
    assert "element vertex 2" in text
    assert np.allclose(
        np.frombuffer(payload, dtype="<f4").reshape(-1, 3), points, atol=1e-6
    )


def test_write_glb_points_is_a_valid_gltf_binary(tmp_path):
    points = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    raw = write_glb_points(points, tmp_path / "cloud.glb").read_bytes()

    magic, version, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67, "glTF magic"
    assert version == 2
    assert total == len(raw), "declared length must match the file size"

    json_len, json_type = struct.unpack("<II", raw[12:20])
    assert json_type == 0x4E4F534A
    gltf = json.loads(raw[20:20 + json_len].decode("utf-8"))
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["meshes"][0]["primitives"][0]["mode"] == 0, "POINTS primitive"
    assert gltf["accessors"][0]["count"] == 2
    assert gltf["accessors"][0]["type"] == "VEC3"
    assert gltf["accessors"][0]["componentType"] == 5126
    assert gltf["accessors"][0]["min"] == [-1.0, -2.0, -3.0]
    assert gltf["accessors"][0]["max"] == [1.0, 2.0, 3.0]

    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack("<II", raw[bin_off:bin_off + 8])
    assert bin_type == 0x004E4942
    payload = raw[bin_off + 8: bin_off + 8 + bin_len]
    assert np.allclose(
        np.frombuffer(payload, dtype="<f4").reshape(-1, 3), points, atol=1e-6
    )
# ---------------------------------------------------------------------------
# End-to-end behaviour and honesty guarantees
# ---------------------------------------------------------------------------

def test_reconstruct_dense_reports_missing_sparse_artifacts(tmp_path):
    result = reconstruct_dense("no_such_job", tmp_path / "data", write_artifacts=False)

    assert result.status == STATUS_DENSE_NO_SPARSE
    assert result.ok is False
    assert result.n_points == 0
    assert result.message == DENSE_FAILURE_MESSAGES[STATUS_DENSE_NO_SPARSE]


def test_reconstruct_dense_reports_no_registered_cameras(tmp_path):
    job_id, _ = _synthetic_job(tmp_path)
    rec = tmp_path / "data" / job_id / "reconstruction"
    (rec / "cameras_global.json").write_text("{}", encoding="utf-8")

    result = reconstruct_dense(job_id, tmp_path / "data", write_artifacts=False)
    assert result.status == STATUS_DENSE_NO_CAMERAS
    assert result.ok is False


def test_reconstruct_dense_reports_when_no_depth_map_can_be_anchored(tmp_path):
    job_id, _ = _synthetic_job(tmp_path)
    depth_dir = tmp_path / "data" / job_id / "analysis" / "depth"
    for path in depth_dir.iterdir():
        cv2.imwrite(str(path), np.zeros((HEIGHT, WIDTH), np.uint8))

    result = reconstruct_dense(
        job_id, tmp_path / "data", parameters={"max_image_side": HEIGHT},
        write_artifacts=False,
    )
    assert result.status == STATUS_DENSE_NO_DEPTH
    assert result.ok is False
    assert result.n_points == 0
    assert all(f["reason"] == "depth_curve_not_anchored" for f in result.frames_skipped)


def test_reconstruct_dense_skips_frames_without_depth_maps(tmp_path):
    job_id, _ = _synthetic_job(tmp_path, n_cams=4)
    depth_dir = tmp_path / "data" / job_id / "analysis" / "depth"
    (depth_dir / "frame_00003.jpg").unlink()

    result = reconstruct_dense(
        job_id, tmp_path / "data", parameters={"max_image_side": HEIGHT},
        write_artifacts=False,
    )
    reasons = {f["frame"]: f["reason"] for f in result.frames_skipped}
    assert reasons.get("frame_00003.jpg") == "depth_map_missing"
    assert "frame_00003.jpg" not in result.frames_used


def test_reconstruct_dense_end_to_end_writes_all_artifacts(tmp_path):
    job_id, world = _synthetic_job(tmp_path)
    out = tmp_path / "data" / job_id / "reconstruction"

    result = reconstruct_dense(
        job_id, tmp_path / "data",
        parameters={"max_image_side": HEIGHT, "pixel_stride": 1},
        write_artifacts=True,
    )

    assert result.status == STATUS_DENSE_SUCCESS, result.message
    assert result.ok is True
    assert result.n_points > 0
    assert np.all(np.isfinite(result.points))
    assert len(result.frames_used) == 3
    # Same world frame as the sparse cloud (not a fresh local frame).
    assert result.points[:, 0].min() > -20 and result.points[:, 0].max() < 20
    assert result.n_points > world.shape[0], "must actually be denser"

    for name in (
        "dense_reconstruction.json",
        "dense_points_global.ply",
        "dense_points_global.npz",
        "dense_points_global.glb",
    ):
        assert (out / name).is_file(), f"missing artifact: {name}"

    with np.load(out / "dense_points_global.npz") as data:
        stored = data["points"]
        assert stored.shape == (result.n_points, 3)
        assert np.allclose(stored, result.points)

    header = (out / "dense_points_global.ply").read_bytes().partition(b"end_header\n")[0]
    assert f"element vertex {result.n_points}" in header.decode("ascii")
def test_dense_manifest_records_provenance_and_limits(tmp_path):
    job_id, world = _synthetic_job(tmp_path)
    out = tmp_path / "data" / job_id / "reconstruction"

    reconstruct_dense(
        job_id, tmp_path / "data",
        parameters={"max_image_side": HEIGHT}, write_artifacts=True,
    )
    manifest = json.loads(
        (out / "dense_reconstruction.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == STATUS_DENSE_SUCCESS
    assert manifest["sparse_point_count"] == len(world)
    assert manifest["dense_point_count"] > 0
    assert manifest["n_source_frames"] == 3
    assert len(manifest["source_frames"]) == 3
    assert manifest["scale_status"] == "UP_TO_SCALE_NO_METRIC_UNITS"
    assert manifest["coordinate_system"] == "global_seed_pair_world_frame"
    assert manifest["georeferenced"] is False
    assert "HEURISTIC_NOT_CALIBRATED" in manifest["intrinsics_status"]
    assert "depth_anything_v2" in manifest["depth_source"]
    assert manifest["reconstruction_kind"] == "denser_point_cloud"
    assert manifest["is_dense"] is False, "must not overstate the result"
    assert manifest["limitations"]
    assert manifest["parameters"]["voxel_ratio"] > 0

    text = json.dumps(manifest).lower()
    for forbidden in ("latitude", "longitude", "utm", "elevation_m", "distance_m"):
        assert forbidden not in text, f"manifest must not claim {forbidden}"


def test_failed_dense_run_writes_manifest_but_no_cloud(tmp_path):
    job_id, _ = _synthetic_job(tmp_path)
    depth_dir = tmp_path / "data" / job_id / "analysis" / "depth"
    for path in depth_dir.iterdir():
        cv2.imwrite(str(path), np.zeros((HEIGHT, WIDTH), np.uint8))
    out = tmp_path / "data" / job_id / "reconstruction"

    reconstruct_dense(
        job_id, tmp_path / "data", parameters={"max_image_side": HEIGHT},
        write_artifacts=True,
    )

    manifest = json.loads((out / "dense_reconstruction.json").read_text(encoding="utf-8"))
    assert manifest["status"] == STATUS_DENSE_NO_DEPTH
    assert not (out / "dense_points_global.ply").exists()
    assert not (out / "dense_points_global.glb").exists()
    assert not (out / "dense_points_global.npz").exists()