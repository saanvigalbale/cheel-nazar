"""Phase 8.3 + 8.4 - denser point cloud and browser-ready 3D export.

Phase 8.2 produces ONE coherent but SPARSE world cloud (~10k points). That is
enough to place cameras but far too sparse to look like anything. This module
densifies it using the depth evidence already computed by the Phase 6.4 pipeline
(Depth Anything V2) -- no new inference is run.

APPROACH (deliberately an MVP, not research-grade MVS)
-----------------------------------------------------
Real multi-view stereo needs a trusted dense correspondence field and careful
handling of textureless regions. The footage here is 4K drone video where Phase
8.2 already proved that pure triangulation yields only ~190 points per pair, so
a classical MVS pass would be both slow and unreliable. Instead we use the
**existing monocular depth maps as the dense correspondence source** and let the
verified Phase 8.2 sparse cloud supply the geometry:

    for each registered frame:
        project the sparse cloud into the frame  ->  (u, v, Z_world_units)
        sample the depth map at (u, v)          ->  disparity d in [0, 255]
        robustly fit   d = A / Z + B            (A > 0)
        invert and back-project a strided pixel grid into that frame
        transform camera-frame points into the global world frame

WHY THE AFFINE-IN-INVERSE-DEPTH FIT MATTERS
--------------------------------------------
Depth Anything V2 emits RELATIVE inverse depth (disparity) that is min-max
normalised *per frame*, so ``A`` and ``B`` differ for every frame and neither is
known. Fitting them against sparse points that already carry the global world
scale is what makes the dense cloud land in the SAME frame and at the SAME
(up-to-scale) magnitude as the sparse cloud, rather than being a per-frame guess.

HONESTY NOTES
-------------
* The result is still **UP-TO-SCALE**: no metres are implied and none are
  claimed. World units are those of the Phase 8.2 seed-pair frame.
* Depth is **monocular and relative**. Densification is evidence-guided, not
  metrically exact; surfaces can be warped or missing where the model errs.
* Only pixels whose disparity falls inside the range actually *anchored* by
  sparse SfM points are back-projected, so the model is never extrapolated into
  a fabricated far-field shell (e.g. sky).
* Frames that cannot be calibrated are skipped and recorded, never faked.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .sfm import _resolve_data_dir, camera_matrix, write_ply

STATUS_DENSE_SUCCESS = "success"
STATUS_DENSE_NO_SPARSE = "no_sparse_reconstruction"
STATUS_DENSE_NO_CAMERAS = "no_registered_cameras"
STATUS_DENSE_NO_FRAMES = "no_frames"
STATUS_DENSE_NO_DEPTH = "no_usable_depth_maps"
STATUS_DENSE_NO_POINTS = "no_dense_points"

DENSE_FAILURE_MESSAGES = {
    STATUS_DENSE_NO_SPARSE: (
        "Phase 8.2 global sparse artifacts are missing; run the global "
        "reconstruction before densification."
    ),
    STATUS_DENSE_NO_CAMERAS: "cameras_global.json contains no registered cameras.",
    STATUS_DENSE_NO_FRAMES: "No frames were found for dense reconstruction.",
    STATUS_DENSE_NO_DEPTH: (
        "No frame had enough sparse-anchored depth samples to calibrate the "
        "relative depth map."
    ),
    STATUS_DENSE_NO_POINTS: (
        "Depth maps were calibrated but no pixel survived back-projection and "
        "filtering."
    ),
}

DEFAULT_DENSE_PARAMETERS: Dict[str, Any] = {
    # Back-projection working resolution (small: we want coverage, not detail).
    "max_image_side": 640,
    "pixel_stride": 2,
    "frame_stride": 1,
    "max_frames": 57,
    "focal_ratio": 1.2,
    # Sparse-anchored calibration of the relative depth map.
    "min_calibration_points": 40,
    "depth_curve_bins": 32,
    "min_bin_anchors": 4,
    "min_distinct_disparity": 8,
    "max_curve_relative_error": 0.25,
    "min_anchor_depth": 1e-3,
    # Range actually anchored by sparse points; never extrapolated past it.
    "anchor_z_percentile_lo": 1.0,
    "anchor_z_percentile_hi": 99.0,
    "depth_range_margin": 0.05,
    # Filtering / sampling.
    "support_radius_ratio": 0.006,
    "min_view_support": 2,
    "max_support_escalations": 4,
    "voxel_ratio": 0.0015,
    "max_points": 400000,
    "max_raw_points": 6000000,
}


# --------------------------------------------------------------------------
# Core geometry helpers (pure functions, no I/O)
# --------------------------------------------------------------------------


def project_world_points(
    world_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a frame.

    Returns ``(uv, z, inside)`` where ``uv`` is (N, 2) pixel coordinates, ``z``
    is the camera-frame depth in world units, and ``inside`` flags points that
    project within the image bounds (with half a pixel of tolerance, so a point
    exactly on the border is not lost to floating-point noise). ``z <= 0`` means
    behind the camera.
    """
    pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    cam = pts @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (cam @ np.asarray(intrinsics, dtype=np.float64).T)[:, :2] / z[:, None]
    inside = (
        np.isfinite(uv).all(axis=1)
        & (z > 0)
        & (uv[:, 0] >= -0.5)
        & (uv[:, 0] < width - 0.5)
        & (uv[:, 1] >= -0.5)
        & (uv[:, 1] < height - 0.5)
    )
    return uv, z, inside


def sample_disparity(
    disparity: np.ndarray,
    uv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour sample of a disparity map at ``uv`` (in-bounds only)."""
    img = np.asarray(disparity)
    height, width = img.shape[:2]
    u = np.rint(np.asarray(uv, dtype=np.float64)[:, 0]).astype(np.int64)
    v = np.rint(np.asarray(uv, dtype=np.float64)[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    values = np.zeros(u.shape[0], dtype=np.float64)
    if inside.any():
        values[inside] = img[v[inside], u[inside]].astype(np.float64)
    return values, inside

def disparity_percentile_map(disparity: np.ndarray) -> np.ndarray:
    """Rank-transform a disparity map into per-frame percentiles in [0, 1].

    Depth Anything's output is min-max normalised *per frame*, so only the ORDER
    of pixel values is meaningful: neither the absolute grey level nor any fixed
    affine law transfers between frames. Working in disparity-percentile space
    removes the unknown per-frame scale and shift entirely.
    """
    shape = np.asarray(disparity).shape
    flat = np.asarray(disparity, dtype=np.float64).reshape(-1)
    n = flat.shape[0]
    if n == 0:
        return np.zeros(shape, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    # Average ranks for tied values. With 8-bit maps over hundreds of thousands
    # of pixels, ties are the norm, and giving them sequential ranks would make
    # a completely uniform map (e.g. a failed/blank depth map) look like it
    # spanned the full percentile range.
    _, starts, counts = np.unique(flat[order], return_index=True, return_counts=True)
    mean_ranks = np.repeat(starts + (counts - 1) / 2.0, counts)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mean_ranks
    return (ranks / max(n - 1, 1)).reshape(shape)


def fit_depth_curve(
    disparity: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Learn one frame's disparity-percentile -> camera-depth curve.

    Sparse points projected into this frame give (disparity percentile, true
    camera depth in world units) pairs. Binning those pairs by percentile and
    taking the MEDIAN depth per bin yields knots that are anchored to real
    geometry at every depth level, instead of extrapolating a two-point law.

    Returns ``None`` when the frame cannot be calibrated.
    """
    percentiles = disparity_percentile_map(disparity)
    u_anchor, inside = sample_disparity(percentiles, uv)
    z = np.asarray(depth, dtype=np.float64)

    # A blank/uniform map carries no depth ordering at all, so no curve may be
    # derived from it however many anchors are available.
    if np.unique(np.asarray(disparity)).size < int(params["min_distinct_disparity"]):
        return None

    min_abs = float(params["min_anchor_depth"])
    good = inside & np.isfinite(u_anchor) & np.isfinite(z) & (z > min_abs)
    n_anchors = int(np.count_nonzero(good))
    if n_anchors < int(params["min_calibration_points"]):
        return None

    min_bin = int(params["min_bin_anchors"])

    u = u_anchor[good]
    zz = z[good]

    n_groups = int(np.clip(
        int(params["depth_curve_bins"]), 2, max(u.shape[0] // max(min_bin, 1), 2)
    ))
    # Equal-COUNT groups rather than equal-width percentile bins: anchors are
    # rarely spread uniformly over the percentile axis, and fixed-width bins
    # collapse to two knots when they are not.
    sorted_u = u[np.argsort(u, kind="stable")]
    sorted_z = zz[np.argsort(u, kind="stable")]
    groups = np.array_split(np.arange(u.shape[0]), n_groups)

    knot_u_arr = np.array([np.median(sorted_u[g]) for g in groups], dtype=np.float64)
    knot_z_arr = np.array([np.median(sorted_z[g]) for g in groups], dtype=np.float64)
    order = np.argsort(knot_u_arr)
    knot_u_arr = knot_u_arr[order]
    # A HIGHER percentile means HIGHER disparity, i.e. NEARER: depth must be
    # non-increasing in the percentile. Enforcing that stops interpolation from
    # inventing a non-physical depth ordering when a group is noisy.
    knot_z_arr = np.minimum.accumulate(knot_z_arr[order])

    predicted = np.interp(u, knot_u_arr, knot_z_arr)
    relative_error = np.abs(predicted - zz) / np.maximum(zz, 1e-9)
    median_error = float(np.median(relative_error))
    if not np.isfinite(median_error) or median_error > float(
        params["max_curve_relative_error"]
    ):
        return None

    depth_lo = float(np.percentile(zz, float(params["anchor_z_percentile_lo"])))
    depth_hi = float(np.percentile(zz, float(params["anchor_z_percentile_hi"])))
    if not np.isfinite(depth_lo) or not np.isfinite(depth_hi) or depth_hi <= depth_lo:
        return None

    return {
        "knot_u": knot_u_arr,
        "knot_z": knot_z_arr,
        "n_anchors": n_anchors,
        "n_bins_filled": int(knot_u_arr.shape[0]),
        "median_relative_error": median_error,
        "depth_lo": depth_lo,
        "depth_hi": depth_hi,
        "u_lo": float(sorted_u[0]),
        "u_hi": float(sorted_u[-1]),
    }


def depth_from_curve(
    percentiles: np.ndarray,
    knot_u: np.ndarray,
    knot_z: np.ndarray,
    depth_lo: float,
    depth_hi: float,
    margin: float = 0.0,
    u_lo: Optional[float] = None,
    u_hi: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the percentile->depth curve; returns ``(depth, valid)``.

    Depths outside the range actually anchored by sparse points are rejected, so
    the curve is never extrapolated into a fabricated far-field shell. Likewise
    a pixel whose percentile lies outside the anchored percentile range is
    rejected, which prevents ``np.interp`` from silently clamping it to a flat
    plateau at the nearest knot.
    """
    p = np.asarray(percentiles, dtype=np.float64)
    z = np.interp(p, np.asarray(knot_u, dtype=np.float64), np.asarray(knot_z, dtype=np.float64))
    span = max(float(depth_hi) - float(depth_lo), 1e-9)
    valid = (
        np.isfinite(z)
        & (z >= float(depth_lo) - margin * span)
        & (z <= float(depth_hi) + margin * span)
    )
    if u_lo is not None and u_hi is not None:
        valid &= (p >= float(u_lo)) & (p <= float(u_hi))
    return np.where(valid, z, 0.0), valid


def backproject_depth(
    disparity: np.ndarray,
    curve: Dict[str, Any],
    intrinsics: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a strided pixel grid into camera-frame 3D points.

    Returns ``(points, valid_mask)`` with ``points`` shaped (H, W, 3); invalid
    pixels are exactly zero.
    """
    percentiles = disparity_percentile_map(disparity)
    depth, valid = depth_from_curve(
        percentiles,
        curve["knot_u"],
        curve["knot_z"],
        curve["depth_lo"],
        curve["depth_hi"],
        float(params["depth_range_margin"]),
        curve.get("u_lo"),
        curve.get("u_hi"),
    )

    height, width = np.asarray(disparity).shape[:2]
    stride = max(1, int(params["pixel_stride"]))
    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    us = np.arange(0, width, stride, dtype=np.float64)
    vs = np.arange(0, height, stride, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs)
    grid_d = depth[::stride, ::stride]
    grid_valid = valid[::stride, ::stride]

    points = np.stack(
        [(grid_u - cx) / fx * grid_d, (grid_v - cy) / fy * grid_d, grid_d], axis=-1
    )
    return np.where(grid_valid[..., None], points, 0.0), grid_valid


def camera_to_world(
    points_camera: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Map camera-frame points to the world frame (inverse of ``world->camera``)."""
    pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    return (pts - t) @ r


def count_view_support(
    points: np.ndarray,
    frame_index: np.ndarray,
    radius: float,
) -> np.ndarray:
    """For each point, how many DISTINCT frames contribute a nearby point.

    Points that only their own frame explains are monocular depth guesses with
    no multi-view corroboration; they are the main source of dense-cloud noise.
    """
    pts = np.asarray(points, dtype=np.float64)
    frames = np.asarray(frame_index, dtype=np.int64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if radius <= 0:
        return np.ones(n, dtype=np.int64)

    keys = np.floor(pts / radius).astype(np.int64)
    offset = keys.min(axis=0) - 1
    keys = keys - offset
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]

    n_frames = int(frames.max()) + 1 if frames.size else 1
    pairs = np.unique(flat.astype(np.int64) * n_frames + frames)
    flat_of_pair = pairs // n_frames
    uniq_flat, counts = np.unique(flat_of_pair, return_counts=True)
    index = np.clip(np.searchsorted(uniq_flat, flat), 0, uniq_flat.shape[0] - 1)
    return counts[index].astype(np.int64)


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Keep one point per occupied voxel (deterministic, first occurrence)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts
    if voxel_size <= 0:
        return pts

    keys = np.floor(pts / voxel_size).astype(np.int64)
    offset = keys.min(axis=0) - 1
    keys = keys - offset
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first_index = np.unique(flat, return_index=True)
    kept = pts[np.sort(first_index)]

    if max_points is not None and kept.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(kept.shape[0], size=int(max_points), replace=False)
        kept = kept[np.sort(choice)]
    return kept


def cloud_diagonal(points: np.ndarray) -> float:
    """Bounding-box diagonal; used to make filter sizes resolution-independent."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return 0.0
    extent = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(extent))


# --------------------------------------------------------------------------
# Browser-friendly / compact export (Phase 8.4)
# --------------------------------------------------------------------------


def write_binary_ply(points: np.ndarray, path: Union[str, Path]) -> Path:
    """Write a binary little-endian PLY point cloud (float32 xyz).

    Binary keeps a dense cloud roughly 3x smaller and far faster to write than
    the ASCII format used for the sparse cloud, and Three.js reads both.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with open(out, "wb") as handle:
        handle.write(header)
        handle.write(pts.tobytes())
    return out


def write_glb_points(points: np.ndarray, path: Union[str, Path]) -> Path:
    """Write a minimal glTF 2.0 binary (.glb) POINTS cloud.

    A hand-built single-buffer GLB avoids adding a glTF dependency and is
    directly loadable by Three.js (GLTFLoader) as a point cloud.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    count = int(pts.shape[0])

    if count:
        lo = pts.min(axis=0).astype(float).tolist()
        hi = pts.max(axis=0).astype(float).tolist()
    else:
        lo = hi = [0.0, 0.0, 0.0]

    gltf = {
        "asset": {"version": "2.0", "generator": "cheel-nazar reconstruction.dense_sfm"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "dense_points"}],
        "meshes": [{"name": "dense_points", "primitives": [
            {"attributes": {"POSITION": 0}, "mode": 0}
        ]}],
        "accessors": [{
            "bufferView": 0,
            "componentType": 5126,          # FLOAT
            "count": count,
            "type": "VEC3",
            "min": lo,
            "max": hi,
        }],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": count * 12}],
        "buffers": [{"byteLength": count * 12}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)     # pad to 4 bytes
    bin_bytes = pts.tobytes()
    bin_bytes += b"\x00" * ((4 - len(bin_bytes) % 4) % 4)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(out, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total))   # 'glTF', v2
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        handle.write(bin_bytes)
    return out


def write_ply_auto(points: np.ndarray, path: Union[str, Path], ascii_limit: int = 200000) -> Path:
    """ASCII for small clouds (human-inspectable), binary for large ones."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] <= ascii_limit:
        return write_ply(pts, path)
    return write_binary_ply(pts, path)



# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------


def load_sparse_cloud(path: Union[str, Path]) -> np.ndarray:
    """Load the Phase 8.2 global cloud (``.npz``) as an (N, 3) float64 array."""
    file = Path(path)
    if not file.is_file():
        return np.zeros((0, 3), np.float64)
    with np.load(file) as data:
        keys = list(data.keys())
        key = "points" if "points" in keys else keys[0]
        pts = np.asarray(data[key], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0, 3), np.float64)
    return pts[np.all(np.isfinite(pts), axis=1)]


def load_global_cameras(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load ``cameras_global.json`` (``{frame: {rotation, translation, ...}}``)."""
    file = Path(path)
    if not file.is_file():
        return {}
    raw = json.loads(file.read_text(encoding="utf-8"))
    cameras: Dict[str, Dict[str, Any]] = {}
    for name, entry in raw.items():
        try:
            rotation = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
            translation = np.asarray(entry["translation"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation)):
            cameras[name] = {"rotation": rotation, "translation": translation}
    return cameras


@dataclass
class DenseReconstructionResult:
    """Outcome of a dense/denser pass over an existing global reconstruction."""

    status: str
    message: str
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    frames_used: List[str] = field(default_factory=list)
    frames_skipped: List[Dict[str, str]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DENSE_SUCCESS

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "module": "cheel_nazar.reconstruction.dense_sfm",
            "phase": "8.3",
            "status": self.status,
            "ok": self.ok,
            "message": self.message,
            "geometry_type": "denser_point_cloud",
            "n_dense_points": self.n_points,
            "n_frames_used": len(self.frames_used),
            "source_frames": list(self.frames_used),
            "frames_skipped": list(self.frames_skipped),
            # Provenance that must travel with the cloud, never inferred later.
            "sparse_point_count": self.manifest.get("sparse_point_count"),
            "scale_status": self.manifest.get("scale_status"),
            "coordinate_system": self.manifest.get("coordinate_system"),
            "georeferenced": self.manifest.get("georeferenced", False),
            "intrinsics_status": self.manifest.get("intrinsics_status"),
            "depth_source": self.manifest.get("depth_source"),
            "parameters": self.manifest.get("parameters"),
            "filtering": {
                key: self.manifest.get(key)
                for key in (
                    "n_raw_points",
                    "n_points_after_support_filter",
                    "support_radius",
                    "support_radius_escalations",
                    "voxel_size",
                    "min_view_support",
                    "sparse_cloud_diagonal",
                )
            },
            "stats": self.stats,
            "limitations": self.manifest.get("limitations", []),
            "not_implemented": self.manifest.get("not_implemented", []),
        }


def _load_work_disparity(
    path: Path,
    max_side: int,
) -> Optional[np.ndarray]:
    """Load a depth map as grayscale and downscale it to the working size."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    if max_side and max(height, width) > max_side:
        scale = max_side / float(max(height, width))
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image





# --------------------------------------------------------------------------
# Dense reconstruction driver
# --------------------------------------------------------------------------

LIMITATIONS = [
    "UP_TO_SCALE: world units are those of the Phase 8.2 seed-pair frame. "
    "No metres, distances or sizes are implied and none are claimed.",
    "Intrinsics are a heuristic (f = focal_ratio * max(width, height)), NOT "
    "calibrated drone intrinsics.",
    "NOT GEOREFERENCED: no GPS/IMU/WGS84 alignment exists. Camera centres are "
    "not geographic coordinates.",
    "Depth is monocular RELATIVE depth (Depth Anything V2), not metric. "
    "Surfaces may be warped or missing where the depth model errs.",
    "Densification is anchored to the Phase 8.2 sparse cloud, so accuracy is "
    "bounded by that (uncalibrated) sparse reconstruction.",
    "No textured mesh: this is a point cloud suitable for MVP visualization.",
]

NOT_IMPLEMENTED = [
    "textured_mesh_generation",
    "loop_closure_bundle_adjustment",
    "georeferencing_gps_wgs84",
]


def reconstruct_dense(
    job_id: str,
    data_dir: Union[str, Path] = "data",
    parameters: Optional[Dict[str, Any]] = None,
    write_artifacts: bool = True,
) -> DenseReconstructionResult:
    """Densify an existing Phase 8.2 global reconstruction using depth maps."""
    params = {**DEFAULT_DENSE_PARAMETERS, **(parameters or {})}
    data_path = _resolve_data_dir(data_dir)
    job_dir = data_path / job_id
    rec_dir = job_dir / "reconstruction"
    frames_dir = job_dir / "frames"
    depth_dir = job_dir / "analysis" / "depth"
    out_dir = rec_dir

    cameras = load_global_cameras(rec_dir / "cameras_global.json")
    sparse = load_sparse_cloud(rec_dir / "sparse_points_global.npz")

    base: Dict[str, Any] = {
        "job_id": job_id,
        "frames_dir": str(frames_dir),
        "depth_dir": str(depth_dir),
        "parameters": params,
        "depth_source": "depth_anything_v2_relative_disparity_gray8",
        "scale_status": "UP_TO_SCALE_NO_METRIC_UNITS",
        "coordinate_system": "global_seed_pair_world_frame",
        "georeferenced": False,
        "intrinsics_status": (
            "HEURISTIC_NOT_CALIBRATED: f = focal_ratio * max(width, height)"
        ),
        "limitations": LIMITATIONS,
        "not_implemented": NOT_IMPLEMENTED,
        "based_on": "reconstruction_global.json (Phase 8.2)",
        "sparse_point_count": int(sparse.shape[0]),
        "n_registered_cameras": len(cameras),
    }

    # Tracks what each frame contributed, so a failure is diagnosable.
    frames_used: List[str] = []
    frames_skipped: List[Dict[str, str]] = []
    frame_log: List[Dict[str, Any]] = []
    total_raw = 0

    def fail(status: str) -> DenseReconstructionResult:
        # A failed dense pass must still leave an honest manifest behind, and
        # must keep the frames it already processed so the failure can be
        # diagnosed from the result itself.
        failed = DenseReconstructionResult(
            status=status,
            message=DENSE_FAILURE_MESSAGES[status],
            frames_used=list(frames_used),
            frames_skipped=list(frames_skipped),
            stats=base,
            manifest=base,
        )
        if write_artifacts:
            _finalize_dense(failed, out_dir, base)
        return failed

    if not (rec_dir / "cameras_global.json").is_file() or sparse.shape[0] == 0:
        return fail(STATUS_DENSE_NO_SPARSE)
    if not cameras:
        return fail(STATUS_DENSE_NO_CAMERAS)
    if not frames_dir.is_dir():
        return fail(STATUS_DENSE_NO_FRAMES)

    names = sorted(cameras.keys())
    stride = max(1, int(params["frame_stride"]))
    names = names[::stride]
    if len(names) > int(params["max_frames"]):
        step = len(names) / float(params["max_frames"])
        names = [names[int(i * step)] for i in range(int(params["max_frames"]))]

    max_side = int(params["max_image_side"])
    focal_ratio = float(params["focal_ratio"])
    diagonal = cloud_diagonal(sparse)
    support_radius = diagonal * float(params["support_radius_ratio"])
    voxel_size = diagonal * float(params["voxel_ratio"])

    frame_chunks: List[np.ndarray] = []
    index_chunks: List[np.ndarray] = []

    for frame_index, name in enumerate(names):
        camera = cameras[name]
        frame_path = frames_dir / name
        depth_path = depth_dir / name
        if not frame_path.is_file():
            frames_skipped.append({"frame": name, "reason": "frame_image_missing"})
            continue
        if not depth_path.is_file():
            frames_skipped.append({"frame": name, "reason": "depth_map_missing"})
            continue

        disparity = _load_work_disparity(depth_path, max_side)
        if disparity is None:
            frames_skipped.append({"frame": name, "reason": "depth_map_unreadable"})
            continue

        height, width = disparity.shape[:2]
        intrinsics = camera_matrix(width, height, focal_ratio)
        rotation = camera["rotation"]
        translation = camera["translation"]

        uv, z, inside = project_world_points(
            sparse, rotation, translation, intrinsics, width, height
        )
        fit = fit_depth_curve(disparity, uv, z, params)
        if fit is None:
            frames_skipped.append(
                {"frame": name, "reason": "depth_curve_not_anchored"}
            )
            continue

        cam_points, valid = backproject_depth(disparity, fit, intrinsics, params)
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            frames_skipped.append({"frame": name, "reason": "no_valid_depth_pixels"})
            continue

        world = camera_to_world(cam_points[valid], rotation, translation)
        world = world[np.all(np.isfinite(world), axis=1)]
        if world.shape[0] == 0:
            frames_skipped.append({"frame": name, "reason": "non_finite_backprojection"})
            continue

        remaining = int(params["max_raw_points"]) - total_raw
        if remaining <= 0:
            frames_skipped.append({"frame": name, "reason": "raw_point_budget_reached"})
            break
        if world.shape[0] > remaining:
            world = world[:remaining]

        frame_chunks.append(world)
        index_chunks.append(np.full(world.shape[0], frame_index, dtype=np.int64))
        total_raw += int(world.shape[0])
        frames_used.append(name)
        frame_log.append({
            "frame": name,
            "status": "used",
            "n_anchors": fit["n_anchors"],
            "n_curve_bins_filled": fit["n_bins_filled"],
            "curve_median_relative_error": round(fit["median_relative_error"], 5),
            "depth_lo": round(fit["depth_lo"], 6),
            "depth_hi": round(fit["depth_hi"], 6),
            "n_backprojected": int(world.shape[0]),
            "n_valid_depth_pixels": n_valid,
        })

    base.update({
        "n_frames_available": len(names),
        "n_frames_used": len(frames_used),
        "n_frames_skipped": len(frames_skipped),
        "n_raw_points": int(total_raw),
        "support_radius": round(float(support_radius), 6),
        "voxel_size": round(float(voxel_size), 6),
        "sparse_cloud_diagonal": round(float(diagonal), 6),
        "frame_log": frame_log,
    })

    if not frames_used:
        return fail(STATUS_DENSE_NO_DEPTH)

    raw_points = np.concatenate(frame_chunks, axis=0)
    raw_index = np.concatenate(index_chunks, axis=0)

    support = count_view_support(raw_points, raw_index, support_radius)
    min_support = int(params["min_view_support"])
    supported = raw_points[support >= min_support]

    # If nothing at all is corroborated at this radius the filter is simply too
    # tight for the scene, not evidence that the geometry is wrong. Coarsen it
    # a bounded number of times before giving up, and record what was used.
    escalations = 0
    for _ in range(int(params["max_support_escalations"])):
        if supported.shape[0] > 0 or min_support <= 1:
            break
        support_radius *= 4.0
        escalations += 1
        support = count_view_support(raw_points, raw_index, support_radius)
        supported = raw_points[support >= min_support]
    base["support_radius"] = round(float(support_radius), 6)
    base["support_radius_escalations"] = escalations

    dense_points = voxel_downsample(
        supported,
        voxel_size,
        max_points=int(params["max_points"]),
    )

    base.update({
        "n_points_after_support_filter": int(supported.shape[0]),
        "n_points_final": int(dense_points.shape[0]),
        "median_view_support": float(np.median(support)) if support.size else None,
        "support_filter_removed": int(raw_points.shape[0] - supported.shape[0]),
    })

    if dense_points.shape[0] == 0:
        return fail(STATUS_DENSE_NO_POINTS)

    result = DenseReconstructionResult(
        status=STATUS_DENSE_SUCCESS,
        message=(
            f"Denser point cloud built from {len(frames_used)} frame(s) anchored "
            f"to the Phase 8.2 global sparse cloud. Up-to-scale, not metric."
        ),
        points=dense_points,
        frames_used=frames_used,
        frames_skipped=frames_skipped,
        stats=base,
        manifest=base,
    )

    if write_artifacts:
        _finalize_dense(result, out_dir, base)
    return result


def _finalize_dense(
    result: DenseReconstructionResult,
    out_dir: Path,
    manifest_base: Dict[str, Any],
) -> None:
    """Write the dense cloud plus a manifest that never overstates the result."""
    manifest = result.to_manifest()
    manifest["dense_point_count"] = result.n_points
    manifest["sparse_point_count"] = manifest_base.get("sparse_point_count")
    manifest["n_source_frames"] = manifest_base.get("n_frames_used")
    manifest["source_frames"] = list(result.frames_used)
    manifest["reconstruction_kind"] = "denser_point_cloud"
    manifest["is_dense"] = False
    manifest["is_sparse"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dense_reconstruction.json").write_text(
        json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8"
    )

    # A failed dense pass must never leave a cloud behind that a consumer
    # could mistake for a successful denser reconstruction.
    if not result.ok:
        return

    write_binary_ply(result.points, out_dir / "dense_points_global.ply")
    np.savez_compressed(
        out_dir / "dense_points_global.npz", points=result.points.astype(np.float64)
    )
    write_glb_points(result.points, out_dir / "dense_points_global.glb")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8.3/8.4 - denser point cloud + browser export"
    )
    parser.add_argument("--job-id", required=True, help="Job ID to densify")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-image-side", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--no-artifacts", action="store_true", help="Do not write output files"
    )
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    for key, value in (
        ("frame_stride", args.frame_stride),
        ("max_frames", args.max_frames),
        ("max_image_side", args.max_image_side),
        ("max_points", args.max_points),
    ):
        if value is not None:
            overrides[key] = value

    result = reconstruct_dense(
        args.job_id,
        args.data_dir,
        parameters=overrides or None,
        write_artifacts=not args.no_artifacts,
    )
    summary = result.to_manifest()
    for key in ("source_frames", "stats", "frames_skipped"):
        summary.pop(key, None)
    print(json.dumps(summary, indent=2, default=_json_safe))
    print(
        f"STATUS={result.status} DENSE_POINTS={result.n_points} "
        f"FRAMES_USED={len(result.frames_used)} SKIPPED={len(result.frames_skipped)}"
    )


if __name__ == "__main__":
    main()