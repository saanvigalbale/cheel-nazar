"""Phase 8.2 - GLOBAL sparse SfM: one coherent cloud from many frame pairs.

Phase 8.1 reconstructs each image pair independently, so every pair has its own
local coordinate frame. This module stitches them into a single world frame
using standard incremental Structure-from-Motion built entirely from OpenCV
primitives already present in the project:

    seed pair (first pair that reconstructs)
        -> world frame := camera A of that pair (R = I, t = 0)
    for each remaining frame, in temporal order:
        gather 3D-2D correspondences by matching the frame against every
        already-registered frame and looking the matches up in that frame's
        observation table
        -> solvePnPRansac  => global pose (R, t) of the new camera
        -> validate inliers / reprojection error, else mark the frame failed
        -> triangulate NEW points between the new camera and already
           registered neighbours, both poses now known in world coordinates

HONESTY NOTES
-------------
* Still **UP-TO-SCALE** monocular reconstruction: no metres are implied.
* Intrinsics remain a **heuristic** (f = focal_ratio * max(w, h)), NOT
  calibrated drone intrinsics. This is stated in every manifest.
* Nothing is fabricated: frames that cannot be registered are reported in
  ``frames_failed`` with a reason; a run with no seed pair or no registered
  camera returns an explicit failure status with zero points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .sfm import (
    STATUS_INSUFFICIENT_FRAMES,
    STATUS_NO_FRAMES,
    STATUS_SUCCESS,
    _resolve_data_dir,
    camera_matrix,
    detect_features,
    estimate_relative_pose,
    load_frames,
    match_features,
    triangulate_and_filter,
    write_ply,
)

STATUS_GLOBAL_NO_SEED = "no_seed_pair"
STATUS_GLOBAL_NO_REGISTRATION = "no_camera_registered"

GLOBAL_FAILURE_MESSAGES = {
    STATUS_NO_FRAMES: "No readable frames were found for global reconstruction.",
    STATUS_INSUFFICIENT_FRAMES: "At least two frames are required for global SfM.",
    STATUS_GLOBAL_NO_SEED: (
        "No selected frame pair produced a valid two-view reconstruction, so "
        "no world frame could be established."
    ),
    STATUS_GLOBAL_NO_REGISTRATION: (
        "A seed pair existed but no additional camera could be registered "
        "with 3D-2D correspondences."
    ),
}

DEFAULT_GLOBAL_PARAMETERS: Dict[str, Any] = {
    "max_image_side": 1280,
    "focal_ratio": 1.2,
    "n_features": 4000,
    "ratio_test": 0.75,
    "min_matches": 30,
    "ransac_thresh_px": 1.5,
    "min_inliers": 15,
    "max_reproj_error_px": 2.0,
    "min_good_points": 50,
    "min_parallax_deg": 0.5,
    "registration_stride": 1,
    "seed_stride": 1,
    "max_frames": 57,
    "min_pnp_points": 30,
    "min_pnp_inliers": 12,
    "pnp_reproj_error_px": 3.0,
    "max_triangulation_partners": 2,
    "max_points": 200000,
}

@dataclass
class GlobalReconstructionResult:
    """Outcome of a global (multi-frame) sparse reconstruction."""

    status: str
    message: str
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    cameras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frames_registered: List[str] = field(default_factory=list)
    frames_failed: List[Dict[str, str]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def trajectory(self) -> List[Dict[str, Any]]:
        """Ordered camera centres (up-to-scale, never GPS)."""
        return [
            {
                "frame": name,
                "center": self.cameras[name].get("center"),
                "rotation": self.cameras[name].get("rotation"),
                "translation": self.cameras[name].get("translation"),
                "n_observations": self.cameras[name].get("n_observations"),
                "method": self.cameras[name].get("method"),
            }
            for name in self.frames_registered
            if name in self.cameras
        ]

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "module": "cheel_nazar.reconstruction.global_sfm",
            "phase": "8.2",
            "status": self.status,
            "ok": self.ok,
            "message": self.message,
            "n_points": self.n_points,
            "n_frames_registered": len(self.frames_registered),
            "frames_registered": self.frames_registered,
            "frames_failed": self.frames_failed,
            "cameras": self.cameras,
            "stats": self.stats,
            "scale_status": "UP_TO_SCALE_NO_METRIC_UNITS",
            "intrinsics_status": (
                "HEURISTIC_NOT_CALIBRATED: f = focal_ratio * max(width, height)"
            ),
            "coordinate_note": (
                "All points and camera poses share ONE world frame defined by the "
                "seed pair's first camera. Up-to-scale monocular reconstruction: "
                "no metres, no GPS, no WGS84."
            ),
            "not_implemented": [
                "bundle_adjustment",
                "loop_closure",
                "dense_reconstruction_mvs",
                "mesh_generation",
                "georeferencing_gps_wgs84",
            ],
        }


def select_frame_indices(
    n_frames: int,
    stride: int = 1,
    max_frames: int = 57,
) -> List[int]:
    """Configurable keyframe selection over the existing extracted frames."""
    stride = max(1, int(stride))
    indices = list(range(0, n_frames, stride))
    if len(indices) > max_frames:
        step = len(indices) / float(max_frames)
        indices = [indices[int(i * step)] for i in range(max_frames)]
    return indices


class FrameStore:
    """Caches SIFT features per frame and mutual matches per frame pair."""

    def __init__(self, frames: List[Dict[str, Any]], params: Dict[str, Any]) -> None:
        self.frames = frames
        self.params = params
        self._features: Dict[int, Tuple[List[Any], Optional[np.ndarray]]] = {}
        self._matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._detector = cv2.SIFT_create(nfeatures=int(params["n_features"]))
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._intrinsics: Dict[int, np.ndarray] = {}

    def intrinsics(self, index: int) -> np.ndarray:
        if index not in self._intrinsics:
            frame = self.frames[index]
            self._intrinsics[index] = camera_matrix(
                frame["width"], frame["height"], float(self.params["focal_ratio"])
            )
        return self._intrinsics[index]

    def features(self, index: int):
        if index not in self._features:
            self._features[index] = detect_features(
                self.frames[index]["gray"], detector=self._detector
            )
        return self._features[index]

    def matches(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Mutual matches as ``(kp_index_in_frame_i, kp_index_in_frame_j)``."""
        if i == j:
            return []
        key = (i, j) if i < j else (j, i)
        if key not in self._matches:
            _, desc_a = self.features(key[0])
            _, desc_b = self.features(key[1])
            self._matches[key] = [
                (m.queryIdx, m.trainIdx)
                for m in match_features(
                    desc_a, desc_b, self.params["ratio_test"], self._matcher
                )
            ]
        cached = self._matches[key]
        if i < j:
            return cached
        return [(b, a) for (a, b) in cached]

# --------------------------------------------------------------------------
# Scene degeneracy diagnostics
# --------------------------------------------------------------------------

def planarity_report(points: np.ndarray) -> Dict[str, Any]:
    """Quantify how planar a point set is.

    Two-view structure from a (near) planar scene is projectively ambiguous and
    absolute PnP on coplanar points is ill-posed, so this is reported rather
    than hidden.
    """
    empty = {
        "n_points": int(points.shape[0]),
        "planar": None,
        "third_axis_ratio": None,
        "plane_rms_residual": None,
        "point_spread": None,
    }
    if points.shape[0] < 3:
        return empty
    centered = points - points.mean(axis=0)
    try:
        _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return empty
    if sv[0] <= 0:
        return empty
    ratio = float(sv[2] / sv[0])
    normal = vt[2]
    residual = centered @ normal
    return {
        "n_points": int(points.shape[0]),
        "planar": bool(ratio < 0.05),
        "third_axis_ratio": round(ratio, 6),
        "plane_rms_residual": round(float(np.sqrt((residual ** 2).mean())), 6),
        "point_spread": round(float(sv[0]), 6),
    }


# --------------------------------------------------------------------------
# Seed pair: establishes the world frame
# --------------------------------------------------------------------------

def seed_pair(
    store: FrameStore,
    i: int,
    j: int,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Two-view reconstruct of (i, j) with per-point keypoint indices retained."""
    keypoints_a, _ = store.features(i)
    keypoints_b, _ = store.features(j)
    matches = store.matches(i, j)
    info: Dict[str, Any] = {
        "frame_a": store.frames[i]["name"],
        "frame_b": store.frames[j]["name"],
        "n_matches": len(matches),
    }
    if len(matches) < int(params["min_matches"]):
        info["status"] = "insufficient_matches"
        return None

    intrinsics = store.intrinsics(i)
    pts_a = np.float32([keypoints_a[a].pt for a, _ in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([keypoints_b[b].pt for _, b in matches]).reshape(-1, 1, 2)

    pose = estimate_relative_pose(
        pts_a, pts_b, intrinsics, params["ransac_thresh_px"], int(params["min_inliers"])
    )
    if pose is None:
        info["status"] = "pose_estimation_failed"
        return None

    mask = pose["inlier_mask"].ravel().astype(bool)
    points, stats, keep = triangulate_and_filter(
        pts_a[mask],
        pts_b[mask],
        intrinsics,
        pose["R"],
        pose["t"],
        float(params["max_reproj_error_px"]),
        float(params["min_parallax_deg"]),
        return_keep_mask=True,
    )
    info.update(stats)
    info["n_inliers"] = pose["n_inliers"]
    if points.shape[0] < int(params["min_good_points"]):
        info["status"] = "insufficient_parallax"
        return None

    # Map surviving points back to original keypoint indices.
    inlier_positions = np.flatnonzero(mask)
    kept_positions = np.flatnonzero(keep)
    full_positions = inlier_positions[kept_positions]
    info["idx_a"] = [int(matches[p][0]) for p in full_positions]
    info["idx_b"] = [int(matches[p][1]) for p in full_positions]
    info["status"] = "success"
    info["R"] = pose["R"]
    info["t"] = pose["t"]
    info["points"] = points
    return info


# --------------------------------------------------------------------------
# Camera registration (PnP against the growing world cloud)
# --------------------------------------------------------------------------

def gather_correspondences(
    store: FrameStore,
    frame_index: int,
    observations: Dict[int, Dict[int, int]],
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect 3D-2D pairs by matching ``frame_index`` to registered frames."""
    keypoints, _ = store.features(frame_index)
    object_points: List[np.ndarray] = []
    image_points: List[Tuple[float, float]] = []

    for other_index, obs in observations.items():
        if other_index == frame_index:
            continue
        for a_in_frame, b_in_other in store.matches(frame_index, other_index):
            point_id = obs.get(b_in_other)
            if point_id is None:
                continue
            if point_id >= points.shape[0]:
                continue
            object_points.append(points[point_id])
            image_points.append(keypoints[a_in_frame].pt)

    if not object_points:
        return np.zeros((0, 3), np.float64), np.zeros((0, 2), np.float64)
    return (
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
    )


def register_camera(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: np.ndarray,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """solvePnPRansac -> world pose of a camera, with explicit validation."""
    if object_points.shape[0] < int(params["min_pnp_points"]):
        return None

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            intrinsics,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=200,
            reprojectionError=float(params["pnp_reproj_error_px"]),
            confidence=0.999,
        )
    except cv2.error:
        return None

    if not ok or rvec is None or tvec is None or inliers is None:
        return None

    n_inliers = int(len(inliers))
    if n_inliers < int(params["min_pnp_inliers"]):
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    # Reprojection must be measured in *pixels* (the image_points convention),
    # so project through the 3x4 matrix [K | [R|t]] rather than the bare
    # extrinsic, which would yield normalized (division-free) image coordinates.
    projection = intrinsics @ np.hstack([rotation, translation])          # (3, 4)
    homogeneous = np.hstack(
        [object_points, np.ones((object_points.shape[0], 1))]
    ).T                                                               # (4, N)
    projected = projection @ homogeneous                                # (3, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (projected[:2] / projected[2]).T                          # (N, 2)
    errors = np.linalg.norm(uv - image_points, axis=1)
    inlier_mask = np.zeros(errors.shape[0], dtype=bool)
    inlier_mask[inliers.ravel()] = True
    valid = inlier_mask & np.isfinite(errors)
    if not valid.any():
        return None
    median_error = float(np.median(errors[valid]))
    if median_error > float(params["pnp_reproj_error_px"]):
        return None

    center = (-rotation.T @ translation).ravel()
    if not np.all(np.isfinite(center)):
        return None

    return {
        "rotation": rotation,
        "translation": translation,
        "center": center,
        "n_correspondences": int(object_points.shape[0]),
        "n_inliers": n_inliers,
        "median_reproj_error_px": round(median_error, 4),
        "method": "solvePnPRansac(EPNP)",
    }

# --------------------------------------------------------------------------
# Global triangulation between two already-registered cameras
# --------------------------------------------------------------------------

def triangulate_global_pair(
    store: FrameStore,
    i: int,
    j: int,
    cameras: Dict[int, Dict[str, Any]],
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Triangulate NEW world points between registered cameras i and j."""
    keypoints_i, _ = store.features(i)
    keypoints_j, _ = store.features(j)
    matches = store.matches(i, j)
    if len(matches) < int(params["min_matches"]):
        return None

    intrinsics_i = store.intrinsics(i)
    intrinsics_j = store.intrinsics(j)
    proj_i = intrinsics_i @ np.hstack([cameras[i]["rotation"], cameras[i]["translation"]])
    proj_j = intrinsics_j @ np.hstack([cameras[j]["rotation"], cameras[j]["translation"]])

    pts_i = np.ascontiguousarray(
        np.float64([keypoints_i[a].pt for a, _ in matches]).reshape(-1, 2).T
    )
    pts_j = np.ascontiguousarray(
        np.float64([keypoints_j[b].pt for _, b in matches]).reshape(-1, 2).T
    )

    homo = np.asarray(
        cv2.triangulatePoints(proj_i, proj_j, pts_i, pts_j), dtype=np.float64
    )
    if homo.ndim != 2:
        return None
    if homo.shape[0] != 4 and homo.shape[1] == 4:
        homo = homo.T
    if homo.shape[0] != 4:
        return None

    w = homo[3]
    valid_w = np.abs(w) > 1e-12
    homo = np.where(valid_w[None, :], homo / np.where(valid_w, w, 1.0)[None, :], 0.0)
    points = homo[:3].T                                     # (N, 3) world

    def depth_in(cam: Dict[str, Any], pts: np.ndarray) -> np.ndarray:
        return (cam["rotation"] @ pts.T + cam["translation"]).T[:, 2]

    in_front = (depth_in(cameras[i], points) > 0) & (depth_in(cameras[j], points) > 0)

    def reproj(proj: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
        p = proj @ np.hstack([points, np.ones((points.shape[0], 1))]).T
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = (p[:2] / p[2][None, :]).T
        return np.linalg.norm(projected - pts2d.T, axis=1)

    flat_i = np.float64([keypoints_i[a].pt for a, _ in matches]).reshape(-1, 2)
    flat_j = np.float64([keypoints_j[b].pt for _, b in matches]).reshape(-1, 2)
    err = np.maximum(reproj(proj_i, pts_i), reproj(proj_j, pts_j))
    keep = in_front & np.isfinite(err) & (err < float(params["max_reproj_error_px"]))
    if not keep.any():
        return None

    new_points = points[keep]
    kept_positions = np.flatnonzero(keep)
    return {
        "points": new_points,
        "idx_i": [int(matches[p][0]) for p in kept_positions],
        "idx_j": [int(matches[p][1]) for p in kept_positions],
        "n_matches": len(matches),
        "median_reproj_error_px": round(float(np.median(err[keep])), 4),
    }


def compose_pose(
    parent: Dict[str, Any],
    child: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two ``world -> camera`` poses into the ``parent -> child`` pose.

    With ``X_cam = R X_world + t``, the relative transform satisfies
    ``X_child = R_rel X_parent + t_rel`` where
    ``R_rel = R_child R_parent^T`` and ``t_rel = t_child - R_rel t_parent``.
    """
    r_parent = np.asarray(parent["rotation"], dtype=np.float64)
    t_parent = np.asarray(parent["translation"], dtype=np.float64)
    r_child = np.asarray(child["rotation"], dtype=np.float64)
    t_child = np.asarray(child["translation"], dtype=np.float64)
    r_rel = r_child @ r_parent.T
    t_rel = t_child - (r_rel @ t_parent)
    return np.asarray(r_rel, dtype=np.float64), np.asarray(t_rel, dtype=np.float64)

# --------------------------------------------------------------------------
# Incremental global reconstruction driver
# --------------------------------------------------------------------------

def reconstruct_global(
    job_id: str,
    data_dir: Union[str, Path] = "data",
    parameters: Optional[Dict[str, Any]] = None,
    write_artifacts: bool = True,
) -> GlobalReconstructionResult:
    """Build ONE coherent sparse cloud for a job from its existing frames."""
    params = {**DEFAULT_GLOBAL_PARAMETERS, **(parameters or {})}
    data_path = _resolve_data_dir(data_dir)
    job_dir = data_path / job_id
    frames_dir = job_dir / "frames"
    out_dir = job_dir / "reconstruction"

    frames = load_frames(frames_dir, int(params["max_image_side"]))
    base_stats: Dict[str, Any] = {
        "job_id": job_id,
        "frames_dir": str(frames_dir),
        "parameters": params,
        "n_frames_found": len(frames),
        "frame_size": [frames[0]["width"], frames[0]["height"]] if frames else None,
        "downscale": frames[0]["scale"] if frames else None,
    }

    if not frames:
        result = GlobalReconstructionResult(
            STATUS_NO_FRAMES, GLOBAL_FAILURE_MESSAGES[STATUS_NO_FRAMES]
        )
        result.stats = base_stats
        _finalize(result, out_dir, base_stats, write_artifacts)
        return result
    if len(frames) < 2:
        result = GlobalReconstructionResult(
            STATUS_INSUFFICIENT_FRAMES,
            GLOBAL_FAILURE_MESSAGES[STATUS_INSUFFICIENT_FRAMES],
        )
        result.stats = base_stats
        _finalize(result, out_dir, base_stats, write_artifacts)
        return result

    store = FrameStore(frames, params)
    indices = select_frame_indices(
        len(frames), params["registration_stride"], int(params["max_frames"])
    )

    # ---- 1. Seed the world frame -------------------------------------------
    seed: Optional[Dict[str, Any]] = None
    seed_pair_indices: Optional[Tuple[int, int]] = None
    seed_attempts: List[Dict[str, Any]] = []
    seed_stride = max(1, int(params["seed_stride"]))
    for a_pos in range(0, len(indices) - 1, seed_stride):
        a_idx, b_idx = indices[a_pos], indices[a_pos + 1]
        attempt = seed_pair(store, a_idx, b_idx, params)
        seed_attempts.append(
            {
                k: v
                for k, v in (attempt or {}).items()
                if k not in ("points", "idx_a", "idx_b", "R", "t")
            }
        )
        if attempt is not None:
            seed, seed_pair_indices = attempt, (a_idx, b_idx)
            break

    if seed is None or seed_pair_indices is None:
        result = GlobalReconstructionResult(
            STATUS_GLOBAL_NO_SEED, GLOBAL_FAILURE_MESSAGES[STATUS_GLOBAL_NO_SEED]
        )
        result.stats = dict(base_stats, seed_attempts=seed_attempts)
        _finalize(result, out_dir, base_stats, write_artifacts)
        return result

    a, b = seed_pair_indices
    points = np.asarray(seed["points"], dtype=np.float64)
    cameras: Dict[int, Dict[str, Any]] = {
        a: {
            "rotation": np.eye(3, dtype=np.float64),
            "translation": np.zeros((3, 1), dtype=np.float64),
            "center": np.zeros(3),
            "method": "seed_identity",
            "n_observations": len(seed["idx_a"]),
        },
        b: {
            "rotation": np.asarray(seed["R"], dtype=np.float64),
            "translation": np.asarray(seed["t"], dtype=np.float64),
            "center": (-np.asarray(seed["R"]).T @ np.asarray(seed["t"])).ravel(),
            "method": "seed_recoverPose",
            "n_observations": len(seed["idx_b"]),
        },
    }
    observations: Dict[int, Dict[int, int]] = {
        a: {kp: idx for idx, kp in enumerate(seed["idx_a"])},
        b: {kp: idx for idx, kp in enumerate(seed["idx_b"])},
    }
    registered_order = [a, b]
    frames_failed: List[Dict[str, str]] = []
    registration_log: List[Dict[str, Any]] = []
    n_new_from_triangulation = 0

    # ---- 2. Incrementally register remaining frames -------------------------
    for frame_index in indices:
        if frame_index in cameras:
            continue
        name = frames[frame_index]["name"]
        object_points, image_points = gather_correspondences(
            store, frame_index, observations, points
        )
        pose = register_camera(
            object_points, image_points, store.intrinsics(frame_index), params
        )
        if pose is None:
            n_corr = int(object_points.shape[0])
            planarity = planarity_report(points)
            reason = (
                f"only {n_corr} 3D-2D correspondences "
                f"(need {int(params['min_pnp_points'])})"
                if n_corr < int(params["min_pnp_points"])
                else (
                    "PnP degenerate: scene is near-planar "
                    f"(3rd singular value ratio {planarity['third_axis_ratio']}), "
                    "so absolute pose is not recoverable from this bootstrap"
                    if planarity["planar"]
                    else "PnP RANSAC failed or inlier/reprojection validation failed"
                )
            )
            frames_failed.append(
                {"frame": name, "status": "registration_failed", "reason": reason}
            )
            registration_log.append(
                {
                    "frame": name,
                    "status": "failed",
                    "n_correspondences": n_corr,
                    "reason": reason,
                }
            )
            continue

        cameras[frame_index] = {
            "rotation": pose["rotation"],
            "translation": pose["translation"],
            "center": pose["center"],
            "method": pose["method"],
            "n_correspondences": pose["n_correspondences"],
            "n_inliers": pose["n_inliers"],
            "median_reproj_error_px": pose["median_reproj_error_px"],
            "n_observations": 0,
        }
        registered_order.append(frame_index)
        registration_log.append(
            {
                "frame": name,
                "status": "registered",
                "n_correspondences": pose["n_correspondences"],
                "n_inliers": pose["n_inliers"],
                "median_reproj_error_px": pose["median_reproj_error_px"],
            }
        )

        # Observations of already-known 3D points, seen in this new frame.
        new_obs: Dict[int, int] = {}
        for other_index, obs in observations.items():
            for a_in_new, b_in_other in store.matches(frame_index, other_index):
                point_id = obs.get(b_in_other)
                if point_id is not None and a_in_new not in new_obs:
                    new_obs[a_in_new] = point_id

        # Triangulate NEW points against recently registered neighbours.
        partners = [
            other for other in reversed(registered_order) if other != frame_index
        ][: int(params["max_triangulation_partners"])]
        for partner in partners:
            addition = triangulate_global_pair(
                store, frame_index, partner, cameras, params
            )
            if addition is None:
                continue
            new_points = addition["points"]
            for offset, kp_new in enumerate(addition["idx_i"]):
                if kp_new in new_obs:
                    continue
                if points.shape[0] >= int(params["max_points"]):
                    break
                pid = int(points.shape[0])
                new_obs[kp_new] = pid
                points = np.vstack([points, new_points[offset]])
                observations[partner][addition["idx_j"][offset]] = pid
                n_new_from_triangulation += 1

        observations[frame_index] = new_obs
        cameras[frame_index]["n_observations"] = len(new_obs)

    # ---- 3. Assemble the global result -------------------------------------
    frames_registered = [frames[i]["name"] for i in registered_order]
    cameras_out: Dict[str, Dict[str, Any]] = {
        frames[i]["name"]: {
            "rotation": np.asarray(cameras[i]["rotation"]).tolist(),
            "translation": np.asarray(cameras[i]["translation"]).ravel().tolist(),
            "center": np.asarray(cameras[i]["center"], dtype=float).ravel().tolist(),
            "method": cameras[i].get("method"),
            "n_observations": cameras[i].get("n_observations"),
            "n_inliers": cameras[i].get("n_inliers"),
            "median_reproj_error_px": cameras[i].get("median_reproj_error_px"),
        }
        for i in registered_order
    }

    only_seed = len(registered_order) == 2
    scene_planarity = planarity_report(points)
    if only_seed:
        status = STATUS_GLOBAL_NO_REGISTRATION
        message = (
            GLOBAL_FAILURE_MESSAGES[STATUS_GLOBAL_NO_REGISTRATION]
            + (
                " Scene is near-planar, so two-view bootstrap leaves an absolute "
                "scale/distance ambiguity and PnP registration is degenerate."
                if scene_planarity["planar"]
                else ""
            )
        )
    else:
        status = STATUS_SUCCESS if points.shape[0] > 0 else STATUS_GLOBAL_NO_REGISTRATION
        message = (
            "Global sparse reconstruction produced in one world frame."
            if status == STATUS_SUCCESS
            else GLOBAL_FAILURE_MESSAGES[STATUS_GLOBAL_NO_REGISTRATION]
        )

    result = GlobalReconstructionResult(
        status=status,
        message=message,
        points=points,
        cameras=cameras_out,
        frames_registered=frames_registered,
        frames_failed=frames_failed,
        stats=dict(
            base_stats,
            seed_pair=[frames[a]["name"], frames[b]["name"]],
            seed_n_points=int(len(seed["idx_a"])),
            n_frames_selected=len(indices),
            n_frames_registered=len(registered_order),
            n_frames_failed=len(frames_failed),
            n_new_points_from_triangulation=int(n_new_from_triangulation),
            scene_planarity=scene_planarity,
            registration_log=registration_log,
            seed_attempts=seed_attempts[:5],
        ),
    )
    _finalize(result, out_dir, base_stats, write_artifacts)
    return result

# --------------------------------------------------------------------------
# Artifacts + CLI
# --------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _finalize(
    result: GlobalReconstructionResult,
    out_dir: Path,
    manifest_base: Dict[str, Any],
    write_artifacts: bool,
) -> None:
    """Always write the manifest; cloud/cameras/trajectory only for a global success.

    A failed global run may still hold seed-pair points, but those are *not* a
    global cloud, so they are never written under the ``sparse_points_global``
    name that downstream consumers treat as a valid reconstruction.
    """
    manifest = result.to_manifest()
    manifest.update(manifest_base)
    manifest["stats"] = result.stats

    if not write_artifacts:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconstruction_global.json").write_text(
        json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8"
    )

    if not result.ok:
        return

    (out_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "scale_status": "UP_TO_SCALE_NO_METRIC_UNITS",
                "georeferenced": False,
                "note": "Camera centres are in the seed-pair world frame. Not GPS.",
                "frames": result.trajectory(),
            },
            indent=2,
            default=_json_safe,
        ),
        encoding="utf-8",
    )
    (out_dir / "cameras_global.json").write_text(
        json.dumps(result.cameras, indent=2, default=_json_safe), encoding="utf-8"
    )

    if result.n_points > 0:
        np.savez_compressed(
            out_dir / "sparse_points_global.npz",
            points=result.points.astype(np.float64),
        )
        write_ply(result.points, out_dir / "sparse_points_global.ply")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8.2 - global sparse SfM from extracted frames"
    )
    parser.add_argument("--job-id", required=True, help="Job ID to reconstruct")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--stride", type=int, default=None, help="Frame selection stride")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames")
    parser.add_argument(
        "--no-artifacts", action="store_true", help="Do not write output files"
    )
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.stride is not None:
        overrides["registration_stride"] = args.stride
    if args.max_frames is not None:
        overrides["max_frames"] = args.max_frames

    result = reconstruct_global(
        args.job_id,
        args.data_dir,
        parameters=overrides or None,
        write_artifacts=not args.no_artifacts,
    )
    summary = result.to_manifest()
    summary.pop("cameras", None)
    print(json.dumps(summary, indent=2, default=_json_safe))
    print(
        f"STATUS={result.status} POINTS={result.n_points} "
        f"CAMERAS={len(result.frames_registered)}"
    )


if __name__ == "__main__":
    main()







