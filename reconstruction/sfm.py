"""Phase 8.1 - real Structure-from-Motion foundation using OpenCV primitives.

Pipeline (per image pair):

    grayscale + downscale
        -> SIFT feature detection
        -> BFMatcher(L2) + Lowe ratio + cross-check filtering
        -> essential matrix (USAC_MAGSAC RANSAC)
        -> recoverPose  -> relative rotation R and translation direction t
        -> triangulatePoints + cheirality check + reprojection filtering
        -> sparse 3D points (in the first camera's frame)

HONESTY NOTES
-------------
* Monocular SfM yields an **up-to-scale** reconstruction. Absolute metres are
  NOT produced (that needs calibrated intrinsics and a metric reference).
* Camera intrinsics are unknown for drone footage, so a standard heuristic
  focal length (``f = focal_ratio * max(width, height)``) is used and recorded
  in the output manifest.
* Each image pair is reconstructed in its own first-camera coordinate frame.
  Global alignment / bundle adjustment is NOT implemented in 8.1, so the cloud
  is a union of independent, up-to-scale local clouds (each point is tagged
  with its pair index). This is stated explicitly in the manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Status vocabulary - the module never invents a result
# --------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_NO_FRAMES = "no_frames"
STATUS_INSUFFICIENT_FRAMES = "insufficient_frames"
STATUS_INSUFFICIENT_FEATURES = "insufficient_features"
STATUS_INSUFFICIENT_MATCHES = "insufficient_matches"
STATUS_POSE_ESTIMATION_FAILED = "pose_estimation_failed"
STATUS_INSUFFICIENT_PARALLAX = "insufficient_parallax"

FAILURE_STATUSES = frozenset(
    {
        STATUS_NO_FRAMES,
        STATUS_INSUFFICIENT_FRAMES,
        STATUS_INSUFFICIENT_FEATURES,
        STATUS_INSUFFICIENT_MATCHES,
        STATUS_POSE_ESTIMATION_FAILED,
        STATUS_INSUFFICIENT_PARALLAX,
    }
)

STATUS_MESSAGES = {
    STATUS_NO_FRAMES: "No readable frames were found in the frames directory.",
    STATUS_INSUFFICIENT_FRAMES: "At least two frames are required for Structure-from-Motion.",
    STATUS_INSUFFICIENT_FEATURES: "Too few SIFT features were detected in the frame pair.",
    STATUS_INSUFFICIENT_MATCHES: "Too few reliable feature matches survived filtering.",
    STATUS_POSE_ESTIMATION_FAILED: "Essential-matrix / relative-pose estimation failed.",
    STATUS_INSUFFICIENT_PARALLAX: "Triangulated points failed cheirality, reprojection or parallax checks.",
    STATUS_SUCCESS: "Sparse reconstruction produced from real image geometry.",
}

DEFAULT_PARAMETERS: Dict[str, Any] = {
    "max_image_side": 1280,      # downscale 4K frames for tractable SIFT
    "focal_ratio": 1.2,          # f = focal_ratio * max(w, h) (no calibration available)
    "n_features": 4000,
    "ratio_test": 0.75,
    "min_matches": 30,
    "ransac_thresh_px": 1.5,
    "min_inliers": 15,
    "max_reproj_error_px": 2.0,
    "min_good_points": 50,
    "min_parallax_deg": 0.5,
    "pair_stride": 3,
    "max_pairs": 4,
}


@dataclass
class ReconstructionResult:
    """Outcome of a reconstruction attempt. ``ok`` is True only for success."""

    status: str
    message: str
    frames_used: List[str] = field(default_factory=list)
    pairs: List[Dict[str, Any]] = field(default_factory=list)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    point_pair_index: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "module": "cheel_nazar.reconstruction.sfm",
            "phase": "8.1",
            "status": self.status,
            "ok": self.ok,
            "message": self.message,
            "n_points": self.n_points,
            "frames_used": list(self.frames_used),
            "pairs": self.pairs,
            "stats": self.stats,
            "coordinate_note": (
                "Monocular SfM output is UP-TO-SCALE. Points are expressed in "
                "the first camera frame of each image pair; each pair is an "
                "independent local cloud (global alignment / bundle adjustment "
                "is not implemented in Phase 8.1). No metres are implied."
            ),
            "not_implemented": [
                "global_alignment_bundle_adjustment",
                "dense_reconstruction_mvs",
                "mesh_generation",
                "georeferencing_gps_wgs84",
            ],
        }


def _resolve_data_dir(data_dir: Union[str, Path]) -> Path:
    path = Path(data_dir)
    if path.exists():
        return path
    project_root = Path(__file__).resolve().parent.parent / path
    if project_root.exists():
        return project_root
    return path

# --------------------------------------------------------------------------
# Frame loading + intrinsics
# --------------------------------------------------------------------------

def load_frames(
    frames_dir: Union[str, Path],
    max_side: int = 1280,
) -> List[Dict[str, Any]]:
    """Load frames grayscale, downscaled so ``max(w, h) <= max_side``.

    Returns ``[{"name", "gray", "scale", "width", "height"}]`` in filename
    order, which is the extraction pipeline's temporal order.
    """
    directory = Path(frames_dir)
    if not directory.is_dir():
        return []

    names = sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )

    frames: List[Dict[str, Any]] = []
    for name in names:
        image = cv2.imread(str(directory / name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            continue
        height, width = image.shape[:2]
        scale = 1.0
        if max_side and max(height, width) > max_side:
            scale = max_side / float(max(height, width))
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        h, w = image.shape[:2]
        frames.append(
            {
                "name": name,
                "gray": image,
                "scale": float(scale),
                "width": int(w),
                "height": int(h),
            }
        )
    return frames


def camera_matrix(width: int, height: int, focal_ratio: float = 1.2) -> np.ndarray:
    """Pinhole intrinsics from the no-calibration heuristic ``f = r * max(w, h)``."""
    f = focal_ratio * float(max(width, height))
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# Features + matching
# --------------------------------------------------------------------------

def detect_features(
    gray: np.ndarray,
    n_features: int = 4000,
    detector: Optional[Any] = None,
) -> Tuple[List[Any], Optional[np.ndarray]]:
    """SIFT detection. Returns (keypoints, descriptors) - descriptors may be None."""
    sift = detector if detector is not None else cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return list(keypoints), descriptors


def match_features(
    descriptors1: Optional[np.ndarray],
    descriptors2: Optional[np.ndarray],
    ratio_test: float = 0.75,
    matcher: Optional[Any] = None,
) -> List[Any]:
    """Lowe ratio test plus a mutual cross-check, for robustness."""
    if descriptors1 is None or descriptors2 is None:
        return []
    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return []

    bf = matcher if matcher is not None else cv2.BFMatcher(cv2.NORM_L2)

    def ratio_pairs(desc_a: np.ndarray, desc_b: np.ndarray) -> Dict[int, Any]:
        pairs: Dict[int, Any] = {}
        for pair in bf.knnMatch(desc_a, desc_b, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < ratio_test * second.distance:
                pairs[best.queryIdx] = best
        return pairs

    forward = ratio_pairs(descriptors1, descriptors2)    # q1 (desc1) -> match
    backward = ratio_pairs(descriptors2, descriptors1)  # j  (desc2) -> match

    # Mutual / cross-checked matches: the desc2 keypoint matched by q1 must
    # choose q1 as ITS best desc1 partner.
    mutual: List[Any] = []
    for q1, m in forward.items():
        reverse = backward.get(m.trainIdx)
        if reverse is not None and reverse.trainIdx == q1:
            mutual.append(m)
    return mutual

# --------------------------------------------------------------------------
# Relative pose + triangulation
# --------------------------------------------------------------------------

def estimate_relative_pose(
    points1: np.ndarray,
    points2: np.ndarray,
    intrinsics: np.ndarray,
    ransac_thresh_px: float = 1.5,
    min_inliers: int = 15,
) -> Optional[Dict[str, Any]]:
    """Essential matrix + recoverPose. Returns None when the pose is not estimable."""
    if points1.shape[0] < max(8, min_inliers):
        return None

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    essential, inlier_mask = cv2.findEssentialMat(
        points1,
        points2,
        intrinsics,
        method=method,
        prob=0.999,
        threshold=float(ransac_thresh_px),
    )
    if essential is None or essential.size == 0:
        return None

    essential = np.asarray(essential, dtype=np.float64)
    if essential.shape != (3, 3):          # stacked solutions -> keep the first
        essential = essential[:3, :3]

    mask_in = inlier_mask.copy() if inlier_mask is not None else None
    n_good, rotation, translation, pose_mask = cv2.recoverPose(
        essential, points1, points2, intrinsics, mask=mask_in
    )
    if rotation is None or translation is None:
        return None

    n_inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else int(n_good)
    if n_inliers < min_inliers:
        return None

    return {
        "R": np.asarray(rotation, dtype=np.float64),
        "t": np.asarray(translation, dtype=np.float64).reshape(3, 1),
        "inlier_mask": pose_mask,
        "n_inliers": n_inliers,
        "essential": essential,
    }


def _tri_failed(return_keep_mask: bool, n_triangulated: int):
    """Uniform empty-result helper for triangulate_and_filter early exits."""
    points = np.zeros((0, 3), np.float64)
    stats = {"n_triangulated": int(n_triangulated), "n_kept": 0}
    if return_keep_mask:
        return points, stats, np.zeros((0,), dtype=bool)
    return points, stats


def triangulate_and_filter(
    points1: np.ndarray,
    points2: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_reproj_error_px: float = 2.0,
    min_parallax_deg: float = 0.5,
    return_keep_mask: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Triangulate and keep points with positive depth and low reprojection error.

    With ``return_keep_mask=True`` returns ``(points, stats, keep_mask)`` so the
    global (Phase 8.2) aligner can map surviving points back to the input
    correspondences. Default behaviour is unchanged.
    """
    # OpenCV 5.x needs CV_64F (2, N)-style points here; float32 or a transposed
    # (N, 1, 2) array makes it compute a bogus allocation size.
    pts1_2d = np.ascontiguousarray(points1, dtype=np.float64).reshape(-1, 2)
    pts2_2d = np.ascontiguousarray(points2, dtype=np.float64).reshape(-1, 2)

    # OpenCV 5.x requires CV_64F points in (2, N) layout here. Passing (N, 2)
    # silently yields a (4, 2) result (one bogus "point" per coordinate), and
    # float32 makes it compute a negative allocation size.
    pts1_t = np.ascontiguousarray(pts1_2d.T, dtype=np.float64)   # (2, N)
    pts2_t = np.ascontiguousarray(pts2_2d.T, dtype=np.float64)   # (2, N)

    zeros = np.zeros((3, 1), dtype=np.float64)
    proj1 = intrinsics @ np.hstack([np.eye(3), zeros])
    proj2 = intrinsics @ np.hstack([rotation, translation])

    homo = np.asarray(cv2.triangulatePoints(proj1, proj2, pts1_t, pts2_t), dtype=np.float64)
    if homo.ndim != 2:
        return _tri_failed(return_keep_mask, 0)
    # Depending on the OpenCV build the homogeneous result comes back as
    # (4, N) or (N, 4); normalise to (4, N) before dehomogenising.
    if homo.shape[0] != 4 and homo.shape[1] == 4:
        homo = homo.T
    if homo.shape[0] != 4:
        return _tri_failed(return_keep_mask, 0)

    w = homo[3]
    valid_w = np.abs(w) > 1e-12
    homo = np.where(valid_w[None, :], homo / np.where(valid_w, w, 1.0)[None, :], 0.0)
    points3d = homo[:3].T.astype(np.float64)   # (N, 3)

    # Cheirality: must be in front of BOTH cameras.
    depth1 = points3d[:, 2]
    cam2 = (rotation @ points3d.T + translation).T
    depth2 = cam2[:, 2]
    in_front = (depth1 > 0) & (depth2 > 0)

    # Reprojection error in both views.
    def reproj(proj: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
        p = proj @ np.hstack([points3d, np.ones((points3d.shape[0], 1))]).T   # (3, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            # p[:2] is (2, N) and p[2] is (N,); the [None, :] keeps the
            # denominator per-column instead of broadcasting across rows.
            projected = (p[:2] / p[2][None, :]).T                              # (N, 2)
        return np.linalg.norm(projected - pts2d, axis=1)

    err = np.maximum(reproj(proj1, pts1_2d), reproj(proj2, pts2_2d))
    low_error = np.isfinite(err) & (err < float(max_reproj_error_px))

    # Parallax angle between the two viewing rays (per point).
    ray1 = points3d.copy()
    ray2 = (rotation @ points3d.T + translation).T
    n1 = np.linalg.norm(ray1, axis=1)
    n2 = np.linalg.norm(ray2, axis=1)
    denom = np.maximum(n1 * n2, 1e-12)
    cos = np.clip((ray1 * ray2).sum(axis=1) / denom, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))

    keep = in_front & low_error
    kept_angles = angles[keep]
    median_angle = float(np.median(kept_angles)) if kept_angles.size else 0.0

    stats = {
        "n_matched": int(points1.shape[0]),
        "n_triangulated": int(points3d.shape[0]),
        "n_in_front": int(np.count_nonzero(in_front)),
        "n_low_reproj_error": int(np.count_nonzero(low_error)),
        "n_kept": int(np.count_nonzero(keep)),
        "median_reproj_error_px": round(float(np.median(err[keep])), 4) if keep.any() else None,
        "median_parallax_deg": round(median_angle, 4),
    }
    if return_keep_mask:
        return points3d[keep], stats, keep
    return points3d[keep], stats

# --------------------------------------------------------------------------
# Pair reconstruction
# --------------------------------------------------------------------------

def reconstruct_pair(
    frame1: Dict[str, Any],
    frame2: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reconstruct one image pair. Returns a dict with ``status`` and ``points``."""
    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    pair_info: Dict[str, Any] = {
        "frame_a": frame1["name"],
        "frame_b": frame2["name"],
        "width": frame1["width"],
        "height": frame1["height"],
        "intrinsics_note": "heuristic f = focal_ratio * max(w, h); no calibration available",
        "points": np.zeros((0, 3)),
    }

    if frame1["width"] != frame2["width"] or frame1["height"] != frame2["height"]:
        pair_info["status"] = STATUS_POSE_ESTIMATION_FAILED
        return pair_info

    intrinsics = camera_matrix(frame1["width"], frame1["height"], params["focal_ratio"])
    detector = cv2.SIFT_create(nfeatures=int(params["n_features"]))

    keypoints1, descriptors1 = detect_features(frame1["gray"], detector=detector)
    keypoints2, descriptors2 = detect_features(frame2["gray"], detector=detector)
    pair_info["n_keypoints_a"] = len(keypoints1)
    pair_info["n_keypoints_b"] = len(keypoints2)

    if len(keypoints1) < 8 or len(keypoints2) < 8:
        pair_info["status"] = STATUS_INSUFFICIENT_FEATURES
        return pair_info

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = match_features(descriptors1, descriptors2, params["ratio_test"], matcher)
    pair_info["n_matches"] = len(matches)

    if len(matches) < int(params["min_matches"]):
        pair_info["status"] = STATUS_INSUFFICIENT_MATCHES
        return pair_info

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    pose = estimate_relative_pose(
        pts1, pts2, intrinsics, params["ransac_thresh_px"], int(params["min_inliers"])
    )
    if pose is None:
        pair_info["status"] = STATUS_POSE_ESTIMATION_FAILED
        return pair_info

    pair_info["n_inliers"] = pose["n_inliers"]
    pair_info["rotation"] = pose["R"]
    pair_info["translation"] = pose["t"]
    pair_info["translation_note"] = "unit translation direction (scale unknown)"

    mask = pose["inlier_mask"].ravel().astype(bool)
    points3d, stats = triangulate_and_filter(
        pts1[mask],
        pts2[mask],
        intrinsics,
        pose["R"],
        pose["t"],
        float(params["max_reproj_error_px"]),
        float(params["min_parallax_deg"]),
    )
    pair_info.update(stats)

    if points3d.shape[0] < int(params["min_good_points"]) or (
        stats["median_parallax_deg"] < float(params["min_parallax_deg"])
    ):
        pair_info["status"] = STATUS_INSUFFICIENT_PARALLAX
        pair_info["points"] = points3d
        return pair_info

    pair_info["status"] = STATUS_SUCCESS
    pair_info["points"] = points3d
    return pair_info


def _select_pairs(n_frames: int, stride: int, max_pairs: int) -> List[Tuple[int, int]]:
    stride = max(1, int(stride))
    candidates = [(i, i + stride) for i in range(0, n_frames - stride)]
    if not candidates:
        candidates = [(i, i + 1) for i in range(0, n_frames - 1)]
    if len(candidates) <= max_pairs:
        return candidates
    step = len(candidates) / float(max_pairs)
    return [candidates[int(i * step)] for i in range(max_pairs)]


def write_ply(points: np.ndarray, path: Union[str, Path]) -> Path:
    """Write an ASCII PLY point cloud (real reconstructed coordinates)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {int(points.shape[0])}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for x, y, z in points:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

# --------------------------------------------------------------------------
# Job-level orchestration
# --------------------------------------------------------------------------

def reconstruct_job(
    job_id: str,
    data_dir: Union[str, Path] = "data",
    parameters: Optional[Dict[str, Any]] = None,
    write_artifacts: bool = True,
) -> ReconstructionResult:
    """Run the SfM foundation over an existing job's extracted frames."""
    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    data_path = _resolve_data_dir(data_dir)
    job_dir = data_path / job_id
    frames_dir = job_dir / "frames"
    out_dir = job_dir / "reconstruction"

    frames = load_frames(frames_dir, int(params["max_image_side"]))
    manifest_base = {
        "job_id": job_id,
        "frames_dir": str(frames_dir),
        "parameters": params,
        "n_frames_found": len(frames),
        "frame_size": [frames[0]["width"], frames[0]["height"]] if frames else None,
        "downscale": frames[0]["scale"] if frames else None,
    }

    if not frames:
        result = ReconstructionResult(STATUS_NO_FRAMES, STATUS_MESSAGES[STATUS_NO_FRAMES])
        result.stats = manifest_base
        _finalize(result, out_dir, manifest_base, write_artifacts)
        return result

    if len(frames) < 2:
        result = ReconstructionResult(
            STATUS_INSUFFICIENT_FRAMES, STATUS_MESSAGES[STATUS_INSUFFICIENT_FRAMES]
        )
        result.frames_used = [frames[0]["name"]]
        result.stats = manifest_base
        _finalize(result, out_dir, manifest_base, write_artifacts)
        return result

    pairs = _select_pairs(len(frames), params["pair_stride"], int(params["max_pairs"]))
    all_points: List[np.ndarray] = []
    pair_indices: List[np.ndarray] = []
    pair_reports: List[Dict[str, Any]] = []
    used_frames: List[str] = []
    best_failure = STATUS_INSUFFICIENT_PARALLAX

    for pair_number, (i, j) in enumerate(pairs):
        info = reconstruct_pair(frames[i], frames[j], params)
        points = np.asarray(info.pop("points", np.zeros((0, 3))), dtype=np.float64)
        info["pair_index"] = pair_number
        pair_reports.append(info)
        used_frames.extend([frames[i]["name"], frames[j]["name"]])

        if info["status"] == STATUS_SUCCESS and points.shape[0] > 0:
            all_points.append(points)
            pair_indices.append(np.full(points.shape[0], pair_number, dtype=np.int32))
        elif info["status"] in (
            STATUS_INSUFFICIENT_FEATURES,
            STATUS_INSUFFICIENT_MATCHES,
            STATUS_POSE_ESTIMATION_FAILED,
        ):
            best_failure = info["status"]

    if all_points:
        status = STATUS_SUCCESS
        message = STATUS_MESSAGES[STATUS_SUCCESS]
        points = np.vstack(all_points)
        pair_index = np.concatenate(pair_indices)
    else:
        status = best_failure
        message = STATUS_MESSAGES[status]
        points = np.zeros((0, 3), dtype=np.float64)
        pair_index = np.zeros((0,), dtype=np.int32)

    result = ReconstructionResult(
        status=status,
        message=message,
        frames_used=sorted(set(used_frames)),
        pairs=pair_reports,
        points=points,
        point_pair_index=pair_index,
    )
    result.stats = dict(
        manifest_base,
        pairs_attempted=len(pairs),
        pairs_succeeded=int(len(all_points)),
    )
    _finalize(result, out_dir, manifest_base, write_artifacts)
    return result

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
    result: ReconstructionResult,
    out_dir: Path,
    manifest_base: Dict[str, Any],
    write_artifacts: bool,
) -> None:
    """Write the manifest and, on success, the point cloud + camera artifacts.

    A manifest is always written (even for failures) so the outcome is auditable.
    """
    manifest = result.to_manifest()
    manifest.update(manifest_base)
    manifest["stats"] = result.stats

    if write_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reconstruction.json").write_text(
            json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8"
        )
        if result.ok and result.n_points > 0:
            np.savez_compressed(
                out_dir / "sparse_points.npz",
                points=result.points.astype(np.float64),
                pair_index=result.point_pair_index.astype(np.int32),
            )
            write_ply(result.points, out_dir / "sparse_points.ply")
            (out_dir / "cameras.json").write_text(
                json.dumps(
                    [
                        {
                            "frame_a": p.get("frame_a"),
                            "frame_b": p.get("frame_b"),
                            "rotation": p.get("rotation"),
                            "translation": p.get("translation"),
                            "n_inliers": p.get("n_inliers"),
                            "n_points": p.get("n_kept"),
                            "median_parallax_deg": p.get("median_parallax_deg"),
                            "status": p.get("status"),
                        }
                        for p in result.pairs
                    ],
                    indent=2,
                    default=_json_safe,
                ),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8.1 - sparse SfM reconstruction from extracted frames"
    )
    parser.add_argument("--job-id", required=True, help="Job ID to reconstruct")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--stride", type=int, default=None, help="Frame-pair stride")
    parser.add_argument("--max-pairs", type=int, default=None, help="Maximum frame pairs")
    parser.add_argument(
        "--no-artifacts", action="store_true", help="Do not write output files"
    )
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.stride is not None:
        overrides["pair_stride"] = args.stride
    if args.max_pairs is not None:
        overrides["max_pairs"] = args.max_pairs

    result = reconstruct_job(
        args.job_id,
        args.data_dir,
        parameters=overrides or None,
        write_artifacts=not args.no_artifacts,
    )
    print(json.dumps(result.to_manifest(), indent=2, default=_json_safe))
    print(f"STATUS={result.status} POINTS={result.n_points}")


if __name__ == "__main__":
    main()





