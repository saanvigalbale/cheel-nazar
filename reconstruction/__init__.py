"""
Cheel Nazar - 3D Reconstruction & Photogrammetry Module.

Phase 8.1 provides a REAL Structure-from-Motion foundation built on OpenCV
primitives (no COLMAP, no new dependencies):

    frames -> SIFT features -> ratio/cross-checked matching
           -> essential matrix (RANSAC/MAGSAC) -> relative camera pose
           -> triangulation (cheirality + reprojection filtering)
           -> sparse 3D point cloud

Phase 8.2 stitches every pair into ONE coherent world cloud:

    seed pair -> world frame
    incremental solvePnPRansac(EPNP) registration
    -> global triangulation -> single sparse cloud + camera trajectory

Phase 8.3/8.4 densifies that cloud using the existing Depth Anything V2 maps,
anchoring each frame's relative depth to the global scale, and exports a
browser-ready point cloud:

    dense_points_global.ply  (binary PLY)
    dense_points_global.npz
    dense_points_global.glb  (glTF 2.0 POINTS, Three.js-ready)

Every output is derived from the actual imagery, and every manifest states the
scale status honestly. Nothing is georeferenced or metric.

Deliberately out of scope:
  * textured mesh generation
  * loop closure / global bundle adjustment
  * georeferencing to GPS/IMU/WGS84
"""

from .sfm import (  # noqa: F401
    FAILURE_STATUSES,
    ReconstructionResult,
    STATUS_INSUFFICIENT_FEATURES,
    STATUS_INSUFFICIENT_FRAMES,
    STATUS_INSUFFICIENT_MATCHES,
    STATUS_INSUFFICIENT_PARALLAX,
    STATUS_NO_FRAMES,
    STATUS_POSE_ESTIMATION_FAILED,
    STATUS_SUCCESS,
    reconstruct_job,
    reconstruct_pair,
)
from .global_sfm import (  # noqa: F401
    GlobalReconstructionResult,
    reconstruct_global,
)
from .dense_sfm import (  # noqa: F401
    DEFAULT_DENSE_PARAMETERS,
    DENSE_FAILURE_MESSAGES,
    DenseReconstructionResult,
    reconstruct_dense,
    write_binary_ply,
    write_glb_points,
)

__version__ = "0.4.0"
