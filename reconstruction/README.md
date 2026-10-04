# 3D Reconstruction & Photogrammetry

Real 3D reconstruction from job footage, built entirely from OpenCV + NumPy
already used elsewhere in this project. **No COLMAP, no new dependencies.**

| Module | Phase | What it does |
|---|---|---|
| `sfm.py` | 8.1 | Per-pair Structure-from-Motion -> sparse cloud |
| `global_sfm.py` | 8.2 | Stitches every pair into ONE global cloud + camera trajectory |
| `dense_sfm.py` | 8.3/8.4 | Densifies it from Depth Anything maps, exports PLY/NPZ/GLB |

## Pipeline

```
frames -> SIFT -> ratio+cross-check matching -> essential matrix (RANSAC/MAGSAC)
       -> relative pose -> triangulation (cheirality + reprojection filtering)

seed pair -> world frame
remaining frames -> incremental solvePnPRansac(EPNP) -> global triangulation

per registered frame:
    project the global sparse cloud -> (u, v, true camera depth)
    sample that frame's Depth Anything map at those pixels
    fit disparity-percentile -> depth curve  (anchors relative depth to global scale)
    back-project the pixel grid, transform to world
multi-view support filter + voxel downsample
```

The percentile step matters: Depth Anything output is min-max normalised *per
frame*, so only the ordering of pixel values is meaningful. Ranking the map and
learning the rank -> depth curve against real sparse geometry is what puts the
dense cloud in the same world frame and at the same scale as Phase 8.2.

## Output (`data/<job_id>/reconstruction/`)

```
reconstruction_global.json     global sparse manifest (8.2)
sparse_points_global.ply/.npz  global sparse cloud
cameras_global.json            per-frame R, t, centre
trajectory.json                 ordered camera centres

dense_reconstruction.json      dense manifest (8.3)
dense_points_global.ply        binary PLY point cloud
dense_points_global.npz        float64 xyz
dense_points_global.glb        glTF 2.0 POINTS (Three.js GLTFLoader ready)
```

## Honesty guarantees

- `scale_status: UP_TO_SCALE_NO_METRIC_UNITS` - **no metres are implied or claimed**
- Intrinsics are a **heuristic** (`f = focal_ratio * max(width, height)`), not calibrated
- **Not georeferenced** - camera centres are local, never GPS/UTM/WGS84
- Depth is **monocular relative** depth, not metric
- Frames that cannot be calibrated are skipped and recorded in `frames_skipped`
- A failed run writes a manifest describing the failure and **no** cloud, so a
  partial result can never be mistaken for a successful reconstruction

## Not implemented

- Textured mesh / Poisson surface reconstruction
- Loop closure and global bundle adjustment
- Georeferencing to GPS/IMU (deliberately out of scope)
