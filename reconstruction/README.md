# 3D Reconstruction & Photogrammetry Module (Placeholder)

This directory is designated for the 3D reconstruction and georeferencing pipeline.

### Planned Integrations:
1. **Frame Extraction**: Efficient keyframe sampling via OpenCV with motion-blur filtering.
2. **Feature Extraction & Matching**: SIFT / SuperPoint / LightGlue for robust sparse correspondences across aerial passes.
3. **Structure-from-Motion (SfM)**: Sparse bundle adjustment and camera pose trajectory calculation via COLMAP.
4. **Dense Reconstruction & Meshing**: Open3D Poisson surface reconstruction and point cloud densification.
5. **Georeferencing**: Alignment with GPS/IMU telemetry logs to project local 3D coordinates into UTM/WGS84.
