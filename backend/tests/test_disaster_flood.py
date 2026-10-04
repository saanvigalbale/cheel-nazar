"""Phase 6B.1 Step 2 - synthetic-fixture tests for the FloodAnalyzer.

These tests exercise the REAL ``ai.disaster`` implementation against tiny
synthetic jobs written into pytest's ``tmp_path``. They load no models, use no
network, and depend only on numpy + Pillow.

Key point: every fixture writes a ``class_id_map`` into
``segmentation/summary.json``, which is the preferred resolution path in
``ai.disaster.base.resolve_class_ids`` - so the real checkpoint configuration
is never consulted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.disaster import ArtifactBundle  # noqa: E402
from ai.disaster.flood import FloodAnalyzer  # noqa: E402

# Tiny synthetic class vocabulary (ids chosen to mirror real ADE20K numbering).
BG = 0
ROAD = 2
WATER = 3

CLASS_ID_MAP = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "car",
    5: "person",
}


def _make_mask(water_px: int = 0, road_px: int = 0, size: int = 100) -> np.ndarray:
    """Mask with the exact given pixel counts, laid out row-major from index 0."""
    total = size * size
    assert 0 <= water_px + road_px <= total
    flat = np.zeros(total, dtype=np.uint8)
    flat[:water_px] = WATER
    flat[water_px : water_px + road_px] = ROAD
    return flat.reshape(size, size)


def _block_mask(size: int, blocks) -> np.ndarray:
    """Mask built from explicit ``(value, y0, y1, x0, x1)`` blocks."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for value, y0, y1, x0, x1 in blocks:
        mask[y0:y1, x0:x1] = value
    return mask


def _det(frame: str, label: str, bbox, confidence: float = 0.5) -> dict:
    return {
        "frame": frame,
        "label": label,
        "confidence": confidence,
        "bbox": [float(v) for v in bbox],
    }


def _write_job(tmp_path, frames: dict, detections=None, job_id: str = "job-test"):
    """Materialise a synthetic job and return (data_dir, job_id)."""
    data_dir = Path(tmp_path) / "data"
    job = data_dir / job_id
    seg = job / "analysis" / "segmentation"
    depth = job / "analysis" / "depth"
    seg.mkdir(parents=True, exist_ok=True)
    depth.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    for frame_name, spec in frames.items():
        Image.fromarray(spec["mask"]).save(seg / f"{Path(frame_name).stem}_mask.png")
        if spec.get("depth") is not None:
            Image.fromarray(np.asarray(spec["depth"], dtype=np.uint8)).save(
                depth / frame_name
            )
        summary_frames.append({"frame": frame_name, "coverage": {}, "categories": {}})

    (job / "analysis" / "detections.json").write_text(
        json.dumps(detections if detections is not None else []), encoding="utf-8"
    )
    (seg / "summary.json").write_text(
        json.dumps(
            {
                "model": "synthetic/segformer-stub",
                "frames_processed": len(summary_frames),
                "class_id_map": CLASS_ID_MAP,
                "categories": {},
                "frames": summary_frames,
            }
        ),
        encoding="utf-8",
    )
    return data_dir, job_id


def _analyze(tmp_path, frames: dict, detections=None, parameters=None, job_id="job-test"):
    data_dir, jid = _write_job(tmp_path, frames, detections, job_id)
    bundle = ArtifactBundle(jid, data_dir)
    return FloodAnalyzer(parameters=parameters).analyze(bundle)


def _only(result: dict) -> dict:
    """Return the single per-frame record (test helper)."""
    frames = result["frames"]
    assert len(frames) == 1, f"expected 1 frame record, got {len(frames)}"
    return frames[0]

# ---------------------------------------------------------------------------
# 1. Water percentage
# ---------------------------------------------------------------------------

def test_water_percentage_is_exact(tmp_path):
    # 250 water pixels out of 100x100 = 10_000 px  ->  2.5 %
    frame = _make_mask(water_px=250, size=100)
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    assert record["water_pixels"] == 250
    assert abs(record["water_coverage_pct"] - 2.5) < 1e-4
    assert abs(result["aggregate"]["water_extent_pct"] - 2.5) < 1e-4
    assert abs(result["aggregate"]["mean_water_coverage_pct"] - 2.5) < 1e-4


# ---------------------------------------------------------------------------
# 2. Road coverage
# ---------------------------------------------------------------------------

def test_road_coverage_is_exact(tmp_path):
    # 500 road pixels out of 10_000 px  ->  5.0 %
    frame = _make_mask(road_px=500, size=100)
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    assert record["road_pixels"] == 500
    assert abs(record["road_coverage_pct"] - 5.0) < 1e-4
    assert abs(result["aggregate"]["road_coverage_pct"] - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# 3. Water near road (spatial-proximity proxy)
# ---------------------------------------------------------------------------

def test_possible_water_near_road_is_positive_when_water_is_adjacent(tmp_path):
    # water strip on top, road strip immediately below (touching)
    frame = _block_mask(100, [(WATER, 0, 40, 0, 100), (ROAD, 40, 60, 0, 100)])
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    assert record["road_pixels"] == 2000
    assert record["road_pixels_near_water"] > 0
    assert record["possible_water_near_road_pct"] > 0
    assert result["aggregate"]["possible_water_near_road_pct"] > 0


def test_possible_water_near_road_is_zero_when_water_is_far(tmp_path):
    # water at the very top, road at the very bottom (>> buffer apart)
    size = 200
    frame = _block_mask(
        size,
        [(WATER, 0, 20, 0, size), (ROAD, 180, 200, 0, size)],
    )
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    assert record["road_pixels"] == 4000
    assert record["road_pixels_near_water"] == 0
    assert record["possible_water_near_road_pct"] == 0.0
    assert result["aggregate"]["possible_water_near_road_pct"] == 0.0


def test_possible_water_near_road_is_none_when_no_road_pixels(tmp_path):
    frame = _make_mask(water_px=250, size=100)
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    assert record["road_pixels"] == 0
    assert record["possible_water_near_road_pct"] is None
    assert result["aggregate"]["possible_water_near_road_pct"] is None


def test_possible_water_near_road_is_zero_when_no_water(tmp_path):
    frame = _make_mask(road_px=500, size=100)
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    assert record["road_pixels"] == 500
    assert record["water_pixels"] == 0
    assert record["possible_water_near_road_pct"] == 0.0
    assert result["aggregate"]["possible_water_near_road_pct"] == 0.0


def test_removed_water_road_overlap_fields_are_absent(tmp_path):
    """The impossible water AND road intersection must no longer be reported."""
    frame = _block_mask(
        100,
        [(WATER, 0, 25, 0, 100), (ROAD, 25, 50, 0, 100)],
    )
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})
    record = _only(result)

    for removed in ("water_on_road_pixels", "water_on_road_pct_of_road"):
        assert removed not in record, removed
        assert removed not in result["aggregate"], removed
    assert "confirmed_flooded_road" in result["not_measured"]
    assert "does NOT confirm" in result["aggregate"]["water_near_road_semantics"]

# ---------------------------------------------------------------------------
# 4. Person in water
# ---------------------------------------------------------------------------

def test_person_detection_in_water_counts_as_in_water_frame_instance(tmp_path):
    frame = _block_mask(200, [(WATER, 0, 100, 0, 200)])
    dets = [_det("frame_00000.png", "person", [10, 10, 60, 60])]
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}}, dets)
    people = _only(result)["people"]

    assert people["frame_instances_in_water"] == 1
    assert people["in_water_breakdown"] == {"person": 1}
    assert people["max_box_water_overlap_pct"] == 100.0
    assert result["aggregate"]["people_frame_instances_in_water"] == 1


# ---------------------------------------------------------------------------
# 5. Vehicle in water
# ---------------------------------------------------------------------------

def test_vehicle_detection_in_water_counts_as_in_water_frame_instance(tmp_path):
    frame = _block_mask(200, [(WATER, 0, 100, 0, 200)])
    dets = [_det("frame_00000.png", "car", [20, 20, 120, 90])]
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}}, dets)
    vehicles = _only(result)["vehicles"]

    assert vehicles["frame_instances_in_water"] == 1
    assert vehicles["in_water_breakdown"] == {"car": 1}
    assert result["aggregate"]["vehicles_frame_instances_in_water"] == 1


# ---------------------------------------------------------------------------
# 6. Near-water detection (inside the 64 px buffer, outside the water)
# ---------------------------------------------------------------------------

def test_detection_near_water_counts_only_as_near_water(tmp_path):
    frame = _block_mask(200, [(WATER, 0, 20, 0, 200)])
    # box sits 40 px below the water edge -> within the 64 px buffer
    dets = [_det("frame_00000.png", "person", [10, 40, 60, 80])]
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}}, dets)
    people = _only(result)["people"]

    assert people["frame_instances_in_water"] == 0
    assert people["frame_instances_near_water"] == 1
    assert result["aggregate"]["people_frame_instances_near_water"] == 1


# ---------------------------------------------------------------------------
# 7. Detection far from water
# ---------------------------------------------------------------------------

def test_detection_far_from_water_is_not_counted(tmp_path):
    frame = _block_mask(200, [(WATER, 0, 20, 0, 200)])
    # box sits 150 px below the water edge -> beyond the 64 px buffer
    dets = [_det("frame_00000.png", "car", [10, 150, 60, 190])]
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}}, dets)
    record = _only(result)

    assert record["people"]["frame_instances_in_water"] == 0
    assert record["people"]["frame_instances_near_water"] == 0
    assert record["vehicles"]["frame_instances_in_water"] == 0
    assert record["vehicles"]["frame_instances_near_water"] == 0
    assert record["vehicles"]["max_box_water_overlap_pct"] == 0.0


# ---------------------------------------------------------------------------
# 8. Overlap threshold behaviour (25 % box overlap)
# ---------------------------------------------------------------------------

def test_min_overlap_threshold_is_respected(tmp_path):
    frame = _block_mask(200, [(WATER, 0, 100, 0, 100)])  # water = x[0:100], y[0:100]
    # box [50,50,150,150] -> window 100x100, intersection 50x50 = 25 %
    dets = [_det("frame_00000.png", "car", [50, 50, 150, 150])]

    lenient = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}}, dets)
    assert _only(lenient)["vehicles"]["max_box_water_overlap_pct"] == 25.0
    assert _only(lenient)["vehicles"]["frame_instances_in_water"] == 1

    strict = _analyze(
        tmp_path,
        {"frame_00000.png": {"mask": frame}},
        dets,
        parameters={"box_water_overlap_min_pct": 30.0},
        job_id="job-strict",
    )
    assert _only(strict)["vehicles"]["frame_instances_in_water"] == 0

# ---------------------------------------------------------------------------
# 9-12. Temporal trend
# ---------------------------------------------------------------------------

def _trend_frames(water_pcts):
    """Build 100x100 frames whose water coverage follows ``water_pcts``."""
    frames = {}
    for index, pct in enumerate(water_pcts):
        # 10_000 px frame -> pct == water_px / 100
        frames[f"frame_{index:05d}.png"] = {
            "mask": _make_mask(water_px=int(round(pct * 100)), size=100)
        }
    return frames


def test_trend_increasing(tmp_path):
    frames = _trend_frames([1, 2, 3, 4, 5, 6, 7, 8])
    trend = _analyze(tmp_path, frames)["aggregate"]["temporal_trend"]
    assert trend["label"] == "increasing"
    assert trend["frames_compared"] == 8
    assert trend["first_half_mean_water_pct"] == 2.5
    assert trend["second_half_mean_water_pct"] == 6.5
    assert trend["relative_change"] > 0


def test_trend_decreasing(tmp_path):
    frames = _trend_frames([8, 7, 6, 5, 4, 3, 2, 1])
    trend = _analyze(tmp_path, frames)["aggregate"]["temporal_trend"]
    assert trend["label"] == "decreasing"
    assert trend["relative_change"] < 0


def test_trend_stable(tmp_path):
    frames = _trend_frames([4, 4, 4, 4, 4, 4, 4, 4])
    trend = _analyze(tmp_path, frames)["aggregate"]["temporal_trend"]
    assert trend["label"] == "stable"
    assert trend["relative_change"] == 0


def test_trend_insufficient_data(tmp_path):
    # only 2 frames -> below trend_min_frames (6)
    frames = _trend_frames([1, 9])
    trend = _analyze(tmp_path, frames)["aggregate"]["temporal_trend"]
    assert trend["label"] == "insufficient_data"
    assert trend["frames_compared"] == 2
    assert trend["min_frames_required"] == 6


def test_relative_change_is_none_when_first_half_has_no_water(tmp_path):
    """Regression: dividing by a ~0 first half must not produce garbage.

    Previously this produced relative_change = 2026286.2069 instead of None.
    """
    frames = _trend_frames([0, 0, 0, 0, 4, 4, 4, 4])
    trend = _analyze(tmp_path, frames)["aggregate"]["temporal_trend"]

    assert trend["frames_compared"] == 8
    assert trend["first_half_mean_water_pct"] == 0.0
    assert trend["second_half_mean_water_pct"] == 4.0
    # absolute change must still be computed correctly
    assert trend["absolute_change_pct_points"] == 4.0
    # relative change is undefined and must be null, not a huge number
    assert trend["relative_change"] is None
    assert trend["label"] == "increasing"


# ---------------------------------------------------------------------------
# 13. Empty / zero-frame input
# ---------------------------------------------------------------------------

def test_empty_and_missing_artifacts_return_honest_empty_result(tmp_path):
    data_dir, job_id = _write_job(tmp_path, {})
    empty = FloodAnalyzer().analyze(ArtifactBundle(job_id, data_dir))

    assert empty["frames"] == []
    assert empty["aggregate"]["frames_analyzed"] == 0
    assert empty["aggregate"]["confidence"] is None
    assert "No segmentation masks" in empty["aggregate"]["note"]

    missing = FloodAnalyzer().analyze(ArtifactBundle("does-not-exist", data_dir))
    assert missing["aggregate"]["frames_analyzed"] == 0
    assert missing["frames"] == []


# ---------------------------------------------------------------------------
# 14. Frame-instance semantics (no unique-object claims)
# ---------------------------------------------------------------------------

def test_counts_are_frame_instances_not_unique_objects(tmp_path):
    frame = _block_mask(200, [(WATER, 0, 100, 0, 200)])
    dets = [
        _det("frame_00000.png", "person", [10, 10, 60, 60]),
        _det("frame_00000.png", "car", [100, 20, 180, 90]),
    ]
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}}, dets)
    aggregate = result["aggregate"]

    assert aggregate["people_frame_instances_in_water"] == 1
    assert aggregate["vehicles_frame_instances_in_water"] == 1

    for key in aggregate:
        assert not key.startswith("unique"), f"unexpected unique-count key: {key}"

    assert "unique_people_count" in result["not_measured"]
    assert "unique_vehicle_count" in result["not_measured"]
    assert "frame_instances" in result["count_semantics"]
    assert "NOT unique" in result["count_semantics"]
    assert "NOT unique" in aggregate["count_semantics"]


# ---------------------------------------------------------------------------
# 15. Provenance / limitations
# ---------------------------------------------------------------------------

def test_provenance_and_limitations_are_preserved(tmp_path):
    frame = _make_mask(water_px=250, size=100)
    result = _analyze(tmp_path, {"frame_00000.png": {"mask": frame}})

    assert result["hazard"] == "flood"
    assert result["nature"] == "proxy_estimate"
    assert "segformer" in result["source"]
    assert "yolo11n" in result["source"]
    assert result["disclaimer"].startswith("Proxy/estimate only")

    # no fabricated overall confidence anywhere
    assert result["confidence"] is None
    assert result["aggregate"]["confidence"] is None
    assert result["confidence_note"]

    aggregate_blob = json.dumps(result["aggregate"])
    for forbidden in (
        "flood_area_sq_m",
        "metric_water_depth",
        "georeferenced_coordinates",
        "flood_severity_score",
        "overall_flood_confidence",
        "flow_direction",
        "flow_velocity",
    ):
        assert forbidden in result["not_measured"], forbidden
        assert forbidden not in result["aggregate"], forbidden
        assert forbidden not in aggregate_blob, forbidden


# ---------------------------------------------------------------------------
# 16. Relative depth evidence stays relative
# ---------------------------------------------------------------------------

def test_relative_depth_evidence_is_relative_not_metric(tmp_path):
    size = 100
    mask = _make_mask(water_px=100, size=size)

    depth = np.full((size, size), 100, dtype=np.uint8)
    flat = depth.reshape(-1)
    flat[:100] = 0  # same row-major cells as the water region

    result = _analyze(tmp_path, {"frame_00000.png": {"mask": mask, "depth": depth}})
    evidence = result["aggregate"]["relative_depth_evidence"]

    assert evidence["frames_with_water_depth_evidence"] == 1
    assert abs(evidence["mean_relative_depth_in_water"] - 0.0) < 1e-6
    # (10_000 - 100) px at 100 / 10_000 = 99.0
    assert abs(evidence["mean_relative_depth_scene"] - 99.0) < 1e-6
    assert abs(evidence["mean_within_frame_depth_separation"] - 99.0) < 1e-6
    assert evidence["metric_depth_available"] is False
    assert evidence["within_frame_only"] is True


