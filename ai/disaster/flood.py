"""Flood proxy analyzer for Cheel Nazar (Phase 6B.1).

Consumes the Phase 6A evidence artifacts and produces **proxy / estimate**
flood indicators. It deliberately produces NO metric quantities:

  * no flood area in m2, no flood boundary, no georeferenced coordinates
  * no water depth in metres, no flow direction/velocity
  * no flood severity score, no single flood confidence score
  * no unique person / vehicle counts - the video is sampled at 2 FPS, so
    every detection count here is a FRAME INSTANCE, not a unique object

Usage (from the repository root)::

    python -m ai.disaster.flood --job-id <job_id> --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .base import (
    PEOPLE_LABELS,
    ROAD_LABELS,
    VEHICLE_LABELS,
    ArtifactBundle,
    HazardAnalyzer,
    label_ids,
    pct,
    resolve_class_ids,
)

# ADE20K semantic water classes (generic water - NOT flood-specific).
WATER_LABELS = {"water", "sea", "river", "lake", "waterfall", "swimming pool"}

DEFAULT_PARAMETERS: Dict[str, Any] = {
    "water_presence_threshold_pct": 0.5,
    "box_water_overlap_min_pct": 20.0,
    "near_water_buffer_px": 64,
    "water_near_road_buffer_px": 32,
    "trend_min_frames": 6,
    "trend_relative_change_threshold": 0.15,
}

NOT_MEASURED = [
    "flood_area_sq_m",
    "flood_boundary",
    "metric_water_depth",
    "flood_severity_score",
    "overall_flood_confidence",
    "unique_people_count",
    "unique_vehicle_count",
    "georeferenced_coordinates",
    "flow_direction",
    "flow_velocity",
    "confirmed_flooded_road",
    "flooded_road_length_or_area",
]

DISCLAIMER = (
    "Proxy/estimate only. ADE20K 'water' is a generic semantic class and does "
    "not distinguish flood water from rivers, lakes, sea or pools. All "
    "detection counts are FRAME INSTANCES (2 FPS sampled video), not unique "
    "objects. Relative depth is per-frame normalised and is supporting "
    "evidence only - it is not a metric depth. "
    "'possible_water_near_road_pct' is a SPATIAL-PROXIMITY proxy: it only "
    "shows that water pixels lie within a small buffer of road pixels, and "
    "does NOT prove that any road is flooded."
)

COUNT_SEMANTICS = (
    "frame_instances (one count per detection per frame; frames overlap at "
    "2 FPS so these are NOT unique people/vehicles)"
)


def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation of ``mask`` with a (2r+1) square structuring element."""
    if radius_px <= 0 or not mask.any():
        return mask
    kernel = np.ones((2 * radius_px + 1, 2 * radius_px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _near_road_semantics(buffer_px: int) -> str:
    """Human-readable meaning of the road/water proximity proxy."""
    return (
        "Proxy: share of road-classified pixels lying within a "
        f"{buffer_px} px buffer of water-classified pixels (spatial proximity "
        "only). ADE20K class ids are mutually exclusive per pixel, so a direct "
        "water-AND-road intersection does not exist and is not reported. "
        "This does NOT confirm that a road is flooded."
    )


class FloodAnalyzer(HazardAnalyzer):
    """Semantic flood indicators from segmentation + detection + relative depth."""

    name = "flood"
    nature = "proxy_estimate"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.parameters: Dict[str, Any] = {**DEFAULT_PARAMETERS, **(parameters or {})}

    def _box_stats(
        self,
        detections: List[Dict[str, Any]],
        labels,
        water_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Frame instances of ``labels`` inside / near the water mask."""
        in_water = 0
        near_water = 0
        in_breakdown: Dict[str, int] = {}
        near_breakdown: Dict[str, int] = {}
        overlaps: List[float] = []

        height, width = water_mask.shape
        threshold = float(params["box_water_overlap_min_pct"])
        buffer_px = int(params["near_water_buffer_px"])

        for det in detections:
            label = det.get("label")
            if label not in labels:
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x1 = max(0.0, min(float(bbox[0]), width))
            y1 = max(0.0, min(float(bbox[1]), height))
            x2 = max(0.0, min(float(bbox[2]), width))
            y2 = max(0.0, min(float(bbox[3]), height))
            if x2 <= x1 or y2 <= y1:
                continue

            ix1, iy1 = int(x1), int(y1)
            ix2 = min(width, max(ix1 + 1, int(np.ceil(x2))))
            iy2 = min(height, max(iy1 + 1, int(np.ceil(y2))))

            window = water_mask[iy1:iy2, ix1:ix2]
            sub_area = float(window.size)
            overlap_pct = 100.0 * float(window.sum()) / sub_area if sub_area else 0.0
            overlaps.append(round(overlap_pct, 2))

            if overlap_pct >= threshold:
                in_water += 1
                in_breakdown[label] = in_breakdown.get(label, 0) + 1

            ex1 = max(0, ix1 - buffer_px)
            ey1 = max(0, iy1 - buffer_px)
            ex2 = min(width, ix2 + buffer_px)
            ey2 = min(height, iy2 + buffer_px)
            expanded = water_mask[ey1:ey2, ex1:ex2]
            if expanded.size and bool(expanded.any()):
                near_water += 1
                near_breakdown[label] = near_breakdown.get(label, 0) + 1

        return {
            "frame_instances_in_water": in_water,
            "frame_instances_near_water": near_water,
            "in_water_breakdown": dict(sorted(in_breakdown.items(), key=lambda kv: -kv[1])),
            "near_water_breakdown": dict(sorted(near_breakdown.items(), key=lambda kv: -kv[1])),
            "max_box_water_overlap_pct": max(overlaps) if overlaps else None,
        }

    def analyze(self, bundle: ArtifactBundle) -> Dict[str, Any]:
        """Analyse one job from Phase 6A artifacts (no model inference)."""
        params = self.parameters
        class_ids = resolve_class_ids(bundle)
        water_ids = label_ids(class_ids, WATER_LABELS)
        road_ids = label_ids(class_ids, ROAD_LABELS)
        threshold = float(params["water_presence_threshold_pct"])

        records: List[Dict[str, Any]] = []

        for frame in bundle.frame_names():
            mask = bundle.load_mask(frame)
            if mask is None:
                continue

            depth = bundle.load_depth(frame)
            height, width = mask.shape
            total_px = int(mask.size)

            if water_ids.size:
                water_mask = np.isin(mask, water_ids)
            else:
                water_mask = np.zeros(mask.shape, dtype=bool)
            if road_ids.size:
                road_mask = np.isin(mask, road_ids)
            else:
                road_mask = np.zeros(mask.shape, dtype=bool)

            water_px = int(water_mask.sum())
            road_px = int(road_mask.sum())

            # Spatial-proximity proxy (replaces the impossible water AND road
            # pixel intersection: semantic class ids are mutually exclusive).
            near_road_buffer = int(params["water_near_road_buffer_px"])
            if road_px:
                water_dilated = _dilate(water_mask, near_road_buffer)
                road_near_water_px = int(
                    np.logical_and(road_mask, water_dilated).sum()
                )
            else:
                road_near_water_px = 0

            detections = bundle.detections_by_frame.get(frame, [])
            people = self._box_stats(detections, PEOPLE_LABELS, water_mask, params)
            vehicles = self._box_stats(detections, VEHICLE_LABELS, water_mask, params)

            depth_in_water = None
            depth_scene = None
            depth_separation = None
            if depth is not None:
                depth_scene = round(float(depth.mean()), 2)
                if water_px:
                    depth_in_water = round(float(depth[water_mask].mean()), 2)
                    depth_separation = round(depth_scene - depth_in_water, 2)

            water_pct = pct(water_px, total_px)
            records.append({
                "frame": frame,
                "frame_width": width,
                "frame_height": height,
                "total_pixels": total_px,
                "water_pixels": water_px,
                "water_coverage_pct": water_pct,
                "water_present": bool(water_pct >= threshold),
                "road_pixels": road_px,
                "road_coverage_pct": pct(road_px, total_px),
                "road_pixels_near_water": road_near_water_px,
                "possible_water_near_road_pct": (
                    pct(road_near_water_px, road_px) if road_px else None
                ),
                "water_near_road_semantics": _near_road_semantics(near_road_buffer),
                "people": people,
                "vehicles": vehicles,
                "mean_relative_depth_in_water": depth_in_water,
                "mean_relative_depth_scene": depth_scene,
                "within_frame_depth_separation": depth_separation,
            })

        return {
            "hazard": self.name,
            "nature": self.nature,
            "job_id": bundle.job_id,
            "source": "segformer-ade20k + yolo11n + depth-anything-v2(relative)",
            "segmentation_model": bundle.model_name,
            "confidence": None,
            "confidence_note": (
                "No single flood confidence is reported: no calibrated "
                "model-level confidence exists for this proxy. Individual "
                "numbers carry their own source in the per-frame evidence."
            ),
            "count_semantics": COUNT_SEMANTICS,
            "parameters": params,
            "disclaimer": DISCLAIMER,
            "not_measured": list(NOT_MEASURED),
            "aggregate": self._aggregate(records, params),
            "frames": records,
        }

    def _trend(
        self,
        water_pcts: List[float],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare first-half vs second-half mean water coverage."""
        n = len(water_pcts)
        min_frames = int(params["trend_min_frames"])
        rel_threshold = float(params["trend_relative_change_threshold"])

        if n < min_frames:
            return {
                "label": "insufficient_data",
                "frames_compared": n,
                "min_frames_required": min_frames,
                "note": "Too few frames to assess a temporal trend.",
            }

        mid = n // 2
        first = water_pcts[:mid]
        second = water_pcts[mid:]
        first_mean = sum(first) / len(first)
        second_mean = sum(second) / len(second)
        delta = second_mean - first_mean

        # Guard against a divide-by-near-zero artifact when the first half of
        # the flight saw no water at all (relative change is then undefined).
        if abs(first_mean) < 1e-3:
            relative_change = None
            if delta > 0:
                label = "increasing"
            elif delta < 0:
                label = "decreasing"
            else:
                label = "stable"
        else:
            relative_change = round(delta / abs(first_mean), 4)
            if abs(relative_change) <= rel_threshold:
                label = "stable"
            elif relative_change > 0:
                label = "increasing"
            else:
                label = "decreasing"

        return {
            "label": label,
            "frames_compared": n,
            "first_half_mean_water_pct": round(first_mean, 4),
            "second_half_mean_water_pct": round(second_mean, 4),
            "absolute_change_pct_points": round(delta, 4),
            "relative_change": relative_change,
            "relative_change_threshold": rel_threshold,
            "note": (
                "Change in scene water coverage between video halves; not a "
                "measurement of flood progression. 'relative_change' is null "
                "when the first half had no water coverage."
            ),
        }

    def _aggregate(
        self,
        records: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        n = len(records)
        if n == 0:
            return {
                "frames_analyzed": 0,
                "note": "No segmentation masks found for this job.",
                "confidence": None,
            }

        total_px = sum(r["total_pixels"] for r in records)
        total_water = sum(r["water_pixels"] for r in records)
        total_road = sum(r["road_pixels"] for r in records)
        total_road_near_water = sum(r["road_pixels_near_water"] for r in records)

        water_pcts = [r["water_coverage_pct"] for r in records]
        present_frames = sum(1 for r in records if r["water_present"])

        people_in = sum(r["people"]["frame_instances_in_water"] for r in records)
        people_near = sum(r["people"]["frame_instances_near_water"] for r in records)
        veh_in = sum(r["vehicles"]["frame_instances_in_water"] for r in records)
        veh_near = sum(r["vehicles"]["frame_instances_near_water"] for r in records)

        def merge_breakdown(key: str, group: str) -> Dict[str, int]:
            merged: Dict[str, int] = {}
            for r in records:
                for label, count in r[group][key].items():
                    merged[label] = merged.get(label, 0) + count
            return dict(sorted(merged.items(), key=lambda kv: -kv[1]))

        depth_records = [
            r for r in records if r["within_frame_depth_separation"] is not None
        ]

        if depth_records:
            k = len(depth_records)
            mean_in_water = sum(r["mean_relative_depth_in_water"] for r in depth_records) / k
            mean_scene = sum(r["mean_relative_depth_scene"] for r in depth_records) / k
            mean_sep = sum(r["within_frame_depth_separation"] for r in depth_records) / k
            depth_evidence = {
                "frames_with_water_depth_evidence": k,
                "mean_relative_depth_in_water": round(mean_in_water, 2),
                "mean_relative_depth_scene": round(mean_scene, 2),
                "mean_within_frame_depth_separation": round(mean_sep, 2),
                "metric_depth_available": False,
                "within_frame_only": True,
                "interpretation": (
                    "Positive separation means water pixels are darker "
                    "(relatively farther) than the rest of the frame. "
                    "Supporting evidence only."
                ),
            }
        else:
            depth_evidence = {
                "frames_with_water_depth_evidence": 0,
                "mean_relative_depth_in_water": None,
                "mean_relative_depth_scene": None,
                "mean_within_frame_depth_separation": None,
                "metric_depth_available": False,
                "within_frame_only": True,
            }

        return {
            "frames_analyzed": n,
            "water_extent_pct": pct(total_water, total_px),
            "water_extent_semantics": (
                "Pixel-weighted share of the scene classified as water "
                "(ADE20K). Not an affected-area figure."
            ),
            "water_present_frames": present_frames,
            "water_present_frame_ratio": pct(present_frames, n),
            "mean_water_coverage_pct": round(sum(water_pcts) / n, 4),
            "median_water_coverage_pct": round(float(median(water_pcts)), 4),
            "max_water_coverage_pct": round(max(water_pcts), 4),
            "road_coverage_pct": pct(total_road, total_px),
            "possible_water_near_road_pct": (
                pct(total_road_near_water, total_road) if total_road else None
            ),
            "water_near_road_semantics": _near_road_semantics(
                int(params["water_near_road_buffer_px"])
            ),
            "people_frame_instances_in_water": people_in,
            "people_frame_instances_near_water": people_near,
            "people_in_water_breakdown": merge_breakdown("in_water_breakdown", "people"),
            "people_near_water_breakdown": merge_breakdown("near_water_breakdown", "people"),
            "vehicles_frame_instances_in_water": veh_in,
            "vehicles_frame_instances_near_water": veh_near,
            "vehicles_in_water_breakdown": merge_breakdown("in_water_breakdown", "vehicles"),
            "vehicles_near_water_breakdown": merge_breakdown("near_water_breakdown", "vehicles"),
            "count_semantics": COUNT_SEMANTICS,
            "relative_depth_evidence": depth_evidence,
            "temporal_trend": self._trend(water_pcts, params),
            "confidence": None,
        }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6B.1 - flood proxy analyzer (reads existing Phase 6A artifacts)"
    )
    parser.add_argument("--job-id", required=True, help="Job ID to analyse")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--frames-output",
        default=None,
        help="Per-frame output path (default: <job>/analysis/disaster/flood_frames.json)",
    )
    parser.add_argument(
        "--aggregate-output",
        default=None,
        help="Optional path to also write the full aggregate payload as JSON",
    )
    parser.add_argument(
        "--print-frames",
        action="store_true",
        help="Also print the per-frame records",
    )
    args = parser.parse_args()

    bundle = ArtifactBundle(args.job_id, args.data_dir)
    result = FloodAnalyzer().analyze(bundle)

    frames_path = (
        Path(args.frames_output)
        if args.frames_output
        else bundle.disaster_dir / "flood_frames.json"
    )
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_payload = {
        "hazard": result["hazard"],
        "job_id": result["job_id"],
        "segmentation_model": result["segmentation_model"],
        "source": result["source"],
        "parameters": result["parameters"],
        "count_semantics": result["count_semantics"],
        "disclaimer": result["disclaimer"],
        "frames_analyzed": result["aggregate"]["frames_analyzed"],
        "frames": result["frames"],
    }
    frames_path.write_text(
        json.dumps(frames_payload, indent=2),
        encoding="utf-8",
    )

    if args.aggregate_output:
        out = Path(args.aggregate_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Per-frame evidence written to: {frames_path}")
    output = dict(result)
    if not args.print_frames:
        output.pop("frames", None)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()




