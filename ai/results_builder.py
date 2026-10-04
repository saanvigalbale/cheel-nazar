"""Phase 6A - real results aggregation for Cheel Nazar.

Builds ``data/<job_id>/results.json`` using ONLY artifacts produced by the
existing models:

  * YOLO11n (COCO)      -> data/<job_id>/analysis/detections.json
  * SegFormer-B0 ADE20K -> data/<job_id>/analysis/segmentation/summary.json
  * Depth Anything V2   -> data/<job_id>/analysis/depth/*.jpg  (relative only)

No fabricated values are produced. Anything the pipeline cannot actually
measure is listed under ``not_measured``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

PEOPLE_CLASSES = {"person"}
VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "boat",
    "train",
    "airplane",
}

DETECTOR_SOURCE = "yolo11n"
SEGMENTATION_SOURCE = "segformer-ade20k"
DEPTH_SOURCE = "depth-anything-v2"

WATER_DISCLAIMER = (
    "Semantic water proxy (ADE20K water/sea/river/lake/waterfall/swimming pool); "
    "not a flood or metric water-depth measurement."
)

NOT_MEASURED = [
    "metric_water_depth",
    "georeferenced_area",
    "structural_damage",
    "debris_extent",
    "landslide_extent",
    "fire_extent",
    "hazard_zones",
    "georeferenced_coordinates",
]

# Hazard modules that are not implemented yet. Listed for transparency only -
# this builder never performs hazard-specific calculations.
PENDING_HAZARD_MODULES = [
    "landslide",
    "earthquake",
    "wildfire",
    "cyclone",
    "avalanche",
    "industrial",
]


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _detection_category(detections: List[Dict[str, Any]], classes: set) -> Dict[str, Any]:
    """Aggregate raw detections belonging to ``classes`` into an honest summary."""
    matched = [d for d in detections if d.get("label") in classes]

    breakdown: Dict[str, int] = {}
    frames = set()
    confidences: List[float] = []

    for det in matched:
        label = det.get("label", "unknown")
        breakdown[label] = breakdown.get(label, 0) + 1
        if det.get("frame") is not None:
            frames.add(det["frame"])
        conf = det.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))

    return {
        "nature": "detected",
        "count": len(matched),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "min_confidence": round(min(confidences), 4) if confidences else None,
        "frames_with_detections": len(frames),
        "source": DETECTOR_SOURCE,
        "breakdown": dict(sorted(breakdown.items(), key=lambda kv: -kv[1])),
    }


def _relative_depth_stats(depth_dir: Path) -> Dict[str, Any]:
    """Relative (non-metric) statistics across the saved grayscale depth maps."""
    stats: Dict[str, Any] = {
        "nature": "relative",
        "metric_depth_available": False,
        "source": DEPTH_SOURCE,
        "maps_generated": 0,
        "units": "normalized 8-bit grayscale (relative)",
        "mean_relative_depth": None,
        "min_relative_depth": None,
        "max_relative_depth": None,
        "disclaimer": "Relative monocular depth - not metric, not water depth.",
    }

    if not depth_dir.is_dir():
        return stats

    total = 0
    count = 0
    vmin: Optional[int] = None
    vmax: Optional[int] = None
    maps = 0

    for depth_file in sorted(depth_dir.iterdir()):
        if not depth_file.is_file() or depth_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            with Image.open(depth_file) as img:
                arr = np.asarray(img.convert("L"), dtype=np.uint8)
        except OSError:
            continue
        maps += 1
        total += int(arr.sum())
        count += int(arr.size)
        f_min = int(arr.min())
        f_max = int(arr.max())
        vmin = f_min if vmin is None else min(vmin, f_min)
        vmax = f_max if vmax is None else max(vmax, f_max)

    stats["maps_generated"] = maps
    if count:
        stats["mean_relative_depth"] = round(total / count, 2)
        stats["min_relative_depth"] = vmin
        stats["max_relative_depth"] = vmax

    return stats


def _coverage(seg_summary: Optional[Any]) -> Dict[str, float]:
    base = {"structures": 0.0, "roads": 0.0, "vegetation": 0.0, "terrain": 0.0, "water": 0.0}
    if isinstance(seg_summary, dict) and isinstance(seg_summary.get("categories"), dict):
        for key in base:
            value = seg_summary["categories"].get(key)
            if isinstance(value, (int, float)):
                base[key] = round(float(value), 2)
    return base


# Detailed per-frame evidence is stored separately (e.g.
# analysis/disaster/<hazard>_frames.json) and must not be duplicated into
# results.json. Only the aggregate payload belongs in the results document.
PER_FRAME_EVIDENCE_KEY = "frames"


def _strip_per_frame_evidence(payload: Any) -> Any:
    """Return a copy of a hazard payload without its per-frame evidence.

    Applied generically to every registered hazard so future analyzers that
    emit a ``frames`` collection get the same treatment without any change to
    the analyzer itself. Only the top-level ``frames`` key is removed -
    aggregate fields such as ``aggregate.frames_analyzed`` are untouched.
    """
    if not isinstance(payload, dict) or PER_FRAME_EVIDENCE_KEY not in payload:
        return payload
    return {k: v for k, v in payload.items() if k != PER_FRAME_EVIDENCE_KEY}


def _disaster_sections(analysis_dir: Path) -> Dict[str, Any]:
    """Auto-discover the generic disaster payload written by the job pipeline.

    The payload is produced by the ``ai.disaster`` registry; this builder only
    forwards it, so it stays hazard-agnostic.
    """
    payload = _load_json(analysis_dir / "disaster" / "disasters_summary.json")
    if not isinstance(payload, dict):
        return {
            "disasters": {},
            "available": [],
            "pending": list(PENDING_HAZARD_MODULES),
        }

    raw = payload.get("disasters")
    source: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    available = sorted(source.keys())

    return {
        "disasters": {
            hazard: _strip_per_frame_evidence(finding)
            for hazard, finding in source.items()
        },
        "available": available,
        "pending": [name for name in PENDING_HAZARD_MODULES if name not in available],
    }


def _merge_not_measured(disasters: Dict[str, Any]) -> List[str]:
    """Base not-measured list plus every hazard module's own list."""
    merged = list(NOT_MEASURED)
    for payload in disasters.values():
        if not isinstance(payload, dict):
            continue
        for item in payload.get("not_measured") or []:
            if item not in merged:
                merged.append(item)
    return merged

def build_results(job_id: str, data_dir: Union[str, Path] = "data") -> Dict[str, Any]:
    """Assemble an honest results payload from real artifacts for ``job_id``."""
    data_path = Path(data_dir)
    if not data_path.exists():
        root_data = Path(__file__).resolve().parent.parent / data_dir
        if root_data.exists():
            data_path = root_data

    analysis_dir = data_path / job_id / "analysis"

    raw = _load_json(analysis_dir / "detections.json")
    detections: List[Dict[str, Any]] = raw if isinstance(raw, list) else []

    seg_summary = _load_json(analysis_dir / "segmentation" / "summary.json")
    cov = _coverage(seg_summary)
    depth_stats = _relative_depth_stats(analysis_dir / "depth")

    disaster = _disaster_sections(analysis_dir)

    return {
        "job_id": job_id,
        "models": {
            "detector": "YOLO11n COCO",
            "segmentation": "SegFormer-B0 ADE20K",
            "depth": "Depth Anything V2 Small (relative)",
        },
        "categories": {
            "people": _detection_category(detections, PEOPLE_CLASSES),
            "vehicles": _detection_category(detections, VEHICLE_CLASSES),
            "structures": {
                "nature": "semantic_estimate",
                "coverage_pct": cov["structures"],
                "source": SEGMENTATION_SOURCE,
            },
            "roads": {
                "nature": "semantic_estimate",
                "coverage_pct": cov["roads"],
                "source": SEGMENTATION_SOURCE,
            },
            "vegetation": {
                "nature": "semantic_estimate",
                "coverage_pct": cov["vegetation"],
                "source": SEGMENTATION_SOURCE,
            },
            "terrain": {
                "nature": "semantic_estimate",
                "coverage_pct": cov["terrain"],
                "source": SEGMENTATION_SOURCE,
            },
            "possible_water_extent": {
                "nature": "semantic_estimate",
                "coverage_pct": cov["water"],
                "source": SEGMENTATION_SOURCE,
                "disclaimer": WATER_DISCLAIMER,
            },
        },
        "relative_depth": depth_stats,
        "disasters": disaster["disasters"],
        "disaster_modules_available": disaster["available"],
        "disaster_modules_not_implemented": disaster["pending"],
        "not_measured": _merge_not_measured(disaster["disasters"]),
        "model_url": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6A - build real results.json for a job"
    )
    parser.add_argument("--job-id", required=True, help="Job ID to aggregate")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this path (default: print to stdout)",
    )
    args = parser.parse_args()

    results = build_results(args.job_id, args.data_dir)
    text = json.dumps(results, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
