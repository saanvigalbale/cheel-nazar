"""Disaster intelligence layer for Cheel Nazar (Phase 6B.1).

Analyzers here are **pure post-processing** over the artifacts already
produced by the Phase 6A evidence layer:

    data/<job_id>/frames/*.jpg
    data/<job_id>/analysis/detections.json          (YOLO11n)
    data/<job_id>/analysis/segmentation/*_mask.png  (SegFormer-B0 ADE20K)
    data/<job_id>/analysis/segmentation/summary.json
    data/<job_id>/analysis/depth/*.jpg              (Depth Anything V2, relative)

They never load model weights, never modify Phase 6A artifacts, and never
claim metric or georeferenced quantities.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# Shared label vocabularies (kept identical to Phase 6A results_builder).
PEOPLE_LABELS = {"person"}
VEHICLE_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "boat",
    "train",
    "airplane",
}
ROAD_LABELS = {"road", "sidewalk", "path", "runway", "dirt track"}

DEFAULT_SEGMENTATION_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


def load_json(path: Path) -> Optional[Any]:
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_data_dir(data_dir: Union[str, Path]) -> Path:
    path = Path(data_dir)
    if path.exists():
        return path
    project_root = Path(__file__).resolve().parent.parent.parent / path
    if project_root.exists():
        return project_root
    return path


class ArtifactBundle:
    """Read-only accessor for the Phase 6A artifacts of a single job."""

    def __init__(self, job_id: str, data_dir: Union[str, Path] = "data") -> None:
        self.job_id = job_id
        self.data_path = _resolve_data_dir(data_dir)
        self.job_dir = self.data_path / job_id
        self.frames_dir = self.job_dir / "frames"
        self.analysis_dir = self.job_dir / "analysis"
        self.segmentation_dir = self.analysis_dir / "segmentation"
        self.depth_dir = self.analysis_dir / "depth"
        self.disaster_dir = self.analysis_dir / "disaster"

        raw = load_json(self.analysis_dir / "detections.json")
        self.detections: List[Dict[str, Any]] = raw if isinstance(raw, list) else []

        summary = load_json(self.segmentation_dir / "summary.json")
        self.seg_summary: Dict[str, Any] = summary if isinstance(summary, dict) else {}

        self.detections_by_frame: Dict[str, List[Dict[str, Any]]] = {}
        for det in self.detections:
            frame = det.get("frame")
            if frame:
                self.detections_by_frame.setdefault(frame, []).append(det)

    @property
    def model_name(self) -> Optional[str]:
        model = self.seg_summary.get("model")
        return str(model) if model else None

    def frame_names(self) -> List[str]:
        """Frames that have a segmentation mask, in pipeline order."""
        names = [
            entry.get("frame")
            for entry in self.seg_summary.get("frames", [])
            if isinstance(entry, dict) and entry.get("frame")
        ]
        if names:
            return names
        if self.frames_dir.is_dir():
            return [p.name for p in sorted(self.frames_dir.glob("*.jpg"))]
        return []

    def mask_path(self, frame: str) -> Path:
        return self.segmentation_dir / (Path(frame).stem + "_mask.png")

    def depth_path(self, frame: str) -> Path:
        return self.depth_dir / frame

    def load_mask(self, frame: str) -> Optional[np.ndarray]:
        path = self.mask_path(frame)
        if not path.is_file():
            return None
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.uint8)

    def load_depth(self, frame: str) -> Optional[np.ndarray]:
        path = self.depth_path(frame)
        if not path.is_file():
            return None
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.int32)

    def class_id_map(self) -> Dict[int, str]:
        return resolve_class_ids(self)


def _load_config_id2label(model_name: str) -> Optional[Dict[int, str]]:
    """Read ``id2label`` from the model *configuration* only (never weights)."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return None

    for local_only in (True, False):
        try:
            config = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
            id2label = getattr(config, "id2label", None)
            if id2label:
                return {int(k): str(v).strip() for k, v in id2label.items()}
        except Exception:
            continue
    return None


def resolve_class_ids(bundle: ArtifactBundle) -> Dict[int, str]:
    """Resolve ``class_id -> label`` for the segmentation masks.

    Prefers ``summary.json["class_id_map"]`` when present; otherwise falls back
    to the checkpoint configuration (no weight loading).
    """
    mapping = bundle.seg_summary.get("class_id_map")
    if isinstance(mapping, dict) and mapping:
        return {int(k): str(v).strip() for k, v in mapping.items()}

    model_name = bundle.model_name or DEFAULT_SEGMENTATION_MODEL
    fallback = _load_config_id2label(model_name)
    if not fallback:
        raise RuntimeError(
            "Unable to resolve ADE20K class ids: summary.json has no "
            "'class_id_map' and the model configuration could not be read."
        )
    return fallback


def label_ids(class_ids: Dict[int, str], labels) -> np.ndarray:
    """Return the sorted class ids whose label is in ``labels``."""
    wanted = {label.strip() for label in labels}
    ids = [cid for cid, label in class_ids.items() if label in wanted]
    return np.array(sorted(ids), dtype=np.uint8)


def pct(numerator: float, denominator: float) -> float:
    """Percentage helper that never divides by zero."""
    if not denominator:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), 4)


class HazardAnalyzer(ABC):
    """Contract for disaster analyzers.

    Analyzers receive an :class:`ArtifactBundle` and return a JSON-serialisable
    payload. They must never invent confidences or metric quantities.
    """

    name: str = "hazard"
    nature: str = "proxy_estimate"

    @abstractmethod
    def analyze(self, bundle: ArtifactBundle) -> Dict[str, Any]:
        """Analyse one job and return the hazard payload."""
        ...

