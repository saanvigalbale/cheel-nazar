"""
YOLO Object Detection Pipeline for Cheel Nazar (Phase 6.2).

Modular detector for aerial drone video frames using YOLO11n.
Processes all JPG frames in a job directory and saves real detections
(frame filename, class label, confidence, bounding box) to detections.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ultralytics import YOLO


def resolve_model_path(model_path: Union[str, Path] = "yolo11n.pt") -> Path:
    """Resolve YOLO weights path across common project locations."""
    path = Path(model_path)
    if path.is_file():
        return path

    # Check project root (ai/.. -> project_root/yolo11n.pt)
    project_root = Path(__file__).resolve().parent.parent
    root_model = project_root / model_path
    if root_model.is_file():
        return root_model

    # Check local ai/ directory
    local_model = Path(__file__).resolve().parent / model_path
    if local_model.is_file():
        return local_model

    return path


class YOLODetector:
    """Reusable YOLO detector for Cheel Nazar aerial frames."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11n.pt",
        conf_threshold: float = 0.25,
    ):
        resolved_path = resolve_model_path(model_path)
        self.model_path = str(resolved_path)
        self.conf_threshold = conf_threshold
        self.model = YOLO(self.model_path)

    def detect_frame(
        self,
        image_path: Union[str, Path],
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run detection on a single image frame.
        Returns a list of detections with frame name, label, confidence, and bbox coordinates.
        """
        image_path = Path(image_path)
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        results = self.model(str(image_path), conf=conf, verbose=False)
        detections: List[Dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[class_id]
                bbox = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                detections.append({
                    "frame": image_path.name,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                })

        return detections

    def detect_frames(
        self,
        frames_dir: Union[str, Path],
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run detection across all JPG/JPEG frames in the given directory.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        frame_files = sorted(
            [f for f in frames_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")]
        )

        all_detections: List[Dict[str, Any]] = []
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        for frame_file in frame_files:
            frame_dets = self.detect_frame(frame_file, conf_threshold=conf)
            all_detections.extend(frame_dets)

        return all_detections


def detect_objects(
    frames_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    model_path: Union[str, Path] = "yolo11n.pt",
) -> List[Dict[str, Any]]:
    """
    Process all JPG frames in frames_dir, detect objects using YOLO11n,
    and save detection output as JSON if output_file is provided.
    """
    frames_dir = Path(frames_dir)
    detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
    detections = detector.detect_frames(frames_dir, conf_threshold=conf_threshold)

    # If no output_file specified but frames_dir is inside a job directory (.../<job_id>/frames)
    if output_file is None and frames_dir.name == "frames":
        output_file = frames_dir.parent / "analysis" / "detections.json"

    if output_file is not None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

    return detections


def process_job(
    job_id: str,
    data_dir: Union[str, Path] = "data",
    conf_threshold: float = 0.25,
    model_path: Union[str, Path] = "yolo11n.pt",
) -> List[Dict[str, Any]]:
    """
    Run reusable YOLO detection pipeline for any given job ID.
    Reads frames from data/<job_id>/frames and saves detections to data/<job_id>/analysis/detections.json.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        root_data = Path(__file__).resolve().parent.parent / data_dir
        if root_data.exists():
            data_path = root_data

    frames_dir = data_path / job_id / "frames"
    output_file = data_path / job_id / "analysis" / "detections.json"

    return detect_objects(
        frames_dir=frames_dir,
        output_file=output_file,
        conf_threshold=conf_threshold,
        model_path=model_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Cheel Nazar Phase 6.2 - YOLO Object Detection Pipeline")
    parser.add_argument(
        "--job-id",
        type=str,
        default="a27e91ee-c3b1-46f9-be63-d6f3c2a27552",
        help="Job ID to process (default: a27e91ee-c3b1-46f9-be63-d6f3c2a27552)",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Explicit path to frames directory (overrides --job-id)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output path for detections.json",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections (default: 0.25)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO model weights (default: yolo11n.pt)",
    )

    args = parser.parse_args()

    if args.frames_dir:
        frames_path = Path(args.frames_dir)
        output_path = Path(args.output) if args.output else (frames_path.parent / "analysis" / "detections.json")
        print(f"Processing frames from: {frames_path}")
        detections = detect_objects(
            frames_dir=frames_path,
            output_file=output_path,
            conf_threshold=args.conf,
            model_path=args.model,
        )
    else:
        print(f"Processing job ID: {args.job_id}")
        detections = process_job(
            job_id=args.job_id,
            conf_threshold=args.conf,
            model_path=args.model,
        )
        output_path = Path("data") / args.job_id / "analysis" / "detections.json"

    print(f"Detections complete.")
    print(f"Total detections saved: {len(detections)}")
    print(f"Saved to: {output_path}")

    # Summary of detected classes
    class_counts: Dict[str, int] = {}
    for d in detections:
        class_counts[d["label"]] = class_counts.get(d["label"], 0) + 1

    print("Class breakdown:")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  - {label}: {count}")


if __name__ == "__main__":
    main()