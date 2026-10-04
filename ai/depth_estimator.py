"""
Monocular Depth Estimation Pipeline for Cheel Nazar (Phase 6.4).

Utilizes Depth Anything V2 (Small) to generate relative depth maps
from extracted aerial video frames. Outputs are stored per-job under:
    data/<job_id>/analysis/depth/<frame_filename>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import pipeline


def select_device(preferred_device: Optional[str] = None) -> str:
    """Select compute device (mps, cuda, or cpu)."""
    if preferred_device:
        return preferred_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DepthEstimator:
    """Reusable Depth Anything V2 estimator for Cheel Nazar drone footage."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: Optional[str] = None,
    ):
        self.device = select_device(device)
        self.model_name = model_name
        self._pipe = None

    @property
    def pipe(self):
        """Lazy-loaded transformers depth estimation pipeline."""
        if self._pipe is None:
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=self.device,
            )
        return self._pipe

    def estimate_frame(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Run depth estimation on a single frame.
        Optionally saves the resulting grayscale depth map preserving dimensions.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image frame not found: {image_path}")

        with Image.open(image_path) as raw_img:
            orig_size = raw_img.size
            rgb_img = raw_img.convert("RGB")

        # Run inference via Hugging Face pipeline
        result = self.pipe(rgb_img)
        depth_map: Image.Image = result["depth"]

        # Ensure spatial resolution strictly matches the source frame
        if depth_map.size != orig_size:
            depth_map = depth_map.resize(orig_size, Image.BILINEAR)

        # Ensure grayscale 8-bit depth representation
        if depth_map.mode != "L":
            depth_map = depth_map.convert("L")

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            depth_map.save(out_file)

        return depth_map


def estimate_depth_for_frames(
    frames_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: Optional[str] = None,
    single_frame: bool = False,
    frame_name: Optional[str] = None,
) -> List[Path]:
    """
    Process frames from frames_dir and save depth maps into output_dir.
    Preserves the original frame filename.
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    if output_dir is None and frames_dir.name == "frames":
        output_dir = frames_dir.parent / "analysis" / "depth"

    if output_dir is None:
        raise ValueError("output_dir must be specified if frames_dir is not in standard layout.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if frame_name:
        target_files = [frames_dir / frame_name]
        if not target_files[0].is_file():
            raise FileNotFoundError(f"Specified frame not found: {target_files[0]}")
    else:
        target_files = sorted(
            [f for f in frames_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")]
        )

    if not target_files:
        raise ValueError(f"No JPG frames found in {frames_dir}")

    if single_frame:
        target_files = target_files[:1]

    estimator = DepthEstimator(model_name=model_name, device=device)
    saved_paths: List[Path] = []
    total = len(target_files)

    for idx, frame_file in enumerate(target_files, 1):
        out_file = output_dir / frame_file.name
        estimator.estimate_frame(frame_file, output_path=out_file)
        saved_paths.append(out_file)
        print(f"[{idx}/{total}] Generated depth map for {frame_file.name}")

    return saved_paths


def process_job_depth(
    job_id: str,
    data_dir: Union[str, Path] = "data",
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: Optional[str] = None,
    single_frame: bool = False,
    frame_name: Optional[str] = None,
) -> List[Path]:
    """
    Execute depth estimation for a specific job ID.
    Reads frames from data/<job_id>/frames and writes to data/<job_id>/analysis/depth/.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        root_data = Path(__file__).resolve().parent.parent / data_dir
        if root_data.exists():
            data_path = root_data

    frames_dir = data_path / job_id / "frames"
    output_dir = data_path / job_id / "analysis" / "depth"

    return estimate_depth_for_frames(
        frames_dir=frames_dir,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        single_frame=single_frame,
        frame_name=frame_name,
    )


def main():
    parser = argparse.ArgumentParser(description="Cheel Nazar Phase 6.4 - Monocular Depth Estimation")
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
        "--output-dir",
        type=str,
        default=None,
        help="Explicit path to depth output directory",
    )
    parser.add_argument(
        "--single-frame",
        action="store_true",
        help="Process only the first frame (for verification)",
    )
    parser.add_argument(
        "--frame-name",
        type=str,
        default=None,
        help="Specific frame filename to process (e.g. frame_00000.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face model checkpoint (default: depth-anything/Depth-Anything-V2-Small-hf)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (mps, cuda, cpu)",
    )

    args = parser.parse_args()
    start_time = time.time()

    if args.frames_dir:
        frames_path = Path(args.frames_dir)
        output_path = Path(args.output_dir) if args.output_dir else (frames_path.parent / "analysis" / "depth")
        print(f"Processing depth for frames in: {frames_path}")
        results = estimate_depth_for_frames(
            frames_dir=frames_path,
            output_dir=output_path,
            model_name=args.model,
            device=args.device,
            single_frame=args.single_frame,
            frame_name=args.frame_name,
        )
    else:
        print(f"Processing depth for job ID: {args.job_id}")
        results = process_job_depth(
            job_id=args.job_id,
            model_name=args.model,
            device=args.device,
            single_frame=args.single_frame,
            frame_name=args.frame_name,
        )

    elapsed_time = time.time() - start_time
    avg_per_frame = elapsed_time / len(results) if results else 0.0

    print(f"\n--- Depth Estimation Summary ---")
    print(f"Total frames processed: {len(results)}")
    print(f"Depth maps successfully created: {len(results)}")
    print(f"Total processing time: {elapsed_time:.2f}s")
    print(f"Average time per frame: {avg_per_frame:.2f}s")


if __name__ == "__main__":
    main()
