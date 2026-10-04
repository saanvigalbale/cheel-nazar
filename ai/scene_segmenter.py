import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


class SceneSegmenter:
    def __init__(self, model_name=MODEL_NAME, device=None):
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def segment_frame(self, image_path, output_path):
        image = Image.open(image_path).convert("RGB")

        original_size = image.size

        max_size = 1024

        scale = min(
            max_size / original_size[0],
            max_size / original_size[1],
            1.0
        )

        resized_size = (
            int(original_size[0] * scale),
            int(original_size[1] * scale)
        )

        small_image = image.resize(resized_size)

        inputs = self.processor(
            images=small_image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        mask_image = Image.fromarray(mask)

        mask_image = mask_image.resize(
            original_size,
            resample=Image.Resampling.NEAREST
        )

        mask = np.array(mask_image).astype(np.uint8)

        Image.fromarray(mask).save(output_path)

        return mask

    def calculate_coverage(self, mask):
        total_pixels = mask.size

        coverage = {}

        for class_id, label in self.id2label.items():
            class_id = int(class_id)
            label = label.strip()

            pixel_count = int(np.sum(mask == class_id))

            if pixel_count > 0:
                coverage[label] = {
                    "pixels": pixel_count,
                    "percentage": round(
                        (pixel_count / total_pixels) * 100,
                        2
                    )
                }

        return coverage

    def process_frames(self, frames_dir, output_dir):
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        results = []

        frame_files = sorted(
            frames_dir.glob("*.jpg")
        )

        if not frame_files:
            raise ValueError(
                f"No JPG frames found in {frames_dir}"
            )

        total_frames = len(frame_files)

        for index, frame_path in enumerate(
            frame_files,
            start=1
        ):
            print(
                f"Processing frame {index}/{total_frames}: "
                f"{frame_path.name}"
            )

            mask_path = (
                output_dir
                / f"{frame_path.stem}_mask.png"
            )

            mask = self.segment_frame(
                frame_path,
                mask_path
            )

            coverage = self.calculate_coverage(mask)

            results.append({
                "frame": frame_path.name,
                "coverage": coverage
            })

        summary_path = output_dir / "summary.json"

        with open(
            summary_path,
            "w"
        ) as file:
            json.dump(
                results,
                file,
                indent=2
            )

        return {
            "frames_processed": len(results),
            "summary_file": str(summary_path),
            "output_dir": str(output_dir)
        }


def process_job(
    job_id,
    data_dir="data",
    device=None
):
    frames_dir = (
        Path(data_dir)
        / job_id
        / "frames"
    )

    output_dir = (
        Path(data_dir)
        / job_id
        / "analysis"
        / "segmentation"
    )

    segmenter = SceneSegmenter(
        device=device
    )

    return segmenter.process_frames(
        frames_dir,
        output_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic scene segmentation for Cheel Nazar"
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="Job ID to process"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory"
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device: mps or cpu"
    )

    args = parser.parse_args()

    result = process_job(
        job_id=args.job_id,
        data_dir=args.data_dir,
        device=args.device
    )

    print("\nSegmentation complete.")
    print(
        json.dumps(
            result,
            indent=2
        )
    )