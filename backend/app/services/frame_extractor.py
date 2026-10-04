import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        cap.release()
        raise ValueError("Could not determine video FPS.")

    frame_interval = max(1, int(video_fps / fps))

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            output_path = output_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_number += 1

    cap.release()

    return {
        "total_frames": frame_number,
        "saved_frames": saved_count,
        "video_fps": video_fps,
        "sample_fps": fps
    }