from app.services.frame_extractor import extract_frames

video_path = "../uploads/b3e95573-4ed3-407b-b4f9-f4c2c08d86fc_14636688-uhd_3840_2160_30fps.mp4"
output_dir = "../data/frames/b3e95573-4ed3-407b-b4f9-f4c2c08d86fc"

result = extract_frames(video_path, output_dir, fps=2.0)

print("Frame extraction completed!")
print("Video FPS:", result["video_fps"])
print("Total video frames:", result["total_frames"])
print("Saved frames:", result["saved_frames"])