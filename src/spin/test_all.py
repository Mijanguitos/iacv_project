import os
from pathlib import Path
import spin_postprocessing

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

for i in range(9, 17):
    clip = f"clip_{i}"
    print(f"\n--- {clip} ---")
    video_path = f"{PROJECT_ROOT}\\src\\ball_detection\\preprocessing_output\\preprocessed_{clip}"
    original_video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}.mp4"
    json_path = f"{PROJECT_ROOT}\\src\\spin\\optical_flow_output\\optical_flow_{clip}"
    save_path = f"{PROJECT_ROOT}\\src\\spin\\postprocessing_output\\{clip}_postprocessing"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}.json"
    
    try:
        spin_postprocessing.spin_post_processing(json_path, save_path, original_video_path)
    except Exception as e:
        print(f"Error on {clip}: {e}")
