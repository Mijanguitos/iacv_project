from pathlib import Path
import numpy as np
import os

# Import your custom modules
import spin.spin_detection as spin_detection
import spin.spin_postprocessing as spin_postprocessing
import spin.spin_visualization as spin_visualization

def ball_spin(trajectory_path: os.PathLike[str],
              video_path: os.PathLike[str],
              optical_flow_path: os.PathLike[str],
              postprocessing_out_path: os.PathLike[str],
              visualization_out_path: os.PathLike[str]) -> None:
    """
    Main function to perform the whole bowling ball spin workflow. 
    It calls the optical flow detection, mathematical postprocessing, 
    and 3D augmented reality visualization.
    
    Args:
        trajectory_path (os.PathLike[str]): Path to the 2D bounding box JSON.
        preprocessed_video_path (os.PathLike[str]): Path to the cropped/CLAHE clip for detection.
        original_video_path (os.PathLike[str]): Path to the raw clip for visualization.
        detection_out_path (os.PathLike[str]): Base path for saving the raw spin JSON.
        postprocessing_out_path (os.PathLike[str]): Base path for saving the smoothed spin JSON.
        visualization_out_path (os.PathLike[str]): Path for saving the final rendered MP4.
    """

    spin_detection.spin_detection(
        trajectory_path=trajectory_path, 
        video_path=video_path, 
        save_path=optical_flow_path
    )

    # Note: spin_detection appends '_spin' to the detection_out_path
    spin_postprocessing.spin_post_processing(
        json_path=optical_flow_path, 
        save_path=postprocessing_out_path, 
        video_path=video_path,
    )

    # Note: spin_postprocessing appends '_postprocessing.json' to its save path
    spin_visualization.render_spin_overlay(
        json_path=f"{postprocessing_out_path}.json", 
        video_path=video_path,
        output_path=visualization_out_path
    )
    

if __name__ == "__main__":
    
    # Define the clip you want to process
    clip = "clip_16"
    extension = ".mp4"
    
    PROJECT_ROOT = Path().resolve()
    print(f"Project Root: {PROJECT_ROOT}")

    # Define Input Paths
    original_video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}.json"

    # Define Output Paths
    spin_dir = f"{PROJECT_ROOT}\\src\\spin"
    
    # We pass the base names; the modules will append '_spin.json' and '_postprocessing.json' automatically
    detection_out_path = f"{spin_dir}\\spin_output\\optical_flow_{clip}"
    postprocessing_out_path = f"{spin_dir}\\postprocessing_output\\{clip}"
    visualization_out_path = f"{spin_dir}\\spin_output\\{clip}_rendered.mp4"

    # Run the pipeline
    ball_spin(
        trajectory_path, 
        original_video_path, 
        detection_out_path, 
        postprocessing_out_path, 
        visualization_out_path
    )