from pathlib import Path
import numpy as np
import os

# Import your custom modules
import spin_detection
import spin_postprocessing
import spin_visualization

def ball_spin(trajectory_path: os.PathLike[str],
              preprocessed_video_path: os.PathLike[str],
              original_video_path: os.PathLike[str],
              detection_out_path: os.PathLike[str],
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

    print("--- Starting Spin Detection ---")
    spin_detection.spin_detection(
        trajectory_path=trajectory_path, 
        video_path=original_video_path, 
        save_path=detection_out_path
    )

    print("--- Starting Spin Post-Processing ---")
    # Note: spin_detection appends '_spin' to the detection_out_path
    spin_postprocessing.spin_post_processing(
        json_path=f"{detection_out_path}_spin", 
        save_path=postprocessing_out_path, 
        video_path=original_video_path,
    )

    print("--- Starting 3D Visualization Render ---")
    # Note: spin_postprocessing appends '_postprocessing.json' to its save path
    spin_visualization.render_spin_overlay(
        json_path=f"{postprocessing_out_path}_postprocessing.json", 
        video_path=original_video_path, 
        output_path=visualization_out_path
    )
    
    print("--- Workflow Complete! ---")


if __name__ == "__main__":
    
    # Define the clip you want to process
    clip = "clip_2"
    extension = ".mp4"
    
    PROJECT_ROOT = Path().resolve()
    print(f"Project Root: {PROJECT_ROOT}")

    # Define Input Paths
    preprocessed_video_path = f"{PROJECT_ROOT}\\src\\ball_detection\\preprocessing_output\\preprocessed_{clip}"
    original_video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}.json"

    # Define Output Paths
    spin_dir = f"{PROJECT_ROOT}\\src\\spin"
    
    # We pass the base names; the modules will append '_spin.json' and '_postprocessing.json' automatically
    detection_out_path = f"{spin_dir}\\spin_output\\{clip}"
    postprocessing_out_path = f"{spin_dir}\\postprocessing_output\\{clip}"
    visualization_out_path = f"{spin_dir}\\spin_output\\{clip}_rendered.mp4"

    # Run the pipeline
    ball_spin(
        trajectory_path, 
        preprocessed_video_path, 
        original_video_path, 
        detection_out_path, 
        postprocessing_out_path, 
        visualization_out_path
    )