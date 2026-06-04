
from pathlib import Path
import numpy as np
import os

# Custom modules
import preprocessing
import detection
import postprocessing
import visualization


def ball_detection(lane_points: np.ndarray,
                   video_path: os.PathLike[str],
                   preprocessing_path: os.PathLike[str],
                   detection_path: os.PathLike[str],
                   postprocessing_path: os.PathLike[str],
                   visualization_path: os.PathLike[str]) -> None:
    """
    Main function to perform the whole bowling ball detection workflow. It calls the preprocessing
    of the video clip, the candidate detection, postprocessing, and visualization. These scripts
    generate their respective outputs both in video and .json format in the provided paths. 
    Args:
        lane_points (np.ndarray): Array of points that define the lane's polygon.
        video_path (os.PathLike[str]): Path where the initial clip is found
        preprocessing_path (os.PathLike[str]): Path where the preprocessing output will be saved
        detection_path (os.PathLike[str]): Path where the detection output will be saved
        postprocessing_path (os.PathLike[str]): Path where the postprocessing output will be saved
        visualization_path (os.PathLike[str]): Path where the visualization output will be saved
    """

    # Call preprocessing module to prepara the clip for ball detection    
    preprocessing.video_preprocessing(video_path, preprocessing_path, lane_points)

    # Call detection module for bowling ball candidate detection
    detection.candidate_detection(preprocessing_path, detection_path)

    # Call post-processing module to get the ball's trajectory
    postprocessing.compute_trajectory(detection_path, postprocessing_path)

    # Call the visualization path to generate the final video and graphs
    visualization.visualization(postprocessing_path, video_path, visualization_path)



if __name__ == "__main__":

    #lane_points = np.array([[872, 684], [1228, 696], [1241, 277], [1146, 273]]) # For clip 1
    #lane_points = [[819, 813], [1308, 819] , [1442, 175], [1254, 175]] # For clip 2
    #lane_points = np.array([[608, 983], [1186, 1016], [1259, 350], [1131, 344]]) # For clip 7
    lane_points = np.array([[608, 983], [1185, 1016], [1258, 354], [1129, 347]]) # For clip 5
    
    clip = "clip_5"
    
    PROJECT_ROOT = Path().resolve()
    print(f"Project Root: {PROJECT_ROOT}")

    extension = ".mov"
    video_path = f"data\\clips\\{clip}{extension}"
    ball_path = f"{PROJECT_ROOT}\\src\\ball_detection\\"
    preprocessing_path = f"{ball_path}preprocessing_output\\preprocessed_{clip}"
    detection_path = f"{ball_path}detection_output\\candidates_{clip}"
    postprocessing_path = f"{ball_path}postprocessing_output\\postprocessed_{clip}"
    visualization_path = f"{ball_path}visualization_plotting\\{clip}"

    ball_detection(lane_points, video_path, preprocessing_path, detection_path, postprocessing_path, visualization_path)
