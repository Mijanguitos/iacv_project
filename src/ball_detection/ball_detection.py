
from pathlib import Path
import numpy as np
import os

# Custom modules
import preprocessing as preprocessing
import candidate_detection as candidate_detection
import postprocessing as postprocessing
import visualization as visualization

def order_lane_points(points):
    """
    Arranges 4 points in a consistent order: 
    Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    # Convert to a numpy array just in case tuples or lists were passed
    # (Notice some of your clips use () instead of [] at the end!)
    pts = np.array(points)
    
    # Sort the points by their y-coordinate (ascending)
    # The first two have the smallest y (top), the last two have largest y (bottom)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    
    # Sort the top points by their x-coordinate to separate left and right
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    
    # Sort the bottom points by their x-coordinate to separate left and right
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    
    # Return in standard order
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

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

    # Order the lane points so they are always in the specific order
    lane_points = order_lane_points(lane_points)

    # Call preprocessing module to prepara the clip for ball detection    
    preprocessing.video_preprocessing(video_path, preprocessing_path, lane_points)

    # Call detection module for bowling ball candidate detection
    candidate_detection.candidate_detection(preprocessing_path, detection_path)

    # Call post-processing module to get the ball's trajectory
    postprocessing.compute_trajectory(detection_path, postprocessing_path)

    # Call the visualization path to generate the final video and graphs
    visualization.visualization(postprocessing_path, video_path, visualization_path)



if __name__ == "__main__":

    #lane_points = np.array([[872, 684], [1228, 696], [1241, 277], [1146, 273]])    # For clip 1
    #lane_points = np.array([[819, 813], [1308, 819] , [1442, 175], [1254, 175]])   # For clip 2
    #lane_points = np.array([[590, 847], [1061, 855], [1204, 190], (1038, 188)])    # For clip 3
    #lane_points = np.array([[608, 983], [1185, 1016], [1258, 354], [1129, 347]])   # For clip 5
    
    #lane_points = np.array([[607, 985], [1192, 1018], [1258, 349], [1131, 346]])    # For clip 6
    #lane_points = np.array([[608, 983], [1186, 1016], [1259, 350], [1131, 344]])   # For clip 7
    #lane_points = np.array([[325, 1074], [1683, 1071], [1573, 385], (1259, 394)])   # For clip 8

    lane_points = np.array([[390, 861], [1085, 878], [1148, 222],  [979, 221]]) # For clip 9, 10, 11 and 12
    #lane_points = np.array([[266, 1048], [1043, 1048], [1103, 169], [912, 171]]) # For clip 13, 14, 15 and 16

    clip = "clip_12"   # Change this variable to select the clip to be processed
    extension = ".mp4"

    PROJECT_ROOT = Path().resolve()
    print(f"Project Root: {PROJECT_ROOT}")

    video_path = f"data\\clips\\{clip}{extension}"
    ball_path = f"{PROJECT_ROOT}\\src\\ball_detection\\"
    preprocessing_path = f"{ball_path}preprocessing_output\\preprocessed_{clip}"
    detection_path = f"{ball_path}detection_output\\candidates_{clip}"
    postprocessing_path = f"{ball_path}postprocessing_output\\postprocessed_{clip}"
    visualization_path = f"{ball_path}visualization_plotting\\{clip}"

    ball_detection(lane_points, video_path, preprocessing_path, detection_path, postprocessing_path, visualization_path)
