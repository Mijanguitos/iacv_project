
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os

def trajectory_plotting(concept: str, 
                        trajectory: dict, 
                        save_path: os.PathLike[str]):
    
    x = trajectory["x"]
    y = trajectory["y"]
    r = trajectory["r"]
    markers = (np.array(r) ** 2) * 0.25

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    plt.plot(x, y, 
             linestyle='--', color='gray', linewidth=1, alpha=0.5, label='Interpolated Path')
    
    
    scatter = plt.scatter(x, y, 
                          s=markers, 
                          c='teal', 
                          edgecolors='white', 
                          linewidths=0.5,
                          alpha=0.8, 
                          label='Detections (size=radius)',
                          zorder=4)
    
    plt.plot(x, y, color='teal', linewidth=1.5, alpha=0.3)

    plt.title("Ball Trajectory (Marker Size Proportional to Radius)", fontsize=14)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    ax.invert_yaxis() 
    plt.grid(True, linestyle=':', alpha=0.6)
    #plt.legend(loc='upper right')
    ax.set_aspect('equal', 'box')
    
    plt.show()
    

def frame_detection(concept: str, 
                    trajectory: dict, 
                    save_path: os.PathLike[str],
                    video_path: os.PathLike[str]):
            
    x = trajectory["x"]
    y = trajectory["y"]
    r = trajectory["r"]
    f = trajectory["f"]

    # Load the video clip and get total number of frames and frame rate
    CAP = cv2.VideoCapture(f"{video_path}")                  # Class for video captured from the clip's file   
    n_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))   # Clip's total number of frames
    fps = CAP.get(cv2.CAP_PROP_FPS)                     # Clip's frame rate
    width  = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
    height = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)   # float `height`
    
    # Video writer class
    output = cv2.VideoWriter(f"{save_path}_{concept}.mp4", -1, fps, (int(width), int(height)))

    # Loops to perform preprocessing
    count = 0
    while True:
        ret, frame = CAP.read()

        if not ret:
            break
        
        if count in f:
            i = f.index(count)
            cv2.circle(frame, (int(x[i]), int(y[i])), int(r[i]), (0, 255, 0), 4)
            cv2.circle(frame, (int(x[i]), int(y[i])), 1, (0, 128, 255), -1)

        # Write preprocessed frame
        output.write(frame)
        count += 1

    output.release()

def visualization(trajectory_path: os.PathLike[str],
                  video_path: os.PathLike[str],
                  save_path: os.PathLike[str]):
    
    with open(f"{trajectory_path}.json", "r") as file:
        loaded_trajectory = json.load(file)
 

    for key in loaded_trajectory.keys():
        trajectory_plotting(key, loaded_trajectory[key], save_path)
        frame_detection(key, loaded_trajectory[key], save_path, video_path)


if __name__ == "__main__":
    PROJECT_ROOT = Path().resolve()

    clip = "clip_7"
    extension = ".mov"
    video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}"
    save_path = f"{PROJECT_ROOT}\\src\\ball_detection\\visualization_plotting\\{clip}"

    visualization(trajectory_path, video_path, save_path)
