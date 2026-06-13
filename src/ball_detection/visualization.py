
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    f = trajectory["f"]  # Extract frames for the timeline colorscale
    
    markers = (np.array(r) ** 2) * 0.25

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Subtle background line to show the continuous path
    plt.plot(x, y, 
             linestyle='-', color='black', linewidth=1.5, alpha=1, label='Interpolated Path', zorder=1)
    
    # Scatter plot using frames (f) for the color mapping
    scatter = plt.scatter(x, y, 
                          s=markers, 
                          c=f,               # Map colors directly to frame numbers
                          cmap='viridis',    # Apply the colormap
                          edgecolors='white', 
                          linewidths=0.5,
                          alpha=0.8, 
                          label='Detections (size=radius)',
                          zorder=4)
    
    scatter = plt.scatter(x, y, 
                          s=10, 
                          c='black', 
                          alpha=1)

    # Add a colorbar linked to the scatter plot
    cbar = plt.colorbar(scatter, ax=ax, label='Frame Number', pad=0.02)

    # Make the title dynamic based on the 'concept' (e.g., Observations vs Estimations)
    #plt.title(f"Ball Trajectory: {concept.capitalize()}", fontsize=14, pad=15)
    plt.xlabel("X Position", fontweight='bold')
    plt.ylabel("Y Position", fontweight='bold')
    
    ax.invert_yaxis() 
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Keeping 'equal' here is correct since we are plotting physical X vs physical Y
    ax.set_aspect('equal', 'box')
    
    # Optional: Save the plot automatically using the provided save_path
    plt.savefig(f"{save_path}_{concept}_trajectory.png", bbox_inches='tight', dpi=300)
    
    #plt.show()

def comparative_trajectory_plot(trajectory: dict, save_path: os.PathLike[str]):
    """
    Plots Observations and Estimations side-by-side, sharing axes and a unified color ramp.
    """
    obs = trajectory["observations"]
    est = trajectory["estimations"]

    # Calculate global timeline boundaries to ensure the color scale matches perfectly
    vmin = min(min(obs["f"]), min(est["f"]))
    vmax = max(max(obs["f"]), max(est["f"]))

    # 1. Set up a 1x2 grid of subplots that share both axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)    
    # ---------------------------------------------------------
    # Plot 1: Observations (Raw Detections)
    # ---------------------------------------------------------
    obs_markers = (np.array(obs["r"]) ** 2) * 0.25
    

    # Add the background path line to show the continuous spline
    axes[0].plot(obs["x"], obs["y"], linestyle='-', color='black', linewidth=1.5, alpha=1, zorder=1)

    sc1 = axes[0].scatter(obs["x"], obs["y"], 
                          s=obs_markers, 
                          c=obs["f"], 
                          cmap='viridis',
                          vmin=vmin, vmax=vmax,
                          edgecolors='white', 
                          linewidths=0.5,
                          alpha=0.8, 
                          zorder=4)
    
    #axes[0].set_title("Observations (Raw Detections)", fontsize=14, pad=10)
    axes[0].set_xlabel("X Position", fontweight='bold')
    axes[0].set_ylabel("Y Position", fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].set_aspect('equal', 'box')

    # ---------------------------------------------------------
    # Plot 2: Estimations (Smoothed Output)
    # ---------------------------------------------------------
    est_markers = (np.array(est["r"]) ** 2) * 0.25
    
    # Add the background path line to show the continuous spline
    axes[1].plot(est["x"], est["y"], linestyle='-', color='black', linewidth=1.5, alpha=1, zorder=1)
    
    sc2 = axes[1].scatter(est["x"], est["y"], 
                          s=est_markers, 
                          c=est["f"], 
                          cmap='viridis',
                          vmin=vmin, vmax=vmax,
                          edgecolors='white', 
                          linewidths=0.5,
                          alpha=0.8, 
                          zorder=4)
    
    #axes[1].set_title("Estimations (Smoothed Spline)", fontsize=14, pad=10)
    axes[1].set_xlabel("X Position", fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].set_aspect('equal', 'box')

    # ---------------------------------------------------------
    # Shared Formatting & Output
    # ---------------------------------------------------------
    # Because sharey=True, inverting the first axis automatically inverts the second
    axes[0].invert_yaxis() 
    
    # 1. Create dividers for BOTH axes so we can manipulate their widths equally
    divider0 = make_axes_locatable(axes[0])
    divider1 = make_axes_locatable(axes[1])
    
    # 2. Add a "ghost" axis to the left of the first plot and hide it. 
    # This steals the exact same amount of space from axes[0] to keep it symmetrical.
    cax0 = divider0.append_axes("left", size="4%", pad=0.15)
    cax0.axis('off') 
    
    # 3. Add the real colorbar axis to the right of the second plot
    cax1 = divider1.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(sc2, cax=cax1, label='Frame Number') 
    
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.05)

    plt.savefig(f"{save_path}_comparative_trajectory.png", bbox_inches='tight', dpi=300)
    #plt.show()
    

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

    comparative_trajectory_plot(loaded_trajectory, save_path)


if __name__ == "__main__":
    PROJECT_ROOT = Path().resolve()

    clip = "clip_7"
    extension = ".mov"
    video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}"
    save_path = f"{PROJECT_ROOT}\\src\\ball_detection\\visualization_plotting\\{clip}"

    visualization(trajectory_path, video_path, save_path)
