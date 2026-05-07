import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_lane_boundaries(frame, lane_borders, base_dir="debug/final_boundaries"):
    vis = frame.copy()

    # Remove None values if any
    points = [p for p in lane_borders if p is not None]

    if len(points) == 4:
        pts = np.array(points, dtype=np.int32)

        # OpenCV expects shape (N,1,2)
        pts = pts.reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # Optional: also draw points
    for point in lane_borders:
        if point is not None:
            x, y = point
            cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(base_dir, "final_lane_boundaries.png"), vis)



def plot_bowling_3d(lane_corners, ball_trajectory, rvec, tvec, output_path="data/debug/rectification/3d_reconstruction.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the Lane Borders
    # world_pts: [[0, 18.29, 0], [1.054, 18.29, 0], [1.054, 0, 0], [0, 0, 0]]
    lane_x = [p[0] for p in lane_corners] + [lane_corners[0][0]]
    lane_y = [p[1] for p in lane_corners] + [lane_corners[0][1]]
    lane_z = [p[2] for p in lane_corners] + [lane_corners[0][2]]
    ax.plot(lane_x, lane_y, lane_z, color='blue', label='Lane Boundaries')

    # 2. Plot the Camera Position
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color='red', s=100, label='Camera')

    # 3. Plot the Ball Trajectory
    if len(ball_trajectory) > 0:
        ball_pts = np.array(ball_trajectory)
        ax.plot(ball_pts[:, 0], ball_pts[:, 1], ball_pts[:, 2], 
                color='green', marker='o', markersize=4, label='Ball Path')

    # Formatting the Plot
    ax.set_xlabel('Width (X) [m]')
    ax.set_ylabel('Length (Y) [m]')
    ax.set_zlabel('Height (Z) [m]')
    ax.set_title('3D Bowling Analysis')
    ax.legend()
    
    # Set axis limits based on standard lane dimensions [cite: 88]
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 20)
    ax.set_zlim(0, 3) # View from floor to 3 meters up
    
    plt.show()
    # plt.savefig(output_path)
    # plt.close(fig)
    print("asd")
