import os

import cv2
import numpy as np


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
