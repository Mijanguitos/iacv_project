import cv2
import yaml
import numpy as np
from pathlib import Path
from lane_detection.lane_detection import (
    get_bottom_lane_boundary,
    get_top_lane_boundary,
    get_lateral_lane_boundaries,
    postprocess_boundary_lines,
)
from rectification.lane_rectification import rectify_bowling_lane
from utils.plot_utils import plot_lane_boundaries, plot_points
from utils.testing_utils import parameter_search, run_all_videos
from utils.utils import load_config, load_json, obtain_corners_from_image
from utils.video_utils import (
    create_3d_bowling_visualization,
    generate_trajectory_video_with_board,
)

import ball_detection.ball_detection
import spin.spin



config = load_config("src/config.yaml")

# SELECTION OF THE CLIP TO PROCESS
CLIP = "clip_1"

def main():
    clip_path = getattr(config.clip_paths, CLIP) 
    print(clip_path)


if __name__ == "__main__":
    main()