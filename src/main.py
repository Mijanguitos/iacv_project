import cv2
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

config = load_config("src/config.yaml")


def main():
    vid = cv2.VideoCapture(config.paths.input_clip_path)
    ret, frame = vid.read()
    vid.release()

    template_pin = cv2.imread(config.paths.template_pin_path)
    # # target_height = 40  # approximate pin height
    # target_height = 75  # approximate pin height second vid
    # # Vid 3 75 px approx

    if not ret or frame is None:
        print("Failed to read the first frame from the video")
        return

    ## Visualization of different methods for lane boundary detection
    run_all_videos()
    
    ## 1. Lane Detection
    # 1.1 Detect bottom boundary
    bottom_line = get_bottom_lane_boundary(
        frame,
        edge_threshold=config.lane_detection.bottom.edge_threshold,
        edge_method=config.lane_detection.bottom.edge_method,
        conv_method=config.lane_detection.bottom.conv_method,
    )

    # 1.2 Detect lateral boundaries
    # Determine lane center for this video. Prefer per-clip override in config,
    # falling back to the global `lane_center_point`.
    video_stem = Path(config.paths.input_clip_path).stem
    lane_center = None
    if hasattr(config.points, "lane_center_points"):
        lane_center = getattr(config.points.lane_center_points, video_stem, None)
    if lane_center is None:
        lane_center = config.points.lane_center_point

    lateral_lines = get_lateral_lane_boundaries(
        frame,
        edge_threshold=config.lane_detection.lateral.edge_threshold,
        edge_method=config.lane_detection.lateral.edge_method,
        conv_method=config.lane_detection.lateral.conv_method,
        direction=config.lane_detection.lateral.direction,
        lane_center=lane_center,
    )

    # 1.3 Detect top boundary
    top_line = get_top_lane_boundary(
        frame, template_pin, mode=config.lane_detection.top.mode
    )

    # 1.4 Post-process boundaries to get final lane corners
    lane_borders = postprocess_boundary_lines(bottom_line, lateral_lines, top_line)
    plot_lane_boundaries(frame, lane_borders, base_dir=config.paths.debug_dir_path)

    ## 2. Rectification
    # 2.1 Obtain homography and rectified lane image
    rectified, H = rectify_bowling_lane(
        image=frame,
        src_points=lane_borders,
        pixels_per_meter=100,
        output_path=config.paths.output_rectified_lane_path,
    )

    ## 3. Obtain image plane ball trajectory
    # {observations: {x:[], y:[], r:[], f:[]},
    # estimations: {x:[], y:[], r:[], f:[]}} from JSON output of the ball tracking module
    if config.misc.calculate_ball_trajectory:
        # TODO: Implement ball trajectory calculation with interpolation
        raise NotImplementedError(
            "Ball trajectory calculation not implemented in this script. Please run the ball detection module first to generate the trajectory data."
        )

    trajectory_data = load_json(config.paths.ball_trajectory_data_path)

    ## 4. Transform trajectory points to rectified space
    # Interpolated values of x, y (ball center) and r for each frame
    x_inter, y_inter, r_inter, f = [
        trajectory_data["estimations"][key]
        for key in ["x", "y", "r", "f"]
    ]

    # Obtain contact points by adding radius to y-coordinate (assuming ball touches lane at its bottom point)
    inter_points_contact = np.array(
        [(x, y + r) for x, y, r in zip(x_inter, y_inter, r_inter)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    # Build image-space contact points (frame, x, y) for the original video overlay
    inter_points_contact_with_frames = np.array(
        [
            (int(frame), float(x), float(y + r))
            for frame, x, y, r in zip(f, x_inter, y_inter, r_inter)
        ],
        dtype=np.float32,
    ).reshape(-1, 3)

    rectified_inter_points_contact = cv2.perspectiveTransform(inter_points_contact, H)
    rectified_inter_points_contact_with_frames = [
        (int(frame), float(x), float(y))
        for frame, (x, y) in zip(f, rectified_inter_points_contact.reshape(-1, 2))
    ]

    # rectified_points_int = np.round(rectified_points).astype(np.int32)
    rectified_inter_points_contact_int = np.round(
        rectified_inter_points_contact
    ).astype(np.int32)
    plot_points(
        rectified_inter_points_contact_int,
        rectified,
        output_path=config.paths.output_rectified_lane_with_trajectory_path,
    )

    # Use helper to obtain image corners (accepts image array or path)
    source_board_corners = obtain_corners_from_image(rectified)

    destination_board_corners, board_template = obtain_corners_from_image(
        config.paths.board_template_path, return_image=True
    )

    ## 5. Generate video overlaying ball trajectory on original video and rectified lane
    # TODO: Add spin to video

    generate_trajectory_video_with_board(
        input_video_path=config.paths.input_clip_path,
        image_points=inter_points_contact_with_frames,
        board_points=rectified_inter_points_contact_with_frames,
        board_template_path=config.paths.board_template_path,
        source_lane_corners=source_board_corners,
        destination_board_corners=destination_board_corners,
        output_video_path=config.paths.output_combined_trajectory_video_path,
        color=(0, 0, 255),
        thickness=config.plotting.point_thickness,
        point_radius=config.plotting.point_radius,
        board_margin=config.plotting.board_margin,
    )



    # 3D Reconstruction and Visualization

    rectified_contact_points = rectified_inter_points_contact.reshape(-1, 2)

    trajectory_3d = []
    ball_radius_m = 0.10915  # Bowling ball radius in meters
    for i, (frame_id, (x_metric, y_metric)) in enumerate(zip(f, rectified_contact_points)):
        r = ball_radius_m  # radius aligned by index

        # Lift into 3D (lane plane is Z=0)
        X = float(x_metric)
        Y = float(y_metric)
        Z = float(r)

        trajectory_3d.append((frame_id, X, Y, Z))

    
    create_3d_bowling_visualization(trajectory_3d, lane_width=1.066, lane_length=19.16)


if __name__ == "__main__":
    main()
