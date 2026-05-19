import cv2
import numpy as np
from lane_detection.lane_detection import get_bottom_lane_boundary, get_top_lane_boundary, get_lateral_lane_boundaries, postprocess_boundary_lines
from rectification.lane_rectification import   rectify_bowling_lane
from utils.plot_utils import plot_lane_boundaries, plot_bowling_3d
from utils.utils import load_json
from utils.video_utils import generate_trajectory_board_image, generate_trajectory_board_video, generate_trajectory_video, generate_trajectory_video_with_board

input_path = "data/clips/clip_1.mp4"
# input_path = "data/clips/clip_2.mp4"
# input_path = "data/clips/clip_3.mp4"
lane_center_point = [1100, 540]  

def main():
    vid = cv2.VideoCapture(input_path)
    ret, frame = vid.read()
    vid.release()

    template_pin = cv2.imread("./data/templates/template_pin_real.png")
    # # target_height = 40  # approximate pin height
    # target_height = 75  # approximate pin height second vid
    # # Vid 3 75 px approx

    if not ret or frame is None:
        print("Failed to read the first frame from the video")
        return
   
    bottom_line = get_bottom_lane_boundary(frame, edge_threshold=30, edge_method="sobel", conv_method="r_g_minus_b")
    print("Best line (sobel + r_g_minus_b):", bottom_line)
    lateral_lines = get_lateral_lane_boundaries(frame, edge_threshold=30, edge_method="sobel", conv_method="r_g_minus_b", direction="vertical"
                                , lane_center=lane_center_point)
    top_line = get_top_lane_boundary(frame, template_pin, mode="bottom")
    lane_borders = postprocess_boundary_lines(bottom_line, lateral_lines, top_line)
    plot_lane_boundaries(frame, lane_borders, base_dir = "debug")
    
    
    # 2D Rectification
    rectified, H = rectify_bowling_lane(
    image=frame,
    src_points=lane_borders,
    pixels_per_meter=120,
    output_path="output/rectified_lane.png"
    )


    # Load ball trajectory 
    # {observations: {x:[], y:[], r:[], f:[]},
    # estimations: {x:[], y:[], r:[], f:[]}} from JSON output of the ball tracking module

    # TODO: Falta bajar el radio para la trayectoria en el lane.
    trajectory_data = load_json("src/ball_detection/postprocessing_output/postprocessed_clip_1.json")
    


    # Transform contact points to rectified space for visualization.
    observation_x = trajectory_data["observations"]["x"]
    observation_y = trajectory_data["observations"]["y"]
    observation_r = trajectory_data["observations"]["r"]
    observation_contact_xy = np.array(
        [(x, y + r) for x, y, r in zip(observation_x, observation_y, observation_r)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    estimation_f = trajectory_data["estimations"]["f"]
    estimation_x = trajectory_data["estimations"]["x"]
    estimation_y = trajectory_data["estimations"]["y"]
    estimation_r = trajectory_data["estimations"]["r"]
    f_x_y_trajectory_int = np.array(
        [(int(frame), float(x), float(y + r)) for frame, x, y, r in zip(estimation_f, estimation_x, estimation_y, estimation_r)],
        dtype=np.float32,
    ).reshape(-1, 1, 3)

    rectified_points = cv2.perspectiveTransform(observation_contact_xy, H)

    rectified_points_int = np.round(rectified_points).astype(np.int32)
    for pt in rectified_points_int:
        cv2.circle(rectified, tuple(pt[0]), 5, (0, 0, 255), -1)
    cv2.imwrite("output/rectified_lane_with_points.png", rectified)

    # Build rectified trajectory points with frame indexes for the board video.
    estimated_contact_xy = np.array(
        [(x, y + r) for x, y, r in zip(estimation_x, estimation_y, estimation_r)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    rectified_estimated_points = cv2.perspectiveTransform(estimated_contact_xy, H).reshape(-1, 2)
    rectified_estimated_points_with_frames = [
        (int(frame), float(x), float(y))
        for frame, (x, y) in zip(estimation_f, rectified_estimated_points)
    ]

    rectified_height, rectified_width = rectified.shape[:2]
    source_board_corners = [
        (0.0, rectified_height - 1.0),
        (rectified_width - 1.0, rectified_height - 1.0),
        (rectified_width - 1.0, 0.0),
        (0.0, 0.0),
    ]

    board_template = cv2.imread("data/templates/board_tracking.jpg")
    if board_template is None:
        raise IOError("Cannot load board template: data/templates/board_tracking.jpg")
    board_height, board_width = board_template.shape[:2]
    destination_board_corners = [
        (0.0, board_height - 1.0),
        (board_width - 1.0, board_height - 1.0),
        (board_width - 1.0, 0.0),
        (0.0, 0.0),
    ]

    generate_trajectory_video_with_board(
        input_video_path=input_path,
        image_points=f_x_y_trajectory_int.reshape(-1, 3),
        board_points=rectified_estimated_points_with_frames,
        board_template_path="data/templates/board_tracking.jpg",
        source_lane_corners=source_board_corners,
        destination_board_corners=destination_board_corners,
        output_video_path="output/combined_trajectory_video.mp4",
        color=(0, 255, 0),
        thickness=2,
        point_radius=4,
        board_margin=20,
    )






    
    
    # Camera parameter estimation 
    # Calculates K, R, and t based on the lane corners [cite: 108, 110]
    # K, rvec, tvec = estimate_camera_parameters(lane_borders, frame.shape)
    # f = find_focal_length_for_height(lane_borders, 1.5, frame.shape)
    # K, rvec, tvec = estimate_camera_parameters_with_vp(lane_borders, frame.shape)

    
    # validate_reprojection(frame, lane_borders, K, rvec, tvec)
    # # --- 3. Ball Tracking & 3D Lifting Loop ---
    # ball_trajectory_3d = [pixel_to_world_3d(lane_center_point, K, rvec, tvec)]  # Start with initial position of the ball (example point)


    # K, rvec, tvec, H_iw = full_calibration_and_lifting(lane_borders, frame.shape)

    # ball_trajectory_3d = [
    # lift_point_to_3d(lane_center_point, H_iw)
    # ]

    # # K= calibrate_K_pure_projective(lane_borders)
    # # ball_trajectory_3d, rvec, tvec = reconstruct_3d_homogeneous(lane_center_point, K, lane_borders)
    # # Standard Lane dimensions for visualization [cite: 88, 139]
    # world_lane_corners = [
    #     [0, 18.29, 0],      # near-left (foul line)
    #     [1.054, 18.29, 0],   # near-right
    #     [1.054, 0, 0],       # far-right (pins)
    #     [0, 0, 0]            # far-left
    # ]


    
    # if ball_trajectory_3d:
    #     plot_bowling_3d(
    #         lane_corners=world_lane_corners, 
    #         ball_trajectory=ball_trajectory_3d, 
    #         rvec=rvec, 
    #         tvec=tvec, 
    #         output_path="output/3d_ball_trajectory.png"
    #     )
    # cv2.circle(frame, (lane_center_point[0], lane_center_point[1]), 10, (255, 0, 0), -1)

    # # Save the result to verify
    # cv2.imwrite("debug/lane_center_visualization.png", frame)


if __name__ == "__main__":
    main()
    