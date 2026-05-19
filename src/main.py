import cv2
import numpy as np
from lane_detection.lane_detection import get_bottom_lane_boundary, get_top_lane_boundary, get_lateral_lane_boundaries, postprocess_boundary_lines
from rectification.lane_rectification import   rectify_bowling_lane
from utils.plot_utils import plot_lane_boundaries, plot_bowling_3d
from utils.utils import load_json
from utils.video_utils import generate_trajectory_video

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
    


    # Transform points to rectified space for visualization
    x_y_trajectory = np.array(list(zip(trajectory_data["observations"]["x"], trajectory_data["observations"]["y"])), dtype=np.float32).reshape(-1, 1, 2)
    f_x_y_trajectory_int = np.array(list(zip(trajectory_data["estimations"]["f"], trajectory_data["estimations"]["x"], trajectory_data["estimations"]["y"])), dtype=np.float32).reshape(-1, 1, 3)
    
    rectified_points = cv2.perspectiveTransform(x_y_trajectory, H)

    rectified_points_int = np.round(rectified_points).astype(np.int32)
    for pt in rectified_points_int:
        cv2.circle(rectified, tuple(pt[0]), 5, (0, 0, 255), -1)
    cv2.imwrite("output/rectified_lane_with_points.png", rectified)


    generate_trajectory_video(
        input_video_path=input_path,
        points = f_x_y_trajectory_int.reshape(-1, 3),  # (frame, x, y)
        output_video_path="output/ball_trajectory_video.mp4")






    
    
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
    