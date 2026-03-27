import cv2
from lane_detection import get_bottom_lane_boundary, get_top_lane_boundary, get_lateral_lane_boundaries, postprocess_boundary_lines
from lane_rectification import rectify_bowling_lane
from plot_utils import plot_lane_boundaries

input_path = "clips/clip_1.mp4"
# input_path = "clips/clip_2.mp4"
# input_path = "clips/clip_3.mp4"
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
    
    
    rectified, H = rectify_bowling_lane(
    image=frame,
    src_points=lane_borders,
    pixels_per_meter=120,
    output_path="output/rectified_lane.png"
    )



if __name__ == "__main__":
    main()
    