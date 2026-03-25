import cv2
from lane_detection import get_bottom_lane_boundary, get_top_lane_boundary, parameter_search, get_lateral_lane_boundaries

input_path = "clips/clip_1.mp4"
lane_center_point = [1100, 540]  

def main():
    vid = cv2.VideoCapture(input_path)
    ret, frame = vid.read()
    vid.release()

    template_pin = cv2.imread("./data/templates/template_pin_real.png")
    
    # target_height = 40  # approximate pin height

    # scale = target_height / template_pin.shape[0]

    # template_pin = cv2.resize(
    #     template_pin,
    #     None,
    #     fx=scale,
    #     fy=scale,
    #     interpolation=cv2.INTER_AREA
    # )

    if not ret or frame is None:
        print("Failed to read the first frame from the video")
        return

    # Run a full parameter search (writes debug outputs for each combo)
    # parameter_search(frame)

    # Use the preferred combination in the main script
    best_line = get_bottom_lane_boundary(frame, edge_method="sobel", conv_method="r_g_minus_b")
    print("Best line (sobel + r_g_minus_b):", best_line)
    get_lateral_lane_boundaries(frame, edge_threshold=30, edge_method="sobel", conv_method="r_g_minus_b", direction="vertical"
                                , lane_center=lane_center_point)
    get_top_lane_boundary(frame, template_pin)


if __name__ == "__main__":
    main()
    