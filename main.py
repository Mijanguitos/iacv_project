import cv2
from lane_detection import get_bottom_lane_boundary, parameter_search

input_path = "clips/clip_1.mp4"


def main():
    vid = cv2.VideoCapture(input_path)
    ret, frame = vid.read()
    vid.release()

    if not ret or frame is None:
        print("Failed to read the first frame from the video")
        return

    # Run a full parameter search (writes debug outputs for each combo)
    # parameter_search(frame)

    # Use the preferred combination in the main script
    best_line = get_bottom_lane_boundary(frame, edge_method="sobel", conv_method="r_g_minus_b")
    print("Best line (sobel + r_g_minus_b):", best_line)


if __name__ == "__main__":
    main()
    