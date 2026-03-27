# List of Frame_lane_boundaries objs
import os
import numpy as np

import cv2
from sklearn.linear_model import RANSACRegressor

from utils import (
    custom_grayscale,
    line_intersection,
    non_max_suppression,
    crop_by_ratio,
)

# Debug output folders
DEBUG_DIR = "debug"
GRAYSCALE_DIR = os.path.join(DEBUG_DIR, "grayscale")
EDGES_DIR = os.path.join(DEBUG_DIR, "edges")
LINES_DIR = os.path.join(DEBUG_DIR, "lines")
BEST_DIR = os.path.join(DEBUG_DIR, "best_line")
os.makedirs(GRAYSCALE_DIR, exist_ok=True)
os.makedirs(EDGES_DIR, exist_ok=True)
os.makedirs(LINES_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


def get_vid_lane_boundaries(vid):
    lane_boundaries = []
    ret, frame = vid.read()
    while ret:
        frame_lane_boundaries = get_frame_lane_boundaries(frame)
        lane_boundaries.append(frame_lane_boundaries)
        ret, frame = vid.read()
    return lane_boundaries


def get_frame_lane_boundaries(frame):
    # Implementation for detecting lane boundaries in a single frame
    top = get_top_lane_boundary(frame)
    bottom = get_bottom_lane_boundary(
        frame, edge_threshold=30, edge_method="sobel", conv_method="r_g_minus_b"
    )
    # left, right = get_lateral_lane_boundaries(frame)
    # return top, bottom, left, right
    return top, bottom


def preprocess_frame(frame, conv_method="r_g_minus_b", blur_kernel=(5, 5)):
    gray = custom_grayscale(frame, method=conv_method)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    return blurred


def get_bottom_lane_boundary(
    frame,
    edge_method="sobel",
    conv_method="pca",
    edge_threshold=0.3,
    hough_threshold=50,
    hough_min_line_length=100,
    hough_max_line_gap=10,
    slope_threshold=0.3,
    crop_region=[0.6, 1.0, 0.15, 0.85],
):
    # Implementation for detecting the bottom lane boundary

    blurred = preprocess_frame(frame, conv_method=conv_method)
    # crop before edge detection (bottom + center region)
    cropped, top, left = crop_by_ratio(blurred, crop_region)

    edges = detect_edges(
        cropped,
        method=edge_method,
        conv_method=conv_method,
        edge_threshold=edge_threshold,
        direction="horizontal",
        debug_prefix="bottom_",
    )

    # hough transform
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=hough_max_line_gap,
    )

    output, _, _ = crop_by_ratio(frame.copy(), crop_region)

    best_candidate = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Get the first line within the slope threshold
            if x2 == x1:
                line_slope = float("inf")
            else:
                line_slope = (y2 - y1) / (x2 - x1)

            if abs(line_slope) > slope_threshold:
                # rejected as red
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
                continue

            if best_candidate is None:
                best_candidate = line[0]

            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(
        os.path.join(LINES_DIR, f"lines_{conv_method}_{edge_method}.png"), output
    )

    filename = f"bottom_best_candidate_line_" f"{conv_method}_{edge_method}.png"

    if best_candidate is not None:
        return save_candidate_line(
            frame, best_candidate, crop_region, [top, left], (0, 255, 0), filename
        )
    return None


def save_candidate_line(frame, line, crop_region, top_left_offset, color, filename):
    vis, _, _ = crop_by_ratio(frame.copy(), crop_region)
    x1, y1, x2, y2 = get_true_coords(line, crop_region=top_left_offset)
    cv2.line(vis, (x1, y1), (x2, y2), color, 3)
    cv2.imwrite(os.path.join(BEST_DIR, filename), vis)
    return [x1, y1, x2, y2]


def get_top_lane_boundary(
    frame,
    template,
    min_points=3,
    crop_region=[0.0, 0.4, 0.0, 1.0],
    mode="mid" # "mid" for line connecting midpoints of pins, "bottom" for connecting the bottom of the pins 
):

    top_region, _, _ = crop_by_ratio(frame, crop_region)

    midpoints = detect_pin_midpoints_template(
        top_region,
        template,
        scales=[
            0.8,
            0.9,
            1.0,
            1.1,
            1.5,
            1.75,
            2.0,
        ],  # multi-scale to handle perspective
        # threshold=0.85, first vid
        threshold=0.75,
        debug_dir="debug_template",
        mode=mode
    )

    if len(midpoints) < min_points:
        print(f"Insufficient midpoints found: {len(midpoints)}")
        return None

    # Fit line using midpoints with RANSAC outlier removal
    points = np.array(midpoints).astype(np.float32)
    x_coords = points[:, 0].reshape(-1, 1)
    y_coords = points[:, 1]
    ransac = RANSACRegressor(
        min_samples=2, residual_threshold=15, max_trials=100, random_state=42
    )
    ransac.fit(x_coords, y_coords)
    inlier_mask = ransac.inlier_mask_
    inliers = points[inlier_mask]
    outliers = points[~inlier_mask]
    if len(outliers) > 0:
        print(f"Outliers removed: {len(outliers)}/{len(points)}")

    # Refit line using only inliers for better accuracy
    x_inliers = inliers[:, 0]
    y_inliers = inliers[:, 1]

    coefficients = np.polyfit(x_inliers, y_inliers, 1)
    m, b = coefficients
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(m * x_min + b)
    y_max = int(m * x_max + b)

    # Viz
    vis_image = top_region.copy()
    cv2.line(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    for pt in inliers.astype(int):
        cv2.circle(vis_image, tuple(pt), 5, (255, 0, 0), -1)
    for pt in outliers.astype(int):
        cv2.circle(vis_image, tuple(pt), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(BEST_DIR, "top_boundary_fitted_line.png"), vis_image)
    cv2.imwrite(os.path.join(LINES_DIR, "top_boundary_midpoints.png"), vis_image)
    print(f"Top boundary detected with {len(points)} midpoints")
    print(f"Line equation: y = {m:.4f}x + {b:.4f}")

    return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)


def get_lateral_lane_boundaries(
    frame,
    edge_method="sobel",
    conv_method="pca",
    edge_threshold=None,
    hough_threshold=50,
    hough_min_line_length=100,
    hough_max_line_gap=10,
    direction="vertical",
    lane_center=None,
):
    blurred = preprocess_frame(frame, conv_method=conv_method)

    # no cropping for lateral edges
    edges = detect_edges(
        blurred,
        method=edge_method,
        conv_method=conv_method,
        edge_threshold=edge_threshold,
        direction=direction,
        debug_prefix="lateral_",
    )
    # hough transform
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=hough_max_line_gap,
    )

    output = frame.copy()

    if lane_center is None:
        raise ValueError("lane_center must be provided for lateral boundary detection")
    center_x, center_y = lane_center[0], lane_center[1]

    left_candidate = None
    right_candidate = None
    min_left_dist = float("inf")
    min_right_dist = float("inf")
    pos_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            m = (y2 - y1) / (x2 - x1)
            if abs(m) < 1:
                continue
            dist, intersection_x = calculate_distance_to_center(
                line[0], center_x, center_y
            )

            # Determine side based on avg_x
            if intersection_x < center_x:
                # left side: pick the line with the smallest distance to center
                if dist < min_left_dist:
                    min_left_dist = dist
                    left_candidate = line[0]
            else:
                # right side: pick the line with the smallest distance to center
                if dist < min_right_dist:
                    min_right_dist = dist
                    right_candidate = line[0]

            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pos_lines += 1

    cv2.circle(
        output, (int(center_x), int(center_y)), 5, (255, 0, 0), -1
    )  # Mark the center point
    cv2.imwrite(
        os.path.join(LINES_DIR, f"lateral_lines_{conv_method}_{edge_method}.png"),
        output,
    )
    print(os.path.join(LINES_DIR, f"lateral_lines_{conv_method}_{edge_method}.png"))
    # Save best left/right boundary candidates
    candidates = {"left": left_candidate, "right": right_candidate}
    for side, line in candidates.items():
        if line is None:
            continue
        vis = frame.copy()
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        filename = f"lateral_best_{side}_line_" f"{conv_method}_{edge_method}.png"
        cv2.imwrite(os.path.join(BEST_DIR, filename),vis)

    return left_candidate, right_candidate


def detect_pin_midpoints_template(
    frame,
    template,
    scales=(0.8, 0.9, 1.0, 1.1, 1.2),
    threshold=0.6,
    nms_threshold=0.1,
    debug_dir=None,
    mode="mid" # "mid" for line connecting midpoints of pins, "bottom" for connecting the bottom of the pins
):
    """
    Detect bowling pins using multi-scale template matching.

    Args:
        frame: Input BGR image
        template: Pin template image
        scales: Template scales to test
        threshold: Matching confidence threshold
        nms_threshold: Overlap threshold for NMS
        debug_dir: Folder to save debug images

    Returns:
        midpoints: list of (x, y)
    """

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    boxes = []
    scores = []

    for scale in scales:
        resized_template = cv2.resize(
            gray_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        th, tw = resized_template.shape
        if th > gray_frame.shape[0] or tw > gray_frame.shape[1]:
            continue

        # Template matching
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            x, y = pt
            boxes.append([x, y, x + tw, y + th])
            scores.append(result[y, x])

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Apply Non-Maximum Suppression
    indices = non_max_suppression(boxes, scores, nms_threshold)
    final_boxes = boxes[indices]
    midpoints = []

    vis = frame.copy()
    for box in final_boxes:
        x1, y1, x2, y2 = box
        if mode == "bottom":
            mx = int((x1 + x2) / 2)
            my = y2  # Use the bottom of the pin
        else:
            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)
        midpoints.append((mx, my))
        # Debug drawing
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (mx, my), 3, (0, 0, 255), -1)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "template_matches.png"), vis)
    print(f"Detected pins: {len(midpoints)}")

    return midpoints


def get_true_coords(line, crop_region):
    x1, y1, x2, y2 = line
    top, left = crop_region
    true_x1 = x1 + left
    true_y1 = y1 + top
    true_x2 = x2 + left
    true_y2 = y2 + top
    return true_x1, true_y1, true_x2, true_y2


def calculate_distance_to_center(line, center_x, center_y):
    """
    Calculate the horizontal distance from the center point to the intersection
    of the line with the horizontal line at center_y.
    """
    x1, y1, x2, y2 = line

    # If the line is vertical (x1 == x2), intersection x is x1
    if x1 == x2:
        intersection_x = x1
    elif y1 == y2:
        # If the line is horizontal, use the center_y directly
        intersection_x = center_x
    else:
        # Calculate slope
        m = (y2 - y1) / (x2 - x1)
        # Ignore nearly horizontal lines for lateral boundary detection
        if abs(m) < 1:
            return [float("inf"), 0]
        intersection_x = x1 + (center_y - y1) / m

    # Horizontal distance
    distance = abs(center_x - intersection_x)
    return [distance, intersection_x]


def detect_edges(
    blurred,
    method="sobel",
    conv_method="pca",
    edge_threshold=None,
    direction="horizontal",
    debug_prefix="",
):
    """Run edge detection on the provided image.

    Cropping (if needed) must be performed by the caller.
    """

    if method == "sobel":
        if (
            direction == "horizontal"
        ):  # For horizontal edges (lane markings), detect vertical gradients
            sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        else:  # For vertical edges (lateral boundaries), detect horizontal gradients
            sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel = np.absolute(sobel)
        sobel = sobel / sobel.max() * 255  # normalize to 0-255
        edges = np.uint8(sobel)
    elif method == "canny":
        edges = cv2.Canny(blurred, 30, 90)
    elif method == "laplacian":
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = laplacian / laplacian.max() * 255  # normalize to 0-255
        edges = np.uint8(laplacian)
    else:
        raise ValueError("Invalid edge detection method")
    if edge_threshold is not None:
        _, edges = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(
        os.path.join(EDGES_DIR, f"{debug_prefix}edges_{method}_{conv_method}.jpg"),
        edges,
    )
    return edges


def postprocess_boundary_lines(bottom_line, lateral_lines, top_line):
    """
    Returns the 4 lane corner points:
    [bottom-left, bottom-right,
     top-right, top-left]
    """
    if len(lateral_lines) < 2:
        return None
    left_line, right_line = lateral_lines
    bottom_left = line_intersection(bottom_line, left_line)
    bottom_right = line_intersection(bottom_line, right_line)
    top_right = line_intersection(top_line, right_line)
    top_left = line_intersection(top_line, left_line)

    corners = [bottom_left, bottom_right, top_right, top_left]
    if any(p is None for p in corners):
        return None
    return corners
