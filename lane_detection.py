


# List of Frame_lane_boundaries objs
import os
import numpy as np

import cv2
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

# Debug output folders
DEBUG_DIR = "debug"
GRAYSCALE_DIR = os.path.join(DEBUG_DIR, "grayscale")
EDGES_DIR = os.path.join(DEBUG_DIR, "edges")
LINES_DIR = os.path.join(DEBUG_DIR, "lines")
BEST_DIR = os.path.join(DEBUG_DIR, "best_line")



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
    bottom = get_bottom_lane_boundary(frame)
    # left, right = get_lateral_lane_boundaries(frame)
    # return top, bottom, left, right
    return top, bottom

# def get_top_lane_boundary(
#     frame,
#     red_threshold=150,
#     min_points=5,
# ):
#     """
#     Detect the top lane boundary using red channel extraction and overdetermined linear interpolation.
    
#     Extracts red markings from the top part of the frame (bowling pin markings),
#     finds the coordinates of red points, and fits a line using least squares interpolation.
    
#     Args:
#         frame: Input image frame
#         red_threshold: Threshold value for red channel (0-255)
#         min_points: Minimum number of points required to fit a line
    
#     Returns:
#         Line endpoints as (x1, y1, x2, y2) or None if insufficient points
#     """
#     # Ensure debug folders exist
#     os.makedirs(GRAYSCALE_DIR, exist_ok=True)
#     os.makedirs(EDGES_DIR, exist_ok=True)
#     os.makedirs(LINES_DIR, exist_ok=True)
#     os.makedirs(BEST_DIR, exist_ok=True)
    
#         # Convert BGR → HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Focus on top region
#     top_region = hsv[:int(hsv.shape[0] * 0.3), :]

#     # Red hue ranges (two ranges!)
#     lower_red1 = np.array([0, 120, 70])
#     upper_red1 = np.array([10, 255, 255])

#     lower_red2 = np.array([170, 120, 70])
#     upper_red2 = np.array([180, 255, 255])

#     # Create masks
#     mask1 = cv2.inRange(top_region, lower_red1, upper_red1)
#     mask2 = cv2.inRange(top_region, lower_red2, upper_red2)

#     red_mask = cv2.bitwise_or(mask1, mask2)

#     # Optional cleanup
#     kernel = np.ones((3,3), np.uint8)
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
#     # Save debug image
#     cv2.imwrite(os.path.join(EDGES_DIR, f"red_channel_top.jpg"), top_region)
#     cv2.imwrite(os.path.join(EDGES_DIR, f"red_mask_top.jpg"), red_mask)
    
#     # Find coordinates of red points
#     points = cv2.findNonZero(red_mask)
    
#     if points is None or len(points) < min_points:
#         print(f"Insufficient red points found: {len(points) if points is not None else 0}")
#         return None
    
#     # Extract x, y coordinates
#     points = points.reshape(-1, 2).astype(np.float32)
#     x_coords = points[:, 0]
#     y_coords = points[:, 1]
    
#     # Perform overdetermined linear interpolation using least squares
#     # Fit: y = mx + b
#     coefficients = np.polyfit(x_coords, y_coords, 1)
#     m, b = coefficients  # slope, intercept
    
#     # Calculate line endpoints for visualization
#     x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
#     y_min = int(m * x_min + b)
#     y_max = int(m * x_max + b)
    
#     # Create visualization image
#     vis_image = frame[:int(frame.shape[0] * 0.3), :].copy()
#     cv2.line(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    
#     # Draw the detected red points
#     for pt in points.astype(int):
#         cv2.circle(vis_image, tuple(pt), 2, (255, 0, 0), -1)
    
#     cv2.imwrite(os.path.join(BEST_DIR, f"top_boundary_fitted_line.png"), vis_image)
#     cv2.imwrite(os.path.join(LINES_DIR, f"top_boundary_red_points.png"), vis_image)
    
#     print(f"Top boundary detected with {len(points)} points")
#     print(f"Line equation: y = {m:.4f}x + {b:.4f}")
    
#     # Return line endpoints adjusted to full frame coordinates
#     # Since we worked with the top region, we need to adjust y coordinates
#     return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)

import os
import cv2
import numpy as np


def get_top_lane_boundary(
    frame,
    template,
    min_points=3,
):
    """
    Detect the top lane boundary using SIFT pin detection
    and linear interpolation of detected pin midpoints.

    Args:
        frame: Input BGR frame
        template: Bowling pin template image
        min_points: Minimum midpoints required

    Returns:
        Line endpoints as (x1, y1, x2, y2) or None
    """

    # Ensure debug folders exist
    os.makedirs(GRAYSCALE_DIR, exist_ok=True)
    os.makedirs(EDGES_DIR, exist_ok=True)
    os.makedirs(LINES_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    # --------------------------------------------------
    # Restrict to top region (VERY IMPORTANT)
    # --------------------------------------------------

    top_h = int(frame.shape[0] * 0.4)

    top_region = frame[:top_h, :]

    # --------------------------------------------------
    # Call SIFT midpoint detection
    # --------------------------------------------------

    midpoints = detect_pin_midpoints_template(
        frame,
        template,
        scales=[0.8, 0.9, 1.0, 1.1],
        # threshold=0.85, first vid
        threshold=0.75,
        debug_dir="debug_template"
        )

    if len(midpoints) < min_points:

        print(
            f"Insufficient midpoints found: {len(midpoints)}"
        )

        return None

    # --------------------------------------------------
    # Fit line using midpoints with RANSAC outlier removal
    # --------------------------------------------------

    points = np.array(midpoints).astype(np.float32)

    x_coords = points[:, 0].reshape(-1, 1)
    y_coords = points[:, 1]

    # RANSAC for robust line fitting (conservative: only removes clear outliers)
    ransac = RANSACRegressor(
        min_samples=2,
        residual_threshold=15,
        max_trials=100,
        random_state=42
    )
    ransac.fit(x_coords, y_coords)

    # Get inlier mask
    inlier_mask = ransac.inlier_mask_
    inliers = points[inlier_mask]
    outliers = points[~inlier_mask]

    if len(outliers) > 0:
        print(f"Outliers removed: {len(outliers)}/{len(points)}")

    # Refit line using only inliers for better accuracy
    x_inliers = inliers[:, 0]
    y_inliers = inliers[:, 1]

    coefficients = np.polyfit(
        x_inliers,
        y_inliers,
        1
    )

    m, b = coefficients

    # --------------------------------------------------
    # Create line endpoints
    # --------------------------------------------------

    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))

    y_min = int(m * x_min + b)
    y_max = int(m * x_max + b)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    vis_image = top_region.copy()

    # Draw fitted line
    cv2.line(
        vis_image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        3
    )

    # Draw inlier midpoints (blue)
    for pt in inliers.astype(int):
        cv2.circle(
            vis_image,
            tuple(pt),
            5,
            (255, 0, 0),
            -1
        )

    # Draw outlier midpoints (red)
    for pt in outliers.astype(int):
        cv2.circle(
            vis_image,
            tuple(pt),
            5,
            (0, 0, 255),
            -1
        )

    cv2.imwrite(
        os.path.join(
            BEST_DIR,
            "top_boundary_fitted_line.png"
        ),
        vis_image
    )

    cv2.imwrite(
        os.path.join(
            LINES_DIR,
            "top_boundary_midpoints.png"
        ),
        vis_image
    )

    print(
        f"Top boundary detected with {len(points)} midpoints"
    )

    print(
        f"Line equation: y = {m:.4f}x + {b:.4f}"
    )

    # --------------------------------------------------
    # Return endpoints
    # --------------------------------------------------

    return np.array(
        [x_min, y_min, x_max, y_max],
        dtype=np.int32
    )




# def detect_pin_midpoints_sift(
#     frame,
#     template,
#     ratio_thresh=0.75,
#     eps=40,
#     min_samples=4,
#     debug=False
# ):
#     """
#     Detect bowling pins using SIFT and return midpoints of detected pins.

#     Args:
#         frame: Input BGR frame
#         template: Image of a single bowling pin
#         ratio_thresh: Lowe ratio test threshold
#         eps: DBSCAN clustering radius
#         min_samples: DBSCAN minimum cluster size
#         debug: Whether to visualize results

#     Returns:
#         midpoints: np.array of shape (N,2)
#     """

#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(
#         os.path.join(
#             LINES_DIR,
#             "gray_frame.png"
#         ),
#         gray_frame
#     )
#     cv2.imwrite(
#         os.path.join(
#             LINES_DIR,
#             "gray_template.png"
#         ),
#         gray_template
#     )
#     # Create SIFT detector
#     sift = cv2.SIFT_create()

#     # Detect keypoints
#     kp1, des1 = sift.detectAndCompute(
#         gray_template,
#         None
#     )

#     kp2, des2 = sift.detectAndCompute(
#         gray_frame,
#         None
#     )

#     if des2 is None:
#         return np.empty((0, 2))

#     # Match descriptors
#     bf = cv2.BFMatcher(cv2.NORM_L2)

#     matches = bf.knnMatch(
#         des1,
#         des2,
#         k=2
#     )
#     all_matches = [m for m, n in matches]
#     debug_img = cv2.drawMatches(
#     template,  # template image
#     kp1,       # template keypoints
#     frame,     # full frame image
#     kp2,       # frame keypoints
#     all_matches,  # list of matches after ratio test
#     None,      # output image
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#     cv2.imwrite(
#         os.path.join(
#             LINES_DIR,
#             "sift_matches.png"
#         ),
#         debug_img
#     )

#     # Lowe ratio test
#     good_matches = []

#     for m, n in matches:
#         if m.distance < ratio_thresh * n.distance:
#             good_matches.append(m)

#     if len(good_matches) == 0:
#         return np.empty((0, 2))

#     # Extract matched points
#     pts = np.float32([
#         kp2[m.trainIdx].pt
#         for m in good_matches
#     ])

#     # Cluster points into pins
#     clustering = DBSCAN(
#         eps=eps,
#         min_samples=min_samples
#     ).fit(pts)

#     labels = clustering.labels_

#     midpoints = []

#     for label in set(labels):

#         if label == -1:
#             continue  # noise

#         cluster_pts = pts[labels == label]

#         # Midpoint = centroid
#         midpoint = np.mean(
#             cluster_pts,
#             axis=0
#         )

#         midpoints.append(midpoint)

#     midpoints = np.array(midpoints)

#     # Debug visualization
#     if debug:

#         vis = frame.copy()

#         # Draw cluster points
#         for pt in pts.astype(int):
#             cv2.circle(
#                 vis,
#                 tuple(pt),
#                 2,
#                 (255, 0, 0),
#                 -1
#             )

#         # Draw midpoints
#         for mp in midpoints.astype(int):
#             cv2.circle(
#                 vis,
#                 tuple(mp),
#                 6,
#                 (0, 255, 0),
#                 -1
#             )

#         cv2.imwrite(
#             "debug_sift_midpoints.png",
#             vis
#         )

#     return midpoints





def detect_pin_midpoints_template(
    frame,
    template,
    scales=(0.8, 0.9, 1.0, 1.1, 1.2),
    threshold=0.6,
    nms_threshold=0.1,
    debug_dir=None
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
    
    # # Optional: use edges (often improves robustness)
    # gray_frame = cv2.Canny(gray_frame, 40, 120)
    # gray_template = cv2.Canny(gray_template, 40, 120)

    # # Save Canny-processed images for debugging
    # if debug_dir is not None:
    #     os.makedirs(debug_dir, exist_ok=True)
    #     cv2.imwrite(os.path.join(debug_dir, "canny_frame.png"), gray_frame)
    #     cv2.imwrite(os.path.join(debug_dir, "canny_template.png"), gray_template)

    boxes = []
    scores = []

    for scale in scales:

        # Resize template
        resized_template = cv2.resize(
            gray_template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        th, tw = resized_template.shape

        if th > gray_frame.shape[0] or tw > gray_frame.shape[1]:
            continue

        # Template matching
        result = cv2.matchTemplate(
            gray_frame,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )

        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):

            x, y = pt

            boxes.append([
                x,
                y,
                x + tw,
                y + th
            ])

            scores.append(result[y, x])

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Apply Non-Maximum Suppression
    indices = non_max_suppression(
        boxes,
        scores,
        nms_threshold
    )

    final_boxes = boxes[indices]

    midpoints = []

    vis = frame.copy()

    for box in final_boxes:

        x1, y1, x2, y2 = box

        mx = int((x1 + x2) / 2)
        my = int((y1 + y2) / 2)

        midpoints.append((mx, my))

        # Debug drawing
        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            vis,
            (mx, my),
            3,
            (0, 0, 255),
            -1
        )

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(
            os.path.join(debug_dir, "template_matches.png"),
            vis
        )

    print(f"Detected pins: {len(midpoints)}")

    return midpoints



def non_max_suppression(boxes, scores, threshold):
    """
    Apply Non-Maximum Suppression.
    """

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:

        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / areas[order[1:]]

        inds = np.where(overlap <= threshold)[0]

        order = order[inds + 1]

    return keep




def get_bottom_lane_boundary(
    frame,
    edge_method="sobel",
    conv_method="pca",
    edge_threshold=None,
    hough_threshold=50,
    hough_min_line_length=100,
    hough_max_line_gap=10,
):
    # Implementation for detecting the bottom lane boundary
    # Ensure debug folders exist
    os.makedirs(GRAYSCALE_DIR, exist_ok=True)
    os.makedirs(EDGES_DIR, exist_ok=True)
    os.makedirs(LINES_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    #grayscale
    # gray = custom_grayscale(frame, method="default")
    gray = custom_grayscale(frame, method=conv_method)
    # gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = gray_clahe.apply(gray)
    # gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(GRAYSCALE_DIR, f"blurred_{conv_method}.jpg"), blurred)

    # crop before edge detection (bottom + center region)
    cropped = blurred[int(blurred.shape[0] * 0.6):, int(blurred.shape[1] * 0.15):int(blurred.shape[1] * 0.85)]
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

    output = frame[int(blurred.shape[0] * 0.6):, int(blurred.shape[1] * 0.15):int(blurred.shape[1] * 0.85)].copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(LINES_DIR, f'lines_{conv_method}_{edge_method}.png'), output)
    
    # Select the first line as the best candidate and plot it in another image
    if lines is not None and len(lines) > 0:
        best_candidate = lines[0][0]  # First line is the best candidate
        best_candidate_image = frame[int(blurred.shape[0]*0.6):, int(blurred.shape[1]*0.15):int(blurred.shape[1]*0.85)].copy()
        x1, y1, x2, y2 = best_candidate
        cv2.line(best_candidate_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(BEST_DIR, f'bottom_best_candidate_line_{conv_method}_{edge_method}.png'), best_candidate_image)
        return best_candidate
    else:
        return None

def get_lateral_lane_boundaries(frame,
    edge_method="sobel",
    conv_method="pca",
    edge_threshold=None,
    hough_threshold=50,
    hough_min_line_length=100,
    hough_max_line_gap=10,
    direction = "vertical",
    lane_center = None
):
    # Implementation for detecting the bottom lane boundary
    # Ensure debug folders exist
    os.makedirs(GRAYSCALE_DIR, exist_ok=True)
    os.makedirs(EDGES_DIR, exist_ok=True)
    os.makedirs(LINES_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    #grayscale
    # gray = custom_grayscale(frame, method="default")
    gray = custom_grayscale(frame, method=conv_method)
    # gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = gray_clahe.apply(gray)
    #gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(GRAYSCALE_DIR, f"blurred_lateral_{conv_method}.jpg"), blurred)
    
    # no cropping for lateral edges
    edges = detect_edges(
        blurred,
        method=edge_method,
        conv_method=conv_method,
        edge_threshold=edge_threshold,
        direction=direction,
        debug_prefix="lateral_",
    )
    #hough transform
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=hough_max_line_gap,
    )

    output = frame.copy()

    # center_x = frame.shape[1] / 2.0
    if lane_center is None:
        raise ValueError("lane_center must be provided for lateral boundary detection")
    center_x, center_y = lane_center[0], lane_center[1]

    left_candidate = None
    right_candidate = None
    min_left_dist = float('inf')
    min_right_dist = float('inf')
    pos_lines=0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            m = (y2 - y1) / (x2 - x1)
            if abs(m)<1:
                continue
            dist, intersection_x = calculate_distance_to_center(line[0], center_x, center_y)
            

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
            pos_lines+=1
    print(pos_lines)
    cv2.circle(output, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)  # Mark the center point
    cv2.imwrite(os.path.join(LINES_DIR, f'lateral_lines_{conv_method}_{edge_method}.png'), output)
    print(os.path.join(LINES_DIR, f'lateral_lines_{conv_method}_{edge_method}.png'))
    # Save best left/right boundary candidates
    if left_candidate is not None:
        left_img = frame.copy()
        x1, y1, x2, y2 = left_candidate
        cv2.line(left_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(BEST_DIR, f'lateral_best_left_line_{conv_method}_{edge_method}.png'), left_img)

    if right_candidate is not None:
        right_img = frame.copy()
        x1, y1, x2, y2 = right_candidate
        cv2.line(right_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(BEST_DIR, f'lateral_best_right_line_{conv_method}_{edge_method}.png'), right_img)

    return left_candidate, right_candidate


def parameter_search(
    frame,
    grayscale_methods=None,
    edge_methods=None,
):
    """Search for the best grayscale+edge combination (saves debug images)."""

    if grayscale_methods is None:
        grayscale_methods = [
            "default",
            "lightness",
            "luminosity",
            "r_g_minus_b",
            "pca",
        ]
    if edge_methods is None:
        edge_methods = ["sobel", "canny", "laplacian"]

    results = {}
    for gray_method in grayscale_methods:
        for edge_method in edge_methods:
            print(f"Searching: grayscale={gray_method}, edge={edge_method}")

            # Reduce noise for Sobel/Laplacian
            if edge_method in ("sobel", "laplacian"):
                edge_threshold = 30
                hough_threshold = 150
                hough_min_line_length = 150
                hough_max_line_gap = 8
            else:
                edge_threshold = None
                hough_threshold = 50
                hough_min_line_length = 100
                hough_max_line_gap = 10

            best_line = get_bottom_lane_boundary(
                frame,
                edge_method=edge_method,
                conv_method=gray_method,
                edge_threshold=edge_threshold,
                hough_threshold=hough_threshold,
                hough_min_line_length=hough_min_line_length,
                hough_max_line_gap=hough_max_line_gap,
            )
            results[(gray_method, edge_method)] = best_line

    return results


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
        intersection_x = center_x  # Horizontal line doesn't affect lateral distance
    else:
        # Calculate slope
        m = (y2 - y1) / (x2 - x1)
        if abs(m)<1:
            return [float('inf'), 0]  # Ignore nearly horizontal lines for lateral boundary detection
        # Solve for x at y = center_y
        # y - y1 = m(x - x1)
        # center_y - y1 = m(intersection_x - x1)
        intersection_x = x1 + (center_y - y1) / m
    
    # Horizontal distance
    distance = abs(center_x - intersection_x)
    return [distance, intersection_x]



def custom_grayscale(frame, method="default"):
    if method == "default":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif method == "lightness":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:,:,1]
    elif method == "luminosity":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:,:,0]
    elif method == "r_g_minus_b":
        r, g, b = cv2.split(frame)
        return cv2.subtract(cv2.add(r, g), b)
    elif method == "pca":
        pixels = frame.reshape(-1, 3)
        mean = np.mean(pixels, axis=0)
        centered = pixels - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]
        grayscale = np.dot(centered, principal_component)
        grayscale = ((grayscale - grayscale.min()) / (grayscale.max() - grayscale.min()) * 255)
        return grayscale.reshape(frame.shape[:2]).astype(np.uint8)
    else:
        raise ValueError("Invalid grayscale method")

def detect_edges(blurred, method="sobel", conv_method="pca", edge_threshold=None, direction="horizontal", debug_prefix=""):
    """Run edge detection on the provided image.

    Cropping (if needed) must be performed by the caller.
    """

    if method == "sobel":
        if direction == "horizontal":  # For horizontal edges (lane markings), detect vertical gradients
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

    cv2.imwrite(os.path.join(EDGES_DIR, f"{debug_prefix}edges_{method}_{conv_method}.jpg"), edges)
    return edges