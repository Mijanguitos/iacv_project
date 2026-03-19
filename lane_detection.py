


# List of Frame_lane_boundaries objs
import os
import numpy as np

import cv2


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
    # top = get_top_lane_boundary(frame)
    bottom = get_bottom_lane_boundary(frame)
    # left, right = get_lateral_lane_boundaries(frame)
    # return top, bottom, left, right
    return  bottom

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
    direction = "vertical"
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

    left_candidate = None
    right_candidate = None
    center_x = frame.shape[1] / 2.0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            avg_x = (x1 + x2) / 2.0

            # Keep track of closest lines to the image center on each side
            if avg_x < center_x:
                # left side: pick the line with the maximum avg_x (closest to center)
                if left_candidate is None or avg_x > (left_candidate[0] + left_candidate[2]) / 2.0:
                    left_candidate = line[0]
            else:
                # right side: pick the line with the minimum avg_x (closest to center)
                if right_candidate is None or avg_x < (right_candidate[0] + right_candidate[2]) / 2.0:
                    right_candidate = line[0]

            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(LINES_DIR, f'lateral_lines_{conv_method}_{edge_method}.png'), output)

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