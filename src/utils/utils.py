import cv2
import numpy as np


def custom_grayscale(frame, method="default"):
    if method == "default":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif method == "lightness":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 1]
    elif method == "luminosity":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
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
        grayscale = (
            (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min()) * 255
        )
        return grayscale.reshape(frame.shape[:2]).astype(np.uint8)
    else:
        raise ValueError("Invalid grayscale method")


def line_intersection(line1, line2):
    """
    Compute intersection point between two lines.
    Lines are given as (x1, y1, x2, y2).
    Returns (x, y) or None if parallel.
    """

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-6:
        return None  # Parallel lines

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom

    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (int(px), int(py))


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


def crop_by_ratio(image, crop_region):
    """
    Crop using normalized region:
    [top, bottom, left, right]
    """

    h, w = image.shape[:2]

    top = int(h * crop_region[0])
    bottom = int(h * crop_region[1])
    left = int(w * crop_region[2])
    right = int(w * crop_region[3])

    cropped = image[top:bottom, left:right]

    return cropped, top, left
