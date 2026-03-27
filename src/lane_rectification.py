


import cv2
import numpy as np
import os


def rectify_bowling_lane(
    image,
    src_points,
    lane_width_m=1.054,     # 41.5 inches
    lane_length_m=18.29,    # 60 feet
    pixels_per_meter=100,
    output_path="rectified_lane.png"
):
    """
    Rectify a bowling lane using perspective transform.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    src_points : list or np.ndarray
        Four points in the input image corresponding to lane corners.
        Order MUST be:
            [near-left,
             near-right,
             far-right,
             far-left]

    lane_width_m : float
        Real lane width in meters.

    lane_length_m : float
        Real lane length in meters.

    pixels_per_meter : float
        Scale factor controlling output resolution.

    output_path : str
        Where to save rectified image.

    Returns
    -------
    rectified : np.ndarray
        Rectified top-down image.

    H : np.ndarray
        Homography matrix.
    """

    # Convert source points
    src_pts = np.array(src_points, dtype=np.float32)

    # Convert real-world size to pixels
    width_px = int(lane_width_m * pixels_per_meter)
    length_px = int(lane_length_m * pixels_per_meter)

    # Destination rectangle
    dst_pts = np.array([
        [0, length_px],          # near-left
        [width_px, length_px],   # near-right
        [width_px, 0],           # far-right
        [0, 0]                   # far-left
    ], dtype=np.float32)

    # Compute homography
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp perspective
    rectified = cv2.warpPerspective(
        image,
        H,
        (width_px, length_px)
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image
    cv2.imwrite(output_path, rectified)

    return rectified, H


def transform_points(points, H):
    pts = np.array(points, dtype=np.float32).reshape(-1,1,2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1,2)