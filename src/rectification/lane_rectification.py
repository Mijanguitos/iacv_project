


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



def estimate_camera_parameters(lane_borders, image_shape):
    """
    Estimates K, R, and t without a calibration checkerboard.
    lane_borders order: [bottom-left, bottom-right, top-right, top-left]
    """
    h, w = image_shape[:2]
    cx, cy = w / 2, h / 2  # Assume principal point is image center [cite: 84]

    # 1. Estimate Focal Length (f) using Vanishing Point [cite: 81]
    # Lateral lines: (top_left to bottom_left) and (top_right to bottom_right)
    # We find where these lines intersect in the image plane
    def get_line(p1, p2):
        return np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])

    line_l = get_line(lane_borders[3], lane_borders[0])
    line_r = get_line(lane_borders[2], lane_borders[1])
    
    # Intersection of parallel lines in 2D is the Vanishing Point (V) [cite: 82]
    v_point = np.cross(line_l, line_r)
    v_u, v_v = v_point[0]/v_point[2], v_point[1]/v_point[2]

    # Simplified focal length estimation [cite: 84]
    # f^2 = -(v_u - cx)*(v_u - cx) - (v_v - cy)*(v_v - cy) is a common heuristic
    # but for bowling, a safe FOV-based guess or solving via PnP is more robust.
    # Let's use a standard FOV guess for a fixed camera (approx 900-1200 pixels)
    f = 1000 
    
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)

    # 2. Define 3D World Coordinates (Meters) [cite: 42, 88]
    # Standard Lane: 1.054m wide, 18.29m long. Foul line is Y=0.
    world_pts = np.array([
        [0, 18.29, 0],      # bottom-left (near foul line)
        [1.054, 18.29, 0],   # bottom-right
        [1.054, 0, 0],       # top-right (at pins)
        [0, 0, 0]            # top-left
    ], dtype=np.float32)

    # 3. Solve PnP for Extrinsics (R, t) [cite: 58, 86]
    image_pts = np.array(lane_borders, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, None)

    return K, rvec, tvec



def pixel_to_world_3d(ball_pixel, K, rvec, tvec):
    """
    Converts 2D pixel to 3D world coordinate assuming the ball center
    is 0.108m above the floor.
    """
    R, _ = cv2.Rodrigues(rvec)
    ball_radius = 0.108 # Standard bowling ball radius [cite: 73]
    
    # Point in camera coords: inv(K) * [u, v, 1]
    uv_homo = np.array([ball_pixel[0], ball_pixel[1], 1.0]).reshape(3, 1)
    inv_k = np.linalg.inv(K)
    ray_cam = inv_k @ uv_homo
    
    # Transform ray to world coords
    inv_R = R.T
    ray_world = inv_R @ ray_cam
    t_world = -inv_R @ tvec
    
    # Solve for scale 's' where Z_world = ball_radius [cite: 67, 95]
    # Equation: P_world = t_world + s * ray_world
    # Z_world = t_world_z + s * ray_world_z
    s = (ball_radius - t_world[2, 0]) / ray_world[2, 0]
    
    world_3d = t_world + s * ray_world
    return world_3d.flatten() # Returns [X, Y, Z]