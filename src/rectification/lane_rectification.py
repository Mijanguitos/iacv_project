


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


def estimate_camera_parameters_with_vp(lane_borders, image_shape):
    """
    Refines K, R, and t by calculating the focal length from the 
    lane's vanishing point.
    lane_borders: [bottom-left, bottom-right, top-right, top-left]
    """
    h, w = image_shape[:2]
    # Principal point assumed at the image center [cite: 84, 183]
    cx, cy = w / 2, h / 2 

    # --- Step 1: Find Vanishing Point (V) [cite: 182] ---
    # Lane lines connect top corners to bottom corners [cite: 108]
    def get_line(p1, p2):
        return np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])

    line_left = get_line(lane_borders[3], lane_borders[0])
    line_right = get_line(lane_borders[2], lane_borders[1])
    
    # Intersection of these lines is the Vanishing Point [cite: 82, 83]
    v_point = np.cross(line_left, line_right)
    if v_point[2] == 0: # Avoid division by zero for parallel 2D lines
        v_u, v_v = cx, -1e6 
    else:
        v_u, v_v = v_point[0]/v_point[2], v_point[1]/v_point[2]

    # --- Step 2: Estimate Focal Length (f) [cite: 183] ---
    # For a camera looking down a lane, f relates to the distance between 
    # the principal point and the vanishing point. 
    # A common geometric estimation for a single VP on the Y-axis is:
    f = np.sqrt(abs((v_u - cx)**2 + (v_v - cy)**2))
    # f = 230
    
    # Construct refined Intrinsic Matrix [cite: 85]
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)

    # --- Step 3: Solve PnP for Extrinsics [cite: 184] ---
    # Standard world coordinates (meters) [cite: 88]
    world_pts = np.array([
        [0, 18.29, 0],      # bottom-left
        [1.054, 18.29, 0],   # bottom-right
        [1.054, 0, 0],       # top-right
        [0, 0, 0]            # top-left
    ], dtype=np.float32)

    image_pts = np.array(lane_borders, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, None)

    print(f"Refined Focal Length: {f:.2f}")
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





# def find_focal_length_for_height(lane_borders, target_height, image_shape):
#     """
#     Iterates through possible focal lengths to find the one that 
#     places the camera at the specified target_height (in meters).
#     """
#     h, w = image_shape[:2]
#     cx, cy = w / 2, h / 2
#     world_pts = np.array([
#         [0, 18.29, 0], [1.054, 18.29, 0], 
#         [1.054, 0, 0], [0, 0, 0]
#     ], dtype=np.float32)
#     image_pts = np.array(lane_borders, dtype=np.float32)

#     best_f = 1000
#     min_error = float('inf')

#     # Search through a range of typical focal lengths (e.g., 200 to 3000)
#     for f in range(200, 3000, 10):
#         K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
#         _, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, None)
        
#         # Calculate camera position: C = -R^T * t
#         R, _ = cv2.Rodrigues(rvec)
#         cam_pos = -R.T @ tvec
#         current_height = cam_pos[2][0]

#         error = abs(current_height - target_height)
#         if error < min_error:
#             min_error = error
#             best_f = f
            
#     print(f"Optimal Focal Length found: {best_f} for target height {target_height}m, error = {min_error:.2f}m")
#     return best_f





# def validate_reprojection(frame, lane_borders, K, rvec, tvec):
#     """
#     Projects 3D world lane corners back onto the 2D image to check calibration accuracy.
#     """
#     # Define the same 3D world points used in solvePnP
#     world_pts = np.array([
#         [0, 18.29, 0],      # near-left (foul line)
#         [1.054, 18.29, 0],   # near-right
#         [1.054, 0, 0],       # far-right (pins)
#         [0, 0, 0]            # far-left
#     ], dtype=np.float32)

#     # Project 3D points to 2D image plane
#     reprojected_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, K, None)
#     reprojected_pts = reprojected_pts.reshape(-1, 2).astype(int)

#     vis_frame = frame.copy()

#     # Draw the original detected corners in Green
#     for pt in lane_borders:
#         cv2.circle(vis_frame, tuple(pt), 7, (0, 255, 0), -1)

#     # Draw the reprojected corners in Red
#     for pt in reprojected_pts:
#         cv2.circle(vis_frame, tuple(pt), 4, (0, 0, 255), -1)

#     # Draw lines connecting reprojected points to show the lane shape
#     cv2.polylines(vis_frame, [reprojected_pts], isClosed=True, color=(0, 0, 255), thickness=2)

#     cv2.imwrite("debug/reprojection_validation.png", vis_frame)
#     print("Reprojection validation image saved to debug/reprojection_validation.png")















# def estimate_camera_with_vp(lane_borders, image_shape):
#     """
#     Computes K, rvec, and tvec using the vanishing point to calibrate focal length.
#     lane_borders order: [bottom-left, bottom-right, top-right, top-left]
#     """
#     h, w = image_shape[:2]
#     cx, cy = w / 2, h / 2 

#     # 1. Compute Vanishing Point (VP) from lateral lane lines
#     def get_line_eq(p1, p2):
#         return np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])

#     line_left = get_line_eq(lane_borders[3], lane_borders[0])   # top-left to bottom-left
#     line_right = get_line_eq(lane_borders[2], lane_borders[1])  # top-right to bottom-right
    
#     vp_homo = np.cross(line_left, line_right)
#     if abs(vp_homo[2]) < 1e-6:
#         v_u, v_v = cx, -1e6 
#     else:
#         v_u = vp_homo[0] / vp_homo[2]
#         v_v = vp_homo[1] / vp_homo[2]

#     # 2. Geometric Focal Length (f) estimation
#     f = np.sqrt(abs((v_u - cx)**2 + (v_v - cy)**2))
    
#     K = np.array([[f, 0, cx],
#                   [0, f, cy],
#                   [0, 0, 1]], dtype=np.float32)

#     # 3. Define the physical 3D world points (in meters)
#     # Standard Lane size: 1.054m wide, 18.29m long. Pins are at Y=0.
#     world_pts = np.array([
#         [0, 18.29, 0],      # bottom-left (near foul line)
#         [1.054, 18.29, 0],   # bottom-right
#         [1.054, 0, 0],       # top-right (at pins)
#         [0, 0, 0]            # top-left
#     ], dtype=np.float32)

#     image_pts = np.array(lane_borders, dtype=np.float32)
    
#     # 4. Solve PnP to locate camera
#     _, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, None)

#     return K, rvec, tvec


# def reconstruct_ball_3d(ball_pixel, K, rvec, tvec, ball_radius=0.108):
#     """
#     Transforms a 2D ball pixel coordinate (u, v) into a 3D world point (X, Y, Z).
#     Assumes the ball center rolls at Z = ball_radius.
#     """
#     u, v = ball_pixel[0], ball_pixel[1]
    
#     # Compute camera rotation matrix R
#     R, _ = cv2.Rodrigues(rvec)
    
#     # Step 1: Compute local viewing ray vector d in camera frame
#     inv_K = np.linalg.inv(K)
#     d_cam = inv_K @ np.array([u, v, 1.0]).reshape(3, 1)
    
#     # Step 2: Transform the ray direction and translation to world coordinates
#     # r = R^T * d_cam
#     r_world = R.T @ d_cam
    
#     # c_trans = -R^T * tvec (Camera position in world coords)
#     c_world = -R.T @ tvec
    
#     r_x, r_y, r_z = r_world[0, 0], r_world[1, 0], r_world[2, 0]
#     c_x, c_y, c_z = c_world[0, 0], c_world[1, 0], c_world[2, 0]
    
#     # Step 3: Solve for scale factor 's' using the floor constraint (Z_world = ball_radius)
#     # Z_world = s * r_z + c_z -> s = (ball_radius - c_z) / r_z
#     if abs(r_z) < 1e-6:
#         return None # Avoid division by zero if ray is parallel to the floor
        
#     s = (ball_radius - c_z) / r_z
    
#     # Step 4: Reconstruct 3D position
#     X_w = s * r_x + c_x
#     Y_w = s * r_y + c_y
#     Z_w = ball_radius
    
#     return np.array([X_w, Y_w, Z_w])























# def calibrate_K_pure_projective(lane_borders):
#     """
#     Calibrates the intrinsic matrix K using zero-guess linear equations.
#     Solves for the Image of the Absolute Conic (IAC) using:
#       1) The orthogonal circular point mappings from the ground plane homography
#       2) The lane vanishing point orthogonality constraint
      
#     lane_borders order: [bottom-left, bottom-right, top-right, top-left]
#     """
#     # --- Step 1: Compute Ground Plane Homography ---
#     # Define physical metric coordinates on the Z=0 plane (meters)
#     # The origin (0,0) is at the top-left pin deck corner.
#     world_pts = np.array([
#         [0, 18.29],       # bottom-left (foul line)
#         [1.054, 18.29],   # bottom-right
#         [1.054, 0],       # top-right (pins)
#         [0, 0]            # top-left
#     ], dtype=np.float32)
    
#     image_pts = np.array(lane_borders, dtype=np.float32)
    
#     # H maps points on the Z=0 plane to the image plane
#     H, _ = cv2.findHomography(world_pts, image_pts)
#     h1 = H[:, 0]
#     h2 = H[:, 1]
    
#     # --- Step 2: Compute Orthogonal Vanishing Points ---
#     # Convert points to homogeneous lines to find the longitudinal vanishing point (Y-axis direction)
#     def get_line_eq(p1, p2):
#         return np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])
        
#     line_left = get_line_eq(lane_borders[3], lane_borders[0])   # top-left to bottom-left
#     line_right = get_line_eq(lane_borders[2], lane_borders[1])  # top-right to bottom-right
    
#     # Longitudinal Vanishing Point (v1)
#     v1_homo = np.cross(line_left, line_right)
#     v1 = v1_homo / v1_homo[2] if abs(v1_homo[2]) >= 1e-6 else v1_homo

#     # Transverse Vanishing Point (v2) - represents the orthogonal X-axis direction of the lane.
#     # Parallel lines in this direction are the foul line (bottom) and pin deck (top).
#     line_bottom = get_line_eq(lane_borders[0], lane_borders[1]) # bottom-left to bottom-right
#     line_top = get_line_eq(lane_borders[3], lane_borders[2])    # top-left to top-right
    
#     v2_homo = np.cross(line_bottom, line_top)
#     v2 = v2_homo / v2_homo[2] if abs(v2_homo[2]) >= 1e-6 else v2_homo

#     # --- Step 3: Setup System of Linear Equations A * w = 0 ---
#     # We solve for the symmetric conic vector: w = [w1, w2, w3, w4]^T
#     # where omega = [ w1,  0, w2;
#     #                  0, w1, w3;
#     #                 w2, w3, w4 ]
#     A = []
    
#     # Equation 1 (Orthogonality from circular points): h1^T * omega * h2 = 0
#     eq1 = [
#         h1[0]*h2[0] + h1[1]*h2[1],             # coeff of w1
#         h1[0]*h2[2] + h1[2]*h2[0],             # coeff of w2
#         h1[1]*h2[2] + h1[2]*h2[1],             # coeff of w3
#         h1[2]*h2[2]                            # coeff of w4
#     ]
#     A.append(eq1)
    
#     # Equation 2 (Scaling from circular points): h1^T * omega * h1 - h2^T * omega * h2 = 0
#     eq2 = [
#         (h1[0]**2 + h1[1]**2) - (h2[0]**2 + h2[1]**2),  # coeff of w1
#         2 * (h1[0]*h1[2] - h2[0]*h2[2]),                 # coeff of w2
#         2 * (h1[1]*h1[2] - h2[1]*h2[2]),                 # coeff of w3
#         (h1[2]**2 - h2[2]**2)                            # coeff of w4
#     ]
#     A.append(eq2)
    
#     # Equation 3 (Orthogonal Vanishing Points): v1^T * omega * v2 = 0
#     eq3 = [
#         v1[0]*v2[0] + v1[1]*v2[1],             # coeff of w1
#         v1[0]*v2[2] + v1[2]*v2[0],             # coeff of w2
#         v1[1]*v2[2] + v1[2]*v2[1],             # coeff of w3
#         v1[2]*v2[2]                            # coeff of w4
#     ]
#     A.append(eq3)
    
#     # Solve A * w = 0 via Singular Value Decomposition
#     A = np.array(A, dtype=np.float32)
#     _, _, Vh = np.linalg.svd(A)
#     w_vec = Vh[-1, :]  # Last row of Vh corresponds to the null space generator
    
#     w1, w2, w3, w4 = w_vec[0], w_vec[1], w_vec[2], w_vec[3]
    
#     # Keep standard positive sign convention for camera metrics
#     if w1 < 0:
#         w1, w2, w3, w4 = -w1, -w2, -w3, -w4
        
#     # --- Step 4: Extract Intrinsic Parameters Analytically ---
#     c_u = -w2 / w1
#     c_v = -w3 / w1
#     f = np.sqrt(abs((w4 - (c_u**2 + c_v**2) * w1) / w1))
    
#     K = np.array([[f, 0, c_u],
#                   [0, f, c_v],
#                   [0, 0, 1.0]], dtype=np.float32)
    
#     print("\n--- Projective Geometric Calibration Successful ---")
#     print(f"Computed Focal Length (f): {f:.2f} pixels")
#     print(f"Computed Principal Point (cx, cy): ({c_u:.2f}, {c_v:.2f})")
#     return K


# def reconstruct_3d_homogeneous(ball_pixel, K, lane_borders, ball_radius=0.108):
#     """
#     Performs 3D trajectory reconstruction using homogeneous coordinates, 
#     the camera projection matrix P, and projective line-plane intersections in P^3.
#     """
#     u, v = ball_pixel[0], ball_pixel[1]
    
#     # Define physical 3D points
#     world_pts = np.array([
#         [0, 18.29, 0], [1.054, 18.29, 0], [1.054, 0, 0], [0, 0, 0]
#     ], dtype=np.float32)
#     image_pts = np.array(lane_borders, dtype=np.float32)
    
#     # Compute pose analytically via solvePnP using the calibrated K
#     _, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, None)
    
#     R, _ = cv2.Rodrigues(rvec)
#     P = K @ np.hstack((R, tvec.reshape(3, 1)))  # 3x4 Homogeneous Projection Matrix
    
#     # 1. Compute homogeneous Camera Center C (Null space of P)
#     _, _, Vh = np.linalg.svd(P)
#     C_homo = Vh[-1, :]
#     C_homo /= C_homo[3]  # Normalize w component to 1
    
#     # 2. Define the flat plane constraint: Z = ball_radius
#     # Homogeneous plane vector: pi = [a, b, c, d]^T -> 0*X + 0*Y + 1*Z - ball_radius = 0
#     pi_floor = np.array([0.0, 0.0, 1.0, -ball_radius])
    
#     # 3. Projective back-projection ray using the pseudo-inverse of P
#     P_pseudo = np.linalg.pinv(P)
#     X_ray_base = P_pseudo @ np.array([u, v, 1.0])
#     X_ray_base /= X_ray_base[3]  # Normalize w component to 1
    
#     # 4. Find intersection factor 'k' such that: pi^T * (X_base + k * C) = 0
#     numerator = np.dot(pi_floor, X_ray_base)
#     denominator = np.dot(pi_floor, C_homo)
    
#     if abs(denominator) < 1e-6:
#          return None  # Ray runs parallel to the lane floor
         
#     k = -numerator / denominator
    
#     # 5. Compute the final 3D homogeneous point and convert back to Euclidean (m)
#     X_homo_final = X_ray_base + k * C_homo
#     X_euclidean = X_homo_final[:3] / X_homo_final[3]
    
#     return [X_euclidean], rvec, tvec















# # Claude

# import numpy as np
# import cv2

# def compute_lane_homography(lane_borders):
#     """
#     lane_borders = [bottom_left, bottom_right, top_right, top_left]
#     each point = (x_px, y_px)
#     Returns H: maps image pixels -> world coords (meters, on Z=0 plane)
#     """
#     # World coords (X, Y in meters, Z=0 plane)
#     world_pts = np.array([
#         [0,     18.29],
#         [1.054, 18.29],
#         [1.054, 0    ],
#         [0,     0    ]
#     ], dtype=np.float64)

#     img_pts = np.array(lane_borders, dtype=np.float64)  # shape (4,2)

#     H, mask = cv2.findHomography(img_pts, world_pts)
#     return H



# def calibrate_K_from_homography(H_world_to_image, image_shape):
#     """
#     Recover K from a single homography using the Zhang method constraints.
#     Assumes: zero skew, square pixels, principal point at image center.
    
#     H_world_to_image: maps world plane -> image (inverse of your H above)
#     """
#     h, w = image_shape[:2]
#     cx, cy = w / 2.0, h / 2.0

#     # Work with H mapping world->image: H_wi = inv(H_iw)
#     H = H_world_to_image  # 3x3

#     h1 = H[:, 0]  # first column
#     h2 = H[:, 1]  # second column

#     # Constraints from r1.T @ r2 = 0  and  |r1| = |r2|
#     # With K = [[f,0,cx],[0,f,cy],[0,0,1]]  (square pixels, zero skew)
#     # We solve for f using the two constraints.

#     # Let h1_n = K^{-1} h1, h2_n = K^{-1} h2
#     # Constraint: h1_n . h2_n = 0
#     # With K^{-1} = [[1/f,0,-cx/f],[0,1/f,-cy/f],[0,0,1]]

#     def Kinv_h(h, f):
#         return np.array([
#             (h[0] - cx) / f,
#             (h[1] - cy) / f,
#             h[2]
#         ])

#     # Solve: dot(K^{-1}h1, K^{-1}h2) = 0  for f
#     # Expanding: (h1[0]-cx)(h2[0]-cx)/f^2 + (h1[1]-cy)(h2[1]-cy)/f^2 + h1[2]*h2[2] = 0
#     A = (h1[0]-cx)*(h2[0]-cx) + (h1[1]-cy)*(h2[1]-cy)
#     B = h1[2]*h2[2]
#     # A/f^2 + B = 0  =>  f^2 = -A/B
#     f_sq = -A / B
#     if f_sq <= 0:
#         print(f"Warning: f^2={f_sq:.2f} negative, using fallback")
#         # Fallback: use |h1|=|h2| constraint instead
#         C = (h1[0]-cx)**2 + (h1[1]-cy)**2 - (h2[0]-cx)**2 - (h2[1]-cy)**2
#         D = h2[2]**2 - h1[2]**2
#         f_sq = C / D if D != 0 else w**2
#         f_sq = max(f_sq, (w*0.5)**2)  # sanity bound

#     f = np.sqrt(abs(f_sq))
#     print(f"Estimated focal length: {f:.1f} px")

#     K = np.array([
#         [f,  0,  cx],
#         [0,  f,  cy],
#         [0,  0,  1 ]
#     ])
#     return K




# def recover_extrinsics(H_world_to_image, K):
#     """
#     H maps world plane (Z=0) -> image pixels.
#     Returns rvec, tvec (OpenCV convention).
#     """
#     K_inv = np.linalg.inv(K)
#     h1 = H_world_to_image[:, 0]
#     h2 = H_world_to_image[:, 1]
#     h3 = H_world_to_image[:, 2]

#     # Scale factor: lambda = 1 / ||K^{-1} h1||
#     lam = 1.0 / np.linalg.norm(K_inv @ h1)

#     r1 = lam * (K_inv @ h1)
#     r2 = lam * (K_inv @ h2)
#     r3 = np.cross(r1, r2)        # orthogonal by construction
#     t  = lam * (K_inv @ h3)

#     # Build rotation matrix and orthogonalize via SVD
#     R_approx = np.column_stack([r1, r2, r3])
#     U, _, Vt = np.linalg.svd(R_approx)
#     R = U @ Vt  # nearest orthogonal matrix

#     rvec, _ = cv2.Rodrigues(R)
#     tvec = t.reshape(3, 1)
#     return rvec, tvec





# def full_calibration_and_lifting(lane_borders, image_shape):
#     """
#     Returns K, rvec, tvec and a function to lift image points to 3D.
#     """
#     # H: image -> world (floor plane)
#     H_iw = compute_lane_homography(lane_borders)
#     # H: world -> image
#     H_wi = np.linalg.inv(H_iw)

#     K = calibrate_K_from_homography(H_wi, image_shape)
#     rvec, tvec = recover_extrinsics(H_wi, K)

#     return K, rvec, tvec, H_iw


# def lift_point_to_3d(pixel_point, H_image_to_world):
#     """
#     Given a pixel (u,v) on the LANE FLOOR, return its (X, Y, Z=0) world coords.
#     This is exact — no Z ambiguity since we know the point is on the plane.
#     """
#     p = np.array([pixel_point[0], pixel_point[1], 1.0])
#     P_w = H_image_to_world @ p
#     P_w /= P_w[2]
#     return np.array([P_w[0], P_w[1], 0.0])




# BALL_RADIUS_M = 0.108  # regulation bowling ball

# def lift_ball_center(contact_pixel, H_iw):
#     floor_pt = lift_point_to_3d(contact_pixel, H_iw)
#     return floor_pt + np.array([0, 0, BALL_RADIUS_M])