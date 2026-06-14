from pathlib import Path
import numpy as np
import cv2 as cv2
import json
import math
import os


def roi_bounds(x1: int, y1: int, 
               x2: int, y2: int,
               r: int, frame_shape:tuple, offset: int):
    
    x_min = max(0, min(
                x1 - r - offset,
                x2 - r - offset))
    x_max = min(frame_shape[1] - 1, max(
                x1 + r + offset,
                x2 + r + offset))
    y_min = max(0, 
                y2 - r - offset)
    y_max = min(frame_shape[0] - 1,
                y1 + r + offset)
    
    return x_min, x_max, y_min, y_max


# Define a single roi based on the radius of the first ball 
def roi_bounds_single(x: int, y: int, r: int, frame_shape:tuple, offset: int):
    x_min = max(0, x - r - offset)
    x_max = min(frame_shape[1] - 1, x + r + offset)
    y_min = max(0, y - r - offset)
    y_max = min(frame_shape[0] - 1, y + r + offset)
    
    return x_min, x_max, y_min, y_max


def pad_roi(roi, th, tw):
    ph = th - roi.shape[0]
    pw = tw - roi.shape[1]
    return cv2.copyMakeBorder(roi, 0, ph, 0, pw, cv2.BORDER_REFLECT)


def frame_preprocessing(frame: cv2.typing.MatLike):
    # Create CLAHE object (localized histogram equalization)
    clahe = cv2.createCLAHE(clipLimit = 2.5,
                            tileGridSize = (15, 15))
    
    return clahe.apply(frame)

def optical_flow(gray_frame_1: cv2.typing.MatLike,
                 gray_frame_2: cv2.typing.MatLike,
                 mask: cv2.typing.MatLike,
                 ball_radius: int):

    # Detect corners Shi-Tomasi
    p0 = cv2.goodFeaturesToTrack(gray_frame_1, 
                                 mask=mask, 
                                 maxCorners=100, 
                                 qualityLevel=0.001, 
                                 minDistance=0, 
                                 blockSize=3)
    if p0 is None:
        raise ValueError("No features detected")
    
    lk_params = {
        "winSize"   : (21, 21),
        "maxLevel"  : 3,
        "criteria"  : (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01),
    }

    p1, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        gray_frame_1, gray_frame_2, p0, None, **lk_params
    )
    p0r, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray_frame_2, gray_frame_1, p1, None, **lk_params
    )

    fb_error = np.linalg.norm(p0 - p0r, axis=2)
    return p0, p1, p0r, status_forward, status_backward, fb_error

def filter_3d_points(
    p0,
    p1,
    p0r,
    status_forward,
    status_backward,
    fb_error,
    center_roi1,
    center_roi2,
    ball_radius,
    fb_threshold=10.0,
    low_threshold_factor=2,
    is_retry=False,  # <-- Added safety flag
):
    movement_threshold = ball_radius
    low_movement_threshold = ball_radius / low_threshold_factor

    old3d, new3d, good_pts = [], [], []
    for old_pt, new_pt, s1, s2, err in zip(
        p0.reshape(-1, 2),
        p1.reshape(-1, 2),
        status_forward.ravel(),
        status_backward.ravel(),
        fb_error.ravel(),
    ):
        if s1 and s2 and err < fb_threshold:
            displacement = np.linalg.norm(new_pt - old_pt)
            if low_movement_threshold < displacement < movement_threshold:
                ox, oy = old_pt - center_roi1
                nx, ny = new_pt - center_roi2
                oz = math.sqrt(max(ball_radius**2 - ox**2 - oy**2, 0))
                nz = math.sqrt(max(ball_radius**2 - nx**2 - ny**2, 0))
                old3d.append([ox, oy, oz])
                new3d.append([nx, ny, nz])
                good_pts.append((old_pt, new_pt))

    old3d = np.array(old3d)
    new3d = np.array(new3d)

    # If we failed to find 3 points, and we HAVEN'T retried yet, do it once.
    if old3d.shape[0] < 3 and not is_retry:
        print(f"Lenient retry triggered (Found {old3d.shape[0]} points)")
        return filter_3d_points(
            p0,
            p1,
            p0r,
            status_forward,
            status_backward,
            fb_error,
            center_roi1,
            center_roi2,
            ball_radius,
            low_threshold_factor=10,
            is_retry=True,  # <-- Flag that we are now in the retry loop
        )
    
    outside = sum(1 for ox, oy, _ in old3d if ox**2 + oy**2 > ball_radius**2)
    if outside > 0:
        print(f"Warning: {outside}/{len(old3d)} points projected outside ball radius")


    # If we already retried (or succeeded), just return the arrays
    return old3d, new3d, good_pts

def compute_rotation(old3d, new3d):
    P, Q = old3d, new3d
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    cos_omega = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    omega = np.arccos(cos_omega)

    if np.isclose(omega, 0):
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * np.sin(omega)
        )
        axis /= np.linalg.norm(axis)

    return axis, omega

def spin_detection(trajectory_path: os.PathLike[str], 
                   video_path: os.PathLike[str],
                   save_path: os.PathLike[str]):
    
    # Load the video clip and get total number of frames and frame rate
    CAP = cv2.VideoCapture(video_path)                  # Class for video captured from the clip's file
    n_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))   # Clip's total number of frames
    fps = CAP.get(cv2.CAP_PROP_FPS)                     # Clip's frame rate
    width  = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
    height = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)   # float `height`

    # Extract the trajectory data from the JSON file
    with open(f"{trajectory_path}.json", "r") as f:
        data = json.load(f)

    trajectory = data["estimations"]
    xs = list(map(int, trajectory["x"]))
    ys = list(map(int, trajectory["y"]))
    rs = list(map(int, trajectory["r"]))
    fs = trajectory["f"]

    # Dictionary to save the final output
    spin_output = dict()

    # Loop through the frames of the video minus one frame
    for i in range(n_frames - 1):
        # Set the capture to the first frame and get the following one
        CAP.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret1, frame1 = CAP.read()
        ret2, frame2 = CAP.read()

        if not (ret1 and ret2):
            break

        if i in fs and i != fs[-1]:
            # Calculate the index for xs, ys, rs based on the frame number
            j = i - fs[0]  

            x1, y1, r1 = xs[j], ys[j], rs[j]
            x2, y2, r2 = xs[j + 1], ys[j + 1], rs[j + 1]

            x_min, x_max, y_min, y_max = roi_bounds_single(x1, y1, r1, frame1.shape, offset=2)                                                        
            roi1 = frame1[y_min:y_max, x_min:x_max]

            # Shift the second ROI to be centered on the second ball position
            # the rois need to be the same shape
            x_min2, x_max2, y_min2, y_max2 = roi_bounds_single(x2, y2, r1, frame2.shape, offset=2)
            roi2 = frame2[y_min2:y_max2, x_min2:x_max2]

            # Pad both ROIs to the same target size in case edge-clipping
            # caused one to be smaller than the other (fixes LK pyramid size error)
            target_h = max(roi1.shape[0], roi2.shape[0])
            target_w = max(roi1.shape[1], roi2.shape[1])


            roi1 = pad_roi(roi1, target_h, target_w)
            roi2 = pad_roi(roi2, target_h, target_w)

            print(roi1.shape, roi2.shape)

            # Compute center in ROI coordinatesq
            center_roi1 = np.array([x1 - x_min, y1 - y_min])
            center_roi2 = np.array([x2 - x_min2, y2 - y_min2])

            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

            mask = np.zeros_like(gray1)
            cv2.circle(
                mask, tuple(center_roi1.astype(int)), int(r1 * 0.8), 255, -1
            )
            try:
                p0, p1, p0r, s_f, s_b, fb_error = optical_flow(
                    gray1, gray2, mask, r1
                )
            except ValueError as e:
                print(f"Frame {i} skipped: {e}") # Prints "No features detected"
                continue
            
            old3d, new3d, good_pts = filter_3d_points(
                p0, p1, p0r, s_f, s_b, fb_error, center_roi1, center_roi2, r1
            )
            
            if old3d.shape[0] < 3:
                print(f"Frame {i} skipped: Not enough valid 3D points ({old3d.shape[0]})")
                continue # Abort the rest of the loop and move to the next frame

            axis, omega = compute_rotation(old3d, new3d)

            # Create a copy of the ROI so we don't draw on the original image data
            # Create copies so we don't draw on the original data
            vis_roi1 = roi1.copy()
            vis_roi2 = roi2.copy()

            # The centers of the ball in this specific ROI
            cx1, cy1 = int(center_roi1[0]), int(center_roi1[1])
            cx2, cy2 = int(center_roi2[0]), int(center_roi2[1])

            # Draw the ball's bounding circles for context
            cv2.circle(vis_roi1, (cx1, cy1), r1, (255, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.circle(vis_roi2, (cx2, cy2), r2, (255, 255, 0), 1, lineType=cv2.LINE_AA)

            # Loop through all the points that survived the 3D filter
            for (old_pt, new_pt) in good_pts:
                ox, oy = int(old_pt[0]), int(old_pt[1])
                nx, ny = int(new_pt[0]), int(new_pt[1])

                # Draw on ROI 1: Just the starting point
                cv2.circle(vis_roi1, (ox, oy), 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                # Draw on ROI 2: The green flow line and the ending point

                cv2.line(vis_roi2, (nx, ny), (ox, oy), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                cv2.circle(vis_roi2, (ox, oy), 1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
                cv2.circle(vis_roi2, (nx, ny), 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            # --- ENHANCEMENT: Resize for Visibility ---
            scale_factor = 8
            vis_roi1_large = cv2.resize(vis_roi1, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            vis_roi2_large = cv2.resize(vis_roi2, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

            # --- ADD TEXT LABELS ---
            # We add text AFTER scaling up so the font is high resolution and readable
            cv2.putText(vis_roi1_large, f"Frame {i}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(vis_roi2_large, f"Frame {i+1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Stack the two images horizontally
            combined_vis = np.hstack((vis_roi1_large, vis_roi2_large))
            
            """
            # Display the combined dashboard
            cv2.imshow("Optical Flow Tracking: Before & After", combined_vis)

            # Pause execution to view the frame (Press 'q' to quit, any other key to advance)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            """

            spin_output[i] = {
                "frame": i,
                "x": x1,
                "y": y1,
                "radius": r1,
                "x_axis": axis[0],
                "y_axis": axis[1],
                "z_axis": axis[2],
                "omega": omega,
            }

    with open(f"{save_path}.json", 'w') as f:
        json.dump(spin_output, f, indent=4)


if __name__ == "__main__":

    PROJECT_ROOT = Path().resolve()

    clip = "clip_1"
    extension = ".mp4"
    video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    save_path = f"{PROJECT_ROOT}\\src\\spin\\optical_flow_output\\optical_flow_{clip}"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}"

    spin_detection(trajectory_path, video_path, save_path)

    cv2.destroyAllWindows()

    print(-1)
