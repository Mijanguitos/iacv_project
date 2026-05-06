import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import cv2 as cv2
import math

# Name of the clip's video file to be used
VIDEO_FILE = str("clip_2.mp4")

LANE_POINTS = [[872, 684], [1228, 696], [1241, 277], [1146, 273]] # For clip 1
LANE_POINTS = [[819, 813], [1308, 819] , [1442, 175], [1254, 175]] # For clip 2

# Manual detection of points in the video
def click_event(event, x, y, flags, params) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
        LANE_POINTS.append([x, y])
        cv2.imshow('frame', frame)
        

def point_detection(frame) -> None:
    cv2.imshow('frame', frame)
    #cv2.resizeWindow("frame", 1000, 750)
    cv2.setMouseCallback('frame', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Extract region of interest (modified polygon of the lane)+
def compute_modified_polygon(points: np.ndarray) -> np.ndarray:
    """
    Computes a modified polygon based on the top two points of the input points.
    Args:
        points (np.ndarray): Array of points representing the polygon.
    Returns:
        np.ndarray: Modified polygon points.
    """
    top_indices = np.argsort(points[:, 1])[:2]
    top_points = points[top_indices]
    left_top, right_top = sorted(top_points, key=lambda pt: pt[0])

    # Padding
    left_top_mod = [left_top[0] - 20, left_top[1] - 15]
    right_top_mod = [right_top[0] + 20, right_top[1] - 15]
    # No padding
    #left_top_mod = [left_top[0], left_top[1]]
    #right_top_mod = [right_top[0], right_top[1]]

    return np.array(
        [
            left_top_mod
            if np.array_equal(pt, left_top)
            else right_top_mod
            if np.array_equal(pt, right_top)
            else pt
            for pt in points
        ],
        dtype=np.int32,
    )

def frame_preprocessing(frame: cv2.typing.MatLike, polygon: np.ndarray) -> cv2.typing.MatLike:
    # Frame operations
    mask = cv2.fillPoly(np.zeros(frame.shape[:2], dtype=np.uint8), [polygon], 255)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    frame = cv2.medianBlur(frame, 9)
    frame = cv2.bilateralFilter(frame, 15, 100, 75)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(15, 15))
    frame = clahe.apply(frame)

    return frame

def circle_detection(frame: cv2.typing.MatLike) -> np.ndarray:
    """
    Performs all the needed operations to the frame and returns and array of ball candidates
    
    """
    print("Detecting ball candidates...")
    circles = cv2.HoughCircles(frame,
                               method=cv2.HOUGH_GRADIENT_ALT,
                               dp=1.2,      #1.2                 # downsample size
                               minDist=50, #100          # minimum distance between detected cirlces
                               param1=300,  #300             # upper threshold for canny edge detection
                               param2=0.7,  #40             # cummulator (lower) threshold for canny edge detection
                               minRadius=10,
                               maxRadius=50)
                
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return np.squeeze(circles, axis=0)
    else:
        #print(None)
        return None
    
def create_graph(ball_candidates: list) -> nx.DiGraph:
    """
    Construct the weighted directed graph, where all the nodes represent the ball candidates, while the
    edges link the candidates in a frame with the candidates in the next two consecutive frames.
    Args:
        ball_candidates (np.ndarray): 2D array containing all detected ball candidates for each frame.
    Returns:
        nx.DiGraph: Weighted directed graph ready for trajectory analysis.
    
    """
    print("Constructing candidate graph...")

    DG = nx.DiGraph()


    colors = cm.plasma(range(len(ball_candidates)))
    color_map = []

    labels = {}

    positions = []
    x = 0
    y = 0

    # Pre-loop setup
    node_ids_by_frame = {} # Dictionary to store {frame_index: [list_of_node_ids]}
    ball_count = 0
    window_size = 5 # How many frames back to look
    max_distance = 40  # Maximum pixels a ball can move between frames

    for frame, balls in enumerate(ball_candidates):
        if balls is not None:
            node_ids_by_frame[frame] = []
            
            for ball in balls:
                # 1. Create the node
                # Assuming 'ball' is a tuple or list: (x_pos, y_pos)

                x_pos = ball[0]
                y_pos = ball[1]
                curr_pos = np.array((x_pos, y_pos))
                DG.add_node(ball_count, frame=frame, pos=curr_pos, rad=ball[2])
                node_ids_by_frame[frame].append(ball_count)
                
                # 2. Look back at the previous 'window_size' frames
                for lookback in range(1, window_size + 1):
                    prev_frame = frame - lookback
                    
                    # Check if we have nodes recorded for that previous frame
                    if prev_frame in node_ids_by_frame:
                        for prev_node_id in node_ids_by_frame[prev_frame]:
                        # Get position of the previous ball
                            prev_pos = DG.nodes[prev_node_id]['pos']
                            
                            # Calculate Euclidean Distance
                            #dist = np.linalg.norm(curr_pos - prev_pos)
                            squared_diffs = (curr_pos - prev_pos) ** 2
                            sum_squared = np.sum(squared_diffs)
                            dist = np.sqrt(sum_squared)
                                                        
                            # 3. Only connect if the movement is realistic
                            if dist < max_distance and dist != 0.00:
                                DG.add_edge(prev_node_id, ball_count, weight=dist)
                                #print(f"Connected: Frame {prev_frame}->{frame} (Dist: {dist:.2f})")

                # Update visualization metadata
                color_map.append(colors[frame])
                labels[ball_count] = f"{frame}"
                positions.append((x_pos, -y_pos))
                y += 1
                ball_count += 1
                
            x += 2
            y = y * (-1)
            
    nx.draw_networkx_nodes(DG, pos=positions, label=labels, node_color=color_map)
    nx.draw_networkx_labels(DG, pos=positions, labels=labels)
    nx.draw_networkx_edges(DG, pos=positions)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    ax.axis('equal')
    plt.axis("off")
    plt.show()

    return DG

def reconstruct_trayectory(DG):
    """
    Finds the longest path in the DAG and extracts the coordinates.
    """
    print("Reconstructing trajectory...")
    # 1. Find the longest path (list of node IDs)
    longest_path_nodes = nx.dag_longest_path(DG, weight=None) # weight=None finds max number of nodes
    
    # 2. Extract positions for plotting
    # Assuming nodes have 'pos' attribute stored as np.array([x, y])
    trajectory_coords = np.array([DG.nodes[n]['pos'] for n in longest_path_nodes])
    radii = np.array([DG.nodes[n]['rad'] for n in longest_path_nodes])

    frames = [DG.nodes[n]['frame'] for n in longest_path_nodes]
    
    return longest_path_nodes, trajectory_coords, frames, radii

def plot_trajectory(coords, frames):
    plt.figure(figsize=(10, 6))
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Plot the line connecting the detections
    plt.plot(x, y, '-o', color='teal', markersize=4, label='Longest Trajectory', alpha=0.6)
    
    # Highlight start and end
    plt.scatter(x[0], y[0], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x[-1], y[-1], color='red', s=100, label='End', zorder=5)
    
    # Annotate frame numbers for context
    for i, txt in enumerate(frames):
        if i % 2 == 0: # Annotate every 2nd frame to keep it clean
            plt.annotate(txt, (x[i], y[i]), fontsize=8, alpha=0.8)

    plt.title("Reconstructed Ball Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis() # Often needed for image coordinates
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.show()

def trajectory_interpolation(coords, frames):
    """
    LS interpolation to accomodate the treyectory
    """
    print("Interpolating...")
    n = len(coords)

    # Interpolate x values for every frame
    # Cuadratic model for x values

    # Design matrix
    f = np.array(frames)
    x = coords[:, 0]
    n = len(f)
    A = np.transpose(np.array([f**2, f, np.ones(n)]))
    N = A.T @ A
    N_inv = np.linalg.inv(N)
    T = A.T @ x

    params_x = N_inv @ T

    new_x = A @ params_x

    # Cuadratic model y = a*x^2 + b*x + c
    # Design matrix
    y = coords[:, 1]
    #A = np.transpose(np.array([x**2, x, np.ones(n)]))

    N = A.T @ A
    N_inv = np.linalg.inv(N)
    T = A.T @ y

    params_y = N_inv @ T

    new_y = A @ params_y

    # Interpolate for every frame
    f = np.arange(frames[0], frames[-1])
    A = np.transpose(np.array([f**2, f, np.ones(len(f))]))

    x = A @ params_x
    y = A @ params_y

    return np.transpose([x, y])

def radius_interpolation(radius, frames):
    r = radius.T

    # Cubic
    f = np.array(frames)
    A = np.transpose(np.array([f, np.ones(len(f))]))

    N = A.T @ A
    N_inv = np.linalg.inv(N)
    T = A.T @ r

    params_r = N_inv @ T
    
    f = np.arange(frames[0], frames[-1])
    A = np.transpose(np.array([f, np.ones(len(f))]))

    r = A @ params_r

    return r
    

def plot_dual_trajectories(orig_coords, interp_coords, orig_frames=None):
    """
    Plots the original detected trajectory and the interpolated version.
    
    Args:
        orig_coords (np.ndarray): Nx2 array of (x, y) from the graph.
        interp_coords (np.ndarray): Mx2 array of (x, y) after interpolation.
        orig_frames (list): Optional frame numbers for the original detections.
    """
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # 1. Plot Interpolated Trajectory (The "Smooth" path)
    plt.plot(interp_coords[:, 0], interp_coords[:, 1], 
             linestyle='--', color='gray', linewidth=1.5, 
             label='Interpolated Path', alpha=0.6)

    # 2. Plot Original Trajectory (The "Detected" path)
    plt.plot(orig_coords[:, 0], orig_coords[:, 1], 
             '-o', color='teal', linewidth=2, markersize=6, 
             label='Original Detections', markeredgecolor='white')

    # 3. Highlight Start and End of the total sequence
    plt.scatter(interp_coords[0, 0], interp_coords[0, 1], 
                color='green', s=120, label='Start', zorder=5, edgecolors='black')
    plt.scatter(interp_coords[-1, 0], interp_coords[-1, 1], 
                color='red', s=120, label='End', zorder=5, edgecolors='black')

    # 4. Optional: Annotate original frame numbers
    if orig_frames is not None:
        for i, frame_num in enumerate(orig_frames):
            plt.annotate(f"F{frame_num}", 
                         (orig_coords[i, 0], orig_coords[i, 1]),
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center', 
                         fontsize=8, 
                         color='teal',
                         fontweight='bold')

    # Formatting
    plt.title("Ball Trajectory: Original vs. Interpolated", fontsize=14)
    plt.xlabel("X Coordinate (px)")
    plt.ylabel("Y Coordinate (px)")
    
    # Standard image coordinate system: Y increases downwards
    ax.invert_yaxis() 
    
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.axis('equal')
    plt.show()
    
if __name__ == "__main__":

    # Load the video clip and get total number of frames and frame rate
    CAP = cv2.VideoCapture(VIDEO_FILE)                  # Class for video captured from the clip's file   
    n_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))   # Clip's total number of frames
    fps = CAP.get(cv2.CAP_PROP_FPS)                     # Clip's frame rate
    print(f"Total frames: {n_frames}, FPS: {fps}")

    # Manually get the lane points
    # TODO: Replace with lane detection code
    """
    CAP.set(cv2.CAP_PROP_POS_FRAMES, -1)
    ret, frame = CAP.read()
    point_detection(frame)
    print(LANE_POINTS)
    """
    # Compute a polygon that covers the lane with extension on the topside of the image
    points = np.array(LANE_POINTS, dtype=np.int32)  # Parse the lane points into int32  
    polygon = compute_modified_polygon(points)      # Compute the extended polygon 

    # Ball detection routine
    ball_candidates = list()    # Create empty array to store all the potential ball candidates 

    # Ball detection iteration
    while True:
        ret, frame = CAP.read()         # Get the next video frame
        if not ret:                     # If there is not frame, break
            break

        preprocessed_frame = frame_preprocessing(frame, polygon)

        circles = circle_detection(preprocessed_frame)
        ball_candidates.append(circles)

        #print(ball_candidates)
        
        if circles is not None:
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(preprocessed_frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(preprocessed_frame, (x, y), 1, (0, 128, 255), -1)
            
        cv2.imshow("Video Frame", preprocessed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    candidates_graph = create_graph(ball_candidates)
    path_nodes, coords, frames, radii = reconstruct_trayectory(candidates_graph)

    plot_trajectory(coords, frames)

    new_coords = trajectory_interpolation(coords, frames)
    plot_trajectory(new_coords, frames)

    plot_dual_trajectories(coords, new_coords, frames)

    new_radii = radius_interpolation(np.array(radii), frames)

    i = 0
    CAP = cv2.VideoCapture(VIDEO_FILE)                  # Class for video captured from the clip's file   

    while True:
        ret, frame = CAP.read()         # Get the next video frame
        if not ret:                     # If there is not frame, break
            break

        if i > frames[0] and i < frames[-1]:
            x = int(new_coords[i - frames[0], 0])
            y = int(new_coords[i - frames[0], 1])
            r = int(new_radii[i - frames[0]])
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame, (x, y), 1,  (0, 128, 255), -1)

        cv2.imshow("Video Frame", frame)

        i += 1
        # Wait for 1ms for key press to continue or exit if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    print(-1)