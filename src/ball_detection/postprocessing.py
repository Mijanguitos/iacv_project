""" Trajectory computation, postprocessing, and visualization of bowling ball.

Retrieves the 

"""

from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import json
import os

def create_graph(ball_candidates: dict) -> nx.DiGraph:
    """
    Construct the weighted directed graph, where all the nodes represent the ball candidates, while the
    edges link the candidates in a frame with the candidates in the next two consecutive frames.
    Args:
        ball_candidates (np.ndarray): 2D array containing all detected ball candidates for each frame.
    Returns:
        nx.DiGraph: Weighted directed graph ready for trajectory analysis.
    
    """

    # Create graph object
    DG = nx.DiGraph()

    # Getting the number of frames from the dictionary
    n_frames = len(ball_candidates)

    colors = cm.pink(range(n_frames))
    color_map = []

    labels = {}

    positions = []
    x = 0
    y = 0

    # Pre-loop setup
    node_ids_by_frame = {} # Dictionary to store {frame_index: [list_of_node_ids]}
    ball_count = 0
    window_size = 15 # How many frames back to look
    max_distance = 80  # Maximum pixels a ball can move between frames

    for frame in range(n_frames):
        if ball_candidates[str(frame)] is not None:
            node_ids_by_frame[frame] = []
            
            for ball in ball_candidates[str(frame)]:
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
                                                        
                            # --- DIRECTIONAL FILTER ---
                            # In OpenCV, Y=0 is the top of the screen. Moving "upwards" means Y decreases.
                            # We use <= with a tiny tolerance (+2 pixels) to account for slight bounding-box jitter 
                            # where the ball might appear perfectly flat for a single frame.
                            is_moving_upwards = curr_pos[1] <= (prev_pos[1] + 2)
                            
                            # Only connect if the movement is realistic AND directional
                            if dist < max_distance and dist != 0.00 and is_moving_upwards:
                                DG.add_edge(prev_node_id, ball_count, weight=dist)
                                #print(f"Connected: Frame {prev_frame}->{frame} (Dist: {dist:.2f})")

                # Update visualization metadata
                color_map.append(colors[frame])
                labels[ball_count] = f"{frame}"
                positions.append((frame, -y_pos))                
                y += 1
                ball_count += 1
                
            x += 2
            y = y * (-1)
            

    # --- Improved Spatiotemporal Visualization ---
    
    # 1. Setup the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # 2. Normalize colors properly for the colormap
    norm = mcolors.Normalize(vmin=0, vmax=n_frames)
    cmap = cm.viridis 
    
    node_colors = [cmap(norm(DG.nodes[n]['frame'])) for n in DG.nodes]

    # 3. Draw nodes
    nx.draw_networkx_nodes(
        DG, 
        pos=positions, 
        node_color=node_colors, 
        node_size=100,         
        alpha=1,           
        edgecolors='white',   
        linewidths=0.5,
        ax=ax
    )

    # 4. Draw edges (Heavier and more visible)
    nx.draw_networkx_edges(
        DG, 
        pos=positions, 
        alpha=0.9,            # Increased opacity
        width=1.5,            # Thicker lines
        edge_color="black",    # Distinct color
        arrows=True,
        arrowsize=5,         # Larger arrowheads
        ax=ax
    )

    # 5. Filter labels (One label per 15th frame)
    filtered_labels = {}
    labeled_frames = set()
    
    for n in DG.nodes:
        frame = DG.nodes[n]['frame']
        if frame % 15 == 0 and frame not in labeled_frames:
            filtered_labels[n] = str(frame)
            labeled_frames.add(frame) 
            
    # 6. Add a colorbar 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Frame Number', pad=0.02)

    # 7. Polish the axes (Turned ON for Spatiotemporal layout)
    ax.set_aspect('auto') # crucial for mixing time and pixel coordinates
    ax.margins(0.10)
    #ax.set_title("Ball Candidate Trajectory: Temporal Flow", fontsize=14, pad=15)
    
    # Explicitly set axis labels
    ax.set_xlabel("Frame Number (Time)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Pixel Y-Coordinate (Depth)", fontsize=11, fontweight='bold')
    
    # Keep axes visible and show ticks
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    # Add a subtle grid to help track values across the span
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    #plt.show()

    return DG

def reconstruct_trayectory(candidates_graph: nx.DiGraph):
    """
    Finds the longest path in the DAG and extracts the coordinates.
    """
    # Find the longest path
    longest_path_nodes = nx.dag_longest_path(candidates_graph, weight=None) # weight=None finds max number of nodes
    
    # Extract positions for plotting
    # Assuming nodes have 'pos' attribute stored as np.array([x, y])
    trajectory_coords = np.array([candidates_graph.nodes[n]['pos'] for n in longest_path_nodes])
    radii = np.array([candidates_graph.nodes[n]['rad'] for n in longest_path_nodes])

    frames = np.array([candidates_graph.nodes[n]['frame'] for n in longest_path_nodes])
    
    return longest_path_nodes, trajectory_coords, frames, radii

def interpolation(data: np.ndarray, 
                  frames_in: np.ndarray, 
                  frames_out: np.ndarray, 
                  degree: int) -> np.ndarray:
    """
    Perform Least Squares Polynomial Interpolation of the given degree.
    
    Args:
        data (np.ndarray): Array of shape (N, 2) containing [x, y] coordinates.
        frames_in (np.ndarray): Array of shape (N,) containing input frame indices.
        frames_out (np.ndarray): Array of shape (M,) containing output frame indices.
        degree (int): The degree of the polynomial (e.g., 2 for quadratic, 3 for cubic).
    
    Returns:
        np.ndarray: Interpolated values over the frame's domain.
    """
    # 1. Shift the timeline to prevent polynomial explosion
    f_shift = frames_out[0]
    
    # 2. Fit the curve using ONLY the clean, filtered frames
    A = np.vander(frames_in - f_shift, degree + 1)
    params, _, _, _ = np.linalg.lstsq(A, data, rcond=None)
    
    # 3. Reconstruct the smooth curve over the FULL, unbroken timeline
    A_full = np.vander(frames_out - f_shift, degree + 1)
    est_data = A_full @ params        

    return est_data

def logarithmic_interpolation(data: np.ndarray, 
                              frames_in: np.ndarray, 
                              frames_out: np.ndarray) -> np.ndarray:
    """
    Perform Logarithmic Least Squares Interpolation.
    Fits the curve r = A * ln(f_shifted) + B.
    
    Args:
        data (np.ndarray): Array containing radius values.
        frames_in (np.ndarray): Array containing input frame indices.
        frames_out (np.ndarray): Array containing output frame indices.
        
    Returns:
        np.ndarray: Logarithmically interpolated radius values.
    """
    f_shift = frames_out[0] - 1
    
    # Fit using filtered frames
    ln_x = np.log(frames_in - f_shift)
    A_matrix = np.vstack([ln_x, np.ones(len(ln_x))]).T
    A, B = np.linalg.lstsq(A_matrix, data, rcond=None)[0]
    
    # Reconstruct over full timeline
    est_data = A * np.log(frames_out - f_shift) + B
    
    return est_data


def spline_interpolation(data: np.ndarray, frames: np.ndarray) -> np.ndarray:
    cs = CubicSpline(frames, data, bc_type='not-a-knot')
    f = np.arange(frames[0], frames[-1])
    est_x = cs(f)
    return est_x

def gaussian_filter(data: np.ndarray, sigma: float = 5) -> np.ndarray:
    """ Apply Gaussian filter to the 1D data for smoothing."""
    return gaussian_filter1d(data, sigma=sigma)

def remove_outliers(data: np.ndarray, frames: np.ndarray, threshold: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers by detecting sudden, unrealistic jumps in velocity between frames.
    
    Args:
        data (np.ndarray): The 1D array of positions (x, y, or r).
        frames (np.ndarray): The corresponding frame numbers.
        threshold (float): Z-score threshold for velocity. Lower is stricter (e.g., 2.0).
    
    Returns:
        tuple: Filtered data array and filtered frames array.
    """
    # Calculate frame-to-frame velocity (change in position / change in time)
    dt = np.diff(frames)
    dy = np.diff(data)
    velocity = dy / dt
    
    # Duplicate the first velocity to maintain the same array length
    velocity = np.insert(velocity, 0, velocity[0])
    
    # Calculate mean and standard deviation of the velocities
    mean_v = np.mean(velocity)
    std_v = np.std(velocity)
    
    # Avoid division by zero if the object is perfectly stationary
    if std_v == 0:
        return data, frames
        
    # Compute Z-scores
    z_scores = np.abs((velocity - mean_v) / std_v)
    
    # Keep only the points where the velocity jump is within the normal distribution
    mask = z_scores < threshold
    
    return data[mask], frames[mask]

def plot_interpolation(frames_raw: np.ndarray, data_raw: np.ndarray, 
                       frames_interp: np.ndarray, data_interp: np.ndarray, 
                       title: str = "Interpolated Trajectory", 
                       ylabel: str = "Position"):
    """
    Plots the original raw detections against the smoothed interpolation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot original raw data as a scatter plot
    ax.scatter(frames_raw, data_raw, color='gray', alpha=0.6, label='Raw Detections', zorder=2)

    # Plot the interpolated/smoothed data as a continuous line
    ax.plot(frames_interp, data_interp, color='#1f77b4', linewidth=2.5, label='Smoothed Spline', zorder=3)

    # Formatting for a clean, professional look
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Frame Number (Time)", fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.legend(loc='best', fontsize=10)
    
    # Tighten up the layout
    ax.margins(0.05)
    #plt.tight_layout()
    #plt.show()

def compute_trajectory(candidates_path: os.PathLike[str],
                       save_path: os.PathLike[str]) -> dict:
    
    print(f"Ball detection: Computing trajectory {candidates_path}")

    with open(f"{candidates_path}.json", "r") as file:
        loaded_candidates = json.load(file)

    candidates_graph = create_graph(loaded_candidates)
    longest_path_nodes, trajectory_coords, frames, radii = reconstruct_trayectory(candidates_graph)

    # Interpolate x and y coordinates, and ball radius
    x = trajectory_coords[:, 0]
    y = trajectory_coords[:, 1]
    r = np.transpose(radii)

    f_global = np.arange(frames[0], frames[-1])

    observations = {"x": x.tolist(),
                    "y": y.tolist(),
                    "r": r.tolist(),
                    "f": frames.tolist()}
    
    x, frames_x = remove_outliers(x, frames)
    y, frames_y = remove_outliers(y, frames)
    r, frames_r = remove_outliers(r, frames)
    
    x_int = interpolation(x, frames_x, f_global, 4)
    #x_int = spline_interpolation(x, frames)
    #x_int = gaussian_filter(x_int, sigma=3)
    #y_int = interpolation(y, frames, 4)
    #y_int = spline_interpolation(y, frames)
    y_int = interpolation(y, frames_y, f_global, 4)
    #y_int = gaussian_filter(y_int, sigma=3)
    
    #r_int = exponential_interpolation(r, frames)   
    #r_int = spline_interpolation(r, frames)
    r_int = logarithmic_interpolation(r, frames_r, f_global)
    r_int = interpolation(r, frames_r, f_global, 4)

    f = np.arange(frames[0], frames[-1])

    """
    plot_interpolation(
        frames_raw=frames_y,  # Matches the length of filtered 'y'
        data_raw=y, 
        frames_interp=f_global, 
        data_interp=y_int, 
        title="Smoothed Y-Coordinate Trajectory", 
        ylabel="Pixel Y-Coordinate (Depth)"
    )

    plot_interpolation(
        frames_raw=frames_x,  # Matches the length of filtered 'x'
        data_raw=x, 
        frames_interp=f_global, 
        data_interp=x_int, 
        title="Smoothed X-Coordinate Trajectory", 
        ylabel="Pixel X-Coordinate"
    )

    plot_interpolation(
        frames_raw=frames_r,  # Matches the length of filtered 'r'
        data_raw=r, 
        frames_interp=f_global, 
        data_interp=r_int, 
        title="Smoothed Radius Trajectory", 
        ylabel="Ball Radius (Pixels)"
    )
    """

    estimations = {"x": x_int.tolist(), 
                   "y": y_int.tolist(),
                   "r": r_int.tolist(),
                   "f": f.tolist()}
    
    trajectory = {"observations": observations,
                  "estimations": estimations}
    
    with open(f"{save_path}.json", 'w') as f:
        json.dump(trajectory, f, indent=4)

    print(f"Ball detection:Trajectory saved {candidates_path}")
    

if __name__ == "__main__":

    clip = "clip_2"
    candidates_path = f"detection_output\\candidates_{clip}"
    save_path = f"postprocessing_output\\trajectory_{clip}"

    compute_trajectory(candidates_path, save_path)