""" Trajectory computation, postprocessing, and visualization of bowling ball.

Retrieves the 

"""

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
    window_size = 5 # How many frames back to look
    max_distance = 40  # Maximum pixels a ball can move between frames

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

def interpolation(data: np.ndarray, frames: np.ndarray, degree: int) -> np.ndarray:
    """
    Perform Least Squares Polynomial Interpolation of the given degree.
    
    Args:
        data (np.ndarray): Array of shape (N, 2) containing [x, y] coordinates.
        frames (np.ndarray): Array of shape (N,) containing frame indices.
        degree (int): The degree of the polynomial (e.g., 2 for quadratic, 3 for cubic).
    
    Returns:
        np.ndarray: Interpolated values over the frame's domain.
    """

    # Number of samples
    n = len(data)

    # Least Squares interpolation
    # Dependent variable
    x = data

    # Design matrix
    f = frames
    #A = np.transpose(np.array([f**3, f**2, f, np.ones(len(f))]))
    A = np.vander(f, degree)

    N_inv = np.linalg.inv(A.T @ A)

    T = A.T @ x

    params_x = N_inv @ T

    est_x = A @ params_x

    v = x - est_x

    s2 = (v.T @ v)/(n - len(params_x))

    # New design matrix for interpolation
    f = np.arange(frames[0], frames[-1])
    #A = np.transpose(np.array([f**3, f**2, f, np.ones(len(f))]))
    A = np.vander(f, degree)

    est_x = A @ params_x        

    return est_x

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

    observations = {"x": x.tolist(),
                    "y": y.tolist(),
                    "r": r.tolist(),
                    "f": frames.tolist()}
    
    x_int = interpolation(x, frames, 4)
    y_int = interpolation(y, frames, 4)
    r_int = interpolation(r, frames, 3)
    f = np.arange(frames[0], frames[-1])


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