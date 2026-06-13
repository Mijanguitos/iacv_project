import numpy as np
import cv2
import json
import os
from pathlib import Path

def create_wireframe_rings(pts=30):
    """
    Creates 3 orthogonal 3D rings (X, Y, Z planes) to act as a wireframe sphere.
    """
    t = np.linspace(0, 2 * np.pi, pts)
    # Ring 1: Y-Z plane
    ring_x = np.column_stack((np.zeros_like(t), np.cos(t), np.sin(t)))
    # Ring 2: X-Z plane
    ring_y = np.column_stack((np.cos(t), np.zeros_like(t), np.sin(t)))
    # Ring 3: X-Y plane
    ring_z = np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t)))
    
    return [ring_x, ring_y, ring_z]

def render_spin_overlay(json_path: os.PathLike[str], 
                        video_path: os.PathLike[str], 
                        output_path: os.PathLike[str]):
    
    # 1. Load the processed JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 2. Setup OpenCV Video Capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. Setup OpenCV Video Writer (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 4. Initialize 3D Geometry
    rings_3d = create_wireframe_rings()
    R_abs = np.eye(3) # The absolute orientation matrix (starts at Identity)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Check if we have data for this frame
        str_idx = str(frame_idx)
        if str_idx in data:
            f_data = data[str_idx]
            
            # Extract tracking and kinematics
            cx, cy = int(f_data["x"]), int(f_data["y"])
            r = int(f_data["radius"])
            ax, ay, az = f_data["x_axis"], f_data["y_axis"], f_data["z_axis"]
            angle = f_data["angle"]
            
            # --- UPDATE ORIENTATION ---
            # Create a rotation vector (Axis * Angle)
            rvec = np.array([ax, ay, az]) * angle
            
            # Convert vector to a 3x3 Rotation Matrix
            R_delta, _ = cv2.Rodrigues(rvec)
            
            # Multiply to accumulate rotation over time
            R_abs = R_delta @ R_abs 
            
            # --- DRAW WIREFRAME ---
            for ring in rings_3d:
                # Rotate the 3D points
                rotated_ring = ring @ R_abs.T
                
                # Orthographic Projection: Drop Z, scale by radius, translate to center
                pts_2d = rotated_ring[:, :2] * r + np.array([cx, cy])
                
                # Draw the ring
                pts_2d = np.int32(pts_2d)
                cv2.polylines(frame, [pts_2d], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                
            # --- DRAW SPIN AXIS ---
            # Calculate where the axis pierces the sphere (scaled out slightly for visibility)
            axis_scale = r * 1.5
            start_pt = (int(cx - ax * axis_scale), int(cy - ay * axis_scale))
            end_pt = (int(cx + ax * axis_scale), int(cy + ay * axis_scale))
            
            # Draw the axis line
            cv2.line(frame, start_pt, end_pt, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, end_pt, 4, (0, 0, 255), -1) # Put a dot on the "forward" side of the axis
            
            # Optional: Draw the bounding circle of the ball
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Write to output video
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Render complete! Saved to {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path().resolve()
    clip = "clip_7"
    extension = ".mov"
    
    # Update these paths to match your directory structure
    json_path = f"{PROJECT_ROOT}\\src\\spin\\postprocessing_output\\{clip}_postprocessing.json"
    video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    output_path = f"{PROJECT_ROOT}\\src\\spin\\spin_output\\{clip}_rendered.mp4"
    
    render_spin_overlay(json_path, video_path, output_path)