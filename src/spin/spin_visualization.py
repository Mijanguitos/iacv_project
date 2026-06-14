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

def generate_texture(width=400, height=200):
    """Generates a procedural irregular texture (4 sections) for the companion ball."""
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if (i < height // 2) == (j < width // 2):
                texture[i, j] = [255, 0, 0] # Blue
            else:
                texture[i, j] = [0, 255, 255] # Yellow
    return texture

def render_companion_ball(frame, cx, cy, r, R_abs, texture):
    """Renders a textured 3D sphere to visualize spin."""
    x0, x1 = int(cx - r), int(cx + r + 1)
    y0, y1 = int(cy - r), int(cy + r + 1)
    
    x0_c = max(0, x0)
    y0_c = max(0, y0)
    x1_c = min(frame.shape[1], x1)
    y1_c = min(frame.shape[0], y1)
    
    if x0_c >= x1_c or y0_c >= y1_c:
        return frame
        
    y, x = np.mgrid[y0_c:y1_c, x0_c:x1_c]
    
    dx = (x - cx) / r
    dy = (y - cy) / r
    
    mask = dx**2 + dy**2 <= 1.0
    
    dx_valid = dx[mask]
    dy_valid = dy[mask]
    dz_valid = np.sqrt(np.clip(1.0 - dx_valid**2 - dy_valid**2, 0.0, 1.0))
    
    # 3D points in the camera frame
    pts_3d = np.vstack((dx_valid, dy_valid, dz_valid)) # 3 x N
    
    # Rotate backwards to find texture coordinates
    pts_tex = R_abs.T @ pts_3d # 3 x N
    
    px, py, pz = pts_tex[0], pts_tex[1], pts_tex[2]
    
    # Calculate UV coordinates
    u = 0.5 + np.arctan2(px, pz) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(py, -1.0, 1.0)) / np.pi
    
    tex_h, tex_w = texture.shape[:2]
    
    tex_x = np.clip((u * tex_w).astype(np.int32), 0, tex_w - 1)
    tex_y = np.clip((v * tex_h).astype(np.int32), 0, tex_h - 1)
    
    # Basic Lambertian shading from a light source
    light_dir = np.array([0.5, -0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.dot(light_dir, pts_3d)
    intensity = np.clip(intensity, 0.2, 1.0) # ambient + diffuse
    
    colors = texture[tex_y, tex_x] * intensity[:, np.newaxis]
    
    res_frame = frame
    for c in range(3):
        channel = res_frame[y0_c:y1_c, x0_c:x1_c, c]
        channel[mask] = colors[:, c]
        res_frame[y0_c:y1_c, x0_c:x1_c, c] = channel
        
    return res_frame

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
    #fourcc = cv2.VideoWriter_fourcc(*'mp4')
    out = cv2.VideoWriter(f"{output_path}", -1, fps, (width, height))
    
    # 4. Initialize 3D Geometry
    rings_3d = create_wireframe_rings()
    R_abs = np.eye(3) # The absolute orientation matrix (starts at Identity)
    texture = generate_texture(400, 200)

    def draw_wireframe_and_axis(target_frame, center_x, center_y, radius, ax_val, ay_val, rings, rotation_matrix):
        # --- DRAW WIREFRAME ---
        for ring in rings:
            rotated_ring = ring @ rotation_matrix.T
            num_points = len(rotated_ring)
            for i in range(num_points):
                p1 = rotated_ring[i]
                p2 = rotated_ring[(i + 1) % num_points]
                # Only draw the segment if it's on the front-facing hemisphere (Z >= 0)
                # We use a tiny negative tolerance to prevent gaps at the edges
                if p1[2] >= -0.01 and p2[2] >= -0.01:
                    pt1 = (int(p1[0] * radius + center_x), int(p1[1] * radius + center_y))
                    pt2 = (int(p2[0] * radius + center_x), int(p2[1] * radius + center_y))
                    cv2.line(target_frame, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            
        # --- DRAW SPIN AXIS ---
        axis_scale = radius * 1.5
        start_pt = (int(center_x - ax_val * axis_scale), int(center_y - ay_val * axis_scale))
        end_pt = (int(center_x + ax_val * axis_scale), int(center_y + ay_val * axis_scale))
        cv2.line(target_frame, start_pt, end_pt, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(target_frame, end_pt, 4, (0, 0, 255), -1)
    
    frame_idx = 0
    has_seen_ball = False
    last_ax, last_ay, last_az = 0.0, 0.0, 0.0
    last_omega = 0.0

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
            omega = f_data.get("omega", f_data.get("angle", 0.0))
            
            # --- UPDATE ORIENTATION ---
            # Create a rotation vector (Axis * Omega)
            rvec = np.array([ax, ay, az]) * omega
            
            # Convert vector to a 3x3 Rotation Matrix
            R_delta, _ = cv2.Rodrigues(rvec)
            
            # Multiply to accumulate rotation over time
            R_abs = R_delta @ R_abs 
            
            # Draw on original ball
            draw_wireframe_and_axis(frame, cx, cy, r, ax, ay, rings_3d, R_abs)
            # Optional: Draw the bounding circle of the ball
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            last_ax, last_ay, last_az = ax, ay, az
            last_omega = omega
            has_seen_ball = True
        
        elif has_seen_ball:
            # Freeze the companion ball at its last known rotation (stop spinning)
            ax, ay = last_ax, last_ay
            omega = last_omega

        if has_seen_ball:
            # --- DRAW COMPANION BALL ---
            # Place the companion ball in the top right corner
            r_comp = 100
            cx_comp = width - r_comp - 30
            cy_comp = r_comp + 30
            frame = render_companion_ball(frame, cx_comp, cy_comp, r_comp, R_abs, texture)
            
            # Draw on companion ball
            draw_wireframe_and_axis(frame, cx_comp, cy_comp, r_comp, ax, ay, rings_3d, R_abs)
            
            # Display angular velocity below the companion ball
            text = f"Omega: {omega:.4f} rad/f"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
            text_x = cx_comp - text_size[0] // 2
            text_y = cy_comp + r_comp + 30
            
            # Draw text with outline for better visibility
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

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
    output_path = f"{PROJECT_ROOT}\\src\\spin\\spin_output\\{clip}_rendered"
    
    render_spin_overlay(json_path, video_path, output_path)