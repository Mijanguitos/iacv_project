from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import statistics
import json
import math
import os

# Helper plotting function
def plotting(frames, x_axes, y_axes, z_axes, angles, stage_name = "Default"):
    # Create a 2-row subplot sharing the X-axis (Frames)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Pipeline Stage: {stage_name}", fontsize=14, fontweight='bold')

    # Top Plot: X, Y, Z axes
    ax1.plot(frames, x_axes, label="X Axis", color="red", marker=".", markersize=4)
    ax1.plot(frames, y_axes, label="Y Axis", color="green", marker=".", markersize=4)
    ax1.plot(frames, z_axes, label="Z Axis", color="blue", marker=".", markersize=4)
    ax1.set_ylabel("Axis Component (-1 to 1)")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best")

    # Bottom Plot: Angle
    ax2.plot(frames, angles, label="Angle (Rads/Frame)", color="purple", marker=".", markersize=4)
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Angle")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

# OUTLIER REMOVAL FUNCTIONS

def remove_axes_outliers(x_axes, y_axes, frames, threshold=0.5):
    def fit_line(x_vals, y_vals):
        n = len(x_vals)
        if n == 0: 
            return 0, 0
            
        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - mean_x) ** 2 for x in x_vals)
        
        m = numerator / denominator if denominator != 0 else 0
        c = mean_y - m * mean_x
        return m, c

    # Compute trendlines for X and Y components over time
    m_x, c_x = fit_line(frames, x_axes)
    m_y, c_y = fit_line(frames, y_axes)

    # Loop through the array values by index
    for i, frame_val in enumerate(frames):
        x_pred = (m_x * frame_val) + c_x
        y_pred = (m_y * frame_val) + c_y

        # Calculate absolute errors
        x_error = abs(x_axes[i] - x_pred)
        y_error = abs(y_axes[i] - y_pred)

        # If either error exceeds the threshold, it's an outlier
        if x_error > threshold or y_error > threshold:
            x_axes[i] = None
            y_axes[i] = None

    return x_axes, y_axes

def remove_angle_outliers(angles, threshold = 1.0):
    """
    Removes statistically abnormal angles using Z-score filtering.
    Operates directly on a list of angles.
    """
        
    # 2. Calculate the Mean and Standard Deviation
    mean_angle = statistics.mean(angles)
    std_angle = statistics.stdev(angles)
    
    # Prevent division by zero in the rare case that all angles are exactly identical
    if std_angle == 0:
        return angles
        
    # 3. Apply the Z-score filter
    for i, angle in enumerate(angles):
        if angle is not None:
            z_score = abs(angle - mean_angle) / std_angle
            
            # If it exceeds the threshold, nullify it
            if z_score > threshold:
                angles[i] = None
                
    return angles

# INTERPOLATION AND SMOOTHING FUNCTIONS
def fill_gaps(series):
    """
    Linearly interpolates missing (None) values in a 1D list.
    Handles leading and trailing gaps via flat extrapolation.
    """
    # Identify the indices and values of the valid (surviving) data points
    valid_indices = [i for i, val in enumerate(series) if val is not None]
    valid_values = [series[i] for i in valid_indices]

    # Edge case: If there's 1 or 0 valid frames, we can't interpolate
    if len(valid_indices) < 2:
        return series

    # Define the target timeline (every index from 0 to the end of the array)
    all_indices = np.arange(len(series))

    # Perform NumPy linear interpolation
    interpolated_values = np.interp(all_indices, valid_indices, valid_values)

    # Convert back to a standard Python list to keep it native
    return interpolated_values.tolist()


def gaussian_smoothing(series, sigma=10.0):
    """
    Applies a 1D Gaussian filter using pure NumPy.
    Replicates scipy.ndimage.gaussian_filter1d(mode='nearest').
    """
    if len(series) < 2:
        return series

    # Generate the Gaussian kernel
    # A radius of 4*sigma captures 99.99% of the bell curve's weight
    radius = int(4 * sigma)
    x = np.arange(-radius, radius + 1)
    
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)  # Normalize so the weights equal exactly 1.0

    # Pad the series to prevent edge drop-off (mode='nearest')
    pad_size = len(kernel) // 2
    padded_series = np.pad(series, pad_size, mode='edge')

    # Apply the convolution
    smoothed_series = np.convolve(padded_series, kernel, mode='valid')

    return smoothed_series.tolist()

# PHYSICAL CONSTRAINTS
def enforce_non_decreasing_x(x_axes):
    """
    Forces the X-axis to only stay flat or increase over time.
    Prevents the spin axis from mathematically "rewinding" mid-throw.
    """
    if not x_axes:
        return x_axes
        
    x_out = list(x_axes) # Create a copy to avoid mutating the original
    last_valid = x_out[0]
    
    for i in range(1, len(x_out)):
        if x_out[i] < last_valid:
            x_out[i] = last_valid
        else:
            last_valid = x_out[i]
            
    return x_out

def scale_x_axis(x_axes):
    """
    Applies a ramping scale from x_start to 1/x_end.
    """
    if not x_axes or len(x_axes) < 2:
        return x_axes
        
    x_start = x_axes[0]
    # Protect against division by zero just in case
    x_end = 1.0 / x_axes[-1] if x_axes[-1] != 0 else 1.0 
    
    scale_factors = np.linspace(x_start, x_end, len(x_axes))
    x_scaled = np.array(x_axes) * scale_factors
    
    return x_scaled.tolist()

def scale_y_axis(y_axes):
    """
    Applies a ramping scale that forces the Y-axis to taper to 0.
    """
    if not y_axes or len(y_axes) < 2:
        return y_axes
        
    scale_factors = np.linspace(1.0, 0.0, len(y_axes))
    y_scaled = np.array(y_axes) * scale_factors
    
    return y_scaled.tolist()

def compute_z_axis(x_axes, y_axes, z_axis_avg):
    """
    Recalculates the Z-axis to enforce the 3D unit vector constraint:
    X^2 + Y^2 + Z^2 = 1.
    """
    z_axes = []
    # Determine which hemisphere the Z-axis belongs in
    sign = -1.0 if z_axis_avg < 0 else 1.0
    
    for x, y in zip(x_axes, y_axes):
        squared_sum = (x ** 2) + (y ** 2)
        
        # Prevent domain errors (math.sqrt of a negative number)
        # If X^2 + Y^2 somehow exceeded 1, we cap Z at 0
        z_val = max(1.0 - squared_sum, 0.0)
        
        z_axes.append(sign * math.sqrt(z_val))
        
    return z_axes


def spin_post_processing(json_path: os.PathLike[str],
                         save_path: os.PathLike[str],
                         video_path: os.PathLike[str]):
    
    # Load .json data
    with open(f"{json_path}.json", "r") as f:
        data = json.load(f)

    x_axes = [frame["x_axis"] for frame in data.values()]
    y_axes = [frame["y_axis"] for frame in data.values()]
    z_axes = [frame["z_axis"] for frame in data.values()]
    frames = [frame["frame"] for frame in data.values()]
    angles = [frame["angle"] for frame in data.values()]
    
    plotting(frames, x_axes, y_axes, z_axes, angles,
             "Imported data")


    # Get the original's video frame count and rate for interpolation reference
    CAP = cv2.VideoCapture(video_path)
    n_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = CAP.get(cv2.CAP_PROP_FPS)
    print(f"Original video has {n_frames} frames at {fps} FPS.")
    CAP.release()

    # Vector unflipping
    # Calculate the median of each axis 
    med_x = statistics.median(x_axes)
    med_y = statistics.median(y_axes)
    med_z = statistics.median(z_axes)

    for key, frame_data in data.items():
        # Calculate the 3D dot product against the reference vector
        dot_product = (
            (frame_data["x_axis"] * med_x) +
            (frame_data["y_axis"] * med_y) +
            (frame_data["z_axis"] * med_z)
        )
        
        # If the dot product is negative, the vector is backwards. Flip it.
        if dot_product < 0:
            frame_data["x_axis"] *= -1
            frame_data["y_axis"] *= -1
            frame_data["z_axis"] *= -1

    # Outlier removal passes
    x_axes = [frame["x_axis"] for frame in data.values()]
    y_axes = [frame["y_axis"] for frame in data.values()]
    z_axes = [frame["z_axis"] for frame in data.values()]
    frames = [frame["frame"] for frame in data.values()]

    x_axes_clean, y_axes_clean = remove_axes_outliers(x_axes, y_axes, frames)
    angles_clean = remove_angle_outliers(angles, threshold=0.5)
    #plotting(frames, x_axes_clean, y_axes_clean, z_axes, angles_clean, "Outlier removal")

    # Interpolation process
    x_axes_filled = fill_gaps(x_axes_clean)
    y_axes_filled = fill_gaps(y_axes_clean)
    angles_filled = fill_gaps(angles)
    #plotting(frames, x_axes_filled, y_axes_filled, z_axes, angles_filled, "Linear interpolation")

    x_axes_smoothed = gaussian_smoothing(x_axes_filled, sigma=10.0)
    y_axes_smoothed = gaussian_smoothing(y_axes_filled, sigma=10.0)
    #plotting(frames, x_axes_smoothed, y_axes_smoothed, z_axes, angles_filled, "Gaussian smoothing")

    # Monotonic x-axis rotation
    x_axes_monotonic = enforce_non_decreasing_x(x_axes_smoothed)
    #plotting(frames, x_axes_monotonic, y_axes_smoothed, z_axes, angles_filled, "Monotonic x-axis rotation")

    # Scale the axes
    x_axes_scaled = scale_x_axis(x_axes_monotonic)
    y_axes_scaled = scale_y_axis(y_axes_smoothed)
    #plotting(frames, x_axes_scaled, y_axes_scaled, z_axes, angles_filled, "X and Y axis scaling")

    # Recalculate the Z-Axis so X^2 + Y^2 + Z^2 = 1
    z_axis_avg = statistics.mean(z_axes) if z_axes else 1.0
    z_axes_final = compute_z_axis(x_axes_scaled, y_axes_scaled, z_axis_avg)
    #plotting(frames, x_axes_scaled, y_axes_scaled, z_axes_final, angles_filled, "Final z-axis calculation")


    final_angles = [frame["angle"] for frame in data.values() if frame["angle"] is not None]
    
    if final_angles:
        # 2. Calculate the average radians per frame
        avg_rads_per_frame = statistics.mean(final_angles)
        
        # 3. Apply the RPM conversion formula
        fps = fps
        rpm = (avg_rads_per_frame * fps / (2 * math.pi)) * 60.0
        
        print(f"Average Angle (Rads/Frame): {avg_rads_per_frame:.4f}")
        print(f"Calculated Rev Rate: {rpm:.1f} RPM")
    else:
        print("No valid angles remaining to calculate RPM.")

    """ 
    first_frames = list(data.values())[:5]
    
    if first_frames:
        avg_x = statistics.mean([f["x_axis"] for f in first_frames])
        avg_y = statistics.mean([f["y_axis"] for f in first_frames])
        avg_z = statistics.mean([f["z_axis"] for f in first_frames])
        
        # Axis Tilt: The vertical angle from the horizon
        # We use abs(avg_y) assuming the Y axis represents vertical tilt
        tilt_rads = math.asin(abs(avg_y))
        tilt_degrees = math.degrees(tilt_rads)
        
        # Axis Rotation: The angle relative to the foul line (X axis)
        # We use atan2 to safely handle division by zero
        rotation_rads = math.atan2(abs(avg_z), abs(avg_x))
        rotation_degrees = math.degrees(rotation_rads)
        
        print(f"Axis Tilt: {tilt_degrees:.1f}°")
        print(f"Axis Rotation: {rotation_degrees:.1f}°")
    """

    for i, (key, frame_data) in enumerate(data.items()):
        frame_data["x_axis"] = x_axes_scaled[i]
        frame_data["y_axis"] = y_axes_scaled[i]
        frame_data["z_axis"] = z_axes_final[i]
        frame_data["angle"] = angles_filled[i]

    with open(f"{save_path}_postprocessing.json", 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    PROJECT_ROOT = f"{Path().resolve()}"

    clip = "clip_1"
    extension = ".mp4"
    video_path = f"{PROJECT_ROOT}\\src\\ball_detection\\preprocessing_output\\preprocessed_{clip}"
    original_video_path = f"{PROJECT_ROOT}\\data\\clips\\{clip}{extension}"
    print(f"Original video path: {original_video_path}")
    json_path = f"{PROJECT_ROOT}\\src\\spin\\spin_output\\{clip}_spin"
    save_path = f"{PROJECT_ROOT}\\src\\spin\\postprocessing_output\\{clip}"
    trajectory_path = f"{PROJECT_ROOT}\\src\\ball_detection\\postprocessing_output\\postprocessed_{clip}.json"

    spin_post_processing(json_path, save_path, video_path, original_video_path, trajectory_path)

    print(-1)