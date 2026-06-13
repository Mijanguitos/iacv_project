# Bowling Kinematics & Lane Detection Pipeline

A comprehensive computer vision pipeline designed for tracking bowling ball trajectories, estimating 3D spin kinematics, and rectifying lane perspectives. This repository orchestrates multiple advanced image processing modules to extract high-fidelity motion data from standard video clips.

## Features

* **Lane Boundary Detection:** Utilizes Sobel edge detection, Hough transforms, and RANSAC regression to identify the bottom, lateral, and top boundaries of the bowling lane.
* **Perspective Rectification:** Computes a homography matrix to project the 2D camera view into a top-down, rectified 2D space.
* **Ball Detection & Tracking:** Isolates the bowling ball frame-by-frame to map continuous trajectory coordinates.
* **Spin Estimation & Visualization:** Processes optical flow data to estimate the ball's angular velocity and spin axis, rendering a 3D augmented reality overlay onto the original video.

## Mathematical Approach

The motion estimation and post-processing steps are grounded in rigorous kinematic calculations to handle noise and axis-sign ambiguity:
* **Feature Tracking:** Outliers and falsely tracked features are filtered out of the video processing pipeline using forward-backward optical flow validation.
* **Kinematics:** The pipeline calculates rotation matrices by solving the orthogonal Procrustes problem, utilizing Singular Value Decomposition (SVD) within the Kabsch algorithm.
* **Coordinate Projection:** Transforming 2D features into 3D space assumes a ball-centric reference frame and a perfect sphere where $R\approx \frac{1}{2}\verb|frame_width|$. Smoothing is applied via a log function and direct interpolation.

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `src/ball_detection/` | Video preprocessing, candidate isolation, and trajectory post-processing. |
| `src/lane_detection/` | Boundary detection using templates and gradient thresholding. |
| `src/rectification/` | Homography transformations to map image coordinates to lane geometry. |
| `src/spin/` | Optical flow detection and mathematical post-processing for spin axes. |
| `src/utils/` | Shared utilities for plotting, JSON parsing, and configuration loading. |

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```
Core dependencies include opencv-python, numpy, scikit-learn, and pyyaml.

