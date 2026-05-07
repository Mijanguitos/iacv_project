# -*- coding: utf-8 -*-
"""Video preprocessing for bowling ball detection.

Generates preprocessed video based on the provided initial clip.
This script contains the functions to perform the pertinent video's 
preprocessing for the bowling ball detection. The proposed workflow
consists of:

    - Masking the pixels that do not belong to the lane where the ball
      should appear (assuming a static camera).
    - Blurring and filtering the frames to reduce noise and artifacts.
    - Applying localized contrast enhancement to improve the distinction
      between the background (wooden lane) and the ball.

Finally, it generates and saves another video in the provided path for 
the next steps.

Usage example:

    * To run tests:

    $ python preprocessing.py

    * Within a program:

    preprocessing.video_preprocessing(video_path, save_path, lane_points)

"""
import numpy as np
import cv2 as cv2
import os


def compute_modified_polygon(points: np.ndarray, 
                             upper_padding: int,
                             side_padding: int) -> np.ndarray:
    """ Defines new region of interest (modified polygon of the lane).
    Computes a modified polygon based on the top two points of the input points.

    Args:
        polygon (np.ndarray): Array of points representing the polygon.
        upper_padding (int): Amount of pixels to add to the poligon on top.
        side_padding (int): Amount of pixels to add to the poligon on the sides.
    Returns:
        np.ndarray: Modified polygon points.
    """
    top_indices = np.argsort(points[:, 1])[:2]
    top_points = points[top_indices]
    left_top, right_top = sorted(top_points, key=lambda pt: pt[0])


    left_top_mod = [left_top[0] - side_padding, left_top[1] - upper_padding]
    right_top_mod = [right_top[0] + side_padding, right_top[1] - upper_padding]
    

    modified_polygon = np.array(
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

    return modified_polygon


def frame_preprocessing(frame: cv2.typing.MatLike, 
                        mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """ Preprocessing of individual frames.
    Apply preprocessing steps for the given frame: bitwise masking the pixels that do not correspond to
    the lane, then median blurring to reduce high-frequency noise in the frame, followed by chainging its
    colorspace to grayscale so that, finally, an adaptative histrogram equialization (CLAHE) can be 
    applyed to increase the contrast and the ball can be more distinguishable for the following steps.
    Args:
        frame (cv2.typing.MatLike): Input frame extracted from the initial video.
        mask (cv2.typing.MatLike): Binary mask of the region of interest.
    Returns:
        cv2.typing.MatLike: Preprocessed frame.
    """
    # Binary mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Reduce high-frequency noise by applying median blurring
    median_frame = cv2.medianBlur(masked_frame, 9)

    # Black and white image for CLAHE
    bnw_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    
    # Create CLAHE object (localized histogram equalization)
    clahe = cv2.createCLAHE(clipLimit = 2.5,
                            tileGridSize = (15, 15))
    preprocessed_frame = clahe.apply(bnw_frame)

    return preprocessed_frame

def video_preprocessing(video_path: os.PathLike[str],
                        save_path: os.PathLike[str],
                        lane_points: np.ndarray) -> None:
    """ Preprocessing of whole video.
    Reads the video file from path, applies the preprocessing steps for each frame and saves it as a 
    video in the provided path.
    Args:
        video_path (os.PathLike[str]): Path where the original clip is found.
        save_path (os.PathLike[str]): Path where the preprocessed output is to be saved.
        lane_points (np.ndarray): Array of points that define the lane's polygon.

    Note: It is assumed that the camera is static or movements are already corrected.
    """

    print(f"Ball detection: preprocessing {video_path}")

    # Load the video clip and get total number of frames and frame rate
    CAP = cv2.VideoCapture(video_path)                  # Class for video captured from the clip's file   
    n_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))   # Clip's total number of frames
    fps = CAP.get(cv2.CAP_PROP_FPS)                     # Clip's frame rate
    width  = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
    height = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)   # float `height`
    
    # Video writer class
    output = cv2.VideoWriter(save_path, -1, fps, (int(width), int(height)), isColor=False)

    # Compute the modified polygon for masking
    modified_polygon = compute_modified_polygon(lane_points, 15, 20)

    # Loops to perform preprocessing
    while True:
        ret, frame = CAP.read()

        if not ret:
            break
        
        # Get the mask on the frame's domain
        mask = cv2.fillPoly(np.zeros(frame.shape[:2], dtype=np.uint8), [modified_polygon], 255)

        # Preprocessing for each frame
        preprocessed_frame = frame_preprocessing(frame, mask)

        # Write preprocessed frame
        output.write(preprocessed_frame)

    output.release()

    print(f"Ball detection: preprocessed video saved at {save_path}")



if __name__ == "__main__":
    
    video_path = "clip_2.mp4"
    save_path = f"preprocessing_output\\preprocessed_{video_path}"
    lane_points = np.array([[819, 813], [1308, 819] , [1442, 175], [1254, 175]])

    modified_polygon = compute_modified_polygon(lane_points, 15, 20)

    video_preprocessing(video_path, save_path, lane_points)