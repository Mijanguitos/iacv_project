# -*- coding: utf-8 -*-
""" Bowling ball candidate detection.

Retrieves a preprocessed video and detects all the the possible bowling balls that might
appear in the scene. Then it proceeds to save them in a .json file for further processing.

Usage example:

    * To run tests:

    $ python detection.py

    * Within a program:

    detection.candidate_detection(video_path, save_path)

"""

import numpy as np
import cv2 as cv2
import json
import os


def circle_detection(frame: cv2.typing.MatLike) -> np.ndarray:
    """ Detects circles in the scene.
    Finds circles in the grayscale frame using the Hough transform.

    Args:
        frame (cv2.typing.MatLike): Input frame extracted from the initial video.

    Returns:
        np.ndarray: Array of detected circles in the scene.

    """
    circles = cv2.HoughCircles(frame,
                               method=cv2.HOUGH_GRADIENT_ALT,
                               dp=1.2,      #1.2                 # downsample size
                               minDist=45, #100          # minimum distance between detected cirlces
                               param1=250,  #300             # upper threshold for canny edge detection
                               param2=0.8,  #40             # cummulator (lower) threshold for canny edge detection
                               minRadius=15,
                               maxRadius=70)
                
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return np.squeeze(circles, axis=0)
    else:
        return None


def candidate_detection(video_path: os.PathLike[str],
                        save_path: os.PathLike[str]):
    """ Ball candidate detection.
    Reads the video file from path and applies circle detection for each framqe. Then generates
    a video that displays where the circles were found and a .json file containing all the
    bowling ball candidates for each frame.

    Args:
        video_path (os.PathLike[str]): Path where the original clip is found.
        save_path (os.PathLike[str]): Path where the ball candidates output is to be saved.
    """

    print(f"Ball detection: Ball candidate detection {video_path}")

    # Load the video clip and get total number of frames and frame rate
    CAP = cv2.VideoCapture(f"{video_path}.mp4")                  # Class for video captured from the clip's file
    n_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))   # Clip's total number of frames
    fps = CAP.get(cv2.CAP_PROP_FPS)                     # Clip's frame rate
    width  = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
    height = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)   # float `height`
    
    # Video writer class
    output = cv2.VideoWriter(f"{save_path}.mp4", -1, fps, (int(width), int(height)))

    # Dictionary that will contain all detected ball candidates
    ball_candidates = dict()
    i_frame = 0

    # Loop to perform ball candidate detection
    while True:
        ret, frame = CAP.read()

        if not ret:
            break

        # Get the grayscale colorspace frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Retrieve ball candidates from preprocessed frame 
        circles = circle_detection(gray_frame)

        # Add the frame's detected circles to the dictionary
        ball_candidates[i_frame] = circles.tolist() if circles is not None else None

        # Draw the candidates in the frame
        if circles is not None:
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a another
                # corresponding to the center of the circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 1, (0, 128, 255), -1)

        # Write frame
        output.write(frame)

        i_frame += 1
#ily
    output.release()

    with open(f"{save_path}.json", 'w') as f:
        json.dump(ball_candidates, f, indent=4)

    print(f"Ball detection: Ball candidates saved {video_path}")



if __name__ == "__main__":
    clip = "clip_2"
    video_path = f"preprocessing_output\\preprocessed_{clip}"
    save_path = f"detection_output\\candidates_{clip}"

    candidate_detection(video_path, save_path)
