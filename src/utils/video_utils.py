import os
from typing import Iterable, Sequence, Tuple

import cv2

Point = Tuple[int, int, int]


def generate_trajectory_video(
    input_video_path: str,
    points: Sequence[Point],
    output_video_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
) -> None:
    """Create a new video with a progressive trajectory drawn from point frames.

    Args:
        input_video_path: Path to the source video.
        points: Sequence of (frame, x, y) tuples.
        output_video_path: Path to write the trajectory video.
        color: BGR color used for the trajectory lines and points.
        thickness: Thickness of the trajectory lines.
        point_radius: Radius of the drawn point markers.
    """

    if not points.any():
        raise ValueError("The points list must contain at least one (frame, x, y) tuple.")

    normalized_points = []
    for frame_index, x, y in points:
        normalized_points.append((int(frame_index), int(x), int(y)))

    normalized_points.sort(key=lambda p: p[0])

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise IOError(f"Cannot open output video for writing: {output_video_path}")

    points_by_frame = {}
    for frame_index, x, y in normalized_points:
        points_by_frame.setdefault(frame_index, []).append((x, y))

    trajectory = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in points_by_frame:
            trajectory.extend(points_by_frame[frame_index])

        if trajectory:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)
            for point in trajectory:
                cv2.circle(frame, point, point_radius, color, -1)

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def generate_trajectory_video_from_iterable(
    input_video_path: str,
    points: Iterable[Point],
    output_video_path: str,
    **kwargs,
) -> None:
    """Helper wrapper that accepts any iterable of points."""
    generate_trajectory_video(input_video_path, list(points), output_video_path, **kwargs)


