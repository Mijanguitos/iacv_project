import os
from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Point = Tuple[int, int, int]
Point2D = Tuple[float, float]


def _normalize_trajectory_points(points: Sequence[Point]) -> list[Point]:
    normalized_points = []
    for frame_index, x, y in points:
        normalized_points.append((int(frame_index), int(x), int(y)))
    normalized_points.sort(key=lambda p: p[0])
    return normalized_points


def _draw_trajectory_points(
    image: np.ndarray,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
    point_radius: int,
) -> None:
    if len(points) < 1:
        return

    for i in range(1, len(points)):
        cv2.line(image, points[i - 1], points[i], color, thickness)
    for point in points:
        cv2.circle(image, point, point_radius, color, -1)


def _ensure_valid_points(points: Sequence[Point]) -> list[Point]:
    if not len(points):
        raise ValueError("The points list must contain at least one (frame, x, y) tuple.")
    return _normalize_trajectory_points(points)


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

    normalized_points = _ensure_valid_points(points)

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


def _make_perspective_transform(
    source_corners: Sequence[Point2D],
    destination_corners: Sequence[Point2D],
) -> np.ndarray:
    if len(source_corners) != 4 or len(destination_corners) != 4:
        raise ValueError("Both source and destination corner lists must contain exactly 4 points.")

    src = np.array(source_corners, dtype=np.float32)
    dst = np.array(destination_corners, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _transform_lane_points_to_board(
    lane_points: Sequence[Point2D],
    H: np.ndarray,
    board_shape: Tuple[int, int, int],
) -> np.ndarray:
    if not lane_points:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.array(lane_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    transformed_int = np.round(transformed).astype(np.int32)
    transformed_int[:, 0] = np.clip(transformed_int[:, 0], 0, board_shape[1] - 1)
    transformed_int[:, 1] = np.clip(transformed_int[:, 1], 0, board_shape[0] - 1)
    return transformed_int


def generate_trajectory_board_image(
    board_template_path: str,
    lane_points: Sequence[Point2D],
    source_lane_corners: Sequence[Point2D],
    output_image_path: Optional[str] = None,
    destination_board_corners: Optional[Sequence[Point2D]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    """Draw a trajectory on a warped lane board using a lane-to-board homography.

    Args:
        board_template_path: Path to the board template image.
        lane_points: Sequence of lane points in the same coordinate space as source_lane_corners.
        source_lane_corners: Four corners of the lane in source coordinates.
        output_image_path: Optional path to save the generated board image.
        destination_board_corners: Four points in board-template coordinates. If None, uses image corners.
        color: BGR color used for the trajectory.
        thickness: Thickness of the trajectory polyline.
        point_radius: Radius of the trajectory markers.

    Returns:
        The board image with the trajectory drawn.
    """

    board = cv2.imread(board_template_path)
    if board is None:
        raise IOError(f"Cannot read board template image: {board_template_path}")

    if destination_board_corners is None:
        height, width = board.shape[:2]
        destination_board_corners = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]

    H = _make_perspective_transform(source_lane_corners, destination_board_corners)
    transformed_points = _transform_lane_points_to_board(lane_points, H, board.shape)
    board_with_trajectory = board.copy()
    _draw_trajectory_points(
        board_with_trajectory,
        [tuple(pt) for pt in transformed_points],
        color,
        thickness,
        point_radius,
    )

    if output_image_path:
        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
        cv2.imwrite(output_image_path, board_with_trajectory)

    return board_with_trajectory


def generate_trajectory_board_video(
    input_video_path: str,
    points: Sequence[Point],
    board_template_path: str,
    source_lane_corners: Sequence[Point2D],
    output_video_path: str,
    destination_board_corners: Optional[Sequence[Point2D]] = None,
    board_position: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
    board_margin: int = 10,
) -> None:
    """Create a video that places a warped board overlay with the trajectory in the top-right area.

    The trajectory points are transformed from the lane coordinate system into the board template
    via homography, and the progressive trajectory is drawn onto the board overlay.
    """

    normalized_points = _ensure_valid_points(points)

    board_template = cv2.imread(board_template_path)
    if board_template is None:
        raise IOError(f"Cannot read board template image: {board_template_path}")

    if destination_board_corners is None:
        height, width = board_template.shape[:2]
        destination_board_corners = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]

    H = _make_perspective_transform(source_lane_corners, destination_board_corners)

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    board_height, board_width = board_template.shape[:2]
    if board_width > frame_width or board_height > frame_height:
        capture.release()
        raise ValueError("Board template is larger than the input video frame.")

    if board_position is None:
        board_position = (frame_width - board_width - board_margin, board_margin)

    position_x, position_y = board_position
    if position_x < 0 or position_y < 0:
        capture.release()
        raise ValueError("Board position must be inside the video frame.")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        capture.release()
        raise IOError(f"Cannot open output video for writing: {output_video_path}")

    points_by_frame = {}
    for frame_index, x, y in normalized_points:
        points_by_frame.setdefault(frame_index, []).append((float(x), float(y)))

    trajectory_points: list[Point2D] = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in points_by_frame:
            trajectory_points.extend(points_by_frame[frame_index])

        board_overlay = board_template.copy()
        if trajectory_points:
            transformed_points = _transform_lane_points_to_board(trajectory_points, H, board_overlay.shape)
            _draw_trajectory_points(
                board_overlay,
                [tuple(pt) for pt in transformed_points],
                color,
                thickness,
                point_radius,
            )

        frame[position_y:position_y + board_height, position_x:position_x + board_width] = board_overlay
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def generate_trajectory_video_with_board(
    input_video_path: str,
    image_points: Sequence[Point],
    board_points: Sequence[Point],
    board_template_path: str,
    source_lane_corners: Sequence[Point2D],
    output_video_path: str,
    destination_board_corners: Optional[Sequence[Point2D]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
    board_margin: int = 10,
) -> None:
    """Create a single video with both frame trajectory and board trajectory.

    The output video shows the input video on the left with the ball trajectory drawn
    progressively, and the board template on the right with the corresponding warped
    trajectory also drawn progressively.
    """

    normalized_image_points = _ensure_valid_points(image_points)
    normalized_board_points = _ensure_valid_points(board_points)

    board_template = cv2.imread(board_template_path)
    if board_template is None:
        raise IOError(f"Cannot read board template image: {board_template_path}")

    if destination_board_corners is None:
        height, width = board_template.shape[:2]
        destination_board_corners = [
            (0.0, 0.0),
            (width - 1.0, 0.0),
            (width - 1.0, height - 1.0),
            (0.0, height - 1.0),
        ]

    H = _make_perspective_transform(source_lane_corners, destination_board_corners)

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    board_height, board_width = board_template.shape[:2]
    output_width = frame_width + board_width + board_margin
    output_height = max(frame_height, board_height)

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        capture.release()
        raise IOError(f"Cannot open output video for writing: {output_video_path}")

    image_points_by_frame = {}
    for frame_index, x, y in normalized_image_points:
        image_points_by_frame.setdefault(frame_index, []).append((x, y))

    board_points_by_frame = {}
    for frame_index, x, y in normalized_board_points:
        board_points_by_frame.setdefault(frame_index, []).append((x, y))

    trajectory_image_points: list[Tuple[int, int]] = []
    trajectory_board_points: list[Point2D] = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in image_points_by_frame:
            trajectory_image_points.extend(image_points_by_frame[frame_index])
        if frame_index in board_points_by_frame:
            trajectory_board_points.extend(board_points_by_frame[frame_index])

        frame_with_trajectory = frame.copy()
        if trajectory_image_points:
            _draw_trajectory_points(
                frame_with_trajectory,
                trajectory_image_points,
                color,
                thickness,
                point_radius,
            )

        board_overlay = board_template.copy()
        if trajectory_board_points:
            transformed_points = _transform_lane_points_to_board(trajectory_board_points, H, board_overlay.shape)
            _draw_trajectory_points(
                board_overlay,
                [tuple(pt) for pt in transformed_points],
                color,
                thickness,
                point_radius,
            )

        output_frame = np.zeros((output_height, output_width, 3), dtype=frame.dtype)
        output_frame[0:frame_height, 0:frame_width] = frame_with_trajectory
        output_frame[0:board_height, frame_width + board_margin: frame_width + board_margin + board_width] = board_overlay

        writer.write(output_frame)
        frame_index += 1

    capture.release()
    writer.release()


def generate_trajectory_board_video_from_iterable(
    input_video_path: str,
    points: Iterable[Point],
    board_template_path: str,
    source_lane_corners: Sequence[Point2D],
    output_video_path: str,
    **kwargs,
) -> None:
    """Helper wrapper that accepts any iterable of points."""
    generate_trajectory_board_video(
        input_video_path,
        list(points),
        board_template_path,
        source_lane_corners,
        output_video_path,
        **kwargs,
    )







