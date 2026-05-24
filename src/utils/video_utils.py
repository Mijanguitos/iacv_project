import os
from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Point = Tuple[int, int, int]
Point2D = Tuple[float, float]
import plotly.graph_objects as go



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
    # if board_template is None:
    #     raise IOError(f"Cannot read board template image: {board_template_path}")

    # if destination_board_corners is None:
    #     height, width = board_template.shape[:2]
    #     destination_board_corners = [
    #         (0.0, 0.0),
    #         (width - 1.0, 0.0),
    #         (width - 1.0, height - 1.0),
    #         (0.0, height - 1.0),
    #     ]

    H = _make_perspective_transform(source_lane_corners, destination_board_corners)

    capture = cv2.VideoCapture(input_video_path)
    # if not capture.isOpened():
    #     raise IOError(f"Cannot open input video: {input_video_path}")

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



def create_3d_bowling_visualization(
    trajectory_3d,
    lane_width=1.066,
    lane_length=18.29
):
    """
    trajectory_3d: list of (frame_id, x, y, z)
    """

    trajectory_3d = sorted(trajectory_3d, key=lambda p: p[0])

    frames = []
    x_vals, y_vals, z_vals = [], [], []

    for frame_id, x, y, z in trajectory_3d:
        x_vals.append(x / 100)  # meters
        y_vals.append(lane_length - y / 100)  # meters
        z_vals.append(z)        # meters (already correct)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    fig = go.Figure()

    # =========================
    # Lane surface
    # =========================
    fig.add_trace(go.Mesh3d(
        x=[0, lane_width, lane_width, 0],
        y=[0, 0, lane_length, lane_length],
        z=[0, 0, 0, 0],
        color="lightgray",
        opacity=0.5
    ))

    # =========================
    # Origin marker (NEW)
    # =========================
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers+text",
        marker=dict(size=6, color="black"),
        text=["Origin"],
        textposition="top center",
        name="Origin"
    ))

    # =========================
    # Axis direction indicators (NEW but minimal)
    # =========================
    fig.add_trace(go.Scatter3d(
        x=[0, lane_width],
        y=[0, 0],
        z=[0, 0],
        mode="lines",
        line=dict(color="blue", width=4),
        name="X axis (width)"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, lane_length],
        z=[0, 0],
        mode="lines",
        line=dict(color="green", width=4),
        name="Y axis (length)"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 0.15],
        mode="lines",
        line=dict(color="red", width=4),
        name="Z axis (height)"
    ))

    # =========================
    # Full trajectory
    # =========================
    fig.add_trace(go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="lines",
        line=dict(color="red", width=5)
    ))

    # =========================
    # Animation frames
    # =========================
    for i in range(len(x_vals)):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=[x_vals[i]],
                    y=[y_vals[i]],
                    z=[z_vals[i]],
                    mode="markers",
                    marker=dict(size=8, color="blue")
                )
            ],
            name=str(i)
        ))

    # Initial ball
    fig.add_trace(go.Scatter3d(
        x=[x_vals[0]],
        y=[y_vals[0]],
        z=[z_vals[0]],
        mode="markers",
        marker=dict(size=8, color="blue")
    ))

    fig.frames = frames

    # =========================
    # Layout (unchanged except camera improvement)
    # =========================
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Lane width (m)", range=[0, lane_width]),
            yaxis=dict(title="Lane length (m)", range=[0, lane_length]),
            zaxis=dict(title="Height (m)", range=[0, 0.15]),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.8, y=-2.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            )
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 30, "redraw": True},
                        "fromcurrent": True
                    }]
                )
            ]
        )]
    )

    fig.show()