from pathlib import Path
import argparse
import cv2
import numpy as np

from lane_detection.lane_detection import (
    detect_edges,
    get_bottom_lane_boundary,
    get_lateral_lane_boundaries,
    get_top_lane_boundary,
    postprocess_boundary_lines,
    preprocess_frame,
)
from utils.utils import crop_by_ratio, load_config

DEFAULT_GRAYSCALE_METHODS = [
    "default",
    "lightness",
    "hue",
    "r_g_minus_b",
    "pca",
]
DEFAULT_EDGE_METHODS = ["sobel", "canny", "laplacian"]


def _draw_line(frame, line, color=(0, 255, 0), thickness=3):
    vis = frame.copy()
    if line is not None:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def _draw_lines(frame, lines, color=(0, 255, 0), thickness=1):
    vis = frame.copy()
    if lines is None:
        return vis
    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def _draw_quad(frame, corners, color=(0, 255, 255), thickness=3):
    vis = frame.copy()
    if corners is None or any(c is None for c in corners):
        return vis
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
    return vis


def _save_image(path, image, label=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = image.copy()
    if label is not None:
        cv2.putText(
            vis,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), vis)


def _hough_parameters(edge_method):
    if edge_method in ("sobel", "laplacian"):
        return {
            "edge_threshold": 30,
            "hough_threshold": 50,
            "hough_min_line_length": 100,
            "hough_max_line_gap": 10,
        }
    return {
        "edge_threshold": None,
        "hough_threshold": 50,
        "hough_min_line_length": 100,
        "hough_max_line_gap": 10,
    }


def _extract_hough_segments(frame, conv_method, edge_method, direction, crop_region=None):
    blurred = preprocess_frame(frame, conv_method=conv_method)
    top_offset = 0
    left_offset = 0
    if crop_region is not None:
        cropped, top_offset, left_offset = crop_by_ratio(blurred, crop_region)
    else:
        cropped = blurred

    params = _hough_parameters(edge_method)
    edges = detect_edges(
        cropped,
        method=edge_method,
        conv_method=conv_method,
        edge_threshold=params["edge_threshold"],
        direction=direction,
        debug_prefix=f"parameter_search_{direction}_",
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=params["hough_threshold"],
        minLineLength=params["hough_min_line_length"],
        maxLineGap=params["hough_max_line_gap"],
    )

    segments = []
    original_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append((x1, y1, x2, y2))
            original_segments.append(
                (x1 + left_offset, y1 + top_offset, x2 + left_offset, y2 + top_offset)
            )
    return segments, original_segments, cropped


def parameter_search(
    frame,
    grayscale_methods=None,
    edge_methods=None,
    output_dir="debug/parameter_search",
):
    """Search for the best grayscale + edge combo and save line visualizations.

    This routine runs bottom lane boundary search for every combination of the
    specified grayscale and edge methods, then saves an annotated image for
    each result.
    """

    if grayscale_methods is None:
        grayscale_methods = DEFAULT_GRAYSCALE_METHODS
    if edge_methods is None:
        edge_methods = DEFAULT_EDGE_METHODS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for gray_method in grayscale_methods:
        for edge_method in edge_methods:
            combo_name = f"{gray_method}_{edge_method}"
            print(f"Searching: grayscale={gray_method}, edge={edge_method}")

            params = _hough_parameters(edge_method)
            best_line = get_bottom_lane_boundary(
                frame,
                edge_method=edge_method,
                conv_method=gray_method,
                edge_threshold=params["edge_threshold"],
                hough_threshold=params["hough_threshold"],
                hough_min_line_length=params["hough_min_line_length"],
                hough_max_line_gap=params["hough_max_line_gap"],
            )
            results[(gray_method, edge_method)] = best_line

            method_label = f"{gray_method}+{edge_method}"
            image_filename = output_path / f"bottom_line_{combo_name}.png"
            _save_image(_draw_line(frame, best_line), image_filename, method_label)

    print(f"Saved parameter search line images to: {output_path}")
    return results


def process_method(
    frame,
    template,
    lane_center,
    conv_method,
    edge_method,
    output_path,
):
    output_path.mkdir(parents=True, exist_ok=True)
    combo_label = f"{conv_method}+{edge_method}"

    bottom_segments, bottom_segments_orig, cropped_bottom = _extract_hough_segments(
        frame,
        conv_method=conv_method,
        edge_method=edge_method,
        direction="horizontal",
        crop_region=[0.6, 1.0, 0.15, 0.85],
    )
    _save_image(
        output_path / "bottom_segments_cropped.png",
        _draw_lines(cropped_bottom, bottom_segments, color=(0, 255, 0), thickness=1),
        f"{combo_label} bottom segments",
    )
    _save_image(
        output_path / "bottom_segments.png",
        _draw_lines(frame, bottom_segments_orig, color=(0, 255, 0), thickness=1),
        f"{combo_label} bottom segments",
    )

    bottom_line = get_bottom_lane_boundary(
        frame,
        edge_method=edge_method,
        conv_method=conv_method,
        edge_threshold=_hough_parameters(edge_method)["edge_threshold"],
        hough_threshold=_hough_parameters(edge_method)["hough_threshold"],
        hough_min_line_length=_hough_parameters(edge_method)["hough_min_line_length"],
        hough_max_line_gap=_hough_parameters(edge_method)["hough_max_line_gap"],
    )
    _save_image(
        output_path / "bottom_best_line.png",
        _draw_line(frame, bottom_line, color=(0, 255, 0), thickness=3),
        f"{combo_label} bottom best",
    )

    lateral_segments, lateral_segments_orig, _ = _extract_hough_segments(
        frame,
        conv_method=conv_method,
        edge_method=edge_method,
        direction="vertical",
        crop_region=None,
    )
    _save_image(
        output_path / "lateral_segments.png",
        _draw_lines(frame, lateral_segments_orig, color=(255, 128, 0), thickness=1),
        f"{combo_label} lateral segments",
    )

    lateral_left, lateral_right = get_lateral_lane_boundaries(
        frame,
        edge_method=edge_method,
        conv_method=conv_method,
        edge_threshold=_hough_parameters(edge_method)["edge_threshold"],
        hough_threshold=_hough_parameters(edge_method)["hough_threshold"],
        hough_min_line_length=_hough_parameters(edge_method)["hough_min_line_length"],
        hough_max_line_gap=_hough_parameters(edge_method)["hough_max_line_gap"],
        direction="vertical",
        lane_center=lane_center,
    )
    lateral_best = frame.copy()
    if lateral_left is not None:
        lateral_best = _draw_line(lateral_best, lateral_left, color=(0, 255, 0), thickness=3)
    if lateral_right is not None:
        lateral_best = _draw_line(lateral_best, lateral_right, color=(0, 255, 0), thickness=3)
    _save_image(
        output_path / "lateral_best_lines.png",
        lateral_best,
        f"{combo_label} lateral best",
    )

    return bottom_line, (lateral_left, lateral_right)


def process_video(
    video_path,
    config_path="src/config.yaml",
    grayscale_methods=None,
    edge_methods=None,
    output_root="debug/parameter_search",
):
    config = load_config(config_path)
    template = cv2.imread(config.paths.template_pin_path)
    if template is None:
        raise IOError(f"Cannot load template pin image: {config.paths.template_pin_path}")

    if grayscale_methods is None:
        grayscale_methods = DEFAULT_GRAYSCALE_METHODS
    if edge_methods is None:
        edge_methods = DEFAULT_EDGE_METHODS

    video_path = Path(video_path)
    output_root = Path(output_root) / video_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(str(video_path))
    ret, frame = vid.read()
    vid.release()
    if not ret or frame is None:
        raise IOError(f"Unable to read frame from video: {video_path}")

    top_line = get_top_lane_boundary(
        frame,
        template,
        mode=config.lane_detection.top.mode,
    )
    _save_image(
        output_root / "top_boundary.png",
        _draw_line(frame, top_line, color=(255, 255, 0), thickness=3),
        "top boundary",
    )

    if top_line is None:
        print(f"Warning: top boundary failed for video {video_path}")

    # Determine lane center for this video. Prefer per-clip override in config,
    # falling back to the global `lane_center_point`.
    video_stem = video_path.stem
    lane_center = None
    if hasattr(config.points, "lane_center_points"):
        # `lane_center_points` is loaded as a SimpleNamespace, so use getattr
        lane_center = getattr(config.points.lane_center_points, video_stem, None)
    if lane_center is None:
        lane_center = config.points.lane_center_point

    for gray_method in grayscale_methods:
        for edge_method in edge_methods:
            combo_output = output_root / f"{gray_method}_{edge_method}"
            print(f"Processing {video_path.name}: {gray_method}+{edge_method}")
            bottom_line, lateral_lines = process_method(
                frame,
                template,
                lane_center,
                gray_method,
                edge_method,
                combo_output,
            )
            quad = None
            if top_line is not None and bottom_line is not None and lateral_lines[0] is not None and lateral_lines[1] is not None:
                quad = postprocess_boundary_lines(bottom_line, lateral_lines, top_line)
            _save_image(
                combo_output / "lane_quadrilateral.png",
                _draw_quad(frame, quad),
                f"{gray_method}+{edge_method} quad",
            )
            summary_path = combo_output / "summary.txt"
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                summary_file.write(f"video={video_path.name}\n")
                summary_file.write(f"method={gray_method}+{edge_method}\n")
                summary_file.write(f"bottom_line={bottom_line is not None}\n")
                summary_file.write(f"left_line={lateral_lines[0] is not None if lateral_lines is not None else False}\n")
                summary_file.write(f"right_line={lateral_lines[1] is not None if lateral_lines is not None else False}\n")
                summary_file.write(f"top_line={top_line is not None}\n")
                summary_file.write(f"quadrilateral={quad is not None}\n")


def run_all_videos(
    clips_dir="data/clips",
    config_path="src/config.yaml",
    output_root="debug/parameter_search",
    grayscale_methods=None,
    edge_methods=None,
):
    clips_dir = Path(clips_dir)
    video_paths = sorted(
        [
            *clips_dir.glob("*.mp4"),
            *clips_dir.glob("*.mov"),
        ]
    )
    if not video_paths:
        raise IOError(f"No video files found in {clips_dir}")

    for video_path in video_paths:
        process_video(
            video_path,
            config_path=config_path,
            grayscale_methods=grayscale_methods,
            edge_methods=edge_methods,
            output_root=output_root,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run lane boundary parameter search for all video clips.")
    parser.add_argument(
        "--clips-dir",
        default="data/clips",
        help="Directory containing the input video clips.",
    )
    parser.add_argument(
        "--config",
        default="src/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug/parameter_search",
        help="Base directory where parameter search outputs are saved.",
    )
    parser.add_argument(
        "--grayscale-methods",
        nargs="+",
        default=DEFAULT_GRAYSCALE_METHODS,
        help="List of grayscale conversion methods.",
    )
    parser.add_argument(
        "--edge-methods",
        nargs="+",
        default=DEFAULT_EDGE_METHODS,
        help="List of edge detection methods.",
    )
    return parser.parse_args()




