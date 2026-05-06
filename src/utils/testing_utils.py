from lane_detection.lane_detection import get_bottom_lane_boundary


def parameter_search(
    frame,
    grayscale_methods=None,
    edge_methods=None,
):
    """Search for the best grayscale+edge combination (saves debug images)."""

    if grayscale_methods is None:
        grayscale_methods = [
            "default",
            "lightness",
            "luminosity",
            "r_g_minus_b",
            "pca",
        ]
    if edge_methods is None:
        edge_methods = ["sobel", "canny", "laplacian"]

    results = {}
    for gray_method in grayscale_methods:
        for edge_method in edge_methods:
            print(f"Searching: grayscale={gray_method}, edge={edge_method}")

            # Reduce noise for Sobel/Laplacian
            if edge_method in ("sobel", "laplacian"):
                edge_threshold = 30
                hough_threshold = 150
                hough_min_line_length = 150
                hough_max_line_gap = 8
            else:
                edge_threshold = None
                hough_threshold = 50
                hough_min_line_length = 100
                hough_max_line_gap = 10

            best_line = get_bottom_lane_boundary(
                frame,
                edge_method=edge_method,
                conv_method=gray_method,
                edge_threshold=edge_threshold,
                hough_threshold=hough_threshold,
                hough_min_line_length=hough_min_line_length,
                hough_max_line_gap=hough_max_line_gap,
            )
            results[(gray_method, edge_method)] = best_line

    return results
