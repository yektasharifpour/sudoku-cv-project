"""
src/phase1_debug_visualize.py

Saves every Phase 1 stage's output as an actual image file, for a handful of
sample images, instead of only the final pass/fail result. This is also
exactly the "processing stage images" deliverable the project brief asks
for (Phase 1's report section wants gray/threshold/contour/etc previews,
not just a final cropped grid).

For each input image, writes into <output_dir>/<image_stem>/:
    01_original.jpg
    02_grayscale.jpg
    03_threshold.jpg          (binary mask, ink = white)
    04_dilated.jpg             (the gap-bridging step -- see
                                phase1_contour_detection.py's fix #2)
    05_contour_overlay.jpg     (largest contour after dilation, in red)
    06_corners_overlay.jpg     (final 4 ordered corners, or a "NOT FOUND"
                                banner if corner detection failed)
    07_warped_grid.jpg         (top-down squared grid, if corners found)
    08_cells_contact_sheet.jpg (all 81 cropped cells in a 9x9 sheet)

Run:
    python src/phase1_debug_visualize.py --images datasets/sudoku_dataset/training --limit 5
    python src/phase1_debug_visualize.py --images path/to/specific/images --limit 0  # all of them
"""

import argparse
import glob
import os
import sys
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.phase1_preprocessing import to_binary_threshold
from src.phase1_contour_detection import (
    DILATION_ITERATIONS,
    DILATION_KERNEL,
    EPSILON_RANGE,
    order_corners,
)
from src.phase1_perspective_warp import SIDE, compute_transform, split_cells, warp_grid
from src.phase1_cell_cleanup import clean_cell


def _save(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def _find_corners_with_trace(thresh: np.ndarray):
    """Same logic as phase1_contour_detection.find_grid_corners(), but also
    returns the intermediate dilated mask + largest contour, so every stage
    can be saved as its own image."""
    dilated = cv2.dilate(thresh, DILATION_KERNEL, iterations=DILATION_ITERATIONS)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dilated, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    perimeter = cv2.arcLength(hull, True)

    for eps in np.arange(*EPSILON_RANGE):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return dilated, largest_contour, order_corners(approx)

    return dilated, largest_contour, None


def visualize_one(image_path: str, output_dir: str):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out = os.path.join(output_dir, stem)

    image = cv2.imread(image_path)
    if image is None:
        print(f"{stem}: could not read image, skipping")
        return

    _save(f"{out}/01_original.jpg", image)

    gray, thresh = to_binary_threshold(image)
    _save(f"{out}/02_grayscale.jpg", gray)
    _save(f"{out}/03_threshold.jpg", thresh)

    dilated, largest_contour, corners = _find_corners_with_trace(thresh)
    _save(f"{out}/04_dilated.jpg", dilated)

    contour_vis = image.copy()
    if largest_contour is not None:
        cv2.drawContours(contour_vis, [largest_contour], -1, (0, 0, 255), 3)
    _save(f"{out}/05_contour_overlay.jpg", contour_vis)

    corners_vis = image.copy()
    if corners is not None:
        for i, (x, y) in enumerate(corners):
            cv2.circle(corners_vis, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(corners_vis, str(i), (int(x) + 12, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(corners_vis, "CORNERS NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    _save(f"{out}/06_corners_overlay.jpg", corners_vis)

    if corners is None:
        print(f"{stem}: FAILED -- no 4-corner quadrilateral found, stopped after stage 06")
        return

    transform_matrix, _ = compute_transform(corners, side=SIDE)
    warped_gray, warped_thresh = warp_grid(gray, thresh, transform_matrix, side=SIDE)
    _save(f"{out}/07_warped_grid.jpg", warped_gray)

    raw_cells = split_cells(warped_thresh, side=SIDE)
    cleaned_cells = [clean_cell(c) for c in raw_cells]

    tile = 64
    sheet = np.zeros((9 * tile, 9 * tile), dtype="uint8")
    for i, cell in enumerate(cleaned_cells):
        r, c = divmod(i, 9)
        resized = cv2.resize(cell, (tile, tile))
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = resized
    _save(f"{out}/08_cells_contact_sheet.jpg", sheet)

    print(f"{stem}: PASSED -- all 8 stages saved to {out}/")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: save every processing stage as an image")
    parser.add_argument("--images", required=True, help="Folder of Sudoku photos, or a single image path")
    parser.add_argument("--limit", type=int, default=5, help="Max images to process (0 = all)")
    parser.add_argument("--output-dir", default="phase1_debug_output", help="Where to write stage images")
    args = parser.parse_args()

    if os.path.isfile(args.images):
        image_paths = [args.images]
    else:
        image_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
        if args.limit:
            image_paths = image_paths[: args.limit]

    print(f"Visualizing {len(image_paths)} image(s) into {args.output_dir}/ ...")
    for path in image_paths:
        visualize_one(path, args.output_dir)


if __name__ == "__main__":
    main()
