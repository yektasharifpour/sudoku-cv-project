"""
src/phase1_pipeline.py

Phase 1 -- orchestrator. Ties together the 4 stage modules into the full
grid-extraction pipeline, plus ground-truth loading and batch testing.

    stage 1: phase1_preprocessing.to_binary_threshold
    stage 2: phase1_contour_detection.find_grid_corners
    stage 3: phase1_perspective_warp.compute_transform / warp_grid / split_cells
    stage 4: phase1_cell_cleanup.clean_cell / is_empty

Run directly to batch-test against a folder of Sudoku images:
    python src/phase1_pipeline.py --images datasets/sudoku_dataset/training --checkpoint-dir checkpoints/phase1/train --limit 0
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from phase1_preprocessing import to_binary_threshold
from phase1_contour_detection import find_grid_corners
from phase1_perspective_warp import SIDE, compute_transform, split_cells, warp_grid
from phase1_cell_cleanup import clean_cell, is_empty


# --- Ground-truth labels from wichtounet/sudoku_dataset's .dat files -------

def load_ground_truth_grid(image_path: str) -> Optional[np.ndarray]:
    """Looks for a `<image_stem>.dat` file next to image_path and parses its
    9x9 digit grid. Returns a (9, 9) int array, or None if no matching .dat
    exists or it doesn't parse cleanly. Robust to a variable-length header
    by taking the last 9 lines that each split into exactly 9 integers."""
    dat_path = os.path.splitext(image_path)[0] + ".dat"
    if not os.path.exists(dat_path):
        return None

    with open(dat_path, "r", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    grid_lines = []
    for line in reversed(lines):
        tokens = line.split()
        if len(tokens) == 9 and all(_is_int(t) for t in tokens):
            grid_lines.append([int(t) for t in tokens])
        if len(grid_lines) == 9:
            break

    if len(grid_lines) != 9:
        return None

    grid_lines.reverse()
    return np.array(grid_lines, dtype=int)


def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


# --- Orchestrator -----------------------------------------------------------

def process_sudoku_image(
    image_path: str,
    side: int = SIDE,
    checkpoint_dir: str = "checkpoints/phase1",
    save_checkpoint: bool = True,
) -> Dict:
    """Runs all 4 stages end to end on a single image file and returns a
    status dict. On success, also writes a .npz checkpoint containing the
    81 cleaned cell images, their empty/digit flags, ground-truth labels
    (if available), and the transform matrix + corners."""
    image = cv2.imread(image_path)
    if image is None:
        return {"path": image_path, "status": "failed", "reason": "could not read image"}

    gray, thresh = to_binary_threshold(image)

    corners = find_grid_corners(thresh)
    if corners is None:
        return {"path": image_path, "status": "failed", "reason": "could not find 4 corners"}

    transform_matrix, inverse_matrix = compute_transform(corners, side=side)
    warped_gray, warped_thresh = warp_grid(gray, thresh, transform_matrix, side=side)
    raw_cells = split_cells(warped_thresh, side=side)

    cleaned_cells = [clean_cell(c) for c in raw_cells]
    empty_flags = [is_empty(c) for c in cleaned_cells]

    ground_truth_grid = load_ground_truth_grid(image_path)
    if ground_truth_grid is not None:
        cell_labels = ground_truth_grid.flatten().tolist()
    else:
        cell_labels = [-1] * 81

    result = {
        "path": image_path,
        "status": "passed",
        "empty_count": int(np.sum(empty_flags)),
        "has_ground_truth": ground_truth_grid is not None,
        "warped": warped_gray,
        "cleaned_cells": cleaned_cells,
        "empty_flags": empty_flags,
        "cell_labels": cell_labels,
        "transform_matrix": transform_matrix,
        "inverse_matrix": inverse_matrix,
        "corners": corners,
    }

    if save_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"{checkpoint_dir}/{checkpoint_name}.npz"
        np.savez(
            save_path,
            cells=np.array(cleaned_cells, dtype=object),
            empty_flags=np.array(empty_flags),
            cell_labels=np.array(cell_labels, dtype=int),
            transform_matrix=transform_matrix,
            corners=corners,
        )
        result["checkpoint"] = save_path

    return result


def batch_test(
    image_dir: str,
    limit: Optional[int] = None,
    checkpoint_dir: str = "checkpoints/phase1",
) -> List[Dict]:
    """Runs process_sudoku_image() over every .jpg in a folder and prints a
    pass/fail summary."""
    test_images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if limit:
        test_images = test_images[:limit]

    results = [process_sudoku_image(p, checkpoint_dir=checkpoint_dir) for p in test_images]

    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    with_gt = [r for r in passed if r["has_ground_truth"]]

    print(f"Passed: {len(passed)} / {len(results)}")
    print(f"Failed: {len(failed)}")
    for r in failed:
        print(" -", r["path"], "->", r["reason"])
    print(f"Images with ground-truth .dat labels found: {len(with_gt)} / {len(passed)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: batch-test Sudoku grid extraction")
    parser.add_argument("--images", default="datasets/sudoku_dataset/images", help="Folder of test images")
    parser.add_argument("--limit", type=int, default=20, help="Max number of images to test")
    parser.add_argument("--checkpoint-dir", default="checkpoints/phase1", help="Where to write .npz checkpoints")
    args = parser.parse_args()
    limit = None if args.limit == 0 else args.limit
    batch_test(args.images, limit, args.checkpoint_dir)
