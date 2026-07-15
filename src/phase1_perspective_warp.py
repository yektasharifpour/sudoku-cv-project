"""
src/phase1_perspective_warp.py

Phase 1, stage 3: given the 4 corners, warp the grid to a square top-down
view and slice it into 81 raw cell images.
"""

from typing import List, Tuple

import cv2
import numpy as np

SIDE = 900  # warped grid resolution, must stay divisible by 9


def compute_transform(corners: np.ndarray, side: int = SIDE) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (transform_matrix, inverse_matrix). The inverse is needed
    later by Phase 4's overlay step to map solved-grid positions back onto
    the original photo -- Phase 1 itself never needed it, but computing it
    here (once, alongside the forward matrix) means every caller gets it
    for free instead of each one recomputing np.linalg.inv separately."""
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype="float32",
    )
    transform_matrix = cv2.getPerspectiveTransform(corners, dst)
    inverse_matrix = np.linalg.inv(transform_matrix)
    return transform_matrix, inverse_matrix


def warp_grid(
    gray: np.ndarray, thresh: np.ndarray, transform_matrix: np.ndarray, side: int = SIDE
) -> Tuple[np.ndarray, np.ndarray]:
    """Warps both the grayscale preview and the binary mask to a `side` x
    `side` top-down square. Returns (warped_gray, warped_thresh)."""
    warped_gray = cv2.warpPerspective(gray, transform_matrix, (side, side))
    warped_thresh = cv2.warpPerspective(thresh, transform_matrix, (side, side))
    return warped_gray, warped_thresh


def split_cells(warped_thresh: np.ndarray, side: int = SIDE) -> List[np.ndarray]:
    """Slices the warped binary grid into 81 raw (uncropped) cell images,
    row-major order (cell 0 = row 0 col 0, cell 1 = row 0 col 1, ...)."""
    cell_size = side // 9
    cells = []
    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            cells.append(warped_thresh[y1:y2, x1:x2])
    return cells
