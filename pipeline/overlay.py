"""
Renders the solved Sudoku back onto the original input image.

Covers bonus option 4 from the brief:
  - display the final answer on the original input image
  - preserve the original grid's perspective
  - correct alignment of overlaid text with the image

Only digits that were originally empty (0 in the recognized grid) are drawn --
the digits the model already read are left alone, so it's visually clear
what the solver actually filled in.
"""

from typing import List

import cv2
import numpy as np

from pipeline.interfaces import GridExtractionResult, warp_point_to_original


def draw_solution_on_original(
    original_image: np.ndarray,
    extraction: GridExtractionResult,
    recognized_grid: List[List[int]],
    solved_grid: List[List[int]],
    warped_size: int = None,
) -> np.ndarray:
    """
    Args:
        original_image: the raw input photo (BGR).
        extraction: the GridExtractionResult used to get here (for the
            inverse perspective matrix).
        recognized_grid: Phase 2's 9x9 output (0 = empty). Used to decide
            which cells to draw into (only originally-empty ones).
        solved_grid: the solver's completed 9x9 grid.
        warped_size: side length in px of the warped grid used during
            extraction. Defaults to `extraction.warped_grid.shape[0]` --
            pass this explicitly only if that's somehow not reliable.
            Real Phase 1 uses 900px; the mock pipeline uses 450px. Don't
            hardcode either value here, since it silently breaks cell
            alignment if Phase 1 changes its output size.

    Returns:
        A copy of original_image with the solved digits drawn in, in the
        correct position/perspective.
    """
    if warped_size is None:
        warped_size = extraction.warped_grid.shape[0]

    output = original_image.copy()
    cell_size = warped_size / 9.0

    for row in range(9):
        for col in range(9):
            if recognized_grid[row][col] != 0:
                continue  # leave originally-filled cells alone

            digit = solved_grid[row][col]
            if digit == 0:
                continue  # shouldn't happen on a successfully solved board

            # Center of this cell in warped-grid coordinates.
            cx = col * cell_size + cell_size / 2
            cy = row * cell_size + cell_size / 2

            ox, oy = warp_point_to_original(
                (cx, cy), extraction.inverse_perspective_matrix
            )

            # Estimate a reasonable font scale from how large a cell is in
            # the original image, so text looks proportional regardless of
            # photo resolution or grid tilt. Sample cell width via two
            # adjacent warped points mapped back to original coordinates.
            right_x, right_y = warp_point_to_original(
                (cx + cell_size / 2, cy), extraction.inverse_perspective_matrix
            )
            cell_px_in_original = ((right_x - ox) ** 2 + (right_y - oy) ** 2) ** 0.5
            font_scale = max(0.4, cell_px_in_original / 25.0)
            thickness = max(1, int(font_scale * 2))

            text = str(digit)
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_origin = (int(ox - text_w / 2), int(oy + text_h / 2))

            cv2.putText(
                output,
                text,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 140, 255),  # BGR orange, distinguishable from black grid ink
                thickness,
                cv2.LINE_AA,
            )

    return output
