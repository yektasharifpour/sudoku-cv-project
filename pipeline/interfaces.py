"""
Pipeline interface contract.

This file defines the exact function signatures the UI (Phase 4 / bonus UI option)
expects from Phase 1 (grid extraction) and Phase 2 (digit recognition).

Nobody should implement logic in this file. It exists so that:
  - The UI can be built and demoed today against `pipeline/mock_pipeline.py`.
  - Whoever finishes Phase 1 / Phase 2 knows exactly what shape of data to return,
    without needing to look at UI code.
  - Swapping mock -> real implementation is a one-line import change in the UI
    (see ui/app.py), not a rewrite.

Coordinate convention: all pixel coordinates are (x, y), origin top-left, matching
OpenCV / PIL convention (not numpy's (row, col)).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class GridExtractionResult:
    """Output of Phase 1 (grid extraction)."""

    # The perspective-corrected, top-down crop of just the 9x9 grid.
    # Expected to be roughly square, e.g. 450x450 or 900x900 px, grayscale or BGR.
    warped_grid: np.ndarray

    # 81 cell crops, in row-major order (cell 0 = row 0 col 0, cell 1 = row 0 col 1, ...).
    # Each is a small grayscale/BGR image of a single cell, cropped from warped_grid,
    # ideally with the grid lines already removed (per the Phase 1 brief).
    cell_images: List[np.ndarray]

    # The 3x3 perspective transform matrix that maps points FROM warped_grid
    # coordinates BACK TO the original input image coordinates. This is the
    # inverse of whatever matrix was used to produce warped_grid (e.g. the
    # inverse of what cv2.getPerspectiveTransform + cv2.warpPerspective used).
    # Required for the "overlay answer on original image" bonus option.
    inverse_perspective_matrix: np.ndarray  # shape (3, 3)

    # Confidence that a table was actually found (0-1). UI shows a warning
    # below some threshold. Optional -- default to 1.0 if not modeling this.
    detection_confidence: float = 1.0


@dataclass
class DigitRecognitionResult:
    """Output of Phase 2 (digit recognition)."""

    # 9x9 grid of ints, 0 = empty cell, 1-9 = recognized digit.
    # Row-major: grid[row][col].
    grid: List[List[int]]

    # Per-cell confidence, same shape as grid, values 0-1. Optional but
    # strongly recommended -- lets the UI flag "low confidence" cells to the
    # user instead of silently trusting a possibly-wrong read, and is useful
    # for the error-analysis section of the report.
    confidences: Optional[List[List[float]]] = None


def extract_grid(input_image: np.ndarray) -> GridExtractionResult:
    """
    Phase 1 entry point.

    Args:
        input_image: raw uploaded photo, BGR (OpenCV convention), any resolution.

    Returns:
        GridExtractionResult. Raise `GridNotFoundError` (below) if no sudoku
        grid could be located in the image -- do not return a best-guess crop
        silently, since the UI needs to distinguish "no grid found" from
        "grid found but hard to read" to give useful feedback.
    """
    raise NotImplementedError("Implemented by Phase 1")


def recognize_digits(extraction: GridExtractionResult) -> DigitRecognitionResult:
    """
    Phase 2 entry point.

    Args:
        extraction: the GridExtractionResult from extract_grid().
            Use extraction.cell_images (81 cells, row-major).

    Returns:
        DigitRecognitionResult with a 9x9 grid of ints (0 = empty).
    """
    raise NotImplementedError("Implemented by Phase 2")


class GridNotFoundError(Exception):
    """Raised by extract_grid() when no sudoku grid is detected in the image."""


def warp_point_to_original(
    point_xy: Tuple[float, float], inverse_perspective_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Helper: maps a single (x, y) point in warped_grid coordinates back to
    original-image coordinates, using the inverse perspective matrix from
    GridExtractionResult. Used by the overlay renderer (bonus option 4).
    """
    x, y = point_xy
    vec = np.array([x, y, 1.0])
    mapped = inverse_perspective_matrix @ vec
    mapped = mapped / mapped[2]
    return float(mapped[0]), float(mapped[1])
