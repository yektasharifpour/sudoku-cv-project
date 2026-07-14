"""
Mock implementation of pipeline/interfaces.py.

Lets the UI be built and demoed before Phase 1 (grid extraction) and Phase 2
(digit recognition) are finished. Does NOT do real computer vision -- it fakes
a plausible grid extraction from whatever image is uploaded, and always
"recognizes" the same fixed example puzzle, so the full pipeline (extract ->
recognize -> solve -> overlay) can be exercised end to end.

Once Phase 1 / Phase 2 are ready, the UI switches to the real functions by
changing a single import in ui/app.py -- nothing else about this file's
callers needs to change, since it implements the same interface.
"""

import cv2
import numpy as np

from pipeline.interfaces import (
    DigitRecognitionResult,
    GridExtractionResult,
    GridNotFoundError,
)

# A fixed example puzzle (the classic Peter Norvig demo puzzle) returned
# regardless of the actual image content -- this is a MOCK, not real digit
# recognition. Swap for the real Phase 2 model when it's ready.
_MOCK_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

_WARPED_SIZE = 450  # px, square


def extract_grid(input_image: np.ndarray) -> GridExtractionResult:
    """
    MOCK. Instead of real grid detection, takes the largest centered square
    crop of the input image and resizes it to a fixed size, pretending that's
    the perspective-corrected grid. Also fakes an inverse perspective matrix
    that maps back to that same center-crop region, so the overlay step has
    something geometrically consistent to work with.

    Raises GridNotFoundError if the image is unreadable / too small, to
    exercise the UI's error-handling path.
    """
    if input_image is None or input_image.size == 0:
        raise GridNotFoundError("Empty image")

    h, w = input_image.shape[:2]
    if h < 50 or w < 50:
        raise GridNotFoundError("Image too small to plausibly contain a grid")

    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    src_pts = np.float32(
        [[left, top], [left + side, top], [left + side, top + side], [left, top + side]]
    )
    dst_pts = np.float32(
        [[0, 0], [_WARPED_SIZE, 0], [_WARPED_SIZE, _WARPED_SIZE], [0, _WARPED_SIZE]]
    )

    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    inverse_matrix = np.linalg.inv(perspective_matrix)

    warped = cv2.warpPerspective(input_image, perspective_matrix, (_WARPED_SIZE, _WARPED_SIZE))

    cell_size = _WARPED_SIZE // 9
    cell_images = []
    for row in range(9):
        for col in range(9):
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            cell_images.append(warped[y0:y1, x0:x1])

    return GridExtractionResult(
        warped_grid=warped,
        cell_images=cell_images,
        inverse_perspective_matrix=inverse_matrix,
        detection_confidence=1.0,
    )


def recognize_digits(extraction: GridExtractionResult) -> DigitRecognitionResult:
    """
    MOCK. Ignores the actual cell images and always returns the same fixed
    example puzzle, with uniform fake confidence. Swap for the real Phase 2
    model's predictions when ready -- same return shape.
    """
    confidences = [[0.99 if v != 0 else 1.0 for v in row] for row in _MOCK_PUZZLE]
    return DigitRecognitionResult(
        grid=[row[:] for row in _MOCK_PUZZLE],
        confidences=confidences,
    )
