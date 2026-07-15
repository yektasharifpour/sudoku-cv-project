"""
src/phase1_preprocessing.py

Phase 1, stage 1: turn a raw photo into a binary mask ready for contour
detection. Split out on its own so this stage's output (the thresholded
image) can be inspected/saved independently -- this is exactly the
"processing stage images" deliverable the project brief asks for, and having
it as a standalone function makes that a one-line call instead of digging it
out of a monolithic pipeline function.
"""

from typing import Tuple

import cv2
import numpy as np


def to_binary_threshold(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image_bgr: raw photo, BGR (OpenCV convention).

    Returns:
        (gray, thresh) --
          gray: grayscale version of the input, same resolution.
          thresh: binary mask (0/255) where ink/grid-lines are WHITE (255)
            and paper background is BLACK (0). THRESH_BINARY_INV is what
            makes this inversion happen -- adaptiveThreshold's normal output
            would have dark ink as 0, which is why INV is used here.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return gray, thresh
