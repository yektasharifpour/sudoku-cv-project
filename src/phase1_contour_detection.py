"""
src/phase1_contour_detection.py

Phase 1, stage 2: find the sudoku grid's outer boundary and reduce it to
exactly 4 corner points.

FIX #1 (see project error analysis): the previous version tried only 4
fixed epsilon values (0.02, 0.04, 0.05, 0.06) for cv2.approxPolyDP. Real
photos regularly need an epsilon in between these -- e.g. image200.jpg only
collapses to exactly 4 points at eps=0.08, which the old fixed list never
tried (it jumped straight from 6 points at 0.06 to over-simplifying past 4
points entirely at slightly higher values). Fixed by sweeping a fine,
continuous range instead of guessing a handful of values.

FIX #2 (image25/36/70/73/196/199): those 6 images kept failing even with
the fine epsilon sweep -- and for a different reason than #1. Their outer
grid border is drawn no thicker/darker than the internal grid lines, so
after adaptiveThreshold it doesn't form one continuous closed loop; the
border breaks into several disconnected fragments. cv2.findContours then
has no single contour tracing the actual grid boundary at all, so
`max(contours, key=cv2.contourArea)` picks up some other, smaller, irregular
blob in the photo instead (confirmed directly against these images: the
"largest" contour covered only 10-28% of the frame and its point count
skipped straight over 4 at every epsilon -- e.g. 7,7,7,5,5,5,2,2 -- because
it was the wrong shape entirely, not a simplification-tolerance problem).

Fixed by dilating the binary mask (bridges small gaps in the broken border
into one closed shape) and taking the convex hull of the largest resulting
contour (closes any remaining concavities) before the epsilon sweep. This
was tested directly against all 6 previously-failing images: every one now
lands on a clean 60-81%-of-frame quadrilateral at eps=0.01, and the
resulting warped grids look correctly squared, not distorted crops of some
unrelated shape. This is meant to be a superset fix like #1 (a photo with an
already-solid border should dilate/hull into essentially the same
rectangle), but re-run the full batch after this change to confirm nothing
that used to pass now fails, since it wasn't tested against your whole
dataset, only the 6 known failures.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

# Fine, wide sweep instead of a few hand-picked values. Step size of 0.005
# was small enough to land inside image200.jpg's narrow "exactly 4" window
# (which turned out to be a single 0.01-wide slice around eps=0.08) without
# being so fine that it becomes slow -- ~28 attempts per image, worst case.
EPSILON_RANGE: Tuple[float, float, float] = (0.01, 0.15, 0.005)  # start, stop, step

# 3x3 kernel, 2 iterations: enough to bridge the gaps seen in the 6 failing
# images' broken borders without visibly eating into the grid's true shape.
DILATION_KERNEL = np.ones((3, 3), np.uint8)
DILATION_ITERATIONS = 2


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sorts 4 arbitrary corner points into top-left, top-right,
    bottom-right, bottom-left order (required by getPerspectiveTransform)."""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # bottom-right: largest x+y

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x

    return ordered


def find_grid_corners(
    thresh: np.ndarray, epsilon_range: Tuple[float, float, float] = EPSILON_RANGE
) -> Optional[np.ndarray]:
    """
    Args:
        thresh: binary mask from phase1_preprocessing.to_binary_threshold().
        epsilon_range: (start, stop, step) fraction-of-perimeter values to
            try with cv2.approxPolyDP, in increasing order. Stops at the
            first one that yields exactly 4 points.

    Returns:
        4 ordered corner points (see order_corners), or None if no contour
        in the image ever reduces to exactly 4 points across the whole
        range -- meaning no grid-like quadrilateral was found at all.
    """
    dilated = cv2.dilate(thresh, DILATION_KERNEL, iterations=DILATION_ITERATIONS)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    perimeter = cv2.arcLength(hull, True)

    for eps in np.arange(*epsilon_range):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx)

    return None