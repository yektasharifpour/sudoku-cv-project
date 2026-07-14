"""
Diagnostic tool: draws where Phase 1 THINKS the grid's 4 corners are, directly
on top of the original photo, plus the 9x9 cell boundaries it derived from
them. Run this on a few real test photos to see at a glance whether the
rightward-drift problem is:

  (a) bad corner detection -- the drawn corners visibly don't line up with the
      puzzle's actual corners in the photo, usually worse on one specific
      side/corner -> fixable by tuning epsilon/thresholding in extract_grid
  (b) lens distortion -- the 4 corners themselves look right, but the drawn
      grid lines don't stay aligned with the printed grid lines as they cross
      the image (especially near the frame edges) -> a harder problem, not
      fixable by touching corner detection at all, since the transform itself
      can't correct a curved (non-planar-projective) source image

Usage:
    python debug/debug_corners.py path/to/photo.jpg
Writes: <name>_corners_debug.png next to the input image.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from pipeline.real_pipeline import extract_grid
from pipeline.interfaces import GridNotFoundError, warp_point_to_original


def debug_corners(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    try:
        extraction = extract_grid(img)
    except GridNotFoundError as e:
        print(f"extract_grid failed entirely: {e}")
        print("(That's a different, more severe problem than drift -- it means")
        print(" no grid was found at all, not that it was found in the wrong place.)")
        return

    side = extraction.warped_grid.shape[0]
    inv = extraction.inverse_perspective_matrix

    debug_img = img.copy()

    # Draw the 4 corners Phase 1 believes it found, mapped back to original coords
    corner_dst_points = [(0, 0), (side - 1, 0), (side - 1, side - 1), (0, side - 1)]
    corner_labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    original_corners = []
    for (dx, dy), label in zip(corner_dst_points, corner_labels):
        ox, oy = warp_point_to_original((dx, dy), inv)
        original_corners.append((ox, oy))
        cv2.circle(debug_img, (int(ox), int(oy)), 12, (0, 0, 255), -1)
        cv2.putText(debug_img, label, (int(ox) + 15, int(oy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Connect the 4 corners so it's obvious if the quadrilateral itself looks right
    for i in range(4):
        p1 = tuple(map(int, original_corners[i]))
        p2 = tuple(map(int, original_corners[(i + 1) % 4]))
        cv2.line(debug_img, p1, p2, (0, 0, 255), 2)

    # Draw all internal 9x9 grid lines mapped back too -- this is what actually
    # reveals lens distortion: if the photo has barrel/pincushion distortion,
    # these lines will visibly NOT sit on top of the real printed grid lines,
    # especially the further a line is from whichever corner anchors best.
    cell = side / 9.0
    for i in range(10):
        # horizontal line i
        pts = [warp_point_to_original((t, i * cell), inv) for t in np.linspace(0, side, 40)]
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(debug_img, [pts], False, (0, 255, 0), 1, cv2.LINE_AA)
        # vertical line i
        pts = [warp_point_to_original((i * cell, t), inv) for t in np.linspace(0, side, 40)]
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(debug_img, [pts], False, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = os.path.splitext(image_path)[0] + "_corners_debug.png"
    cv2.imwrite(out_path, debug_img)
    print(f"Wrote {out_path}")
    print("Red = the 4 corners Phase 1 detected (and the quadrilateral between them)")
    print("Green = the 9x9 grid lines Phase 1 derived from those corners")
    print("Look closely at whether the GREEN lines drift away from the real")
    print("printed grid lines as they move away from whichever red corner is")
    print("most accurate -- that confirms lens distortion rather than a simple")
    print("corner-detection miss.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug/debug_corners.py path/to/photo.jpg")
        sys.exit(1)
    debug_corners(sys.argv[1])