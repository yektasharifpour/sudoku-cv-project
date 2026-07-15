"""
src/phase1_cell_cleanup.py

Phase 1, stage 4: strip leftover grid-line fragments off each cell's edges,
and a quick heuristic for whether a cell looks empty.

This is the single source of truth for clean_cell() -- pipeline/real_pipeline.py
and digit_recognition/domain_matched_augmentation.py both import it from here
rather than keeping their own copies. That's a deliberate fix: an earlier
version had a second, hand-copied clean_cell() that silently drifted out of
sync with this one (different margin_ratio), which corrupted training data
without any error ever being raised. One shared function, one place to change
the margin ratio, no drift possible.
"""

import numpy as np

MARGIN_RATIO = 0.2  # see phase1 error analysis: raised from 0.18 to strip
                      # thicker grid-line fragments seen in real photos


def clean_cell(cell: np.ndarray, margin_ratio: float = MARGIN_RATIO) -> np.ndarray:
    """Crops a fixed margin off each edge of a cell."""
    h, w = cell.shape
    m = int(h * margin_ratio)
    return cell[m : h - m, m : w - m]


def is_empty(binary_cell: np.ndarray, white_pixel_threshold: float = 0.03) -> bool:
    """A cell is considered empty if very few pixels are 'ink' (white,
    since THRESH_BINARY_INV makes dark ink show up as 255)."""
    white_ratio = np.sum(binary_cell == 255) / binary_cell.size
    return white_ratio < white_pixel_threshold
