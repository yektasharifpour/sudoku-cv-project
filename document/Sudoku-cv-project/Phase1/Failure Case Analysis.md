## Failure Case Analysis

> [!info] Deliverable: Phase 1 — Grid Extraction
> Analysis of images where the grid extraction pipeline fails to produce a correct result. Full processing logs and debug visuals are in [[Processing Stage Images#Block 1.8—Batch Testing Function|Block 1.8 — Batch Testing]].

---

### Batch Test Results (v10 — final pipeline)

The final pipeline (`src/phase1_pipeline.py`) was run on the full **wichtounet/sudoku_dataset**:

| Fold | Passed | Failed | Total |
|------|--------|--------|-------|
| Training | 119 / 119 | 0 | 119 |
| Testing  |  38 / 38  | 0 | 38 |

> [!success] No failures on the v10 pipeline
> The earlier batch-test run (Block 1.8 in the Colab notebook, v1 pipeline) reported 3 failures out of 20. Subsequent improvements (epsilon tuning for corner detection, proper global threshold reuse, margin/crop fixes) eliminated those failures and scaled successfully to all 157 dataset images.

---

### Original Failure Cases (Block 1.8 — v1 pipeline)

See [[Processing Stage Images#Block 1.8—Batch Testing Function|Block 1.8]] for the full debug output.

**Passed:** 17 / 20
**Failed:** 3

![[b1.8r1s1-output.png]]

| Image | Error | Root Cause |
|-------|-------|------------|
| `image76.jpg` | `could not find 4 corners` | Grid extends almost to the photo's edges with barely any margin/background around it. `findContours` with `RETR_EXTERNAL` likely picked up the _photo frame itself_ as the largest contour instead of the grid border. |
| `image74.jpg` | `could not find 4 corners` | Red text/graphic block on the right side of the page outside the grid. If that block's contour area rivals or exceeds the grid's, `max(contours, key=cv2.contourArea)` grabs the wrong region, or splits attention from the true grid border. |
| `image196.jpg` | `could not find 4 corners` | Same issue as image74 (colored pie-chart graphic and extra page content on the right), **plus** a visible shadow diagonal across the page, which breaks the grid's contour into disconnected pieces under thresholding. |

### Common Failure Patterns

1. **Insufficient background margin around the grid** — When the Sudoku grid fills most of the photo with little to no border, the outer contour detector confuses the photo frame with the grid border.
2. **Distracting content adjacent to the grid** — Colored graphics, text blocks, or charts near the grid produce competing contours that can be selected over the true grid outline.
3. **Shadows and uneven lighting** — Diagonal shadows across the page split the grid's contour into disconnected fragments under binary thresholding, preventing a clean quadrilateral from being detected.

### Mitigations Applied

- **Increased epsilon tolerance** in `cv2.approxPolyDP` to handle tighter margins around the grid (see [[Processing Stage Images#Testing the result on real data set|real dataset testing section]]).
- **Global threshold reuse** from Block 1.2 instead of per-cell Otsu — eliminated false empty-cell detection on noisy blank cells (see [[Processing Stage Images## Block 1.6 — Clean Cells & Detect Empty Cells|Block 1.6 debug log]]).
- **Cell margin increase** (`margin_ratio: 0.12 → 0.18`) and threshold fix (`0.02 → 0.3`) to strip residual grid-line pixels from cell crops.
