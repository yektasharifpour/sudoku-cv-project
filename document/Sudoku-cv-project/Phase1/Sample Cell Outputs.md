## Sample Outputs of the Extracted 81 Cells

> [!info] Deliverable: Phase 1 — Grid Extraction
> Visual outputs from each block of the grid extraction pipeline, demonstrating successful processing from raw input to 81 individual cell crops.

---

### Block 1.1 — Load & Grayscale
![[Pasted image 20260711214038.png]]
Raw input photo converted to grayscale as the first step.

### Block 1.2 — Noise Removal & Thresholding
![[b1.2-output.png|601]]
Noise reduction and binary thresholding to highlight grid lines for contour detection.

### Block 1.3 — Grid Contour Detection
![[Pasted image 20260712003120.png|340]]
Detected outer border of the Sudoku grid with the 4 corner points extracted.

### Block 1.4 — Perspective Transform
![[Pasted image 20260712003134.png|329]]
Warped grid to a flat top-down 900×900 square view using the perspective transform matrix.

### Block 1.5 — Cell Splitting
![[Pasted image 20260712003151.png|340]]
The warped grid divided into 81 equal cells (9×9). Total cells extracted: 81.

### Block 1.6 — Empty Cell Detection (debug iterations)

Debug output showing the raw white-pixel ratio analysis used to determine the empty-cell threshold:

![[Pasted image 20260712003249.png|347]]

---

![[Pasted image 20260712003335.png|351]]

---

![[Pasted image 20260712003458.png|380]]

After fixing the threshold and reusing the global threshold from Block 1.2, empty-cell detection was verified against the ground truth. Final result: 49 empty cells detected, matching the actual puzzle exactly. See [[Processing Stage Images## Block 1.6 — Clean Cells & Detect Empty Cells|Block 1.6 debug log]] for full details.

### Testing Dataset Results

Pipeline applied to images from the `wichtounet/sudoku_dataset`:

![[Pasted image 20260712004654.png]]

---

![[Pasted image 20260712004720.png]]

---

![[Pasted image 20260712004741.png|414]]

---

![[Pasted image 20260712004803.png|410]]

---

![[Pasted image 20260712004826.png|405]]

---

![[Pasted image 20260712004904.png|406]]

---

![[Pasted image 20260712004942.png|421]]

### Block 1.8 — Batch Test Summary

![[Pasted image 20260712005057.png]]
