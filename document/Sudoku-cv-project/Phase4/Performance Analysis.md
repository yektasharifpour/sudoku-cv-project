## Performance Analysis

> [!info] Deliverable: Phase 4 — Final Integrated System
> End-to-end accuracy, execution time, and error analysis for the complete pipeline.

---

### Pipeline Stages

The integrated pipeline (`pipeline/real_pipeline.py` + `solver/sudoku_solver.py` + `pipeline/overlay.py`) chains four stages:

1. **Grid extraction** — `extract_grid()`: grayscale → threshold → contour detection → perspective warp → cell splitting
2. **Digit recognition** — `recognize_digits()`: CNN inference on 81 cells + gatekeeper empty-cell override
3. **Sudoku solving** — `solve_sudoku()`: backtracking solver with pre-validation
4. **Solution overlay** — `draw_solution_on_original()`: inverse-perspective warp of solved digits onto original photo

---

### 1. Accuracy

#### Phase 1 — Grid Extraction Accuracy

| Metric | Value | Notes |
|--------|-------|-------|
| Training set (119 images) | 119/119 (100%) | All grids detected and corners found |
| Testing set (38 images) | 38/38 (100%) | All grids detected and corners found |
| **Overall** | **157/157 (100%)** | `wichtounet/sudoku_dataset` |

> [!note] Caveat
> This measures whether the pipeline successfully extracts a 9×9 grid — not whether individual cell crops are perfectly aligned. Cell-level quality affects Phase 2 accuracy downstream.

#### Phase 2 — Digit Recognition Accuracy

| Model | Dataset | Accuracy | Notes |
|-------|---------|----------|-------|
| v1 (2-layer CNN) | Combined MNIST+Hoda (30k test) | 99.18% | Clean digit images, synthetic empty cells |
| v10 (3-layer CNN + BN) | Real Sudoku test cells (3,321 cells, 41 images) | 95.06% | Held-out real photo cells through full pipeline |

#### Phase 4 — End-to-End Puzzle Accuracy

<!-- TODO: Run the full pipeline (extract → recognize → solve) on all 157 test images and measure: -->
<!-- - How many puzzles produce a correct solution WITHOUT user correction -->
<!-- - How many puzzles produce a correct solution WITH user correction -->
<!-- - How many puzzles fail (grid not found, unsolvable after correction) -->

> [!warning] Not yet measured
> End-to-end puzzle-level accuracy (percentage of test images where the full pipeline produces a correct solved grid without manual intervention) has not been benchmarked yet. Given the ~5% per-cell error rate, approximately 1 in 5 puzzles will have at least one misread digit. The user-verification step in the UI mitigates this in practice.

---

### 2. Execution Time

<!-- TODO: Benchmark each pipeline stage on a representative test image and record wall-clock time: -->
<!-- - extract_grid() → ? seconds -->
<!-- - recognize_digits() → ? seconds (CPU vs CUDA) -->
<!-- - solve_sudoku() → ? seconds -->
<!-- - draw_solution_on_original() → ? seconds -->
<!-- - Total end-to-end → ? seconds -->

> [!warning] Not yet measured
> Per-stage and total execution time benchmarks have not been collected. The pipeline runs on CPU by default; CUDA is used only during training. A typical single-puzzle run is expected to complete in under 5 seconds on CPU, but this needs to be verified.

---

### 3. Error Analysis

#### Error Sources by Stage

| Stage | Error Type | Impact | Mitigation |
|-------|-----------|--------|------------|
| **Phase 1** | Grid not found (no 4 corners) | Pipeline stops; no result | Increased epsilon tolerance; tested 157/157 pass |
| **Phase 1** | Poor perspective warp | Cell crops are skewed, reducing recognition accuracy | Manual corner detection fallback (not yet implemented) |
| **Phase 1** | Grid lines not fully removed from cells | Classifier sees border pixels as digit content | 18% margin crop in `clean_cell()` |
| **Phase 2** | Digit → empty misclassification | Given digit lost; solver has fewer constraints; may produce wrong solution | Gatekeeper interceptor (connected-components check); user verification |
| **Phase 2** | Empty → digit misclassification | False digit appears in grid; solver may find no solution or wrong solution | Gatekeeper interceptor; user verification |
| **Phase 2** | Digit A → Digit B confusion | Wrong digit in grid; solver may produce wrong solution | User verification with low-confidence cell flagging |
| **Phase 3** | Unsolvable grid (conflicting digits) | Solver returns failure; user must correct | Pre-validation check; UI error message |
| **Phase 4** | Overlay misalignment | Solved digits drawn at wrong positions | Inverse perspective matrix from Phase 1; adaptive font scaling |

#### Compound Error Propagation

Errors cascade across stages. A single misread digit in Phase 2 can cause the solver to either:
- Produce a **wrong solution** (different valid completion of the puzzle)
- **Fail entirely** (if the misread creates a conflict)

The probability of a perfect unassisted read for a puzzle with ~32 given digits at 95.06% per-cell accuracy is approximately:

$$P(\text{all correct}) \approx 0.9506^{32} \approx 0.195$$

This is the primary motivation for the **user verification step** in the UI — it decouples the solver's correctness from the recognizer's accuracy.

---

### Bonus Features

- **Bonus #3 — UI**: Streamlit web app (`ui/app.py`) with image upload, grid verification/correction, solving, and solution overlay display.
- **Bonus #4 — Overlay**: Solved digits drawn on the original photo in the correct perspective using inverse warp (`pipeline/overlay.py`).
