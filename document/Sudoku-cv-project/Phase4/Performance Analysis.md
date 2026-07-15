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



---

### 2. Execution Time

Measured with `benchmarks/benchmark_pipeline.py` on a single test image (`datasets/sudoku_dataset/testing/image170.jpg`), 5 repetitions per stage, **CPU only** (no CUDA available on the benchmark machine). The script warms up the model cache before timing, so the numbers reflect steady-state inference cost, not the one-time model load.

| Stage | Mean (s) | Std (s) | Min (s) |
|-------|---------|---------|---------|
| `extract_grid()` | 0.0037 | 0.0007 | 0.0029 |
| `recognize_digits()` | 0.0151 | 0.0011 | 0.0140 |
| `solve_sudoku()` | 0.0509 | 0.0080 | 0.0428 |
| `draw_solution_on_original()` | 0.0044 | 0.0082 | 0.0007 |
| **Total end-to-end** | **0.0959** | **0.0318** | **0.0678** |

> [!note] Interpretation
> The full pipeline solves a single puzzle in **~0.1 seconds** on CPU — well under the 5-second budget initially estimated. The solver (`solve_sudoku`) is the slowest single stage (~53% of total), which is expected since backtracking is the only non-vectorized, iterative component. Digit recognition is fast (~16% of total) because the CNN is small and runs as a single batched forward pass over 81 cells.
>
> These numbers exclude the one-time `torch.load` model weight load (~0.5–1 s), which the Streamlit UI pays only once per session. The benchmark script isolates steady-state cost via a warm-up pass.

Reproduce with:

```bash
python benchmarks/benchmark_pipeline.py [path/to/image.jpg]
```

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
