## Error Analysis

> [!info] Deliverable: Phase 2 — Digit Recognition
> Analysis of misclassified samples, confusion patterns, and systematic failure modes. Full confusion matrix and misclassification visuals: [[Training progress#Block 2.9 — Confusion Matrix & Error Analysis|Block 2.9]] and [[Training progress#Block 2.10 — Misclassification Examples|Block 2.10]].

---

### v1 Model — Misclassification Examples

![[b2.10-output.png]]

**Total misclassified: 240 out of 29,387 (0.82%)**

> [!note] Error pattern (v1)
> Errors are concentrated in visually ambiguous stroke-shape pairs (3/5, 4/6, 4/9, 1/7), consistent with genuine handwriting ambiguity rather than model deficiency. No systematic failure mode was found for the empty-cell class, which achieved perfect precision/recall.

---

### v10 Model — Error Breakdown on Real Sudoku Cells

The v10 model was evaluated on **3,321 held-out real Sudoku photo cells** from 41 test images. At 95.06% accuracy, the dominant error pattern shifted significantly compared to v1:

#### Primary Error Mode: Digits Misclassified as Empty

The most frequent error across all classes is the model predicting **class 0 (empty)** when the cell actually contains a digit. This accounts for the majority of the ~165 total errors:

| True Class | False → Empty | False Positives into this class |
|-----------|---------------|-------------------------------|
| 0 (empty) | — | 133 (2106 correct) |
| 1 | 16 | 8 |
| 2 | 11 | 4 |
| 3 | 14 | 4 |
| 4 | 11 | 4 |
| 5 | 14 | 2 |
| 6 | 17 | 3 |
| 7 | 14 | 5 |
| 8 | 16 | 4 |
| 9 | 9 | 2 |

**Root cause analysis:**
- Real Sudoku photos have thin, printed digit strokes that can be partially lost during the binary thresholding step in Phase 1. When a digit's ink is faint or the threshold is slightly off, the resulting cell crop contains very little foreground content — resembling an empty cell.
- The `gatekeeper_is_empty` function in the pipeline adds a connected-components check on top of the model, specifically designed to catch cases where the model predicts a digit but the cell is actually empty. However, the **inverse problem** (model says empty but the cell has a digit) is harder to fix with heuristics alone.

#### Secondary Error Mode: Digit-to-Digit Confusion

A smaller set of errors involves one digit being confused with another:

| True → Predicted | Count | Notes |
|-----------------|-------|-------|
| 1 → 7 | 3 | Similar vertical stroke structure |
| 7 → 1 | 2 | Horizontal bar at top missed |
| 8 → 9 | 1 | Lower loop interpreted as tail |

These are genuine visual ambiguities in small (28×28) cell crops.

#### Impact on the Full Pipeline

Even a ~5% per-cell error rate compounds across 81 cells per puzzle. For a typical Sudoku with ~32 givens, the probability that **all given digits are recognized correctly** is approximately:

$$P(\text{perfect read}) \approx 0.95^{32} \approx 0.195$$

This means roughly **1 in 5 puzzles** will have at least one digit misread — which is why the UI (Phase 5) includes a **verify-and-correct** step where the user can manually fix misrecognized cells before solving.

#### Mitigations Applied

1. **Gatekeeper interceptor** (`gatekeeper_is_empty`): Connected-components analysis that overrides the model's empty/non-empty decision based on geometric properties of the cell content. Handles edge cases like paper creases and thin grid-line artifacts.
2. **Domain-matched training**: MNIST digits are re-photographed through the real threshold+crop pipeline so the model trains on the same artifact patterns it will see at inference time.
3. **User verification step**: The Streamlit UI flags low-confidence cells and provides an editable grid for manual correction before solving.


