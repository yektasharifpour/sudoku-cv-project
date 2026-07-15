## Evaluation Results

> [!info] Deliverable: Phase 2 — Digit Recognition
> Classification report, confusion matrix, and held-out test accuracy for the digit recognition model. Detailed training log: [[Training progress]].

---

### v1 Model (Colab — MNIST + Hoda + synthetic empty cells)

**Dataset:** 120k combined train / 30k combined test (MNIST + Hoda + real empty + synthetic empty)
**Architecture:** 2-layer CNN (`DigitClassifier`) — 16 + 32 filters, single FC layer

#### Per-class Classification Report (29,387 test samples)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (empty) | 1.00 | 1.00 | 1.00 | 2,367 |
| 1 | 1.00 | 1.00 | 1.00 | 3,135 |
| 2 | 0.99 | 0.99 | 0.99 | 3,032 |
| 3 | 0.99 | 0.98 | 0.99 | 3,010 |
| 4 | 0.98 | 0.99 | 0.99 | 2,982 |
| 5 | 0.99 | 0.99 | 0.99 | 2,892 |
| 6 | 0.99 | 0.99 | 0.99 | 2,958 |
| 7 | 1.00 | 0.99 | 0.99 | 3,028 |
| 8 | 0.99 | 1.00 | 1.00 | 2,974 |
| 9 | 0.99 | 0.99 | 0.99 | 3,009 |

**Overall accuracy: 0.99** (macro avg 0.99, weighted avg 0.99)

**Total misclassified: 240 / 29,387 (0.82%)**

#### Confusion Matrix

![[b2.9-output.png]]

---

### v10 Model (Final — domain-matched training on real Sudoku cells)

**Dataset:** 109,204 train / 19,280 validation (real labeled cells from Phase 1 checkpoints + domain-matched MNIST + printed font digits + synthetic empty cells)
**Architecture:** `DigitClassifierV2` — 3 conv layers (32/64/128) + BatchNorm + 256-unit FC + Dropout(0.4)
**Device:** CUDA

#### Final Evaluation on Held-out Real Sudoku Photo Cells (v2_test)

**Test accuracy: 0.9506** (3,321 real Sudoku test cells from 41 test images)

#### Confusion Matrix (rows = true label, cols = predicted)

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|--|---|---|---|---|---|---|---|---|---|---|
| **0** | 2106 | 8 | 2 | 3 | 3 | 2 | 2 | 2 | 2 | 2 |
| **1** | 16 | 117 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| **2** | 11 | 0 | 120 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **3** | 14 | 0 | 1 | 117 | 0 | 0 | 0 | 0 | 0 | 0 |
| **4** | 11 | 0 | 1 | 0 | 118 | 0 | 0 | 1 | 1 | 0 |
| **5** | 14 | 0 | 0 | 0 | 0 | 109 | 0 | 0 | 0 | 0 |
| **6** | 17 | 0 | 0 | 0 | 0 | 0 | 101 | 0 | 0 | 0 |
| **7** | 14 | 2 | 0 | 0 | 0 | 0 | 0 | 125 | 1 | 0 |
| **8** | 16 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 134 | 1 |
| **9** | 9 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 110 |

> [!note] Key observation
> The dominant error mode is **digits being misclassified as empty (class 0)**. This accounts for the majority of all misclassifications. The likely cause is that thin or faint digit strokes in real Sudoku photos get partially lost during thresholding, leaving very little ink for the classifier to work with. The inverse error (empty cells classified as digits) is far less common (e.g., only 8 false positives for class 0 → class 1).
>
> Among digit-to-digit confusions, the most notable pairs are 7↔1 (2 cases) and 8↔9 (1 case) — consistent with visually ambiguous stroke shapes.


