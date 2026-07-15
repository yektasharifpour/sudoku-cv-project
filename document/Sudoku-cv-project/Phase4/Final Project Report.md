## Final Project Report

> [!info] Deliverable: Phase 4 — Final Project Report
> This note serves as a roadmap for the final report. Fill in each section as the corresponding analysis is completed.

---

### Report Structure

<!-- TODO: Expand each section below into full report content. -->

#### 1. Introduction
- Project objective: detect a Sudoku grid from a photo, recognize digits, solve the puzzle, and overlay the solution on the original image.
- Team roles and division of work
- Tools and technologies used (Python, OpenCV, PyTorch, Streamlit, Git)

#### 2. Phase 1 — Grid Extraction
- Pipeline overview (grayscale → threshold → contour → perspective warp → cell splitting)
- Processing pipeline images: [[../Phase1/Sample Cell Outputs]]
- Extracted cell samples
- Failure case analysis: [[../Phase1/Failure Case Analysis]]

<!-- TODO: Add a summary of the final extraction accuracy (157/157 on the test dataset) -->

#### 3. Phase 2 — Digit Recognition
- Dataset preparation (MNIST, Hoda, real empty cells, synthetic empty cells, domain-matched augmentation)
- Model architecture (DigitClassifierV2: 3 conv + BatchNorm)
- Training curves: [[../Phase2/Training Curve]]
- Evaluation results: [[../Phase2/Evaluation Results]]
- Error analysis: [[../Phase2/Error Analysis]]

<!-- TODO: Add comparison between v1 and v10 models with lessons learned -->

#### 4. Phase 3 — Sudoku Solver
- Backtracking algorithm description
- Input format (9×9 matrix, 0 = empty)
- Validation rules and handling of invalid/unsolvable puzzles
- Full documentation: [[../Phase3/Sudoku Solver]]

#### 5. Phase 4 — Integrated System
- Pipeline architecture and wiring
- Performance analysis: [[Performance Analysis]]
- End-to-end accuracy, execution time, error propagation

<!-- TODO: Add benchmark results once measured -->

#### 6. Bonus Features
- **Bonus #3 — UI**: Streamlit web application with verify-and-correct workflow
- **Bonus #4 — Overlay**: Solution drawn on original photo with correct perspective

<!-- TODO: Add screenshots of the UI in action -->

#### 7. Challenges and Lessons Learned
<!-- TODO: Document key challenges encountered during development and how they were resolved -->
<!-- Examples: per-cell Otsu instability, Colab GPU limitations, checkpoint directory issues -->

#### 8. Future Work
<!-- TODO: Potential improvements beyond the current scope -->
<!-- Examples: handwritten digit support, real-time video input, mobile deployment -->

#### 9. References
<!-- TODO: Add citations for datasets, algorithms, and external resources used -->
<!-- - wichtounet/sudoku_dataset -->
<!-- - MNIST -->
<!-- - Hoda dataset -->
<!-- - Backtracking algorithm references -->
