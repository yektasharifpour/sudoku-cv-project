## Sudoku Solver

> [!info] Phase 3 — Backtracking Sudoku Solver
> Implementation in `solver/sudoku_solver.py`. Unit tests in `solver/test_solver.py`.

---

### Overview

The solver receives a 9×9 integer matrix from Phase 2's digit recognition output and fills in all empty cells (represented as `0`) to produce a complete, valid Sudoku solution. The implementation uses a classic **backtracking** algorithm with row/column/3×3-box constraint checking.

### Input Format: Converting Model Output to a 9×9 Matrix

Phase 2's `recognize_digits()` function returns a `DigitRecognitionResult` with a `grid` field — a list of 9 lists, each containing 9 integers:

```
grid[row][col] ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

- **`0`** = empty cell (the model classified the cell as empty or the gatekeeper interceptor overrode the prediction)
- **`1`–`9`** = recognized digit

This grid is passed directly to `solve_sudoku()` as a list-of-lists or NumPy array. The solver works on a deep copy so the original board is not mutated.

### Solver Algorithm: Backtracking

```
solve_backtracking(board):
    find the first empty cell (value == 0)
    if no empty cell found → puzzle is solved, return True

    for num in 1..9:
        if placing num is safe (no conflicts):
            place num
            recursively solve the rest
            if successful → return True
            remove num (backtrack)

    return False  (no valid number fits → trigger backtracking)
```

The algorithm guarantees correctness by exhaustively trying every valid placement. For standard 9×9 Sudoku puzzles this runs in well under a second.

### Validation Rules

A placement is **safe** if the number does not already appear in:

1. **Same row** — horizontal check across all 9 columns
2. **Same column** — vertical check across all 9 rows
3. **Same 3×3 box** — the sub-grid containing the cell, computed as `block_idx = (row // 3) * 3 + (col // 3)`

Additionally, the entire board is **pre-validated** before solving begins via `is_valid_board()`, which checks that no row, column, or 3×3 box contains duplicate non-zero values. This catches invalid input early rather than wasting time on an unsolvable puzzle.

### Handling Invalid or Unsolvable Puzzles

The solver's top-level function `solve_sudoku(board)` returns a tuple `(success, solved_grid)`:

| Scenario | Return Value |
|----------|-------------|
| Valid puzzle, successfully solved | `(True, solved_grid)` |
| Board is not 9×9 | `(False, None)` |
| Board contains duplicate digits (invalid) | `(False, None)` |
| Board is valid but has no solution | `(False, None)` |

The UI handles the failure cases gracefully:

- **Invalid board** — The Streamlit app displays: *"Couldn't solve this grid — it's either internally inconsistent (e.g. a repeated digit in some row/column/box) or has no valid solution."*
- The user is prompted to edit the grid in the data editor and try again.

### Tests

Unit tests in `solver/test_solver.py` cover:

- Standard valid solvable puzzle
- NumPy array input (type flexibility)
- Initially invalid puzzle (duplicate digits → returns `False`)
- Unsolvable puzzle (a cell trapped with all 1–9 blocked → returns `False`)
- Empty board (all zeros → solver fills it, returns `True`)
- Fully solved board (no empty cells → returns `True`)
- Wrong dimensions (8×9 → returns `False`)

All tests pass. Run with: `pytest solver/test_solver.py`
