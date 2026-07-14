import pytest
import numpy as np
from sudoku_solver import solve_sudoku, is_valid_board

def test_solve_sudoku_valid_puzzle():
    # A standard valid solvable puzzle
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    success, solved = solve_sudoku(board)
    assert success is True
    assert solved is not None

    # Check that there are no empty cells
    for row in solved:
        assert 0 not in row

    # Check that the board remains valid after solving
    assert is_valid_board(solved)

def test_solve_sudoku_numpy_array():
    # A standard valid solvable puzzle using NumPy
    board = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    success, solved = solve_sudoku(board)
    assert success is True
    assert solved is not None
    assert isinstance(solved, np.ndarray)

def test_initially_invalid_puzzle():
    # Invalid: two 5s in the first row
    board = [
        [5, 5, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    success, solved = solve_sudoku(board)
    assert success is False
    assert solved is None

def test_unsolvable_puzzle():
    # Valid initial state, but unsolvable configuration.
    # We can create an unsolvable puzzle by removing a clue needed for unique solution
    # and creating a contradiction or by blocking all possibilities for a cell.
    # An easy unsolvable puzzle is to box in a cell so no valid number can go in.
    board = [
        [5, 1, 6, 8, 4, 9, 7, 3, 2],
        [3, 0, 7, 6, 0, 5, 0, 0, 0],
        [8, 0, 9, 7, 0, 0, 0, 6, 5],
        [1, 3, 5, 0, 6, 0, 9, 0, 7],
        [4, 7, 2, 5, 9, 1, 0, 0, 6],
        [9, 6, 8, 3, 7, 0, 0, 5, 0],
        [2, 5, 3, 1, 8, 6, 0, 7, 4],
        [6, 8, 4, 2, 0, 7, 5, 0, 0],
        [7, 9, 1, 0, 5, 0, 6, 0, 8]
    ]

    # We can force a contradiction to make it unsolvable
    # In row 8, col 4 (0-indexed), there's a 0. Let's make it such that no number can go there.
    # Actually, a simpler unsolvable case is one that requires backtracking but ultimately fails.
    # Let's take a board that fails due to constraints.
    unsolvable_board = [
        [5, 1, 6, 8, 4, 9, 7, 3, 2],
        [3, 2, 7, 6, 1, 5, 8, 4, 9],
        [8, 4, 9, 7, 2, 3, 1, 6, 5],
        [1, 3, 5, 2, 6, 4, 9, 8, 7],
        [4, 7, 2, 5, 9, 1, 3, 0, 6],  # Missing a couple
        [9, 6, 8, 3, 7, 8, 2, 5, 1],  # 8 is duplicated here in row 5 (invalid)
        [2, 5, 3, 1, 8, 6, 0, 7, 4],
        [6, 8, 4, 2, 3, 7, 5, 1, 0],
        [7, 9, 1, 4, 5, 0, 6, 2, 8]
    ]
    # The above is invalid. Let's construct a valid-but-unsolvable board.

    board = [
        [1, 2, 3, 4, 5, 6, 7, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # Cannot put 9 in corner because 1 is taken in row, etc.
        [0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 8]
    ]
    # The top-right corner is (0,8). It's empty. The top row has 1-8. So it must be 9.
    # The rightmost col has 1-8 below it. So the top-right corner must be 9.
    # However, if we put 9 in the rightmost col, what goes in the rest? This is actually solvable possibly.
    # Let's use a known unsolvable configuration by setting a trap.
    board2 = [
        [5, 1, 6, 8, 4, 9, 7, 3, 2],
        [3, 0, 7, 6, 0, 5, 0, 0, 0],
        [8, 0, 9, 7, 0, 0, 0, 6, 5],
        [1, 3, 5, 0, 6, 0, 9, 0, 7],
        [4, 7, 2, 5, 9, 1, 0, 0, 6],
        [9, 6, 8, 3, 7, 0, 0, 5, 0],
        [2, 5, 3, 1, 8, 6, 0, 7, 4],
        [6, 8, 4, 2, 0, 7, 5, 0, 0],
        [7, 9, 1, 0, 5, 0, 6, 0, 8]
    ]
    # Let's explicitly trap a cell:
    board_trap = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # Add numbers to restrict one cell to no possibilities
    # Cell (0,0) needs a number. Let's populate its row, col, block with 1..9
    # Row: 2 3 4
    board_trap[0][1] = 2
    board_trap[0][2] = 3
    board_trap[0][3] = 4
    # Col: 5 6 7
    board_trap[1][0] = 5
    board_trap[2][0] = 6
    board_trap[3][0] = 7
    # Block: 8 9
    board_trap[1][1] = 8
    board_trap[2][2] = 9
    # Now cell (0,0) cannot be 2,3,4 (row), 5,6,7 (col), 8,9 (block).
    # Wait, 1 is still possible.
    board_trap[4][0] = 1 # Now 1 is in the col.
    # So (0,0) cannot be 1,2,3,4,5,6,7,8,9. It's trapped.

    success, solved = solve_sudoku(board_trap)
    assert success is False
    assert solved is None

def test_empty_board():
    board = [[0] * 9 for _ in range(9)]
    success, solved = solve_sudoku(board)
    assert success is True
    assert solved is not None
    assert is_valid_board(solved)

def test_full_solved_board():
    board = [
        [4, 3, 5, 2, 6, 9, 7, 8, 1],
        [6, 8, 2, 5, 7, 1, 4, 9, 3],
        [1, 9, 7, 8, 3, 4, 5, 6, 2],
        [8, 2, 6, 1, 9, 5, 3, 4, 7],
        [3, 7, 4, 6, 8, 2, 9, 1, 5],
        [9, 5, 1, 7, 4, 3, 6, 2, 8],
        [5, 1, 9, 3, 2, 6, 8, 7, 4],
        [2, 4, 8, 9, 5, 7, 1, 3, 6],
        [7, 6, 3, 4, 1, 8, 2, 5, 9]
    ]
    success, solved = solve_sudoku(board)
    assert success is True
    assert solved is not None
    assert is_valid_board(solved)

def test_wrong_dimensions():
    # 8x9 board
    board = [[0] * 9 for _ in range(8)]
    success, solved = solve_sudoku(board)
    assert success is False
    assert solved is None
