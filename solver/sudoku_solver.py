import numpy as np
from typing import List, Union, Tuple, Optional

def is_valid_board(board: Union[List[List[int]], np.ndarray]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    blocks = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val != 0:
                block_idx = (r // 3) * 3 + (c // 3)
                if val in rows[r] or val in cols[c] or val in blocks[block_idx]:
                    return False
                rows[r].add(val)
                cols[c].add(val)
                blocks[block_idx].add(val)
    return True

def is_safe(board: List[List[int]], row: int, col: int, num: int) -> bool:
    # Check row
    for x in range(9):
        if board[row][x] == num:
            return False

    # Check column
    for x in range(9):
        if board[x][col] == num:
            return False

    # Check 3x3 block
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False

    return True

def find_empty_location(board: List[List[int]]) -> Optional[Tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None

def solve_backtracking(board: List[List[int]]) -> bool:
    """
    Solves the Sudoku board using backtracking
    """
    empty_loc = find_empty_location(board)
    if not empty_loc:
        return True # Puzzle is solved

    row, col = empty_loc

    for num in range(1, 10):
        if is_safe(board, row, col, num):
            board[row][col] = num

            if solve_backtracking(board):
                return True

            # Backtrack
            board[row][col] = 0

    return False

def solve_sudoku(board: Union[List[List[int]], np.ndarray]) -> Tuple[bool, Optional[Union[List[List[int]], np.ndarray]]]:
    """
    Solves a 9x9 Sudoku board.
    If unsolvable or initially invalid, returns (False, None).
    """
    # Ensure it's a 9x9 structure
    if len(board) != 9 or any(len(row) != 9 for row in board):
        return False, None

    is_numpy = isinstance(board, np.ndarray)

    if is_numpy:
        working_board = board.tolist()
    else:
        working_board = [row[:] for row in board]

    if not is_valid_board(working_board):
        return False, None

    success = solve_backtracking(working_board)

    if not success:
        return False, None

    if is_numpy:
        return True, np.array(working_board)
    else:
        return True, working_board
