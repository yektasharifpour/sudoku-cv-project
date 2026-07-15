"""
Benchmarks wall-clock time for each pipeline stage on a representative test image.

Usage:
    python benchmarks/benchmark_pipeline.py [path/to/image.jpg]

If no image path is given, defaults to the first image in datasets/sudoku_dataset/testing.
Runs each stage N times and reports mean + std so the numbers are stable.
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch

from pipeline.real_pipeline import extract_grid, recognize_digits
from pipeline.overlay import draw_solution_on_original
from solver.sudoku_solver import solve_sudoku

DEFAULT_IMAGE = "datasets/sudoku_dataset/testing/image170.jpg"
N_RUNS = 5  # number of repetitions per stage for stable timing


def time_it(fn, *args, **kwargs):
    """Run fn once, return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark(image_path: str, n_runs: int = N_RUNS):
    print(f"{'=' * 60}")
    print(f"Benchmarking pipeline stages on: {image_path}")
    print(f"Repetitions per stage: {n_runs}")
    print(f"PyTorch device available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}\n")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # --- Warm up the model cache (first load is slow due to torch.load) ---
    print("Warming up model cache (not counted)...")
    warmup_extraction = extract_grid(img)
    warmup_recognition = recognize_digits(warmup_extraction)
    print("Warm-up complete.\n")

    # --- Stage timings ---
    timings = {}

    # 1. extract_grid
    samples = []
    for _ in range(n_runs):
        _, t = time_it(extract_grid, img)
        samples.append(t)
    timings["extract_grid()"] = samples

    # 2. recognize_digits
    samples = []
    for _ in range(n_runs):
        _, t = time_it(recognize_digits, warmup_extraction)
        samples.append(t)
    timings["recognize_digits()"] = samples

    # 3. solve_sudoku
    samples = []
    for _ in range(n_runs):
        _, t = time_it(solve_sudoku, warmup_recognition.grid)
        samples.append(t)
    timings["solve_sudoku()"] = samples

    # 4. draw_solution_on_original
    success, solved_grid = solve_sudoku(warmup_recognition.grid)
    samples = []
    for _ in range(n_runs):
        _, t = time_it(
            draw_solution_on_original,
            img,
            warmup_extraction,
            warmup_recognition.grid,
            solved_grid,
        )
        samples.append(t)
    timings["draw_solution_on_original()"] = samples

    # --- End-to-end (all stages in sequence) ---
    samples = []
    for _ in range(n_runs):
        start = time.perf_counter()
        extraction = extract_grid(img)
        recognition = recognize_digits(extraction)
        success, solved = solve_sudoku(recognition.grid)
        if success:
            _ = draw_solution_on_original(img, extraction, recognition.grid, solved)
        samples.append(time.perf_counter() - start)
    timings["Total end-to-end"] = samples

    # --- Report ---
    print(f"{'Stage':<35} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10}")
    print("-" * 67)
    for stage, samples in timings.items():
        mean = statistics.mean(samples)
        std = statistics.stdev(samples) if len(samples) > 1 else 0.0
        mn = min(samples)
        print(f"{stage:<35} {mean:>10.4f} {std:>10.4f} {mn:>10.4f}")

    print(f"\n{'-' * 67}")
    print(f"Solved successfully: {success}")
    print(f"Recognized grid (row 0): {warmup_recognition.grid[0]}")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    benchmark(image_path)
