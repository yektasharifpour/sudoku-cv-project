"""
Real implementation of pipeline/interfaces.py, built directly from the tested
logic in phase1_grid_extraction.ipynb (Block 1.8: process_sudoku_image) and
phase2_digit_recognition.ipynb (Block 2.5: DigitClassifier, Block 2.6:
preprocessing).

Differences from the notebook code (adapted for library/UI use, not
behavior changes):
  - extract_grid() takes an in-memory BGR image (what the UI already has
    from the file uploader), not a file path -- no cv2.imread() here.
  - No checkpoint .npz is written -- the UI doesn't need Phase 1's
    Colab-restart-recovery checkpoints, it runs the pipeline live.
  - Failures raise GridNotFoundError instead of returning a status dict,
    matching pipeline/interfaces.py's contract.
  - inverse_perspective_matrix is added (np.linalg.inv of the forward
    matrix Phase 1 computes) since Phase 1 itself never needed the inverse.

NOT YET VALIDATED END-TO-END: I could not run PyTorch in the environment
that built this file (no network access to install it), so recognize_digits()
below has not been executed against a real image. The model-loading and
preprocessing code was written to exactly match Block 2.5 / Block 2.6 of
phase2_digit_recognition.ipynb, and the checkpoint files were verified
structurally (layer shapes match the DigitClassifier class exactly), but
please do the first real run yourself and tell me what happens if anything
looks off.
"""

import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.interfaces import (
    DigitRecognitionResult,
    GridExtractionResult,
    GridNotFoundError,
)

# --- Grid extraction (Phase 1, Block 1.8) -----------------------------------

_SIDE = 900  # matches process_sudoku_image's `side = 900`
_EPSILON_VALUES = (0.02, 0.04, 0.05, 0.06)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Same as Phase 1's order_corners (Block 1.4): sorts 4 points into
    top-left, top-right, bottom-right, bottom-left order."""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def clean_cell(cell: np.ndarray, margin_ratio: float = 0.18) -> np.ndarray:
    """Same as Phase 1's clean_cell (Block 1.6)."""
    h, w = cell.shape
    m = int(h * margin_ratio)
    return cell[m : h - m, m : w - m]


def extract_grid(input_image: np.ndarray) -> GridExtractionResult:
    """
    Real grid extraction, transplanted from process_sudoku_image()
    (phase1_grid_extraction.ipynb, Block 1.8) -- same algorithm, same
    epsilon-retry loop, just adapted to work on an in-memory image and
    raise instead of returning a status dict.
    """
    if input_image is None or input_image.size == 0:
        raise GridNotFoundError("Empty image")

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise GridNotFoundError("No contours found in image")

    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)

    approx = None
    for eps in _EPSILON_VALUES:
        candidate = cv2.approxPolyDP(largest_contour, eps * perimeter, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is None:
        raise GridNotFoundError("Could not find 4 grid corners")

    corners = order_corners(approx)
    dst = np.array(
        [[0, 0], [_SIDE - 1, 0], [_SIDE - 1, _SIDE - 1], [0, _SIDE - 1]],
        dtype="float32",
    )
    transform_matrix = cv2.getPerspectiveTransform(corners, dst)
    inverse_matrix = np.linalg.inv(transform_matrix)

    warped = cv2.warpPerspective(gray, transform_matrix, (_SIDE, _SIDE))
    warped_thresh = cv2.warpPerspective(thresh, transform_matrix, (_SIDE, _SIDE))

    cell_size = _SIDE // 9
    cell_images = []
    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            cell_images.append(clean_cell(warped_thresh[y1:y2, x1:x2]))

    return GridExtractionResult(
        warped_grid=warped,  # grayscale, matches Phase 1's own preview
        cell_images=cell_images,  # 81 cells, ~64x64, binary (0/255), row-major
        inverse_perspective_matrix=inverse_matrix,
        detection_confidence=1.0,  # Phase 1 doesn't compute a real score for this
    )


# --- Digit recognition (Phase 2, Block 2.5 / 2.6) ---------------------------


class DigitClassifier(nn.Module):
    """Original architecture (2 conv + 2 fc), used by digit_classifier_final.pt (v1)
    and digit_classifier_final_v2.pt. Exact copy of phase2_digit_recognition.ipynb's
    Block 2.5."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DigitClassifierV2(nn.Module):
    """Deeper architecture (3 conv + batchnorm), used by digit_classifier_final_v3arch.pt
    and digit_classifier_final_v4.pt. Exact copy of phase2_digit_recognition_final.ipynb's
    Block 2.14 (the version actually used to train v4 -- verified structurally against the
    real checkpoint's saved tensor shapes, not just the notebook code)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Maps the "architecture" string stored inside each checkpoint to the class that can
# actually load it. This has already changed once (v1/v2 -> v3arch/v4), so rather than
# hardcoding a single class, pick based on what the checkpoint itself says it is -- the
# next architecture change only needs a new entry here, not a rewrite of _load_model.
_ARCHITECTURE_REGISTRY = {
    "DigitClassifier (2 conv layers + 2 fc layers)": DigitClassifier,
    "DigitClassifierV2 (3 conv layers + batchnorm)": DigitClassifierV2,
}

# v4 is the current best: deeper architecture, 99.51% test accuracy, and -- unlike
# earlier checkpoints -- trained on data that includes printed AND cursive synthetic
# fonts, not just handwriting. Point this at wherever you place the file.
DEFAULT_WEIGHTS_PATH = os.environ.get(
    "DIGIT_MODEL_PATH", "models/digit_classifier_final_v4.pt"
)

_model_cache = {}


def _load_model(weights_path: str = DEFAULT_WEIGHTS_PATH) -> nn.Module:
    if weights_path in _model_cache:
        return _model_cache[weights_path]

    checkpoint = torch.load(weights_path, map_location="cpu")
    arch_name = checkpoint.get("architecture")
    model_class = _ARCHITECTURE_REGISTRY.get(arch_name)
    if model_class is None:
        raise ValueError(
            f"Checkpoint '{weights_path}' declares architecture '{arch_name}', which "
            f"isn't in _ARCHITECTURE_REGISTRY. Known architectures: "
            f"{list(_ARCHITECTURE_REGISTRY.keys())}. Add the new class and register it "
            f"here rather than guessing -- a silent mismatch fails with a confusing "
            f"tensor-shape error instead of this clear one."
        )

    model = model_class(num_classes=checkpoint.get("num_classes", 10))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model_cache[weights_path] = model
    return model


def recognize_digits(
    extraction: GridExtractionResult, weights_path: str = DEFAULT_WEIGHTS_PATH
) -> DigitRecognitionResult:
    """
    Real digit recognition. Every one of the 81 cells is classified by the
    model directly -- including cells Phase 1 might independently consider
    "empty" -- since the model was trained with its own dedicated empty
    class (class 0) and is designed to make that call itself, exactly the
    way Block 2.9's evaluation in the notebook does it.

    Preprocessing matches Block 2.6's DigitDataset exactly: resize to
    28x28, scale to [0, 1] float32, no augmentation at inference time.
    """
    model = _load_model(weights_path)

    batch = []
    for cell in extraction.cell_images:
        resized = cv2.resize(cell, (28, 28))
        normalized = resized.astype("float32") / 255.0
        batch.append(normalized)

    batch_tensor = torch.tensor(np.stack(batch)).unsqueeze(1)  # (81, 1, 28, 28)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)

    predictions = predictions.numpy().reshape(9, 9).tolist()
    confidences = confidences.numpy().reshape(9, 9).tolist()

    return DigitRecognitionResult(grid=predictions, confidences=confidences)