"""
pipeline/real_pipeline.py

Real implementation of pipeline/interfaces.py.
Grid extraction (extract_grid) is a thin wrapper around shared src/phase1_*.py modules.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.phase1_preprocessing import to_binary_threshold
from src.phase1_contour_detection import find_grid_corners
from src.phase1_perspective_warp import SIDE, compute_transform, split_cells, warp_grid
from src.phase1_cell_cleanup import clean_cell


def gatekeeper_is_empty(cell_tensor_or_np) -> bool:
    """
    PRODUCTION INTERCEPTOR: Analyzes any cell image resolution using connected components.
    Protects live inference against paper creases using calibrated v10 relative anchors.
    """
    if hasattr(cell_tensor_or_np, "cpu"):
        cell_np = cell_tensor_or_np.detach().cpu().numpy().squeeze()
    else:
        cell_np = cell_tensor_or_np.squeeze()

    if cell_np.ndim != 2:
        return False

    if cell_np.dtype != np.uint8:
        img_uint8 = (cell_np * 255).astype(np.uint8)
    else:
        img_uint8 = cell_np.astype(np.uint8)
    img_uint8 = cv2.resize(img_uint8, (28, 28))

    _, binary = cv2.threshold(img_uint8, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    if num_labels <= 1:
        return True

    max_area = 0
    largest_idx = -1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            largest_idx = i

    if max_area < 15:
        return True

    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    touches_top = (y == 0)
    touches_bottom = (y + h >= 27)
    touches_left = (x == 0)
    touches_right = (x + w >= 27)

    # Rule A: Vertical Crease / Grid-Line Interceptor
    if touches_top and touches_bottom:
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio <= 0.25:
            return True

    # Rule B: Horizontal Crease Interceptor
    if touches_left and touches_right:
        aspect_ratio = h / float(w) if w > 0 else 0
        if aspect_ratio <= 0.25:
            return True

    # Rule C: Perimeter Noise Sweeper (Calibrated to 65px for v10 specifications)
    if (touches_top or touches_bottom or touches_left or touches_right):
        if max_area < 65:
            return True

    return False


def extract_grid(input_image: np.ndarray) -> GridExtractionResult:
    if input_image is None or input_image.size == 0:
        raise GridNotFoundError("Empty image")

    gray, thresh = to_binary_threshold(input_image)

    corners = find_grid_corners(thresh)
    if corners is None:
        raise GridNotFoundError("Could not find 4 grid corners")

    transform_matrix, inverse_matrix = compute_transform(corners, side=SIDE)
    warped, warped_thresh = warp_grid(gray, thresh, transform_matrix, side=SIDE)
    raw_cells = split_cells(warped_thresh, side=SIDE)

    # Calls your custom 18% clean_cell from src/phase1_cell_cleanup.py
    cell_images = [clean_cell(c) for c in raw_cells]

    return GridExtractionResult(
        warped_grid=warped,
        cell_images=cell_images,
        inverse_perspective_matrix=inverse_matrix,
        detection_confidence=1.0,
    )


# --- Digit recognition (Phase 2) ---------------------------

class DigitClassifierV2(nn.Module):
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


_ARCHITECTURE_REGISTRY = {
    "DigitClassifierV2 (3 conv layers + batchnorm)": DigitClassifierV2,
    "DigitClassifierV2 (3 conv layers + batchnorm + Gatekeeper Override)": DigitClassifierV2,
}

DEFAULT_WEIGHTS_PATH = os.environ.get(
    "DIGIT_MODEL_PATH", "models/digit_classifier_final_v10.pt"
)

_model_cache = {}


def _load_model(weights_path: str = DEFAULT_WEIGHTS_PATH) -> nn.Module:
    if weights_path in _model_cache:
        return _model_cache[weights_path]

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    arch_name = checkpoint.get("architecture")
    model_class = _ARCHITECTURE_REGISTRY.get(arch_name)
    if model_class is None:
        raise ValueError(f"Checkpoint '{weights_path}' declares unknown architecture '{arch_name}'.")

    model = model_class(num_classes=checkpoint.get("num_classes", 10))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model_cache[weights_path] = model
    return model


def recognize_digits(
    extraction: GridExtractionResult, weights_path: str = DEFAULT_WEIGHTS_PATH
) -> DigitRecognitionResult:
    model = _load_model(weights_path)

    batch = []
    for cell in extraction.cell_images:
        resized = cv2.resize(cell, (28, 28))
        normalized = resized.astype("float32") / 255.0
        batch.append(normalized)

    batch_tensor = torch.tensor(np.stack(batch)).unsqueeze(1)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)

    predictions_np = predictions.numpy()
    confidences_np = confidences.numpy()

    for idx in range(81):
        if gatekeeper_is_empty(extraction.cell_images[idx]):
            predictions_np[idx] = 0
            confidences_np[idx] = 1.0

    predictions = predictions_np.reshape(9, 9).tolist()
    confidences = confidences_np.reshape(9, 9).tolist()

    return DigitRecognitionResult(grid=predictions, confidences=confidences)