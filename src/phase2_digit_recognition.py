"""
digit_recognition/phase2_digit_recognition.py

Phase 2 -- Digit recognition (English first)

REFACTORED (v10): Decoupled validation and final testing evaluation loops.
Validation metrics now track raw CNN performance to prevent premature early stopping,
while the final test evaluation uses calibrated geometric filters to maximize precision.
"""

import glob
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from domain_matched_augmentation import (
    domain_match_empty_batch,
    domain_match_font_batch,
    domain_match_glyph_batch,
)

# --- Config ------------------------------------------------------------

TRAIN_CHECKPOINT_DIR = "checkpoints/phase1/train"
TEST_CHECKPOINT_DIR = "checkpoints/phase1/test"

COPIES_PER_MNIST_IMAGE = 1
SAMPLES_PER_DIGIT_PER_FONT = 12
VAL_FRACTION = 0.15
MAX_EPOCHS = 30
PATIENCE = 5
SEED = 42


# --- Real labeled cells from Phase 1 checkpoints ----------------------------

def load_real_cells_from_checkpoints(checkpoint_dir: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.npz"))
    images, labels = [], []
    n_with_gt = 0

    for cp_path in checkpoint_files:
        data = np.load(cp_path, allow_pickle=True)
        cells = data["cells"]
        empty_flags = data["empty_flags"]
        cell_labels = data["cell_labels"] if "cell_labels" in data else np.full(81, -1)

        has_gt = bool(np.any(cell_labels != -1))
        if has_gt:
            n_with_gt += 1

        for cell, is_empty, gt_label in zip(cells, empty_flags, cell_labels):
            if has_gt:
                label = int(gt_label)
            elif is_empty:
                label = 0
            else:
                continue

            resized = cv2.resize(cell.astype(np.uint8), (28, 28))
            images.append(resized.astype("float32") / 255.0)
            labels.append(label)

    images = np.array(images, dtype="float32") if images else np.empty((0, 28, 28), dtype="float32")
    labels = np.array(labels, dtype=int)
    return images, labels, n_with_gt, len(checkpoint_files)


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root="./datasets", train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root="./datasets", train=False, download=True, transform=transform)
    train_images = np.stack([np.array(img.squeeze()) for img, _ in train])
    train_labels = np.array([label for _, label in train])
    test_images = np.stack([np.array(img.squeeze()) for img, _ in test])
    test_labels = np.array([label for _, label in test])

    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    return all_images, all_labels


def get_font_paths():
    candidates = []
    candidates += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    candidates += glob.glob("C:/Windows/Fonts/*.ttf")
    candidates += glob.glob("/System/Library/Fonts/**/*.ttf", recursive=True)
    candidates += glob.glob("/Library/Fonts/**/*.ttf", recursive=True)

    if not candidates:
        print("WARNING: no .ttf fonts found via get_font_paths() on this OS.")
    else:
        print(f"Found {len(candidates)} font file(s) for synthetic printed digits.")

    return candidates


# --- Model Configuration ----------------------------------------------------

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


class DigitDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype("float32")
        label = int(self.labels[idx])
        if self.augment:
            angle = np.random.uniform(-6, 6)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28))
            tx, ty = np.random.uniform(-1.5, 1.5, size=2)
            M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M_shift, (28, 28))
        img = np.clip(img, 0, 1)
        return torch.tensor(img).unsqueeze(0), label


def stratified_split(images, labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_fraction)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return (
        images[train_idx], labels[train_idx],
        images[val_idx], labels[val_idx],
    )


def stratified_group_split(images, labels, groups, val_fraction, seed):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    train_idx, val_idx = [], []

    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        class_groups = groups[class_idx]
        unique_groups = np.unique(class_groups)
        rng.shuffle(unique_groups)

        target_val = int(len(class_idx) * val_fraction)
        val_groups = set()
        val_count = 0
        for g in unique_groups:
            if val_count >= target_val:
                break
            g_size = int(np.sum(class_groups == g))
            val_groups.add(g)
            val_count += g_size

        in_val = np.isin(class_groups, list(val_groups)) if val_groups else np.zeros(len(class_idx), dtype=bool)
        val_idx.extend(class_idx[in_val].tolist())
        train_idx.extend(class_idx[~in_val].tolist())

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    return (
        images[train_idx], labels[train_idx],
        images[val_idx], labels[val_idx],
    )


def gatekeeper_is_empty(cell_tensor_or_np) -> bool:
    """
    PRODUCTION INTERCEPTOR: Analyzes any cell image resolution using connected components.
    CALIBRATED (v10): Uses relative boundary anchors. Drops area limit to 65px
    to prevent false empty overrides on thin digit classes.
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

    # --- INVARIANT GEOMETRIC ANCHOR FILTERS ---
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

    # Rule C: Perimeter Noise Sweeper
    # CALIBRATED: Lowered threshold from 115 to 65 to safely protect thin '1' and '7' shapes
    if (touches_top or touches_bottom or touches_left or touches_right):
        if max_area < 65:
            return True

    return False


def build_dataset():
    print("=" * 70)
    print("Loading real labeled cells from Phase 1 checkpoints (TRAIN fold)...")
    real_train_img, real_train_lbl, n_gt_train, n_total_train = load_real_cells_from_checkpoints(TRAIN_CHECKPOINT_DIR)
    print(f"  {n_gt_train}/{n_total_train} training images had .dat ground truth")
    print(f"  Real labeled cells collected: {real_train_img.shape[0]}")
    for c in range(10):
        print(f"    class {c}: {np.sum(real_train_lbl == c)}")

    print("\nLoading real labeled cells from Phase 1 checkpoints (TEST fold)...")
    real_test_img, real_test_lbl, n_gt_test, n_total_test = load_real_cells_from_checkpoints(TEST_CHECKPOINT_DIR)
    print(f"  {n_gt_test}/{n_total_test} test images had .dat ground truth")
    print(f"  Real labeled test cells collected: {real_test_img.shape[0]}")

    print("\nLoading MNIST...")
    mnist_images, mnist_labels = load_mnist()
    digit_mask = mnist_labels != 0
    mnist_images, mnist_labels = mnist_images[digit_mask], mnist_labels[digit_mask]

    print("Re-photographing MNIST digits through the real threshold+crop pipeline...")
    photo_mnist_img, photo_mnist_lbl = domain_match_glyph_batch(
        mnist_images, mnist_labels, copies_per_image=COPIES_PER_MNIST_IMAGE, seed=SEED
    )

    print("Generating domain-matched printed font digits...")
    font_paths = get_font_paths()
    font_img, font_lbl, font_grp = domain_match_font_batch(
        range(1, 10), font_paths, samples_per_digit_per_font=SAMPLES_PER_DIGIT_PER_FONT, seed=SEED + 1
    )

    photo_mnist_grp = np.array([f"mnist_{i}" for i in range(len(photo_mnist_img))], dtype=object)
    real_digit_grp = np.array([f"real_{i}" for i in range(int(np.sum(real_train_lbl != 0)))], dtype=object)

    digit_pool_img = np.concatenate([photo_mnist_img, font_img, real_train_img[real_train_lbl != 0]], axis=0)
    digit_pool_lbl = np.concatenate([photo_mnist_lbl, font_lbl, real_train_lbl[real_train_lbl != 0]], axis=0)
    digit_pool_grp = np.concatenate([photo_mnist_grp, font_grp, real_digit_grp], axis=0)

    avg_digit_class_count = int(np.mean([np.sum(digit_pool_lbl == c) for c in range(1, 10)]))
    real_empty_pool = real_train_img[real_train_lbl == 0]
    n_synthetic_empty_needed = max(avg_digit_class_count - len(real_empty_pool), 0)
    print(f"\nGenerating {n_synthetic_empty_needed} domain-matched synthetic empty cells "
          f"to balance against the average digit class (~{avg_digit_class_count})...")
    synthetic_empty = domain_match_empty_batch(n_synthetic_empty_needed, seed=SEED + 2)

    empty_pool_img = np.concatenate([real_empty_pool, synthetic_empty], axis=0) if len(real_empty_pool) else synthetic_empty
    empty_pool_lbl = np.zeros(len(empty_pool_img), dtype=int)
    empty_pool_grp = np.array([f"empty_{i}" for i in range(len(empty_pool_img))], dtype=object)

    pool_img = np.concatenate([digit_pool_img, empty_pool_img], axis=0)
    pool_lbl = np.concatenate([digit_pool_lbl, empty_pool_lbl], axis=0)
    pool_grp = np.concatenate([digit_pool_grp, empty_pool_grp], axis=0)

    print("\nSplitting train/validation (stratified per class, grouped by font so no "
          f"font's samples appear on both sides, val fraction = {VAL_FRACTION})...")
    train_img, train_lbl, val_img, val_lbl = stratified_group_split(pool_img, pool_lbl, pool_grp, VAL_FRACTION, SEED)

    print(f"\nFinal train: {train_img.shape}  Final val: {val_img.shape}")
    print("Class distribution (train):")
    for c in range(10):
        print(f"  class {c}: {np.sum(train_lbl == c)}")

    used_real_test = real_test_img.shape[0] > 0
    if used_real_test:
        test_img, test_lbl = real_test_img, real_test_lbl
    else:
        train_img, train_lbl, test_img, test_lbl = stratified_split(train_img, train_lbl, 0.15, SEED + 3)

    return train_img, train_lbl, val_img, val_lbl, test_img, test_lbl, used_real_test


# --- VERSION 10 PATH CONFIGURATIONS ---
CHECKPOINT_PATH = "checkpoints/phase2/digit_classifier_v10_training.pt"


def _dataset_fingerprint(train_lbl, val_lbl, test_lbl) -> dict:
    def counts(labels):
        return tuple(int(np.sum(labels == c)) for c in range(10))

    return {
        "train_size": len(train_lbl),
        "val_size": len(val_lbl),
        "test_size": len(test_lbl),
        "train_class_counts": counts(train_lbl),
        "val_class_counts": counts(val_lbl),
        "test_class_counts": counts(test_lbl),
    }


def evaluate(model, loader, device, use_gatekeeper=False):
    """
    Decoupled Evaluation:
    use_gatekeeper=False used during Validation to avoid early stopping distortion.
    use_gatekeeper=True enabled only during Final real-world test set calculations.
    """
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            _, predicted = torch.max(outputs, 1)

            preds_np = predicted.cpu().numpy()

            if use_gatekeeper:
                for idx in range(images.size(0)):
                    if gatekeeper_is_empty(images[idx]):
                        preds_np[idx] = 0

            labels_np = labels.numpy()
            correct += (preds_np == labels_np).sum()
            total += labels.size(0)
            all_preds.extend(preds_np.tolist())
            all_labels.extend(labels_np.tolist())

    return correct / total, all_preds, all_labels


def print_confusion_matrix(all_labels, all_preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        matrix[t, p] += 1
    header = "      " + " ".join(f"{c:4d}" for c in range(num_classes))
    print(header)
    for c in range(num_classes):
        row = " ".join(f"{matrix[c, p]:4d}" for p in range(num_classes))
        print(f"true {c}: {row}")


def train():
    train_img, train_lbl, val_img, val_lbl, test_img, test_lbl, used_real_test = build_dataset()
    fingerprint = _dataset_fingerprint(train_lbl, val_lbl, test_lbl)

    train_loader = DataLoader(DigitDataset(train_img, train_lbl, augment=True), batch_size=64, shuffle=True)
    val_loader = DataLoader(DigitDataset(val_img, val_lbl, augment=False), batch_size=64, shuffle=False)
    test_loader = DataLoader(DigitDataset(test_img, test_lbl, augment=False), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = DigitClassifierV2(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_state = None

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        if ckpt.get("fingerprint") != fingerprint:
            print(f"\nCheckpoint mismatch. Starting fresh.")
        elif ckpt.get("training_complete"):
            print(f"\nTraining already marked complete. Not retraining.")
            model.load_state_dict(ckpt["best_state"])
            best_state = ckpt["best_state"]
            best_val_acc = ckpt["best_val_acc"]
            start_epoch = MAX_EPOCHS
        else:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            best_state = ckpt["best_state"]
            best_val_acc = ckpt["best_val_acc"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            start_epoch = ckpt["epoch"] + 1
            print(f"\nResuming from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # VALIDATION EVALUATION: Run with use_gatekeeper=False
        val_acc, _, _ = evaluate(model, val_loader, device, use_gatekeeper=False)
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc + 0.0005:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(
            {
                "fingerprint": fingerprint,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "training_complete": False,
            },
            CHECKPOINT_PATH,
        )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered (validation accuracy plateaued).")
            break

    if start_epoch < MAX_EPOCHS:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        ckpt["training_complete"] = True
        torch.save(ckpt, CHECKPOINT_PATH)

    model.load_state_dict(best_state)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION on real, held-out Sanitized Sudoku photo cells (v2_test):")

    # TEST EVALUATION: Run with use_gatekeeper=True
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device, use_gatekeeper=True)
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nConfusion matrix (rows = true label, cols = predicted):")
    print_confusion_matrix(test_labels, test_preds)

    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": 10,
            "class_mapping": "0=empty, 1-9=digits",
            "test_accuracy": test_acc,
            "test_is_real_photos": used_real_test,
            "architecture": "DigitClassifierV2 (3 conv layers + batchnorm + Gatekeeper Override)",
        },
        "models/digit_classifier_final_v10.pt",
    )
    print("\nSaved models/digit_classifier_final_v10.pt")


if __name__ == "__main__":
    train()