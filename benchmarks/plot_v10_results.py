"""
Generate v10 training curve and confusion matrix plots from the real training output.

Saves PNGs to document/Sudoku-cv-project/images/ so they can be embedded in Obsidian.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- v10 Training Data (from MAIN_OUTPUT/digit_recognition_v10.txt) ---
epochs = list(range(1, 18))  # 17 epochs, early stopping
train_acc = [
    0.8652, 0.9322, 0.9426, 0.9480, 0.9513, 0.9543, 0.9568,
    0.9578, 0.9594, 0.9600, 0.9612, 0.9619, 0.9634, 0.9636,
    0.9642, 0.9651, 0.9652,
]
val_acc = [
    0.9233, 0.9335, 0.9345, 0.9412, 0.9478, 0.9413, 0.9497,
    0.9499, 0.9452, 0.9487, 0.9516, 0.9536, 0.9504, 0.9504,
    0.9491, 0.9468, 0.9501,
]

# --- v10 Confusion Matrix (from MAIN_OUTPUT/digit_recognition_v10.txt) ---
cm = np.array([
    [2106,   8,   2,   3,   3,   2,   2,   2,   2,   2],
    [  16, 117,   0,   0,   0,   0,   0,   3,   1,   0],
    [  11,   0, 120,   0,   0,   0,   1,   0,   0,   0],
    [  14,   0,   1, 117,   0,   0,   0,   0,   0,   0],
    [  11,   0,   1,   0, 118,   0,   0,   1,   1,   0],
    [  14,   0,   0,   0,   0, 109,   0,   0,   0,   0],
    [  17,   0,   0,   0,   0,   0, 101,   0,   0,   0],
    [  14,   2,   0,   0,   0,   0,   0, 125,   1,   0],
    [  16,   0,   1,   0,   0,   1,   0,   0, 134,   1],
    [   9,   0,   0,   0,   0,   1,   0,   0,   0, 110],
])
labels = [str(i) for i in range(10)]
class_names = ["0\n(empty)"] + [str(i) for i in range(1, 10)]

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "document", "Sudoku-cv-project", "images"
)
os.makedirs(OUT_DIR, exist_ok=True)


def plot_training_curve():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, "o-", label="Train accuracy", color="#2563eb", markersize=5)
    ax.plot(epochs, val_acc, "s-", label="Validation accuracy", color="#dc2626", markersize=5)
    ax.axvline(x=17, color="gray", linestyle="--", alpha=0.6, label="Early stop (epoch 17)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("DigitClassifierV2 (v10) — Training Curves\n"
                 "Domain-matched training: real Sudoku cells + MNIST + printed fonts + synthetic empty")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.0)

    path = os.path.join(OUT_DIR, "v10_training_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(9, 7))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_pct, annot=True, fmt=".2f", cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Proportion (row-normalized)"},
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("DigitClassifierV2 (v10) — Confusion Matrix\n"
                 "3,321 real Sudoku test cells (41 test images)")

    # Annotate the diagonal with absolute counts
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j + 0.5, i + 0.82, f"n={cm[i, j]}", fontsize=7,
                        ha="center", va="center", color="gray")

    path = os.path.join(OUT_DIR, "v10_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_training_curve()
    plot_confusion_matrix()
    print("\nDone. Copy the images into the vault's images/ folder if needed.")
