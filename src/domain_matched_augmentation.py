"""
digit_recognition/domain_matched_augmentation.py

Why this file exists
---------------------
Phase 2's model was trained on MNIST + Hoda + PIL-rendered font glyphs: all
smooth, grayscale, fully-visible, centered digits. But `real_pipeline.py`'s
extract_grid() feeds the model something very different at inference time:
a hard binary mask (cv2.adaptiveThreshold, values are only 0 or 255) that has
also been cropped by clean_cell()'s fixed 18% margin. The model was never
shown a single training example that looked like what it actually receives.

This module closes that gap. Instead of training on raw MNIST/Hoda/font
glyphs, it "re-photographs" them: pastes each glyph onto a synthetic paper
cell, then pushes it through the *exact same* clean_cell() used in
pipeline/real_pipeline.py (imported directly, not reimplemented, so training
data and inference data are guaranteed to go through identical cropping
logic -- this was a real, separate bug we saw in the test notebook, where a
second hand-copied clean_cell() had silently drifted from Phase 1's).

Usage sketch (see retrain_domain_matched.py for the full driver):

    from digit_recognition.domain_matched_augmentation import (
        domain_match_glyph_batch, domain_match_font_batch, domain_match_empty_batch,
    )

    # mnist_train_images / hoda_train_images: (N, 28, 28) float32 in [0,1],
    # white digit on black background (standard MNIST convention)
    photo_mnist, photo_mnist_labels = domain_match_glyph_batch(
        mnist_train_images, mnist_train_labels, copies_per_image=1
    )
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.real_pipeline import clean_cell  # single source of truth

# --- Constants matching real_pipeline.py's extract_grid() exactly ----------
_CELL_PX_IN_WARP = 900 // 9  # = 100, matches real_pipeline's cell_size
_CANVAS_PX = 300             # synthetic "pre-warp" cell resolution we render at
_ADAPTIVE_BLOCK = 11
_ADAPTIVE_C = 2
_FINAL_PX = 28


# --- Step 1: build a synthetic "photographed paper cell" -------------------

def _paper_background(size: int = _CANVAS_PX, rng: np.random.Generator = None) -> np.ndarray:
    """A plain paper-like background: bright, with mild lighting gradient +
    sensor noise, matching how a real photographed cell looks before any
    ink is placed on it."""
    rng = rng or np.random.default_rng()
    base = rng.uniform(205, 245)
    img = np.full((size, size), base, dtype="float32")

    # mild directional lighting gradient (simulates uneven real-world lighting)
    if rng.random() < 0.7:
        gy, gx = np.mgrid[0:size, 0:size].astype("float32") / size
        angle = rng.uniform(0, 2 * np.pi)
        gradient = (np.cos(angle) * gx + np.sin(angle) * gy) * rng.uniform(-25, 25)
        img += gradient

    # faint grid-line remnants near the cell edges (real warped cells often
    # carry a sliver of the neighboring grid line)
    if rng.random() < 0.35:
        edge = rng.choice(["top", "bottom", "left", "right"])
        thickness = rng.integers(2, 5)
        dark = rng.uniform(40, 110)
        if edge == "top":
            img[:thickness, :] = dark
        elif edge == "bottom":
            img[-thickness:, :] = dark
        elif edge == "left":
            img[:, :thickness] = dark
        else:
            img[:, -thickness:] = dark

    img += rng.normal(0, 4, size=(size, size))
    return np.clip(img, 0, 255).astype("float32")


def _paste_glyph_as_ink(
    canvas: np.ndarray,
    glyph_white_on_black: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float] = (0.55, 0.9),
    rotation_range: Tuple[float, float] = (-9, 9),
) -> np.ndarray:
    """Takes a standard white-digit-on-black glyph (MNIST/Hoda convention),
    inverts it to dark-ink-on-light-paper (how a real printed/handwritten
    digit looks before any thresholding), scales/rotates/jitters it, and
    pastes it onto the paper canvas."""
    size = canvas.shape[0]
    glyph = glyph_white_on_black.astype("float32")
    if glyph.max() > 1.5:
        glyph = glyph / 255.0

    ink = 1.0 - glyph  # invert: stroke becomes dark, background becomes light
    ink_u8 = (ink * 255).astype("uint8")

    target = int(size * rng.uniform(*scale_range))
    resized = cv2.resize(ink_u8, (target, target), interpolation=cv2.INTER_CUBIC)

    angle = rng.uniform(*rotation_range)
    M = cv2.getRotationMatrix2D((target / 2, target / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        resized, M, (target, target), borderValue=255, flags=cv2.INTER_CUBIC
    )

    max_offset = size - target
    x0 = int(rng.integers(0, max(max_offset, 1)))
    y0 = int(rng.integers(0, max(max_offset, 1)))

    out = canvas.copy()
    region = out[y0 : y0 + target, x0 : x0 + target]
    # darken canvas wherever ink is darker than the paper (soft composite,
    # not a hard paste, so ink blends with the paper's own noise/gradient)
    out[y0 : y0 + target, x0 : x0 + target] = np.minimum(region, rotated.astype("float32"))
    return out


def _render_font_glyph_as_ink(
    canvas: np.ndarray,
    digit: int,
    font_path: str,
    rng: np.random.Generator,
    size_range: Tuple[int, int] = (150, 220),
) -> np.ndarray:
    """Same idea as _paste_glyph_as_ink, but the ink comes from rendering a
    font character directly, for printed/cursive-font training samples.

    Returns None if the font can't be used (corrupt file, unreadable, etc.)
    -- callers MUST skip on None rather than treat it as a valid sample.
    Silently falling back to the blank canvas here would mean a genuinely
    blank image gets kept and labeled as the intended digit, which is
    exactly the kind of mislabeled training data that previously caused the
    model to over-predict "empty" on real digits. Fail loudly instead."""
    canvas_size = canvas.shape[0]
    pil_canvas = Image.fromarray(np.full((canvas_size, canvas_size), 255, dtype="uint8"))
    draw = ImageDraw.Draw(pil_canvas)
    font_size = int(rng.integers(*size_range))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"  WARNING: skipping font '{font_path}' (failed to load: {e})")
        return None

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size - text_w) // 2 - bbox[0] + int(rng.integers(-20, 21))
    y = (canvas_size - text_h) // 2 - bbox[1] + int(rng.integers(-20, 21))
    draw.text((x, y), text, fill=0, font=font)

    angle = rng.uniform(-9, 9)
    rendered = np.array(pil_canvas).astype("float32")
    M = cv2.getRotationMatrix2D((canvas_size / 2, canvas_size / 2), angle, 1.0)
    rotated = cv2.warpAffine(rendered, M, (canvas_size, canvas_size), borderValue=255)

    return np.minimum(canvas, rotated)


# --- Step 2: run it through the REAL pipeline's threshold + crop -----------

def _photograph_and_pipeline_crop(canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies camera-like degradation, then the exact same adaptiveThreshold
    parameters and clean_cell() crop that pipeline/real_pipeline.py uses, so
    the result is statistically indistinguishable from a real inference-time
    cell image."""
    img = canvas.copy()

    # mild camera blur / defocus
    if rng.random() < 0.6:
        k = int(rng.choice([3, 3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    img = np.clip(img + rng.normal(0, 3, size=img.shape), 0, 255).astype("uint8")

    # === identical to real_pipeline.extract_grid()'s threshold step ===
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        _ADAPTIVE_BLOCK, _ADAPTIVE_C,
    )

    # simulate the warpPerspective resize down to the real cell_size (100px)
    cell_100 = cv2.resize(thresh, (_CELL_PX_IN_WARP, _CELL_PX_IN_WARP), interpolation=cv2.INTER_AREA)
    cell_100 = ((cell_100 > 127) * 255).astype("uint8")  # re-binarize after resize blending

    # === identical to real_pipeline.clean_cell(), imported not copied ===
    cropped = clean_cell(cell_100)

    # === identical to real_pipeline.recognize_digits()'s final preprocessing ===
    resized = cv2.resize(cropped, (_FINAL_PX, _FINAL_PX))
    return (resized.astype("float32") / 255.0)


# --- Public batch builders ---------------------------------------------------

def domain_match_glyph_batch(
    images: np.ndarray,
    labels: np.ndarray,
    copies_per_image: int = 1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-photographs an array of (N, 28, 28) white-on-black glyphs (MNIST or
    Hoda convention) `copies_per_image` times each, running every copy
    through the real threshold+crop pipeline. Use this on your existing
    MNIST/Hoda arrays instead of feeding them to the model raw."""
    rng = np.random.default_rng(seed)
    out_images, out_labels = [], []
    for img, label in zip(images, labels):
        for _ in range(copies_per_image):
            canvas = _paper_background(rng=rng)
            canvas = _paste_glyph_as_ink(canvas, img, rng=rng)
            out_images.append(_photograph_and_pipeline_crop(canvas, rng))
            out_labels.append(label)
    return np.array(out_images, dtype="float32"), np.array(out_labels)


def domain_match_font_batch(
    digits: Sequence[int],
    font_paths: Sequence[str],
    samples_per_digit_per_font: int = 40,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same idea as Block 2.12/2.13's printed & cursive font generation, but
    every sample goes through the real threshold+crop pipeline instead of
    being used as a clean PIL render.

    Also returns `groups`: one entry per returned sample, equal to the
    source font's path. Each font only contributes a handful of visually
    distinct glyphs (per digit) even though it's sampled many times with
    small rotation/scale/position jitter -- if those near-duplicate samples
    get split randomly between train and val (as a plain per-class split
    would), val ends up full of near-duplicates of what the model already
    trained on, which inflates val accuracy without the model actually
    generalizing. Callers should split by `groups`, not just by label, so
    every sample from a given font stays entirely on one side."""
    rng = np.random.default_rng(seed)
    out_images, out_labels, out_groups = [], [], []
    n_skipped = 0
    bad_fonts = set()
    for digit in digits:
        for font_path in font_paths:
            for _ in range(samples_per_digit_per_font):
                canvas = _paper_background(rng=rng)
                canvas = _render_font_glyph_as_ink(canvas, digit, font_path, rng=rng)
                if canvas is None:
                    n_skipped += 1
                    bad_fonts.add(font_path)
                    continue
                out_images.append(_photograph_and_pipeline_crop(canvas, rng))
                out_labels.append(digit)
                out_groups.append(font_path)

    if n_skipped:
        print(
            f"domain_match_font_batch: skipped {n_skipped} sample(s) from "
            f"{len(bad_fonts)} unusable font(s): {sorted(bad_fonts)}"
        )

    if not out_images:
        return (
            np.empty((0, _FINAL_PX, _FINAL_PX), dtype="float32"),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=object),
        )

    return (
        np.array(out_images, dtype="float32"),
        np.array(out_labels),
        np.array(out_groups, dtype=object),
    )


def domain_match_empty_batch(n_samples: int, seed: int = 0) -> np.ndarray:
    """Blank paper cells (no ink at all) run through the same pipeline.
    Because adaptiveThreshold responds to the paper's own lighting
    gradient/noise, this naturally reproduces the same speckle pattern real
    empty cells show -- more faithfully than hand-tuned synthetic noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        canvas = _paper_background(rng=rng)
        out.append(_photograph_and_pipeline_crop(canvas, rng))
    return np.array(out, dtype="float32")