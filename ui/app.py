"""
Sudoku CV Project -- UI (bonus options 3 and 4)

Pipeline: input image -> extracted grid -> recognized digits -> USER VERIFIES/
CORRECTS the recognized grid -> solve -> overlay solution on original photo.

The verify-and-correct step exists because Phase 2's model is still being
improved -- it decouples "does the solver + overlay work correctly" from
"is the digit recognizer accurate yet". You confirm/fix the grid by hand,
then everything downstream (solving, overlay) runs on your confirmed grid,
not the model's raw (possibly wrong) output.

Run with:
    streamlit run ui/app.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pipeline.real_pipeline import extract_grid, recognize_digits
# from pipeline.mock_pipeline import extract_grid, recognize_digits

from pipeline.interfaces import GridNotFoundError
from pipeline.overlay import draw_solution_on_original
from solver.sudoku_solver import solve_sudoku


def to_display_rgb(img: np.ndarray) -> np.ndarray:
    """Handles both grayscale (real pipeline) and BGR (mock pipeline) warped grids."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Sudoku Solver", layout="wide")
st.title("Sudoku Solver")
st.caption(
    "Upload a photo of a Sudoku puzzle, verify/correct the recognized digits, "
    "then see it solved and overlaid on your original photo."
)

uploaded_file = st.file_uploader("Upload a Sudoku photo", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to get started.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_id = hashlib.md5(file_bytes).hexdigest()

if st.session_state.get("file_id") != file_id:
    st.session_state["file_id"] = file_id
    st.session_state["solve_attempted"] = False
    st.session_state["solved"] = False
    st.session_state["solved_grid"] = None
    st.session_state["confirmed_grid"] = None

    pil_image = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    st.session_state["pil_image"] = pil_image
    st.session_state["original_bgr"] = original_bgr

    try:
        extraction = extract_grid(original_bgr)
        recognition = recognize_digits(extraction)
        st.session_state["extraction_error"] = None
        st.session_state["extraction"] = extraction
        st.session_state["recognition_grid"] = recognition.grid
        st.session_state["recognition_confidences"] = recognition.confidences
    except GridNotFoundError as e:
        st.session_state["extraction_error"] = str(e)
        st.session_state["extraction"] = None

pil_image = st.session_state["pil_image"]
original_bgr = st.session_state["original_bgr"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Input image")
    st.image(pil_image, width=350)

if st.session_state["extraction_error"] is not None:
    with col2:
        st.subheader("2. Extracted grid")
        st.error(f"Could not find a Sudoku grid in this image: {st.session_state['extraction_error']}")
    st.stop()

extraction = st.session_state["extraction"]

with col2:
    st.subheader("2. Extracted grid")
    st.image(to_display_rgb(extraction.warped_grid), width=350)
    if extraction.detection_confidence < 0.6:
        st.warning(
            f"Low grid-detection confidence ({extraction.detection_confidence:.0%}). "
            "Result may be unreliable."
        )

st.subheader("3. Verify / correct the recognized digits")
st.caption(
    "0 means the model thinks this cell is empty. Fix any wrong cells directly "
    "in the table below, then click Confirm & Solve. You can re-edit and "
    "re-solve as many times as you need."
)

confidences = st.session_state["recognition_confidences"]
if confidences is not None:
    low_conf_cells = [
        (r, c)
        for r in range(9)
        for c in range(9)
        if st.session_state["recognition_grid"][r][c] != 0 and confidences[r][c] < 0.7
    ]
    if low_conf_cells:
        st.caption(f"Model wasn't confident about these cells (row, col), double-check them: {low_conf_cells}")

editor_key = f"editor_{file_id}"
edited_df = st.data_editor(
    pd.DataFrame(st.session_state["recognition_grid"]),
    key=editor_key,
    use_container_width=True,
    column_config={
        str(c): st.column_config.NumberColumn(min_value=0, max_value=9, step=1, width="small")
        for c in range(9)
    },
)

if st.button("Confirm & Solve", type="primary"):
    try:
        confirmed_grid = edited_df.astype(int).values.tolist()
    except (ValueError, TypeError):
        st.error("All cells must be whole numbers 0-9 (0 = empty). Please fix and try again.")
        st.stop()

    st.session_state["confirmed_grid"] = confirmed_grid
    success, solved_grid = solve_sudoku(confirmed_grid)
    st.session_state["solve_attempted"] = True
    st.session_state["solved"] = success
    st.session_state["solved_grid"] = solved_grid

if st.session_state["solve_attempted"]:
    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("4. Final answer")
        if st.session_state["solved"]:
            st.dataframe(pd.DataFrame(st.session_state["solved_grid"]), use_container_width=True)
        else:
            st.error(
                "Couldn't solve this grid -- it's either internally inconsistent "
                "(e.g. a repeated digit in some row/column/box) or has no valid "
                "solution. Edit the table above and click Confirm & Solve again."
            )

    if st.session_state["solved"]:
        with col4:
            st.subheader("Solution overlaid on original photo")
            overlay_bgr = draw_solution_on_original(
                original_bgr,
                extraction,
                st.session_state["confirmed_grid"],
                st.session_state["solved_grid"],
            )
            st.image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)