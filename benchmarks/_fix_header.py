"""One-off fix: remove the misplaced duplicate Block 2.10 header."""

path = "document/Sudoku-cv-project/Phase2/Training progress.md"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the pattern: line with "weighted avg", blank, "## Block 2.10 ...", "### v10 Model"
# We want to delete the "## Block 2.10" line that sits between weighted avg and ### v10 Model
out = []
i = 0
while i < len(lines):
    line = lines[i]
    # Check if this is the misplaced Block 2.10 header (the one followed by ### v10 Model)
    if (
        line.startswith("## Block 2.10")
        and i + 1 < len(lines)
        and lines[i + 1].startswith("### v10 Model")
    ):
        # Check that two lines before is the weighted avg line
        if len(out) >= 2 and "weighted avg" in out[-2]:
            print(f"Deleting misplaced header at line {i + 1}: {line.rstrip()!r}")
            i += 1  # skip this line, keep the rest
            continue
    out.append(line)
    i += 1

with open(path, "w", encoding="utf-8") as f:
    f.writelines(out)
print("Done.")
