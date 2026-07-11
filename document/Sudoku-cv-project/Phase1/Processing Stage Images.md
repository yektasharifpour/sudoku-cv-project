## Block 1.1 — Load & Grayscale
load a Sudoku photo and convert it to grayscale as the first step of grid extraction.

### Output :
![[Pasted image 20260711214038.png|595]]
## Block 1.2 — Noise removal + edge/threshold detection
reduce noise and highlight the grid lines so contours can be detected in the next block

### Output:
![[b1.2-output.png|601]]
## Block 1.3 — Grid Contour Detection
locate the Sudoku grid's outer border in the thresholded image and extract its 4 corners.

### Output:
![[b1.3-output.png|363]]
## Block 1.4 — Perspective transform
warp the detected grid to a flat top-down square view, and store the transform matrix for later use 

### Output:
![[b1.4-output.png|352]]

## Block 1.5 — Cell Splitting
divide the warped 900x900 grid into 81 equal cells (9x9).

### Output:
![[b1.5-output.png|371]]
Total cells extracted: 81
## Block 1.6 — Clean Cells & Detect Empty Cells
### Output:
![[b1.6-output 2.png|427]]
`[[False False False False False False False False False]`
`[False False False False False False False False False]` 
`[False False False False False False False False False]`
`[False False False False False False False False False]`
`[False False False False False False False False False]`
`[False False False False False False False False False]`
`[False False False False False False False False False]`
`[False False False False False False False False False]`
`[False False False False False False False False False]]`

Problem:
	All cells are being flagged as "has digit" — that's not right for a real puzzle (should be a mix of empty and filled).Add one more temporary cell to inspect the actual white-pixel ratios instead of just the True/False flags.
output:

> [!That's actually a very clean bimodal split — values cluster either **low (~0.05–0.18)** or **high (~0.8+)**, with a clear gap between them. So the _detection_ logic works, but our threshold (0.02) was set way too low, and the high values (~0.8) look suspiciously large for a digit (a digit normally covers maybe 15–30% of a cell, not 80%+) — that suggests leftover grid-line border is still baked into the "high" cells even after our 12% margin crop.]
> 		`[[0.089 0.084 0.829 0.819 0.069 0.817 0.152 0.084 0.114]` 
> 		`[0.103 0.834 0.059 0.815 0.821 0.829 0.106 0.83 0.821]` 
> 		`[0.833 0.103 0.112 0.829 0.818 0.161 0.104 0.837 0.817]` 
> 		`[0.114 0.81 0.835 0.175 0.103 0.845 0.177 0.159 0.084]` 
> 		`[0.087 0.841 0.162 0.113 0.161 0.085 0.151 0.153 0.051]` 
> 		`[0.07 0.817 0.841 0.832 0.084 0.826 0.174 0.106 0.81 ]` 
> 		`[0.176 0.85 0.104 0.157 0.848 0.113 0.158 0.829 0.845]` 
> 		`[0.831 0.09 0.81 0.81 0.831 0.821 0.06 0.105 0.089]` 
> 		`[0.106 0.085 0.848 0.823 0.092 0.826 0.115 0.84 0.07 ]]`



### Step 1 to solve:
Increase margin_ratio to strip more border:
	 margin_ratio: 0.12 -> 0.18
Fix the empty-cell threshold to sit between the two clusters
	white_pixel_threshold: 0.02 -> 0.3
	output:
		![[b1.6.r1.s2 4.png|405]]
		problem:
			most off them are wrong
			

### Step 2 to solve:
Sort the ratios and look at them in order
	output:
		0 0.051
		1 0.059
		2 0.06
		3 0.069
		4 0.07
		5 0.07
		6 0.084
		7 0.084
		8 0.084
		9 0.084
		10 0.085
		11 0.085
		12 0.087
		13 0.089
		14 0.089
		15 0.09
		16 0.092
		17 0.103
		18 0.103
		19 0.103
		20 0.104
		21 0.104
		22 0.105
		23 0.106
		24 0.106
		25 0.106
		26 0.112
		27 0.113
		28 0.113
		29 0.114
		30 0.114
		31 0.115
		32 0.151
		33 0.152
		34 0.153
		35 0.157
		36 0.158
		37 0.159
		38 0.161
		39 0.161
		40 0.162
		41 0.174
		42 0.175
		43 0.176
		44 0.177
		45 0.81
		46 0.81
		47 0.81
		48 0.81
		49 0.815
		50 0.817
		51 0.817
		52 0.817
		53 0.818
		54 0.819
		55 0.821
		56 0.821
		57 0.821
		58 0.823
		59 0.826
		60 0.826
		61 0.829
		62 0.829
		63 0.829
		64 0.829
		65 0.83
		66 0.831
		67 0.831
		68 0.832
		69 0.833
		70 0.834
		71 0.835
		72 0.837
		73 0.84
		74 0.841
		75 0.841
		76 0.845
		77 0.845
		78 0.848
		79 0.848
		80 0.85
	Problem
		That gap is actually very clean and large — everything splits into two tight clusters: **0.05–0.177** (45 cells) and **0.81–0.85** (36 cells), with a big empty gap between 0.177 and 0.81. So a threshold around **0.5** would cleanly separate those two clusters.
		But here's the catch: 45 low-cluster cells ≠ your actual 49 empty cells. That's a **4-cell discrepancy** — meaning 4 truly-empty cells are landing in the _high_ cluster (ratio ~0.81+), not the low one. A ratio of 0.81+ is way too high to be a normal digit anyway (digits shouldn't cover 80%+ of a cell) — this smells like specific cells still have leftover grid-border pixels that our uniform margin crop isn't fully removing.
	### Step 3 to solve:
	find exactly which cells are the problem before changing any thresholds further.
	output:
		`row 0: 0.09 0.08 0.83 0.82 0.07 0.82 0.15 0.08 0.11`
		`row 1: 0.10 0.83 0.06 0.82 0.82 0.83 0.11 0.83 0.82` 
		`row 2: 0.83 0.10 0.11 0.83 0.82 0.16 0.10 0.84 0.82` 
		`row 3: 0.11 0.81 0.84 0.18 0.10 0.84 0.18 0.16 0.08` 
		`row 4: 0.09 0.84 0.16 0.11 0.16 0.08 0.15 0.15 0.05`
		`row 5: 0.07 0.82 0.84 0.83 0.08 0.83 0.17 0.11 0.81`
		`row 6: 0.18 0.85 0.10 0.16 0.85 0.11 0.16 0.83 0.84` 
		`row 7: 0.83 0.09 0.81 0.81 0.83 0.82 0.06 0.11 0.09` 
		`row 8: 0.11 0.09 0.85 0.82 0.09 0.83 0.11 0.84 0.07`
		problem:
			Cell 0 has digit "5" → ratio **0.09** (low). Cell 2 is empty → ratio **0.83** (high). That's **backwards** from our assumption — digits are getting _low_ ratios and empty cells are getting _high_ ratios, consistently, across every row you posted.**Root cause:** `cv2.threshold(..., THRESH_OTSU)` picks a different cutoff _per cell_, independently. For a truly empty cell, the pixel values are almost uniform (just background, maybe faint shading) — there's no real bimodal split for Otsu to find, so it picks an unstable, near-arbitrary threshold, and ends up classifying most of that near-uniform noise as "foreground" after inversion. For digit cells, Otsu actually has a real bimodal split (ink vs. background) and behaves correctly, isolating just the thin digit — which is a _smaller_ area than the noise-inflated "empty" cells. So per-cell Otsu is unreliable specifically on blank cells. This isn't a threshold-tuning problem — it's a wrong-method problem.
	## Step 4 to solve:
	reuse the reliable global threshold from Block 1.2 instead of recomputing Otsu per cell.
	1. Go back to Block 1.4 and add `warped_thresh = cv2.warpPerspective(thresh, transform_matrix, (side, side))` right after `warped = cv2.warpPerspective(gray, transform_matrix, (side, side))`
	2. Go back to Block 1.5 and add a second loop, splitting `warped_thresh` the same way as `warped`
	3. Replace your Block 1.6 `is_empty` logic to use `cells_thresh` directly (already binary, no Otsu needed)
	4. result:
		 `[[False False True True False True True False False]`
		 `[False True False True True True False True True]`
		 `[ True False False True True True False True True]`
		 `[False True True True False True True True False]`
		 `[False True True False True False True True False]`
		 `[False True True True False True True False True]` 
		 `[ True True False True True False True True True]`
		 `[ True False True True True True False False False]`
		 `[False False True True False True False True False]]`
		 Empty cells detected: 49
	5. verify it's not just the right _count_ by coincidence, but the right _cells_.
		 `row 0: D D E E D E E D D` 
		 `row 1: D E D E E E D E E` 
		 `row 2: E D D E E E D E E` 
		 `row 3: D E E E D E E E D` 
		 `row 4: D E E D E D E E D` 
		 `row 5: D E E E D E E D E` 
		 `row 6: E E D E E D E E E` 
		 `row 7: E D E E E E D D D` 
		 `row 8: D D E E D E D E D`
##### All 9 rows match your actual puzzle exactly — Phase 1's core logic (Blocks 1.1–1.6) is now correct on this test image.
output:
	![[b.1.6.output2 4.png|395]]

## Testing the result on real data set
1. Clone the real-photo dataset into Colab
	`!git clone https://github.com/wichtounet/sudoku_dataset.git`
2. Update Block 1.1's image path
		image_path = "sudoku_dataset/images/image32.jpg"
			outpout:
				![[b1.1ds-output.png|553]]
				![[b1.2ds-output.png]]
				![[b1.3ds-output.png|350]]
				this result is not true. we have to fix it:
					Increase the epsilon tolerance
						output:
							![[1.3.dsr2-output.png|355]]
				![[b1.4ds-output.png|345]]
				![[b1.5ds-output.png|368]]
				![[b1.6ds-output.png|401]]
3. Re-run Blocks 1.1 through 1.6 in order
4. Compare Block 1.3's detected corners against the ground truth
5. After Block 1.3 runs and prints `Corner points: ...`, compare it to the CSV's row for `image32.jpg`
## Block 1.7—Save checkpoint
persist extracted cells, empty/digit flags, and transform matrices for the test set, so Phase 2 doesn't require re-running Phase 1 from scratch after a
### Output
Saved checkpoint to: checkpoints/phase1/image32.npz

## Block 1.8—Batch Testing Function
Blocks 1.1–1.7 into a single reusable function and run it across multiple test images in a loop
### output:
Passed: 17 / 20 
Failed: 3 - datasets/sudoku_dataset/images/image76.jpg → could not find 4 corners 
- datasets/sudoku_dataset/images/image74.jpg → could not find 4 corners 
- datasets/sudoku_dataset/images/image196.jpg → could not find 4 corners
==7/20 passing with 3 documented failures — that's a solid testing-point result, and those 3 failures are exactly the failure-case analysis data the spec wants.==
#### Step1: Visualize why the 3 failures happened
##### output:
![[b1.8r1s1-output.png]]
##### The reason:
- **image76.jpg** — the grid extends almost to the photo's edges with barely any margin/background around it. `findContours` with `RETR_EXTERNAL` likely picked up the _photo frame itself_ as the largest contour instead of the grid border, since there's little separation between them.
- **image74.jpg** — there's a red text/graphic block on the right side of the page, outside the grid. If that block's contour area rivals or exceeds the grid's, `max(contours, key=cv2.contourArea)` could grab the wrong region, or it's splitting attention from the true grid border.
- **image196.jpg** — same issue as image74 (colored pie-chart graphic and extra page content on the right), **plus** a visible shadow diagonal across the page, which can break the grid's contour into disconnected pieces under thresholding.

## Push the results into GitHub
### Step1:
Clone  actual GitHub repo into Colab (not just the dataset)
### Step2:
go to branch phase1/feature/grid-extraction
### Step 3:
copy data set to datasets/sudoku_dataset
### Step 4:
push changes