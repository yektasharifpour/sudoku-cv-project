## Block 2.1 — Load Digit Datasets (MNIST + Hoda)
 load English (MNIST) and Persian (Hoda) digit datasets as the foundation for a combined 10-class digit classifier (1-9 + empty).
### Output(MNIST):
![[b2.1-output.png]]
`MNIST train size: 60000` 
`MNIST test size: 10000`

### Output(Hoda):
![[b2.1aHoda-output.png]]
`Hoda train shape: (60000, 28, 28, 1) (60000,)` 
`Hoda test shape: (20000, 28, 28, 1) (20000,)`

## Block 2.2 — Merge Datasets
combine MNIST (English) and Hoda (Persian) digit images into a single unified dataset for training one 10-class classifier that recognizes both scripts.

### Output:
![[b2.2-output.png]]
`Combined train: (120000, 28, 28) (120000,)`
`Combined test: (30000, 28, 28) (30000,)`

## Block 2.3 — Empty Cell Class (Real + Synthetic)
Purpose: build the 10th class (label 9... or 10, see note) representing empty cells, combining real examples from Phase 1 checkpoints with synthetic generated blanks.

### Output(1):
`Found 17 Phase 1 checkpoints`
`Real empty cells collected: (833, 28, 28)`

### Output(2):
`Synthetic empty cells generated: (11000, 28, 28)`
![[b2.3-output(2).png]]

the synthetic ones are noticeably grainier than reality. Two things are going on:

1. **Display scaling issue:** `plt.imshow` without `vmin`/`vmax` auto-stretches each image's own min/max to fill the full black-white range. Real cells are almost perfectly uniform (tiny variance), so they display as flat black. Synthetic cells have subtle noise (std=0.03) that gets stretched to look much noisier than it actually is.
2. **Real noise magnitude issue:** even accounting for that, actual real cells are far more uniform than what we're generating — our noise level is genuinely too strong.
#### Fix: reduce noise and fix the display scaling
##### Output:
![[b2.3-output(3).png]]
both rows show mostly flat dark backgrounds with subtle noise, and the synthetic edge-artifact (visible as a gray line on the right in 2 of the 5 samples) closely resembles the small artifacts you see in the real ones (the fragment in Real image 1, the dot in Real image 3/4/5). This looks realistic enough to train on.

## Block 2.4 — Final Dataset Assembly
Purpose: combine MNIST digits 1-9, Hoda digits 1-9, real empty cells, and synthetic empty cells into one final labeled dataset (label 0 = empty, labels 1-9 = digits).

### Output:
`Final train: (117543, 28, 28) (117543,)`
`Final test: (29387, 28, 28) (29387,)`

`Class distribution (train):`
  `Class 0: 9466`
  `Class 1: 12742`
  `Class 2: 11958`
  `Class 3: 12131`
  `Class 4: 11842`
  `Class 5: 11421`
  `Class 6: 11918`
  `Class 7: 12265`
  `Class 8: 11851`
  `Class 9: 11949`
## Block 2.5 — Model Architecture
Purpose: define a CNN for 10-class digit classification (0=empty, 1-9=digits).
### Output:
`DigitClassifier(`
  `(conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`
  `(conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`
  `(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)`
  `(fc1): Linear(in_features=1568, out_features=128, bias=True)`
  `(dropout): Dropout(p=0.3, inplace=False)`
  `(fc2): Linear(in_features=128, out_features=10, bias=True)`
`)`

`Using device: cpu`


> [!NOTE] Device: CPU
> because of the colab limitation in using GPU and because the digit classifier here is lightweight (small CNN on ~28×28 grayscale cells) — it doesn't strictly need a GPU. Training will just be slower on CPU, but for a 10-class digit model on a modest dataset, CPU training is very feasible in Colab's free tier. So we can proceed without GPU for now; if training time becomes painful later, we can revisit (e.g. use MobileNet at reduced resolution, or request GPU only for the final full training run).
>

## Data Loaders and Augmentation
Purpose: wrap the final dataset into PyTorch DataLoaders with augmentation (rotation, shift) for more robust training, per spec requirement.

### Output:
`Batch images shape: torch.Size([64, 1, 28, 28])`
`Batch labels shape: torch.Size([64])`

## Block 2.7 — Training Loop
Purpose: train the CNN on the combined dataset using CrossEntropy Loss, tracking accuracy per epoch (per spec requirement).

### Output:
![[b2.7-output.png|490]]

## Block 2.8 — Training Curves
Purpose: visualize loss and accuracy over epochs, per spec deliverable requirement.
### Output:
![[b2.8-output.png|697]]
## Block 2.9 — Confusion Matrix & Error Analysis
### Output:
Purpose: evaluate per-class performance, especially checking if the "empty" class (0) is confused with digits, per spec deliverable requirement.
![[b2.9-output.png]]
			  `precision    recall   f1-score  support`

           0       1.00      1.00      1.00      2367
           1       1.00      1.00      1.00      3135
           2       0.99      0.99      0.99      3032
           3       0.99      0.98      0.99      3010
           4       0.98      0.99      0.99      2982
           5       0.99      0.99      0.99      2892
           6       0.99      0.99      0.99      2958
           7       1.00      0.99      0.99      3028
           8       0.99      1.00      1.00      2974
           9       0.99      0.99      0.99      3009

    accuracy                          0.99     29387
   `macro avg       0.99      0.99       0.99     29387`
`weighted avg       0.99      0.99      0.99     29387`

## Block 2.10 — Misclassification Examples
Purpose: visually inspect actual misclassified samples for the error-analysis deliverable.

### Output:
![[b2.10-output.png]]
`Total misclassified: 240 out of 29387`

> [!NOTE] error-analysis
> Errors are concentrated in visually ambiguous stroke-shape pairs (3/5, 4/6, 4/9, 1/7), consistent with genuine handwriting ambiguity rather than model deficiency. No systematic failure mode was found for the empty-cell class, which achieved perfect precision/recall.

## Block 2.11 — Final model checkpoint
### Issue: 
`torch.save()` failed with `RuntimeError: Parent directory models does not exist`, initially suspected to be a Colab runtime disconnect (which would normally lose all trained progress in memory).
### Diagnosis: 
Checked whether training state survived by running `print(history["test_acc"][-1])`. It returned 0.9918... successfully, confirming the session was still active — no data was lost.
### Root cause: 
The models/ folder was empty when the repo skeleton was created (Phase 0), so git never tracked it. Cloning the repo fresh for Phase 2 didn't recreate it.
### Fix: 
`os.makedirs("models", exist_ok=True)` before re-saving — no retraining required.


> [!NOTE] Why this was low-risk either way
> even if the runtime had actually disconnected, the per-epoch checkpointing built into Block 2.7 (checkpoints/phase2/digit_classifier_latest.pt, saved after every epoch) meant training could have resumed from the last completed epoch instead of restarting from scratch — exactly per the project's Checkpoint Policy. This incident confirmed that safeguard was working as intended, even though it wasn't ultimately needed this time.

### Outputs:
`0.9918331234899785`

`Final model saved to models/digit_classifier_final.pt`

## Push changes to GitHub
Enumerating objects: 47, done.
Counting objects: 100% (47/47), done.
Delta compression using up to 2 threads
Compressing objects: 100% (41/41), done.
Writing objects: 100% (43/43), 30.78 MiB | 5.41 MiB/s, done.
Total 43 (delta 6), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (6/6), completed with 1 local object.
To https://github.com/yektasharifpour/sudoku-cv-project.git
   f016082..c710d74  phase2/feature/digit-recognition -> phase2/feature/digit-recognition