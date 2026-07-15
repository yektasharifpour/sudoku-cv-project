## Training Curves

> [!info] Deliverable: Phase 2 — Digit Recognition
> Loss and accuracy curves over training epochs. Visualizations are in [[Training progress#Block 2.8 — Training Curves|Block 2.8 — Training Curves]].

---

### v1 Model (Colab — MNIST + Hoda + synthetic empty cells)

Trained on CPU in Google Colab using the initial 2-conv-layer CNN architecture.

![[b2.8-output.png|697]]

### v10 Model (Final — domain-matched training pipeline)

Trained locally on CUDA with the improved 3-conv-layer + BatchNorm architecture (`DigitClassifierV2`). Uses domain-matched augmentation: MNIST digits re-photographed through the real threshold+crop pipeline, printed font digits, and calibrated synthetic empty cells.

![[v10_training_curve.png]]

> [!note] Early Stopping
> Training stopped at epoch 17 because validation accuracy plateaued — no improvement for several consecutive epochs. The model converged to ~96.5% train accuracy and ~95% validation accuracy on the augmented dataset.


