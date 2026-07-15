## Training Curves

> [!info] Deliverable: Phase 2 — Digit Recognition
> Loss and accuracy curves over training epochs. Visualizations are in [[Training progress#Block 2.8 — Training Curves|Block 2.8 — Training Curves]].

---

### v1 Model (Colab — MNIST + Hoda + synthetic empty cells)

Trained on CPU in Google Colab using the initial 2-conv-layer CNN architecture.

![[b2.8-output.png|697]]

### v10 Model (Final — domain-matched training pipeline)

Trained locally on CUDA with the improved 3-conv-layer + BatchNorm architecture (`DigitClassifierV2`). Uses domain-matched augmentation: MNIST digits re-photographed through the real threshold+crop pipeline, printed font digits, and calibrated synthetic empty cells.

**Training log (17/30 epochs — early stopping triggered):**

| Epoch | Train Accuracy | Validation Accuracy |
|-------|---------------|-------------------|
|  1    | 0.8652        | 0.9233            |
|  2    | 0.9322        | 0.9335            |
|  3    | 0.9426        | 0.9345            |
|  4    | 0.9480        | 0.9412            |
|  5    | 0.9513        | 0.9478            |
|  6    | 0.9543        | 0.9413            |
|  7    | 0.9568        | 0.9497            |
|  8    | 0.9578        | 0.9499            |
|  9    | 0.9594        | 0.9452            |
| 10    | 0.9600        | 0.9487            |
| 11    | 0.9612        | 0.9516            |
| 12    | 0.9619        | 0.9536            |
| 13    | 0.9634        | 0.9504            |
| 14    | 0.9636        | 0.9504            |
| 15    | 0.9642        | 0.9491            |
| 16    | 0.9651        | 0.9468            |
| 17    | 0.9652        | 0.9501            |

> [!note] Early Stopping
> Training stopped at epoch 17 because validation accuracy plateaued — no improvement for several consecutive epochs. The model converged to ~96.5% train accuracy and ~95% validation accuracy on the augmented dataset.


