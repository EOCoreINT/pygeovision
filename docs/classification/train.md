# Train a classifier

`GeoTrainer` trains **segmentation and detection** models. A scene classifier is simpler than either: the data is a stack
of chips and one label per chip, so the training loop is a few lines of PyTorch around a model from `get_model`. This
page shows that loop, verified on synthetic data, and what to add to it for real work.

```{admonition} Why not GeoTrainer?
:class: note
`TrainingConfig(task=...)` accepts `segmentation` or `detection`. Its loss, metrics and checkpoints are built for per-pixel
masks and boxes. For classification, a short loop gives you full control of the loss, the sampling and the metric. If you
want one network that classifies *and* segments, use [multi-task learning](../training/multitask.md).
```

## Before you start

- [ ] A folder of chips, each a GeoTIFF or array with your bands, and **one class per chip** (a CSV, or one folder per class).
- [ ] A **validation set from different scenes** than the training set. Chips cut from the same scene are neighbours; splitting them randomly leaks information.
- [ ] A fixed list of class names in a fixed order: the label is the position in that list.
- [ ] The same pixel scaling for training and prediction.

## The loop

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pygeovision.models import get_model


class Chips(Dataset):
    """(image, label) pairs. Replace with chips cut from your own scenes:
    image float32 (bands, H, W) scaled the way you will scale at prediction time, label an int."""
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.x, self.y = images, labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.x[i]), int(self.y[i])


train_loader = DataLoader(Chips(train_images, train_labels), batch_size=16, shuffle=True)
val_loader = DataLoader(Chips(val_images, val_labels), batch_size=16)

model = get_model("convnext-t", num_classes=3, in_channels=4, pretrained=False)   # pretrained=True with real data
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

best = 0.0
for epoch in range(1, 9):
    model.train()
    for x, y in train_loader:
        optimiser.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimiser.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
    accuracy = correct / len(val_loader.dataset)
    print(f"epoch {epoch}  loss {loss.item():.3f}  val_accuracy {accuracy:.3f}")
    if accuracy > best:                                   # keep the best epoch, as GeoTrainer does
        best = accuracy
        torch.save({"model_state_dict": model.state_dict(), "classes": ["a", "b", "c"]}, "best.pth")
```

Output of a real run on **synthetic** chips (192 training and 48 validation chips of 64 × 64 pixels, three classes that
differ in brightness, `convnext-t` from scratch on a CPU). It shows the shape of a log; accuracy on your imagery will
differ, and a model trained from scratch on so few chips is erratic in the first epochs, as here:

```text
epoch 1  loss 1.267  val_accuracy 0.312
epoch 2  loss 1.279  val_accuracy 0.396
epoch 3  loss 1.232  val_accuracy 0.312
epoch 4  loss 1.334  val_accuracy 0.312
epoch 5  loss 1.054  val_accuracy 0.396
epoch 6  loss 0.974  val_accuracy 0.708
epoch 7  loss 0.413  val_accuracy 0.708
epoch 8  loss 0.357  val_accuracy 0.688
```

Chance for three balanced classes is 0.333, so the first epochs were no better than guessing; the model began to learn in
epoch 6. With real data use `pretrained=True`, which starts far above chance.

## What to add for real work

| Need | How |
|---|---|
| Pretrained weights | `get_model(..., pretrained=True)`. With more than 3 bands the extra bands start untrained, so allow more epochs |
| Train only the new head first | `for p in model.parameters(): p.requires_grad = False`, then unfreeze the last layers; or use a frozen DINOv2 ([few-shot](few-shot.md)) |
| A learning-rate schedule | `torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)`; call `scheduler.step()` once per epoch |
| Class imbalance | `CrossEntropyLoss(weight=...)` with weights inversely proportional to class frequency, or a `WeightedRandomSampler` |
| Augmentation | Flips and 90° rotations are safe for overhead imagery; apply the same to the image only (there is no mask) |
| Mixed precision on a GPU | `torch.autocast("cuda")` around the forward pass and a `torch.cuda.amp.GradScaler` |
| Experiment tracking | [Experiment tracking](../experiments/tracking.md) (`ExperimentTracker`) |

## Evaluate it

Overall accuracy hides class mix-ups. Build a confusion matrix on a held-out set:

```python
import torch

model.load_state_dict(torch.load("best.pth", weights_only=True)["model_state_dict"]); model.eval()
n = 3
confusion = torch.zeros(n, n, dtype=torch.long)           # rows: reference class, columns: predicted class
with torch.no_grad():
    for x, y in val_loader:
        for ref, pred in zip(y, model(x).argmax(dim=1)):
            confusion[ref, pred] += 1
recall = confusion.diag() / confusion.sum(dim=1).clamp(min=1)       # share of each class that was found
precision = confusion.diag() / confusion.sum(dim=0).clamp(min=1)    # share of each prediction that was right
```

A class with low **recall** is being missed; a class with low **precision** is being over-predicted. Report both next to
the overall accuracy.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| Validation accuracy stays at chance | Learning rate too high or too low; labels do not match the images; too few chips | Try `1e-4` and `1e-3`; open a few chips with their labels; add chips |
| Accuracy is high on training, low on validation | Over-fitting or leakage | Split by scene; more augmentation; pretrained weights |
| `ValueError` about window size (ViT/Swin) | Fixed chip size | `get_model(..., img_size=<chip size>)` |
| Loss is `nan` | Learning rate too high, or `nan` values in the input | Lower the rate; replace `nan` pixels before training |
| Out of memory | Batch or model too large | Smaller batch; `resnet50` instead of `convnext-l`; mixed precision |

Next: [Predict with a classifier](predict.md).
