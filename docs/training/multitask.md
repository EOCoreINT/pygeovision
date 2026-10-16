# Multi-task learning

**Package:** `pygeovision.advanced.multitask` · **Class:** `MultiTaskLearner` · **Needs:** `torch`, `torchvision`

## Why

When two outputs rest on the same visual evidence (a land-cover map and a "what kind of scene is this" label; a building
mask and a building count), training one network for both can help: the shared encoder is pushed to learn features useful
to each task, and you run one network at inference instead of two. It can also hurt: if the tasks pull the encoder in
different directions one of them gets worse (*negative transfer*). `MultiTaskLearner` makes the experiment cheap: one
encoder, one small head per task, a weighted loss, and per-task loss curves so you can see which task is paying.

```{admonition} What it is, exactly
:class: note
A **torchvision backbone** (ResNet-50, ResNet-101 or VGG-16, 32× down-sampling) feeding **light heads**. The segmentation
head is two transposed convolutions and a 1×1 convolution, which is far simpler than the decoders in
[`pygeovision.models`](../models/tasks.md) (U-Net, DeepLab, SegFormer …). Expect a coarser, less accurate map than a
dedicated segmentation model. Use this to *test whether multi-task training helps*, or when a light shared network is
enough; if segmentation quality is the goal, train a dedicated model.
```

## How to use

```{literalinclude} ../_examples/advanced/multitask_demo.py
:language: python
:caption: multitask_demo.py
:linenos:
```

{download}`Download multitask_demo.py <../_examples/advanced/multitask_demo.py>`. It builds 4-band chips (a bright square of one of three
sizes on noise), trains a segmentation head (where is the square?) and a classification head (how big is it?) together for
8 epochs on a CPU, then predicts on held-out chips. It takes about 35 seconds on 2 CPU cores.

### Supported tasks and what each head is

| Task | Head | Target in the batch | Loss |
|---|---|---|---|
| `segmentation` | 2 × `ConvTranspose2d` + 1×1 conv, bilinearly resized to the input size | `(B, H, W)` long; **255 = ignore** | Dice + cross-entropy (`ComboLoss`) |
| `classification` | global average pool → 2-layer perceptron | `(B,)` long | Label-smoothed cross-entropy |
| `change_detection` | same as classification, applied to **one** tensor. Stack the two dates along channels (`in_channels = 2 × bands`); there is no siamese branch | `(B,)` long | Cross-entropy |
| `detection` | pooled features → perceptron with `classes × 5` outputs (x, y, w, h, confidence) | `(B, classes × 5)` float | Smooth L1 |

`detection` is a toy: one box per class per image. For real object detection use the detectors in
[`pygeovision.models`](../detection/index.md#object-detection).

### Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `backbone` | `"resnet50"` | `"resnet50"`, `"resnet101"`, `"vgg16"` (anything else raises `ValueError`) |
| `tasks` | `["segmentation", "classification"]` | Which heads to build |
| `n_classes` | `{}` | `{task: number of classes}`. Defaults: classification 4, others 2 |
| `task_weights` | equal, summing to 1 | `{task: weight}` in the total loss. Raise the weight of a task that is lagging |
| `pretrained` | **`True`** | Download ImageNet weights for the backbone (needs a network). `False` trains from scratch (what the example uses) |
| `in_channels` | 3 | Input bands. For 4 bands the first-layer filters for bands 1–3 keep their pretrained values and extra bands start at the **mean** of the three |
| `device` | auto | |

### Batches

`train()` and `predict()` work on dictionaries. A `DataLoader` item must be:

```python
{"inputs": tensor(C, H, W) float,          # C == in_channels
 "segmentation": tensor(H, W) long,        # one key per task being trained
 "classification": tensor(()) long}
```

Anything else raises a `ValueError` that lists the keys it found. A task that has an output but no target in a batch is
skipped for that batch; if *no* task matches, `compute_loss` raises instead of training on a constant.

### Methods

| Method | Purpose |
|---|---|
| `build()` | Create (or rebuild, resetting weights) the model and return the `nn.Module`. `train()` calls it if needed |
| `train(train_loader, val_loader, epochs=100, lr=1e-4, save_path=None, log_every=10)` | AdamW (weight decay 1e-4), cosine schedule, gradient clipping at norm 1. `val_loader` may be `None` (then the training loss picks the "best" epoch) |
| `compute_loss(outputs, targets)` | `(total_loss, {task: float})`; the weighted sum |
| `load_weights(path)` | Load the state dict written by `train(save_path=...)` |
| `predict(inputs, tasks=None)` | NumPy results without gradients. `tasks=[...]` runs only those heads |

## Expected output

```text
model parameters: 32.7 M
trained in 31 s
keys: ['best_epoch', 'best_val_loss', 'epochs_trained', 'history', 'success']
best_epoch: 8 best_val_loss: 0.425941
epoch 1: train 1.096  val 0.893  per task (val) {'segmentation': 0.649, 'classification': 1.136}
epoch 8: train 0.494  val 0.426  per task (val) {'segmentation': 0.422, 'classification': 0.43}
{'segmentation': (16, 64, 64), 'classification': (16,)}
held-out foreground IoU: 0.45   classification accuracy: 0.94
only one task: {'classification': (2,)}
```

What each line tells you:

* **`train()` returns** `success`, `best_val_loss`, `best_epoch`, `epochs_trained` and `history` with `train_loss`,
  `val_loss` (`None` entries if there is no validation loader), `train_task_losses` and `val_task_losses` (one dictionary per
  epoch). **Read the per-task curves**: here both fell (segmentation 0.65 → 0.42, classification 1.14 → 0.43), i.e. no
  task was sacrificed. If one rises while the other falls, the tasks are competing: change `task_weights` or train separate
  models.
* **Held-out numbers** come from `predict` plus your own metric. Classification reached 0.94 and foreground IoU 0.45 after 8
  epochs from random weights on 48 images. The coarse head explains the modest IoU; this is a CPU demonstration, not a
  benchmark.
* **`predict` shapes:** segmentation → class map `(B, H, W)` at the **input size**; classification / change detection →
  class ids `(B,)`; detection → raw `(B, classes × 5)` values. A `(C, H, W)` array gets a batch axis added.
* **Saved weights:** `save_path` receives the weights of the *best* epoch. The model object in memory holds the **last**
  epoch. Call `load_weights(save_path)` to get the best ones back (the example does).

## What to do when…

| Situation | What to do |
|---|---|
| `ValueError: backbone must be one of ['resnet50', 'resnet101', 'vgg16']` | Only those are built in |
| `ValueError: unknown task(s) [...]` | Use `segmentation`, `classification`, `change_detection` or `detection` |
| `ValueError: each batch must be a dict with an 'inputs' tensor …` | Your dataset returns tuples. Wrap it so `__getitem__` returns `{"inputs": x, "segmentation": mask, ...}` |
| `ValueError: no task has both an output and a target` | A key in your batch does not match a task name (typos such as `"seg"`) |
| It tries to download weights and fails | `pretrained=True` needs a network. Use `pretrained=False`, or put torchvision's ResNet weights in the torch cache first |
| `RuntimeError: ... expected input to have 3 channels` | `in_channels` does not match your data. Set it to the number of bands (or stacked bands) |
| One task improves while another gets worse | Negative transfer. Change `task_weights`, lower the learning rate, or train separate models and compare with the [benchmark tools](../experiments/benchmark.md) |
| Segmentation looks blocky | The head is deliberately light. Try a dedicated model from [`pygeovision.models`](../models/tasks.md), or use this network's *features* only |
| Out-of-memory | Smaller batch or chip size; `resnet50` instead of `resnet101`; the segmentation head at 2048 channels is the heaviest layer |
| You need class imbalance handling in the segmentation loss | `compute_loss` uses fixed `ComboLoss()` and cannot be configured. Subclass and override `compute_loss`, or use [`GeoTrainer`](index.md) with a single task |
| You need the best validation epoch in memory | Call `load_weights(save_path)` after training |

## Limits

* One shared learning rate and schedule for all heads; no per-task gradient balancing beyond `task_weights`.
* No mixed precision, multi-GPU or experiment tracking: for those use [`GeoTrainer`](fine-tuning.md) on a single task.
* `change_detection` here is classification of a stacked tensor ("did anything change in this chip?"), not a change **map**;
  use a change-detection model for maps.
