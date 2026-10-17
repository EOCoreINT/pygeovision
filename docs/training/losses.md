# Loss functions

The **loss** is the number the network tries to make small. It measures how wrong the predicted map is. Choosing it
decides what "wrong" means, and for geospatial data that matters more than for photographs, because the things we map
are often rare: water is 2% of a scene, buildings 5%, burn scars 1%.

## Why the default can fail on rare classes

The default loss is ordinary **cross-entropy**: for every pixel, how far is the predicted probability of the true class
from 1, averaged over all pixels. If 98% of pixels are background, a network that predicts "background" everywhere is
98% right and its loss is already low. The few target pixels hardly move the average, so the model can sit in that
comfortable answer for a long time (or for good).

The signs, in the training log:

- `val_iou` stays near half the background's IoU (about 0.5 for a very rare class);
- predicted maps contain none, or almost none, of your class;
- overall accuracy is very high.

Do not trust accuracy for a rare class: a map of nothing scores 98%. Watch the IoU or F1 **of the class you care
about** (`val_iou_class_1`, [metric names](config.md#metric-names)).

## Using a loss

Pass any loss as `loss_fn`. It is called as `loss_fn(logits, target)`, with `logits` of shape `(batch, classes, H, W)`
and `target` of shape `(batch, H, W)` holding class numbers.

```python
from pygeovision.losses import ComboLoss
from pygeovision.training import GeoTrainer, TrainingConfig

result = GeoTrainer(model, cfg).fit(train, val, loss_fn=ComboLoss())
```

Everything in `pygeovision.losses` ignores pixels labelled **255** by default (`ignore_index=255`), matching the
package-wide "no data" value. The trainer's default, `CrossEntropyLoss(ignore_index=255)`, does the same.

```{admonition} Two loss modules
:class: note
Use `pygeovision.losses` (this page). The older `pygeovision.ai.training.losses.get_loss(name, **kwargs)` ignores the value
`-100`, not 255: with 255-labelled nodata it fails or learns the nodata as a class. If you use it, pass
`ignore_index=255` as a keyword.
```

## Which loss

| Your situation | Start with | Why |
|---|---|---|
| Classes are reasonably balanced (each over ~15% of pixels) | The default cross-entropy | Simple, stable |
| One rare class that you care about | `DiceLoss(classes=[1])`, or `ComboLoss()`, or `FocalLoss(alpha=[…])` | Measure overlap of the rare class directly, or weight its pixels up |
| You want fewer **missed** targets and accept some false alarms (flood, damage, fire) | `TverskyLoss(alpha=0.3, beta=0.7, classes=[1])` | Penalises misses (false negatives) more than false alarms |
| Several rare classes | `FocalLoss(alpha=[w0, w1, w2, …])`, or a `WeightedSumLoss` of focal and Dice | One weight per class |
| Edges matter (buildings, roads, shorelines) | `BoundaryAwareLoss()` | Adds extra penalty on pixels at object borders |
| Large uniform areas dominate and training stalls | `OhemCrossEntropy()` | Learns from the hardest pixels only |
| You report IoU and want to optimise it directly | `LovaszLoss()` | A smooth stand-in for IoU. Often best used for a final stretch of training |
| Labels are noisy | `LabelSmoothingCrossEntropy()` | Stops the model becoming over-confident in wrong labels |
| You want a ready all-round mix | `GeospatialMixedLoss()` | Combo + boundary + hard-pixel terms, weights adjustable |

## A real comparison

To show what the choice does, the same small network was trained on a synthetic scene set where the target covers
**1.8%** of the pixels, with each loss, for 60 epochs on a CPU, and scored on 32 held-out chips by the IoU of the target
class.

| Loss | Target-class IoU | Precision | Recall |
|---|---:|---:|---:|
| cross-entropy (default) | 0.802 | 0.94 | 0.84 |
| cross-entropy with class counts (`ClassBalancedCrossEntropy`) | 0.802 | 0.94 | 0.84 |
| cross-entropy, weights `[1, 20]` | 0.796 | 0.70 | 0.97 |
| `DiceLoss()` (all classes averaged) | **0.000** | n/a | 0.00 |
| `DiceLoss(classes=[1])` | **0.844** | 0.92 | 0.91 |
| `ComboLoss()` | 0.816 | 0.92 | 0.88 |
| `FocalLoss(alpha=[0.25, 0.75])` | 0.830 | 0.89 | 0.93 |
| `TverskyLoss()` (all classes averaged) | **0.000** | n/a | 0.00 |
| `TverskyLoss(classes=[1])` | 0.839 | 0.89 | 0.94 |

How to read it, and what not to conclude:

- **This is a synthetic, easy problem** and a tiny network. The numbers show *behaviour*, not what you will get on
  real imagery. On a harder problem plain cross-entropy usually fares worse.
- **Time matters.** In an otherwise identical 25-epoch run, plain cross-entropy had not yet found the target class
  (IoU 0.00) while focal loss already had (0.78): cross-entropy found it later, with more epochs. A rare class
  often looks "stuck" at first and then moves suddenly.
- **`DiceLoss()` and `TverskyLoss()` over all classes collapsed.** They average the overlap of background and target.
  Background overlap is nearly perfect from the start, so the average is already good and the target class gets little
  push. Restricting them to the class you care about with `classes=[1]` fixed it. This is the most important row.
- **Weights trade precision for recall.** Weighting the target 20× raised recall (0.97) and lowered precision (0.70):
  it finds more targets and also marks more non-targets. Pick the balance that suits your use. A flood warning wants
  recall; an inventory of buildings wants precision.
- `ClassBalancedCrossEntropy` gave the same result as plain cross-entropy here. Its weights come from an "effective
  number of samples" formula and are gentle (about 2:1 for this imbalance). For a strong push use explicit weights
  in `torch.nn.CrossEntropyLoss(weight=...)`.

Reproduce the setup with your own data: train once per candidate loss with `early_stopping_metric` and
`checkpoint_metric` set to the target class's IoU, then compare precision and recall on a held-out area, not just IoU.

## The losses in detail

All take `(logits, target)` and return one number.

### Cross-entropy variants

| Loss | Arguments | Notes |
|---|---|---|
| `torch.nn.CrossEntropyLoss(weight=tensor, ignore_index=255)` | `weight`: one factor per class | The plain tool. Useful weights: inverse frequency (`1 / share`) or its square root for a gentler push |
| `ClassBalancedCrossEntropy` | `class_counts=[n0, n1, …]` (pixels per class), `beta=0.9999` | Weights from the effective number of samples. Without `class_counts` it is plain cross-entropy |
| `LabelSmoothingCrossEntropy` | `smoothing=0.1` | Softens the targets slightly |
| `FocalCrossEntropy` | `gamma=2.0` | Focal loss without per-class weights |

### Overlap losses

| Loss | Arguments | Notes |
|---|---|---|
| `DiceLoss` | `smooth=1.0`, `per_class=True`, `classes=None`, `reduction="mean"` | 1 − Dice overlap. **Pass `classes=[…]` for a rare class.** `smooth` avoids dividing by zero on chips that contain none of the class |
| `TverskyLoss` | `alpha=0.3` (false-alarm weight), `beta=0.7` (miss weight), `classes=None` | Dice with separate penalties. `beta > alpha` favours recall; swap them to favour precision |
| `LovaszLoss` | `per_image=True` | Smooth stand-in for IoU. Classes that never occur in a batch are skipped |
| `ComboLoss` | `dice_weight=0.5`, `ce_weight=0.5` | Dice plus cross-entropy: the usual safe choice. Cross-entropy keeps gradients flowing early; Dice sharpens overlap |

### Weighting hard pixels

| Loss | Arguments | Notes |
|---|---|---|
| `FocalLoss` | `alpha` (one number, or one weight per class), `gamma=2.0` | Down-weights pixels the model already gets right. With a single number `alpha` scales the whole loss and does **not** favour any class; for a rare class give a list: `alpha=[0.25, 0.75]`. The list length must equal the number of classes or an error is raised |
| `OhemCrossEntropy` | `thresh=0.7` (a *loss value*, not a probability), `min_kept=100000` | Keeps pixels whose loss is above `thresh`, and at least `min_kept` of the hardest ones. With small batches `min_kept` exceeds the pixel count, so every labelled pixel with a non-zero loss is used |
| `BoundaryAwareLoss` | `boundary_weight=5.0`, `base_loss="combo"` or `"dice"`, `kernel_size=3` | Base loss plus extra cross-entropy on pixels at class borders. Any other `base_loss` raises an error |

### Combinations

```python
from pygeovision.losses import WeightedSumLoss, FocalLoss, DiceLoss, GeospatialMixedLoss

loss_fn = WeightedSumLoss([(0.5, FocalLoss(alpha=[0.25, 0.75])), (0.5, DiceLoss(classes=[1]))])

loss_fn = GeospatialMixedLoss(weights={"combo": 0.5, "boundary": 0.3, "ohem": 0.2})
```

`GeospatialMixedLoss` accepts the components `combo`, `boundary`, `ohem`, `dice`, `focal`, `tversky`; any other key raises
a `ValueError` (a misspelt component is not silently dropped). The numeric `num_classes` argument is accepted for
compatibility and not used.

### Box losses for detection

`CIoULoss`, `DIoULoss`, `GIoULoss` and `SIoULoss` compare **paired boxes** `(N, 4)` in `x1, y1, x2, y2` form and return
the mean of `1 − IoU-like score` (0 for identical boxes). They are building blocks for a detector of your own.

```python
import torch
from pygeovision.losses import CIoULoss

pred = torch.tensor([[10., 10., 50., 50.], [0., 0., 20., 20.]])
true = torch.tensor([[10., 10., 50., 50.], [10., 10., 30., 30.]])
print(CIoULoss()(pred, true))          # tensor(0.4841)  (the first pair is perfect, the second overlaps by a quarter)
```

The built-in detectors (`faster-rcnn-r50`, `fcos-r50`) compute their own loss inside the model when trained with
`GeoTrainer(task="detection")`, so you do not pass a `loss_fn` there.

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| The model predicts only background | A rare class under cross-entropy, or a Dice/Tversky loss averaged over all classes | `DiceLoss(classes=[1])`, `ComboLoss()` or `FocalLoss(alpha=[…])`; train longer before judging |
| Many false alarms, few misses | Weights or `beta` too strong | Lower the class weight, `beta`, or focal `alpha`; or raise the decision threshold by using probabilities at prediction time |
| Many misses, few false alarms | Too little push on the rare class | Raise the weight or `beta`; try `TverskyLoss(classes=[1])` |
| `loss` is `nan` | A batch whose labels are all 255: cross-entropy (also inside `ComboLoss`) averages over zero pixels. Or a learning rate that is too high | Drop chips whose label is entirely 255; lower `learning_rate` |
| `ValueError … alpha has N per-class weights but the model outputs M classes` | `FocalLoss(alpha=[…])` list has the wrong length | One entry per class |
| `ValueError: … classes=[…] out of range` | A class index ≥ number of classes | Check `num_classes` |
| `ValueError: … unknown component` | A misspelt `GeospatialMixedLoss` key | Use one of the listed names |
| Loss values differ wildly between losses | They are on different scales | Compare **metrics** (`val_iou_class_1`), not loss values, across losses |
| Training metric improves, map is still noisy at edges | Pixel-wise loss ignores shape | `BoundaryAwareLoss`, or clean the prediction afterwards ([Postprocessing](../data/postprocessing.md)) |
