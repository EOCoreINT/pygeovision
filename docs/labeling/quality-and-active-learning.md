# Quality Assessment & Active Learning

Two real tools for improving label quality and prioritizing what to
label next — neither is a labeler itself.

## `LabelQualityAssessor`

```python
from pygeovision.labeling.quality import LabelQualityAssessor

result = LabelQualityAssessor().assess(
    label_path="./labels/buildings.tif",
    image_path="./scene.tif",   # optional, enables image-aware checks
    checks=["class_balance", "coverage", "slivers"],
)
```

Real, substantive scoring, not placeholder metrics — hand-verified
formulas:

- **`class_balance`** — a real imbalance-ratio-based score (how skewed
  the class distribution is).
- **`coverage`** — a real valid-pixel-fraction check (how much of the
  raster is real data vs. nodata/background).
- **`slivers`** — real connected-component analysis
  (`scipy.ndimage`) flagging suspiciously small, likely-spurious
  labeled regions.

```bash
pygeovision label quality ./labels/buildings.tif --html
```

Real output includes a letter `quality_grade`, a `quality_score`
(0–1), per-check scores, and real, actionable `recommendations`
generated from whichever checks scored poorly. `--html` saves a real,
shareable HTML report alongside the raster.

## `ActiveLearner`

Real, published active-learning strategies for deciding which
unlabeled samples are most worth a human's time to label next —
useful when you have far more unlabeled imagery than labeling budget.

```python
from pygeovision.labeling.active import ActiveLearner

learner = ActiveLearner(strategy="entropy", budget=100)
selected = learner.select(model, unlabeled_pool, n_select=50)
```

**Real, valid `strategy` values:** `entropy`, `least_confidence`,
`margin` (three real, classic uncertainty-sampling formulas — each
scores samples differently based on the model's own real predicted
class probabilities), `coreset` (real feature-space diversity
selection — picks samples that are maximally different from what's
already labeled, not just individually uncertain), `committee` (real
Monte Carlo dropout — enables dropout at inference time and measures
real prediction disagreement across multiple real stochastic forward
passes as an uncertainty signal), `random` (a real baseline for
comparison).

```{note}
`select()` requires a real, already-trained PyTorch model in eval
mode — this tool prioritizes among your *unlabeled* data using a
model you already have, it doesn't replace the need to train one
first from an initial labeled set.
```
