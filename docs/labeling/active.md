# Active learning: label the right images first

**Class:** `pygeovision.labeling.ActiveLearner` · **Client:** `client.labeling.active_learner(strategy, budget)`

## Why use it

Hand labeling is the expensive part. If you can label 200 images, **which 200** matters a lot. Labeling the easy ones
the model already gets right teaches it nothing. Active learning asks the **current model** which unlabelled images it
is least sure about (or which look most different from what it has seen) and hands you those first.

The loop:

```text
1. Label a small starter set (say 30 images). Train a model.
2. ActiveLearner.select(model, pool)  → the 20 images the model finds most informative
3. A person labels those 20.
4. Retrain with 50 labelled images. Record the score: learner.update(batch, metrics={"val_iou": …})
5. Repeat until the budget is spent or the score stops improving.
```

It is a tool for **choosing**; it does not label, train or retrain anything. You keep those steps.

## Strategies

| Strategy | What it looks at | Good for | Watch out |
|---|---|---|---|
| `entropy` (default) | How spread the predicted class probabilities are, averaged over the image | General use | Favours images with a lot of genuinely ambiguous ground |
| `least_confidence` | 1 − the top class probability | Same idea, simpler | |
| `margin` | The gap between the top two classes | When two classes are often confused (crop A vs crop B) | |
| `coreset` | Picks images that look **different from each other** (4 × 4 pooled probability maps; greedy farthest-first) | Making sure the batch covers variety, not ten near-duplicates | Ignores whether the model is unsure |
| `committee` | Disagreement between `mc_passes` runs with dropout switched on (Monte-Carlo dropout) | A model with dropout layers | Raises `ValueError` if the model has none |
| `random` | Nothing | **The baseline you should compare against** | |

Classification models (`(1, C)` logits) and segmentation models (`(1, C, H, W)`; a single channel is treated as a sigmoid)
work. Detection models do not.

```{admonition} Always compare with random
:class: tip
Active selection does not always win, especially with a tiny starter set or a model that is confident and wrong. Run
the same budget with `strategy="random"` at least once. If random does as well, use random: it is cheaper and unbiased.
```

## How to use

```python
from pygeovision.labeling import ActiveLearner

learner = ActiveLearner(strategy="entropy", budget=200, batch_size=20, n_iterations=10)

pool = [{"path": str(p)} for p in sorted(Path("chips/unlabelled").glob("*.tif"))]

while not learner.finished:
    batch = learner.select(model, pool)                 # most informative first; items are copies with "score"
    paths = [b["path"] for b in batch]
    # ... send `paths` to annotators (e.g. Label Studio), wait for labels ...
    # ... retrain the model with all labels so far (GeoTrainer) ...
    iou = evaluate(model)
    learner.update(batch, metrics={"val_iou": iou})
    pool = [p for p in pool if p["path"] not in paths]  # do not offer them again

learner.plot_learning_curve("active_curve.png", metric="val_iou")
```

### Constructor

| Argument | Default | Meaning |
|---|---|---|
| `strategy` | `"entropy"` | One of the table above. Unknown names raise `ValueError` |
| `budget` | 100 | Total samples you are willing to annotate |
| `n_iterations` | 10 | Rounds you plan |
| `batch_size` | 16 | Samples per round |
| `seed` | 42 | For `random` and the first pick in `coreset` |
| `loader` | `default_loader` | Function that turns a path into a `(1, C, H, W)` float tensor. **See below** |
| `mc_passes` | 10 | Dropout passes for `committee` (≥ 2) |

### `select(model, unlabeled_pool, n_select=None, device=None)`

- **`model`**: a trained `torch.nn.Module`. It is moved to `device` and put in eval mode. Pass `None` for `random`.
- **`unlabeled_pool`**: items that are dicts with a `"path"` key, plain paths, tensors, or dataset samples.
- **`n_select`**: defaults to `batch_size`, capped by what is left of the budget.
- Returns the chosen items, **most informative first**. Dict items come back as copies with a `"score"` added; other items as
  `{"idx": position, "score": s}`.

Real output shape (random weights and random data, so the numbers are meaningless; the structure is what matters):

```python
[('x0', 0.6081), ('x1', 0.5254)]            # (path, score), highest first
learner.update(batch, metrics={"val_iou": 0.5})
learner.remaining_budget   # 4
learner.finished           # False
learner.learning_curve     # [{'n_labeled': 2, 'iteration': 1, 'val_iou': 0.5}]
```

### The `loader` matters

`default_loader` reads a raster, replaces NaN with 0 and **min-max scales each image on its own**. That is a convenient
default and **probably not what your model was trained on**. If you trained on 0–1 reflectance, on percentile-stretched
data, or with per-band normalisation, pass a `loader` that does the same, or the uncertainty scores will not reflect how
the model behaves in real use:

```python
def loader(path):
    x = read_and_normalise_like_training(path)           # your own function
    return torch.from_numpy(x).unsqueeze(0)

ActiveLearner(strategy="entropy", loader=loader)
```

### Bookkeeping

| Member | Meaning |
|---|---|
| `remaining_budget` | `budget` minus samples recorded with `update` |
| `finished` | `True` when the budget is spent or `n_iterations` rounds are recorded |
| `update(batch, metrics=None)` | Counts the batch against the budget. `metrics` are stored for the curve |
| `learning_curve` | `[{"n_labeled", "iteration", <metric>…}]` |
| `plot_learning_curve(save_path=None, metric="val_iou")` | Metric versus labelled samples (needs `matplotlib`; warns if nothing was recorded) |
| `train_iteration(model, config=None, unlabeled_pool=None)` | **Does not train, despite the name.** Selects a batch and records the round. `config` is ignored with a warning |
| `history` | One record per `train_iteration` call |

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `RuntimeError: no pool item could be scored. First error: …` | Every item failed (bad loader, wrong channel count, unreadable file) | Read the quoted first error. Check the loader gives the channels the model expects |
| Warning "N of M pool items could not be scored" | Some files are bad | They are skipped; fix or remove them |
| `ValueError … committee … no dropout layers` | The model has no `nn.Dropout` | Add dropout, or use `entropy` |
| `TypeError … model returned …` | Output is not a logits tensor | Wrap the model so it returns a tensor |
| `ValueError … must return a batch of one` | The model changes the batch size | Fix the model's forward pass |
| Empty list returned and a warning | Budget is already spent | Check `remaining_budget` |
| The same images selected again | They were not removed from the pool | Remove them after annotation |
| Learning curve is flat or noisy | Validation set too small; labels noisy | Use a fixed, hand-checked validation set, never selected samples |
| Selected images are all near-duplicates | Uncertainty strategies do not enforce variety | Use `coreset`, or select a larger batch and pick every k-th |
| Everything gets a score of about the same | Model is untrained or random | Train first; active learning needs a model that has learned something |

```{note}
Selection logic (scores, ordering, budget accounting, device handling, loader use) is tested on small PyTorch models.
Whether active selection beats random on **your** data is an empirical question this package cannot answer for you; the
`random` strategy exists so you can measure it.
```

```{seealso}
[Tile labelers and Label Studio](tile-labelers.md) for the annotation step · [Training](../training/index.md) for the
retraining step
```
