# Hyper-parameter search

**Why.** Learning rate, batch size and weight decay interact, and good values differ between datasets. A search tries
many combinations and reports the best, instead of you guessing.

**How.** Needs `pip install optuna` (in `pygeovision[train]`). Each trial must train a **fresh model with a fresh
trainer**, and return the score to maximise.

```python
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer, TrainingConfig, OptunaHPO

def objective(params):
    cfg = TrainingConfig(output_dir=f"runs/hpo/{params['learning_rate']:.1e}", num_classes=2, in_channels=4,
                         max_epochs=10, num_workers=0, **params)
    model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=True)
    return GeoTrainer(model, cfg).fit(train, val)["best_val_iou"]

best = OptunaHPO(n_trials=20, direction="maximize").run(
    objective_fn=objective,
    search_space={
        "learning_rate": ("float_log", 1e-5, 1e-2),
        "weight_decay":  ("float_log", 1e-6, 1e-2),
        "batch_size":    ("categorical", [4, 8, 16]),
    },
)
print(best)   # {'best_params': {...}, 'best_value': 0.83, 'n_trials': 20, 'study_name': 'pgv_hpo'}
```

Search-space entries are `("float", lo, hi)`, `("float_log", lo, hi)`, `("int", lo, hi)`, `("int_log", lo, hi)` and
`("categorical", [choices])`. Anything else raises `ValueError`. Without Optuna installed you get an `ImportError`
telling you to install it (not an empty result).

Notes:

- Use a **short** `max_epochs` per trial; you are ranking settings, not producing the final model. Retrain the winner
  for the full length afterwards.
- `storage="sqlite:///hpo.db"` saves the study so it can be continued; `n_jobs>1` runs trials in parallel threads
  (mind GPU memory).
- The `pruner` setting is accepted, but trials report only their final score, so nothing is pruned early.
- `hpo.importance()` returns which parameters mattered most (needs enough trials).
- *Verified here with a real Optuna install on a toy objective (15 trials; `importance()` returned shares that sum to 1);
  a search whose trials train real networks was not run.* The search-space handling and error paths are also covered by tests.
- A second, more general search tool exists: [`GeoAutoML`](automl.md) takes any function (not only
  `GeoTrainer`), supports pruning, and records failed trials instead of aborting.
