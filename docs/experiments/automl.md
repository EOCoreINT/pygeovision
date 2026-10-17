# AutoML

**Package:** `pygeovision.advanced.automl` · **Class:** `GeoAutoML` · **Needs:** `pip install optuna` (plotting: `plotly`)

## Why

Learning rate, loss function, backbone, number of epochs: these choices change a score by several points, and good values
differ from dataset to dataset. Trying them by hand is slow and biased towards the combinations you already believe in.
`GeoAutoML` automates the loop: it proposes a configuration, calls **your** training function with it, records the number
that comes back, and uses the history to propose better configurations ([Optuna](https://optuna.org)'s TPE sampler by
default).

It does not know anything about models. What is searched is whatever your function does with the configuration dictionary
it receives. That makes it usable for a U-Net on Sentinel-2, a random forest on indices, or a pipeline parameter; it also
means it cannot tell you *which* parameters are worth searching.

```{admonition} GeoAutoML or OptunaHPO?
:class: tip
PyGeoVision has two Optuna wrappers. [`OptunaHPO`](search.md#hyper-parameter-search) lives in
`pygeovision.training`, and its objective is one `GeoTrainer` run. `GeoAutoML` accepts any function, supports **pruning**
(stopping bad trials early), treats a crashing trial as a *failed trial* instead of aborting the study, and has a
different search-space notation (`("float", lo, hi, "log")` rather than `("float_log", lo, hi)`). Use either; do not mix
their notations.
```

## How to use

```{literalinclude} ../_examples/advanced/automl_demo.py
:language: python
:caption: automl_demo.py
:linenos:
```

{download}`Download automl_demo.py <../_examples/advanced/automl_demo.py>`. The "training" is a formula with a known optimum (learning
rate near 1e-3, loss `combo`) plus a backbone value that always crashes, so you can see how a search behaves in about a
second. In a real project `train_fn` trains a model and returns the validation score, which is why each trial may take
minutes or hours.

### The search function

```python
from pygeovision.advanced.automl import GeoAutoML

automl = GeoAutoML(metric="val_iou", direction="maximize", n_trials=50, seed=42)
result = automl.search(train_fn, search_space, pruner="median", sampler="tpe")
```

`train_fn(config) -> float` receives a dictionary with one value per search-space entry. Return the number to optimise.

### Search-space entries

| Entry | Meaning | Example |
|---|---|---|
| `("float", low, high)` | Uniform real number | `("float", 0.0, 0.5)` |
| `("float", low, high, "log")` | Log-uniform: equal chance per order of magnitude. Use for learning rates | `("float", 1e-5, 1e-2, "log")` |
| `("int", low, high)` | Whole number, both ends included | `("int", 5, 30)` |
| `("categorical", [a, b, c])` | One of a list | `("categorical", ["combo", "focal"])` |
| `("constant", value)` | Passed through unchanged (keeps the config complete) | `("constant", 7)` |

An empty space, an unknown kind, a missing element, `low >= high` or an empty choice list raises `ValueError` **before**
any trial runs, naming the entry.

### Constructor arguments

| Argument | Default | What it does |
|---|---|---|
| `metric` | `"val_iou"` | A *label* for reports. It does not select anything: whatever `train_fn` returns is the metric |
| `direction` | `"maximize"` | `"maximize"` or `"minimize"`. Use `"minimize"` for a loss |
| `n_trials` | 50 | Maximum number of trials |
| `timeout_s` | `None` | Stop after this many seconds (a running trial is allowed to finish) |
| `backend` | `"optuna"` | `"optuna"` or `"ray_tune"` (see [the caveat](#ray-tune)) |
| `n_jobs` | 1 | Parallel trials in threads. Mind GPU memory; with 1 the search is reproducible |
| `storage` | `None` | e.g. `"sqlite:///study.db"` to save the study and continue later. Re-using a `study_name` **continues** that study, so use a new name for a new experiment |
| `study_name` | `"pgv_automl"` | Name of the study |
| `seed` | 42 | Seed of the sampler (same seed, same function, `n_jobs=1` gives the same trials) |
| `show_progress_bar` | `True` | Optuna's progress bar |

`search()` also takes `pruner` (`"median"` default, `"hyperband"`, or `None`) and `sampler` (`"tpe"` default, `"cmaes"`
for numeric-only spaces, `"random"`). An unknown name raises `ValueError`.

### Pruning: stopping bad trials early

Pruning only works if the trial tells Optuna how it is doing. Give `train_fn` a **second argument** and `GeoAutoML` will
pass the Optuna trial:

```python
import optuna

def train_fn(config, trial):
    for epoch in range(config["epochs"]):
        val_iou = train_one_epoch_and_validate(...)
        trial.report(val_iou, epoch)              # tell Optuna the score so far
        if trial.should_prune():                  # it is clearly worse than other trials at this epoch
            raise optuna.TrialPruned()
    return val_iou
```

Without the `trial` argument no intermediate values exist, so no trial is ever pruned, whatever `pruner` says. The
median pruner starts pruning after 5 completed trials and ignores the first 5 reported steps.

## Expected output

Running the script above produced:

```text
success          : True
best_value       : 0.871757
best_params      : {'lr': 0.00183, 'loss': 'combo', 'epochs': 10, 'backbone': 'resnet50'}
completed/pruned/failed: 16 0 24
first error      : ['RuntimeError: CUDA out of memory (simulated)']
importance       : {'lr': 0.9858, 'epochs': 0.0104, 'loss': 0.002, 'backbone': 0.0018}
with pruning     : completed 21 pruned 19
all NaN          : False - no trial completed; first error: train_fn returned nan for {'x': 0.374
```

plus one log line per failed trial (`Trial 8 failed: RuntimeError: CUDA out of memory (simulated)`) on standard error.

How to read it:

* **`best_params` / `best_value`**: the best *completed* trial. With a smooth objective the search found the right learning
  rate (0.0018, true optimum 0.001) and the right loss in 40 trials; real objectives are noisier.
* **`completed / pruned / failed`**: 24 of 40 trials hit the "broken" backbone and failed. They never count towards the
  best value, and the study carried on. A high failed count is a signal to look at `errors`.
* **`importance`**: the share of the variation in the score explained by each parameter, summing to 1. Here the learning
  rate explains 99 %. With few trials these numbers are noise; they need at least two completed trials and a parameter that
  varies.
* **`with pruning`**: 19 of 40 trials were stopped early, saving most of their cost.

Full return value on success:

```text
{"success": True, "best_params": {...}, "best_value": float, "n_trials_completed": int, "n_trials_pruned": int,
 "n_trials_failed": int, "metric": str, "direction": str, "study_name": str, "errors": [str, ...]}
```

`errors` holds the first five distinct failure messages. If **no** trial completed, you get
`{"success": False, "error": "no trial completed; first error: ...", "n_trials_pruned", "n_trials_failed", "errors"}`: it is
never reported as a success with an empty result.

### What counts as a failed trial

An exception raised by `train_fn`, a return of `None`, or a return of `NaN`. All are recorded and skipped, so a
configuration that runs out of memory costs you that trial, not the whole search. (`TrialPruned` is not a failure.)

### After the search

```python
automl.importance()                                    # {'lr': 0.98, ...}; {} before a search or if not computable
automl.plot_optimization_history("history.html")       # interactive plot; needs plotly
fig = automl.plot_optimization_history()               # or get the figure and show it yourself
```

`.html` output needs only plotly; `.png` / `.svg` also need `kaleido`. Calling it before `search()` raises
`RuntimeError`.

## What to do when…

| Situation | What to do |
|---|---|
| `success` is `False` and `error` says "no trial completed" | Read `result["errors"]`: the first message is the cause (out of memory, a bad key in `config`, a data path). Run `train_fn({...one valid config...})` by hand until it returns a number |
| `n_trials_failed` is large but the study succeeded | Some configurations crash. Narrow the space (smaller batch sizes, fewer epochs) or accept it; failed trials never become "best" |
| Every trial has nearly the same value | The metric barely depends on the searched parameters, the validation set is too small/easy, or your function ignores `config`. Check `importance()` and that `config` values are actually used |
| Results are not repeatable | Set `seed`, use `n_jobs=1`, and make `train_fn` seed its own training. GPU training is not bit-for-bit deterministic |
| Nothing is ever pruned | `train_fn` needs the second `trial` argument and must call `trial.report(...)` / `trial.should_prune()` each epoch |
| The search takes too long | Lower `max epochs` per trial (you are ranking settings, not producing the final model), set `timeout_s`, use pruning, and retrain the winner for the full length afterwards |
| The best value is much better than your final model | The search over-fitted the validation set (many trials, one small set). Re-evaluate the winner on a separate test set ([benchmark](benchmark.md)) |
| `ImportError: ... optuna` | `pip install optuna` (it is in the `pygeovision[train]` extra) |
| `search()` returns `{"success": False, "error": "pip install ray[tune]"}` | Ray Tune is not installed. Prefer `backend="optuna"` |
| You want a SQLite study you can resume | `GeoAutoML(storage="sqlite:///hpo.db", study_name="exp1")`. Keep the same `study_name` to continue, a new one to start fresh |

### Ray Tune

`backend="ray_tune"` converts the same search space to Ray's and runs it with `tune.run`. **This path was not run here**
(Ray is not installed in the test environment), it does not support pruning or the `trial` argument, and Ray's API changes
between versions. Treat it as experimental; the Optuna backend is the supported one.

## Limits

* The metric is only as good as the validation data you give your function. A search will find whatever flatters that set.
* Categorical choices that are *names* (backbone, loss) only matter if `train_fn` looks them up. A typo in a name fails
  every trial that picks it, which shows up as `n_trials_failed`.
* Trial count needed grows with the number of parameters: 50 trials for 3–5 parameters is reasonable, 50 for 15 is not.
