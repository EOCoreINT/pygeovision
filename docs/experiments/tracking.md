# Experiment tracking

After twenty runs you will not remember which learning rate gave which curve. A **tracker** records the settings and the
per-epoch numbers of every run in one place, so runs can be compared side by side, long after the terminal has closed. PyGeoVision
connects the trainer to three of them (TensorBoard, MLflow, Weights & Biases) with one setting, and it never lets a tracker stop training.

```{figure} ../_static/img/flow-tracking.svg
:alt: Experiment tracking: the config names the backends, the trainer starts them, logs validation numbers every epoch, ends the run, and you look at the result in the backend's own interface
:width: 100%
:class: pgv-figure

One setting selects the backends; the trainer does the rest.
```

```{admonition} What was tested
:class: note
Run here: real training runs with the trackers requested but not installed (the failure path), a stand-in MLflow module that records
every call it receives (the exact calls are listed below), and the tests in `tests/test_trainer_tracking.py`.
**Not run here:** a real TensorBoard, MLflow server or Weights & Biases account (none is installed or reachable from the test
environment). What those tools then display is their own behaviour.
```

## Turn it on

```python
from pygeovision.ai.training.trainer import GeoTrainer, TrainingConfig

cfg = TrainingConfig(
    output_dir="runs/crops",
    experiment_name="crops",             # groups runs: an MLflow experiment, a TensorBoard folder
    run_name="unet-lr3e-4",              # one run; leave empty and a name like run_1760000000 is made
    tracker="mlflow,tensorboard",        # none (default) | mlflow | wandb | tensorboard | all | a comma list
    mlflow_uri="mlruns",                 # where MLflow stores runs (a folder or a server address)
    wandb_project="pygeovision",
    ...
)
GeoTrainer(model, cfg).fit(train_set, val_set)
```

| Setting | Default | Meaning |
|---|---|---|
| `tracker` | `"none"` | Which backends to start. `"all"` is all three. Spaces are ignored and case does not matter. A name that is not one of the three is ignored with a warning that lists the valid ones |
| `experiment_name` | `"pgv_experiment"` | The grouping name used by all backends |
| `run_name` | `""` | The name of this run. Empty creates `run_<unix time>` |
| `mlflow_uri` | `"mlruns"` | MLflow's tracking location: a local folder, or `http://host:5000` for a server |
| `wandb_project` | `"pygeovision"` | The Weights & Biases project |
| `log_every_n_steps` | 10 | How often a per-step message is written at `DEBUG` level only. It does not change what the trackers receive, which is one record per epoch |

There is no command-line tracker option, because `pygeovision ai train` only prepares a config and does not train
([Training](../training/index.md)). Set `tracker` in `TrainingConfig`, including a config loaded from a YAML file
([example](../training/config.md#a-full-example-file)).

## The three backends

| Backend | Install | Where the data goes | How to look at it |
|---|---|---|---|
| `tensorboard` | `pip install tensorboard` | Files under `<output_dir>/tb_logs/<experiment_name>/<run_name>/` | `tensorboard --logdir runs/crops/tb_logs`, then open the address it prints |
| `mlflow` | `pip install mlflow` | The `mlflow_uri` location (default a `mlruns/` folder in the **current directory**, not in `output_dir`) | `mlflow ui` started in the folder that contains `mlruns/` |
| `wandb` | `pip install wandb`, then `wandb login` once | Weights & Biases' servers, in `wandb_project` | The project page on wandb.ai |

Use TensorBoard for a quick look on one machine, MLflow when several people compare runs (put `mlflow_uri` on a shared server) and
Weights & Biases when you want a hosted dashboard. Combining them costs almost nothing, so `"mlflow,tensorboard"` is a sensible default for a
serious project.

## What is recorded

**Once, at the start:** the whole `TrainingConfig` (MLflow gets each setting as a text parameter, cut at **250 characters** per value; W&B
gets it as its run config).

**After every validation pass** (so once per epoch, and only when a validation set is given) these numbers, all as plain floats:

| Name | Meaning |
|---|---|
| `val_loss`, `train_loss` | Mean loss on the validation and training data |
| `val_iou`, `val_f1`, `val_accuracy` | Mean intersection-over-union, mean F1 and pixel accuracy (segmentation) |
| `val_iou_class_<k>`, `val_f1_class_<k>`, `val_precision_class_<k>`, `val_recall_class_<k>` | The same, for each class, so a rare class is visible |
| `val_map50`, `val_map50_95` | Mean average precision, for the detection task |
| `lr`, `epoch` | The learning rate of the first parameter group, and the epoch number |

Verified by listing the `history` of a real 4-epoch run: `lr, train_loss, val_f1_class_0, val_f1_class_1, val_iou, val_iou_class_0,
val_iou_class_1, val_loss, val_map50, val_precision_class_*, val_recall_class_*`. (`val_map50` is present but empty for segmentation.)
Non-numeric entries (a text note about mAP, say) are left out. The step number is the epoch.

**Not recorded automatically:** per-batch losses, the model files, prediction images and system metrics. Add them yourself with the
tracker class below. The trainer does not upload checkpoints; `best.pth` stays in `<output_dir>/checkpoints/`.

## If a tracker cannot start, training goes on

Tracking is a convenience and must not cost you a run. Real log of a run with `tracker="mlflow,tensorboard"` on a machine with
neither installed:

```text
mlflow not installed
tensorboard not installed (pip install tensorboard)
Experiment tracking requested (mlflow, tensorboard) but no backend could start; continuing without it.
Starting training: 4 epochs | device=cpu | precision=auto | strategy=auto
```

A line `Experiment tracking: mlflow, tensorboard` means those backends really started; with only some available, only those are named.
Any later failure of a backend call (a server that goes away mid-run) is logged as
`Experiment tracking call failed (continuing training): ...` and training continues. **Read the log after the first epoch**: a typo in
the tracker name or a missing login shows up there, not as an error.

## Using the tracker on its own

The class the trainer uses can log anything, from any script (a hyper-parameter search, an evaluation, a pipeline run):

```python
from pygeovision.training import ExperimentTracker

tracker = ExperimentTracker("crops", run_name="eval-2026-10", backends=["mlflow", "tensorboard"],
                            mlflow_uri="mlruns", tb_log_dir="./tb_logs")
tracker.start_run(config={"model": "unet-r50", "chip": 256})
for epoch in range(3):
    tracker.log({"val_iou": 0.50 + 0.05 * epoch}, step=epoch)
tracker.log_image("prediction", image_array, step=2)        # TensorBoard and W&B
tracker.log_model("runs/crops/checkpoints/best.pth", "best_model")   # MLflow and W&B
print(tracker.active_backends)                              # the backends that really started
summary = tracker.end_run()
```

| Method | Does |
|---|---|
| `start_run(config=None)` | Starts each backend and logs `config`; returns the tracker |
| `log(metrics, step=None)` | Sends a dictionary of numbers. Also kept in `tracker.history` (a dictionary of lists) |
| `log_model(path, artifact_name="model")` | Uploads a file as an MLflow artifact or a W&B artifact. TensorBoard has no such thing and ignores it |
| `log_image(key, image, step=None)` | TensorBoard and W&B only |
| `end_run()` | Closes the backends; returns `{"experiment", "run_name", "duration_seconds", "metrics"}` where `metrics` holds the **last** value of each name |
| `active_backends` | The list of backends that started |

What an MLflow backend receives for the three-epoch example (verified with a stand-in module that records its calls):

```text
set_tracking_uri('mlruns') · set_experiment('crops') · start_run(run_name='eval-2026-10')
log_params({...})                      each value as text, at most 250 characters
log_metrics({'val_iou': 0.5}, step=0) ... one call per log()
log_artifact('best.pth', artifact_path='best_model')
log_metric('duration_seconds') · end_run()
```

```{note}
The default tracker, if you give no `backends`, is TensorBoard. The TensorBoard step is `step` when you give one; with no `step`
it counts the calls (note that a `step` of 0 is treated as "no step" and also counts calls).
```

## Comparing runs

- **In MLflow:** run `mlflow ui`, tick the runs of one experiment and choose *Compare*; parameters that differ are highlighted, which is
  the fastest way to see what changed between two runs.
- **In TensorBoard:** each run is a folder; pointing `--logdir` at `tb_logs` overlays all of them on one chart.
- **Without a tracker:** `fit()` returns a `history` dictionary, and `last.pth` carries the same history, so you can plot any run from files
  ([Resuming](../training/resuming.md)). The [leaderboard](leaderboard.md) compares models on one dataset with the same metric.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `Experiment tracking requested (...) but no backend could start` | The package is not installed, or W&B is not logged in | Install it; `wandb login` |
| `Unknown tracker(s) ['tensorbord'] ignored` | A typo | Use `mlflow`, `wandb`, `tensorboard`, `all` or `none` |
| `mlflow ui` shows nothing | Started in another folder than the one that holds `mlruns/` | Start it where `mlruns/` is, or set `mlflow_uri` to an absolute path |
| The TensorBoard chart is empty | `logdir` points at the wrong level, or no validation set was given | Use `<output_dir>/tb_logs`; give `fit()` a validation set |
| `Experiment tracking call failed (continuing training)` | The server is unreachable or a value could not be sent | Training is unaffected; check the server. The numbers for those epochs are still in `history` |
| MLflow shows a parameter cut off | Values are limited to 250 characters | Expected; a long list (many class names) is shortened |
| Two runs overwrite each other in TensorBoard | The same `experiment_name` and `run_name` | Give each run its own `run_name` |

## Related

- [Resuming an interrupted run](../training/resuming.md), [Training callbacks](../training/callbacks.md), [Training configuration](../training/config.md)
- [Experiments overview](index.md), [Leaderboard](leaderboard.md), [Hyper-parameter search](search.md)
- [`ExperimentTracker` in the Python API](../reference/api/training.md)
