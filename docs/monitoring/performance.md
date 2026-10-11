# Performance tracking

**Classes:** `PerformanceDrift`, `ModelPerformanceTracker` (`pygeovision.monitoring`), `PerformanceTracker` (`pygeovision.ai.monitoring`) ·
**Python:** `client.monitoring.performance_tracker(model_name)`, `client.ai.monitor.performance_tracker()`

## Why

[Data drift](drift.md) tells you the inputs changed. Only measured accuracy tells you the **model got worse**. This
needs reference labels: every month, label a small fresh sample (even 20–50 chips), score the model on it, and log the
numbers. These classes store the history, compare it with the baseline you validated and raise a flag.

There are three of them, for three jobs:

| Class | Job | Compares with |
|---|---|---|
| `PerformanceDrift` | "Has accuracy dropped below the baseline by more than X?" | A **fixed baseline** you set |
| `ModelPerformanceTracker` | A per-model logbook with `trend` and `summary` | The previous record |
| `ai.monitoring.PerformanceTracker` | A light logbook that logs a warning on regression | The mean of the last 10 records |

## Performance drift: the detector API

```python
from pygeovision.monitoring import PerformanceDrift

pd_ = PerformanceDrift(window_size=100,
                       alert_thresholds={"mean_iou": 0.05, "accuracy": 0.03},
                       storage_path="./monitoring/performance_log.jsonl")
pd_.set_baseline({"mean_iou": 0.80, "accuracy": 0.93})        # from your validation set
pd_.log({"mean_iou": 0.79, "accuracy": 0.93})                 # each time you score fresh labelled data
report = pd_.detect(recent_n=2)
```

| Member | Meaning |
|---|---|
| `PerformanceDrift(window_size=100, alert_thresholds=None, storage_path="./monitoring/performance_log.jsonl")` | `alert_thresholds` = **absolute** drop allowed per metric. Default `mean_iou` 0.05, `accuracy` 0.03, `mean_f1` 0.05. Creates the folder; loads an existing log |
| `set_baseline(metrics)` | The reference values, from validation data |
| `log(metrics, timestamp=None)` | Appends one JSON line to the log file (UTC ISO time by default) |
| `detect(recent_n=None)` | Averages the latest `recent_n` records (default `window_size`) and compares each metric with the baseline |
| `plot_history(metric="mean_iou", save_path=None)` | Line plot with the baseline and the alert line (needs matplotlib) |

Only metrics that are in `alert_thresholds` **and** in the baseline are compared. A metric you log but did not list is
stored and ignored.

Real output after logging mean IoU 0.79, 0.78, 0.74, 0.70 and accuracy 0.93, 0.92, 0.91, 0.90 against a baseline of
0.80 / 0.93:

```text
detect()              -> n_measurements 4, mean_iou recent 0.7525 (drop 0.0475, threshold 0.05)  drift_detected: False
detect(recent_n=2)    -> n_measurements 2, mean_iou recent 0.72   (drop 0.08)                    drift_detected: True
                         alerts: ['mean_iou dropped by 8.0% (threshold: 5.0%)']
                         recommendation: Minor performance drop in mean_iou dropped by 8.0% ... Consider re-calibration.
```

**Read this carefully.** The default window averages all records, so a *sudden* drop is diluted by the earlier good
ones: the newest score here is 0.70 (a 0.10 drop), yet the 4-record average only shows 0.0475, just under the threshold.
Pick `recent_n` to match how fast you need to react: `1` reacts to every single noisy measurement, a window of 3–5 trades
speed for stability. The `drop` is an absolute difference in the metric (0.08 = 8 points), although the alert text
formats it with a % sign.

Edge cases: no baseline → `{"error": "No baseline. Call set_baseline() with validation metrics."}`; nothing logged →
`{"error": "No performance history logged yet."}`. The log file persists, so `detect` after a restart sees the old records.

### Through `DriftDetector`

`DriftDetector.fit(images, reference_metrics={...})` sets the baseline and `check(images, current_metrics={...}, recent_n=...)`
logs a record and runs both comparisons ([Data drift](drift.md)). The history is `storage_dir/perf.jsonl`. With no
baseline the performance part of the report is an `error`, which is harmless and listed in `report["errors"]`.

## The logbook: `ModelPerformanceTracker`

```python
from pygeovision.monitoring import ModelPerformanceTracker

tr = ModelPerformanceTracker("unet-r50", storage_path="./monitoring/tracker.json")
tr.record({"mean_iou": 0.80, "loss": 0.30}, split="production", n_samples=100, notes="June batch")
tr.trend("mean_iou", window=10)
tr.summary()
```

`record(metrics, split="production", n_samples=0, notes="")` appends an entry (values rounded to 6 digits) and rewrites
the JSON file. `trend(metric, window=10)`:

```text
{'metric': 'mean_iou', 'n_records': 4, 'current': 0.75, 'mean_recent': 0.7833,
 'direction': 'declining', 'trend': 'declining', 'min': 0.75, 'max': 0.81}
```

`direction` compares **only the last two records** ("improving" if the newest is higher), so it is noisy and says
nothing about how large the change is. It also assumes *higher is better*: for a loss, "declining" is good news. A metric
with no change reads `stable`. `summary()` gives the number of records, first and latest timestamps, the latest value
of every metric and its trend.

Limits: the file is rewritten as a whole on every `record` (not safe for several writers at once), and a corrupt JSON file
makes the constructor raise.

## The light logbook: `ai.monitoring.PerformanceTracker`

```python
from pygeovision.ai.monitoring import PerformanceTracker

pt = PerformanceTracker(metrics=["miou"], storage_path="./monitoring/history.json", alert_on_regression=0.05)
pt.log({"miou": 0.80}, model_name="unet")
pt.get_trend("miou")          # [0.8, ...]
pt.summary()                  # {'miou': {'mean':..., 'min':..., 'max':..., 'last':...}}
```

After at least 5 earlier records for a metric, `log` compares the new value with the mean of the previous (up to 10)
and logs a **warning** if it dropped by more than `alert_on_regression` *relative* (5 % of the baseline by default). With
mean IoU 0.80, 0.81, 0.79, 0.80, 0.82, 0.80 and then 0.74 you get
`Performance regression detected: miou dropped from 0.8033 → 0.7400 (7.9%)` in the log. That is all it does: it neither
raises nor calls an alert handler, so wire the warning into your logging.

The default `storage_path` is **`~/.pygeovision/performance_history.json`**, one file for everything you run as that user.
Pass a project-specific path. A history file that cannot be parsed is renamed to `<name>.corrupt.json` and a warning is
logged, so it is not overwritten.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `{"error": "No baseline..."}` | `set_baseline` not called | Provide the validation metrics |
| `{"error": "No performance history logged yet."}` | Nothing logged | `log` after each evaluation |
| A clear recent drop is not flagged | Window averages it away | Lower `recent_n` |
| Alerts flip on and off | Metric measured on tiny, noisy samples | Larger evaluation samples; `recent_n` of 3–5 |
| `drop` is negative | Performance improved | Nothing |
| Metric not compared | It is not in `alert_thresholds` or not in the baseline | Add it to both |
| History from an old run still counts | The log file persists | Use a new `storage_path` per model/campaign |
| `trend` says "declining" for a falling loss | It assumes higher is better | Interpret per metric |
| `ValueError`/`JSONDecodeError` loading a tracker | Corrupt JSON | Fix or remove the file |
| Records of different models mixed | Shared default path in `ai.monitoring` | Use one path per model |

```{seealso}
[Data drift](drift.md) · [Alerts](alerts.md) · [Benchmarking](../benchmark/index.md)
```
