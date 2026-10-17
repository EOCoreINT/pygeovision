# Monitoring

**Packages:** `pygeovision.monitoring` (imagery-based, the main layer) and `pygeovision.ai.monitoring` (feature-array based) ·
**Python:** `client.monitoring`, `client.ai.monitor` · **CLI:** `pygeovision monitor drift`

## Why

A model is validated once, on data from one place and time. Afterwards the world changes: a new satellite replaces the old
one, the season turns, haze or a different sun angle shifts every pixel, a province you never trained on is added to the
job list. The model keeps producing maps, and they look just as confident as before. **Nothing in the output tells you the
inputs have left the territory the model knows.**

```{figure} ../_static/img/flow-monitoring.svg
:alt: Monitoring a deployed model: fit a reference from training data once, check each new batch of imagery for input drift, check accuracy on a fresh labelled sample, alert someone, then retrain or investigate
:width: 100%
:class: pgv-figure

Fit a reference once; check every run; alert; and only then decide what to do.
```

Monitoring answers two different questions, and it is important not to confuse them:

| Question | Needs labels? | What it catches | Tool |
|---|---|---|---|
| Have the **inputs** changed? (data drift) | No | New sensor, haze, season, calibration, a new region | [Data drift](drift.md) |
| Has the **accuracy** changed? (performance drift) | Yes: a fresh reference sample | The model actually getting worse | [Performance tracking](performance.md) |
| Which **objects** are growing or shrinking over time? | No | Mines, clear-cuts, floods, built-up growth | [Site tracking](site-tracking.md) |
| **Who is told**, and how? | | | [Alerts](alerts.md) |

Data drift is a warning light, not a verdict: inputs can drift and the model stay fine (a brighter but otherwise similar
season), or inputs stay put and the model fail (an unseen crop type). Performance drift is the real test, but it costs you
reference labels. Use both.

## Which of the two stacks?

| | `pygeovision.monitoring` | `pygeovision.ai.monitoring` |
|---|---|---|
| Input | **GeoTIFF files** (per-band histograms, PSI) | **NumPy feature arrays** `(N, D)` (KS test or MMD) |
| Class | `DriftDetector`, `DistributionDrift`, `PerformanceDrift` | `DriftDetector`, `PerformanceTracker` |
| Also has | `ModelPerformanceTracker`, `AlertManager`, site tracking | `DriftReport` dataclass |
| Use when | You monitor an imagery pipeline | You already extract features (embeddings, band statistics) |
| Python entry | `client.monitoring` | `client.ai.monitor` |

The two classes called `DriftDetector` are **different**; check the import path.

## How often, and on what

| Decision | Guidance |
|---|---|
| How often to check inputs | On every batch you serve, or at least weekly. It is cheap and needs no labels |
| How big a batch | Enough pixels per band for a stable histogram: whole scenes, not a handful of chips. A tiny batch swings the statistic and can look like drift |
| What reference to fit | The data the model **was validated on**, not whatever you have lying around. The reference is the definition of "normal" |
| How often to check accuracy | Whenever you can afford a fresh labelled sample: monthly or per season is typical. Between samples you only have the input check |
| What to do on an alert | Look first (the scenes, the dates, the sensor), then confirm with a labelled sample; retrain only if accuracy actually fell |
| After you retrain | Fit a **new reference** from the new training data, or every later run looks like drift against the old one |

## A quick end-to-end

```python
from pygeovision.monitoring import DriftDetector, AlertManager

det = DriftDetector(storage_dir="./monitoring/")
det.fit(training_tifs, reference_metrics={"mean_iou": 0.80, "accuracy": 0.93})   # once, from training/validation data

report = det.check(this_weeks_tifs, current_metrics={"mean_iou": 0.77})           # every run
AlertManager().check_drift_report(report)

if report["incomplete"]:
    print("could not assess:", report["errors"])
elif report["action_required"]:
    print("retrain or investigate:", report["data_drift"]["recommendation"])
```

or from the shell, for data drift only:

```console
$ pygeovision monitor drift ./train_tifs ./this_week --output drift_report.json
  Drift level: major (PSI 2.4081, worst band band_3 3.1377) | Report: drift_report.json
```

## What each page covers

| Page | Read it for |
|---|---|
| [Data drift](drift.md) | How the input check works, the PSI statistic and its levels, the report layout, the feature-array version |
| [Performance tracking](performance.md) | Recording scores over time against a reference and reading a decline |
| [Site tracking](site-tracking.md) | Following objects (a mine, a clear-cut) through time with a filter that smooths noisy measurements |
| [Alerts](alerts.md) | Turning a report into alerts and sending them somewhere, with copyable Slack, e-mail and file handlers |

## Related

- [Deploying models](../deployment/index.md): monitor what you deploy, from the first week
- [Inference](../inference/index.md#matching-the-input-to-training): many apparent drifts are a scaling mismatch
- [`pygeovision monitor` on the command line](../reference/cli/monitor.md)

```{toctree}
:hidden:
:maxdepth: 1

drift
performance
alerts
site-tracking
```
