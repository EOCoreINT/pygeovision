# Data drift

**Classes:** `DriftDetector`, `DistributionDrift` (`pygeovision.monitoring`), `DriftDetector` (`pygeovision.ai.monitoring`) ·
**CLI:** `pygeovision monitor drift`

## Why

You want to know, **before** the maps go wrong, that the imagery arriving today does not look like the imagery the model was
trained on. Comparing pixel distributions needs no labels and runs in seconds.

## Imagery: `pygeovision.monitoring`

### The idea

For each band, the **reference** images are summarised as a histogram with fixed bin edges. Current images are counted into
those same bins and the two histograms are compared with the **Population Stability Index (PSI)**:

PSI = Σ (current share − reference share) × ln(current share ÷ reference share), summed over the bins.

| PSI | Meaning | Level in the report |
|---|---|---|
| < 0.1 | Stable | `none` |
| 0.1 – 0.2 | Some shift | `minor` |
| ≥ 0.2 | Large shift | `major` |

(These are the conventional PSI thresholds; they come from credit scoring, not from remote sensing. Treat them as a scale
for ranking, and calibrate on your own data: see the tested values below.) Current values outside the reference range are counted in the first or last bin, so
a shift beyond the reference range **is** seen. A bin that is empty on one side is floored at a 0.01 % share, so a
completely new range of values produces a large PSI (several units) rather than infinity.

### How to use

```python
from pygeovision.monitoring import DriftDetector

det = DriftDetector(storage_dir="./monitoring/")           # creates the folder
det.fit(reference_tifs)                                     # list of GeoTIFF paths
report = det.check(current_tifs)
print(report["data_drift"]["drift_level"], report["data_drift"]["overall_psi"])
det.save_report(report, "drift_report.json")
```

`DriftDetector(model=None, storage_dir="./monitoring/")` builds a `DistributionDrift` and a [`PerformanceDrift`](performance.md#performance-drift-the-detector-api)
(history in `storage_dir/perf.jsonl`). `model` is stored and not used.

| Method | What it does |
|---|---|
| `fit(reference_images, reference_metrics=None)` | Builds the reference histograms (and sets the performance baseline if you pass metrics). Returns the detector |
| `check(current_images, current_metrics=None, recent_n=None)` | Data drift + (if metrics are given) performance drift. `recent_n`: how many latest metric records are averaged (default all, up to the window of 100) |
| `save_report(report, path=None)` | Writes the report as JSON; default `storage_dir/drift_report_<time>.json` |

What is sampled: up to **100 reference images** and **50 current images**; from each image up to **10 000 random valid
pixels per band** (fixed random seed, so the same files give the same numbers). NaN, infinite and **nodata pixels are
skipped**. Unreadable files are skipped with a warning and counted. Bands are matched **by position** (`band_1` …), so
the current images must have the same band order as the reference.

### The report

```text
{'timestamp': '<time>', 'drift_detected': <bool>, 'action_required': <bool>,
 'incomplete': False, 'errors': [],
 'data_drift': {
    'overall_psi': <value>, 'drift_level': '<none|minor|major>', 'max_psi': <value>, 'worst_band': 'band_3',
    'per_band': {'band_1': {'psi': <value>, 'kl': ..., 'level': '...', 'mean_reference': ..., 'mean_current': ...}, ...},
    'n_reference': <n>, 'n_current': <n>,
    'recommendation': '<text matching the level>'},
 'performance_drift': {...}}
```

| Key | Meaning |
|---|---|
| `drift_detected` | Data drift is `minor` or `major`, **or** performance drift fired |
| `action_required` | Data drift `major` or performance drift fired |
| `incomplete` | **True when the data comparison could not be made** (see below). Then `drift_detected: False` means *unknown* |
| `errors` | The reasons |
| `overall_psi` | Mean of the per-band PSI. `max_psi` / `worst_band` show the most drifted band, which the mean can hide |
| `kl` | KL divergence of the two histograms (a second view; `level` uses PSI) |
| `n_reference`, `n_current` | Images actually read |

When nothing can be compared (empty or unreadable reference or current images, no matching bands) the answer is
`{"error": "..."}` inside `data_drift` and `incomplete: True`. It is **never** reported as "no drift". Check
`report["incomplete"]` in any automated job.

### Reading the numbers

Judge a PSI against a **baseline from your own data**: compare the reference scenes with other scenes from the same place and
season that you know are fine. That PSI is not zero, because even an identical process gives different samples, and it is the
level to beat before you call anything drift. Typical readings: below about 0.1 none, 0.1 to 0.25 minor, above 0.25 major.
A brightness or contrast change (haze, a different sensor gain, a processing change) pushes the PSI of the affected bands far
above a same-season baseline, and `worst_band` tells you which.

One trap: fill-values. A file with a stripe of zeros will look like drift unless the zeros are declared as no data. **Declare
`nodata` in your GeoTIFFs**: when the file declares it, those pixels are ignored; when it does not, they are counted as data.

`DistributionDrift` on its own works with arrays: `compute_psi(expected_values, actual_values)` takes two samples and
returns the PSI (a normal sample against itself shifted by 0.5 standard deviations gives 0.25, shifted by 5 gives 12.4), and
`compute_kl_divergence(p, q)` takes two histograms.

### From the shell

```console
$ pygeovision monitor drift ref_folder current_folder --output drift.json
  Drift level: <none|minor|major> (PSI <value>, worst band <band> <value>) | Report: drift.json
$ pygeovision monitor drift ref_folder empty_folder --output e.json
  ✗ drift could not be assessed: Nothing could be compared: no readable current images, or their bands do not match the reference. | Report: e.json
```

Uses up to 50 `*.tif` per folder (the first 50 in directory order), and writes the full report. If the comparison cannot be
made it prints `✗ drift could not be assessed: <reason>` and exits with status 1, so a scheduler sees the failure.

## Feature arrays: `pygeovision.ai.monitoring`

```python
from pygeovision.ai.monitoring import DriftDetector

det = DriftDetector(threshold=0.1, method="ks")      # or "mmd"
det.fit_reference(train_features)                    # (N, D) array or list
report = det.check(new_features)                     # DriftReport
print(report.drift_score, report.is_drifted, report.feature_drift)
```

Features can be anything numeric: band statistics per tile, embeddings from a foundation model, model confidence scores.

| `method` | Score | Per-feature output |
|---|---|---|
| `"ks"` (default) | Mean over features of the Kolmogorov–Smirnov statistic (0 = same, 1 = fully separated) | `feature_0 … feature_D` |
| `"mmd"` | Maximum Mean Discrepancy with an RBF kernel (`gamma = 1/D`), on random subsamples of 500 | `{"mmd": value}` |

`DriftReport(timestamp, drift_score, is_drifted, feature_drift, threshold)`. `is_drifted` is `score > threshold`.
`get_history()` returns all past reports; `window_size` is stored but not used.

Errors are explicit: `RuntimeError` before `fit_reference`, `ValueError` for empty features, a different number of columns,
or a method other than `ks`/`mmd`.

### Tested behaviour (reference: 2 000 × 4 standard-normal values)

| Production data (500 × 4) | KS score | KS flagged (>0.1) | MMD score | MMD flagged (>0.1) |
|---|---:|:-:|---:|:-:|
| Same distribution | 0.046 | no | 0.002 | no |
| Mean +0.3 | 0.142 | yes | 0.019 | no |
| Mean +1 | 0.389 | yes | 0.197 | yes |
| Spread ×2 | 0.175 | yes | 0.119 | yes |
| One of four columns +1 | 0.139 (that column 0.43) | yes | 0.080 | **no** |

What the table says:

- **The two scores do not share a scale.** The same threshold of 0.1 means a mean shift of 0.3 standard deviations for KS
  and about 0.7 for MMD. Set the threshold per method, from data you know to be stable.
- **MMD uses unscaled features.** `gamma = 1/D` assumes comparable scales. Standardise the features first, or one large-valued feature dominates.
- **The KS score is a mean over features**, so one drifted column among many is diluted; read `feature_drift`.
- **Small production batches false-alarm.** 30 samples from the *same* distribution scored 0.163 (flagged); 2 000 samples scored 0.022. Use at least a few hundred.
- MMD subsamples randomly (unseeded), so scores vary slightly between runs (0.022–0.024 in four repeats).

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `incomplete: True`, "No reference stats" | `fit` not called | Call `det.fit(reference_tifs)` |
| `incomplete: True`, "Reference statistics are empty" | No readable reference images | Check the paths and that the files are GeoTIFFs; warnings name unreadable files |
| `incomplete: True`, "Nothing could be compared" | No readable current images, or band names do not match | Check files; same band count and order as the reference |
| "Reference statistics have no bin edges" | Statistics saved by an older version | Run `fit` again |
| PSI is huge on a normal-looking scene | Fill value (0, -9999) not declared as nodata | Set `nodata` in the files |
| PSI high after a legitimate change (new season) | That is a real shift of the inputs | Decide: add the season to training, or accept; confirm with [performance](performance.md) |
| `overall_psi` is low but `max_psi` high | One band drifted, the mean hides it | Read `worst_band` and `per_band` |
| Level flips between `none` and `minor` | Near the threshold, sampling noise | Use more images; do not alert on a single crossing |
| `n_current` smaller than the number of files | Unreadable files skipped (a warning names them) | Check the log |
| Reference has more than 100 images and the rest is ignored | `MAX_REFERENCE_IMAGES = 100` | Sample the reference yourself to cover all seasons |
| KS/MMD: `ValueError … columns` | Production features have a different width | Use the same feature extractor |
| KS flags drift on a small batch | KS is noisy for N < a few hundred | Collect larger batches |

```{seealso}
[Performance tracking](performance.md) · [Alerts](alerts.md) · [Preprocessing](../data/preprocessing.md)
```
