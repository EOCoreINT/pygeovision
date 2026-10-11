# Site tracking over time

**Module:** `pygeovision.monitoring.site_tracker` · **Functions:** `extract_sites`, `match_sites_across_epochs`,
`KalmanAreaTracker`, `classify_site_activity`, `track_and_classify_sites`

## Why

Many questions are about **individual things that change**: is this mine pit growing, is that clear-cut recovering, which
flooded area is shrinking? Given one change or classification map per date, you want a table: *site 3 grows by about 14 000 m²
per epoch; site 7 is stable; site 9 appeared at epoch 5*. A single area measurement per date is noisy (clouds, thresholds,
classifier errors), so the module **smooths** each site's area series with a Kalman filter and reports the **rate of
change together with its uncertainty**, instead of treating every wobble as a trend.

It works on binary masks (1 = disturbed pixel), for example a thresholded probability map from a segmentation model or an
[index-based map](../data/indices.md).

## How to use: one call

```python
from pygeovision.monitoring.site_tracker import track_and_classify_sites

results = track_and_classify_sites(
    binary_masks,                  # list of 2-D bool/0-1 arrays, one per epoch, same grid
    epoch_indices=list(range(8)),  # time labels (years, months, ...); uneven gaps are respected
    pixel_area_m2=100.0,           # 10 m pixels
    min_area_m2=300.0,             # ignore specks smaller than this
)
for site_id, r in results.items():
    print(site_id, r["classification"], round(r["final_rate_m2_per_epoch"]), r["areas_m2"])
```

On a synthetic example (8 epochs, 100 m² pixels, speckle noise; an expanding disc, a stable one, a shrinking one and a
site that appears at epoch 5):

```text
site_0000 expanding   14157  [11300, 17700, 25300, 34900, 44100, 57700, 70900, 86100]
site_0001 stable          1  [31800, 31700, 31700, 31700, 31700, 31700, 31700, 31700]
site_0002 recovering  -5165  [61300, 48900, 37700, 29300, 19700, 13700, 8100, 3800]
site_0003 stable          0  [4900, 4900, 4900]
```

The true growth rate of the expanding disc is about 14 800 m² per epoch (the filter reports 14 157, a lag typical of
smoothing). The 31 one-pixel speckles are dropped by `min_area_m2=300` in that run. `epoch_indices` are the time labels; rates
are "per epoch" in **those units**: use years and you get m² per year.

Returned per site: `epochs`, `areas_m2` (the raw measurements), `filtered_areas_m2`, `rate_m2_per_epoch` (the filtered rate
at every step), `final_rate_m2_per_epoch` and `classification`.

| Parameter | Meaning |
|---|---|
| `binary_masks`, `epoch_indices` | Same length, in time order; all masks on the same grid |
| `pixel_area_m2` | The real area of one pixel. **Required** so that areas are in m² rather than pixel counts (use `abs(transform.a * transform.e)` of a projected CRS) |
| `min_area_m2` | Components smaller than this are dropped at every epoch (default 100) |
| `iou_threshold` | Minimum overlap to say two detections are the same site (default 0.1) |
| `stable_threshold_m2_per_epoch` | Rate magnitude below which a site counts as stable (default 100) |
| `**kalman_kwargs` | `process_noise_area`, `process_noise_rate`, `measurement_noise_m2` for the filter |

## The steps, one by one

### 1. Extract sites: `extract_sites`

```python
sites = extract_sites(mask, epoch_index=0, pixel_area_m2=100.0, min_area_m2=300.0, connectivity=8)
sites[0].area_m2, sites[0].centroid_rc, sites[0].n_pixels
```

Labels the connected components of the mask (8- or 4-connectivity; needs `scipy`) and returns `Site` objects:
`label`, `epoch_index`, `pixel_mask` (a full-size boolean array), `centroid_rc` (row, column), `area_m2`, `n_pixels`. On the
example there were 3 sites above 300 m²; with `min_area_m2=0` there were **31** (the speckle). Always set a
minimum: single pixels are almost never sites.

Memory note: every `Site` keeps a **full-size boolean mask**. For a large raster with hundreds of sites over many epochs
that is a lot of memory; crop the masks to the area of interest first.

### 2. Follow identities: `match_sites_across_epochs`

```python
tracks = match_sites_across_epochs([sites_t0, sites_t1, ...], iou_threshold=0.1)   # {site_id: {epoch: Site}}
```

For each epoch, each site is matched to the existing track whose last detected shape overlaps it most (intersection over
union, IoU), provided the overlap is at least `iou_threshold` and the track is not already taken in that epoch. Otherwise a new
track `site_NNNN` starts. A site that is missed in some epoch simply has **no entry** for it (nothing is invented).

Consequences to be aware of:

- Matching is **greedy** and compares with the track's last mask only.
- A site that **splits** keeps one part under its old id; the other part is a new site. **Merging** sites: one id continues
  and the other track ends.
- A site that moves less than its own size is fine; a site that moves completely (no overlap) becomes a new site.
- The IoU threshold of 0.1 is lenient. A fast-growing site still overlaps its earlier shape; raise it if neighbouring sites
  get confused.

### 3. Smooth and estimate the rate: `KalmanAreaTracker`

```python
from pygeovision.monitoring.site_tracker import KalmanAreaTracker
k = KalmanAreaTracker(process_noise_area=500.0, process_noise_rate=2000.0, measurement_noise_m2=2000.0)
r = k.track(epochs=[0, 1, 2, 3, 4], areas_m2=[1000, 1400, None, 2200, 2600])
```

```text
filtered area  [1000, 1320, 1522, 2162, 2591]
rate (m²/epoch) [0, 202, 202, 396, 415]
rate std dev    [64, 60, 75, 59, 59]
```

The state is `[area, rate]`. The model says: next area = area + rate, and the rate itself changes slowly. Each epoch the
filter predicts, then corrects with the measured area, weighting the measurement by how noisy you say it is. A `None`
(site missed) gives a prediction-only step with growing uncertainty, as in epoch 2 above.

| Argument | Meaning |
|---|---|
| `process_noise_area` | How far (variance, m²²) the true area may depart from the straight-line prediction in one epoch |
| `process_noise_rate` | How much the true rate itself may change per epoch ((m²/epoch)²). Larger: follows changes in the rate faster |
| `measurement_noise_m2` | Variance of one area measurement. Larger: trusts single epochs less, smooths more |
| `initial_rate_guess` (in `track`) | Prior rate, default 0 |

The defaults suit areas in the **hundreds to tens of thousands of m²**. For hectare-scale sites scale the three noise terms with the square
of the area unit, or the filter will track noise. `epochs` must be increasing and need not be evenly spaced (the time gap is
used). The rate starts at the prior (0) and takes several epochs to find a trend, which is why it lags.

### 4. Classify: `classify_site_activity`

```python
classify_site_activity(rate, rate_uncertainty, stable_threshold_m2_per_epoch=100.0, n_sigma=1.0)
```

| Condition | Result |
|---|---|
| `rate > threshold + n_sigma × uncertainty` | `expanding` |
| `rate < -threshold − n_sigma × uncertainty` | `recovering` (shrinking) |
| otherwise | `stable` |

Uncertainty is part of the decision: a rate of 150 m²/epoch with a standard deviation of 200 is called `stable`,
because the filter cannot tell it from zero. Examples: `(400, 50)` → expanding, `(60, 30)` → stable, `(-400, 50)` →
recovering, `(150, 200)` → stable. `track_and_classify_sites` calls this with `n_sigma = 1` on the **last** epoch.

A site seen in only one epoch is returned with `classification: "insufficient_data"` and rate 0: one point does not make
a trend.

"Recovering" only means the area is shrinking. Whether that is vegetation regrowth or a model not detecting something any
more is for you to decide.

## Limits

- **Detection quality dominates.** The filter smooths random noise, not a **systematic** bias such as a threshold that
  misses the pit's edge in one season. Use the same model and threshold at every epoch.
- Masks must be **co-registered** on the same grid, with a constant pixel size.
- Cloud gaps look like shrinkage. Mask clouds and treat them as missing rather than as zero.
- The motion model is a straight line with slow change; abrupt events (a sudden clearing) are smoothed over several epochs.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| Hundreds of tiny sites | No size filter | Raise `min_area_m2` |
| One site has two ids over time | Overlap below `iou_threshold` between epochs (fast moves or gaps) | Lower `iou_threshold`, or use closer epochs |
| Two real sites became one | They merged or touch (8-connectivity joins diagonals) | `connectivity=4`; or accept the merged site |
| `ValueError: need at least one real area measurement` | All areas `None` | Check detection |
| `ValueError: epochs and areas_m2 must be the same length` | Mismatched lists | Same length |
| `insufficient_data` | A site seen once | Expected for new or noisy detections |
| Rate seems to lag the true change | Filter smoothing; the rate starts at 0 | Raise `process_noise_rate`; longer series |
| Everything `stable` for a slowly changing site | Rate under `stable_threshold_m2_per_epoch` | Lower it to your meaningful change |
| Everything `expanding` for noise | Noise terms too small for your areas | Raise `measurement_noise_m2` |
| `ImportError: scipy required` | scipy missing | `pip install scipy` |
| `MemoryError` | Full-size masks per site | Crop to the region of interest |

```{seealso}
[Change detection pipelines](../pipelines/index.md) · [Data drift](drift.md)
```
