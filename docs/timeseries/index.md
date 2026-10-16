# Time series of a spectral index

**Package:** `pygeovision.advanced.timeseries` · **Class:** `GeoTimeSeries` · **CLI:** `pygeovision timeseries analyze | anomaly` ·
**Needs:** `rasterio`, `numpy` (`scipy` for p-values, `matplotlib` for plots)

## Why

A single image shows the state of the land; a stack of images shows its **behaviour**: the green-up in spring, a drought that
holds NDVI down for a season, a slow decline after land clearing. `GeoTimeSeries` reduces each image of a stack to a few
numbers for one index (mean, spread, how many pixels were usable), and gives you the three questions you usually ask next:
*is any date unusual?* (`detect_anomalies`), *is there a trend?* (`compute_trend`), *what is happening at this one spot?*
(`extract_point`).

```{admonition} One number per date, for the whole image
:class: warning
The mean over an image hides everything that varies inside it: a drought on 10 % of the area moves the mean very little.
`GeoTimeSeries` suits **uniform study areas** (a field, a catchment, a forest block, a lake) and monitoring at the level of
"how is this site doing". For **maps** of change, use the per-pixel pipelines
[`vegetation_anomaly`](../pipelines/catalog/vegetation_anomaly.md) and
[`water_extent_timeseries`](../pipelines/catalog/water_extent_timeseries.md). It also does not classify crops or map urban
change; it measures one index through time.
```

## How to use

```{literalinclude} ../_examples/advanced/timeseries_demo.py
:language: python
:caption: timeseries_demo.py
:linenos:
```

{download}`Download timeseries_demo.py <../_examples/advanced/timeseries_demo.py>`. It makes 36 monthly 4-band scenes with a known
seasonal cycle, a known upward trend (+0.002 NDVI per month), a known drought (July 2023, 0.25 lower than normal), one
half-cloudy scene with a cloud mask and one corrupt file, so you can check every result against the truth.

### Steps

```python
from glob import glob
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries(sensor="sentinel2")                  # or band_map={"red": 3, "nir": 4, "blue": 1}
paths = sorted(glob("s2_*.tif"))                        # TIME ORDER is the order you give
series = ts.compute_index_series(paths, "ndvi", date_strings=["2021-01", "2021-02", ...], mask_paths=masks)
ts.detect_anomalies(series, method="mad", period=12)
ts.compute_trend(series, period=12)
ts.plot(series, save_path="ndvi.png")
```

### Band positions: say where your bands are

`sensor` selects a preset of **1-based band positions in your files** (not Sentinel-2 band names):

| Preset | blue | green | red | nir | swir | Meaning |
|---|---|---|---|---|---|---|
| `sentinel2` | 1 | 2 | 3 | 4 | 10 | A stack ordered B2, B3, B4, B8 …; SWIR is position 10, so `ndbi` only works on a 10+-band stack in that order |
| `landsat8` | 1 | 2 | 3 | 4 | 5 | A stack ordered B2, B3, B4, B5, B6 (blue … SWIR-1) |
| `planet` | 1 | 2 | 3 | 4 | – | Blue, green, red, NIR |
| `naip` | 3 | 2 | 1 | 4 | – | Red, green, blue, NIR |

If your files are ordered differently, give the positions yourself: `GeoTimeSeries(band_map={"red": 3, "nir": 4, "blue": 1})`.
An unknown `sensor` raises `ValueError`; an index that needs a band the map lacks (e.g. `ndbi` with `planet`) raises
`ValueError` naming the band. **A wrong band map produces a plausible but wrong index; check one scene by hand.**

### Indices

| `index` | Formula (reflectance 0–1) | Needs |
|---|---|---|
| `ndvi` | (NIR − Red) / (NIR + Red) | nir, red |
| `ndwi` | (Green − NIR) / (Green + NIR) | green, nir |
| `ndbi` | (SWIR − NIR) / (SWIR + NIR) | swir, nir |
| `evi` | 2.5 (NIR − Red) / (NIR + 6 Red − 7.5 Blue + 1) | nir, red, blue |
| `savi` | 1.5 (NIR − Red) / (NIR + Red + 0.5) | nir, red |

(EVI uses the **blue band**; earlier versions used a constant instead, which gave wrong values. More indices are in the
[index reference](../data/indices.md).)

### What counts as a valid pixel

A pixel is used only if it is **not** nodata (the file's own nodata value), not NaN, not masked by `mask_paths`, **not zero
in every band the index uses** (empty areas outside a swath, even when nodata is not declared), and its index is within
[−1, 1] (values outside are artefacts). `valid_fraction` reports the share used. Mean, std, P10 and P90 are computed over
these pixels only.

### Reflectance scale

The formulas need reflectance in 0–1. By default each image is divided by 10 000 if its bands are integers or its values go
above 2 (surface-reflectance products such as Sentinel-2 L2A and Landsat C2 scaled), and left alone if they are floats
already in 0–1. Override with `reflectance_scale=`; add `reflectance_offset=-1000` for Sentinel-2 processing baseline 04.00
and later, whose digital numbers carry a +1000 offset. NDVI barely depends on this (it is a ratio), but SAVI and EVI do.

### Cloud masks

`cloud_mask=True` used to be accepted and did nothing; it now gives a `DeprecationWarning`. Supply `mask_paths=[...]`, one
raster per image of the same size where a **non-zero** pixel is excluded (cloud, shadow, snow). Build them from your
product's quality layer (Sentinel-2 SCL, Landsat QA_PIXEL). Without masks, clouds enter the mean and look like a crash in NDVI.

## Expected output

### The series

```text
== without cloud mask ==
scene 2022-08 : 0.3776 valid 1.0
failed scenes : {'s2_2021-03.tif': "RasterioIOError: '/tmp/.../s2_2021-03.tif' not recog"}

== with mask_paths ==
scene 2022-08 : 0.7549 valid 0.5 (season says 0.755)
2023-07 mean/std/p10/p90: [0.5593, 0.0298, 0.5209, 0.5974]
```

`compute_index_series` returns:

```python
{"index": "ndvi", "dates": [...], "mean": [...], "std": [...], "p10": [...], "p90": [...],
 "valid_fraction": [...], "n_images": 36, "errors": {path: "ExceptionType: message"}}
```

* The cloudy August scene gave **0.378** without a mask (clouds drag NDVI to 0.05) and **0.755** with one, matching what the
  season predicts: the mask used half the pixels (`valid 0.5`) and recovered the right value.
* A scene that cannot be read appears as `None` in every list and its reason in `errors`, and the run continues. Always
  check `series["errors"]` and the `None`s before analysing.
* `None` also means "no valid pixel" (all cloud/nodata).

### Anomalies

```text
zscore            : []
mad               : []
zscore, period=12 : [('2023-07', -4.09, 'low')]
```

This is the most important result on the page. The drought (July 2023, −0.25) was **invisible** to a plain z-score and to
MAD, because the normal seasonal swing (NDVI between 0.3 and 0.8) is far larger than the anomaly. Only
`period=12` found it: each month is compared with the same month in other years, and the −4.09 says "4 standard deviations
below a typical July".

`detect_anomalies(series, method="zscore", threshold=None, period=None)`:

| Parameter | Meaning |
|---|---|
| `method="zscore"` | Distance from the mean in standard deviations; default threshold 2.5 |
| `method="mad"` | Modified z-score from the median and the median absolute deviation, which one outlier cannot distort; default threshold 3.5 |
| `threshold` | Absolute score above which a date is reported |
| `period` | Steps per seasonal cycle (12 monthly, 23 for 16-day data …). Needs at least two full cycles and two observations per cycle position |

Returns `[{"date", "value", "expected", "zscore", "type": "low"|"high"}]` in time order. `None` dates are skipped and never
shift the dates of the others.

Two properties to know: with *n* points a plain z-score can never exceed (*n*−1)/√*n*, so a threshold of 2.5 **cannot flag
anything in fewer than 9 points** (use `mad` or a lower threshold for short series); and fewer than 5 usable points
raises `ValueError` rather than answering "no anomalies".

### Trend

```text
plain fit         : {'slope': 0.001865, 'r_squared': 0.012, 'p_value': 0.5318, 'significant': False, 'direction': 'increasing'}
period=12 (seasonal): {'slope': 0.001077, 'r_squared': 0.963, 'p_value': 0.16, 'significant': False, 'direction': 'increasing'}
period=12, anomalies removed: {'slope': 0.002006, 'r_squared': 1.0, 'p_value': 0.0, 'significant': True, 'direction': 'increasing'}
true slope        : 0.002 per month
```

Three lessons from one dataset:

1. The **plain fit** says "increasing" but with p = 0.53 and R² = 0.01 it is noise: the seasonal swing swamps the line. Always
   read `p_value` / `significant` beside `direction`.
2. With **`period=12`** (one level per month plus a common slope) the cycle no longer hides the trend, but the drought,
   sitting at the end of the record, still pulls the slope down to 0.0011 (p = 0.16).
3. **Remove the flagged anomalies and fit again** and the slope is 0.002006: the true 0.002.

`compute_trend(series, stable_threshold=0.001, period=None)` returns `slope` (per time step), `intercept`, `r_squared`,
`p_value`, `significant` (p < 0.05), `direction` ("stable" when |slope| ≤ `stable_threshold`, a threshold on the **index**
units per step, so set it for your index and cadence) and, for the plain fit when all dates parse (`YYYY`, `YYYY-MM`,
`YYYY-MM-DD`, `YYYYMMDD`), `slope_per_year`. Missing dates leave gaps (they do not squeeze the line). Fewer than 3 usable
points gives `{"trend": "insufficient_data", "direction": "insufficient_data"}`.

### One location

```python
ts.extract_point(paths, lon=-0.12, lat=5.60, index="ndvi", date_strings=dates)
# -> {"index", "dates", "mean": [value or None per date], "location": {"lon", "lat"}, "n_images", "errors"}
```

The value of the single pixel containing the point (WGS84 longitude, latitude), `None` where the point is outside the
image, nodata or invalid. The result feeds `detect_anomalies`, `compute_trend` and `plot` directly. A single pixel is noisy:
for a field, prefer the mean of a clipped area.

### Plot

`ts.plot(series, title=None, save_path=None)` draws the mean with the P10–P90 band and returns the matplotlib figure; with
`save_path` the figure is written (and closed). A series with no data raises `ValueError("nothing to plot")`.

## The command line

```text
$ pygeovision timeseries analyze s2_2023-*.tif --dates 2023-01,2023-02,...,2023-12
  NDVI series: 12 images
     2023-01  0.2990   (valid pixels 100%)
     2023-02  0.3399   (valid pixels 100%)
     ...
     2023-06  0.7600   (valid pixels 100%)
     2023-07  0.3502   (valid pixels 100%)
     ...
  Trend: decreasing (slope=-0.004368 per step, R2=0.01, p=0.7691; not significant)

$ pygeovision timeseries anomaly s2_2023-*.tif --dates ... -m mad
  No anomalies detected.

$ pygeovision timeseries anomaly s2_2023-01.tif s2_2023-02.tif
Error: need at least 5 dates with data, got 2
```

(The twelve images were a single year of a seasonal cycle with one dry July: only a year of data, so July's dip is not
distinguishable from the seasonal swing, and the trend is not significant. Add years and `--period 12`.)

`analyze` options: `--index/-i` (ndvi, ndwi, evi, savi, ndbi), `--sensor/-s`, `--dates` (comma-separated labels, same number as
images), `--period`, `--output/-o` (writes the full series as JSON). `anomaly` options: `--threshold/-t`, `--method/-m`,
`--index`, `--sensor`, `--dates`, `--period`. **Images must be listed in time order** (shell globs sort alphabetically, which is
chronological for `YYYY-MM-DD` names). Unreadable images print a warning on standard error and `N/A`.

## What to do when…

| Situation | What to do |
|---|---|
| A value is `None` | The scene failed (see `series["errors"][path]`) or has no valid pixel. Check the path, the band count and the mask |
| `ValueError: unknown index` / `unknown sensor` / `needs band(s) […]` | Use an index from the table, a preset, or `band_map` |
| `ValueError: date_strings has N entries for M images` | One label per image, same order |
| NDVI values look wrong (near 0 or constant) | Wrong band positions (check `band_map` on one scene), or the file is not reflectance. Try `reflectance_scale=` |
| A cloudy date shows a dip | Provide `mask_paths`, or drop that date |
| Anomaly detection reports nothing, but you know a drought happened | The seasonal cycle is hiding it: pass `period` (12 for monthly). Plain `zscore` needs ≥ 9 points to flag anything at 2.5 |
| `ValueError: period=12 needs at least two full cycles` | You need ≥ 24 monthly dates; or drop `period` and analyse one season |
| Trend says "increasing" but `significant` is `False` | The data do not support a trend. Do not report the direction on its own |
| The trend changes a lot when one date is removed | One anomaly is pulling the line. Flag and remove anomalies, then re-fit (as in the example) |
| Different image sizes in the stack | Fine for `compute_index_series` (each image is summarised alone), but a different area per date makes the series incomparable. Clip all to the same polygon first |
| `ImportError … rasterio` | `pip install rasterio` |

## Verified here

Index values on scenes of known reflectance (including EVI with its blue band), nodata/NaN/all-zero handling, masks,
reflectance scale and offset, failure recording, anomaly dates staying aligned when a scene fails, the effect of `period`, trend
slope and p-value (plain and seasonal), the point sampler, the plot, and the CLI, all in `tests/test_timeseries_module.py` and
by running the script above.
