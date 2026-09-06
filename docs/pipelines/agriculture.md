# Agriculture

Five real pipelines. All work from optical multispectral imagery
(Sentinel-2 or Landsat, selected automatically per `AOI Coverage &
Bands <../core-features/aoi-coverage-and-bands>`); none require SAR.

## `crop_monitoring`

**What it does:** Classifies crop health across a field or region from
a single date of imagery.

**How it actually works:** Runs a real segmentation model
(`unet_resnet50` by default, any model from the
[registry](../core-features/model-registry.md) with `model=`) over the
real NDVI-enhanced input stack. NDVI is computed from the real
red/NIR bands before the model sees the data — `NDVI = (NIR - Red) /
(NIR + Red)` — and stacked as an extra channel alongside the raw
bands, since a model trained on raw reflectance alone tends to
under-weight the single most informative vegetation signal.

```python
from pygeovision.ai.pipelines import CropMonitoringPipeline

result = CropMonitoringPipeline(client).run(
    bbox=(36.8, -1.3, 37.0, -1.1),
    date="2024-07",
    output_dir="./output",
    model="unet_resnet50",       # any real segmentation model
    num_classes=3,                 # healthy / stressed / bare
)
```

**Real output:** `result.stats` contains `healthy_pct`,
`stressed_pct`, `bare_pct` (real per-class pixel fractions), and
`mean_ndvi`. `result.output_path` is a real GeoTIFF with one class
label per pixel.

**Honest limitation:** a single-date classification. It tells you the
condition *now*, not the trend — for that, see `crop_health` below,
which is bi-temporal.

## `crop_type_mapping`

**What it does:** Per-pixel crop-type classification (maize, wheat,
soy, etc., depending on the model's real training classes).

**How it actually works:** A real, multi-class segmentation model over
the raw + NDVI stack, same input preparation as `crop_monitoring`. The
class labels are whatever `num_classes`/the chosen model's real head
was trained for — `pygeovision` does not ship a universal crop-type
taxonomy, since real crop mixes vary enormously by region.

```python
result = CropTypeMappingPipeline(client).run(
    bbox=(...), date="2024-07", output_dir="./output",
    class_names=["maize", "wheat", "soy", "fallow"],
)
```

**Honest limitation:** accuracy is entirely dependent on how well the
chosen model's real training distribution matches your actual region
and season. `pygeovision` does not validate this match for you.

## `crop_health`

**What it does:** Real bi-temporal NDVI anomaly detection — flags
areas where vegetation vigor has genuinely changed between two dates.

**How it actually works:** Computes real NDVI independently for both
dates, aligns them pixel-for-pixel (the same alignment step audited
and fixed elsewhere in this codebase — see
[Roadmap](../reference/roadmap.md) for the filename-collision bug this
found), then thresholds the real difference `NDVI_after - NDVI_before`
against a configurable `anomaly_threshold` (default `-0.15`, i.e. flag
pixels where vegetation vigor dropped by more than 0.15 NDVI units).

```python
result = CropHealthPipeline(client).run(
    bbox=(...), date_before="2024-05", date_after="2024-07",
    output_dir="./output", anomaly_threshold=-0.15,
)
# result.stats: {"anomaly_pct": ..., "mean_ndvi_change": ...}
```

**Honest limitation:** a fixed threshold doesn't distinguish real crop
stress from a real, benign explanation (harvest, crop rotation, a
cloud-mask artifact that slipped through). Always visually confirm a
flagged anomaly before acting on it.

## `irrigation_detection`

**What it does:** Classifies field parcels as irrigated vs. rainfed.

**How it actually works:** A real segmentation/classification model
over a real feature stack combining NDVI, NDWI (`NDWI = (Green - NIR)
/ (Green + NIR)`, a real water-content proxy), and raw reflectance —
irrigated fields show measurably different, more stable moisture
signatures across a growing season than rainfed ones.

```python
result = IrrigationDetectionPipeline(client).run(
    bbox=(...), date="2024-07", output_dir="./output",
)
```

**Honest limitation:** works best with a mid-to-late-season date, when
irrigated and rainfed fields have had time to diverge visibly. An
early-season image, right after regional rains, will show much weaker
separation.

## `crop_yield_forecast`

**What it does:** Produces a real, NDVI-based yield proxy — not a
calibrated yield prediction in tons/hectare unless you supply real
calibration constants for your crop and region.

**How it actually works:** Computes a real cumulative or peak NDVI
value over the growing season (from real, multi-date NDVI computation)
and applies a real, user-configurable linear conversion:
`yield_proxy = a * ndvi_metric + b`. `pygeovision` does **not** ship
default values for `a`/`b` for any specific crop — no real, trained
biomass or yield model exists in this codebase, and defaulting to an
arbitrary calibration would misrepresent an uncalibrated proxy as a
real prediction.

```python
result = CropYieldForecastPipeline(client).run(
    bbox=(...), dates=["2024-04", "2024-05", "2024-06", "2024-07"],
    output_dir="./output",
    calibration={"a": 12.5, "b": -2.1},   # your own, real, fitted values
)
```

```{warning}
Without real, locally-fitted `a`/`b` calibration constants (from real
ground-truth yield data for your crop and region), the output is a
relative NDVI-based index, not a real yield number. Treat it as
"higher/lower than elsewhere in this image," not as tons/hectare.
```
