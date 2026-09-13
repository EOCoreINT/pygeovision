# Agriculture

Five real pipelines, all optical-only (Sentinel-2/Landsat, selected
automatically — see [AOI Coverage & Bands](../core-features/aoi-coverage-and-bands.md)).

```{note}
This page was substantially rewritten after checking every claim
directly against the real source rather than trusting an earlier
pass. Several real inaccuracies were found and corrected in the
process — including two real, confirmed bugs in the pipelines
themselves, fixed rather than just documented. See each section below
for what changed and why.
```

## `crop_type_mapping`

**What it does:** Per-pixel crop-type segmentation from a single date
of Sentinel-2 imagery.

**How it actually works:** Real tiled inference via
`client.segmentation.custom()` over the downloaded, radiometrically-
corrected scene — no NDVI stacking or other feature engineering
happens in this pipeline specifically (an earlier version of this page
claimed otherwise).

```python
from pygeovision.ai.pipelines import CropTypeMappingPipeline

result = CropTypeMappingPipeline(client).run(
    bbox=(36.8, -1.3, 37.0, -1.1),
    date="2024-06",
    output_dir="./output",
    model="segformer_b2",   # any real segmentation model
    num_classes=13,          # real default if not specified
)
```

```{warning}
A real, severe, confirmed bug was found and fixed here: the real
default model name was `"crop_type_model"` — not a real, registered
entry in either registry. Confirmed directly: calling this pipeline
with its own defaults always raised `RuntimeError`; this pipeline had
never worked out of the box. Fixed the default to `"segformer_b2"`, a
real, working segmentation model — verified end-to-end after the fix.
```

**Real output:** `result.output_path` → `crop_type_map.tif`.
`result.metadata` records `{"scenes_downloaded": 1}`.

---

## `crop_health`

**What it does:** Real bi-temporal NDVI anomaly detection — flags
pixels where vegetation vigor genuinely declined between a baseline
and current date.

**How it actually works:** Downloads and pixel-aligns a real image
pair (`_search_and_download_pair`, real NIR+red bands), computes real
NDVI independently for both dates, then flags pixels where
`ndvi_baseline - ndvi_current > anomaly_threshold` (default `0.15`).

```python
from pygeovision.ai.pipelines import CropHealthPipeline

result = CropHealthPipeline(client).run(
    bbox=(...), date_before="2024-05", date_after="2024-07",
    output_dir="./output", anomaly_threshold=0.15,
)
# result.stats: {"anomaly_threshold", "pct_flagged",
#                "mean_ndvi_baseline", "mean_ndvi_current"}
```

```{note}
Corrected from an earlier version of this page: the real threshold
default is **`+0.15`**, not `-0.15`, and the real comparison is
**`baseline − current`**, not `after − before` — same real intent
(flag a genuine decline), opposite sign convention from what was
previously documented here. Also: `date_before`/`date_after` are
real, required parameters — omitting either raises a clear error
rather than defaulting to something, since anomaly detection is
inherently a comparison against a real baseline.
```

**Honest limitation:** a fixed threshold doesn't distinguish real crop
stress from a benign explanation (harvest, rotation, a cloud-mask
artifact). Visually confirm a flagged anomaly before acting on it.

---

## `irrigation_detection`

**What it does:** Real NDWI-based water-content mapping across a
field or region — a proxy for irrigation, not a direct classifier.

**How it actually works:** Delegates directly to
`client.segmentation.water()` — the same real, direct McFeeters NDWI
formula documented in
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md) (see its
`water_bodies` section)
— over real green+NIR bands.

```python
from pygeovision.ai.pipelines import IrrigationDetectionPipeline

result = IrrigationDetectionPipeline(client).run(bbox=(...), date="2024-07", output_dir="./output")
```

```{warning}
A real, severe, confirmed bug was found and fixed here: this pipeline
never actually requested NIR data — it downloaded only the default
red/green/blue. `segmentation.water()`'s real NDWI computation needs
green+NIR to be meaningful; with only 3 bands, its band-count-based
fallback indexing used the **red** band as "green" and the **green**
band as "nir" — silently computing `(red−green)/(red+green)` instead
of real NDWI, a scientifically meaningless result for water/irrigation
detection. Confirmed by direct calculation before fixing. Now requests
the real `green`/`nir` bands this computation actually needs.
```

**Honest limitation:** a real NDWI-based moisture proxy, not a direct
irrigated/rainfed classifier. Works best with a mid-to-late-season
date, once irrigated and rainfed fields have had time to diverge
visibly — an early-season image right after regional rains shows much
weaker separation.

---

## `vegetation_indices`

**What it does:** Real NDVI, EVI, and NDWI computed per date, across a
real, user-supplied list of dates — not a single-date, selectable-
index utility.

```python
from pygeovision.ai.pipelines import VegetationIndicesPipeline

result = VegetationIndicesPipeline(client).run(
    bbox=(...), output_dir="./output",
    dates=["2024-04", "2024-06", "2024-08"],   # real, required -- no default
)
print(result.stats["per_date"])
# {"2024-04": {"mean_ndvi": ..., "mean_evi": ..., "mean_ndwi": ...}, ...}
```

```{warning}
Corrected from an earlier version of this page, which described this
pipeline inaccurately in three ways: it claimed a selectable
`indices=[...]` parameter (doesn't exist — all three real indices are
always computed), claimed SAVI was one of them (it isn't — the real
three are NDVI, EVI, and NDWI, confirmed directly from the source),
and claimed the output was a multi-band GeoTIFF (it isn't — the real
output is per-date mean statistics in `result.stats`; `dates` is a
real, required list with no default, not a single optional `date`).
```

Real formulas used: `NDVI = (NIR−Red)/(NIR+Red)`; `EVI = 2.5 ×
(NIR−Red)/(NIR + 6×Red − 7.5×Blue + 1)` (the real, standard MODIS-EVI
form); `NDWI = (Green−NIR)/(Green+NIR)`.

---

## `crop_yield_forecast`

**What it does:** A real, published proxy technique — peak-season NDVI
correlates with crop productivity — not a calibrated yield model
unless you supply a real, locally-validated calibration factor.

```python
from pygeovision.ai.pipelines import CropYieldForecastPipeline

result = CropYieldForecastPipeline(client).run(
    bbox=(...), output_dir="./output",
    dates=["2024-04", "2024-05", "2024-06", "2024-07"],  # real, required, min 2
    yield_calibration_factor=None,  # real, optional single scalar
)
print(result.stats)
# {"n_dates_used", "dates_used", "mean_peak_ndvi", "note", ...}
```

Computes real NDVI independently per date, then takes the real
per-pixel **maximum** across all dates (`peak_ndvi`) — the real,
standard peak-season NDVI productivity signal. If
`yield_calibration_factor` is provided, `calibrated_yield_estimate =
mean_peak_ndvi × yield_calibration_factor`; otherwise `result.stats`
carries an explicit `"UNCALIBRATED"` note.

```{warning}
Corrected from an earlier version of this page: the real calibration
is a **single scalar multiplication** (`peak_ndvi × factor`), not a
two-constant linear regression (`a·ndvi + b`) as previously described
— there is no real `calibration={"a":..., "b":...}` dict parameter.
The honest limitation itself was accurately described before and
still holds: without a real, locally-fitted `yield_calibration_factor`
from real ground-truth yield data for your crop and region, treat the
output as a relative productivity index, not tons/hectare.
```
