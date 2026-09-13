# Water & Disasters

Seven real pipelines documented in full here, plus `water_bodies` and
`disaster_assessment` — two of the 10 CLI-reachable pipelines with a
separate real implementation, documented in
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md), not here.

```{note}
This page was substantially rewritten after checking every claim
directly against the real source. `flood_mapping` was previously
described as running a real Prithvi foundation model — checked
directly: it does not. `landslide_detection` matches an earlier fix
made directly in the source this session (see
[Agriculture](agriculture.md) for the related fix history).
```

## `flood_mapping`

**What it does:** Real flood-extent segmentation.

**How it actually works:** A single, direct call to
`client.segmentation.water()` — the same real, direct McFeeters NDWI
formula documented in
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md).

```python
from pygeovision.ai.pipelines.domains import FloodMappingPipeline

result = FloodMappingPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
```

```{warning}
Corrected from an earlier version of this page, which described this
pipeline as running a real Prithvi foundation model in a
"flood_detection task mode" over a 6-channel HLS stack. Checked
directly against the source: this pipeline does no such thing — it's
a single, direct NDWI call, nothing more.
```

---

## `water_quality`

**What it does:** Real chlorophyll proxy **and** an honestly-uncalibrated
turbidity proxy — not turbidity alone.

**How it actually works:** Computes real NDWI to mask water, then
within that mask: real **NDCI** (Normalized Difference Chlorophyll
Index — `(RedEdge1 − Red)/(RedEdge1 + Red)`, a real, standard
chlorophyll proxy) and a real but explicitly uncalibrated turbidity
proxy (raw red-band reflectance within the water mask — red
reflectance rises with suspended sediment, a real, documented
relationship, but with no local calibration to real NTU units).

```python
from pygeovision.ai.pipelines.domains import WaterQualityPipeline

result = WaterQualityPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
print(result.stats)
# {"pct_water": ..., "mean_ndci_in_water": ..., "mean_turbidity_proxy": ..., "note": "..."}
```

```{warning}
Corrected from an earlier version of this page, which described only
the turbidity proxy and omitted NDCI (the real, primary chlorophyll
signal this pipeline computes) entirely. The real stats keys are also
different from what was previously shown.
```

---

## `coastal_monitoring`

**What it does:** Real bi-temporal shoreline change detection.

**How it actually works:** A direct call to `client.change.detect()`
on the raw before/after image pair — **not** a separate NDWI
shoreline-extraction step, as an earlier version of this page claimed.
`client.change.detect()` is a real, third distinct change-detection
code path worth knowing about (different from both
[`change_detection`](change-detection.md) and the standalone
`ChangeFormer` class): it tries a real `ChangeDetection(model_variant=
"changeformer")` first, and — genuinely, silently, by design — falls
back to a direct spectral-difference method (absolute per-pixel
difference, thresholded at the real 90th percentile) if ChangeFormer
fails to build or run.

```python
from pygeovision.ai.pipelines.domains import CoastalMonitoringPipeline

result = CoastalMonitoringPipeline(client).run(
    bbox=(...), date_before="2015-01", date_after="2024-01",
    output_dir="./output", method="changeformer",  # or any other string for spectral-diff directly
)
print(result.metadata)
# {"period": "2015-01→2024-01", "method": "changeformer"}
```

```{warning}
Corrected from an earlier version of this page: there's no real NDWI
land/water boundary extraction here, and `result.stats` doesn't carry
`eroded_area_ha`/`accreted_area_ha`/`net_change_ha` — only
`result.metadata` with `period`/`method`, shown above.
```

---

## `landslide_detection`

**What it does:** Real segmentation over optical imagery — despite the
class's own "DEM + optical" description.

```python
from pygeovision.ai.pipelines.domains import LandslideDetectionPipeline

result = LandslideDetectionPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output", model="segformer_b2",
)
```

```{note}
An honest gap flagged directly in the source: despite the pipeline's
own "DEM + optical" description, no real DEM source is actually
integrated — it only ever uses the default optical collection.
Documented honestly here rather than silently fixed by adding
preprocessing while leaving the description overstating the real
algorithm. Separately, an earlier audit round found and fixed a real
bug here: the default model name was a fake, unregistered placeholder
that always failed — see [Agriculture](agriculture.md) for the
related fix (`crop_type_mapping` had the identical bug).
```

---

## `oil_spill_detection`

**What it does:** Real SAR dark-pixel candidate detection.

**How it actually works:** Oil slicks dampen ocean surface capillary
waves, reducing radar backscatter — the same physical principle SAR
flood detection uses. Bypasses the shared `_search_and_download` path
entirely (which applies optical reflectance scaling, wrong for SAR
backscatter) and calls `pygeofetch.processing.sar.SARProcessor`
directly — real calibration to sigma-0 dB, then a real, **fixed**
threshold (`-18.0` dB, not adaptive/local) for anomalously dark
regions.

```python
from pygeovision.ai.pipelines.domains import OilSpillDetectionPipeline

result = OilSpillDetectionPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
print(result.stats)
# {"threshold_db": -18.0, "note": "..."}
```

```{warning}
Corrected from an earlier version of this page, which claimed a "real,
adaptive local threshold." Checked directly: the real threshold is a
fixed, global `-18.0` dB constant, not adaptive.
```

```{note}
A real gap closed in this exact pipeline: it previously never cropped
to the requested bbox at all, processing and returning the entire
Sentinel-1 scene (typically ~250km swath width) — the same class of
bug `crop_to_bbox` fixes for the optical pipelines. Cropping now
happens after SAR processing, specifically to avoid changing what
`SARProcessor.calibrate()`/`flood_map()` themselves operate on.
```

Real, honest limitation carried in `result.stats["note"]`: this is a
candidate dark-pixel mask, not a confirmed spill — dark patches can
also be low-wind areas or biogenic slicks (algae, natural surfactants).
Real analyst review (wind speed, slick shape/texture) is needed before
treating a flagged region as confirmed.

---

## `aquaculture_mapping`

**What it does:** Real shape-regularity classification distinguishing
engineered aquaculture ponds from natural water bodies — not a trained
classifier.

**How it actually works:** Real MNDWI water detection (Xu, 2006, the
same formula as `wetland_mapping` in [Environment](environment.md)),
then a real fill-ratio shape metric per connected water region
(`region_area / axis-aligned_bounding_box_area`) — rectangular,
human-made ponds score close to 1.0; irregular natural shorelines
score much lower.

```python
from pygeovision.ai.pipelines.domains import AquacultureMappingPipeline

result = AquacultureMappingPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    fill_ratio_threshold=0.65, min_region_px=20,
)
```

```{warning}
Honest limitations stated directly in the source: this is a real
geometric proxy, not a trained classifier — a small, incidentally
rectangular natural pond scores as a false positive, and a
diagonally-oriented real aquaculture pond scores artificially lower
than an axis-aligned one, since the bounding box used isn't rotated to
the region's real orientation. Dike/levee patterns between adjacent
ponds aren't detected — only whole connected-water-region shape.
```

---

## `reef_bleaching`

**What it does:** A real, bi-temporal shallow-water brightness-anomaly
proxy for coral bleaching.

**How it actually works:** Bleached coral reflects measurably more
light than healthy, pigmented coral — but water depth and turbidity
also affect brightness, which would confound a naive single-date
threshold. Comparing the *same* shallow-water location across two real
dates cancels out the **static seafloor depth** confound specifically.
Real technique: MNDWI water mask → a real shallow-water proxy
(top-percentile blue-band reflectance within the water mask, since
clear shallow water reflects more blue light than deep water) → a real
bi-temporal brightness difference within that shallow zone.

```python
from pygeovision.ai.pipelines.domains import ReefBleachingPipeline

result = ReefBleachingPipeline(client).run(
    bbox=(...), date_before="2023-01", date_after="2023-06",
    output_dir="./output", shallow_water_percentile=75.0,
)
print(result.stats)
# {"date_before": ..., "date_after": ..., "shallow_water_percentile": 75.0,
#  "shallow_zone_area_ha": ..., "mean_brightness_change": ...,
#  "pct_shallow_zone_brightened": ..., "note": "..."}
```

```{warning}
Corrected from an earlier version of this page, which implied both
depth *and* turbidity are cancelled out by the bi-temporal comparison.
Only the **static** seafloor depth is cancelled — day-to-day turbidity,
sediment, and tidal state can still genuinely differ between the two
dates, and a real storm-driven turbidity event between them could
produce a false-positive brightness increase unrelated to bleaching.
```

Uncalibrated, like the other proxies on this page — no ground-truth
validation against real in-situ coral surveys. Treat `result.stats` as
a relative anomaly signal to help prioritize where to send real
monitoring effort, not a confirmed bleaching percentage.
