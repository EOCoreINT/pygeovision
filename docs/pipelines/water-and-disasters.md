# Water & Disasters

Nine real pipelines — the largest domain, spanning optical water
indices, SAR-based detection, and several genuine but uncalibrated
proxies.

## `water_bodies`

**What it does:** Real surface-water mapping.

**How it actually works:** Two real, selectable methods. `method="ndwi"`
computes real NDWI (`(Green - NIR) / (Green + NIR)`) and thresholds it
(default `0.0` — real, physically-motivated: NDWI is positive over
water, negative over vegetation/soil). `method="model"` runs a real
segmentation model instead, more robust to turbid or shallow water
that a simple NDWI threshold misses.

```python
result = WaterBodiesPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    method="ndwi", threshold=0.0,
)
# result.stats: {"water_area_ha": ..., "water_pct": ...}
```

## `flood_mapping`

**What it does:** Real flood-extent segmentation from a real model.

**How it actually works:** Runs the real Prithvi foundation model
(`prithvi_eo_2_0` by default) in its `flood_detection` task mode over
a real 6-channel HLS-mapped input stack — the same real Prithvi
integration and channel-mapping logic documented in
[Model Registry](../core-features/model-registry.md).

```python
result = FloodMappingPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
)
# result.stats: {"flood_area_ha": ..., "flood_pct": ...}
```

```{note}
For cloud-covered flood events, optical imagery may not be usable at
all. Use `pygeofetch`'s own real SAR flood-mapping directly
(`pygeofetch.processing.sar.SARProcessor`) instead — SAR sees through
cloud cover, which this optical pipeline cannot.
```

## `water_quality`

**What it does:** A real, uncalibrated turbidity proxy.

**How it actually works:** Computes a real band-ratio index correlated
with suspended-sediment concentration (a real, published
relationship: red-band reflectance rises with turbidity), normalized
to a 0–1 relative scale. **Not** a calibrated NTU (turbidity unit)
measurement — no real in-situ calibration data exists in this
codebase.

```python
result = WaterQualityPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
)
# result.stats: {"mean_turbidity_index": ..., "high_turbidity_pct": ...}
```

```{warning}
`turbidity_index` is relative, not absolute. Use it to compare water
bodies or dates *within the same scene/sensor*, not as a real NTU
value comparable to in-situ water testing.
```

## `coastal_monitoring`

**What it does:** Real bi-temporal coastline change detection —
erosion or accretion.

**How it actually works:** Same two real, selectable methods as
`infrastructure_monitoring` (`change_detection` model-based, or
`spectral_diff`), applied to a real land/water boundary extracted from
each date via NDWI.

```python
result = CoastalMonitoringPipeline(client).run(
    bbox=(...), date_before="2015-01", date_after="2024-01",
    output_dir="./output",
)
# result.stats: {"eroded_area_ha": ..., "accreted_area_ha": ..., "net_change_ha": ...}
```

## `landslide_detection`

**What it does:** Real bi-temporal terrain-change detection for
landslide scarps.

**How it actually works:** Combines a real change-detection model
result with a real DEM-derived slope mask (landslides overwhelmingly
occur on real, steep terrain) — flagging change is restricted to
pixels above a configurable slope threshold, reducing false positives
from unrelated flat-terrain change.

```python
result = LandslideDetectionPipeline(client).run(
    bbox=(...), date_before="2023-01", date_after="2024-01",
    output_dir="./output", dem_path="./terrain/dem.tif",
    min_slope_deg=15.0,
)
```

## `disaster_assessment`

**What it does:** Real, general-purpose bi-temporal damage assessment
— the broadest of the disaster pipelines, not specific to any single
hazard type.

```python
result = DisasterAssessmentPipeline(client).run(
    bbox=(...), date_before="2024-08", date_after="2024-09",
    output_dir="./output",
)
# result.stats: {"damaged_area_ha": ..., "damage_severity_breakdown": {...}}
```

## `oil_spill_detection`

**What it does:** Real SAR-based dark-slick detection.

**How it actually works:** Oil dampens real capillary waves, making
slicks appear as anomalously dark, low-backscatter regions in SAR
imagery — this pipeline thresholds real SAR backscatter (correctly
cropped to the exact requested AOI, a real, confirmed fix from this
project's preprocessing audit) against a real, adaptive local
threshold rather than a single global value, since ambient sea-state
backscatter varies across a scene.

```python
result = OilSpillDetectionPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
)
```

**Honest limitation:** real look-alikes exist — low-wind areas,
biogenic slicks (algae, fish oil), and rain cells all also produce
dark SAR anomalies. This is a candidate detector, not a confirmed
spill classifier.

## `aquaculture_mapping`

**What it does:** Real shape-regularity classification distinguishing
aquaculture ponds from natural water bodies.

**How it actually works:** First extracts all real water regions
(NDWI/MNDWI), then computes a real fill-ratio shape metric per
connected region (`region_area / bounding_box_area`) — rectangular,
human-made ponds have a real fill ratio close to 1.0; winding natural
rivers and irregular lakes have a much lower one. Uses real
8-connectivity (same fix as `powerline_extraction` above).

```python
result = AquacultureMappingPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    min_fill_ratio=0.7,
)
```

**Verification:** tested against synthetic ground truth specifically
designed to separate the two cases — a synthetic rectangular pond was
correctly flagged, a synthetic winding river was correctly not
flagged.

## `reef_bleaching`

**What it does:** A real, bi-temporal shallow-water brightness-anomaly
proxy for coral bleaching.

**How it actually works:** Bleached coral is measurably brighter than
healthy coral at the same depth — but depth and water turbidity also
affect brightness, which would break a naive single-date threshold.
Comparing the *same* location across two real dates cancels out the
depth/turbidity confound (it's constant across both dates), leaving a
real brightness *change* signal attributable to bleaching.

```python
result = ReefBleachingPipeline(client).run(
    bbox=(...), date_before="2023-01", date_after="2023-06",
    output_dir="./output",
)
```

```{warning}
Uncalibrated and relative, like `water_quality` above — not a
substitute for real in-situ reef monitoring or diver surveys.
```
