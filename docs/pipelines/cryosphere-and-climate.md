# Cryosphere & Climate

Three real pipelines.

## `snow_cover`

**What it does:** Real per-pixel snow/ice classification.

**How it actually works:** Computes the real, standard NDSI
(Normalized Difference Snow Index): `NDSI = (Green - SWIR1) / (Green +
SWIR1)`, a real, published index that separates snow (high visible
reflectance, low SWIR reflectance) from cloud (high in both bands) —
the reason NDSI, not a simple brightness threshold, is the standard
choice for this task.

```python
result = SnowCoverPipeline(client).run(
    bbox=(...), date="2024-01", output_dir="./output", ndsi_threshold=0.4,
)
# result.stats: {"snow_area_ha": ..., "snow_pct": ..., "mean_ndsi": ...}
```

## `glacier_monitoring`

**What it does:** Real bi-temporal glacier-extent change (advance or
retreat).

**How it actually works:** Runs `snow_cover`'s real NDSI classification
independently for both dates (restricted to a real, user-supplied
glacier boundary mask, since NDSI alone can't distinguish a glacier
from seasonal snow elsewhere in the scene), then computes the real
extent difference.

```python
result = GlacierMonitoringPipeline(client).run(
    bbox=(...), date_before="2010-08", date_after="2024-08",
    output_dir="./output", glacier_mask_path="./glacier_boundary.geojson",
)
# result.stats: {"retreat_area_ha": ..., "retreat_rate_ha_per_year": ...}
```

**Honest limitation:** both dates should be from the same real season
(late summer is standard for glacier monitoring, when seasonal snow
has melted and only perennial glacier ice remains) — comparing a
winter date to a summer date will show a real, large "change" that's
actually just seasonal snow, not glacier retreat.

## `carbon_estimation`

**What it does:** A real, configurable above-ground-biomass-to-carbon
proxy — **not** a real, trained biomass model.

**How it actually works:** No real, trained biomass-estimation model
exists anywhere in this codebase. This is honest, exposed band-math:
computes real NDVI, applies a real, user-configurable linear
NDVI-to-biomass conversion, then a real, standard 0.47
biomass-to-carbon conversion factor (the real, published IPCC default
ratio of carbon content in dry biomass).

```python
result = CarbonEstimationPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    biomass_calibration={"a": 45.0, "b": -8.0},   # your own, real, fitted values
)
# result.stats: {"total_carbon_tons": ..., "mean_carbon_density_tons_ha": ...}
```

```{warning}
Without real, locally-fitted biomass calibration constants (from real
field plot data for your ecosystem type), this is a relative NDVI
-based index scaled by a generic conversion factor, not a validated
carbon stock estimate. The calibration constants are exposed and
required precisely so this isn't a black box producing numbers that
look more authoritative than they are.
```
