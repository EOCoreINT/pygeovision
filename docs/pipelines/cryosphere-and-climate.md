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

```{note}
`carbon_estimation` is one of the 10 CLI-reachable pipelines with a
separate, real implementation — see
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md) for the
real, correct formula and parameters (a quadratic NDVI relationship
with `agb_scale`/`agb_max`/`carbon_fraction`, not the linear
calibration dict described in an earlier version of this page). The
honest core message is the same either way: without real, local
calibration, this is an uncalibrated proxy, not a validated carbon
stock estimate.
```
