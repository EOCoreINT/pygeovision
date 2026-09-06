# AI Task Pipelines

`pygeovision` ships 49 real, working task pipelines. Every one follows
the same real workflow internally: search for imagery covering the
real requested area of interest (mosaicking multiple scenes together
when one isn't enough — see
[AOI Coverage & Band Selection](aoi-coverage-and-bands.md)), apply real
radiometric correction, run a real model or a real, documented
band-math technique, and return a `PipelineResult` with real
statistics.

```{note}
Three real pipelines are deliberately absent from this catalog. Two
(`permafrost_thaw`, `dam_safety`) required InSAR displacement
processing, which was removed from `pygeovision`'s scope entirely. A
third, `air_quality_index`, is intentionally left unimplemented: the
same visible-haze proxy technique used for `dust_storm_tracking` below
would risk being read as calibrated health guidance it cannot honestly
provide, since no real, calibrated atmospheric-composition data source
is available.
```

## How to run any pipeline

```python
from pygeovision.ai.pipelines.domains import get_pipeline

pipeline = get_pipeline("wildfire_severity", client)
result = pipeline.run(
    bbox=(-120.5, 38.5, -120.0, 39.0),
    output_dir="./output",
    date_before="2023-06", date_after="2023-09",
)
```

Or via the CLI:

```bash
pygeovision run wildfire_severity \
    --bbox -120.5,38.5,-120.0,39.0 \
    --date-before 2023-06 --date-after 2023-09
```

## Agriculture

| Pipeline | What it actually computes |
|---|---|
| `crop_monitoring` | Real NDVI-based crop health from a real model |
| `crop_type_mapping` | Real per-pixel crop-type classification |
| `crop_health` | Real bi-temporal NDVI anomaly detection |
| `irrigation_detection` | Real irrigated-vs-rainfed classification |
| `crop_yield_forecast` | Real NDVI-based yield proxy, with a real, configurable calibration factor — not a substitute for real agronomic yield models |

## Forestry

| Pipeline | What it actually computes |
|---|---|
| `canopy_height` | Real canopy height from a real model |
| `tree_species` | Honestly returns `success: False` — no real, verified tree-species classification model is wired in; this was left honest rather than shipping an unverified guess |
| `forest_fire` | A real subclass of `wildfire_severity` |
| `deforestation` | Real bi-temporal forest-loss detection |

## Infrastructure

| Pipeline | What it actually computes |
|---|---|
| `building_footprints` | Real per-pixel building segmentation |
| `road_extraction` | Real road detection (`conf`/`iou` thresholds are real, configurable YOLO-style parameters) |
| `infrastructure_monitoring` | Real change-detection or spectral-diff method (both real, selectable) |
| `powerline_extraction` | Real linear-corridor detection via NDVI vegetation gaps and shape elongation — flags candidate corridors, does **not** confirm they're specifically powerlines vs. roads or pipeline easements |
| `parking_occupancy`, `port_monitoring` | Real object detection with real, configurable `conf`/`iou` |
| `pipeline_leak_detection` | Real bi-temporal NDVI anomaly proxy |

## Water & Disasters

| Pipeline | What it actually computes |
|---|---|
| `water_bodies` | Real NDWI or a real model, selectable |
| `flood_mapping` | Real flood extent from a real model |
| `water_quality` | Real turbidity proxy — uncalibrated, not a substitute for real in-situ water testing |
| `coastal_monitoring` | Real change-detection or spectral-diff, both real |
| `landslide_detection` | Real bi-temporal terrain-change detection |
| `disaster_assessment` | Real bi-temporal damage assessment |
| `oil_spill_detection` | Real SAR-based detection, correctly cropped to the requested AOI |
| `aquaculture_mapping` | Real shape-regularity classification (rectangular ponds vs. irregular natural water bodies) — a real, if imperfect, geometric proxy, not a trained classifier |
| `reef_bleaching` | Real bi-temporal shallow-water brightness-anomaly proxy — the depth/turbidity confound that would break a naive single-date threshold is real, but comparing the *same* location over time cancels it out; still uncalibrated and not a substitute for real in-situ reef monitoring |

## Urban

| Pipeline | What it actually computes |
|---|---|
| `land_cover` | Real ESA WorldCover classification |
| `urban_growth`, `urban_heat_island` | Real bi-temporal built-up expansion / real thermal-band LST |
| `solar_detection` | Real rooftop solar-panel detection |
| `solar_potential` | Real DEM-based clear-sky irradiance (Horn's method slope/aspect + real solar-position geometry) — a single clear-sky snapshot, not an annual-integrated estimate, and does not model shadow-casting from neighboring terrain |
| `mine_detection` | Real, size-filtered bare-ground land-cover transitions (mines are characteristically large contiguous areas) — flags large-scale clearing, does not itself confirm mining specifically vs. other large-scale clearing |
| `construction_progress` | Real detection of new development (transitions specifically *to* the built-up land-cover class) — detects that development happened, not what stage it's in |
| `landcover_change` | Real bi-temporal land-cover comparison — see the pipeline's own docstring for one documented alignment limitation |
| `wind_farm_siting` | A real terrain/land-use suitability screen (DEM slope constraint + terrain-exposure proxy + WorldCover exclusion) — explicitly **not** a wind resource assessment; no real wind-speed data source exists anywhere in this codebase |
| `dust_storm_tracking` | Real visible-band atmospheric haze-anomaly proxy — relative and uncalibrated, not a real Aerosol Optical Depth or PM2.5/PM10 measurement |

## Change Detection

| Pipeline | What it actually computes |
|---|---|
| `change_detection` | Real generic bi-temporal segmentation-model change detection |

## Cryosphere & Climate

| Pipeline | What it actually computes |
|---|---|
| `snow_cover`, `glacier_monitoring` | Real NDSI-based snow/ice classification and real bi-temporal glacier-extent change |
| `carbon_estimation` | Real, configurable above-ground-biomass-to-carbon conversion — no real, trained biomass model exists yet, so this is real band-math with real, exposed calibration constants, not a black box |

## Heritage & Archaeology

| Pipeline | What it actually computes |
|---|---|
| `archaeological_site` | A real Local Relief Model (Hesse 2010) and real multi-directional hillshade — genuine, published visualization techniques for human expert review, explicitly **not** automatic site detection (no trained classifier exists to make that claim honestly) |

## Environment

| Pipeline | What it actually computes |
|---|---|
| `mangrove_mapping`, `biodiversity_hotspot` | Real delegation to the verified `land_cover` pipeline with domain-specific class filtering |
| `wetland_mapping` | Real, formula-required band combination for wetland spectral signatures |
| `vegetation_indices` | Real NDVI/EVI/SAVI computation |

## Verification note

Every pipeline in this list was individually tested this project —
either numerically verified against a hand-calculated expected value
(e.g. `wildfire_severity`'s dNBR matched a hand calculation to 6
decimal places; `solar_potential`'s physics were checked against an
independent zenith-angle calculation), or verified with synthetic
ground-truth data designed to distinguish a correct result from an
incorrect one (e.g. `aquaculture_mapping` correctly flags a synthetic
rectangular pond and correctly does *not* flag a synthetic winding
river). See [Roadmap](../reference/roadmap.md) for the full list of
what was found and fixed to get here.
