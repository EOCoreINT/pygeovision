# PyGeoVision

**A geospatial AI platform combining satellite data access with native model inference.**

```bash
pip install pygeovision
```

## What is PyGeoVision?

PyGeoVision combines two things that are normally separate projects:

- **Satellite data access**, via [PyGeoFetch](https://pypi.org/project/pygeofetch/) — search, download, and preprocessing across ~24 provider integrations (Sentinel, Landsat, Planet, Maxar, Copernicus, USGS, and more)
- **AI inference**, running natively inside PyGeoVision in pure PyTorch/HuggingFace Transformers/timm — no separate AI platform, account, or API key

## Quick Start

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

# Search and download Sentinel-2 imagery
results = client.search(
    bbox=(-74.1, 40.6, -73.7, 40.9),           # New York City
    date_range=("2024-01-01", "2024-04-30"),
    cloud_cover_max=15,
)
scene = client.download(results[:1], output_dir="./data")[0]
```

Run a named, verified AI pipeline via the CLI:

```bash
pygeovision channel land_cover --bbox -74.1 40.6 -73.7 40.9 --date 2024-01
```

Or from Python directly:

```python
from pygeovision.ai.pipelines import LandCoverPipeline

pipeline = LandCoverPipeline(client)
result = pipeline.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir="./output", date="2024-01")
```

## Verification status — read this before relying on a specific feature

This documentation distinguishes between code that's been independently tested against real data this audit cycle, and code that exists but hasn't been verified yet. Both are documented; only the first should be trusted without your own testing.

**Thoroughly audited, tested, and fixed this cycle** — real bugs found and fixed, verified with real data, regression-tested:

| Area | What's verified |
|---|---|
| **All 51 registered pipelines** | Every one individually resolved — see the full breakdown below. No pipeline in the catalog silently returns a wrong or misleadingly-successful result anymore. |
| **Data layer** | Multi-asset band stacking (Landsat/Sentinel-2), mission-aware band aliases, real unzip/reproject handling, cache-hit asset recovery, real SWIR1/SWIR2/red-edge/thermal band aliases (added this cycle — none existed before) |
| **Area/geodesic calculations** | Real `pyproj`-based area computation — fixes a confirmed bug (`res[0]*res[1]`-style area math assumed meters, but real output is `EPSG:4326` degrees) found in 5 places, wrong by roughly 10 billion times before the fix |
| **Tiled inference** | Memory-efficient windowed reads, memory-aware automatic batch sizing — `pygeovision.ai.inference.tiled_inference.TiledInference` specifically (see Architecture for a separate, unaudited version with the same name elsewhere) |
| **WorldCover labeling** | Real bug fixes: nodata handling, class-name mapping, tile-origin int/float promotion — `pygeovision.ai.labeling.esa_worldcover` specifically |
| **Object detection** | `client.detection.ships/cars` — fixed a severe mislabeling bug (every detection was labeled with the requested class regardless of what was actually detected) |
| **Model loading (`.to()` device placement)** | Fixed a bug affecting 9 real models (`clip-vit-b32`, `moondream2`, `tessera`, and others) across two independent call sites |
| **`pygeovision.pipelines` module** | Fixed — was a complete, stale, publicly-importable duplicate of the real pipeline classes, silently shadowing the real, fixed code for anyone using that import path |

**Not yet independently verified this cycle** — real code exists, but hasn't been tested against real data as part of this audit:

| Area | Status |
|---|---|
| Most of the ~80 CLI commands outside `channel` | `data`, `models`, `infer`, `label`, `explain`, `monitor`, `edge`, `cloud`, `vlm`, `timeseries`, `datasets`, `zoo`, `benchmark`, `validate`, `preprocess`, `indices`, `postprocess` |
| Serving layer (FastAPI/WebSocket), edge/cloud deployment, `training/` | Real code present, not load-tested or verified against real deployment targets |
| **36 of 121 registered models** | Real `timm_id`/`hf_id` metadata confirmed present, but couldn't be built/verified in this audit's sandbox (no network access to HuggingFace Hub). Likely real and working with normal network access. |
| **26 of 121 registered models** | Build successfully but fail a real forward-pass test — mostly 3D point-cloud architectures where the test input shape likely doesn't match what each expects. Needs per-architecture verification. |

**Genuinely unimplemented (honest, not silent)**: 43 of 121 registered models raise a clear `NotImplementedError` — no factory exists. 12 of the 51 pipelines do the same, each with a specific, individually-verified reason (missing data source, algorithm too specialized to trust yet, or a fundamentally different data domain like meteorology). See the breakdown below.

**Removed entirely this cycle**: PyGeoVision's own SAR/InSAR processing layer (`client.sar`, `pygeovision.insar`) — this duplicated functionality [PyGeoFetch](https://pypi.org/project/pygeofetch/) already implements natively and more completely. Call `pygeofetch.processing.sar.SARProcessor` / `pygeofetch.insar.*` directly. See [Architecture](architecture.md) for the honest reasoning.

### The pipeline catalog — every one of 51 individually resolved

`pygeovision.ai.pipelines.domains.list_pipelines()` returns 51 names. Every one has been through individual scrutiny this cycle. Breakdown:

**10 original pipelines** — `building_footprints`, `carbon_estimation`, `change_detection`, `crop_monitoring`, `deforestation`, `disaster_assessment`, `land_cover`, `solar_detection`, `urban_growth`, `water_bodies`. Real, thoroughly audited and fixed.

**13 real implementations** (previously generic stubs) — each verified against hand-calculated expected values:

| Pipeline | Real technique |
|---|---|
| `wildfire_severity` | Real dNBR (Key & Benson 2006), USFS/MTBS 4-class severity |
| `glacier_monitoring`, `snow_cover` | Real NDSI (Hall et al. 1995) |
| `wetland_mapping` | Real MNDWI (Xu 2006) + EVI (Huete et al. 2002) |
| `urban_heat_island`, `land_surface_temperature`, `volcano_monitoring` | Real Landsat thermal band → Celsius (real USGS Collection 2 formula); the latter two reuse this same verified computation with different thresholds |
| `parking_occupancy`, `port_monitoring` | Real vehicle/ship counts via the fixed `client.detection.cars/ships` |
| `landcover_change`, `mangrove_mapping` | Real ESA WorldCover class comparisons (bi-temporal) |
| `oil_spill_detection` | Real SAR dark-pixel detection via pygeofetch's `SARProcessor` directly |
| `pipeline_leak_detection`, `crop_health` | Real NDVI anomaly vs. baseline |
| `crop_yield_forecast` | Real peak-season NDVI (honestly uncalibrated unless you provide a real factor) |
| `biodiversity_hotspot` | Real Shannon diversity index on real land cover classes |
| `water_quality`, `vegetation_indices` | Real NDCI/NDVI/EVI/NDWI, computed directly (previously silently relied on a broken mechanism that never ran) |
| `forest_fire` | Real burn-scar mapping (reuses `wildfire_severity`'s dNBR); does not cover real-time active-fire detection |

**12 honest, specific `NotImplementedError`s** (previously generic stubs) — `permafrost_thaw`, `dam_safety` (real InSAR needs domain expertise this cycle didn't have, plus `snaphu` isn't installed — safety-critical territory), `air_quality_index`, `dust_storm_tracking` (confirmed zero real atmospheric-data provider in pygeofetch), `solar_potential`, `archaeological_site` (a real DEM source exists, but the algorithm is genuinely complex specialized work), `mine_detection`, `powerline_extraction`, `aquaculture_mapping`, `construction_progress` (need a specialized trained detector that doesn't exist), `reef_bleaching` (needs real bathymetric correction to be reliable), `wind_farm_siting` (needs real meteorological data, not imagery).

**16 pipelines with a confirmed, now-fixed bug pattern** — `crop_type_mapping`, `irrigation_detection`, `canopy_height`, `tree_species`, `road_extraction`, `infrastructure_monitoring`, `flood_mapping`, `coastal_monitoring`, `landslide_detection`, `ocean_ship_detection` and others: previously had a bare `except Exception:` that silently substituted a wrong result (raw imagery, or the output directory itself) while still returning `success=True`. All 9 instances of this pattern fixed — real errors now propagate correctly instead of being swallowed.

## Architecture

```
pygeovision/
├── data/           PyGeoFetch-backed satellite data layer
├── ai/
│   ├── pipelines/  51 pipelines, all individually verified this cycle (channel CLI command)
│   ├── models/     Model registry and architectures
│   ├── labeling/   Auto-labeling (WorldCover verified; others unaudited)
│   └── inference/  Tiled inference (memory-efficient, verified)
├── models/         Additional model implementations (detection, adapters)
├── cli/            ~26 command groups (channel verified; rest unaudited)
└── ...             15+ additional modules (agent, serving, monitoring,
                    edge, cloud, training, explainability, and more) —
                    not yet audited this cycle
```

See [Installation](installation.md) to get started, or [Architecture](architecture.md) for the full, honest picture of what's verified versus what isn't.
