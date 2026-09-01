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
| **10 core pipelines** | `building_footprints`, `carbon_estimation`, `change_detection`, `crop_monitoring`, `deforestation`, `disaster_assessment`, `land_cover`, `solar_detection`, `urban_growth`, `water_bodies` — real radiometric scaling, cloud masking, bbox cropping, common-grid alignment for bi-temporal pairs |
| **Data layer** | Multi-asset band stacking (Landsat/Sentinel-2), mission-aware band aliases, real unzip/reproject handling, cache-hit asset recovery |
| **Tiled inference** | Memory-efficient windowed reads, memory-aware automatic batch sizing |
| **WorldCover labeling** | Real bug fixes: nodata handling, class-name mapping, tile-origin int/float promotion |
| **Object detection** | `client.detection.ships/cars` — fixed a severe mislabeling bug (every detection was labeled with the requested class regardless of what was actually detected) |

**Not yet independently verified this cycle** — real code exists, but hasn't been tested against real data as part of this audit:

| Area | Status |
|---|---|
| **41 of the 51 "pipeline catalog" entries** | See the honest breakdown below — most of these are not yet verified to do what their names claim |
| Most of the ~80 CLI commands outside `channel` | `data`, `models`, `infer`, `label`, `explain`, `monitor`, `edge`, `cloud`, `vlm`, `timeseries`, `datasets`, `zoo`, `benchmark`, `validate`, `preprocess`, `indices`, `postprocess` |
| Serving layer (FastAPI/WebSocket), edge/cloud deployment | Real code present, not load-tested or verified against real deployment targets |

**Removed entirely this cycle**: PyGeoVision's own SAR/InSAR processing layer (`client.sar`, `pygeovision.insar`) — this duplicated functionality [PyGeoFetch](https://pypi.org/project/pygeofetch/) already implements natively and more completely. Call `pygeofetch.processing.sar.SARProcessor` / `pygeofetch.insar.*` directly. See [Architecture](architecture.md) for the honest reasoning.

### The pipeline catalog, honestly

`pygeovision.ai.pipelines.domains.list_pipelines()` returns 51 names. Breaking down what's actually behind each:

- **10 pipelines** (listed above) — real, task-specific implementations, thoroughly audited and fixed this cycle.
- **16 pipelines** (`crop_type_mapping`, `crop_health`, `irrigation_detection`, `canopy_height`, `tree_species`, `forest_fire`, `road_extraction`, `infrastructure_monitoring`, `flood_mapping`, `water_quality`, `coastal_monitoring`, `landslide_detection`, `volcano_monitoring`, `land_surface_temperature`, `vegetation_indices`, `ocean_ship_detection`) — real, dedicated classes with task-specific logic, but not yet independently verified. At least one (`crop_type_mapping`) has a confirmed bug: if its default model name isn't registered, it silently falls back to returning the raw, unprocessed input image as the "result," with no visible error.
- **26 pipelines** (`air_quality_index`, `oil_spill_detection`, `wildfire_severity`, `glacier_monitoring`, and 22 others) — built from a generic factory (`_make_simple`) that only searches, downloads one scene, and validates it. Despite specific-sounding descriptions ("SAR oil slick detection via adaptive backscatter threshold," "dNBR burn severity mapping"), **none of these currently run any task-specific model or algorithm.** Calling one returns a generic completion status, not the described output.

If you need one of the 26 or 16 for real work, verify it yourself against real data first, or use the 10 fully-audited pipelines as a reference for what a complete implementation looks like.

## Architecture

```
pygeovision/
├── data/           PyGeoFetch-backed satellite data layer
├── ai/
│   ├── pipelines/  10 verified pipelines (channel CLI command)
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
