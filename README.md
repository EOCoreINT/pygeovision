<div align="center">

# pygeovision

**A unified open-source Python platform for satellite Earth observation AI.**

[![PyPI version](https://img.shields.io/pypi/v/pygeovision.svg)](https://pypi.org/project/pygeovision/)
[![Python](https://img.shields.io/pypi/pyversions/pygeovision.svg)](https://pypi.org/project/pygeovision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/eocoreint/pygeovision/actions/workflows/tests.yml/badge.svg)](https://github.com/eocoreint/pygeovision/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/codecov/c/github/eocoreint/pygeovision)](https://codecov.io/gh/eocoreint/pygeovision)
[![JOSS](https://joss.theoj.org/papers/XXXXX/status.svg)](https://doi.org/10.21105/joss.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

`pygeovision` is built on top of [`pygeofetch`](https://github.com/eocoreint/pygeofetch),
which handles satellite data access across 22+ providers (Copernicus Data Space,
Planetary Computer, USGS EarthExplorer, ASF Vertex, and more).
Together they cover the complete EO workflow in Python:

```
pygeofetch          →  search + download (22+ providers, any satellite)
pygeovision.data    →  preprocessing  (SAR 9-step pipeline, optical SCL masking)
pygeovision.models  →  AI inference   (Prithvi-EO-2.0, DINOv3, ChangeFormer)
pygeovision.insar   →  InSAR          (SNAP + SNAPHU, mm-precision displacement)
pygeovision.agent   →  GeoAgent       (natural-language query interface)
```

---

## Why pygeovision?

| Requirement | GEE | SNAP | eo-learn | **pygeovision** |
|---|---|---|---|---|
| Runs locally / on your server | ✗ | ✓ | ✓ | **✓** |
| Full audit trail (provenance JSON) | ✗ | ✗ | ✗ | **✓** |
| SAR preprocessing pipeline | ✗ | ✓ | ✗ | **✓** |
| Foundation model inference | ✗ | ✗ | ✗ | **✓** |
| InSAR (mm displacement) | ✗ | ✓ | ✗ | **✓** |
| Natural-language query | ✗ | ✗ | ✗ | **✓** |
| Free, open-source, OSI-approved | ✓ | ✓ | ✓ | **✓** |

---

## Installation

```bash
# Core (data access + preprocessing + spectral indices)
pip install pygeovision

# With AI inference (CPU)
pip install "pygeovision[ai]"

# With AI inference (NVIDIA GPU)
pip install "pygeovision[ai-gpu]"

# With InSAR support (also requires SNAP 9 and snaphu installed separately)
pip install "pygeovision[insar]"

# Everything
pip install "pygeovision[all]"
```

**System requirements:** Python 3.10+, GDAL (via conda recommended).

```bash
# Recommended: install GDAL via conda-forge first
conda install -c conda-forge gdal rasterio geopandas
pip install pygeovision
```

---

## Quick start

### 1. Search and download a Sentinel-2 scene

```python
from pygeovision import PyGeoVision

client = PyGeoVision()

# No credentials needed for Planetary Computer
scenes = client.search(
    bbox=(-0.30, 5.50, -0.05, 5.70),      # Odaw Basin, Accra
    date_range=("2024-06-01", "2024-06-30"),
    satellite="Sentinel-2",
    cloud_cover_max=20,
    providers=["planetary_computer"],
    use_cache=False,
)
print(f"Found {len(scenes)} scenes")

result = client.download(
    [scenes[0]],
    output_dir="./data/",
    post_process=["reproject:EPSG:32630", "cog"],
)
print(f"Downloaded: {result[0].path}")
```

### 2. Run SAR flood detection

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
client.add_credentials("copernicus", username="you@email.com", password="***")

result = client.run_pipeline(
    "flood_detection",
    bbox=(-0.30, 5.50, -0.05, 5.70),
    date_range=("2024-06-25", "2024-06-30"),
    output_dir="./results/flood/",
)
print(result.summary())
# Flood area: 847 ha  |  Confidence: high  |  SAR date: 2024-06-27
```

### 3. Foundation model inference (Prithvi-EO-2.0)

```python
from pygeovision.models.foundation.prithvi import PrithviTasks

prithvi = PrithviTasks(task="land_cover", backbone="prithvi_eo_v2_600")
land_cover = prithvi.run(sentinel2_scene)
print(land_cover.class_fractions)
# {'water': 0.04, 'built': 0.31, 'vegetation': 0.48, 'bare': 0.14, ...}
```

### 4. Natural-language query (GeoAgent)

```python
result = client.agent.run(
    query="Map deforestation in Atewa Forest Reserve, Ghana between 2020 and 2024",
    bbox=(-0.65, 6.15, -0.35, 6.40),
    output_dir="./results/atewa/",
)
print(f"Cleared area: {result.change_area_ha:.0f} ha")
```

### 5. InSAR deformation mapping

```python
from pygeovision.insar.slc import SLCInSARPipeline

pipeline = SLCInSARPipeline(
    master_zip="S1C_IW_SLC__1SDV_20260601T053000.zip",
    slave_zip="S1C_IW_SLC__1SDV_20260613T053000.zip",
    output_dir="./insar/",
    subswath="IW2",
    polarisation="VV",
)
result = pipeline.run()
print(result.summary())
# LOS displacement: [-0.043, 0.037] m  |  Mean coherence: 0.62
```

---

## What's included

### 10 end-to-end pipelines

| Pipeline | Sensor | Application |
|---|---|---|
| `flood_detection` | SAR | Flood extent, community impact, FloodWatch GeoJSON |
| `deforestation` | Optical | ChangeFormer bi-temporal change, REDD+ area stats |
| `crop_mapping` | Optical | Prithvi-EO-2.0 10-class crop type map |
| `building_footprints` | VHR / Optical | DINOv3 segmentation, LoRA fine-tuning |
| `urban_change` | Optical | Landsat 30-year NDBI time series |
| `oil_spill` | SAR | Dark-pixel marine detection |
| `burn_scar` | Optical | dNBR from pre/post Sentinel-2 |
| `subsidence_proxy` | SAR | Bi-temporal amplitude change |
| `water_bodies` | Optical | NDWI + SAR joint classifier |
| `biomass` | Optical | DINOv3 CHMv2 canopy height → allometric AGB |

### Foundation models

| Model | Task heads | Pre-training data |
|---|---|---|
| Prithvi-EO-2.0 (300M / 600M) | land_cover, crop, flood, burn_scar, biomass | 4.2M HLS tiles |
| DINOv3 (ViT-S/B/L/G/7B, ConvNeXt) | classifier, segmentor, detector, depther, CHMv2, dino.txt | SAT-493M satellite images |

### SAR InSAR pipeline

- Full TOPSAR chain via ESA SNAP + snapista Python API
- SNAPHU phase unwrapping (DEFO mode, MCF initialisation)
- Phase-to-displacement: $d_\text{LOS} = \frac{\lambda}{4\pi} \varphi$
- Ascending + descending decomposition into vertical and east-west components
- GPS validation support

---

## Documentation

Full documentation at **[pygeovision.readthedocs.io](https://pygeovision.readthedocs.io)**

- [Installation guide](https://pygeovision.readthedocs.io/guides/installation)
- [SAR flood mapping walkthrough](https://pygeovision.readthedocs.io/tutorials/sar-flood)
- [Foundation models guide](https://pygeovision.readthedocs.io/guides/foundation-models)
- [InSAR tutorial](https://pygeovision.readthedocs.io/tutorials/insar)
- [API reference](https://pygeovision.readthedocs.io/api)
- [Example notebooks](examples/)

---

## Testing

```bash
# Install dev dependencies
pip install "pygeovision[dev]"

# Run all fast tests (no download, no GPU required)
pytest -m "not slow and not gpu and not insar"

# Run full test suite (requires Copernicus credentials and GPU)
pytest
```

The test suite contains 652 tests. All tests marked `slow`, `gpu`, or `insar`
are skipped by default in CI and require additional hardware or credentials.

---

## Citing pygeovision

If you use pygeovision in your research, please cite the JOSS paper:

```bibtex
@article{appiahkubi2026pygeovision,
  author  = {Appiah Kubi, Samuel},
  title   = {{pygeovision: A Unified Open-Source Python Platform for
              Satellite Earth Observation AI}},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {10.21105/joss.XXXXX},
}
```

Or use the `CITATION.cff` file — GitHub renders a "Cite this repository" button
automatically.

---

## Contributing

Contributions are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

- **Bug reports**: open a [GitHub Issue](https://github.com/eocoreint/pygeovision/issues)
- **Feature requests**: open a [GitHub Discussion](https://github.com/eocoreint/pygeovision/discussions)
- **Pull requests**: fork → branch → PR against `main`

All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2024–2026 Samuel Appiah Kubi / EOCoreINT
