# PyGeoVision

**AI that sees Earth from space.**

PyGeoVision is a production-ready Python platform that unifies satellite data acquisition
(22+ providers via PyGeoFetch) and geospatial AI (full model stack) in one coherent API.

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
results = client.search(bbox=(-0.30, 5.50, -0.05, 5.70), date_range=("2026-06-01","2026-06-30"))
downloads = client.download(results[:1], bands=["B02","B03","B04","B08","B11","B12"])
ready = client.prepare_for_ai(downloads[0].path)
```

## Key Features

| Feature | Description |
|---------|-------------|
| 🛰️ **22+ Providers** | Sentinel, Landsat, Planet, Maxar, USGS, Copernicus, JAXA and more |
| 🤖 **119 AI Models** | Segmentation, detection, change detection, foundation models |
| 📡 **SAR Pipeline** | Cloud-independent Sentinel-1 processing with 3 production bug-fixes |
| 🧠 **GeoAgent** | Natural language → complete geospatial pipeline |
| 📐 **InSAR** | Interferogram, coherence, displacement, deformation rate |
| 🗺️ **Visualization** | Interactive maps, raster/vector viewers, time-series, dashboards |
| 🏢 **Enterprise** | RBAC, audit logging, GDPR/SOC2 compliance |

## Installation

```bash
pip install pygeovision                     # core
pip install "pygeovision[geo]"             # + rasterio, geopandas
pip install "pygeovision[train]"           # + PyTorch, segmentation-models-pytorch
pip install "pygeovision[viz]"             # + leafmap, plotly, ipyleaflet
pip install "pygeovision[insar]"           # + isce2, mintpy, snaphu
pip install "pygeovision[all]"             # everything
```

## Quick Links

- 🚀 [Quick Start](quickstart.md)
- 📚 [Tutorials](tutorials/index.md)
- 🔌 [API Reference](api/index.md)
- 🤖 [GeoAgent](api/agent.md)
- 📐 [InSAR](api/insar.md)
- 🗺️ [Visualization](api/viz.md)
