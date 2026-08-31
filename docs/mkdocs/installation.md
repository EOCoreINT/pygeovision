# Installation

## Requirements
- Python 3.10 or later
- pip 23+

## Core install (data + basic AI)
```bash
pip install pygeovision
```

## With geospatial dependencies (recommended)
```bash
pip install "pygeovision[geo]"
# Includes: rasterio, geopandas, pyproj, shapely, fiona
```

## With training support
```bash
pip install "pygeovision[train]"
# Includes: torch, torchvision, segmentation-models-pytorch, timm
```

## With visualization
```bash
pip install "pygeovision[viz]"
# Includes: matplotlib, ipyleaflet, plotly, folium
```

## With InSAR support
```bash
pip install "pygeovision[insar]"
# Includes: scipy, statsmodels (snaphu/isce2 installed separately)
```

## Full installation
```bash
pip install "pygeovision[all]"
```

## From source
```bash
git clone https://github.com/EOCoreINT/pygeovision
cd PyGeoVision
pip install -e ".[dev]"
```

## Verify
```python
import pygeovision as pgv
print(pgv.__version__)   # 2.1.7
client = pgv.PyGeoVision()
print(client)
```
