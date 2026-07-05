# Quick Start

## 5 minutes to your first satellite AI result

### 1. Install
```bash
pip install "pygeovision[geo]"
```

### 2. Search and download
```python
import pygeovision as pgv

client = pgv.PyGeoVision()

# Accra, Ghana — flood-prone Odaw River Basin
results = client.search(
    bbox=(-0.30, 5.50, -0.05, 5.70),
    date_range=("2026-06-01", "2026-06-30"),
    cloud_cover_max=20,
)
print(f"Found {len(results)} scenes")

downloads = client.download(results[:1], bands=["B02","B03","B04","B08","B11","B12"])
```

### 3. Prepare for AI
```python
ready = client.prepare_for_ai(downloads[0].path, model_type="foundation")
print(f"Shape: {ready['shape']}  Range: [{ready['array'].min():.3f}, {ready['array'].max():.3f}]")
```

### 4. Run AI inference
```python
from pygeovision.models.foundation.prithvi import PrithviTasks

tasks = PrithviTasks("prithvi_eo_2_0")
result = tasks.land_cover(ready["array"], source="sentinel2")
print(f"Land cover classes: {result['class_distribution']}")
```

### 5. Visualize
```python
from pygeovision.viz import RasterViewer

rv = RasterViewer(downloads[0].path)
rv.rgb(red=2, green=1, blue=0).show()
rv.ndvi(nir_band=3, red_band=2).export("ndvi.png")
```

### 6. GeoAgent (natural language)
```python
from pygeovision.agent import GeoAgent

agent = GeoAgent(client, output_dir="./results/")
agent.set_context(bbox=(-0.30, 5.50, -0.05, 5.70), date="2026-06")
trace = agent.run("Map flood inundation in the Odaw River Basin")
print(trace.final_output)  # → ./results/flood_mask.geojson
```
