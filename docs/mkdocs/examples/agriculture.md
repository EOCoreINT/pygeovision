# Agriculture Examples

## Crop Type Mapping — Nile Delta

```python
import pygeovision as pgv

client  = pgv.PyGeoVision()
results = client.search(
    bbox=(30.5, 29.8, 32.5, 31.5), date_range=("2026-04-01", "2026-06-30"),
    cloud_cover_max=15
)
downloads = client.download(results[:1], bands=["B02","B03","B04","B08","B11","B12"])
ready = client.prepare_for_ai(downloads[0].path, model_type="foundation")

from pygeovision.models.foundation.prithvi import PrithviTasks
tasks = PrithviTasks("prithvi_eo_2_0")
result = tasks.crop_mapping(ready["array"], source="sentinel2")
print(result["class_distribution"])
```

## SAR Soil Moisture Proxy

```python
from pygeovision.data.processors.sar import despeckle_sar, linear_to_db

# VH/VV ratio correlates with soil moisture
# Lower VH relative to VV → drier soil
despeckle_sar("vv.tif", "vv_desp.tif")
despeckle_sar("vh.tif", "vh_desp.tif")
```

## NDVI Time-Series for Crop Health

```python
from pygeovision.viz import TimeSeriesViewer

tsv = TimeSeriesViewer(
    paths=["ndvi_apr.tif", "ndvi_may.tif", "ndvi_jun.tif"],
    dates=["2026-04-15", "2026-05-15", "2026-06-15"],
    colormap="RdYlGn",
)
tsv.trend(title="Crop Health Trend").export("crop_trend.png")
tsv.anomaly(n_sigma=2.0).export("anomalies.png")
```
