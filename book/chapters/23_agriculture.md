# Chapter 23: Agriculture

## 23.1 Crop Type Mapping

```python
import pygeovision as pgv
from pygeovision.models.foundation.prithvi import PrithviTasks, CROP_CLASSES

client = pgv.PyGeoVision()

# Download seasonal stack (key growth stages)
seasons = [("2026-04-01","2026-04-30"), ("2026-06-01","2026-06-30")]
stack   = []
for start, end in seasons:
    r  = client.search(bbox=BBOX, date_range=(start,end), cloud_cover_max=15)
    dl = client.download(r[:1], output_dir=f"./agri/{start[:7]}/", bands=BANDS)
    if dl and dl[0].success:
        stack.append(dl[0].path)

# Prithvi crop mapping
ready = client.prepare_for_ai(stack[-1], model_type="foundation")
tasks = PrithviTasks("prithvi_eo_2_0")
crops = tasks.crop_mapping(ready["array"], source="sentinel2")

print("Crop distribution:")
for crop, pct in crops["class_distribution"].items():
    print(f"  {crop:<20}: {pct:.1f}%")
```

## 23.2 NDVI Time-Series for Crop Health

```python
from pygeovision.viz import TimeSeriesViewer

# Monthly NDVI stack
tsv = TimeSeriesViewer(paths=stack, dates=seasons,
                         colormap="RdYlGn")
tsv.trend(title="Crop NDVI Seasonal Pattern").export("crop_ndvi.png")
tsv.anomaly(n_sigma=1.5).export("crop_stress.png")  # drought stress
```

## 23.3 Field Boundary Delineation

```python
result = client.pipeline("crop_monitoring",
                          bbox=BBOX, date="2026-06", output_dir="./agri/")
fields = gpd.read_file(result.output_path)
print(f"Fields: {len(fields)}")
print(f"Mean area: {fields.area.mean() / 1e4:.1f} ha")
```

## 23.4 Yield Prediction

```python
from pygeovision.models import get_model
import torch

# Regression model: NDVI time-series -> yield estimate
yield_model = get_model("lstm_regressor",
                          input_size=12,    # 12 monthly NDVI values
                          hidden_size=64,
                          output_size=1)    # tonnes/ha

# Prepare sequence input
ndvi_seq = torch.tensor(monthly_ndvi).unsqueeze(0).float()
yield_pred = yield_model(ndvi_seq).item()
print(f"Predicted yield: {yield_pred:.1f} t/ha")
```

## Exercises

1. Map crop types in the Nile Delta using Prithvi.
2. Identify drought-stressed fields using NDVI anomaly detection.
3. Delineate field boundaries and compute mean field size.

## 23.7 Summary

Agriculture is the largest application domain for geospatial AI.
Sentinel-2 time-series + Prithvi crop mapping + NDVI anomaly detection
provides a complete precision agriculture monitoring system.

## Exercises

1. Map crop types in the Nile Delta using Prithvi.
2. Identify drought-stressed fields using NDVI anomaly (2-sigma).
3. Detect irrigation events using SAR backscatter change.
4. Predict yield for a test field.
