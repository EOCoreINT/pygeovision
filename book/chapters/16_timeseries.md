# Chapter 16: Time-Series Analysis

## 16.1 Multi-Temporal Data Acquisition

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

# Download seasonal Sentinel-2 stack (2 years)
stack_paths = []
for year in [2024, 2025, 2026]:
    for month in ["03", "06", "09", "12"]:
        date = f"{year}-{month}"
        results = client.search(bbox=BBOX,
                                 date_range=(f"{date}-01", f"{date}-28"),
                                 cloud_cover_max=20)
        if results:
            dl = client.download(results[:1], output_dir=f"./stack/{date}/",
                                  bands=BANDS)
            if dl and dl[0].success:
                stack_paths.append(dl[0].path)

print(f"Time-series: {len(stack_paths)} scenes")
```

## 16.2 NDVI Time-Series

```python
import numpy as np, rasterio

ndvi_series = []
dates       = []

for path in stack_paths:
    with rasterio.open(path) as src:
        nir = src.read(4).astype("float32")
        red = src.read(3).astype("float32")
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi_series.append(float(ndvi.mean()))
    dates.append(path.parent.name)

# Visualise
from pygeovision.viz import TimeSeriesViewer

tsv = TimeSeriesViewer(
    paths     = stack_paths,
    dates     = dates,
    colormap  = "RdYlGn",
    band      = 0,   # will compute NDVI internally
)
tsv.trend(title="NDVI 2-Year Trend").export("ndvi_trend.png")
tsv.anomaly(n_sigma=2.0).export("ndvi_anomalies.png")
```

## 16.3 Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

series = pd.Series(ndvi_series, index=pd.to_datetime(dates))
decomp = seasonal_decompose(series, model="additive", period=4)

print(f"Trend: {decomp.trend.mean():.4f}")
print(f"Seasonal amplitude: {decomp.seasonal.std():.4f}")
```

## 16.4 Change Point Detection

```python
# Detect sudden shifts (e.g., deforestation event)
import numpy as np

def detect_changepoints(series, threshold=2.5):
    mu    = np.mean(series)
    sigma = np.std(series)
    changes = []
    for i in range(1, len(series)):
        diff = abs(series[i] - series[i-1])
        if diff > threshold * sigma:
            changes.append((i, series[i] - series[i-1]))
    return changes

changes = detect_changepoints(ndvi_series)
for idx, magnitude in changes:
    print(f"  Change at {dates[idx]}: {magnitude:+.4f}")
```

## Exercises

1. Download a 2-year quarterly stack. Plot NDVI time-series.
2. Identify the seasonal pattern in your study area.
3. Detect a deforestation or drought event using change point detection.

## 16.5 Phenology Monitoring

```python
import numpy as np
from scipy.signal import savgol_filter

ndvi_smooth = savgol_filter(ndvi_series, window_length=5, polyorder=2)
threshold   = ndvi_smooth.max() * 0.5
above       = ndvi_smooth > threshold
crossings   = np.diff(above.astype(int))

sos_idx = np.where(crossings == 1)[0]
eos_idx = np.where(crossings ==-1)[0]

if len(sos_idx) > 0 and len(eos_idx) > 0:
    print(f"SOS: {dates[sos_idx[0]]}")
    print(f"EOS: {dates[eos_idx[0]]}")
```

## Summary

Time-series analysis reveals temporal dynamics invisible in single-date imagery.
Use seasonal decomposition, SOS/EOS phenology, and anomaly detection.

## Exercises

1. Download a 3-year quarterly stack. Plot NDVI time-series.
2. Find growing season start/end using SOS/EOS detection.
3. Detect a drought event using anomaly detection.
