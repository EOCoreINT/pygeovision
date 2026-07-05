# Chapter 28: Climate

## 28.1 Carbon Estimation

```python
import numpy as np, rasterio

with rasterio.open("./data/prepared.tif") as src:
    nir  = src.read(4).astype("float32")
    red  = src.read(3).astype("float32")
    blue = src.read(1).astype("float32")
    res  = abs(src.transform.a)

ndvi = (nir - red) / (nir + red + 1e-8)
evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)

# AGB allometric model (pantropical simplified)
agb_t_ha = 90.0 * evi + 20.0
agb_t_ha *= (ndvi > 0.3)  # vegetation mask

area_ha   = (ndvi > 0.3).sum() * (res**2 / 1e4)
total_agb = float(agb_t_ha.mean()) * area_ha
carbon_t  = total_agb * 0.47  # 47% of AGB is carbon

print(f"Vegetation area: {area_ha:.0f} ha")
print(f"AGB estimate:    {total_agb:.0f} tonnes")
print(f"Carbon stock:    {carbon_t:.0f} tonnes C")
print(f"CO2 equivalent:  {carbon_t * 3.67:.0f} tonnes CO2")
```

## 28.2 Land Surface Temperature

```python
# Landsat 9 Band 10 thermal
K1, K2 = 774.8853, 1321.0789

with rasterio.open("./data/LC09_B10.tif") as src:
    toa = src.read(1).astype("float32")

lst_k = K2 / np.log(K1 / toa + 1)
lst_c = lst_k - 273.15
print(f"LST: {lst_c.min():.1f} to {lst_c.max():.1f} C")
```

## 28.3 Urban Heat Island

```python
# Urban vs rural LST comparison
urban_mask = ndbi > 0
rural_mask = ndvi > 0.3

urban_lst  = lst_c[urban_mask].mean()
rural_lst  = lst_c[rural_mask].mean()
uhi        = urban_lst - rural_lst

print(f"Urban mean LST: {urban_lst:.1f} C")
print(f"Rural mean LST: {rural_lst:.1f} C")
print(f"UHI intensity:  {uhi:.1f} C")
```

## 28.4 Multi-Year NDVI Trend

```python
from pygeovision.viz import TimeSeriesViewer
from scipy.stats import linregress

ndvi_means = [compute_ndvi(path).mean() for path in annual_paths]
years      = list(range(2016, 2027))

slope, intercept, r, p, se = linregress(years, ndvi_means)
print(f"NDVI trend: {slope:+.4f}/year  (p={p:.3f})")
if slope < -0.003 and p < 0.05:
    print("WARNING: Significant vegetation decline detected")

tsv = TimeSeriesViewer(paths=annual_paths, dates=[str(y) for y in years],
                         colormap="RdYlGn")
tsv.trend(title="10-Year NDVI Trend").export("ndvi_trend_10yr.png")
```

## 28.5 Climate Change Analysis Pipeline

```python
# Automate annual climate monitoring
result = client.pipeline("carbon_estimation",
                          bbox=BBOX, date="2026-06", output_dir="./climate/")
print(f"Carbon stock: {result.stats.get('carbon_tonnes', 0):,.0f} t C")

# Glacier monitoring
result_glac = client.pipeline("glacier_monitoring",
                               bbox=(7.8, 46.4, 8.1, 46.6),  # Aletsch
                               date="2026-09", output_dir="./climate/glacier/")
print(f"Glacier area: {result_glac.stats.get('glacier_km2', 0):.1f} km2")
```

## Exercises

1. Estimate carbon stock for a national park.
2. Compute 10-year LST trend for your city. Is it warming?
3. Quantify the urban heat island in a city of your choice.
4. Compare glacier area in the Alps from 2000 to 2026.

## 28.8 Summary

Climate applications require the longest time-series and broadest coverage.
PyGeoVision enables pixel-level analysis from 30-year Landsat records alongside
current Sentinel-2, providing the depth needed for climate trend detection.

## Exercises

1. Estimate carbon stock for a national park.
2. Compute urban heat island for 5 cities.
3. Track glacier area from 2000-2026.
4. Map 30-year vegetation trend in the Sahel.
