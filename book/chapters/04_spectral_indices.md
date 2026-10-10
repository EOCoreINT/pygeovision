# Chapter 4: Spectral Indices

## 4.1 What Are Spectral Indices?

Spectral indices are mathematical combinations of satellite bands that highlight
specific surface properties. They exploit characteristic reflectance signatures
of different materials across the electromagnetic spectrum.

## 4.2 Vegetation Indices

### NDVI (Normalised Difference Vegetation Index)

```python
import numpy as np, rasterio

with rasterio.open("./data/S2_prepared.tif") as src:
    nir = src.read(4).astype("float32")
    red = src.read(3).astype("float32")

ndvi = (nir - red) / (nir + red + 1e-8)
print(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")

# Via RasterViewer
from pygeovision.viz import RasterViewer
rv = RasterViewer("./data/S2_prepared.tif")
rv.ndvi(nir_band=3, red_band=2).export("ndvi.png")
```

### EVI (Enhanced Vegetation Index) — better for dense canopy

```python
evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8)
evi = np.clip(evi, -1, 1)
```

### SAVI (Soil-Adjusted) — for sparse vegetation

```python
L    = 0.5  # soil factor (0=dense, 1=sparse)
savi = (1 + L) * (nir - red) / (nir + red + L + 1e-8)
```

## 4.3 Water Indices

### NDWI — highlights open water

```python
with rasterio.open("./data/S2_prepared.tif") as src:
    green = src.read(2).astype("float32")
    nir   = src.read(4).astype("float32")

ndwi = (green - nir) / (green + nir + 1e-8)
water_pct = (ndwi > 0).mean() * 100
print(f"Water coverage: {water_pct:.1f}%")
```

### MNDWI — better for urban water

```python
with rasterio.open("./data/S2_prepared.tif") as src:
    green = src.read(2).astype("float32")
    swir  = src.read(5).astype("float32")

mndwi = (green - swir) / (green + swir + 1e-8)
```

## 4.4 Urban Indices

### NDBI — highlights built-up surfaces

```python
with rasterio.open("./data/S2_prepared.tif") as src:
    swir = src.read(5).astype("float32")
    nir  = src.read(4).astype("float32")

ndbi = (swir - nir) / (swir + nir + 1e-8)
built_pct = (ndbi > 0).mean() * 100
print(f"Built-up: {built_pct:.1f}%")
```

## 4.5 Burn Indices

```python
def compute_nbr(nir, swir2):
    return (nir - swir2) / (nir + swir2 + 1e-8)

pre_nbr  = compute_nbr(nir_pre,  swir2_pre)
post_nbr = compute_nbr(nir_post, swir2_post)
dnbr     = pre_nbr - post_nbr  # Positive = burned

# USFS 4-class severity
severity = np.zeros_like(dnbr, dtype="uint8")
severity[dnbr > 0.66]                         = 4  # high
severity[(dnbr > 0.26) & (dnbr <= 0.66)]      = 3  # moderate
severity[(dnbr > 0.10) & (dnbr <= 0.26)]      = 2  # low
severity[dnbr <= 0.10]                         = 1  # unchanged
```

## 4.6 Snow / Ice Indices

```python
with rasterio.open("./data/S2_prepared.tif") as src:
    green = src.read(2).astype("float32")
    swir  = src.read(5).astype("float32")

ndsi = (green - swir) / (green + swir + 1e-8)
snow_mask = (ndsi > 0.4).astype("uint8")
```

## 4.7 Tasseled Cap Transformation

```python
TC_COEFF = {
    "brightness": [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949],
    "greenness":  [-0.3599,-0.3533,-0.4734, 0.6633,-0.0059,-0.2856],
    "wetness":    [0.2578, 0.2305, 0.0883, 0.1071,-0.7611,-0.5308],
}

def tasseled_cap(bands_6ch):
    result = {}
    for name, coef in TC_COEFF.items():
        c = np.array(coef, dtype="float32")[:, None, None]
        result[name] = (bands_6ch * c).sum(axis=0)
    return result
```

## 4.8 Index-Based Classification

```python
land_cover = np.zeros_like(ndvi, dtype="uint8")
land_cover[ndvi > 0.3]                                         = 1  # Vegetation
land_cover[(ndwi > 0.0) & (ndvi <= 0.3)]                      = 2  # Water
land_cover[(ndbi > 0.1) & (ndvi <= 0.3) & (ndwi <= 0.0)]     = 3  # Built-up
land_cover[land_cover == 0]                                    = 4  # Bare soil
```

## Summary

- NDVI (NIR-Red)/(NIR+Red): vegetation density
- NDWI (Green-NIR)/(Green+NIR): open water
- NDBI (SWIR-NIR)/(SWIR+NIR): built-up surfaces
- dNBR: wildfire burn severity (4-class USFS scale)
- Tasseled Cap: brightness, greenness, wetness

## Exercises

1. Compute NDVI, NDWI, NDBI side by side. Which areas are misclassified?
2. Apply NDSI to a winter Sentinel-2 scene. Map snow extent.
3. Compare NDWI vs MNDWI for a coastal urban area.
4. Compute dNBR for a wildfire event from the last 2 years.
