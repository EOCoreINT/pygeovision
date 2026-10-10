# Chapter 11: Pixel-Level Regression

## 11.1 Canopy Height Estimation

```python
from pygeovision.models import get_model
import torch

# UNet regression model for canopy height
model = get_model("unet_regressor",
                   encoder_name="resnet50",
                   in_channels=6,
                   output_channels=1)

with torch.no_grad():
    x      = torch.randn(1, 6, 512, 512)
    height = model(x)  # continuous values in metres

print(f"Height range: {height.min():.1f} to {height.max():.1f} m")
```

## 11.2 Biomass Estimation

```python
# AGB (Above-Ground Biomass) from NDVI + allometric equation
ndvi     = (nir - red) / (nir + red + 1e-8)
evi      = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)

# Empirical model (pantropical)
agb_t_ha = 90.0 * evi + 20.0   # tonnes per hectare (simplified)
agb_t_ha = agb_t_ha * (ndvi > 0.3)  # mask non-vegetation

total_agb = agb_t_ha.mean() * area_ha
print(f"Estimated AGB: {total_agb:.0f} tonnes")
print(f"Carbon stock:  {total_agb * 0.47:.0f} tonnes C")
```

## 11.3 Land Surface Temperature

```python
# Landsat 8/9 Band 10 (thermal)
import rasterio, numpy as np

with rasterio.open("./data/LC09_B10.tif") as src:
    toa_radiance = src.read(1).astype("float32")

K1, K2 = 774.8853, 1321.0789  # Landsat 9 constants
lst_kelvin = K2 / np.log(K1 / toa_radiance + 1)
lst_celsius = lst_kelvin - 273.15

print(f"LST range: {lst_celsius.min():.1f} to {lst_celsius.max():.1f} C")
```

## Exercises

1. Estimate canopy height for a forested study area.
2. Compute carbon stock from NDVI + EVI combination.
3. Map urban heat island intensity using LST difference (urban vs rural).

## 11.4 Soil Moisture from SAR

```python
import numpy as np, rasterio

with rasterio.open("./sar/S1_VV_db.tif") as src:
    vv_db = src.read(1).astype("float32")

vv_linear  = 10 ** (vv_db / 10)
soil_moist = np.clip((vv_linear - 0.01) / 0.49, 0, 1) * 0.5
print(f"Mean soil moisture: {soil_moist.mean():.3f} m3/m3")
```

## 11.5 Yield Prediction with LSTM

```python
import torch, torch.nn as nn

class LSTMYield(nn.Module):
    def __init__(self, n_feat=4, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:,-1]).squeeze(-1)

model    = LSTMYield(n_feat=4, hidden=128)
optimizer= torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn  = nn.MSELoss()
```

## Summary

Pixel regression predicts continuous values (height, biomass, LST, moisture).
LSTM models capture temporal crop growth for yield prediction.

## Exercises

1. Estimate canopy height for a forest patch.
2. Predict wheat yield from 12-month NDVI time-series.
3. Map urban heat island using Landsat LST.
