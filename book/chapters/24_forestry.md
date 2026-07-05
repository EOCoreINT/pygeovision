# Chapter 24: Forestry

## 24.1 Deforestation Detection

```python
result = client.pipeline("deforestation",
                          bbox=(-55.0, -3.0, -54.5, -2.5),  # Amazon
                          date="2026-06",
                          output_dir="./forest/")

loss = result.stats.get("forest_loss_km2", 0)
rate = result.stats.get("annual_loss_pct", 0)
print(f"Forest loss: {loss:.1f} km2 ({rate:.2f}%/year)")
```

## 24.2 Canopy Height Estimation

```python
from pygeovision.models import get_model

height_model = get_model("unet_regressor", encoder_name="resnet50",
                           in_channels=6, output_channels=1)
# Load pre-trained weights
height_model.load_state_dict(torch.load("./models/canopy_height.pth"))
height_model.eval()

with torch.no_grad():
    x      = torch.tensor(ready_array).unsqueeze(0).float()
    height = height_model(x).squeeze().numpy()

print(f"Mean canopy height: {height.mean():.1f} m")
print(f"Max canopy height:  {height.max():.1f} m")
```

## 24.3 Biomass Estimation

```python
# Simple allometric model from EVI + height
evi       = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
agb_t_ha  = 90 * evi + 0.5 * height  # tonnes/ha (simplified)
agb_t_ha  = agb_t_ha * (ndvi > 0.3)  # vegetation mask

total_agb = agb_t_ha.mean() * area_ha
carbon    = total_agb * 0.47
print(f"AGB: {total_agb:.0f} t  Carbon: {carbon:.0f} t C")
```

## 24.4 Forest Fire Detection

```python
result = client.pipeline("wildfire_severity",
                          bbox=BBOX, date="2026-09", output_dir="./fire/")
burn = result.stats

for cls, label in enumerate(["Unburned","Low","Moderate","High"]):
    pct = burn.get(f"severity_{cls}_pct", 0)
    print(f"  {label:<10}: {pct:.1f}%")
```

## Exercises

1. Map deforestation in the Amazon for the last 3 years.
2. Estimate biomass for a temperate forest patch.
3. Detect wildfire burn scars from the last fire season in your region.

## 24.7 Summary

Forestry monitoring from satellite imagery covers the complete carbon cycle:
deforestation detection, canopy height, biomass estimation, and REDD+ accounting.
Sentinel-1 SAR + Sentinel-2 optical fusion gives the most complete characterisation.

## Exercises

1. Detect deforestation in a tropical region 2020-2026.
2. Estimate AGB and compare to national forest inventory.
3. Map wildfire burn severity (4 USFS classes).
4. Calculate REDD+ carbon credits for a protected area.
