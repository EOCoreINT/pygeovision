# Chapter 26: Water

## 26.1 Flood Mapping

```python
# SAR-based flood mapping (cloud-independent)
from pygeovision.agent import GeoAgent

agent = GeoAgent(client, output_dir="./water/floods/")
agent.set_context(bbox=(-0.30, 5.50, -0.05, 5.70), date="2026-06")
trace = agent.run("Map SAR flood extent — cloud cover 100%, rainy season")

flood_gdf    = gpd.read_file(trace.final_output)
flooded_km2  = flood_gdf.area.sum() / 1e6
print(f"Flooded area: {flooded_km2:.1f} km2")
print(f"Affected communities: {len(flood_gdf):,} flood polygons")
```

## 26.2 Water Quality Monitoring

```python
# Turbidity proxy from red band
with rasterio.open("./data/prepared.tif") as src:
    red   = src.read(3).astype("float32")
    nir   = src.read(4).astype("float32")

# NDWI to isolate water pixels
ndwi       = (src.read(2).astype("float32") - nir) / (src.read(2) + nir + 1e-8)
water_mask = ndwi > 0

# Turbidity index (higher red in water = more sediment)
turbidity = red * water_mask
clear = (turbidity < 0.05).sum() / water_mask.sum() * 100
print(f"Clear water: {clear:.1f}%")
```

## 26.3 Wetland Monitoring

```python
result = client.pipeline("water_bodies",
                          bbox=BBOX, date="2026-06", output_dir="./water/")
water_gdf  = gpd.read_file(result.output_path)
print(f"Water bodies: {len(water_gdf)}")
print(f"Total area: {water_gdf.area.sum()/1e6:.1f} km2")
```

## 26.4 Coastal Monitoring

```python
# Track shoreline change
pre_water  = gpd.read_file("./water/shoreline_2016.geojson")
post_water = gpd.read_file("./water/shoreline_2026.geojson")

# Shoreline retreat (simplified)
retreat_m = (pre_water.geometry.boundary.length -
              post_water.geometry.boundary.length) / pre_water.geometry.length
print(f"Estimated retreat: {retreat_m.mean():.1f} m over 10 years")
```

## Exercises

1. Map flood extent for a recent flood event in your region.
2. Monitor water quality (turbidity) in a river before and after rainfall.
3. Track wetland extent changes over 5 years.

## 26.7 Summary

Water monitoring from EO covers flood extent, water quality, wetland dynamics,
coastal change, and ocean chlorophyll. SAR gives cloud-independent flood mapping;
optical provides water quality indices.

## Exercises

1. Map flood extent for a recent flood event.
2. Monitor turbidity in a river before/after rainfall.
3. Track wetland extent changes over 5 years.
4. Detect algal bloom from Sentinel-3 chlorophyll retrieval.
