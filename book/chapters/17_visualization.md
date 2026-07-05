# Chapter 17: Interactive Visualization

## 17.1 Interactive Map

```python
from pygeovision.viz import Map

# Create map centered on study area
m = Map(center=(5.6, -0.2), zoom=12, basemap="satellite")

# Add raster layer with colour mapping
m.add_raster("./data/flood_mask.tif",
              colormap="Blues", band=0, opacity=0.8,
              name="Flood Extent")

# Add vector overlay
m.add_vector("./data/risk_zones.geojson",
              fill_color="#e74c3c", fill_opacity=0.3,
              name="Risk Zones")

# Export standalone HTML (no server needed)
m.export("./maps/flood_risk.html")

# Side-by-side comparison
split = Map.split_view("./data/before.tif", "./data/after.tif")
split.export("./maps/before_after.html")
```

## 17.2 Raster Viewer

```python
from pygeovision.viz import RasterViewer

rv = RasterViewer("./data/sentinel2.tif", figsize=(12, 10))

# Composites
rv.rgb(red=2, green=1, blue=0).show()           # True colour
rv.false_color(nir=3, red=2, green=1).show()    # NIR false colour

# Indices
rv.ndvi(nir_band=3, red_band=2).export("ndvi.png", dpi=300)
rv.ndwi(green_band=1, nir_band=3).show()

# Analysis
rv.histogram().show()                             # All bands
rv.profile(start_px=(100,50), end_px=(400,300)).show()  # Transect
```

## 17.3 Change Viewer

```python
from pygeovision.viz import ChangeViewer

cv = ChangeViewer("./data/before.tif", "./data/after.tif")

cv.split(before_label="2023 Pre-flood",
          after_label="2023 Post-flood").export("comparison.png")

cv.difference(title="Amplitude Change").show()

stats = cv.statistics("./data/flood_mask.tif")
print(f"Flooded area: {stats.get('changed_area_km2', 0):.1f} km2")
```

## 17.4 Time-Series Viewer

```python
from pygeovision.viz import TimeSeriesViewer

tsv = TimeSeriesViewer(
    paths     = ndvi_paths,
    dates     = ndvi_dates,
    colormap  = "RdYlGn",
)

tsv.animate("ndvi.gif", fps=3)                   # Animated GIF
tsv.trend(title="NDVI 3-Year Trend").show()      # Trend line
tsv.seasonal(period=4).show()                    # Seasonal decomp
tsv.anomaly(n_sigma=2.0).export("anomalies.png") # Anomaly detection
tsv.mosaic(max_cols=4).export("mosaic.png")      # Overview grid
```

## Exercises

1. Create an interactive map of your study area with raster and vector layers.
2. Build a before/after split view for a recent event.
3. Animate a 2-year NDVI time-series as GIF.

## 17.5 VectorViewer

```python
from pygeovision.viz import VectorViewer
import geopandas as gpd

vv = VectorViewer("./data/buildings.geojson")
vv.view(color="#3388ff", fill_opacity=0.4).show()
vv.table(max_rows=10)
vv.style_by_attribute("area", cmap="Reds").export("choropleth.png")
filtered = vv.filter("area > 500")
vv.heatmap().export("heatmap.png")
```

## Summary

PyGeoVision's visualization layer covers interactive maps, raster indices,
vector styling, change comparison, and time-series animation — all with
standalone HTML export requiring no server.

## Exercises

1. Build an interactive map with flood extent and risk zones.
2. Animate a 2-year NDVI time-series as GIF.
3. Create a choropleth map of building area distribution.
