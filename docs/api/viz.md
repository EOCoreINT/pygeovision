# Visualization API Reference

**Verification status**: class and method signatures confirmed real against
the installed code this cycle (`Map`, `RasterViewer`, `add_raster`,
`add_vector`, `export`, `split_view` all exist with matching signatures).
Rendering output has not been visually verified this cycle.

## `Map`

Interactive map viewer with raster/vector overlays.

```python
from pygeovision.viz import Map

m = Map(center=(5.6, -0.2), zoom=12, basemap="satellite")
m.add_raster("scene.tif", colormap="RdYlGn", band=0, opacity=0.8)
m.add_vector("buildings.geojson", color="red")
m.export("map.html")

# Split view
split = Map.split_view("before.tif", "after.tif")
split.export("comparison.html")
```

## `RasterViewer`

```python
from pygeovision.viz import RasterViewer

rv = RasterViewer("sentinel2.tif")
rv.rgb(red=2, green=1, blue=0).show()          # True-colour
rv.false_color(nir=3, red=2, green=1).show()   # NIR false-colour
rv.ndvi(nir_band=3, red_band=2).export("ndvi.png")
rv.histogram(band=0).show()
rv.profile(start_px=(100,100), end_px=(200,300)).show()
```

## `VectorViewer`

```python
from pygeovision.viz import VectorViewer

vv = VectorViewer("buildings.geojson")
vv.view().show()
vv.table(max_rows=20)
vv.style_by_attribute("area", cmap="Reds").export("choropleth.png")
filtered = vv.filter("area > 500")
vv.heatmap().export("heatmap.png")
```

## `ChangeViewer`

```python
from pygeovision.viz import ChangeViewer

cv = ChangeViewer("before.tif", "after.tif")
cv.split().show()
cv.difference().export("diff.png")
stats = cv.statistics("change_mask.tif")
# stats: {"pct_increased": 12.3, "pct_decreased": 4.5, ...}
```

## `TimeSeriesViewer`

```python
from pygeovision.viz import TimeSeriesViewer

tsv = TimeSeriesViewer(
    paths=["ndvi_2021.tif", "ndvi_2022.tif", "ndvi_2023.tif"],
    dates=["2021-07-01", "2022-07-01", "2023-07-01"],
    colormap="RdYlGn",
)
tsv.animate("ndvi.gif", fps=2)
tsv.trend(title="NDVI Trend").export("trend.png")
tsv.seasonal(period=12).show()
tsv.anomaly(n_sigma=2.0).export("anomalies.png")
tsv.mosaic(max_cols=3).export("mosaic.png")
```
