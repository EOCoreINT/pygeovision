"""
pygeovision.viz
================
Visualization layer for PyGeoVision.

Quick reference::

    from pygeovision.viz import Map, RasterViewer, VectorViewer, ChangeViewer, TimeSeriesViewer

    # Interactive map
    m = Map(center=(5.6, -0.2), zoom=12, basemap="satellite")
    m.add_raster("scene.tif", colormap="RdYlGn")
    m.add_vector("buildings.geojson", color="red")
    m.export("map.html")

    # Raster viewer
    rv = RasterViewer("sentinel2.tif")
    rv.rgb(red=2, green=1, blue=0).show()
    rv.ndvi(nir_band=3, red_band=2).export("ndvi.png")
    rv.histogram().show()

    # Vector viewer
    vv = VectorViewer("buildings.geojson")
    vv.style_by_attribute("area").show()
    vv.table(max_rows=20)

    # Change viewer
    cv = ChangeViewer("before.tif", "after.tif")
    cv.split().show()
    cv.difference().export("diff.png")
    stats = cv.statistics("change_mask.tif")

    # Time-series viewer
    tsv = TimeSeriesViewer(
        paths=["ndvi_2020.tif", "ndvi_2021.tif", "ndvi_2022.tif"],
        dates=["2020-07-01", "2021-07-01", "2022-07-01"],
        colormap="RdYlGn",
    )
    tsv.animate("ndvi.gif")
    tsv.trend().show()
    tsv.anomaly().export("anomalies.png")

    # Before/after split map
    split = Map.split_view("before.tif", "after.tif")
    split.export("comparison.html")
"""

from pygeovision.viz.map        import Map, SplitMap, Layer, RasterLayer, VectorLayer
from pygeovision.viz.raster     import RasterViewer
from pygeovision.viz.vector     import VectorViewer
from pygeovision.viz.change     import ChangeViewer
from pygeovision.viz.timeseries import TimeSeriesViewer

__all__ = [
    # Map viewer
    "Map", "SplitMap", "Layer", "RasterLayer", "VectorLayer",
    # Specialized viewers
    "RasterViewer",
    "VectorViewer",
    "ChangeViewer",
    "TimeSeriesViewer",
]
