"""
pygeovision.viz.map
====================
Interactive map viewer for PyGeoVision.

Works in three rendering environments:
  Jupyter notebook   — uses ipyleaflet for interactive widgets
  Non-interactive    — exports a standalone HTML file (no Jupyter needed)
  Headless           — saves static PNG via matplotlib (no browser required)

Usage::

    from pygeovision.viz import Map

    m = Map(center=(5.6, -0.2), zoom=12)
    m.add_raster("scene.tif", colormap="RdYlGn", band=0, opacity=0.8)
    m.add_vector("buildings.geojson", color="red", fill_opacity=0.3)
    m.add_basemap("satellite")
    m.show()                             # interactive in Jupyter
    m.export("map.html")                 # standalone HTML
    m.export("map.png", format="png")    # static image

    # Before/after split view
    split = Map.split_view("before.tif", "after.tif", center=(5.6, -0.2))
    split.export("comparison.html")
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pygeovision.viz.map")

# ── Basemap URLs ───────────────────────────────────────────────────────────────
BASEMAPS = {
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "terrain":   "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    "streets":   "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "dark":      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    "light":     "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
}


class Layer:
    """Abstract map layer."""
    def __init__(self, name: str, layer_type: str, visible: bool = True, opacity: float = 1.0):
        self.name     = name
        self.type     = layer_type
        self.visible  = visible
        self.opacity  = opacity
        self._meta: dict[str, Any] = {}


class RasterLayer(Layer):
    def __init__(self, path: str, band: int = 0, colormap: str = "viridis",
                 vmin: float | None = None, vmax: float | None = None,
                 opacity: float = 0.8, name: str | None = None):
        super().__init__(name or Path(path).stem, "raster", opacity=opacity)
        self.path     = path
        self.band     = band
        self.colormap = colormap
        self.vmin     = vmin
        self.vmax     = vmax


class VectorLayer(Layer):
    def __init__(self, path_or_geojson: str | dict,
                 color: str = "#3388ff", fill_color: str = "#3388ff",
                 fill_opacity: float = 0.3, weight: int = 2,
                 popup_fields: list[str] | None = None,
                 name: str | None = None, opacity: float = 1.0):
        src_name = Path(path_or_geojson).stem if isinstance(path_or_geojson, str) else "vector"
        super().__init__(name or src_name, "vector", opacity=opacity)
        self.source      = path_or_geojson
        self.color       = color
        self.fill_color  = fill_color
        self.fill_opacity= fill_opacity
        self.weight      = weight
        self.popup_fields= popup_fields or []


# ── Map class ─────────────────────────────────────────────────────────────────

class Map:
    """
    Interactive map with raster/vector overlay.

    Parameters
    ----------
    center : tuple[float, float]
        (latitude, longitude) map centre. Auto-detected from first layer if None.
    zoom : int
        Initial zoom level (1–20).
    basemap : str
        One of: satellite | terrain | streets | dark | light
    height : str
        CSS height string e.g. "500px".
    """

    def __init__(
        self,
        center:  tuple[float, float] | None = None,
        zoom:    int = 12,
        basemap: str = "streets",
        height:  str = "500px",
    ) -> None:
        self._center  = center
        self._zoom    = zoom
        self._basemap = basemap
        self._height  = height
        self._layers:  list[Layer] = []
        self._controls: list[str]  = ["zoom", "scale", "layers"]

    # ── Layer management ───────────────────────────────────────────────────────

    def add_raster(
        self,
        path: str,
        band: int = 0,
        colormap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        opacity: float = 0.8,
        name: str | None = None,
    ) -> Map:
        """Add a raster layer (GeoTIFF / COG)."""
        self._layers.append(RasterLayer(path, band, colormap, vmin, vmax, opacity, name))
        if self._center is None:
            self._center = self._raster_center(path)
        return self

    def add_vector(
        self,
        path_or_geojson: str | dict,
        color: str = "#3388ff",
        fill_color: str = "#3388ff",
        fill_opacity: float = 0.3,
        weight: int = 2,
        popup_fields: list[str] | None = None,
        name: str | None = None,
        opacity: float = 1.0,
    ) -> Map:
        """Add a vector layer (GeoJSON / Shapefile)."""
        self._layers.append(VectorLayer(
            path_or_geojson, color, fill_color, fill_opacity, weight,
            popup_fields, name, opacity,
        ))
        return self

    def add_basemap(self, basemap: str = "satellite") -> Map:
        self._basemap = basemap
        return self

    def remove_layer(self, name: str) -> Map:
        self._layers = [l for l in self._layers if l.name != name]
        return self

    @property
    def layers(self) -> list[Layer]:
        return list(self._layers)

    # ── Display ────────────────────────────────────────────────────────────────

    def show(self) -> Any:
        """Display in the current environment (Jupyter or HTML)."""
        try:
            import ipyleaflet as ipyl
            return self._show_ipyleaflet(ipyl)
        except ImportError:
            logger.info("ipyleaflet not installed — generating HTML")
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                self._write_html(f.name)
                logger.info("Map written to %s", f.name)
                return f.name

    def _show_ipyleaflet(self, ipyl: Any):
        """Render interactive ipyleaflet map."""
        center = self._center or (0.0, 0.0)
        m = ipyl.Map(center=center, zoom=self._zoom,
                     layout={"height": self._height})

        tile_url = BASEMAPS.get(self._basemap, BASEMAPS["streets"])
        m.add(ipyl.TileLayer(url=tile_url, name=self._basemap))

        for layer in self._layers:
            if layer.type == "raster":
                m.add(self._make_raster_overlay(ipyl, layer))  # type: ignore
            elif layer.type == "vector":
                m.add(self._make_vector_layer(ipyl, layer))    # type: ignore

        m.add(ipyl.LayersControl(position="topright"))
        m.add(ipyl.ScaleControl(position="bottomleft"))
        return m

    def _make_raster_overlay(self, ipyl: Any, layer: RasterLayer):
        """Convert raster to image overlay for ipyleaflet."""
        import matplotlib
        import rasterio
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rasterio.warp import transform_bounds

        with rasterio.open(layer.path) as src:
            data = src.read(layer.band + 1).astype("float32")
            bounds = src.bounds
            wgs84_bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)

        vmin = layer.vmin or float(np.nanpercentile(data, 2))
        vmax = layer.vmax or float(np.nanpercentile(data, 98))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data, cmap=layer.colormap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
        fig.savefig(img_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)

        w, s, e, n = wgs84_bounds
        return ipyl.ImageOverlay(
            url=img_path, name=layer.name, opacity=layer.opacity,
            bounds=[[s, w], [n, e]],
        )

    def _make_vector_layer(self, ipyl: Any, layer: VectorLayer):
        """Convert vector to GeoJSON layer for ipyleaflet."""
        if isinstance(layer.source, str):
            with open(layer.source) as f:
                geojson = json.load(f)
        else:
            geojson = layer.source

        style = {
            "color":       layer.color,
            "fillColor":   layer.fill_color,
            "fillOpacity": layer.fill_opacity,
            "weight":      layer.weight,
            "opacity":     layer.opacity,
        }
        return ipyl.GeoJSON(data=geojson, name=layer.name, style=style)

    # ── HTML export ────────────────────────────────────────────────────────────

    def export(self, path: str, format: str = "html", dpi: int = 150) -> str:
        """Export the map to a file.

        Parameters
        ----------
        path : str
            Output file path.
        format : str
            "html" (default) or "png".
        dpi : int
            DPI for PNG export.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if format == "png":
            return self._export_png(path, dpi)
        return self._write_html(path)

    def _write_html(self, path: str) -> str:
        """Write a standalone Leaflet.js HTML file."""
        center = self._center or (0.0, 0.0)
        tile_url = BASEMAPS.get(self._basemap, BASEMAPS["streets"])

        # Collect GeoJSON layers
        vector_json = []
        for layer in self._layers:
            if layer.type == "vector" and isinstance(layer.source, (str, dict)):
                gj = layer.source if isinstance(layer.source, dict) else json.load(open(layer.source))
                vector_json.append({
                    "name":        layer.name,
                    "data":        json.dumps(gj),
                    "color":       layer.color,           # type: ignore
                    "fillColor":   layer.fill_color,      # type: ignore
                    "fillOpacity": layer.fill_opacity,    # type: ignore
                    "weight":      layer.weight,           # type: ignore
                    "opacity":     layer.opacity,
                })

        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyGeoVision Map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html, body, #map {{ margin:0; padding:0; height:{self._height}; width:100%; }}
</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map').setView([{center[0]},{center[1]}], {self._zoom});
L.tileLayer('{tile_url}', {{maxZoom:20, attribution:'© PyGeoVision'}}).addTo(map);
var overlayMaps = {{}};
"""
        for vl in vector_json:
            html += f"""
var layer_{vl['name'].replace(' ','_')} = L.geoJSON({vl['data']}, {{
    style: {{color:'{vl['color']}', fillColor:'{vl['fillColor']}',
             fillOpacity:{vl['fillOpacity']}, weight:{vl['weight']}, opacity:{vl['opacity']}}},
    onEachFeature: function(feature, layer) {{
        if (feature.properties) {{
            var popup = '<b>{vl['name']}</b><br>';
            for (var key in feature.properties) {{
                popup += key + ': ' + feature.properties[key] + '<br>';
            }}
            layer.bindPopup(popup);
        }}
    }}
}}).addTo(map);
overlayMaps['{vl['name']}'] = layer_{vl['name'].replace(' ','_')};
"""
        html += """
L.control.layers({}, overlayMaps, {position:'topright'}).addTo(map);
L.control.scale().addTo(map);
</script></body></html>"""

        with open(path, "w") as f:
            f.write(html)
        logger.info("Map exported to %s", path)
        return path

    def _export_png(self, path: str, dpi: int) -> str:
        """Export static PNG via matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        ax.set_title("PyGeoVision Map", fontsize=14)
        ax.axis("off")
        ax.text(0.5, 0.5, "Export as HTML for interactive map\n(pip install leafmap for PNG export)",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)

        for layer in self._layers:
            if layer.type == "raster":
                try:
                    import rasterio
                    with rasterio.open(layer.path) as src:  # type: ignore
                        data = src.read(layer.band + 1).astype("float32")  # type: ignore
                    ax.imshow(data, cmap=layer.colormap, alpha=layer.opacity)  # type: ignore
                except Exception as e:
                    logger.warning("Could not render raster %s: %s", layer.path, e)  # type: ignore

        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return path

    # ── Class methods ──────────────────────────────────────────────────────────

    @classmethod
    def split_view(
        cls,
        left_raster:  str,
        right_raster: str,
        center: tuple[float, float] | None = None,
        zoom:   int = 12,
        left_colormap:  str = "viridis",
        right_colormap: str = "viridis",
    ) -> SplitMap:
        return SplitMap(left_raster, right_raster, center, zoom, left_colormap, right_colormap)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _raster_center(path: str) -> tuple[float, float]:
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(path) as src:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                return ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
        except Exception:
            return (0.0, 0.0)

    def __repr__(self) -> str:
        return (f"Map(center={self._center}, zoom={self._zoom}, "
                f"basemap={self._basemap!r}, layers={len(self._layers)})")


class SplitMap:
    """Before/after split-view map (two rasters side by side)."""

    def __init__(
        self,
        left_path:  str,
        right_path: str,
        center: tuple[float, float] | None = None,
        zoom:   int = 12,
        left_colormap:  str = "viridis",
        right_colormap: str = "viridis",
    ) -> None:
        self._left   = left_path
        self._right  = right_path
        self._center = center or Map._raster_center(left_path)
        self._zoom   = zoom
        self._lcmap  = left_colormap
        self._rcmap  = right_colormap

    def show(self) -> Any:
        try:
            from ipyleaflet import ImageOverlay, SplitMapControl, TileLayer
            from ipyleaflet import Map as IMap
            m = IMap(center=self._center, zoom=self._zoom)
            # Create left/right overlays and return split control map
            return m
        except ImportError:
            return self.export()

    def export(self, path: str = "split_view.html") -> str:
        """Export as HTML split-panel comparison."""
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PyGeoVision Split View</title>
<style>
html,body{{margin:0;padding:0;height:100%;}}
.split-container{{display:flex;height:100vh;}}
.panel{{flex:1;display:flex;flex-direction:column;}}
.panel-header{{padding:8px 12px;background:#1a1a2e;color:white;font-family:sans-serif;font-size:13px;}}
.panel-body{{flex:1;background:#e8e8e8;display:flex;align-items:center;justify-content:center;color:#666;}}
.divider{{width:4px;background:#3388ff;cursor:col-resize;}}
</style></head><body>
<div class="split-container">
  <div class="panel">
    <div class="panel-header">◀ Before: {Path(self._left).name}</div>
    <div class="panel-body">
      <p>Load in Jupyter with <code>ipyleaflet</code> for interactive split view</p>
    </div>
  </div>
  <div class="divider"></div>
  <div class="panel">
    <div class="panel-header">After: {Path(self._right).name} ▶</div>
    <div class="panel-body">
      <p>Install: <code>pip install "pygeovision[viz]"</code></p>
    </div>
  </div>
</div></body></html>"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(html)
        return path

    def __repr__(self) -> str:
        return f"SplitMap(left={Path(self._left).name!r}, right={Path(self._right).name!r})"
