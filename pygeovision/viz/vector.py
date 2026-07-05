"""
pygeovision.viz.vector
=======================
Vector data visualization: GeoJSON, Shapefile, GeoParquet.

Usage::

    from pygeovision.viz import VectorViewer

    vv = VectorViewer("buildings.geojson")
    vv.view()                                    # basic map
    vv.table(max_rows=20)                        # attribute table
    vv.style_by_attribute("area", cmap="Reds")   # choropleth
    vv.filter("area > 1000")                     # spatial filter
    vv.heatmap(lat_col="lat", lon_col="lon")     # heatmap
    vv.export("buildings_map.html")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeovision.viz.vector")


def _read_geodata(path: str) -> Tuple[Any, dict]:
    """Read a vector file and return (GeoDataFrame, geojson_dict)."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        geojson = json.loads(gdf.to_json())
        return gdf, geojson
    except ImportError:
        # Fallback: load GeoJSON directly
        with open(path) as f:
            geojson = json.load(f)
        return None, geojson


class VectorViewer:
    """
    Vector data viewer with choropleth, filtering, and attribute table.

    Parameters
    ----------
    path : str | dict
        Path to GeoJSON / Shapefile, or a GeoJSON dict.
    """

    def __init__(self, path: Union[str, dict]) -> None:
        self._path   = path
        self._gdf    = None
        self._geojson: Optional[dict] = None
        self._fig    = None
        self._filter_expr: Optional[str] = None

    def _load(self) -> Tuple[Any, dict]:
        if self._geojson is not None:
            return self._gdf, self._geojson
        if isinstance(self._path, dict):
            self._geojson = self._path
        else:
            self._gdf, self._geojson = _read_geodata(str(self._path))
        return self._gdf, self._geojson

    # ── Basic view ─────────────────────────────────────────────────────────────

    def view(self, color: str = "#3388ff", fill_opacity: float = 0.4,
             title: Optional[str] = None) -> "VectorViewer":
        plt = _mpl()
        gdf, _ = self._load()
        if gdf is not None:
            if getattr(self, "_fig", None) is not None:
                plt.close(self._fig)
                self._fig = None
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(ax=ax, color=color, alpha=fill_opacity, edgecolor="white", linewidth=0.5)
            ax.set_title(title or _layer_name(self._path), fontsize=13)
            ax.axis("off")
            self._fig = fig
            plt.tight_layout()
        else:
            logger.info("Install geopandas for interactive vector plots: pip install geopandas")
        return self

    # ── Attribute table ────────────────────────────────────────────────────────

    def table(self, max_rows: int = 20) -> "VectorViewer":
        """Display attribute table."""
        gdf, geojson = self._load()
        if gdf is not None:
            print(gdf.head(max_rows).to_string(index=False))
            print(f"\n({len(gdf)} features, {len(gdf.columns)} columns)")
        else:
            features = geojson.get("features", [])[:max_rows]
            if features:
                cols = list(features[0].get("properties", {}).keys())
                print("\t".join(cols))
                for feat in features:
                    props = feat.get("properties", {})
                    print("\t".join(str(props.get(c, "")) for c in cols))
        return self

    # ── Choropleth ─────────────────────────────────────────────────────────────

    def style_by_attribute(self, attribute: str, cmap: str = "viridis",
                            title: Optional[str] = None) -> "VectorViewer":
        """Colour features by a numeric attribute (choropleth map)."""
        plt = _mpl()
        gdf, _ = self._load()
        if gdf is None:
            raise ImportError("pip install geopandas")
        if attribute not in gdf.columns:
            raise ValueError(f"Attribute '{attribute}' not in columns: {list(gdf.columns)}")
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column=attribute, ax=ax, cmap=cmap, legend=True,
                 legend_kwds={"label": attribute, "shrink": 0.6},
                 edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_title(title or f"{attribute} choropleth", fontsize=13)
        ax.axis("off")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Filter ─────────────────────────────────────────────────────────────────

    def filter(self, expression: str) -> "VectorViewer":
        """Filter features using a query expression (GeoPandas .query() syntax)."""
        gdf, _ = self._load()
        if gdf is None:
            raise ImportError("pip install geopandas")
        filtered = gdf.query(expression)
        result = VectorViewer(json.loads(filtered.to_json()))
        result._gdf = filtered
        logger.info("Filter '%s' → %d / %d features", expression, len(filtered), len(gdf))
        return result

    # ── Heatmap ────────────────────────────────────────────────────────────────

    def heatmap(self, lat_col: str = "latitude", lon_col: str = "longitude",
                bandwidth: float = 0.01, title: str = "Density Heatmap") -> "VectorViewer":
        plt = _mpl()
        gdf, _ = self._load()
        if gdf is None:
            raise ImportError("pip install geopandas")

        if lat_col in gdf.columns and lon_col in gdf.columns:
            lats = gdf[lat_col].astype(float).values
            lons = gdf[lon_col].astype(float).values
        else:
            # Extract centroids from geometry
            try:
                centroids = gdf.geometry.centroid
                lats = centroids.y.values
                lons = centroids.x.values
            except Exception:
                raise ValueError(f"Cannot extract coordinates from {list(gdf.columns)}")

        try:
            from scipy.stats import gaussian_kde
            kernel = gaussian_kde(np.vstack([lons, lats]), bw_method=bandwidth)
            xl = np.linspace(lons.min(), lons.max(), 200)
            yl = np.linspace(lats.min(), lats.max(), 200)
            XX, YY = np.meshgrid(xl, yl)
            ZZ = kernel(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            if getattr(self, "_fig", None) is not None:
                plt.close(self._fig)
                self._fig = None
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.pcolormesh(XX, YY, ZZ, cmap="hot_r", shading="auto")
            ax.scatter(lons, lats, s=2, color="white", alpha=0.3)
            plt.colorbar(im, ax=ax, label="Density")
        except ImportError:
            if getattr(self, "_fig", None) is not None:
                plt.close(self._fig)
                self._fig = None
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(lons, lats, alpha=0.5, s=10, c="red")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Export ─────────────────────────────────────────────────────────────────

    def export(self, path: str, dpi: int = 150, format: str = "auto") -> str:
        suffix = Path(path).suffix.lower()
        if suffix in (".html", "") and format == "auto":
            return self._export_html(path)
        plt = _mpl()
        if self._fig is None:
            self.view()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def _export_html(self, path: str) -> str:
        from pygeovision.viz.map import Map
        m = Map()
        _, geojson = self._load()
        m.add_vector(geojson, name=_layer_name(self._path))
        return m.export(path)

    def show(self) -> "VectorViewer":
        plt = _mpl()
        if self._fig:
            plt.show()
        return self

    def __repr__(self) -> str:
        return f"VectorViewer({_layer_name(self._path)!r})"


def _mpl():
    import matplotlib
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            matplotlib.use("Agg")
    except Exception:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _layer_name(path) -> str:
    if isinstance(path, str):
        return Path(path).stem
    return "vector"
