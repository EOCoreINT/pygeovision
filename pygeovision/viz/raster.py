"""
pygeovision.viz.raster
=======================
Raster visualization tools — RGB composites, false-color, spectral indices,
histograms, transect profiles, and band-combination tools.

Usage::

    from pygeovision.viz import RasterViewer

    rv = RasterViewer("sentinel2_scene.tif")
    rv.rgb(red=0, green=1, blue=2)          # True-colour composite
    rv.false_color(nir=3, red=0, green=1)   # NIR false-colour
    rv.ndvi(nir_band=3, red_band=0)         # NDVI map
    rv.histogram(band=0)                     # Per-band histogram
    rv.profile(start=(lat1,lon1), end=(lat2,lon2))  # Transect profile
    rv.show()
    rv.export("output.png", dpi=300)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeovision.viz.raster")

_MPL_BACKEND_SET = False


def _get_plt():
    global _MPL_BACKEND_SET
    import matplotlib
    if not _MPL_BACKEND_SET:
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                matplotlib.use("module://matplotlib_inline.backend_inline")
            else:
                matplotlib.use("Agg")
        except Exception:
            matplotlib.use("Agg")
        _MPL_BACKEND_SET = True
    import matplotlib.pyplot as plt
    return plt


class RasterViewer:
    """
    Comprehensive raster viewer for single-file or multi-band GeoTIFF scenes.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF file.
    figsize : tuple
        Default figure size (width, height) in inches.
    """

    def __init__(self, path: str, figsize: Tuple[float, float] = (10, 8)) -> None:
        self.path    = path
        self.figsize = figsize
        self._fig    = None
        self._cache: dict = {}

    def _read(self, bands: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """Read and cache raster data."""
        key = str(bands)
        if key in self._cache:
            return self._cache[key]
        try:
            import rasterio
        except ImportError:
            raise ImportError("pip install rasterio")
        with rasterio.open(self.path) as src:
            if bands is None:
                data = src.read().astype("float32")
            elif isinstance(bands, int):
                data = src.read(bands + 1).astype("float32")
            else:
                data = np.stack([src.read(b + 1) for b in bands]).astype("float32")
        self._cache[key] = data
        return data

    def _normalize(self, arr: np.ndarray, pct: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """Percentile stretch to [0, 1]."""
        lo = np.nanpercentile(arr, pct[0])
        hi = np.nanpercentile(arr, pct[1])
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    # ── Composites ─────────────────────────────────────────────────────────────

    def rgb(self, red: int = 2, green: int = 1, blue: int = 0,
            title: str = "True-Colour RGB") -> "RasterViewer":
        """Display a true-colour RGB composite."""
        plt = _get_plt()
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        data = self._read([red, green, blue])
        rgb_arr = np.dstack([self._normalize(data[i]) for i in range(3)])
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(rgb_arr)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        self._fig = fig
        plt.tight_layout()
        return self

    def false_color(self, nir: int = 3, red: int = 2, green: int = 1,
                    title: str = "False-Colour (NIR-R-G)") -> "RasterViewer":
        """Display a false-colour composite (NIR / Red / Green)."""
        return self.rgb(red=nir, green=red, blue=green, title=title)

    def swir_composite(self, swir1: int = 4, nir: int = 3, blue: int = 0,
                        title: str = "SWIR Composite") -> "RasterViewer":
        """Display a SWIR composite highlighting burn scars and geology."""
        return self.rgb(red=swir1, green=nir, blue=blue, title=title)

    # ── Index maps ─────────────────────────────────────────────────────────────

    def ndvi(self, nir_band: int = 3, red_band: int = 2,
             title: str = "NDVI") -> "RasterViewer":
        plt = _get_plt()
        data = self._read([nir_band, red_band])
        nir, red = data[0], data[1]
        index = (nir - red) / (nir + red + 1e-8)
        return self._plot_index(index, cmap="RdYlGn", vmin=-0.2, vmax=0.8, title=title)

    def ndwi(self, green_band: int = 1, nir_band: int = 3,
             title: str = "NDWI") -> "RasterViewer":
        data = self._read([green_band, nir_band])
        green, nir = data[0], data[1]
        index = (green - nir) / (green + nir + 1e-8)
        return self._plot_index(index, cmap="Blues", vmin=-0.5, vmax=0.5, title=title)

    def ndbi(self, swir_band: int = 4, nir_band: int = 3,
             title: str = "NDBI") -> "RasterViewer":
        data = self._read([swir_band, nir_band])
        swir, nir = data[0], data[1]
        index = (swir - nir) / (swir + nir + 1e-8)
        return self._plot_index(index, cmap="YlOrBr", vmin=-0.5, vmax=0.5, title=title)

    def _plot_index(self, arr: np.ndarray, cmap: str, vmin: float, vmax: float,
                    title: str) -> "RasterViewer":
        plt = _get_plt()
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=title, fraction=0.03)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        self._fig = fig
        plt.tight_layout()
        return self

    def single_band(self, band: int = 0, colormap: str = "gray",
                    title: Optional[str] = None) -> "RasterViewer":
        plt = _get_plt()
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        data = self._read(band)
        fig, ax = plt.subplots(figsize=self.figsize)
        vmin, vmax = float(np.nanpercentile(data, 2)), float(np.nanpercentile(data, 98))
        im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(title or f"Band {band}", fontsize=13)
        ax.axis("off")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Histogram ──────────────────────────────────────────────────────────────

    def histogram(self, band: Optional[int] = None, bins: int = 128,
                  title: Optional[str] = None) -> "RasterViewer":
        plt = _get_plt()
        all_data = self._read()
        if all_data.ndim == 2:
            all_data = all_data[np.newaxis]
        bands_to_plot = [band] if band is not None else range(min(all_data.shape[0], 6))
        n = len(list(bands_to_plot))
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        colors = plt.cm.tab10.colors  # type: ignore
        for i, b in enumerate(bands_to_plot):
            data_b = all_data[b].ravel()
            data_b = data_b[np.isfinite(data_b)]
            axes[i].hist(data_b, bins=bins, color=colors[i % 10], edgecolor="none", alpha=0.85)
            axes[i].set_title(f"Band {b}", fontsize=11)
            axes[i].set_xlabel("Pixel value")
            axes[i].set_ylabel("Count")
            axes[i].grid(axis="y", alpha=0.3)
        fig.suptitle(title or "Band Histogram", fontsize=13)
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Profile (transect) ─────────────────────────────────────────────────────

    def profile(self, start_px: Tuple[int, int], end_px: Tuple[int, int],
                bands: Optional[List[int]] = None,
                title: str = "Spectral Profile") -> "RasterViewer":
        """Plot pixel values along a line (transect)."""
        plt = _get_plt()
        data = self._read()
        if data.ndim == 2:
            data = data[np.newaxis]

        r0, c0 = start_px
        r1, c1 = end_px
        n_points = max(abs(r1 - r0), abs(c1 - c0), 10)
        rows = np.linspace(r0, r1, n_points).astype(int)
        cols = np.linspace(c0, c1, n_points).astype(int)

        rows = np.clip(rows, 0, data.shape[1] - 1)
        cols = np.clip(cols, 0, data.shape[2] - 1)

        n_b = data.shape[0]
        bands_to_plot = bands or list(range(min(n_b, 6)))

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=self.figsize)
        for b in bands_to_plot:
            profile_vals = data[b][rows, cols]
            ax.plot(profile_vals, label=f"Band {b}", linewidth=1.5)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Position along transect (pixels)")
        ax.set_ylabel("Pixel value")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Output ─────────────────────────────────────────────────────────────────

    def show(self) -> "RasterViewer":
        plt = _get_plt()
        if self._fig is not None:
            plt.show()
        return self

    def export(self, path: str, dpi: int = 150) -> str:
        plt = _get_plt()
        if self._fig is None:
            raise RuntimeError("Nothing to export — call rgb(), ndvi(), histogram(), etc. first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Raster view exported to %s", path)
        return path

    def __repr__(self) -> str:
        return f"RasterViewer({Path(self.path).name!r})"
