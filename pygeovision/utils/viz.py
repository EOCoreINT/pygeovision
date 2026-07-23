"""
pygeovision.utils.viz
=====================
Dark-themed matplotlib helpers for pygeovision project notebooks.

Replaces these patterns that appear in ALL 6 notebooks:

Pattern 1 — figure setup (identical across every notebook)::

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0D1B2A")
    for ax in axes.ravel():
        ax.set_facecolor("#1C2C3C"); ax.axis("off")

Pattern 2 — show panel with colorbar (repeated 3–6 times per notebook)::

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cb.ax.tick_params(colors="white", labelsize=7)
    if cbar_label: cb.set_label(cbar_label, color="white", fontsize=7)
    if note:
        ax.text(0.02, 0.03, note, ...)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.axis("off")

After::

    from pygeovision.utils import DarkFigure, show_panel

    with DarkFigure(2, 3, figsize=(20, 12)) as (fig, axes):
        show_panel(axes[0,0], ndvi, "RdYlGn", -0.1, 0.9, "NDVI 2024")
        show_panel(axes[0,1], risk_map, risk_cmap, 0, 4, "Risk Map",
                   cbar=False, note="4-class RF")
        fig.suptitle("My Analysis", color="white")
"""
from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# EOCoreINT dark palette (consistent across all notebooks)
BG_DARK   = "#0D1B2A"
BG_PANEL  = "#1C2C3C"
BG_SPINE  = "#546E7A"
TXT_WHITE = "white"
TXT_MUTED = "#90A4AE"
TXT_SUB   = "#B0BEC5"


class DarkFigure:
    """
    Context manager for a dark-themed pygeovision matplotlib figure.

    Replaces the 4-line figure setup at the start of every visualisation cell::

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.patch.set_facecolor("#0D1B2A")
        for ax in axes.ravel():
            ax.set_facecolor("#1C2C3C"); ax.axis("off")

    Usage::

        from pygeovision.utils import DarkFigure, show_panel

        with DarkFigure(2, 3, figsize=(20, 12)) as (fig, axes):
            show_panel(axes[0,0], ndvi_arr,  "RdYlGn", -0.1, 0.9, "NDVI")
            show_panel(axes[0,1], risk_map,  risk_cmap, 0, 4, "Risk Zones")
            fig.suptitle("Analysis Title", color="white", fontsize=13,
                         fontweight="bold", y=1.01)
        # fig.savefig("result.png", ...) after the with-block
    """

    def __init__(
        self,
        nrows:    int = 1,
        ncols:    int = 1,
        figsize:  tuple[float, float] | None = None,
        hspace:   float = 0.35,
        wspace:   float = 0.20,
        gridspec_kw: dict | None = None,
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize or (ncols * 5.5, nrows * 5.5)
        self.hspace  = hspace
        self.wspace  = wspace
        self.gridspec_kw = gridspec_kw or {}
        self.fig  = None
        self.axes = None

    def __enter__(self):
        gskw = {"hspace": self.hspace, "wspace": self.wspace,
                **self.gridspec_kw}
        self.fig, self.axes = plt.subplots(
            self.nrows, self.ncols,
            figsize      = self.figsize,
            gridspec_kw  = gskw,
        )
        self.fig.patch.set_facecolor(BG_DARK)
        # Normalise to always be an ndarray of axes
        ax_arr = np.array(self.axes).ravel()
        for ax in ax_arr:
            ax.set_facecolor(BG_PANEL)
            ax.axis("off")
        return self.fig, self.axes

    def __exit__(self, *args):
        plt.tight_layout()

    def save(
        self,
        path:  str | pathlib.Path,
        dpi:   int = 150,
        title: str = "",
        **kwargs,
    ) -> str:
        """Save the figure and return the path string."""
        if title:
            self.fig.suptitle(title, color=TXT_WHITE, fontsize=13,
                              fontweight="bold", y=1.01)
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(
            str(path), dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
            **kwargs,
        )
        print(f"Saved: {path}")
        plt.show()
        return str(path)


def show_panel(
    ax:           Any,
    data:         np.ndarray | None,
    cmap:         str | Any,
    vmin:         float,
    vmax:         float,
    title:        str,
    cbar:         bool = True,
    cbar_label:   str = "",
    note:         str = "",
    contour_val:  float | None = None,
    contour_color:str = "yellow",
    contour_lw:   float = 1.2,
    title_color:  str = TXT_WHITE,
    title_size:   int = 9,
) -> None:
    """
    Render one imshow panel with a white colorbar and optional annotations.

    Replaces the 10–12 line block repeated 3–6 times per notebook::

        if data is None:
            ax.text(0.5, 0.5, "No data", ...)
            ax.set_title(title, color="#90A4AE", ...)
            return
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
        cb.ax.tick_params(colors="white", labelsize=7)
        if cbar_label: cb.set_label(cbar_label, color="white", fontsize=7)
        if note:
            ax.text(0.02, 0.03, note, transform=ax.transAxes, ...)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
        ax.axis("off")

    Args:
        ax:            Matplotlib axes object.
        data:          ``(H, W)`` or ``(C, H, W)`` float array. ``None`` → placeholder.
        cmap:          Colormap name or ListedColormap.
        vmin, vmax:    Display range.
        title:         Panel title text.
        cbar:          Show colorbar. Default ``True``.
        cbar_label:    Colorbar axis label.
        note:          Small annotation at bottom-left (e.g. metric values).
        contour_val:   If set, draw a contour at this value (e.g. shoreline at 0).
        contour_color: Contour line colour. Default ``"yellow"``.
        contour_lw:    Contour line width.
        title_color:   Title text colour.
        title_size:    Title font size.

    Example::

        from pygeovision.utils import show_panel

        show_panel(axes[0,0], ndvi, "RdYlGn", -0.1, 0.9,
                   "NDVI 2024", cbar_label="NDVI",
                   note=f"mean={ndvi.mean():.3f}")

        show_panel(axes[0,1], mndwi, "RdBu", -0.6, 0.6,
                   "MNDWI Shoreline", contour_val=0.0,
                   contour_color="yellow")

        show_panel(axes[0,2], None, "gray", 0, 1, "Pending download")
    """
    ax.set_facecolor(BG_PANEL)

    if data is None:
        ax.text(0.5, 0.5, "No data\n(check download)",
                ha="center", va="center",
                color=TXT_MUTED, fontsize=10,
                transform=ax.transAxes)
        ax.set_title(title, color=TXT_MUTED, fontsize=title_size, pad=6)
        ax.axis("off")
        return

    # Flatten to 2-D for display
    arr = data if data.ndim == 2 else data[0]

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
        cb.ax.tick_params(colors=TXT_WHITE, labelsize=7)
        if cbar_label:
            cb.set_label(cbar_label, color=TXT_WHITE, fontsize=7)

    if contour_val is not None:
        try:
            ax.contour(arr, levels=[contour_val],
                       colors=[contour_color], linewidths=[contour_lw])
        except Exception:
            pass

    if note:
        ax.text(0.02, 0.03, note,
                transform=ax.transAxes,
                fontsize=7, color=TXT_SUB, va="bottom",
                bbox=dict(facecolor=BG_DARK, alpha=0.75,
                          pad=2, edgecolor="none"))

    ax.set_title(title, color=title_color, fontsize=title_size,
                 fontweight="bold", pad=6)
    ax.axis("off")


def add_station_markers(
    ax:       Any,
    stations: dict,
    forecast: dict | None = None,
    low_color:  str = "#2ecc71",
    mid_color:  str = "#f1c40f",
    high_color: str = "#e74c3c",
    threshold_high: float = 0.6,
    threshold_mid:  float = 0.4,
    ms:       int = 9,
) -> None:
    """
    Plot GNSS / weather station markers on a map panel.

    Used by IonoForecaster to overlay station dots on TEC maps.

    Args:
        ax:             Matplotlib axes with geographic extent set.
        stations:       Dict ``{code: {"lon": ..., "lat": ..., "city": ...}}``.
        forecast:       Dict ``{code: float}`` — values control marker colour.
        low/mid/high_color: Colour thresholds.
        threshold_high/mid: Forecast value thresholds.
        ms:             Marker size.
    """
    for code, info in stations.items():
        val = forecast.get(code, 0.0) if forecast else 0.0
        c   = (high_color if val > threshold_high
               else mid_color if val > threshold_mid
               else low_color)
        ax.plot(info["lon"], info["lat"],
                "o", color=c, ms=ms, mec=TXT_WHITE, mew=1.2)
        ax.text(info["lon"] + 0.05, info["lat"] + 0.05,
                code, color=TXT_WHITE, fontsize=7)


def save_figure(
    fig:   Any,
    path:  str | pathlib.Path,
    title: str = "",
    dpi:   int = 150,
) -> str:
    """
    Save a figure with standard dark background settings and show it.

    Replaces::

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR/"result.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.show()
        print(f"Saved: ...")

    Args:
        fig:   Matplotlib figure.
        path:  Output path.
        title: If set, add as suptitle before saving.
        dpi:   Resolution.

    Returns:
        ``str(path)``
    """
    if title:
        fig.suptitle(title, color=TXT_WHITE, fontsize=13,
                     fontweight="bold", y=1.01)
    plt.tight_layout()
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {path}")
    return str(path)
