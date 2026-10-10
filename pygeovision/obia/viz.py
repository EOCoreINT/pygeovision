"""Small matplotlib helpers for looking at objects.  Each returns a Figure.

matplotlib is imported lazily so the rest of the package works without it.
"""
from __future__ import annotations

import numpy as np

from .io import rescale_bands

_PALETTE = ["#1f77b4", "#2ca02c", "#98df8a", "#d2b48c", "#d62728", "#7f7f7f",
            "#9467bd", "#ff7f0e", "#17becf", "#bcbd22"]


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt


def show_overlay(seg, bands=(0, 1, 2), title: str | None = None, ax=None):
    """Image with object boundaries.  ``bands`` picks the display bands (indices)."""
    plt = _plt()
    fig, ax = (plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax))
    ax.imshow(seg.overlay(bands))
    ax.set_title(title or f"{seg.method}: {seg.n_objects} objects")
    ax.axis("off")
    return fig


def show_classes(result, labels, raster=None, bands=(0, 1, 2), colors: dict | None = None,
                 title: str | None = None, ax=None):
    """Classified objects (colour per class), optionally beside the image."""
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    names = result.class_names
    cols = [(colors or {}).get(n, _PALETTE[i % len(_PALETTE)]) for i, n in enumerate(names)]
    cmap = ListedColormap(["#ffffff"] + cols)
    code = result.to_raster(labels)
    if ax is None:
        n = 2 if raster is not None else 1
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        axes = np.atleast_1d(axes)
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)
    if raster is not None and len(axes) > 1:
        axes[0].imshow(np.moveaxis(rescale_bands(raster.image[list(bands)]).astype(np.uint8), 0, -1))
        axes[0].set_title("image"); axes[0].axis("off")
    a = axes[-1]
    a.imshow(code, cmap=cmap, vmin=0, vmax=len(names), interpolation="nearest")
    a.set_title(title or f"{result.method}: {len(result.classes)} objects"); a.axis("off")
    a.legend(handles=[Patch(color=c, label=n) for c, n in zip(cols, names)], loc="lower right",
             fontsize=8, framealpha=0.9)
    return fig


def plot_scale(df, ax=None):
    """Plot a table from ``scale_diagnostics`` or ``scale_vs_reference``."""
    plt = _plt()
    fig, ax = (plt.subplots(figsize=(7, 4)) if ax is None else (ax.figure, ax))
    x = df.index.to_numpy()
    if "mean_D" in df.columns:
        ax.plot(x, df["mean_over_seg"], "o-", label="over-segmentation")
        ax.plot(x, df["mean_under_seg"], "s-", label="under-segmentation")
        ax.plot(x, df["mean_D"], "k^-", lw=2, label="D (lower = better)")
        best = df["mean_D"].idxmin()
        ax.axvline(best, color="grey", ls="--"); ax.set_title(f"best scale by D: {best:g}")
        ax.set_ylabel("score")
    else:
        ax.plot(x, df["roc_lv"], "o-", label="ROC-LV (%)")
        cand = df.index[df["candidate"]]
        ax.plot(cand, df.loc[cand, "roc_lv"], "r*", ms=14, label="candidate scale")
        ax.set_ylabel("rate of change of LV (%)"); ax.set_title("scale diagnostics (heuristic)")
    ax.set_xlabel("scale parameter"); ax.legend(); ax.grid(alpha=0.3)
    return fig
