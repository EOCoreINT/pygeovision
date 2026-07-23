"""
pygeovision.viz.change
=======================
Change detection visualization tools.

Usage::

    from pygeovision.viz import ChangeViewer

    cv = ChangeViewer("before.tif", "after.tif")
    cv.split()                             # side-by-side panels
    cv.difference()                        # amplitude change map
    cv.overlay("change_mask.tif")          # overlay mask on imagery
    cv.statistics("change_mask.tif")       # area and class breakdown
    cv.export("comparison.html")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pygeovision.viz.change")


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


class ChangeViewer:
    """
    Visualize before/after changes between two rasters.

    Parameters
    ----------
    before : str
        Path to the 'before' raster.
    after : str
        Path to the 'after' raster.
    band : int
        Band index (0-based) to compare.  Default 0.
    """

    def __init__(self, before: str, after: str, band: int = 0) -> None:
        self._before = before
        self._after  = after
        self._band   = band
        self._fig    = None

    def _read_band(self, path: str) -> np.ndarray:
        try:
            import rasterio
        except ImportError:
            raise ImportError("pip install rasterio")
        with rasterio.open(path) as src:
            return src.read(self._band + 1).astype("float32")

    def _pct_stretch(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    # ── Side-by-side split ─────────────────────────────────────────────────────

    def split(self, before_label: str = "Before", after_label: str = "After",
               colormap: str = "gray", title: str = "Change Comparison") -> ChangeViewer:
        """Display before and after rasters side by side."""
        plt = _mpl()
        b_data = self._pct_stretch(self._read_band(self._before))
        a_data = self._pct_stretch(self._read_band(self._after))

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(16, 7))
        ax_b.imshow(b_data, cmap=colormap, vmin=0, vmax=1)
        ax_b.set_title(f"◀  {before_label}\n{Path(self._before).name}", fontsize=12)
        ax_b.axis("off")

        ax_a.imshow(a_data, cmap=colormap, vmin=0, vmax=1)
        ax_a.set_title(f"{after_label}  ▶\n{Path(self._after).name}", fontsize=12)
        ax_a.axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Difference map ─────────────────────────────────────────────────────────

    def difference(self, title: str = "Amplitude Difference") -> ChangeViewer:
        """Show normalised amplitude change map."""
        plt = _mpl()
        b_data  = self._read_band(self._before)
        a_data  = self._read_band(self._after)
        diff    = a_data - b_data
        vabs    = max(abs(float(np.nanpercentile(diff, 2))),
                      abs(float(np.nanpercentile(diff, 98))))

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, data, label in zip(axes,
            [self._pct_stretch(b_data), self._pct_stretch(a_data), None],
            ["Before", "After", title]):
            if data is not None:
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            else:
                im = ax.imshow(diff, cmap="RdBu", vmin=-vabs, vmax=vabs)
                plt.colorbar(im, ax=ax, fraction=0.04, label="Δ value")
            ax.set_title(label, fontsize=12)
            ax.axis("off")

        fig.suptitle("Change Analysis", fontsize=14, fontweight="bold")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Overlay ────────────────────────────────────────────────────────────────

    def overlay(self, mask_path: str, change_color: str = "#ff4444",
                alpha: float = 0.5, title: str = "Change Overlay") -> ChangeViewer:
        """Overlay a binary/classified change mask on the 'after' raster."""
        plt = _mpl()
        import matplotlib.colors as mcolors

        a_data = self._pct_stretch(self._read_band(self._after))
        try:
            import rasterio
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype("float32")
        except Exception as exc:
            logger.warning("Could not read mask %s: %s", mask_path, exc)
            mask = np.zeros_like(a_data)

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(a_data, cmap="gray", vmin=0, vmax=1)

        # Overlay change pixels with a semi-transparent colour
        rgba_mask = np.zeros((*mask.shape, 4), dtype=float)
        changed   = mask > 0
        rgba_mask[changed] = [*mcolors.to_rgb(change_color), alpha]
        ax.imshow(rgba_mask)

        n_changed = int(changed.sum())
        n_total   = int(mask.size)
        ax.set_title(f"{title} — {n_changed:,} changed pixels ({100*n_changed/n_total:.1f}%)",
                     fontsize=12)
        ax.axis("off")

        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color="gray", label="Unchanged"),
            mpatches.Patch(color=change_color, label="Changed"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=10)
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Statistics ─────────────────────────────────────────────────────────────

    def statistics(self, mask_path: str | None = None,
                   pixel_area_m2: float = 100.0) -> dict[str, Any]:
        """Compute and display change statistics."""
        plt = _mpl()

        b_data = self._read_band(self._before)
        a_data = self._read_band(self._after)
        diff   = a_data - b_data

        stats: dict[str, Any] = {
            "total_pixels":   int(diff.size),
            "mean_change":    round(float(np.nanmean(np.abs(diff))), 4),
            "max_change":     round(float(np.nanmax(np.abs(diff))), 4),
            "pct_increased":  round(float((diff > 0.05).mean() * 100), 2),
            "pct_decreased":  round(float((diff < -0.05).mean() * 100), 2),
            "pct_stable":     round(float((np.abs(diff) <= 0.05).mean() * 100), 2),
        }

        if mask_path:
            try:
                import rasterio
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    pix_m2 = abs(src.transform.a * src.transform.e)
                changed_pix = int((mask > 0).sum())
                stats["changed_pixels"] = changed_pix
                stats["changed_area_m2"]  = changed_pix * pix_m2
                stats["changed_area_ha"]  = round(changed_pix * pix_m2 / 1e4, 2)
                stats["changed_area_km2"] = round(changed_pix * pix_m2 / 1e6, 4)
            except Exception as exc:
                logger.warning("Mask stats error: %s", exc)

        # Bar chart — close previous figure before creating new one
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=(8, 5))
        labels  = ["Increased\n(>0.05)", "Stable\n(±0.05)", "Decreased\n(<-0.05)"]
        values  = [stats["pct_increased"], stats["pct_stable"], stats["pct_decreased"]]
        colors  = ["#e74c3c", "#95a5a6", "#3498db"]
        bars    = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("% of pixels", fontsize=12)
        ax.set_title("Change Statistics", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._fig = fig
        plt.tight_layout()
        return stats

    # ── Export / show ──────────────────────────────────────────────────────────

    def show(self) -> ChangeViewer:
        plt = _mpl()
        if self._fig:
            plt.show()
        return self

    def export(self, path: str, dpi: int = 150) -> str:
        _mpl()
        if self._fig is None:
            self.split()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".html"):
            from pygeovision.viz.map import SplitMap
            return SplitMap(self._before, self._after).export(path)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def __repr__(self) -> str:
        return (f"ChangeViewer(before={Path(self._before).name!r}, "
                f"after={Path(self._after).name!r})")
