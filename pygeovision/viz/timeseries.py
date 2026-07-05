"""
pygeovision.viz.timeseries
===========================
Multi-temporal data visualization — animation, trend charts, seasonal decomposition.

Usage::

    from pygeovision.viz import TimeSeriesViewer

    tsv = TimeSeriesViewer(
        paths=["ndvi_2020.tif", "ndvi_2021.tif", "ndvi_2022.tif"],
        dates=["2020-07-01", "2021-07-01", "2022-07-01"],
    )
    tsv.animate(output="ndvi_animation.gif")
    tsv.trend(roi=(-0.2, 5.6, -0.1, 5.7))   # trend over region of interest
    tsv.seasonal()
    tsv.anomaly()
    tsv.export("timeseries.html")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeovision.viz.timeseries")


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


class TimeSeriesViewer:
    """
    Multi-temporal raster viewer with animation, trend, and anomaly tools.

    Parameters
    ----------
    paths : list[str]
        Ordered list of GeoTIFF paths (one per time step).
    dates : list[str] | None
        ISO date strings corresponding to each path.
    band : int
        Band index (0-based) to visualize.  Default 0.
    colormap : str
        Matplotlib colormap for raster display.
    """

    def __init__(
        self,
        paths: List[str],
        dates: Optional[List[str]] = None,
        band: int = 0,
        colormap: str = "RdYlGn",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        self._paths   = [str(p) for p in paths]
        self._dates   = dates or [f"T{i}" for i in range(len(paths))]
        self._band    = band
        self._cmap    = colormap
        self._vmin    = vmin
        self._vmax    = vmax
        self._fig     = None
        self._data_cache: Optional[np.ndarray] = None   # (T, H, W)

    def _load_stack(self) -> np.ndarray:
        if self._data_cache is not None:
            return self._data_cache
        try:
            import rasterio
        except ImportError:
            raise ImportError("pip install rasterio")
        stack = []
        for path in self._paths:
            with rasterio.open(path) as src:
                stack.append(src.read(self._band + 1).astype("float32"))
        self._data_cache = np.stack(stack, axis=0)   # (T, H, W)
        return self._data_cache

    # ── Animation ─────────────────────────────────────────────────────────────

    def animate(self, output: str = "animation.gif", fps: int = 2,
                title: str = "Time Series Animation") -> str:
        """Save an animated GIF of the raster time-series."""
        plt  = _mpl()
        import matplotlib.animation as animation

        stack = self._load_stack()
        T     = stack.shape[0]
        vmin  = self._vmin or float(np.nanpercentile(stack, 2))
        vmax  = self._vmax or float(np.nanpercentile(stack, 98))

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(stack[0], cmap=self._cmap, vmin=vmin, vmax=vmax, animated=True)
        plt.colorbar(im, ax=ax, fraction=0.03)
        txt = ax.set_title(f"{title} — {self._dates[0]}", fontsize=12)
        ax.axis("off")

        def update(frame: int):
            im.set_data(stack[frame])
            txt.set_text(f"{title} — {self._dates[frame]}")
            return im, txt

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=int(1000 / fps), blit=False,
        )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        try:
            anim.save(output, writer="pillow", fps=fps)
        except Exception:
            try:
                anim.save(output, writer="ffmpeg", fps=fps)
            except Exception as exc:
                logger.warning("Animation save failed: %s — install Pillow", exc)
        plt.close(fig)
        logger.info("Animation saved to %s", output)
        return output

    # ── Trend over ROI ────────────────────────────────────────────────────────

    def trend(self, roi: Optional[Tuple[float, float, float, float]] = None,
               title: str = "Temporal Trend") -> "TimeSeriesViewer":
        """Plot mean pixel value over time for a region of interest."""
        plt = _mpl()
        stack = self._load_stack()

        if roi is not None:
            try:
                import rasterio
                from rasterio.windows import from_bounds
                with rasterio.open(self._paths[0]) as src:
                    win    = from_bounds(*roi, src.transform)
                    r_off  = max(int(win.row_off), 0)
                    c_off  = max(int(win.col_off), 0)
                    r_stop = min(int(win.row_off + win.height), stack.shape[1])
                    c_stop = min(int(win.col_off + win.width),  stack.shape[2])
                    stack  = stack[:, r_off:r_stop, c_off:c_stop]
            except Exception as exc:
                logger.warning("ROI crop failed: %s — using full scene", exc)

        means  = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)
        stds   = np.nanstd(stack.reshape(stack.shape[0], -1),  axis=1)

        # Linear trend
        x     = np.arange(len(means))
        slope, intercept = np.polyfit(x, means, 1)
        trend_line       = slope * x + intercept

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self._dates, means, "o-", color="#2196F3", linewidth=2, label="Mean value")
        ax.fill_between(self._dates, means - stds, means + stds,
                        alpha=0.15, color="#2196F3", label="±1 std")
        ax.plot(self._dates, trend_line, "--", color="#e74c3c", linewidth=1.5,
                label=f"Trend  (slope={slope:+.4f}/step)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean pixel value")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Seasonal decomposition ────────────────────────────────────────────────

    def seasonal(self, period: int = 12, title: str = "Seasonal Pattern") -> "TimeSeriesViewer":
        """Show seasonal pattern of the mean time-series."""
        plt = _mpl()
        stack  = self._load_stack()
        means  = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)
        T      = len(means)

        if T < 2 * period:
            logger.warning("Not enough time steps (%d) for seasonal analysis (need %d)", T, 2*period)
            period = max(2, T // 2)

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            import pandas as pd
            s = pd.Series(means)
            decomp = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")

            if getattr(self, "_fig", None) is not None:
                plt.close(self._fig)
                self._fig = None
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            for ax, comp, label in zip(
                axes,
                [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
                ["Observed", "Trend", "Seasonal", "Residual"],
            ):
                ax.plot(self._dates if len(self._dates) == len(comp) else range(len(comp)),
                        comp, linewidth=1.5)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(alpha=0.2)
        except ImportError:
            # Fallback: simple monthly aggregation
            if getattr(self, "_fig", None) is not None:
                plt.close(self._fig)
                self._fig = None
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(range(T), means, "o-", linewidth=1.5, color="#2196F3")
            ax.set_title("Time Series (statsmodels not installed for full decomposition)", fontsize=12)
            ax.set_xlabel("Step"); ax.set_ylabel("Mean value")
            ax.grid(alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def anomaly(self, n_sigma: float = 2.0,
                title: str = "Anomaly Detection") -> "TimeSeriesViewer":
        """Flag time steps where the mean value deviates more than n_sigma."""
        plt = _mpl()
        stack = self._load_stack()
        means = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)

        mu       = means.mean()
        sigma    = means.std()
        upper    = mu + n_sigma * sigma
        lower    = mu - n_sigma * sigma
        is_anom  = (means > upper) | (means < lower)

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=(12, 5))
        dates_x = self._dates
        ax.plot(dates_x, means, "o-", color="#2196F3", linewidth=2, label="Mean", zorder=3)
        ax.axhline(upper, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"+{n_sigma}σ")
        ax.axhline(lower, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"-{n_sigma}σ")
        ax.axhspan(lower, upper, alpha=0.06, color="green", label="Normal range")

        anom_dates = [d for d, a in zip(dates_x, is_anom) if a]
        anom_vals  = means[is_anom]
        if len(anom_dates):
            ax.scatter(anom_dates, anom_vals, color="#e74c3c", zorder=5, s=80,
                       label=f"Anomalies ({len(anom_dates)})")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Mean value")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._fig = fig
        plt.tight_layout()

        return self

    # ── Mosaic overview ───────────────────────────────────────────────────────

    def mosaic(self, max_cols: int = 4, title: str = "Time-Series Mosaic") -> "TimeSeriesViewer":
        """Display all time steps as a small multiples grid."""
        plt  = _mpl()
        stack = self._load_stack()
        T     = stack.shape[0]
        vmin  = self._vmin or float(np.nanpercentile(stack, 2))
        vmax  = self._vmax or float(np.nanpercentile(stack, 98))

        ncols = min(max_cols, T)
        nrows = int(np.ceil(T / ncols))
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.array(axes).ravel()

        for i, (ax, date) in enumerate(zip(axes, self._dates)):
            ax.imshow(stack[i], cmap=self._cmap, vmin=vmin, vmax=vmax)
            ax.set_title(date, fontsize=9)
            ax.axis("off")
        for ax in axes[T:]:
            ax.axis("off")

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._fig = fig
        plt.tight_layout()
        return self

    # ── Output ────────────────────────────────────────────────────────────────

    def show(self) -> "TimeSeriesViewer":
        plt = _mpl()
        if self._fig:
            plt.show()
        return self

    def export(self, path: str, dpi: int = 150) -> str:
        plt = _mpl()
        if self._fig is None:
            self.trend()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def __repr__(self) -> str:
        return f"TimeSeriesViewer(n={len(self._paths)}, band={self._band})"
