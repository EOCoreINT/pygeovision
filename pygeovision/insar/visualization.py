"""
pygeovision.insar.visualization
================================
Visualization tools for InSAR products.

Usage::

    from pygeovision.insar.visualization import InSARViz

    viz = InSARViz()
    viz.interferogram("interferogram.tif")
    viz.coherence("coherence.tif")
    viz.displacement("displacement.tif")
    viz.time_series(["disp_t0.tif", ..., "disp_tn.tif"], dates=[...])
    viz.export("insar_dashboard.html")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pygeovision.insar.visualization")


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


class InSARViz:
    """
    Visualization suite for InSAR products: interferograms, coherence,
    displacement maps, deformation rate, and time-series.
    """

    def __init__(self, figsize: tuple[float, float] = (10, 8)) -> None:
        self.figsize = figsize
        self._fig    = None

    def _read(self, path: str) -> tuple[np.ndarray, Any]:
        try:
            import rasterio
        except ImportError:
            raise ImportError("pip install rasterio")
        with rasterio.open(path) as src:
            data    = src.read(1).astype("float32")
            profile = src.profile.copy()
            abs(src.transform.a)
        data = np.where(data == -9999.0, np.nan, data)
        return data, profile

    # ── Interferogram ─────────────────────────────────────────────────────────

    def interferogram(self, path: str, title: str = "Interferogram (Wrapped Phase)",
                       output_path: str | None = None) -> InSARViz:
        """
        Display wrapped phase interferogram with standard rainbow colour cycle.
        """
        plt = _mpl()
        data, _ = self._read(path)
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(data, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                       interpolation="bilinear")
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, label="Phase (rad)")
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(["−π", "−π/2", "0", "+π/2", "+π"])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        _add_caption(ax, "One colour cycle = 2π rad = λ/2 displacement"
                         f" (Sentinel-1 C-band: {0.05546576/2*100:.1f} cm)")
        self._fig = fig
        plt.tight_layout()
        if output_path:
            self.export(output_path)
        return self

    # ── Coherence ─────────────────────────────────────────────────────────────

    def coherence(self, path: str, title: str = "Coherence Map",
                   output_path: str | None = None) -> InSARViz:
        plt = _mpl()
        import matplotlib.colors as mcolors

        data, _ = self._read(path)
        # Custom colormap: red (decorrelated) → yellow → green (coherent)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "coherence", ["#d32f2f","#f57c00","#fbc02d","#388e3c"], N=256
        )
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.035, label="Coherence (0–1)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        mean_coh = float(np.nanmean(data))
        pct_high = float((data > 0.7).mean() * 100)
        _add_caption(ax, f"Mean coherence: {mean_coh:.3f} | "
                         f"High coherence (>0.7): {pct_high:.1f}%")
        self._fig = fig
        plt.tight_layout()
        if output_path:
            self.export(output_path)
        return self

    # ── Displacement ──────────────────────────────────────────────────────────

    def displacement(self, path: str, title: str = "LOS Displacement (m)",
                      vmax_cm: float = 20.0,
                      output_path: str | None = None) -> InSARViz:
        plt = _mpl()
        data, _ = self._read(path)
        vmax = vmax_cm / 100.0

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: displacement map
        im = axes[0].imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax,
                             interpolation="bilinear")
        plt.colorbar(im, ax=axes[0], fraction=0.04, label="Displacement (m)")
        axes[0].set_title(title, fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Right: displacement histogram
        valid = data[np.isfinite(data)].ravel()
        axes[1].hist(valid * 100, bins=128, color="#2196F3", edgecolor="none", alpha=0.85)
        axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
        axes[1].axvline(float(np.nanmean(valid)) * 100, color="#e74c3c",
                         linewidth=1.5, linestyle="-", label=f"Mean={np.nanmean(valid)*100:.1f} cm")
        axes[1].set_xlabel("Displacement (cm)", fontsize=11)
        axes[1].set_ylabel("Pixel count", fontsize=11)
        axes[1].set_title("Displacement Distribution", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        self._fig = fig
        plt.tight_layout()
        if output_path:
            self.export(output_path)
        return self

    # ── Deformation rate ──────────────────────────────────────────────────────

    def deformation_rate(self, path: str, title: str = "Deformation Rate (mm/yr)",
                          output_path: str | None = None) -> InSARViz:
        plt = _mpl()
        data, _ = self._read(path)
        vabs = max(abs(float(np.nanpercentile(data, 2))),
                   abs(float(np.nanpercentile(data, 98))))

        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(data, cmap="RdBu", vmin=-vabs, vmax=vabs, interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.035, label="mm/year")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        sub_pct = float((data < -5).mean() * 100) if np.isfinite(data).any() else 0
        upl_pct = float((data >  5).mean() * 100) if np.isfinite(data).any() else 0
        _add_caption(ax, f"Subsidence (< −5 mm/yr): {sub_pct:.1f}% | "
                         f"Uplift (> +5 mm/yr): {upl_pct:.1f}%")
        self._fig = fig
        plt.tight_layout()
        if output_path:
            self.export(output_path)
        return self

    # ── Time-series displacement ───────────────────────────────────────────────

    def time_series(
        self,
        displacement_paths: list[str],
        dates: list[str],
        pixel_coords: list[tuple[int, int]] | None = None,
        title: str = "Displacement Time-Series",
        output_path: str | None = None,
    ) -> InSARViz:
        """
        Plot displacement vs time for selected pixels or spatial mean.

        Parameters
        ----------
        pixel_coords : list[(row, col)] | None
            Specific pixel coordinates to sample.  None = mean over valid pixels.
        """
        plt = _mpl()

        stack = []
        for path in displacement_paths:
            data, _ = self._read(path)
            stack.append(data)
        arr = np.stack(stack, axis=0)  # (T, H, W)
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, ax = plt.subplots(figsize=(12, 5))

        if pixel_coords:
            for row, col in pixel_coords:
                values = arr[:, row, col] * 100   # → cm
                ax.plot(dates, values, "o-", linewidth=1.8, markersize=5,
                        label=f"Pixel ({row},{col})")
            ax.legend(fontsize=9, loc="best")
        else:
            means = np.nanmean(arr.reshape(arr.shape[0], -1), axis=1) * 100
            stds  = np.nanstd(arr.reshape(arr.shape[0], -1),  axis=1) * 100
            ax.plot(dates, means, "o-", color="#2196F3", linewidth=2, label="Mean LOS")
            ax.fill_between(dates, means - stds, means + stds, alpha=0.15,
                            color="#2196F3", label="±1σ")
            ax.legend(fontsize=10)

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("LOS displacement (cm)")
        ax.grid(alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._fig = fig
        plt.tight_layout()
        if output_path:
            self.export(output_path)
        return self

    # ── Dashboard: 2×2 summary ────────────────────────────────────────────────

    def dashboard(
        self,
        interferogram_path: str | None = None,
        coherence_path:     str | None = None,
        displacement_path:  str | None = None,
        rate_path:          str | None = None,
        title: str = "InSAR Analysis Dashboard",
        output_path: str | None = None,
    ) -> InSARViz:
        """Display a 2×2 grid of the main InSAR products."""
        plt = _mpl()
        if getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        fig.suptitle(title, fontsize=15, fontweight="bold")

        panels = [
            (interferogram_path, "Interferogram", "hsv",    -np.pi, np.pi,   "Phase (rad)"),
            (coherence_path,     "Coherence",     "RdYlGn",  0,     1,       "Coherence"),
            (displacement_path,  "Displacement",  "RdBu",   -0.2,   0.2,     "Metres"),
            (rate_path,          "Deformation Rate","RdBu", -30,    30,       "mm/year"),
        ]

        for ax, (path, label, cmap, vmin, vmax, cb_label) in zip(axes.ravel(), panels):
            if path:
                try:
                    data, _ = self._read(path)
                    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.colorbar(im, ax=ax, fraction=0.04, label=cb_label)
                    ax.set_title(label, fontsize=12, fontweight="bold")
                except Exception:
                    ax.text(0.5, 0.5, f"{label}\n(not available)", ha="center",
                            va="center", transform=ax.transAxes, fontsize=11, color="#666")
            else:
                ax.text(0.5, 0.5, f"{label}\n(not provided)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=11, color="#999")
                ax.set_facecolor("#f5f5f5")
            ax.axis("off")

        self._fig = fig
        plt.tight_layout()
        if output_path:
            self.export(output_path)
        return self

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, path: str, dpi: int = 150) -> str:
        _mpl()
        if self._fig is None:
            raise RuntimeError("Nothing to export — call interferogram(), coherence(), etc. first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("InSAR viz exported to %s", path)
        return path

    def show(self) -> InSARViz:
        plt = _mpl()
        if self._fig:
            plt.show()
        return self

    def __repr__(self) -> str:
        return "InSARViz()"


def _add_caption(ax, text: str) -> None:
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8.5, color="#555",
            va="bottom", style="italic",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.6})
