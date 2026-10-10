"""Map and chart images (PNG) for the reports.

Drawn with matplotlib's object-oriented API on the Agg canvas, so it works
headless and from threads. No web basemap is used (the pipelines run offline
as well as online); the map is the analysis layer on a neutral background with
a graticule, scale bar, north arrow, legend and area statistics.
"""
from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision._version import __version__
from pygeovision.reporting.fmt import fmt_value
from pygeovision.reporting.palettes import ACCENT, INK, MUTED, NEUTRAL_BG

logger = logging.getLogger(__name__)

DPI = 200
MAP_SIZE = (11.0, 7.4)
CHART_SIZE = (10.0, 4.6)
_MAX_LEGEND_ROWS = 12


@dataclass
class LayerRender:
    """Everything needed to draw one layer."""
    key: str
    title: str
    kind: str                                   # "classes" | "continuous"
    data: np.ndarray                            # classes: ints, invalid = -1 ; continuous: float, invalid = NaN
    extent: tuple[float, float, float, float]   # left, right, bottom, top (CRS units)
    geographic: bool
    metres_per_unit: float = 1.0
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    center: float | None = None
    label: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)   # class rows (value,name,color,area_ha,share)
    muted: tuple[int, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)         # continuous summary
    aoi: tuple[float, float, float, float] | None = None        # requested bbox in CRS units (for the outline)


def _figure(size: tuple[float, float]):
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=DPI, facecolor="white")
    return fig


def _save(fig: Any, path: Path) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=DPI, facecolor="white")
    return path


def _nice_length(target: float) -> float:
    """Round ``target`` down to 1, 2 or 5 x 10^n."""
    if target <= 0:
        return 1.0
    exp = math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * 10 ** exp <= target:
            return m * 10 ** exp
    return 10 ** exp


def _fmt_deg(v: float, axis: str) -> str:
    hemi = ("E" if v >= 0 else "W") if axis == "x" else ("N" if v >= 0 else "S")
    return f"{abs(v):.2f}°{hemi}"


def _draw_scale_bar(ax: Any, r: LayerRender) -> None:
    from matplotlib.patches import Rectangle
    l, rt, b, t = r.extent
    width_units = rt - l
    if r.geographic:
        lat = (b + t) / 2.0
        m_per_unit = 111_320.0 * math.cos(math.radians(lat))
    else:
        m_per_unit = r.metres_per_unit
    width_m = width_units * m_per_unit
    length_m = _nice_length(width_m * 0.22)
    length_u = length_m / m_per_unit
    x0 = l + 0.04 * width_units
    y0 = b + 0.045 * (t - b)
    h = 0.012 * (t - b)
    pad_x, pad_y = 0.012 * width_units, 0.012 * (t - b)
    ax.add_patch(Rectangle((x0 - pad_x, y0 - pad_y), length_u + 2 * pad_x + 0.08 * width_units * 0.4, h + 3.6 * pad_y,
                           facecolor="white", alpha=0.92, edgecolor="none", zorder=5))
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i * length_u / 2, y0), length_u / 2, h, facecolor=INK if i == 0 else "white",
                               edgecolor=INK, linewidth=0.7, zorder=6))
    label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(x0, y0 + h + 0.5 * pad_y, "0", fontsize=9.1, color=INK, va="bottom", ha="center", zorder=6)
    ax.text(x0 + length_u, y0 + h + 0.5 * pad_y, label, fontsize=9.1, color=INK, va="bottom", ha="center", zorder=6)


def _draw_north_arrow(ax: Any) -> None:
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.905, 0.80), 0.08, 0.19, transform=ax.transAxes, facecolor="white", alpha=0.85, edgecolor="none", zorder=6))
    ax.annotate("", xy=(0.945, 0.955), xytext=(0.945, 0.865), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=14), zorder=7)
    ax.text(0.945, 0.845, "N", transform=ax.transAxes, ha="center", va="top", fontsize=11.7, fontweight="bold", color=INK, zorder=7)


def _display_color(color: str, muted: bool) -> str:
    """The class's own colour, or a paler tint of it for background classes (kept opaque and in the same hue,
    so two background classes such as land and water stay distinguishable)."""
    if not muted:
        return color
    from matplotlib.colors import to_hex, to_rgb
    return to_hex(tuple(0.72 * c + 0.28 for c in to_rgb(color)))


def _class_rgba(r: LayerRender) -> np.ndarray:
    from matplotlib.colors import to_rgb
    h, w = r.data.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for row in r.rows:
        m = r.data == row["value"]
        if not m.any():
            continue
        rgba[m, :3] = to_rgb(_display_color(row["color"], row["value"] in r.muted))
        rgba[m, 3] = 1.0
    return rgba


def _draw_data(ax: Any, r: LayerRender) -> Any:
    """Draw the raster; returns the mappable for continuous layers."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, TwoSlopeNorm
    ext = (r.extent[0], r.extent[1], r.extent[2], r.extent[3])
    if r.kind == "classes":
        ax.imshow(_class_rgba(r), extent=ext, origin="upper", interpolation="nearest", zorder=2)
        return None
    cm = colormaps[r.cmap].copy()
    cm.set_bad(alpha=0.0)
    vals = r.data[np.isfinite(r.data)]
    vmin = r.vmin if r.vmin is not None else (float(np.percentile(vals, 2)) if vals.size else 0.0)
    vmax = r.vmax if r.vmax is not None else (float(np.percentile(vals, 98)) if vals.size else 1.0)
    if r.center is not None:
        span = max(abs(vmin - r.center), abs(vmax - r.center)) or 1.0
        norm: Any = TwoSlopeNorm(vcenter=r.center, vmin=r.center - span, vmax=r.center + span)
    else:
        if vmax <= vmin:
            vmax = vmin + 1e-9
        norm = Normalize(vmin=vmin, vmax=vmax)
    return ax.imshow(np.ma.masked_invalid(r.data), extent=ext, origin="upper", cmap=cm, norm=norm,
                     interpolation="nearest", zorder=2)


def _legend_classes(ax: Any, r: LayerRender, y: float) -> float:
    """Class legend with area and share, in sidebar axes coordinates. Returns the y below it."""
    from matplotlib.patches import Rectangle
    ax.text(0.0, y, "Legend", fontsize=13.0, fontweight="bold", color=ACCENT, va="top")
    y -= 0.058
    rows = sorted(r.rows, key=lambda d: d["value"])
    shown = rows[:_MAX_LEGEND_ROWS] if len(rows) <= _MAX_LEGEND_ROWS else sorted(rows, key=lambda d: -(d["area_ha"] or 0))[:_MAX_LEGEND_ROWS]
    for row in shown:
        color = _display_color(row["color"], row["value"] in r.muted)
        ax.add_patch(Rectangle((0.0, y - 0.04), 0.07, 0.04, facecolor=color, edgecolor=INK, linewidth=0.5,
                               transform=ax.transAxes, clip_on=False))
        name = textwrap.shorten(row["name"], width=24, placeholder="…")
        share = fmt_value(row["share"], "frac") if row.get("share") is not None else "n/a"
        ax.text(0.10, y + 0.004, name, fontsize=10.4, color=INK, va="top", fontweight="bold")
        ax.text(0.10, y - 0.027, f"{fmt_value(row['area_ha'], 'ha')}  ·  {share}", fontsize=9.4, color=MUTED, va="top")
        y -= 0.082
    if len(rows) > len(shown):
        ax.text(0.0, y, f"+ {len(rows) - len(shown)} smaller classes (see report)", fontsize=9.1, color=MUTED, va="top")
        y -= 0.05
    return y


def _legend_continuous(fig: Any, ax: Any, mappable: Any, r: LayerRender, y: float) -> float:
    ax.text(0.0, y, r.label or "Value", fontsize=13.0, fontweight="bold", color=ACCENT, va="top")
    y -= 0.06
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 + pos.height * (y - 0.045), pos.width * 0.92, pos.height * 0.035])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    cb.locator = MaxNLocator(4)
    fmt_ = ScalarFormatter(useOffset=False)
    fmt_.set_scientific(False)
    cb.formatter = fmt_
    cb.update_ticks()
    cb.ax.tick_params(labelsize=9, colors=INK)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(0.5)
    y -= 0.12
    s = r.stats
    if s.get("n_valid"):
        lines = [("Mean", s.get("mean")), ("Median", s.get("median")), ("Min / Max", None)]
        for lab, val in lines:
            txt = f"{s['min']:.3g} / {s['max']:.3g}" if lab == "Min / Max" else f"{val:.3g}"
            ax.text(0.0, y, lab, fontsize=10.4, color=MUTED, va="top")
            ax.text(0.98, y, txt, fontsize=10.4, color=INK, va="top", ha="right", fontweight="bold")
            y -= 0.052
    return y - 0.02


def render_map(
    r: LayerRender, out_path: Path, *, title: str, subtitle: str, footer: str,
    kpis: Sequence[tuple[str, str]] = (), warning: str = "",
) -> Path:
    """One map: the layer with graticule, scale bar, north arrow, legend and key figures."""
    fig = _figure(MAP_SIZE)
    fig.text(0.04, 0.955, title, fontsize=22.1 if len(title) <= 44 else max(14.0, 22.1 * 44 / len(title)), fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.905, subtitle, fontsize=13.0 if len(subtitle) <= 90 else max(9.5, 13.0 * 90 / len(subtitle)), color=MUTED, va="top")
    l, rt, b, t_ = r.extent
    if r.geographic:
        disp_aspect = (rt - l) * max(0.05, math.cos(math.radians((b + t_) / 2.0))) / max(t_ - b, 1e-12)
    else:
        disp_aspect = (rt - l) / max(t_ - b, 1e-12)
    max_w_in, max_h_in = 6.5, 5.45
    w_in = min(max_w_in, max_h_in * disp_aspect)
    h_in = w_in / disp_aspect
    fw, fh = MAP_SIZE
    ax_x0, ax_y0 = 0.075, 0.115 + (max_h_in - h_in) / fh / 2
    ax = fig.add_axes([ax_x0, ax_y0, w_in / fw, h_in / fh])
    ax.set_facecolor(NEUTRAL_BG)
    mappable = _draw_data(ax, r)
    ax.set_xlim(l, rt)
    ax.set_ylim(b, t_)
    if r.geographic:
        ax.set_aspect(1.0 / max(0.05, math.cos(math.radians((b + t_) / 2.0))))
    else:
        ax.set_aspect("equal")
    if r.aoi is not None:
        al, ab, ar, at = r.aoi
        ax.plot([al, ar, ar, al, al], [ab, ab, at, at, ab], color=INK, lw=0.9, ls=(0, (4, 3)), zorder=4)
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if r.geographic:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_deg(v, "x")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_deg(v, "y")))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f} km"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f} km"))
    ax.tick_params(labelsize=9.5, colors=INK, length=3)
    ax.grid(color="white", lw=0.6, alpha=0.9, zorder=3)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK)
        sp.set_linewidth(0.8)
    _draw_scale_bar(ax, r)
    _draw_north_arrow(ax)

    side_x = min(0.72, ax_x0 + w_in / fw + 0.06)
    side = fig.add_axes([side_x, 0.115, min(0.28, 0.985 - side_x), 0.73])
    side.axis("off")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    y = 0.995
    y = _legend_classes(side, r, y) if r.kind == "classes" else _legend_continuous(fig, side, mappable, r, y)
    if kpis:
        side.text(0.0, y - 0.01, "Key figures", fontsize=13.0, fontweight="bold", color=ACCENT, va="top")
        y -= 0.08
        for lab, val in kpis[:5]:
            side.text(0.0, y, textwrap.shorten(lab, width=30, placeholder="…"), fontsize=9.9, color=MUTED, va="top")
            side.text(0.0, y - 0.036, val, fontsize=13.7, color=INK, va="top", fontweight="bold")
            y -= 0.105
    if warning:
        fig.text(0.04, 0.062, "⚠ " + warning, fontsize=10.4, color="#b45309", va="top")
    fig.text(0.04, 0.032, footer, fontsize=9.1, color=MUTED, va="top")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style_axes(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="x" if ax.get_xlabel() == "" and False else "y", color="#e4e7eb", lw=0.7)
    ax.set_axisbelow(True)


def chart_class_areas(rows: Sequence[dict[str, Any]], out_path: Path, *, title: str, muted: Sequence[int] = ()) -> Path:
    """Horizontal bars: area per class. Background classes are left out when there are feature classes."""
    show = [r for r in rows if r["value"] not in muted] or list(rows)
    if len(show) > _MAX_LEGEND_ROWS:               # keep the largest classes, then show them in class order
        show = sorted(show, key=lambda d: -(d["area_ha"] or 0))[:_MAX_LEGEND_ROWS]
    show = sorted(show, key=lambda d: -d["value"])   # barh draws bottom-up: class 0 ends up on top
    fig = _figure(CHART_SIZE)
    ax = fig.add_axes([0.26, 0.14, 0.68, 0.70])
    ys = np.arange(len(show))
    vals = [d["area_ha"] or 0.0 for d in show]
    ax.barh(ys, vals, color=[d["color"] for d in show], edgecolor=INK, linewidth=0.4, height=0.62)
    ax.set_yticks(ys)
    ax.set_yticklabels([textwrap.shorten(d["name"], 30, placeholder="…") for d in show], fontsize=11.1)
    top = max(vals) if vals and max(vals) > 0 else 1.0
    for y, d, v in zip(ys, show, vals):
        share = fmt_value(d["share"], "frac") if d.get("share") is not None else ""
        ax.text(v + top * 0.01, y, f"{v:,.1f} ha  {share}", va="center", fontsize=10.4, color=INK)
    ax.set_xlim(0, top * 1.32)
    ax.set_xlabel("Area (hectares)", fontsize=11.1, color=INK)
    _style_axes(ax)
    ax.grid(axis="x", color="#e4e7eb", lw=0.7)
    ax.grid(axis="y", visible=False)
    fig.text(0.04, 0.95, title, fontsize=16.9, fontweight="bold", color=INK, va="top")
    return _save(fig, out_path)


def chart_histogram(values: np.ndarray, summary: dict[str, Any], out_path: Path, *, title: str, xlabel: str, color: str = ACCENT) -> Path:
    v = values[np.isfinite(values)]
    fig = _figure(CHART_SIZE)
    ax = fig.add_axes([0.09, 0.15, 0.86, 0.69])
    if v.size:
        lo, hi = np.percentile(v, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1e-9
        ax.hist(v, bins=60, range=(lo, hi), color=color, alpha=0.88, edgecolor="white", linewidth=0.3)
        if summary.get("median") is not None:
            ax.axvline(summary["median"], color=INK, lw=1.2)
            ax.text(summary["median"], ax.get_ylim()[1] * 0.97, f" median {summary['median']:.3g}", fontsize=10.4, color=INK, va="top")
        if summary.get("p5") is not None:
            ax.axvspan(summary["p5"], summary["p95"], color=color, alpha=0.10)
            ax.text(summary["p95"], ax.get_ylim()[1] * 0.85, f" 5-95 %: {summary['p5']:.3g} to {summary['p95']:.3g}", fontsize=10.4, color=MUTED, va="top")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.set_xlabel(xlabel, fontsize=11.1, color=INK)
    ax.set_ylabel("Pixels", fontsize=11.1, color=INK)
    _style_axes(ax)
    fig.text(0.04, 0.95, title, fontsize=16.9, fontweight="bold", color=INK, va="top")
    return _save(fig, out_path)


def chart_timeseries(rows: Sequence[dict[str, Any]], out_path: Path, *, title: str) -> Path:
    fig = _figure(CHART_SIZE)
    ax = fig.add_axes([0.09, 0.17, 0.86, 0.67])
    x = np.arange(len(rows))
    series = (("mean_ndvi", "NDVI", "#2e7d32"), ("mean_evi", "EVI", "#1565c0"), ("mean_ndwi", "NDWI", "#8d6e63"))
    for key, label, color in series:
        ys = np.array([np.nan if r.get(key) is None else r[key] for r in rows], dtype=float)
        ax.plot(x, ys, marker="o", lw=2, color=color, label=label)
        for xi, yi in zip(x, ys):
            if np.isfinite(yi):
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.8, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["date"]) for r in rows], fontsize=11.1)
    ax.set_ylabel("Scene mean (valid pixels)", fontsize=11.1, color=INK)
    ax.legend(frameon=False, fontsize=11.1, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.09))
    _style_axes(ax)
    fig.text(0.04, 0.95, title, fontsize=16.9, fontweight="bold", color=INK, va="top")
    return _save(fig, out_path)


def chart_before_after(before: float | None, after: float | None, labels: tuple[str, str], out_path: Path, *, title: str, unit: str = "ha") -> Path:
    fig = _figure(CHART_SIZE)
    ax = fig.add_axes([0.14, 0.14, 0.62, 0.70])
    vals = [before or 0.0, after or 0.0]
    bars = ax.bar(labels, vals, color=["#9ecae1", "#3182bd"], edgecolor=INK, linewidth=0.5, width=0.5)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,.1f} {unit}", ha="center", va="bottom", fontsize=11.7, color=INK)
    if before and after is not None:
        delta = after - before
        ax.text(1.05, 0.5, f"Change\n{delta:+,.1f} {unit}\n({delta / before * 100:+.1f} %)", transform=ax.transAxes, fontsize=14.3,
                fontweight="bold", color=INK, va="center")
    ax.set_ylabel(f"Area ({unit})", fontsize=11.1, color=INK)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    _style_axes(ax)
    fig.text(0.04, 0.95, title, fontsize=16.9, fontweight="bold", color=INK, va="top")
    return _save(fig, out_path)


def footer_text(crs_text: str, extra: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"CRS: {crs_text}  ·  Areas measured from the output raster  ·  No basemap  ·  "
            f"PyGeoVision {__version__}  ·  {stamp}" + (f"  ·  {extra}" if extra else ""))


def render_panel(renders: Sequence[LayerRender], out_path: Path, *, title: str, subtitle: str, footer: str, ncols: int = 3) -> Path:
    """Small multiples: the same variable on several dates, one shared colour bar."""
    n = len(renders)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig = _figure((11.0, 3.0 + 3.6 * nrows))
    fig.text(0.04, 0.975, title, fontsize=22.1, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.975 - 0.075 / nrows * 1.2, subtitle, fontsize=13.0, color=MUTED, va="top")
    top, bottom = 0.80, 0.22
    cell_h = (top - bottom) / nrows
    last = None
    for i, r in enumerate(renders):
        row, col = divmod(i, ncols)
        x0 = 0.05 + col * (0.92 / ncols)
        ax = fig.add_axes([x0, top - (row + 1) * cell_h + 0.02, 0.92 / ncols - 0.035, cell_h - 0.06])
        ax.set_facecolor(NEUTRAL_BG)
        last = _draw_data(ax, r) or last
        l, rt, b, t = r.extent
        ax.set_xlim(l, rt)
        ax.set_ylim(b, t)
        ax.set_aspect(1.0 / max(0.05, math.cos(math.radians((b + t) / 2.0))) if r.geographic else "equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK)
            sp.set_linewidth(0.6)
        ax.set_title(r.title, fontsize=12.3, color=INK, fontweight="bold", pad=4)
    if last is not None and renders[0].kind == "continuous":
        cax = fig.add_axes([0.3, 0.135, 0.4, 0.02])
        cb = fig.colorbar(last, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=9.5, colors=INK)
        cb.set_label(renders[0].label, fontsize=11.1, color=INK)
    fig.text(0.04, 0.03, footer, fontsize=9.1, color=MUTED, va="top")
    return _save(fig, out_path)