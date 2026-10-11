"""Spectral-index time series over a stack of images: statistics per date, anomalies, trend, plot.

Scope: each image becomes **one number per date** (the mean of an index over the pixels that are valid),
plus a spread (std, P10, P90). It is a summary of an area through time, not a per-pixel analysis; for
per-pixel anomaly or trend maps use the ``vegetation_anomaly`` and ``water_extent_timeseries`` pipelines.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%Y%m%d", "%Y%m", "%Y")


def _parse_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


class GeoTimeSeries:
    """Summarise a spectral index through time for a list of georeferenced images.

    What it does: reads the bands an index needs from every image, computes the index per pixel, ignores
    invalid pixels, and records mean/std/P10/P90 and the fraction of valid pixels for each date. On top of
    that series it can flag anomalous dates (:meth:`detect_anomalies`), fit a linear trend
    (:meth:`compute_trend`), sample one location (:meth:`extract_point`) and draw it (:meth:`plot`).
    It does not classify crop types or map urban change.

    Band positions are 1-based positions **in your files**. The presets describe the usual band order of a
    stack (``sentinel2``: blue, green, red, nir = bands 1-4, swir = band 10; ``landsat8``: bands 1-5 =
    blue, green, red, nir, swir1). If your files are ordered differently pass ``band_map``.

    Example::

        ts = GeoTimeSeries(sensor="sentinel2")
        series = ts.compute_index_series(sorted(glob("s2_*.tif")), "ndvi",
                                         date_strings=["2024-01", "2024-02", ...])
        ts.detect_anomalies(series, method="mad")
        ts.compute_trend(series)
    """

    #: index name -> (bands it needs, function of those bands in reflectance 0-1)
    INDICES = {
        "ndvi": (("nir", "red"), lambda n, r: (n - r) / (n + r + 1e-8)),
        "ndwi": (("green", "nir"), lambda g, n: (g - n) / (g + n + 1e-8)),
        "ndbi": (("swir", "nir"), lambda s, n: (s - n) / (s + n + 1e-8)),
        "evi": (("nir", "red", "blue"), lambda n, r, b: 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0)),
        "savi": (("nir", "red"), lambda n, r: 1.5 * (n - r) / (n + r + 0.5)),
    }

    BAND_PRESETS = {
        "sentinel2": {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 10},
        "landsat8": {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5},
        "planet": {"blue": 1, "green": 2, "red": 3, "nir": 4},
        "naip": {"red": 1, "green": 2, "blue": 3, "nir": 4},
    }

    def __init__(self, sensor: str = "sentinel2", band_map: dict[str, int] | None = None) -> None:
        """``band_map`` ({"red": 3, "nir": 4, ...}, 1-based) overrides the preset for ``sensor``."""
        if band_map is None and sensor not in self.BAND_PRESETS:
            raise ValueError(f"unknown sensor {sensor!r}; choose from {sorted(self.BAND_PRESETS)} "
                             "or pass band_map={'red': 3, 'nir': 4, ...}")
        self.sensor = sensor
        self.band_map = dict(band_map) if band_map else dict(self.BAND_PRESETS[sensor])

    # ── index series ────────────────────────────────────────────────────────

    def compute_index_series(
        self,
        image_paths: list[str],
        index: str = "ndvi",
        date_strings: list[str] | None = None,
        cloud_mask: bool | None = None,
        mask_paths: list[str] | None = None,
        reflectance_scale: float | None = None,
        reflectance_offset: float = 0.0,
    ) -> dict[str, Any]:
        """One row of statistics per image.

        Args:
            image_paths: rasters in time order.
            index: one of ``INDICES`` (``ndvi``, ``ndwi``, ``ndbi``, ``evi``, ``savi``).
            date_strings: a label per image (``"2024-06"``, ``"2024-06-15"``, ...); defaults to 0, 1, 2...
            cloud_mask: removed; it never did anything. Use ``mask_paths``. Passing it emits a warning.
            mask_paths: optional raster per image (same size) where a non-zero pixel is excluded (cloud,
                shadow, snow...). Build it from your product's quality band.
            reflectance_scale: divide bands by this to get reflectance 0-1. Default (``None``): 10000 for
                integer or large-valued bands (Sentinel-2/Landsat SR products), 1 for floats already in 0-1.
            reflectance_offset: added before scaling, e.g. ``-1000`` for Sentinel-2 processing baseline
                04.00 and later (their digital numbers carry a +1000 offset).

        Returns:
            ``{"index", "dates", "mean", "std", "p10", "p90", "valid_fraction", "n_images", "errors"}``.
            ``None`` in a list means no usable pixel (or the image failed: see ``errors``, which maps the
            path to the reason). ``valid_fraction`` is the share of pixels that were used.

        Pixels are ignored when they are nodata, NaN, masked by ``mask_paths``, have every band used equal
        to 0, or give an index outside [-1, 1] (artefacts).

        Raises:
            ValueError: unknown index; ``date_strings`` or ``mask_paths`` of the wrong length; a band the
                index needs missing from the band map.
            ImportError: rasterio is not installed.
        """
        import numpy as np
        import rasterio

        if cloud_mask is not None:
            warnings.warn("cloud_mask has no effect and will be removed; pass mask_paths=[...] instead",
                          DeprecationWarning, stacklevel=2)
        if index not in self.INDICES:
            raise ValueError(f"unknown index {index!r}; choose from {list(self.INDICES)}")
        bands, fn = self.INDICES[index]
        missing = [b for b in bands if b not in self.band_map]
        if missing:
            raise ValueError(f"index {index!r} needs band(s) {missing}, which sensor {self.sensor!r} does not "
                             f"define; pass band_map= with their positions in your files")
        n = len(image_paths)
        if date_strings is not None and len(date_strings) != n:
            raise ValueError(f"date_strings has {len(date_strings)} entries for {n} images")
        if mask_paths is not None and len(mask_paths) != n:
            raise ValueError(f"mask_paths has {len(mask_paths)} entries for {n} images")

        series: dict[str, Any] = {"dates": list(date_strings) if date_strings is not None else list(range(n)),
                                  "mean": [], "std": [], "p10": [], "p90": [], "valid_fraction": []}
        errors: dict[str, str] = {}
        for i, path in enumerate(image_paths):
            try:
                with rasterio.open(path) as src:
                    arrs = [src.read(self.band_map[b], masked=True) for b in bands]
                    dtypes = [src.dtypes[self.band_map[b] - 1] for b in bands]
                bad = np.zeros(arrs[0].shape, dtype=bool)
                vals = []
                for a in arrs:
                    bad |= np.ma.getmaskarray(a)
                    vals.append(a.astype("float64").filled(np.nan))
                is_int = any(np.issubdtype(np.dtype(d), np.integer) for d in dtypes)
                for v in vals:
                    bad |= ~np.isfinite(v)
                allzero = np.all([v == 0 for v in vals], axis=0)
                scale = reflectance_scale
                if scale is None:
                    top = np.nanmax(np.where(bad, np.nan, vals[0])) if (~bad).any() else 0.0
                    scale = 10000.0 if (is_int or top > 2.0) else 1.0
                refl = [(v + reflectance_offset) / scale for v in vals]
                with np.errstate(invalid="ignore", divide="ignore"):
                    idx = fn(*refl)
                ok = ~bad & ~allzero & np.isfinite(idx) & (idx >= -1) & (idx <= 1)
                if mask_paths is not None:
                    with rasterio.open(mask_paths[i]) as m:
                        mk = m.read(1)
                    if mk.shape != ok.shape:
                        raise ValueError(f"mask {mask_paths[i]} is {mk.shape}, image is {ok.shape}")
                    ok &= mk == 0
                series["valid_fraction"].append(round(float(ok.mean()), 3))
                if ok.any():
                    v = idx[ok]
                    series["mean"].append(round(float(v.mean()), 4))
                    series["std"].append(round(float(v.std()), 4))
                    series["p10"].append(round(float(np.percentile(v, 10)), 4))
                    series["p90"].append(round(float(np.percentile(v, 90)), 4))
                else:
                    for k in ("mean", "std", "p10", "p90"):
                        series[k].append(None)
            except Exception as exc:
                errors[str(path)] = f"{type(exc).__name__}: {exc}"
                logger.warning("Time series step failed for %s: %s: %s", path, type(exc).__name__, exc)
                for k in ("mean", "std", "p10", "p90", "valid_fraction"):
                    series[k].append(None)
        series["index"] = index
        series["n_images"] = n
        series["errors"] = errors
        return series

    # ── analysis ────────────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        series: dict,
        method: str = "zscore",
        threshold: float | None = None,
        period: int | None = None,
    ) -> list[dict]:
        """Flag dates whose mean is unusual for the series.

        Args:
            series: output of :meth:`compute_index_series`.
            method: ``"zscore"`` (distance from the mean in standard deviations; default threshold 2.5) or
                ``"mad"`` (modified z-score from the median and the median absolute deviation, which one
                outlier cannot distort; default threshold 3.5).
            threshold: absolute score above which a date is reported.
            period: number of steps in one seasonal cycle (12 for monthly data over several years). When
                given, each value is compared with the others at the same position in the cycle, so the
                normal summer-high / winter-low swing is not reported as anomalous. Needs at least two
                full cycles.

        Returns:
            ``[{"date", "value", "expected", "zscore", "type": "low"|"high"}]`` in time order. Dates whose
            value is ``None`` are skipped and never shift the dates of the others.

        Notes:
            With ``n`` points a plain z-score can never exceed ``(n-1)/sqrt(n)``, so a threshold of 2.5
            cannot flag anything in fewer than 9 points; use ``"mad"`` or a lower threshold for short series.
            Fewer than 5 usable points raises ``ValueError`` rather than answering "no anomalies".
        """
        import numpy as np

        if method not in ("zscore", "mad"):
            raise ValueError(f"unknown method {method!r}; use 'zscore' or 'mad'")
        if threshold is None:
            threshold = 2.5 if method == "zscore" else 3.5
        dates = series.get("dates", [])
        means = series.get("mean", [])
        pairs = [(i, dates[i] if i < len(dates) else i, v) for i, v in enumerate(means) if v is not None]
        if len(pairs) < 5:
            raise ValueError(f"need at least 5 dates with data, got {len(pairs)}")
        pos = np.array([p[0] for p in pairs])
        y = np.array([p[2] for p in pairs], dtype="float64")

        expected = np.empty_like(y)
        if period:
            if period < 2 or (pos.max() + 1) < 2 * period:
                raise ValueError(f"period={period} needs at least two full cycles ({2 * period} dates)")
            resid = np.empty_like(y)
            for ph in range(period):
                sel = (pos % period) == ph
                if sel.sum() < 2:
                    raise ValueError(f"cycle position {ph} has fewer than 2 observations; cannot judge anomalies there")
                expected[sel] = np.median(y[sel]) if method == "mad" else y[sel].mean()
                resid[sel] = y[sel] - expected[sel]
        else:
            expected[:] = np.median(y) if method == "mad" else y.mean()
            resid = y - expected

        if method == "zscore":
            scale = resid.std()
            scores = resid / scale if scale > 1e-12 else np.zeros_like(resid)
        else:
            centred = resid - np.median(resid)
            mad = np.median(np.abs(centred))
            if mad > 1e-12:
                scores = 0.6745 * centred / mad
            else:       # more than half the values sit exactly on the expectation: use the mean deviation instead
                mean_ad = np.mean(np.abs(centred))
                scores = centred / (1.253314 * mean_ad) if mean_ad > 1e-12 else np.zeros_like(resid)
        return [
            {"date": pairs[k][1], "value": float(y[k]), "expected": round(float(expected[k]), 4),
             "zscore": round(float(scores[k]), 2), "type": "low" if scores[k] < 0 else "high"}
            for k in range(len(pairs)) if abs(scores[k]) > threshold
        ]

    def compute_trend(self, series: dict, stable_threshold: float = 0.001, period: int | None = None) -> dict[str, Any]:
        """Least-squares line through the series.

        ``slope`` is the change **per time step** (steps count positions in the series, so a missing date
        leaves a gap rather than squeezing the line); when every date parses (``YYYY``, ``YYYY-MM``,
        ``YYYY-MM-DD``, ``YYYYMMDD``) ``slope_per_year`` is also given and uses the real spacing.
        ``p_value`` tests slope = 0 and ``significant`` is ``p_value < 0.05``. ``direction`` is
        "increasing"/"decreasing" when ``|slope| > stable_threshold`` else "stable".

        A seasonal series (summer high, winter low) hides a long-term trend: the plain fit then has a huge
        residual and a large p-value. Give ``period`` (steps per cycle, e.g. 12 for monthly data) to fit
        one level per position in the cycle plus a **common slope**, which is the trend with the seasonal
        cycle removed. It needs at least two full cycles and two observations per cycle position;
        ``slope_per_year`` is not computed in this mode.

        Fewer than 3 usable points gives ``{"trend": "insufficient_data", "direction": "insufficient_data"}``.
        """
        import numpy as np

        means = series.get("mean", [])
        dates = series.get("dates", [])
        pts = [(i, v) for i, v in enumerate(means) if v is not None]
        if len(pts) < 3:
            return {"trend": "insufficient_data", "direction": "insufficient_data"}
        x = np.array([p[0] for p in pts], dtype="float64")
        y = np.array([p[1] for p in pts], dtype="float64")

        if period:
            if period < 2 or len(means) < 2 * period:
                raise ValueError(f"period={period} needs at least two full cycles ({2 * period} dates)")
            phase = (x.astype(int) % period)
            counts = np.bincount(phase, minlength=period)
            if (counts < 2).any():
                raise ValueError("every position in the cycle needs at least 2 observations; "
                                 f"positions {np.flatnonzero(counts < 2).tolist()} have fewer")
            design = np.column_stack([(phase == k).astype("float64") for k in range(period)] + [x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            slope = float(beta[-1])
            fitted = design @ beta
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            dof = len(y) - design.shape[1]
            p_value = None
            if dof > 0:
                try:
                    from scipy import stats
                    cov = np.linalg.pinv(design.T @ design) * (ss_res / dof)
                    se = float(np.sqrt(cov[-1, -1]))
                    p_value = 1.0 if se == 0 and slope == 0 else (0.0 if se == 0 else float(2 * stats.t.sf(abs(slope / se), dof)))
                except ImportError:
                    pass
            return {
                "slope": round(slope, 6),
                "intercept": None,
                "r_squared": round(0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot, 4),
                "p_value": None if p_value is None else round(p_value, 4),
                "significant": None if p_value is None else p_value < 0.05,
                "direction": "increasing" if slope > stable_threshold else "decreasing" if slope < -stable_threshold else "stable",
                "period": period,
            }

        slope, intercept = np.polyfit(x, y, 1)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
        p_value = None
        try:
            from scipy import stats
            if ss_tot >= 1e-12:
                p_value = float(stats.linregress(x, y).pvalue)
        except ImportError:
            pass
        out: dict[str, Any] = {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(r2, 4),
            "p_value": None if p_value is None else round(p_value, 4),
            "significant": None if p_value is None else p_value < 0.05,
            "direction": "increasing" if slope > stable_threshold else "decreasing" if slope < -stable_threshold else "stable",
        }
        parsed = [_parse_date(dates[i]) if i < len(dates) else None for i, _ in pts]
        if all(d is not None for d in parsed):
            yrs = np.array([d.year + (d.timetuple().tm_yday - 1) / 365.25 for d in parsed])
            if np.ptp(yrs) > 0:
                out["slope_per_year"] = round(float(np.polyfit(yrs, y, 1)[0]), 6)
        return out

    def extract_point(self, image_paths: list[str], lon: float, lat: float, index: str = "ndvi",
                      date_strings: list[str] | None = None, reflectance_scale: float | None = None,
                      reflectance_offset: float = 0.0) -> dict[str, Any]:
        """The index at one location (WGS84 ``lon``, ``lat``) for every image.

        Returns ``{"index", "dates", "mean", "location", "n_images", "errors"}`` where ``mean`` holds the
        value at the pixel containing the point, or ``None`` if the point is outside the image, nodata, or
        the value is invalid. The result also works with :meth:`detect_anomalies`, :meth:`compute_trend`
        and :meth:`plot`. Scaling arguments behave as in :meth:`compute_index_series`.
        """
        import numpy as np
        import rasterio
        from rasterio.warp import transform

        if index not in self.INDICES:
            raise ValueError(f"unknown index {index!r}; choose from {list(self.INDICES)}")
        bands, fn = self.INDICES[index]
        missing = [b for b in bands if b not in self.band_map]
        if missing:
            raise ValueError(f"index {index!r} needs band(s) {missing}, which sensor {self.sensor!r} does not define")
        n = len(image_paths)
        if date_strings is not None and len(date_strings) != n:
            raise ValueError(f"date_strings has {len(date_strings)} entries for {n} images")
        vals: list[float | None] = []
        errors: dict[str, str] = {}
        for path in image_paths:
            try:
                with rasterio.open(path) as src:
                    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                    row, col = src.index(xs[0], ys[0])
                    if not (0 <= row < src.height and 0 <= col < src.width):
                        vals.append(None)
                        continue
                    win = ((row, row + 1), (col, col + 1))
                    px = [src.read(self.band_map[b], window=win, masked=True) for b in bands]
                    is_int = any(np.issubdtype(np.dtype(src.dtypes[self.band_map[b] - 1]), np.integer) for b in bands)
                if any(np.ma.getmaskarray(p).any() for p in px):
                    vals.append(None)
                    continue
                v = [float(p[0, 0]) for p in px]
                scale = reflectance_scale if reflectance_scale is not None else (
                    10000.0 if (is_int or max(v) > 2.0) else 1.0)
                value = float(fn(*[(x + reflectance_offset) / scale for x in v])) if any(x != 0 for x in v) else float("nan")
                vals.append(round(value, 4) if np.isfinite(value) and -1 <= value <= 1 else None)
            except Exception as exc:
                errors[str(path)] = f"{type(exc).__name__}: {exc}"
                vals.append(None)
        return {"index": index, "dates": list(date_strings) if date_strings is not None else list(range(n)),
                "mean": vals, "location": {"lon": lon, "lat": lat}, "n_images": n, "errors": errors}

    def plot(self, series: dict, title: str | None = None, save_path: str | None = None) -> Any:
        """Line plot of the mean with the P10-P90 band. Returns the matplotlib figure.

        With ``save_path`` the figure is written there and closed; without it the figure is returned (and
        shown when a display backend is active). Raises ``ValueError`` if no date has data and
        ``ImportError`` without matplotlib.
        """
        import matplotlib.pyplot as plt

        dates = series.get("dates", [])
        means = series.get("mean", [])
        rows = [(i, dates[i] if i < len(dates) else i, m) for i, m in enumerate(means) if m is not None]
        if not rows:
            raise ValueError("nothing to plot: no date has a value")
        xs = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(12, 4))
        label = str(series.get("index", "index")).upper()
        ax.plot(xs, [r[2] for r in rows], "b-o", linewidth=2, markersize=4, label=label)
        p10, p90 = series.get("p10", []), series.get("p90", [])
        if len(p10) == len(means) == len(p90) and all(p10[i] is not None and p90[i] is not None for i in xs):
            ax.fill_between(xs, [p10[i] for i in xs], [p90[i] for i in xs], alpha=0.2, color="blue",
                            label="P10-P90 range")
        step = max(1, len(dates) // 12)
        ticks = list(range(0, len(dates), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(dates[i]) for i in ticks], rotation=45, ha="right")
        ax.set_title(title or f"{label} time series")
        ax.set_xlabel("Date")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
        elif "agg" not in plt.get_backend().lower():
            plt.show()
        return fig


__all__ = ["GeoTimeSeries"]
