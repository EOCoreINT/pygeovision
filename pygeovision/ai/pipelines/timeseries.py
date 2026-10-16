"""Pipelines that compare a date with its own history: vegetation anomaly and water extent through time.

Both use the same imagery search and cloud masking as the single-date pipelines (Sentinel-2 where available) and
align every date to the grid of the first. A pixel that is cloud-masked on a date contributes nothing to that date --
it is never counted as "dry" or "normal" -- and every share and area is over the pixels actually observed.
"""
from __future__ import annotations

import csv
import datetime as _dt
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import BasePipeline, PipelineResult
from pygeovision.ai.pipelines._raster import (
    normalized_difference,
    read_stack,
    write_classes,
    write_float,
)
from pygeovision.data.radiometric import real_pixel_area_ha

logger = logging.getLogger(__name__)

R_VCI = "Kogan, F.N. (1995). Application of vegetation index and brightness temperature for drought detection. Advances in Space Research 15(11), 91-100."


def _to_grid(arr: np.ndarray, src_profile: dict[str, Any], ref_profile: dict[str, Any]) -> np.ndarray:
    """Resample ``arr`` (NaN = no data) onto the reference grid, bilinear, without turning NaN into numbers."""
    from rasterio.warp import Resampling, reproject
    out = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=src_profile["transform"], src_crs=src_profile["crs"],
              dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"], resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    return out


def _same_grid(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (a["width"], a["height"], a["crs"]) == (b["width"], b["height"], b["crs"]) and tuple(a["transform"])[:6] == tuple(b["transform"])[:6]


def _pixel_ha(path: Path) -> float:
    import rasterio
    with rasterio.open(path) as src:
        return real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)


# ═══════════════════════════════════════════════════════════════════════════
# Vegetation anomaly
# ═══════════════════════════════════════════════════════════════════════════

ANOMALY_CLASSES = ("Much below normal (z <= -2)", "Below normal (-2 < z <= -1)", "Near normal (|z| < 1)",
                   "Above normal (1 <= z < 2)", "Much above normal (z >= 2)")


class VegetationAnomalyPipeline(BasePipeline):
    """How green is this month compared with the same month in earlier years?

    NDVI of ``date`` (``YYYY-MM``) is compared, pixel by pixel, with the NDVI of the same month in the
    ``baseline_years`` before it (an integer N for the previous N years, or an explicit list of years; at least 3
    years must produce a usable image). Written: ``ndvi_anomaly.tif`` (NDVI minus the baseline mean),
    ``ndvi_zscore.tif`` (that, divided by the baseline standard deviation), ``vci.tif`` (the Vegetation Condition
    Index of Kogan 1995, ``100 (NDVI - min) / (max - min)`` over the baseline years and the target, 0 = the
    lowest on record, 100 = the highest) and ``anomaly_class.tif`` (five standard-deviation classes).

    This is a *vegetation* anomaly. Low values accompany drought but also late planting, harvest, burning, grazing,
    flooding and land-cover change, and cloud residue; the pipeline sees none of rainfall, soil moisture or
    crop type, so do not read it as a drought declaration.
    """
    name = "vegetation_anomaly"
    description = "NDVI anomaly, z-score and Vegetation Condition Index of a month against the same month in earlier years"
    domain = "agriculture"
    tags = ["vegetation", "ndvi", "anomaly", "vci", "time-series"]

    MIN_BASELINE_YEARS = 3

    def run(self, bbox, output_dir="./output", date=None, baseline_years=5, cloud_cover_max=30.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not date or len(str(date)) < 7 or str(date)[4] != "-":
                return self._fail("vegetation_anomaly needs date as YYYY-MM (the month to assess).", t0)
            year, month = int(str(date)[:4]), int(str(date)[5:7])
            if isinstance(baseline_years, int):
                years = [year - k for k in range(1, baseline_years + 1)]
            else:
                years = sorted({int(y) for y in baseline_years})
            if any(y >= year for y in years):
                return self._fail("baseline_years must all be earlier than the year of `date`.", t0)
            if len(years) < self.MIN_BASELINE_YEARS:
                return self._fail(f"at least {self.MIN_BASELINE_YEARS} baseline years are needed to say what is normal.", t0)
            dates = {y: f"{y}-{month:02d}" for y in years}
            target_path = self._search_and_download(bbox, str(date), out / "target", providers=kwargs.get("providers"),
                                                    cloud_cover_max=cloud_cover_max, bands=("red", "nir"))
            if target_path is None:
                return self._fail(f"No usable imagery for {date}.", t0)
            cur, ref_profile = read_stack(target_path)
            ndvi_t = normalized_difference(cur[1], cur[0])
            layers, used, missing = [], [], []
            for y in sorted(years):
                p = self._search_and_download(bbox, dates[y], out / f"baseline_{y}", providers=kwargs.get("providers"),
                                              cloud_cover_max=cloud_cover_max, bands=("red", "nir"))
                if p is None:
                    missing.append(y)
                    continue
                d, prof = read_stack(p)
                nd = normalized_difference(d[1], d[0])
                layers.append(nd if _same_grid(prof, ref_profile) else _to_grid(nd, prof, ref_profile))
                used.append(y)
            if len(used) < self.MIN_BASELINE_YEARS:
                return self._fail(f"Only {len(used)} of the {len(years)} baseline years had usable imagery "
                                  f"({', '.join(map(str, used)) or 'none'}); at least {self.MIN_BASELINE_YEARS} are needed. "
                                  f"No imagery for: {', '.join(map(str, missing))}.", t0)
            base = np.stack(layers)                                  # years x rows x cols, NaN = masked
            n_obs = np.isfinite(base).sum(axis=0)
            enough = n_obs >= self.MIN_BASELINE_YEARS
            with np.errstate(all="ignore"):
                mean = np.nanmean(base, axis=0)
                std = np.nanstd(base, axis=0, ddof=1)
                lo = np.fmin(np.nanmin(base, axis=0), ndvi_t)
                hi = np.fmax(np.nanmax(base, axis=0), ndvi_t)
            valid = np.isfinite(ndvi_t) & enough
            anomaly = np.where(valid, ndvi_t - mean, np.nan).astype("float32")
            ok_std = valid & (std > 1e-4)
            z = np.where(ok_std, (ndvi_t - mean) / np.where(std > 1e-4, std, 1), np.nan).astype("float32")
            ok_vci = valid & (hi - lo > 1e-4)
            vci = np.where(ok_vci, 100.0 * (ndvi_t - lo) / np.where(hi - lo > 1e-4, hi - lo, 1), np.nan).astype("float32")
            cls = np.full(ndvi_t.shape, 2, np.uint8)
            cls[z <= -1], cls[z <= -2], cls[z >= 1], cls[z >= 2] = 1, 0, 3, 4
            cls_valid = np.isfinite(z)
            prof = dict(ref_profile)
            write_float(out / "ndvi_anomaly.tif", anomaly, prof, ["NDVI minus baseline mean"])
            write_float(out / "ndvi_zscore.tif", z, prof, ["(NDVI - baseline mean) / baseline std"])
            write_float(out / "vci.tif", vci, prof, ["Vegetation Condition Index 0-100"])
            path = write_classes(out / "anomaly_class.tif", cls, cls_valid, prof,
                                 [f"{i}={n}" for i, n in enumerate(ANOMALY_CLASSES)] + ["255=nodata"])
            px_ha = _pixel_ha(target_path)
            share = {n: float((cls[cls_valid] == i).mean()) if cls_valid.any() else None for i, n in enumerate(ANOMALY_CLASSES)}
            stats = {
                "mean_ndvi_target": float(np.nanmean(ndvi_t[valid])) if valid.any() else None,
                "mean_ndvi_baseline": float(np.nanmean(mean[valid])) if valid.any() else None,
                "mean_anomaly": float(np.nanmean(anomaly[valid])) if valid.any() else None,
                "mean_vci": float(np.nanmean(vci[ok_vci])) if ok_vci.any() else None,
                "pct_below_normal": (share[ANOMALY_CLASSES[0]] or 0) + (share[ANOMALY_CLASSES[1]] or 0) if cls_valid.any() else None,
                "pct_above_normal": (share[ANOMALY_CLASSES[3]] or 0) + (share[ANOMALY_CLASSES[4]] or 0) if cls_valid.any() else None,
                "area_ha_much_below_normal": float((cls[cls_valid] == 0).sum()) * px_ha,
                "class_shares": share, "baseline_years_used": used, "baseline_years_missing": missing,
                "min_baseline_observations": int(n_obs[valid].min()) if valid.any() else 0,
                "valid_fraction": float(valid.mean()),
            }
            return self._ok(path, stats, {"date": str(date), "baseline_dates": {y: dates[y] for y in used}}, t0)
        except (ValueError, RuntimeError, OSError) as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Water extent through time
# ═══════════════════════════════════════════════════════════════════════════

class WaterExtentTimeseriesPipeline(BasePipeline):
    """Surface-water area on each of several dates, and how often every pixel was water.

    ``dates`` is a list (``YYYY-MM`` ...). On each date water is MNDWI > ``mndwi_threshold`` (Xu 2006), computed on
    the cloud-free pixels only. Output ``water_frequency.tif`` is, per pixel, the share of its cloud-free
    observations in which it was water (needs at least ``min_observations``); ``water_timeseries.csv`` lists, for each
    date, the observed water area, the share of the AOI that was observed and the water share of the observed pixels.

    Compare dates through ``water_fraction_of_valid`` and the observed share, not the raw area: a cloudier date
    shows less water because it sees less ground. Dates observed on fewer than ``min_valid_fraction`` of the AOI
    are listed but left out of the frequency map. Water that is shadow, wet soil, burn scar or dark pavement can be
    mistaken for water, as in any single-index mask.
    """
    name = "water_extent_timeseries"
    description = "Surface-water area for each of several dates and a map of how often each pixel was water (MNDWI)"
    domain = "water"
    tags = ["water", "time-series", "mndwi", "reservoir"]

    def run(self, bbox, output_dir="./output", dates=None, mndwi_threshold=0.0, min_valid_fraction=0.8,
            min_observations=2, cloud_cover_max=30.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not dates or len(dates) < 2:
                return self._fail("water_extent_timeseries needs `dates`: a list of two or more dates.", t0)
            ref_profile: dict[str, Any] | None = None
            ref_path: Path | None = None
            masks: list[tuple[str, np.ndarray, np.ndarray]] = []
            rows: list[dict[str, Any]] = []
            for d in dates:
                sub = out / str(d).replace("-", "_")
                p = self._search_and_download(bbox, d, sub, providers=kwargs.get("providers"),
                                              cloud_cover_max=cloud_cover_max, bands=("green", "swir1"))
                if p is None:
                    rows.append({"date": d, "usable": False, "note": "no usable imagery"})
                    continue
                data, prof = read_stack(p)
                m = normalized_difference(data[0], data[1])
                if ref_profile is None:
                    ref_profile, ref_path = prof, p
                elif not _same_grid(prof, ref_profile):
                    m = _to_grid(m, prof, ref_profile)
                valid = np.isfinite(m)
                water = (m > mndwi_threshold) & valid
                ha = _pixel_ha(ref_path)
                vf = float(valid.mean())
                usable = vf >= min_valid_fraction
                row = {"date": d, "usable": usable, "valid_fraction": vf,
                       "water_area_ha": float(water.sum()) * ha,
                       "water_fraction_of_valid": float(water.sum()) / max(int(valid.sum()), 1)}
                if not usable:
                    row["note"] = f"observed on only {100 * vf:.0f}% of the area; left out of the frequency map"
                rows.append(row)
                if usable:
                    masks.append((d, water, valid))
                write_classes(sub / "water_mask.tif", water.astype(np.uint8), valid, prof,
                              ["0=not water", "1=water", "255=not observed (cloud or no data)"])
            if len(masks) < 2:
                return self._fail(f"Fewer than two dates were observed on at least {100 * min_valid_fraction:.0f}% of the area "
                                  f"({len(masks)} usable of {len(dates)}).", t0)
            wsum = np.sum([w for _, w, _ in masks], axis=0).astype("float32")
            nobs = np.sum([v for _, _, v in masks], axis=0).astype("float32")
            enough = nobs >= min_observations
            freq = np.where(enough, wsum / np.where(nobs > 0, nobs, 1), np.nan).astype("float32")
            path = write_float(out / "water_frequency.tif", freq, ref_profile, ["share of cloud-free observations with water"])
            write_float(out / "valid_observations.tif", nobs, ref_profile, ["number of cloud-free observations"])
            ha = _pixel_ha(ref_path)
            always = enough & (wsum == nobs) & (wsum > 0)
            ever = enough & (wsum > 0)
            with (out / "water_timeseries.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["date", "usable", "valid_fraction", "water_area_ha", "water_fraction_of_valid", "note"])
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in w.fieldnames})
            usable_rows = [r for r in rows if r.get("usable")]
            areas = [r["water_area_ha"] for r in usable_rows]
            stats = {
                "series": rows, "n_dates_requested": len(dates), "n_dates_usable": len(usable_rows),
                "always_water_ha": float(always.sum()) * ha, "ever_water_ha": float(ever.sum()) * ha,
                "variable_water_ha": float((ever & ~always).sum()) * ha,
                "min_water_area_ha": min(areas), "max_water_area_ha": max(areas),
                "mndwi_threshold": mndwi_threshold, "min_valid_fraction": min_valid_fraction,
                "valid_fraction": float(enough.mean()),
                "timeseries_file": str(out / "water_timeseries.csv"),
            }
            return self._ok(path, stats, {"dates": list(dates), "date_before": usable_rows[0]["date"],
                                          "date_after": usable_rows[-1]["date"]}, t0)
        except (ValueError, RuntimeError, OSError) as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Mine closure and rehabilitation
# ═══════════════════════════════════════════════════════════════════════════

REHAB_CLASSES = ("Outside the assessed area (or reference)", "Recovered (>= recovered_rri of reference)",
                 "Recovering (rising)", "Stalled (partial, not rising)", "Bare or sparse", "Declining (falling)",
                 "Water (pit lake, pond)")
def _read_polygons(path: str, what: str):
    """Read a polygon file (GeoJSON, shapefile, GeoPackage ...) or raise ``ValueError`` saying what is wrong with it."""
    if not Path(str(path)).exists():
        raise ValueError(f"`{what}` file not found: {path}")
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - geopandas is a core dependency of the geo extra
        raise ValueError(f"`{what}` needs geopandas (pip install 'pygeovision[geo]').") from exc
    gdf = gpd.read_file(path)
    gdf = gdf[[g is not None and not g.is_empty for g in gdf.geometry]]
    if gdf.empty:
        raise ValueError(f"`{what}` file has no geometries: {path}")
    if gdf.crs is None:
        raise ValueError(f"`{what}` file has no coordinate system, so it cannot be placed on the imagery: {path}")
    if not all(t in ("Polygon", "MultiPolygon") for t in gdf.geometry.geom_type):
        raise ValueError(f"`{what}` must contain polygons only (found {sorted(set(gdf.geometry.geom_type))}): {path}")
    return gdf


def _burn(gdf, profile: dict[str, Any]) -> np.ndarray:
    """Polygons -> boolean mask on the raster grid described by ``profile``."""
    from rasterio.features import rasterize
    g = gdf.to_crs(profile["crs"])
    ras = rasterize([(geom, 1) for geom in g.geometry], out_shape=(profile["height"], profile["width"]),
                    transform=profile["transform"], fill=0, dtype="uint8", all_touched=False)
    return ras.astype(bool)


def _when(value: Any) -> _dt.date:
    """``YYYY-MM`` (taken as the 15th) or ``YYYY-MM-DD`` -> a date. A year alone cannot place a date in a season."""
    text = str(value).strip()
    try:
        if len(text) == 7:
            return _dt.date(int(text[:4]), int(text[5:7]), 15)
        if len(text) == 10:
            return _dt.date.fromisoformat(text)
    except ValueError:
        pass
    raise ValueError(f"date {value!r} must be YYYY-MM or YYYY-MM-DD (the season matters, so a bare year is not accepted)")


def _ols_slope(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-pixel least-squares slope of ``values`` (T x rows x cols, NaN = missing) against ``t`` (T,), NaN where < 2 points."""
    w = np.isfinite(values)
    n = w.sum(axis=0).astype("float64")
    tt = t.reshape(-1, 1, 1).astype("float64")
    v0 = np.where(w, values, 0.0).astype("float64")
    with np.errstate(all="ignore"):
        tm = (w * tt).sum(axis=0) / n
        ym = v0.sum(axis=0) / n
        sxx = (w * (tt - tm) ** 2).sum(axis=0)
        sxy = (w * (tt - tm) * (v0 - ym)).sum(axis=0)
        slope = np.where((n >= 2) & (sxx > 1e-9), sxy / np.where(sxx > 1e-9, sxx, 1), np.nan)
    return slope.astype("float32")


class MineRehabilitationPipeline(BasePipeline):
    """Is the disturbed ground of a mine, tailings facility or quarry coming back, and where is it not?

    Greenness (NDVI) of the site is compared, on every date, with an undisturbed **reference** observed on the same
    date, which takes the season out: ``RRI = site NDVI / median reference NDVI``. RRI of 1 means the site is as
    green as the reference; 0.8 (``recovered_rri``) is read as recovered. Give ``dates`` (three or more,
    ``YYYY-MM`` or ``YYYY-MM-DD``, in the **same season** of different years) and at least one of ``site`` and
    ``reference`` (polygon files, any coordinate system):

    * ``site`` only: the reference is chosen automatically as persistently vegetated land (median NDVI >=
      ``reference_min_ndvi``) outside the site and its ``reference_buffer_m`` margin.
    * ``reference`` only: everything outside the reference is assessed.
    * both: the site is assessed against your reference polygons (any part inside the site is ignored).

    Each assessed pixel gets the mean RRI of the last ``recent_dates`` dates and the least-squares trend of RRI
    over all dates, and is classed: recovered (RRI >= ``recovered_rri``); recovering (trend >= +``trend_threshold``
    per year); declining (<= -``trend_threshold``); bare or sparse (RRI < ``bare_rri``); otherwise stalled. Pixels that were
    water on at least half their observations (MNDWI > ``mndwi_threshold``) are a separate class, not "bare". Cloud is
    nodata, never a class, and a pixel needs ``min_observations`` clear dates. Dates seen on less than
    ``min_valid_fraction`` of the site, or with fewer than ``min_reference_px`` reference pixels, are listed and
    left out.

    Written: ``rehab_status.tif`` (classes), ``recovery_index.tif`` (recent RRI), ``recovery_trend.tif`` (RRI per
    year), ``reference_mask.tif`` and ``rehabilitation_timeseries.csv`` (site median RRI per date).

    Limits. Greenness is not ecosystem recovery: a dense weed or plantation monoculture is "recovered" by this
    measure and a dry-season woodland reference makes the ratio unstable. The reference must be the ecosystem the closure
    plan is aiming for. Per-pixel trends come from a handful of noisy dates; read the site-level series
    (``site_trend_per_year``) before the per-pixel trend map, and sieve the class raster before counting patches.
    The pipeline does not predict when a site will meet its closure criteria, and it is not a compliance finding:
    completion criteria are set by regulators and are normally field-based.
    """
    name = "mine_rehabilitation"
    description = "Revegetation of a mine or quarry site against an undisturbed reference over several same-season dates"
    domain = "mining"
    tags = ["mining", "rehabilitation", "closure", "revegetation", "ndvi", "time-series"]

    MIN_DATES = 3
    MAX_SEASON_SPREAD_DAYS = 75

    def run(self, bbox, output_dir="./output", dates=None, site=None, reference=None, recovered_rri=0.8,
            bare_rri=0.4, trend_threshold=0.03, recent_dates=2, min_observations=3, min_valid_fraction=0.6,
            min_reference_px=30, reference_min_ndvi=0.5, reference_buffer_m=100.0, mndwi_threshold=0.0,
            cloud_cover_max=30.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            # ---- everything that can be checked before a download is checked first ----
            if not dates or len(dates) < self.MIN_DATES:
                return self._fail(f"mine_rehabilitation needs `dates`: {self.MIN_DATES} or more dates (YYYY-MM), one per year, "
                                  "in the same season.", t0)
            when = {str(d): _when(d) for d in dates}
            if len(set(when)) != len(when):
                return self._fail("`dates` contains the same date twice.", t0)
            if site is None and reference is None:
                return self._fail("Give `site` (a polygon file of the disturbed land) or `reference` (a polygon file of "
                                  "undisturbed land to compare with), or both. Without either the pipeline cannot tell "
                                  "which ground is being rehabilitated.", t0)
            if not 0 < bare_rri < recovered_rri:
                return self._fail("Need 0 < bare_rri < recovered_rri.", t0)
            if min_observations < 3:
                return self._fail("min_observations must be at least 3: two dates cannot show a trend.", t0)
            if recent_dates < 1:
                return self._fail("recent_dates must be at least 1.", t0)
            site_gdf = _read_polygons(site, "site") if site is not None else None
            ref_gdf = _read_polygons(reference, "reference") if reference is not None else None

            # ---- imagery: NDVI and MNDWI of every date on the grid of the first ----
            ordered = sorted(when, key=lambda d: when[d])
            ref_profile: dict[str, Any] | None = None
            ref_path: Path | None = None
            ndvi: dict[str, np.ndarray] = {}
            wet: dict[str, np.ndarray] = {}
            seen: dict[str, np.ndarray] = {}
            rows: dict[str, dict[str, Any]] = {}
            for d in ordered:
                p = self._search_and_download(bbox, d, out / d.replace("-", "_"), providers=kwargs.get("providers"),
                                              cloud_cover_max=cloud_cover_max, bands=("green", "red", "nir", "swir1"))
                if p is None:
                    rows[d] = {"date": d, "usable": False, "note": "no usable imagery"}
                    continue
                data, prof = read_stack(p)
                nd = normalized_difference(data[2], data[1])
                mw = normalized_difference(data[0], data[3])
                if ref_profile is None:
                    ref_profile, ref_path = prof, p
                elif not _same_grid(prof, ref_profile):
                    nd, mw = _to_grid(nd, prof, ref_profile), _to_grid(mw, prof, ref_profile)
                ok = np.isfinite(nd) & np.isfinite(mw)
                ndvi[d] = np.where(ok, nd, np.nan).astype("float32")
                wet[d] = ok & (mw > mndwi_threshold)
                seen[d] = ok
                rows[d] = {"date": d, "usable": True}
            if ref_profile is None or ref_path is None or len(ndvi) < self.MIN_DATES:
                return self._fail(f"Only {len(ndvi)} of {len(ordered)} dates had usable imagery; at least {self.MIN_DATES} are needed. "
                                  f"No imagery for: {', '.join(d for d in ordered if d not in ndvi)}.", t0)
            px_ha = _pixel_ha(ref_path)
            side_m = math.sqrt(px_ha * 1e4)
            shape = (ref_profile["height"], ref_profile["width"])

            # ---- where is the site and where is the reference ----
            site_mask = _burn(site_gdf, ref_profile) if site_gdf is not None else None
            user_ref = _burn(ref_gdf, ref_profile) if ref_gdf is not None else None
            if site_mask is not None and not site_mask.any():
                return self._fail("The `site` polygons do not overlap the imagery (check the bbox and the file's coordinates).", t0)
            if user_ref is not None and not user_ref.any():
                return self._fail("The `reference` polygons do not overlap the imagery (check the bbox and the file's coordinates).", t0)
            assessed = site_mask if site_mask is not None else ~user_ref          # type: ignore[operator]
            stack_all = np.stack([ndvi[d] for d in ndvi])
            n_seen = np.isfinite(stack_all).sum(axis=0)
            wet_n = np.sum([wet[d] for d in ndvi], axis=0)
            water = (n_seen >= min_observations) & (wet_n >= 0.5 * np.maximum(n_seen, 1))
            overlap_removed = 0
            if user_ref is not None:
                ref_mask = user_ref & ~assessed
                overlap_removed = int((user_ref & assessed).sum())
                ref_source = "your reference polygons"
            else:
                from scipy.ndimage import binary_dilation
                margin = int(math.ceil(reference_buffer_m / side_m))
                near = binary_dilation(assessed, structure=np.ones((3, 3), bool), iterations=max(margin, 1)) if margin > 0 else assessed
                with np.errstate(all="ignore"):
                    med = np.nanmedian(stack_all, axis=0)
                ref_mask = (~near) & (n_seen >= min_observations) & (med >= reference_min_ndvi)
                ref_source = (f"automatic: land outside the site and a {reference_buffer_m:g} m margin whose median NDVI is "
                              f">= {reference_min_ndvi:g}")
            ref_mask &= ~water
            if int(ref_mask.sum()) < min_reference_px:
                how = ("lower `reference_min_ndvi`, widen the bbox so it takes in undisturbed vegetation, or give `reference` polygons"
                       if user_ref is None else "draw a larger reference or widen the bbox")
                return self._fail(f"Only {int(ref_mask.sum())} reference pixels were found ({ref_source}); at least "
                                  f"{min_reference_px} are needed to say what 'recovered' looks like. To fix: {how}.", t0)

            # ---- per date: relative recovery index and which dates can be trusted ----
            land = assessed & ~water
            rri: dict[str, np.ndarray] = {}
            n_assessed = int(assessed.sum())
            for d in ndvi:
                v = ndvi[d]
                row = rows[d]
                ref_vals = v[ref_mask & np.isfinite(v)]
                site_valid = land & np.isfinite(v)
                row.update(site_valid_fraction=float(site_valid.sum()) / max(int(land.sum()), 1),
                           reference_pixels=int(ref_vals.size))
                if ref_vals.size < min_reference_px:
                    row.update(usable=False, note=f"only {ref_vals.size} reference pixels observed")
                    continue
                ref_med = float(np.median(ref_vals))
                row["reference_ndvi_median"] = ref_med
                if ref_med < 0.1:
                    row.update(usable=False, note=f"reference NDVI {ref_med:.2f} is too low for a ratio (dormant or burnt?)")
                    continue
                if row["site_valid_fraction"] < min_valid_fraction:
                    row.update(usable=False, note=f"site observed on only {100 * row['site_valid_fraction']:.0f}%; left out")
                    continue
                r = np.where(land & np.isfinite(v), v / ref_med, np.nan).astype("float32")
                rri[d] = r
                row.update(site_ndvi_median=float(np.median(v[site_valid])), site_rri_median=float(np.median(r[site_valid])),
                           share_recovered=float((r[site_valid] >= recovered_rri).mean()))
            used = [d for d in ordered if d in rri]
            if len(used) < max(self.MIN_DATES, min_observations):
                reasons = "; ".join(f"{d}: {rows[d].get('note', 'ok')}" for d in ordered if d not in rri)
                return self._fail(f"Only {len(used)} usable dates (need {max(self.MIN_DATES, min_observations)}). {reasons}.", t0)

            # ---- per pixel: recent level and trend ----
            stack = np.stack([rri[d] for d in used])
            years = np.array([(when[d] - when[used[0]]).days / 365.25 for d in used])
            nobs = np.isfinite(stack).sum(axis=0)
            k = min(recent_dates, len(used))
            recent_n = np.isfinite(stack[-k:]).sum(axis=0)
            with np.errstate(all="ignore"):
                recent = np.where(recent_n > 0, np.nansum(stack[-k:], axis=0) / np.maximum(recent_n, 1), np.nan).astype("float32")
            slope = _ols_slope(stack, years)
            classified = land & (nobs >= min_observations) & np.isfinite(recent) & np.isfinite(slope)
            cls = np.zeros(shape, np.uint8)
            up, down = slope >= trend_threshold, slope <= -trend_threshold
            cls[classified] = 3
            cls[classified & (recent < bare_rri)] = 4
            cls[classified & down] = 5
            cls[classified & up] = 2
            cls[classified & (recent >= recovered_rri)] = 1
            cls[assessed & water] = 6
            decided = (~assessed) | classified | (assessed & water)       # everything else in the assessed area is nodata
            prof = dict(ref_profile)
            path = write_classes(out / "rehab_status.tif", cls, decided, prof,
                                 [f"{i}={n}" for i, n in enumerate(REHAB_CLASSES)] + ["255=not enough clear observations"])
            write_float(out / "recovery_index.tif", np.where(classified, recent, np.nan), prof,
                        ["recent relative recovery index (site NDVI / reference NDVI)"])
            write_float(out / "recovery_trend.tif", np.where(classified, slope, np.nan), prof, ["RRI change per year"])
            write_classes(out / "reference_mask.tif", ref_mask.astype(np.uint8), np.ones(shape, bool), prof,
                          ["0=not reference", "1=reference"])

            # ---- site-level series and numbers ----
            site_series = np.array([rows[d]["site_rri_median"] for d in used])
            site_trend = float(np.polyfit(years, site_series, 1)[0]) if len(used) >= 3 and years[-1] > years[0] else None
            doy = [when[d].timetuple().tm_yday for d in used]
            spread = max(min(abs(a - b), 365 - abs(a - b)) for a in doy for b in doy)
            season_warning = (f"the dates are up to {spread} days apart within the year; vegetation differs between seasons "
                              "and the reference only partly removes that. Use the same month in each year."
                              if spread > self.MAX_SEASON_SPREAD_DAYS else None)
            with (out / "rehabilitation_timeseries.csv").open("w", newline="", encoding="utf-8") as fh:
                names = ["date", "usable", "site_valid_fraction", "reference_pixels", "reference_ndvi_median",
                         "site_ndvi_median", "site_rri_median", "share_recovered", "note"]
                w = csv.DictWriter(fh, fieldnames=names)
                w.writeheader()
                for d in ordered:
                    w.writerow({n: rows[d].get(n, "") for n in names})
            land_cls = classified & (cls != 6)
            n_land = int(land_cls.sum())

            def ha(m: np.ndarray) -> float:
                return float(m.sum()) * px_ha

            def share(c: int) -> float | None:
                return float((cls[land_cls] == c).sum()) / n_land if n_land else None

            stats = {
                "assessed_ha": n_assessed * px_ha, "classified_land_ha": ha(land_cls),
                "area_ha_recovered": ha(cls == 1), "area_ha_recovering": ha(cls == 2), "area_ha_stalled": ha(cls == 3),
                "area_ha_bare": ha(cls == 4), "area_ha_declining": ha(cls == 5), "area_ha_water": ha(assessed & water),
                "recovered_fraction": share(1), "recovering_fraction": share(2), "stalled_fraction": share(3),
                "bare_fraction": share(4), "declining_fraction": share(5),
                "recovered_but_declining_ha": ha(classified & (recent >= recovered_rri) & down),
                "mean_rri_latest": float(np.nanmean(recent[classified])) if classified.any() else None,
                "site_rri_first": float(site_series[0]), "site_rri_latest": float(site_series[-1]),
                "site_trend_per_year": site_trend,
                "first_date": used[0], "latest_date": used[-1], "dates_used": used,
                "series": [rows[d] for d in ordered], "series_used": [rows[d] for d in used],
                "n_dates_requested": len(ordered),
                "dates_left_out": {d: rows[d].get("note", "") for d in ordered if d not in rri},
                "reference_pixels": int(ref_mask.sum()), "reference_source": ref_source,
                "reference_overlap_removed_px": overlap_removed,
                "season_warning": season_warning,
                "valid_fraction": float((classified | (assessed & water)).sum()) / max(n_assessed, 1),
                "timeseries_file": str(out / "rehabilitation_timeseries.csv"),
            }
            return self._ok(path, stats, {"dates": used, "date_before": used[0], "date_after": used[-1],
                                          "reference_source": ref_source, "recovered_rri": recovered_rri,
                                          "bare_rri": bare_rri, "trend_threshold": trend_threshold}, t0)
        except (ValueError, RuntimeError, OSError) as exc:
            return self._fail(str(exc), t0)


TIMESERIES_PIPELINES: dict[str, type] = {c.name: c for c in (VegetationAnomalyPipeline, WaterExtentTimeseriesPipeline,
                                                              MineRehabilitationPipeline)}
