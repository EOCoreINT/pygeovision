"""Pipelines that compare a date with its own history: vegetation anomaly and water extent through time.

Both use the same imagery search and cloud masking as the single-date pipelines (Sentinel-2 where available) and
align every date to the grid of the first. A pixel that is cloud-masked on a date contributes nothing to that date --
it is never counted as "dry" or "normal" -- and every share and area is over the pixels actually observed.
"""
from __future__ import annotations

import csv
import logging
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


TIMESERIES_PIPELINES: dict[str, type] = {c.name: c for c in (VegetationAnomalyPipeline, WaterExtentTimeseriesPipeline)}
