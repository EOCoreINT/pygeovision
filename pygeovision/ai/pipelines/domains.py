"""Domain pipelines: published indices, terrain maths and reference products.

Every pipeline here is computed directly from imagery, a DEM or a published
reference map -- there is no neural network in this module, so nothing
here needs trained weights. Each one:

* names the formula it computes and the threshold it applies;
* writes cloud/shadow-masked pixels as nodata (255 for classes, NaN for
  floats) and computes every percentage and area over *valid* pixels only;
* fails with a message that says what to fix (dates missing, API key
  missing, no imagery) instead of returning a result that looks fine.

Pipelines that need a bi-temporal pair take ``date_before`` / ``date_after``
(``YYYY-MM`` or ``YYYY-MM-DD``).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import (  # noqa: F401  (re-exported for callers)
    BasePipeline,
    PipelineResult,
    get_pipeline,
    list_pipelines,
)
from pygeovision.ai.pipelines._raster import (
    area_ha,
    fraction,
    normalized_difference,
    read_stack,
    valid_mask,
    write_classes,
    write_float,
)
from pygeovision.data.radiometric import real_pixel_area_ha

logger = logging.getLogger(__name__)

NDSI_SNOW_THRESHOLD = 0.4  # Hall et al. 1995; used by the MODIS MOD10 snow product


def _need_two_dates(pipeline: BasePipeline, date_before: Any, date_after: Any, what: str) -> str | None:
    if not date_before or not date_after:
        return (f"{pipeline.name} needs date_before and date_after: {what}.")
    return None


def _evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Enhanced Vegetation Index (Huete et al. 2002); NaN where undefined."""
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    out = np.full(nir.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(denom) & (np.abs(denom) > 1e-6)
    out[ok] = 2.5 * (nir[ok] - red[ok]) / denom[ok]
    return np.clip(out, -1.0, 1.0)


def _pixel_ha(path: Path) -> float:
    import rasterio
    with rasterio.open(path) as src:
        return real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)


# ═══════════════════════════════════════════════════════════════════════════
# Vegetation
# ═══════════════════════════════════════════════════════════════════════════

class CropHealthPipeline(BasePipeline):
    """Vegetation-stress anomaly: where NDVI dropped between two dates.

    NDVI = (NIR - red) / (NIR + red). A pixel is flagged when
    ``NDVI_before - NDVI_after > anomaly_threshold`` (default 0.15).
    It flags *any* vegetation that browned or was cleared -- crop or not --
    so mask to your fields before reading it as crop stress. Pixels that
    are cloud-masked in either date are nodata.

    Outputs: ``crop_health_anomaly.tif`` (1 flagged, 0 not, 255 nodata) and
    ``ndvi_change.tif`` (after - before).
    """
    name = "crop_health"
    description = "NDVI drop between two dates (vegetation stress / loss anomaly)"
    domain = "agriculture"
    tags = ["vegetation", "ndvi", "bitemporal"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            anomaly_threshold=0.15, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        err = _need_two_dates(self, date_before, date_after, "an anomaly is a comparison against a baseline")
        if err:
            return self._fail(err, t0)
        try:
            before, after = self._search_and_download_pair(
                bbox, date_before, date_after, out, providers=kwargs.get("providers"),
                bands=("nir", "red"))
            if before is None or after is None:
                return self._fail("No usable imagery for one or both dates.", t0)
            b, profile = read_stack(before)
            a, _ = read_stack(after)
            ndvi_b = normalized_difference(b[0], b[1])
            ndvi_a = normalized_difference(a[0], a[1])
            valid = valid_mask(ndvi_b, ndvi_a)
            drop = ndvi_b - ndvi_a
            flagged = (drop > anomaly_threshold) & valid
            flag_path = write_classes(out / "crop_health_anomaly.tif", flagged, valid, profile,
                                      ["0=not flagged", "1=NDVI dropped", "255=nodata"])
            write_float(out / "ndvi_change.tif", np.where(valid, -drop, np.nan), profile,
                        ["NDVI after - NDVI before"])
            stats = {
                "anomaly_threshold": anomaly_threshold,
                "pct_flagged": fraction(flagged, valid),
                "mean_ndvi_before": float(np.nanmean(ndvi_b[valid])) if valid.any() else None,
                "mean_ndvi_after": float(np.nanmean(ndvi_a[valid])) if valid.any() else None,
                "valid_fraction": float(valid.mean()),
            }
            return self._ok(flag_path, stats, {"date_before": date_before, "date_after": date_after}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class VegetationIndicesPipeline(BasePipeline):
    """NDVI, EVI and NDWI for each date in a time series.

    ``dates`` is a list of dates (``YYYY-MM`` ...). For every date with
    usable imagery the pipeline writes ``<date>/indices.tif`` (bands:
    NDVI, EVI, NDWI) and reports the scene mean of each index over valid
    pixels. ``output_path`` is the output directory.
    """
    name = "vegetation_indices"
    description = "NDVI, EVI and NDWI per date across a time series"
    domain = "agriculture"
    tags = ["vegetation", "ndvi", "evi", "time-series"]

    def run(self, bbox, output_dir="./output", dates=None, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        if not dates:
            return self._fail("vegetation_indices needs `dates`: a list of one or more dates.", t0)
        try:
            per_date: dict[str, dict[str, Any]] = {}
            for date in dates:
                sub = out / str(date).replace("-", "_")
                img = self._search_and_download(
                    bbox, date, sub, providers=kwargs.get("providers"),
                    bands=("blue", "red", "nir", "green"))
                if img is None:
                    continue
                d, profile = read_stack(img)
                blue, red, nir, green = d
                ndvi = normalized_difference(nir, red)
                evi = _evi(blue, red, nir)
                ndwi = normalized_difference(green, nir)
                valid = valid_mask(ndvi)
                path = write_float(sub / "indices.tif", [ndvi, evi, ndwi], profile, ["NDVI", "EVI", "NDWI"])
                per_date[date] = {
                    "mean_ndvi": float(np.nanmean(ndvi[valid])) if valid.any() else None,
                    "mean_evi": float(np.nanmean(evi[valid])) if valid.any() else None,
                    "mean_ndwi": float(np.nanmean(ndwi[valid])) if valid.any() else None,
                    "valid_fraction": float(valid.mean()),
                    "indices_path": str(path),
                }
            if not per_date:
                return self._fail(f"No usable imagery for any of the {len(dates)} requested dates.", t0)
            return self._ok(out, {"n_dates_used": len(per_date), "per_date": per_date}, {"dates": list(dates)}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Water
# ═══════════════════════════════════════════════════════════════════════════

class WaterQualityPipeline(BasePipeline):
    """NDCI chlorophyll index inside the water mask (Sentinel-2).

    NDCI = (red-edge1 - red) / (red-edge1 + red) (Mishra & Mishra 2012)
    is a *relative* chlorophyll-a indicator for turbid inland/coastal
    water; it is not a concentration in ug/l without local calibration.
    Water = NDWI > 0. Band 2 of the output is red reflectance, a relative
    turbidity indicator (not NTU). Needs Sentinel-2 (red-edge band).
    """
    name = "water_quality"
    description = "Relative chlorophyll index (NDCI) and red-reflectance turbidity indicator inside water"
    domain = "water"
    tags = ["water", "ndci", "sentinel-2"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"),
                bands=("green", "red", "rededge1", "nir"))
            if img is None:
                return self._fail("No usable Sentinel-2 imagery found.", t0)
            d, profile = read_stack(img)
            green, red, rededge, nir = d
            ndwi = normalized_difference(green, nir)
            valid = valid_mask(ndwi, red, rededge)
            water = (ndwi > 0) & valid
            ndci = np.where(water, normalized_difference(rededge, red), np.nan)
            turbidity = np.where(water, red, np.nan)
            path = write_float(out / "water_quality.tif", [ndci, turbidity], profile,
                               ["NDCI (relative chlorophyll indicator)",
                                "red reflectance (relative turbidity indicator, not NTU)"])
            stats = {
                "pct_water": fraction(water, valid),
                "mean_ndci_in_water": float(np.nanmean(ndci)) if water.any() else None,
                "mean_red_reflectance_in_water": float(np.nanmean(turbidity)) if water.any() else None,
                "valid_fraction": float(valid.mean()),
                "note": "Relative indicators; calibrate against water samples for concentrations.",
            }
            return self._ok(path, stats, {"date": date}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class _WaterChange(BasePipeline):
    """Shared MNDWI before/after water masks (Xu 2006): (green - SWIR1) / (green + SWIR1) > 0."""

    def _water_pair(self, bbox, out, date_before, date_after, mndwi_threshold, cloud_max, providers):
        before, after = self._search_and_download_pair(
            bbox, date_before, date_after, out, providers=providers,
            cloud_cover_max=cloud_max, bands=("green", "swir1"))
        if before is None or after is None:
            return None
        b, profile = read_stack(before)
        a, _ = read_stack(after)
        m_b = normalized_difference(b[0], b[1])
        m_a = normalized_difference(a[0], a[1])
        valid = valid_mask(m_b, m_a)
        return (m_b > mndwi_threshold) & valid, (m_a > mndwi_threshold) & valid, valid, profile, _pixel_ha(after)


class FloodMappingPipeline(_WaterChange):
    """Flood extent = water on the flood date that was not water before.

    Optical, so it needs a reasonably cloud-free flood-date image. Water is
    MNDWI > ``mndwi_threshold`` (default 0). ``date_before`` is a dry
    reference; permanent water (water on both dates) is separated from
    flood water, which a single-date water mask cannot do.

    Output ``flood_extent.tif``: 0 dry, 1 flooded, 2 permanent water,
    255 nodata (cloud-masked in either date).
    """
    name = "flood_mapping"
    description = "Flooded area = MNDWI water on the flood date minus permanent water from a dry reference"
    domain = "disaster"
    tags = ["flood", "water", "bitemporal", "mndwi"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            mndwi_threshold=0.0, cloud_cover_max=30.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        err = _need_two_dates(self, date_before, date_after,
                              "date_before is a dry reference, date_after the flood date")
        if err:
            return self._fail(err, t0)
        try:
            res = self._water_pair(bbox, out, date_before, date_after, mndwi_threshold,
                                   cloud_cover_max, kwargs.get("providers"))
            if res is None:
                return self._fail("No usable imagery for one or both dates.", t0)
            w_b, w_a, valid, profile, px_ha = res
            classes = np.zeros(w_a.shape, dtype=np.uint8)
            classes[w_a & ~w_b] = 1
            classes[w_a & w_b] = 2
            path = write_classes(out / "flood_extent.tif", classes, valid, profile,
                                 ["0=dry", "1=flooded", "2=permanent water", "255=nodata"])
            stats = {
                "flooded_area_ha": area_ha(classes == 1, valid, px_ha),
                "permanent_water_area_ha": area_ha(classes == 2, valid, px_ha),
                "flooded_fraction": fraction(classes == 1, valid),
                "valid_fraction": float(valid.mean()),
                "mndwi_threshold": mndwi_threshold,
            }
            return self._ok(path, stats, {"date_before": date_before, "date_after": date_after}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class CoastalMonitoringPipeline(_WaterChange):
    """Shoreline change from MNDWI land/water masks on two dates.

    Output ``shoreline_change.tif``: 0 land (both dates), 1 water (both),
    2 land -> water (retreat / inundation), 3 water -> land (advance),
    255 nodata. Use dates at the same tidal state: a tide difference
    between the two images appears as shoreline change.
    """
    name = "coastal_monitoring"
    description = "Land/water change between two dates from MNDWI (shoreline retreat and advance)"
    domain = "coast"
    tags = ["coast", "water", "bitemporal", "mndwi"]

    def run(self, bbox, output_dir="./output", date_before="2020-01", date_after="2024-01",
            mndwi_threshold=0.0, cloud_cover_max=25.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            res = self._water_pair(bbox, out, date_before, date_after, mndwi_threshold,
                                   cloud_cover_max, kwargs.get("providers"))
            if res is None:
                return self._fail("No usable imagery for one or both dates.", t0)
            w_b, w_a, valid, profile, px_ha = res
            classes = np.zeros(w_a.shape, dtype=np.uint8)
            classes[w_b & w_a] = 1
            classes[~w_b & w_a] = 2
            classes[w_b & ~w_a] = 3
            path = write_classes(out / "shoreline_change.tif", classes, valid, profile,
                                 ["0=land", "1=water", "2=land->water", "3=water->land", "255=nodata"])
            stats = {
                "land_to_water_ha": area_ha(classes == 2, valid, px_ha),
                "water_to_land_ha": area_ha(classes == 3, valid, px_ha),
                "net_water_change_ha": area_ha(classes == 2, valid, px_ha) - area_ha(classes == 3, valid, px_ha),
                "valid_fraction": float(valid.mean()),
                "mndwi_threshold": mndwi_threshold,
            }
            return self._ok(path, stats, {"date_before": date_before, "date_after": date_after}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class WetlandMappingPipeline(BasePipeline):
    """Open water vs inundated (wet) vegetation.

    Water = MNDWI > 0; vegetation = EVI > 0.1. Water without vegetation is
    open water; water with vegetation is wet/inundated vegetation. This
    finds wetlands that are wet *at the acquisition date*; dry-season or
    saturated-soil wetlands are not detected.

    Output ``wetland_classification.tif``: 0 other, 1 open water,
    2 wet vegetation, 255 nodata.
    """
    name = "wetland_mapping"
    description = "Open water vs inundated vegetation from MNDWI and EVI"
    domain = "water"
    tags = ["wetland", "water", "mndwi", "evi"]
    _EVI_VEGETATION_THRESHOLD = 0.1

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"),
                bands=("blue", "red", "nir", "green", "swir1"))
            if img is None:
                return self._fail("No usable imagery found.", t0)
            d, profile = read_stack(img)
            blue, red, nir, green, swir1 = d
            mndwi = normalized_difference(green, swir1)
            evi = _evi(blue, red, nir)
            valid = valid_mask(mndwi, evi)
            water = (mndwi > 0) & valid
            veg = (evi > self._EVI_VEGETATION_THRESHOLD) & valid
            classes = np.zeros(mndwi.shape, dtype=np.uint8)
            classes[water & ~veg] = 1
            classes[water & veg] = 2
            path = write_classes(out / "wetland_classification.tif", classes, valid, profile,
                                 ["0=other", "1=open water", "2=wet vegetation", "255=nodata"])
            px_ha = _pixel_ha(img)
            stats = {
                "wet_vegetation_area_ha": area_ha(classes == 2, valid, px_ha),
                "open_water_area_ha": area_ha(classes == 1, valid, px_ha),
                "pct_wet_vegetation": fraction(classes == 2, valid),
                "pct_open_water": fraction(classes == 1, valid),
                "valid_fraction": float(valid.mean()),
            }
            return self._ok(path, stats, {"date": date}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Fire, snow and ice
# ═══════════════════════════════════════════════════════════════════════════

class WildfireSeverityPipeline(BasePipeline):
    """Burn severity from dNBR (Key & Benson 2006).

    NBR = (NIR - SWIR2) / (NIR + SWIR2); dNBR = NBR_before - NBR_after.
    Classes (USGS/MTBS thresholds, simplified to four): dNBR < 0.10
    unburned, 0.10-0.27 low, 0.27-0.66 moderate, >= 0.66 high.
    Pick ``date_before`` shortly before and ``date_after`` shortly after
    the fire, in the same season. Cloud-masked pixels are nodata, not
    "unburned".

    Outputs: ``dnbr.tif`` and ``severity_class.tif`` (0-3, 255 nodata).
    """
    name = "wildfire_severity"
    description = "Burn severity from dNBR (USGS classes) between a pre-fire and a post-fire date"
    domain = "fire"
    tags = ["fire", "dnbr", "bitemporal"]
    _SEVERITY_THRESHOLDS = (0.10, 0.27, 0.66)

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        err = _need_two_dates(self, date_before, date_after, "dNBR compares a pre-fire and a post-fire image")
        if err:
            return self._fail(err, t0)
        try:
            pre, post = self._search_and_download_pair(
                bbox, date_before, date_after, out, providers=kwargs.get("providers"),
                bands=("nir", "swir2"))
            if pre is None or post is None:
                return self._fail("No usable pre-fire or post-fire imagery found.", t0)
            p, profile = read_stack(pre)
            q, _ = read_stack(post)
            nbr_pre = normalized_difference(p[0], p[1])
            nbr_post = normalized_difference(q[0], q[1])
            dnbr = nbr_pre - nbr_post
            valid = valid_mask(dnbr)
            t1, t2, t3 = self._SEVERITY_THRESHOLDS
            classes = np.zeros(dnbr.shape, dtype=np.uint8)
            classes[dnbr >= t1] = 1
            classes[dnbr >= t2] = 2
            classes[dnbr >= t3] = 3
            write_float(out / "dnbr.tif", dnbr, profile, ["dNBR = NBR_before - NBR_after"])
            path = write_classes(out / "severity_class.tif", classes, valid, profile,
                                 ["0=unburned", "1=low", "2=moderate", "3=high", "255=nodata"])
            px_ha = _pixel_ha(pre)
            stats = {
                "mean_dnbr": float(np.nanmean(dnbr[valid])) if valid.any() else None,
                "pct_unburned": fraction(classes == 0, valid),
                "pct_low_severity": fraction(classes == 1, valid),
                "pct_moderate_severity": fraction(classes == 2, valid),
                "pct_high_severity": fraction(classes == 3, valid),
                "burned_area_ha": area_ha(classes >= 1, valid, px_ha),
                "valid_fraction": float(valid.mean()),
            }
            return self._ok(path, stats, {"date_before": date_before, "date_after": date_after}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class SnowCoverPipeline(BasePipeline):
    """Snow/ice extent from NDSI = (green - SWIR1) / (green + SWIR1) > 0.4.

    The threshold is the one used by the MODIS MOD10 snow product (Hall et
    al. 1995). Extent only: snow water equivalent cannot be derived from
    optical imagery. Output ``snow_extent.tif`` (1 snow/ice, 0 none,
    255 nodata).
    """
    name = "snow_cover"
    description = "Snow/ice extent from NDSI > 0.4 on one date"
    domain = "cryosphere"
    tags = ["snow", "ndsi"]

    def run(self, bbox, output_dir="./output", date="2024-01", **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img = self._search_and_download(bbox, date, out, providers=kwargs.get("providers"),
                                            bands=("green", "swir1"))
            if img is None:
                return self._fail("No usable imagery found.", t0)
            d, profile = read_stack(img)
            ndsi = normalized_difference(d[0], d[1])
            valid = valid_mask(ndsi)
            snow = (ndsi > NDSI_SNOW_THRESHOLD) & valid
            path = write_classes(out / "snow_extent.tif", snow, valid, profile,
                                 ["0=no snow/ice", "1=snow/ice", "255=nodata"])
            stats = {
                "snow_area_ha": area_ha(snow, valid, _pixel_ha(img)),
                "snow_cover_fraction": fraction(snow, valid),
                "valid_fraction": float(valid.mean()),
                "ndsi_threshold": NDSI_SNOW_THRESHOLD,
            }
            return self._ok(path, stats, {"date": date}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class GlacierMonitoringPipeline(BasePipeline):
    """Snow/ice extent change between two dates (NDSI > 0.4).

    Compares extent only on pixels that are cloud-free in **both** dates,
    so cloud cover cannot masquerade as glacier retreat. NDSI also maps
    seasonal snow and misses debris-covered ice: choose two dates at the
    end of the melt season (late summer in the northern hemisphere) for a
    glacier signal. Output ``glacier_extent_after.tif`` (1 snow/ice on the
    later date, 0 none, 255 nodata in either date).
    """
    name = "glacier_monitoring"
    description = "Snow/ice area change between two dates from NDSI, on commonly cloud-free pixels"
    domain = "cryosphere"
    tags = ["glacier", "ndsi", "bitemporal"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        err = _need_two_dates(self, date_before, date_after, "area change is a comparison between two dates")
        if err:
            return self._fail(err, t0)
        try:
            before, after = self._search_and_download_pair(
                bbox, date_before, date_after, out, providers=kwargs.get("providers"),
                bands=("green", "swir1"))
            if before is None or after is None:
                return self._fail("No usable imagery for one or both dates.", t0)
            b, profile = read_stack(before)
            a, _ = read_stack(after)
            ndsi_b = normalized_difference(b[0], b[1])
            ndsi_a = normalized_difference(a[0], a[1])
            valid = valid_mask(ndsi_b, ndsi_a)
            snow_b = (ndsi_b > NDSI_SNOW_THRESHOLD) & valid
            snow_a = (ndsi_a > NDSI_SNOW_THRESHOLD) & valid
            path = write_classes(out / "glacier_extent_after.tif", snow_a, valid, profile,
                                 ["0=none", "1=snow/ice (later date)", "255=nodata"])
            px_ha = _pixel_ha(after)
            area_b, area_a = area_ha(snow_b, valid, px_ha), area_ha(snow_a, valid, px_ha)
            stats = {
                "area_ha_before": area_b, "area_ha_after": area_a,
                "area_change_ha": area_a - area_b,
                "area_change_pct": (area_a - area_b) / area_b * 100.0 if area_b > 0 else None,
                "valid_fraction": float(valid.mean()),
                "ndsi_threshold": NDSI_SNOW_THRESHOLD,
            }
            return self._ok(path, stats, {"date_before": date_before, "date_after": date_after}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Thermal (Landsat Collection 2 Level-2 surface temperature)
# ═══════════════════════════════════════════════════════════════════════════

class UrbanHeatIslandPipeline(BasePipeline):
    """Land surface temperature and its hot areas, from Landsat thermal.

    LST (K) = DN x 0.00341802 + 149.0 (USGS Collection 2 Level-2 Surface
    Temperature product), reported in deg C. "Hot" pixels are those more
    than ``threshold_c`` above the **scene mean** -- a relative measure,
    not an urban-minus-rural difference. Needs Landsat (Sentinel-2 has no
    thermal band). Outputs ``lst_celsius.tif`` and a mask file named by the
    subclass.
    """
    name = "urban_heat_island"
    description = "Landsat land surface temperature and areas hotter than the scene mean + threshold"
    domain = "urban"
    tags = ["thermal", "lst", "landsat"]
    _DEFAULT_THRESHOLD_C = 2.0
    _STAT_KEY = "pct_heat_island"
    _MASK_FILE = "heat_island_mask.tif"

    def run(self, bbox, output_dir="./output", date="2024-07", threshold_c=None, **kwargs) -> PipelineResult:
        from pygeovision.data.radiometric import LANDSAT_ST_OFFSET, LANDSAT_ST_SCALE
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        threshold = self._DEFAULT_THRESHOLD_C if threshold_c is None else float(threshold_c)
        try:
            img = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"),
                collections=["landsat-c2-l2"], bands=("thermal",), apply_scale=False)
            if img is None:
                return self._fail("No usable Landsat thermal imagery found (Sentinel-2 has no "
                                  "thermal sensor; this needs Landsat).", t0)
            dn, profile = read_stack(img)
            dn = dn[0]
            valid = np.isfinite(dn) & (dn > 0)
            temp_c = np.where(valid, dn * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET - 273.15, np.nan).astype(np.float32)
            if not valid.any():
                return self._fail("No valid thermal pixels (all masked).", t0)
            mean_c = float(np.nanmean(temp_c))
            hot = ((temp_c - mean_c) > threshold) & valid
            lst_path = write_float(out / "lst_celsius.tif", temp_c, profile, ["land surface temperature (deg C)"])
            write_classes(out / self._MASK_FILE, hot, valid, profile,
                          [f"1=more than {threshold:g} C above scene mean", "255=nodata"])
            stats = {
                "mean_temp_celsius": mean_c,
                "max_temp_celsius": float(np.nanmax(temp_c)),
                "min_temp_celsius": float(np.nanmin(temp_c)),
                self._STAT_KEY: fraction(hot, valid),
                "threshold_c": threshold,
                "valid_fraction": float(valid.mean()),
                "mask_path": str(out / self._MASK_FILE),
            }
            return self._ok(lst_path, stats, {"date": date}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class LandSurfaceTemperaturePipeline(UrbanHeatIslandPipeline):
    """Landsat land surface temperature (same product as ``urban_heat_island``)."""
    name = "land_surface_temperature"
    description = "Landsat Collection 2 Level-2 land surface temperature in deg C"
    domain = "climate"
    tags = ["thermal", "lst", "landsat"]
    _DEFAULT_THRESHOLD_C = 2.0
    _STAT_KEY = "pct_above_threshold"
    _MASK_FILE = "above_threshold_mask.tif"


class VolcanoMonitoringPipeline(UrbanHeatIslandPipeline):
    """Thermal anomalies: pixels more than 20 deg C above the scene mean (default).

    A temperature anomaly is a *candidate* only: it does not distinguish
    lava or fumaroles from wildfire or industrial heat in the same scene.
    Landsat's 100 m thermal band also saturates over active lava, so use it
    for persistent warm ground, not eruption detection.
    """
    name = "volcano_monitoring"
    description = "Landsat thermal anomalies (scene mean + 20 C by default); candidates, not confirmed activity"
    domain = "geology"
    tags = ["thermal", "volcano", "landsat"]
    _DEFAULT_THRESHOLD_C = 20.0
    _STAT_KEY = "pct_thermal_anomaly"
    _MASK_FILE = "thermal_anomaly_mask.tif"


# ═══════════════════════════════════════════════════════════════════════════
# Habitat diversity from the WorldCover reference map
# ═══════════════════════════════════════════════════════════════════════════

def _box_sum(a: np.ndarray, size: int) -> np.ndarray:
    """Sum of ``a`` over a ``size`` x ``size`` window (zero padded, centred)."""
    r = size // 2
    p = np.pad(a.astype(np.float64), ((r + 1, r), (r + 1, r)), mode="constant")
    c = p.cumsum(0).cumsum(1)
    return c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]


def local_shannon(classes: np.ndarray, window_px: int, nodata: int = 0) -> np.ndarray:
    """Shannon diversity H' = -sum(p ln p) of class proportions in a moving window.

    Proportions are taken over the valid (non-nodata) pixels inside each
    window, so borders are not biased. NaN where the window has no valid
    pixel or the centre pixel is nodata.
    """
    if window_px < 1 or window_px % 2 == 0:
        raise ValueError("window_px must be an odd integer >= 1")
    valid = classes != nodata
    total = _box_sum(valid, window_px)
    h = np.zeros(classes.shape, dtype=np.float64)
    for c in np.unique(classes[valid]):
        p = np.divide(_box_sum(classes == c, window_px), total, out=np.zeros_like(total), where=total > 0)
        h -= np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    h[(total == 0) | ~valid] = np.nan
    return h.astype(np.float32)


class BiodiversityHotspotPipeline(BasePipeline):
    """Habitat diversity map from the ESA WorldCover 2021 reference map.

    Computes the Shannon diversity index of land-cover classes inside a
    moving ``window_px`` x ``window_px`` window (10 m pixels, default 21 =
    210 m) and flags the top ``100 - hotspot_percentile`` percent as
    hotspots. Habitat heterogeneity is an indirect, widely used proxy for
    biodiversity potential; it is not a species count. The land-cover map
    is the static 2021 product, so this describes 2021 whatever the date.

    Outputs: ``shannon_diversity.tif`` (the returned ``output_path``),
    ``hotspot_mask.tif`` and ``land_cover/land_cover.tif``.
    """
    name = "biodiversity_hotspot"
    description = "Local Shannon habitat-diversity map from WorldCover 2021, with top-decile hotspots"
    domain = "ecology"
    tags = ["ecology", "reference-product", "worldcover"]

    def run(self, bbox, output_dir="./output", window_px=21, hotspot_percentile=90.0, **kwargs) -> PipelineResult:
        from pygeovision.ai.labeling.esa_worldcover import WORLDCOVER_CLASSES
        from pygeovision.ai.pipelines import LandCoverPipeline
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            lc = LandCoverPipeline(self.pgv).run(bbox, out / "land_cover", source="worldcover")
            if not lc.success or lc.output_path is None:
                return self._fail(f"land cover unavailable: {lc.error}", t0)
            classes, profile = read_stack(lc.output_path)
            classes = np.nan_to_num(classes[0], nan=0).astype(np.uint8)
            valid = classes != 0
            if not valid.any():
                return self._fail("WorldCover has no data for this bounding box (open ocean?).", t0)
            h = local_shannon(classes, int(window_px))
            threshold = float(np.nanpercentile(h, hotspot_percentile))
            # h > 0: on a uniform area the percentile is 0 and every pixel would otherwise be a "hotspot"
            hotspot = (h >= threshold) & (h > 0) & np.isfinite(h)
            h_path = write_float(out / "shannon_diversity.tif", h, profile,
                                 [f"Shannon H' over a {window_px}x{window_px} px window (ln units)"])
            write_classes(out / "hotspot_mask.tif", hotspot, np.isfinite(h), profile,
                          [f"1=top {100 - hotspot_percentile:g}% diversity", "255=nodata"])
            uniq, counts = np.unique(classes[valid], return_counts=True)
            props = counts / counts.sum()
            region_h = float(-(props * np.log(props)).sum())
            stats = {
                "window_px": int(window_px),
                "mean_local_shannon": float(np.nanmean(h)),
                "max_local_shannon": float(np.nanmax(h)),
                "hotspot_threshold": threshold,
                "hotspot_fraction": fraction(hotspot, np.isfinite(h)),
                "region_shannon_index": region_h,
                "region_evenness": region_h / float(np.log(len(uniq))) if len(uniq) > 1 else 0.0,
                "n_habitat_classes": int(len(uniq)),
                "class_breakdown": {WORLDCOVER_CLASSES.get(int(c), f"class_{c}"): float(p)
                                    for c, p in zip(uniq.tolist(), props.tolist())},
                "note": "Habitat-heterogeneity proxy from the static WorldCover 2021 map; not a species count.",
            }
            return self._ok(h_path, stats, {"worldcover_year": 2021}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Terrain (DEM from OpenTopography; needs a free API key)
# ═══════════════════════════════════════════════════════════════════════════

_DEM_CODES = {
    "srtm90": "SRTMGL3", "srtm1arc": "SRTMGL1", "cop30": "COP30", "cop90": "COP90",
    "nasadem": "NASADEM", "alos": "AW3D30", "gedtm30": "GEDTM30",
}


class _DemPipeline(BasePipeline):
    """Shared DEM download and Horn slope/aspect for the terrain pipelines."""
    _KEY_HINT = ("confirm an OpenTopography API key is configured "
                 "(client.add_credentials('opentopography', api_key=...); free at "
                 "https://portal.opentopography.org)")

    def _fetch_dem(self, bbox, out: Path, dem_type: str):
        """Download the DEM product ``dem_type`` and return ``(path, dem, profile, bounds)``.

        ``dem_type`` is an OpenTopography product key (``srtm1arc``,
        ``cop30``, ``nasadem`` ...) or API code (``SRTMGL1`` ...). The
        provider offers several products for one bbox; the requested one
        is selected explicitly rather than taking whichever comes first.
        """
        import rasterio

        code = _DEM_CODES.get(dem_type.lower(), dem_type)
        results = self.pgv.search(
            bbox=bbox, date_range=("2000-01-01", "2000-01-02"),
            providers=["opentopography"], max_results=50,
        )
        if not results:
            raise RuntimeError(f"No DEM returned for this bbox; {self._KEY_HINT}.")
        chosen = [r for r in results
                  if str(getattr(r, "id", "")).startswith(f"opentopo_{code}_")
                  or (getattr(r, "properties", None) or {}).get("dem_type") == code]
        if not chosen:
            offered = sorted({str(getattr(r, "id", "")).split("_")[1] for r in results if "_" in str(getattr(r, "id", ""))})
            raise RuntimeError(f"DEM product {dem_type!r} ({code}) was not offered; got {offered}.")
        downloads = self.pgv.download(chosen[:1], str(out / "dem"))
        if not (downloads and downloads[0].success and downloads[0].path):
            raise RuntimeError("DEM download failed.")
        path = Path(downloads[0].path)
        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            bounds = src.bounds
            nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            dem[dem == nodata] = np.nan
        return path, dem, profile, bounds

    @staticmethod
    def _fill_nan_nearest(dem: np.ndarray) -> np.ndarray:
        """Fill NaN with the nearest valid value (no artificial cliffs)."""
        from scipy import ndimage
        bad = ~np.isfinite(dem)
        if not bad.any():
            return dem
        if bad.all():
            raise RuntimeError("DEM is entirely nodata for this bbox.")
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        return dem[tuple(idx)]

    @staticmethod
    def _cell_size_m(bounds, height: int, width: int) -> tuple[float, float]:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        lat = (bounds.top + bounds.bottom) / 2.0
        dx = abs(geod.inv(bounds.left, lat, bounds.right, lat)[2]) / width
        dy = abs(geod.inv(bounds.left, bounds.bottom, bounds.left, bounds.top)[2]) / height
        return dx, dy

    @staticmethod
    def _horn(dem_filled: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Slope and aspect (radians) by Horn's 3x3 method (as gdaldem).

        Aspect is the compass bearing, clockwise from north, of the
        downhill direction (a south-facing slope is pi).
        """
        z = np.pad(dem_filled, 1, mode="edge")
        dzdx = ((z[2:, :-2] + 2 * z[1:-1, :-2] + z[:-2, :-2])
                - (z[2:, 2:] + 2 * z[1:-1, 2:] + z[:-2, 2:])) / (8 * dx_m)
        dzdy = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
                - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8 * dy_m)
        return np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)), np.arctan2(dzdx, dzdy)


class SolarPotentialPipeline(_DemPipeline):
    """Clear-sky relative solar irradiance on terrain at one date and hour.

    Computes the cosine of the sun's incidence angle on each DEM cell from
    Horn slope/aspect and the sun position (Spencer 1971 declination, the
    standard hour-angle equations; Duffie & Beckman). Output is a 0-1
    modulation factor for **one instant**, not annual kWh/m2: no terrain
    shadows, no atmosphere. Needs an OpenTopography API key.
    """
    name = "solar_potential"
    description = "Clear-sky relative irradiance (0-1) on terrain for a date and hour, from a DEM"
    domain = "energy"
    tags = ["energy", "dem", "needs-api-key"]

    def run(self, bbox, output_dir="./output", date="2024-06-21", hour=12.0, dem_type="srtm1arc", **kwargs) -> PipelineResult:
        from datetime import datetime
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            try:
                day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
            except ValueError:
                return self._fail(f"date must be YYYY-MM-DD, got {date!r}.", t0)
            dem_path, dem, profile, bounds = self._fetch_dem(bbox, out, dem_type)
            nodata_mask = ~np.isfinite(dem)
            height, width = dem.shape
            dx_m, dy_m = self._cell_size_m(bounds, height, width)
            slope, aspect = self._horn(self._fill_nan_nearest(dem), dx_m, dy_m)

            gamma = 2 * np.pi * (day_of_year - 1) / 365.0
            decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                    - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
                    - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
            hour_angle = np.radians(15.0 * (hour - 12.0))
            lat = np.radians((bounds.top + bounds.bottom) / 2.0)
            cos_z = float(np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle), -1, 1))
            zenith = float(np.arccos(cos_z))
            sin_z = np.sin(zenith)
            cos_az = np.clip((np.sin(decl) - np.sin(lat) * cos_z) / (np.cos(lat) * sin_z + 1e-12), -1, 1) if abs(sin_z) > 1e-6 else 1.0
            sun_az = float(np.arccos(cos_az))
            if hour_angle > 0:
                sun_az = 2 * np.pi - sun_az

            cos_inc = np.clip(np.cos(slope) * cos_z + np.sin(slope) * sin_z * np.cos(sun_az - aspect), 0.0, 1.0)
            cos_inc[nodata_mask] = np.nan
            if cos_z <= 0:
                cos_inc[~nodata_mask] = 0.0  # sun below the horizon
            path = write_float(out / "solar_potential.tif", cos_inc, profile,
                               ["relative clear-sky irradiance modulation (0-1, cosine of incidence angle)"])
            valid = np.isfinite(cos_inc)
            stats = {
                "date": date, "hour": hour, "dem_type": dem_type,
                "sun_zenith_deg": float(np.degrees(zenith)), "sun_azimuth_deg": float(np.degrees(sun_az)),
                "mean_relative_irradiance": float(cos_inc[valid].mean()) if valid.any() else None,
                "pct_self_shaded": fraction(cos_inc <= 0, valid),
                "note": "Single-instant relative factor; no terrain shadowing, no atmosphere.",
            }
            return self._ok(path, stats, {"dem_path": str(dem_path)}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class ArchaeologicalSitePipeline(_DemPipeline):
    """Relief-enhancement layers for visual archaeological survey (not detection).

    Writes a Local Relief Model (Hesse 2010: DEM minus a smoothed trend)
    and a multi-directional hillshade (Devereux et al. 2008; 8 azimuths).
    There is no "site found" output -- an expert interprets the layers.
    Resolution caps what is visible: features need to span several DEM
    cells, so a 30 m SRTM/Copernicus DEM shows only landscape-scale
    earthworks (use LiDAR-derived DTMs for small mounds and walls). The
    LRM is unreliable within ``smoothing_radius_px`` of the border; ask
    for a larger bbox than your area of interest. Needs an OpenTopography
    API key.
    """
    name = "archaeological_site"
    description = "Local Relief Model and multi-directional hillshade from a DEM, for expert interpretation"
    domain = "heritage"
    tags = ["heritage", "dem", "needs-api-key"]

    def run(self, bbox, output_dir="./output", smoothing_radius_px=15, dem_type="srtm1arc", **kwargs) -> PipelineResult:
        from scipy.ndimage import uniform_filter
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            dem_path, dem, profile, bounds = self._fetch_dem(bbox, out, dem_type)
            nodata_mask = ~np.isfinite(dem)
            filled = self._fill_nan_nearest(dem)
            lrm = filled - uniform_filter(filled, size=2 * smoothing_radius_px + 1, mode="nearest")
            dx_m, dy_m = self._cell_size_m(bounds, *dem.shape)
            slope, aspect = self._horn(filled, dx_m, dy_m)
            alt = np.radians(45.0)
            shades = [np.clip(np.cos(np.pi / 2 - alt) * np.cos(slope)
                              + np.sin(np.pi / 2 - alt) * np.sin(slope) * np.cos(np.radians(az) - aspect), 0, 1)
                      for az in range(0, 360, 45)]
            hillshade = np.mean(shades, axis=0)
            lrm[nodata_mask] = np.nan
            hillshade[nodata_mask] = np.nan
            lrm_path = write_float(out / "local_relief_model.tif", lrm, profile,
                                   ["Local Relief Model: DEM minus smoothed regional trend (m)"])
            hs_path = write_float(out / "multidirectional_hillshade.tif", hillshade, profile,
                                  ["multi-directional hillshade (8 azimuths, averaged)"])
            stats = {
                "smoothing_radius_px": smoothing_radius_px, "dem_type": dem_type,
                "dem_cell_size_m": [dx_m, dy_m],
                "lrm_std_m": float(np.nanstd(lrm)), "lrm_min_m": float(np.nanmin(lrm)),
                "lrm_max_m": float(np.nanmax(lrm)), "hillshade_path": str(hs_path),
                "note": "Visualisation layers for expert review; no site verdict is produced.",
            }
            return self._ok(lrm_path, stats, {"dem_path": str(dem_path)}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class WindFarmSitingPipeline(_DemPipeline):
    """Terrain and land-use screen for wind-farm siting (not a wind resource assessment).

    Excludes cells steeper than ``max_slope_deg`` (default 15) and, when
    ``use_landcover`` is true, cells that the WorldCover 2021 map marks as
    built-up, snow/ice, water, wetland or mangrove. The score of the
    remaining cells is terrain exposure (elevation minus a regional mean
    over ``exposure_smoothing_px``): locally prominent ground tends to be
    windier, but this does **not** estimate wind speed or energy yield --
    that needs measurements or a wind model. Needs an OpenTopography API
    key; the land-cover mask also needs network access to the WorldCover
    bucket (pass ``use_landcover=False`` to screen on terrain alone).
    """
    name = "wind_farm_siting"
    description = "Slope and land-cover exclusion screen with a terrain-exposure score (no wind speed)"
    domain = "energy"
    tags = ["energy", "dem", "planning", "needs-api-key"]
    _EXCLUDED_CLASSES = (50, 70, 80, 90, 95)  # built-up, snow/ice, water, wetland, mangroves

    def run(self, bbox, output_dir="./output", max_slope_deg=15.0, exposure_smoothing_px=25,
            dem_type="srtm1arc", use_landcover=True, **kwargs) -> PipelineResult:
        import rasterio
        from scipy.ndimage import uniform_filter

        from pygeovision.ai.data.dataset import TileMetadata
        from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            dem_path, dem, profile, bounds = self._fetch_dem(bbox, out, dem_type)
            nodata_mask = ~np.isfinite(dem)
            filled = self._fill_nan_nearest(dem)
            height, width = dem.shape
            dx_m, dy_m = self._cell_size_m(bounds, height, width)
            slope, _ = self._horn(filled, dx_m, dy_m)
            slope_deg = np.degrees(slope)
            exposure = filled - uniform_filter(filled, size=2 * exposure_smoothing_px + 1, mode="nearest")
            allowed = (slope_deg <= max_slope_deg) & ~nodata_mask

            landcover_applied = False
            if use_landcover:
                # WorldCover is resampled straight onto the DEM grid (the
                # labeler only reads the raster's georeferencing).
                with rasterio.open(dem_path) as src:
                    meta = TileMetadata(
                        tile_id=dem_path.stem, source_file=dem_path, bounds=tuple(src.bounds),
                        crs=str(src.crs), transform=list(src.transform)[:6], row_off=0, col_off=0,
                        height=src.height, width=src.width, bands=[1])
                lc_path = out / "landcover_for_exclusion.tif"
                r = ESAWorldCoverLabeler(year=2021).label_tile(dem_path, meta, lc_path)
                if not r.success:
                    return self._fail(f"WorldCover land-use mask unavailable ({r.error}). "
                                      f"Pass use_landcover=False to screen on terrain only.", t0)
                with rasterio.open(lc_path) as src:
                    lc = src.read(1)
                allowed &= ~np.isin(lc, self._EXCLUDED_CLASSES)
                landcover_applied = True

            score = np.where(allowed, exposure, np.nan)
            path = write_float(out / "wind_suitability.tif", score, profile,
                               ["terrain exposure (m) where slope/land-use constraints pass; NaN = excluded (NOT wind speed)"])
            usable = ~nodata_mask
            stats = {
                "max_slope_deg": max_slope_deg, "dem_type": dem_type,
                "landcover_exclusion_applied": landcover_applied,
                "pct_area_suitable_by_slope": fraction((slope_deg <= max_slope_deg), usable),
                "pct_area_suitable_overall": fraction(allowed, usable),
                "suitable_area_ha": float(allowed.sum()) * dx_m * dy_m / 10_000.0,
                "note": "Screening only: no wind speed, direction or energy yield is estimated.",
            }
            return self._ok(path, stats, {"dem_path": str(dem_path)}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_PIPELINES: dict[str, type] = {
    cls.name: cls for cls in (
        CropHealthPipeline, VegetationIndicesPipeline,
        WaterQualityPipeline, FloodMappingPipeline, CoastalMonitoringPipeline, WetlandMappingPipeline,
        WildfireSeverityPipeline, SnowCoverPipeline, GlacierMonitoringPipeline,
        UrbanHeatIslandPipeline, LandSurfaceTemperaturePipeline, VolcanoMonitoringPipeline,
        BiodiversityHotspotPipeline,
        SolarPotentialPipeline, ArchaeologicalSitePipeline, WindFarmSitingPipeline,
    )
}
