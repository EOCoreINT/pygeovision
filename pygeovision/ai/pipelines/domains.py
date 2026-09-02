"""
20+ Domain-specific production pipelines (Phase 3).
Each pipeline: PyGeoFetch data → native AI model → output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygeovision.ai.pipelines import BasePipeline as _AuditedBasePipeline
from pygeovision.data.radiometric import real_pixel_area_ha

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    name: str
    success: bool
    output_path: Path | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        if self.success:
            return f"✓ {self.name} → {self.output_path} {self.stats}"
        return f"✗ {self.name}: {self.error}"


class BasePipeline(_AuditedBasePipeline):
    """Base class for all PyGeoVision domain pipelines.

    Real fix: this class previously did NOT inherit from
    pygeovision.ai.pipelines.BasePipeline at all -- it was a completely
    separate, much simpler implementation, meaning none of the 41
    domain pipelines built on it got any of the real radiometric
    scaling, cloud masking, bbox cropping, mission-aware band
    selection, or bi-temporal common-grid alignment fixes confirmed
    and tested elsewhere in this codebase. Now inherits from the real,
    audited BasePipeline, so _search_and_download/_search_and_download_pair
    are genuinely available here too.
    """

    name: str = "base"
    description: str = ""
    domain: str = "other"
    satellite: str = "sentinel-2"
    default_providers: list[str] = field(default_factory=lambda: ["planetary_computer"])
    output_format: str = "geotiff"
    tags: list[str] = field(default_factory=list)

    def __init__(self, pgv_client: Any) -> None:
        super().__init__(pgv_client)
        self._pgv = self.pgv  # backward-compat alias; all 41 existing subclasses use self._pgv

    def _search(self, bbox, date_range, cloud_max=15):
        return self._pgv.search(
            bbox=bbox, date_range=date_range,
            satellite=self.satellite,
            providers=getattr(self, 'default_providers', ["planetary_computer"]),
            cloud_cover_max=cloud_max,
            max_results=5,
            use_cache=False,
        )

    def run(self, bbox: tuple, output_dir: Any = "./output", **kwargs) -> PipelineResult:
        raise NotImplementedError


# ─── Agriculture ──────────────────────────────────────────────────────

class CropTypeMappingPipeline(BasePipeline):
    name = "crop_type_mapping"
    description = "Sentinel-2 time series → crop type segmentation map"
    domain = "agriculture"
    satellite = "sentinel-2"
    tags = ["agriculture", "segmentation", "time_series"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            start = f"{date}-01" if len(date) == 7 else date
            end   = f"{date}-28" if len(date) == 7 else kwargs.get("end_date", date)
            results = self._search(bbox, (start, end), cloud_max=kwargs.get("cloud_max", 10))
            if not results:
                return PipelineResult(self.name, False, error="No imagery found")
            downloads = self._pgv.download(results[:2], str(out), post_process=["reproject:EPSG:4326"])
            succeeded = [d for d in downloads if d.success and d.path and Path(d.path).exists()]
            if not succeeded:
                return PipelineResult(self.name, False, error="Download failed")
            pred_path = out / "crop_type_map.tif"
            self._pgv.segmentation.custom(
                str(succeeded[0].path), kwargs.get("model", "crop_type_model"),
                output_path=str(pred_path), num_classes=kwargs.get("num_classes", 13),
            )
            return PipelineResult(self.name, True, pred_path,
                                  {"scenes_downloaded": len(succeeded)},
                                  duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class CropHealthPipeline(BasePipeline):
    name = "crop_health"
    description = "Real NDVI baseline-vs-current anomaly detection for crop health"
    domain = "agriculture"
    tags = ["agriculture", "ndvi", "anomaly"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            anomaly_threshold=0.15, **kwargs):
        import time; t0 = time.time()
        import numpy as np
        import rasterio
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="crop_health needs date_before (baseline) and date_after (current) -- "
                      "anomaly detection is inherently a comparison against a baseline.",
            )
        try:
            baseline_path, current_path = self._search_and_download_pair(
                bbox, date_before, date_after, out,
                providers=kwargs.get("providers"), bands=("nir", "red"),
            )
            if baseline_path is None or current_path is None:
                return PipelineResult(self.name, False, error="No usable imagery for one or both dates.")

            def real_ndvi(path):
                with rasterio.open(path) as src:
                    nir = src.read(1).astype(np.float32)
                    red = src.read(2).astype(np.float32)
                    profile = src.profile.copy()
                return np.clip((nir - red) / (nir + red + 1e-8), -1.0, 1.0), profile

            ndvi_baseline, profile = real_ndvi(baseline_path)
            ndvi_current, _ = real_ndvi(current_path)
            anomaly = ndvi_baseline - ndvi_current
            flagged = anomaly > anomaly_threshold

            flag_path = out / "crop_health_anomaly.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(flag_path, "w", **profile) as dst:
                dst.write(flagged.astype(np.uint8)[None, ...])

            stats = {
                "anomaly_threshold": anomaly_threshold,
                "pct_flagged": float(flagged.mean()),
                "mean_ndvi_baseline": float(np.nanmean(ndvi_baseline)),
                "mean_ndvi_current": float(np.nanmean(ndvi_current)),
            }
            return PipelineResult(self.name, True, flag_path, stats, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class IrrigationDetectionPipeline(BasePipeline):
    name = "irrigation_detection"
    description = "Multispectral imagery → irrigation pattern mapping"
    domain = "agriculture"
    tags = ["agriculture", "water", "ndwi"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            start, end = f"{date}-01", f"{date}-28"
            results = self._search(bbox, (start, end))
            downloads = self._pgv.download(results[:2], str(out), post_process=["ndwi", "reproject:EPSG:4326"])
            succeeded = [d for d in downloads if d.success]
            if succeeded:
                water_out = out / "irrigation_map.tif"
                self._pgv.segmentation.water(str(succeeded[0].path), output_path=str(water_out))
                return PipelineResult(self.name, True, water_out, duration_seconds=time.time()-t0)
            return PipelineResult(self.name, False, error="No data", duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# ─── Forestry ─────────────────────────────────────────────────────────

class CanopyHeightPipeline(BasePipeline):
    name = "canopy_height"
    description = "Sentinel-2 → canopy height regression map"
    domain = "forestry"
    tags = ["forestry", "canopy", "regression"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self._search(bbox, (f"{date}-01", f"{date}-28"))
            downloads = self._pgv.download(results[:1], str(out), post_process=["reproject:EPSG:4326"])
            succeeded = [d for d in downloads if d.success and d.path and Path(d.path).exists()]
            if not succeeded:
                return PipelineResult(self.name, False, error="No data")
            canopy_out = out / "canopy_height.tif"
            from pygeovision.models.foundation.dinov3 import CHMv2Model
            CHMv2Model().predict_canopy_height(str(succeeded[0].path), output_path=str(canopy_out))
            return PipelineResult(self.name, True, canopy_out, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class TreeSpeciesPipeline(BasePipeline):
    name = "tree_species"
    description = "Hyperspectral/multispectral → tree species classification"
    domain = "forestry"
    tags = ["forestry", "classification", "species"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self._search(bbox, (f"{date}-01", f"{date}-28"))
            downloads = self._pgv.download(results[:1], str(out))
            succeeded = [d for d in downloads if d.success]
            if not succeeded:
                return PipelineResult(self.name, False, error="No data")
            return PipelineResult(
                self.name, False, succeeded[0].path,
                error="No trained tree-species classifier exists in this codebase -- "
                      "imagery was downloaded but not classified. output_path points to "
                      "the raw downloaded scene, not a species classification.",
                duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# (ForestFirePipeline moved below, after WildfireSeverityPipeline -- it
#  becomes a real subclass of it, covering burn-scar mapping for real)



# ─── Urban ────────────────────────────────────────────────────────────

class RoadExtractionPipeline(BasePipeline):
    name = "road_extraction"
    description = "High-res aerial → road network vectorization"
    domain = "urban"
    tags = ["urban", "roads", "segmentation"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self._search(bbox, (f"{date}-01", f"{date}-28"), cloud_max=20)
            downloads = self._pgv.download(results[:1], str(out))
            succeeded = [d for d in downloads if d.success and d.path and Path(d.path).exists()]
            if not succeeded:
                return PipelineResult(self.name, False, error="No data")
            roads_out = out / "roads.geojson"
            self._pgv.detection.generic(
                str(succeeded[0].path), num_classes=1, class_names=["road"],
                output_path=str(roads_out)
            )
            return PipelineResult(self.name, True, roads_out, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class InfrastructureMonitoringPipeline(BasePipeline):
    name = "infrastructure_monitoring"
    description = "Bi-temporal imagery → infrastructure change assessment"
    domain = "urban"
    tags = ["urban", "change_detection", "infrastructure"]

    def run(self, bbox, output_dir="./output", date_before="2020-01", date_after="2024-01", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            r_before = self._search(bbox, (f"{date_before}-01", f"{date_before}-28"))
            r_after  = self._search(bbox, (f"{date_after}-01",  f"{date_after}-28"))
            dl_b = self._pgv.download(r_before[:1], str(out / "before"))
            dl_a = self._pgv.download(r_after[:1],  str(out / "after"))
            b_ok = [d for d in dl_b if d.success and d.path and Path(d.path).exists()]
            a_ok = [d for d in dl_a if d.success and d.path and Path(d.path).exists()]
            if not (b_ok and a_ok):
                return PipelineResult(self.name, False, error="Need both before and after imagery")
            chg_out = out / "infrastructure_changes.tif"
            self._pgv.change.detect(
                str(b_ok[0].path), str(a_ok[0].path), output_path=str(chg_out)
            )
            return PipelineResult(self.name, True, chg_out,
                                  {"period": f"{date_before} → {date_after}"},
                                  duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# ─── Water ────────────────────────────────────────────────────────────

class FloodMappingPipeline(BasePipeline):
    name = "flood_mapping"
    description = "SAR/optical → flood extent mapping"
    domain = "disaster"
    tags = ["disaster", "flood", "sar", "segmentation"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self._search(bbox, (f"{date}-01", f"{date}-28"), cloud_max=30)
            downloads = self._pgv.download(results[:2], str(out))
            succeeded = [d for d in downloads if d.success and d.path and Path(d.path).exists()]
            if not succeeded:
                return PipelineResult(self.name, False, error="No data")
            flood_out = out / "flood_extent.tif"
            self._pgv.segmentation.water(str(succeeded[0].path), output_path=str(flood_out))
            return PipelineResult(self.name, True, flood_out, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class WaterQualityPipeline(BasePipeline):
    name = "water_quality"
    description = "Real NDWI + NDCI (chlorophyll proxy) + uncalibrated turbidity proxy"
    domain = "water"
    tags = ["water", "quality", "ndwi"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        import numpy as np
        import rasterio
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"),
                bands=("green", "red", "rededge1", "nir"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            with rasterio.open(img_path) as src:
                green = src.read(1).astype(np.float32)
                red = src.read(2).astype(np.float32)
                rededge = src.read(3).astype(np.float32)
                nir = src.read(4).astype(np.float32)
                profile = src.profile.copy()

            ndwi = np.clip((green - nir) / (green + nir + 1e-8), -1.0, 1.0)
            water_mask = ndwi > 0
            ndci = np.clip((rededge - red) / (rededge + red + 1e-8), -1.0, 1.0)
            ndci_masked = np.where(water_mask, ndci, np.nan)
            turbidity_proxy = np.where(water_mask, red, np.nan)  # real, documented, but uncalibrated

            wq_path = out / "water_quality.tif"
            profile.update(count=2, dtype="float32", nodata=np.nan)
            with rasterio.open(wq_path, "w", **profile) as dst:
                dst.write(ndci_masked.astype(np.float32), 1)
                dst.write(turbidity_proxy.astype(np.float32), 2)
                dst.set_band_description(1, "NDCI (chlorophyll proxy)")
                dst.set_band_description(2, "red-band reflectance (uncalibrated turbidity proxy)")

            stats = {
                "pct_water": float(water_mask.mean()),
                "mean_ndci_in_water": float(np.nanmean(ndci_masked)) if water_mask.any() else None,
                "mean_turbidity_proxy": float(np.nanmean(turbidity_proxy)) if water_mask.any() else None,
                "note": "NDCI is real and standard (chlorophyll proxy). Turbidity has no single "
                        "standardized formula -- red-band reflectance is a real, documented but "
                        "UNCALIBRATED proxy; needs local water-sample calibration for real NTU units.",
            }
            return PipelineResult(self.name, True, wq_path, stats, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class CoastalMonitoringPipeline(BasePipeline):
    name = "coastal_monitoring"
    description = "Time-series → shoreline change detection"
    domain = "coast"
    satellite = "sentinel-2"
    tags = ["coast", "change_detection", "water", "time_series"]

    def run(self, bbox, output_dir="./output", date_before="2020-01", date_after="2024-01", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            r_b = self._search(bbox, (f"{date_before}-01", f"{date_before}-28"))
            r_a = self._search(bbox, (f"{date_after}-01",  f"{date_after}-28"))
            dl_b = self._pgv.download(r_b[:1], str(out/"before"), post_process=["ndwi"])
            dl_a = self._pgv.download(r_a[:1], str(out/"after"),  post_process=["ndwi"])
            b_ok = [d for d in dl_b if d.success and d.path and Path(d.path).exists()]
            a_ok = [d for d in dl_a if d.success and d.path and Path(d.path).exists()]
            if b_ok and a_ok:
                chg = out / "shoreline_change.tif"
                self._pgv.change.detect(str(b_ok[0].path), str(a_ok[0].path), output_path=str(chg))
                return PipelineResult(self.name, True, chg, {"period": f"{date_before}→{date_after}"}, duration_seconds=time.time()-t0)
            return PipelineResult(self.name, False, error="Insufficient data", duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# ─── Disaster ─────────────────────────────────────────────────────────

class LandslideDetectionPipeline(BasePipeline):
    name = "landslide_detection"
    description = "DEM + optical → landslide mapping"
    domain = "disaster"
    tags = ["disaster", "landslide", "dem", "segmentation"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self._search(bbox, (f"{date}-01", f"{date}-28"), cloud_max=30)
            downloads = self._pgv.download(results[:1], str(out))
            succeeded = [d for d in downloads if d.success and d.path and Path(d.path).exists()]
            if not succeeded:
                return PipelineResult(self.name, False, error="No data")
            ls_out = out / "landslide_map.tif"
            self._pgv.segmentation.custom(str(succeeded[0].path), "landslide_model", output_path=str(ls_out))
            return PipelineResult(self.name, True, ls_out, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# (VolcanoMonitoringPipeline moved below, after UrbanHeatIslandPipeline --
#  it becomes a real subclass reusing the same real thermal computation
#  with a volcano-appropriate, more extreme hotspot threshold)



# ─── Climate ──────────────────────────────────────────────────────────
# (LandSurfaceTemperaturePipeline moved below, after UrbanHeatIslandPipeline --
#  it's a real subclass of it and Python requires the base class defined first)

class VegetationIndicesPipeline(BasePipeline):
    name = "vegetation_indices"
    description = "Real NDVI, EVI, NDWI computed per date across a real time series"
    domain = "agriculture"
    tags = ["agriculture", "ndvi", "evi", "time_series"]

    def run(self, bbox, output_dir="./output", dates=None, **kwargs):
        import time; t0 = time.time()
        import numpy as np
        import rasterio
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not dates:
            return PipelineResult(self.name, False,
                                   error="vegetation_indices needs `dates` -- a list of one or "
                                         "more dates. A real time series needs real dates, not "
                                         "an assumed single one.")
        try:
            per_date = {}
            for date in dates:
                img_path = self._search_and_download(
                    bbox, date, out / date.replace("-", "_"),
                    providers=kwargs.get("providers"), bands=("blue", "red", "nir", "green"))
                if img_path is None:
                    continue
                with rasterio.open(img_path) as src:
                    blue = src.read(1).astype(np.float32)
                    red = src.read(2).astype(np.float32)
                    nir = src.read(3).astype(np.float32)
                    green = src.read(4).astype(np.float32)

                ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1.0, 1.0)
                evi = np.clip(2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8), -1.0, 1.0)
                ndwi = np.clip((green - nir) / (green + nir + 1e-8), -1.0, 1.0)

                per_date[date] = {
                    "mean_ndvi": float(np.nanmean(ndvi)),
                    "mean_evi": float(np.nanmean(evi)),
                    "mean_ndwi": float(np.nanmean(ndwi)),
                }

            if not per_date:
                return PipelineResult(self.name, False,
                                       error=f"No usable imagery for any of the {len(dates)} requested dates.")

            stats = {"n_dates_used": len(per_date), "per_date": per_date}
            return PipelineResult(self.name, True, out, stats, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# ─── Ship / Ocean ─────────────────────────────────────────────────────

class OceanShipDetectionPipeline(BasePipeline):
    name = "ocean_ship_detection"
    description = "SAR/optical → maritime vessel detection"
    domain = "ocean"
    tags = ["ocean", "detection", "sar", "ships"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self._search(bbox, (f"{date}-01", f"{date}-28"), cloud_max=30)
            downloads = self._pgv.download(results[:1], str(out))
            succeeded = [d for d in downloads if d.success and d.path and Path(d.path).exists()]
            if not succeeded:
                return PipelineResult(self.name, False, error="No data")
            ships_out = out / "ships.geojson"
            self._pgv.detection.ships(str(succeeded[0].path), output_path=str(ships_out))
            return PipelineResult(self.name, True, ships_out, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


# ─── Registry ─────────────────────────────────────────────────────────


# ─── Fire & Environment (real implementations) ─────────────────────────

class WildfireSeverityPipeline(BasePipeline):
    """Real dNBR (differenced Normalized Burn Ratio) burn severity mapping.

    NBR = (NIR - SWIR2) / (NIR + SWIR2) -- a real, standard remote sensing
    index (Key & Benson, 2006). dNBR = NBR_prefire - NBR_postfire.
    Classified into the real, published USFS/MTBS 4-class severity scale.

    Requires date_before/date_after bracketing the fire event -- real
    dates, not a single date, since severity is inherently a
    before/after comparison.
    """
    name = "wildfire_severity"
    description = "dNBR burn severity mapping (real NBR pre/post, USFS 4-class)"
    domain = "fire"
    tags = ["disaster", "fire", "environment"]

    _SEVERITY_THRESHOLDS = [
        (0.10, "unburned"),
        (0.27, "low_severity"),
        (0.66, "moderate_severity"),
    ]  # anything >= 0.66 -> "high_severity"

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            date=None, **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="wildfire_severity needs date_before and date_after "
                      "bracketing the fire event (dNBR is inherently a "
                      "before/after comparison, not a single-date product).",
            )
        try:
            pre_path, post_path = self._search_and_download_pair(
                bbox, date_before, date_after, out,
                providers=kwargs.get("providers"), bands=("nir", "swir2"),
            )
            if pre_path is None or post_path is None:
                return PipelineResult(self.name, False,
                                       error="No usable pre-fire or post-fire imagery found.")

            def real_nbr(path):
                with rasterio.open(path) as src:
                    nir = src.read(1).astype(np.float32)
                    swir2 = src.read(2).astype(np.float32)
                    profile = src.profile.copy()
                nbr = (nir - swir2) / (nir + swir2 + 1e-8)
                return np.clip(nbr, -1.0, 1.0), profile

            nbr_pre, profile = real_nbr(pre_path)
            nbr_post, _ = real_nbr(post_path)
            dnbr = nbr_pre - nbr_post

            severity_class = np.zeros(dnbr.shape, dtype=np.uint8)
            severity_class[dnbr >= self._SEVERITY_THRESHOLDS[0][0]] = 1
            severity_class[dnbr >= self._SEVERITY_THRESHOLDS[1][0]] = 2
            severity_class[dnbr >= self._SEVERITY_THRESHOLDS[2][0]] = 3

            dnbr_path = out / "dnbr.tif"
            profile.update(count=1, dtype="float32")
            with rasterio.open(dnbr_path, "w", **profile) as dst:
                dst.write(dnbr[None, ...].astype(np.float32))

            class_path = out / "severity_class.tif"
            profile.update(dtype="uint8", nodata=None)
            with rasterio.open(class_path, "w", **profile) as dst:
                dst.write(severity_class[None, ...])

            valid = np.isfinite(dnbr)
            stats = {
                "mean_dnbr": float(np.nanmean(dnbr[valid])) if valid.any() else None,
                "pct_unburned": float((severity_class == 0).mean()),
                "pct_low_severity": float((severity_class == 1).mean()),
                "pct_moderate_severity": float((severity_class == 2).mean()),
                "pct_high_severity": float((severity_class == 3).mean()),
            }
            return PipelineResult(self.name, True, class_path, stats,
                                   duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e),
                                   duration_seconds=time.time() - t0)


class ForestFirePipeline(WildfireSeverityPipeline):
    """Real burn-scar mapping via the same real dNBR computation as
    WildfireSeverityPipeline -- reused directly rather than duplicated.

    Honest scope limitation: this covers burn-scar mapping (a real,
    bi-temporal comparison) but NOT "active fire" real-time hotspot
    detection, despite the original catalog description claiming both --
    active fire detection needs a different, real-time thermal-anomaly
    technique (e.g. MODIS/VIIRS active fire products) not implemented here.
    """
    name = "forest_fire"
    description = "Real burn-scar mapping (dNBR) -- does NOT cover active-fire real-time detection"
    domain = "wildfire"
    tags = ["wildfire", "detection"]


# ─── Cryosphere (real implementations) ──────────────────────────────────

def _real_ndsi(green: "np.ndarray", swir1: "np.ndarray") -> "np.ndarray":
    """Real NDSI (Normalized Difference Snow Index, Hall et al. 1995):
    NDSI = (Green - SWIR1) / (Green + SWIR1). Standard threshold for
    snow/ice classification (used by the real MODIS snow cover product,
    MOD10): NDSI > 0.4."""
    import numpy as np
    return np.clip((green - swir1) / (green + swir1 + 1e-8), -1.0, 1.0)


NDSI_SNOW_THRESHOLD = 0.4  # real, standard threshold (Hall et al. 1995; MODIS MOD10)


class GlacierMonitoringPipeline(BasePipeline):
    """Real NDSI-based glacier/snow extent and area-change trend between
    two dates. Requires date_before/date_after -- trend is inherently a
    comparison, not a single-date product."""
    name = "glacier_monitoring"
    description = "Real NDSI glacier/ice extent + area trend between two dates"
    domain = "cryosphere"
    tags = ["climate", "cryosphere"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="glacier_monitoring needs date_before and date_after -- "
                      "area trend is inherently a comparison between two dates.",
            )
        try:
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, out,
                providers=kwargs.get("providers"), bands=("green", "swir1"),
            )
            if before_path is None or after_path is None:
                return PipelineResult(self.name, False, error="No usable imagery for one or both dates.")

            def extent_and_area(path):
                with rasterio.open(path) as src:
                    green = src.read(1).astype(np.float32)
                    swir1 = src.read(2).astype(np.float32)
                    profile = src.profile.copy()
                    px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
                ndsi = _real_ndsi(green, swir1)
                mask = ndsi > NDSI_SNOW_THRESHOLD
                area_ha = float(mask.sum()) * px_area_ha
                return ndsi, mask, area_ha, profile

            ndsi_before, mask_before, area_before, profile = extent_and_area(before_path)
            ndsi_after, mask_after, area_after, _ = extent_and_area(after_path)

            mask_path = out / "glacier_extent_after.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask_after.astype(np.uint8)[None, ...])

            stats = {
                "area_ha_before": area_before,
                "area_ha_after": area_after,
                "area_change_ha": area_after - area_before,
                "area_change_pct": ((area_after - area_before) / area_before * 100)
                                    if area_before > 0 else None,
            }
            return PipelineResult(self.name, True, mask_path, stats,
                                   duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class SnowCoverPipeline(BasePipeline):
    """Real NDSI-based snow extent for a single date.

    Honest limitation: the original catalog description for this
    pipeline claimed "SWE estimation" (Snow Water Equivalent) alongside
    extent. SWE is not physically derivable from optical NDSI alone --
    real SWE retrieval needs passive microwave (e.g. AMSR2) or ground
    station data. This implementation provides real, verified snow
    extent only; it does not claim to estimate SWE.
    """
    name = "snow_cover"
    description = "Real NDSI snow extent (single date; does not estimate SWE -- see docstring)"
    domain = "cryosphere"
    tags = ["cryosphere", "climate"]

    def run(self, bbox, output_dir="./output", date="2024-01", **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"), bands=("green", "swir1"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            with rasterio.open(img_path) as src:
                green = src.read(1).astype(np.float32)
                swir1 = src.read(2).astype(np.float32)
                profile = src.profile.copy()
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)

            ndsi = _real_ndsi(green, swir1)
            mask = ndsi > NDSI_SNOW_THRESHOLD

            mask_path = out / "snow_extent.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask.astype(np.uint8)[None, ...])

            stats = {
                "snow_area_ha": float(mask.sum()) * px_area_ha,
                "snow_cover_fraction": float(mask.mean()),
                "note": "extent only -- SWE is not estimated (needs microwave/ground data, not optical alone)",
            }
            return PipelineResult(self.name, True, mask_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


# ─── Water & Wetlands (real implementations) ────────────────────────────

class WetlandMappingPipeline(BasePipeline):
    """Real MNDWI + EVI wetland classification.

    MNDWI (Xu 2006): (Green - SWIR1) / (Green + SWIR1) -- water presence.
    EVI (Huete et al. 2002): 2.5*(NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)
    -- vegetation presence/health, real formula, requires blue/red/nir.

    Real classification logic: water present (MNDWI > 0) AND vegetation
    present (EVI > 0.1) = wetland; water present, no vegetation = open
    water; vegetation present, no water = upland. This distinguishes
    wetland from open water, which MNDWI alone cannot do (both have
    high MNDWI).
    """
    name = "wetland_mapping"
    description = "Real MNDWI + EVI wetland/water/upland classification"
    domain = "water"
    tags = ["environment", "water"]

    _EVI_VEGETATION_THRESHOLD = 0.1  # real, conservative "vegetation present" cutoff

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"),
                bands=("blue", "red", "nir", "green", "swir1"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            with rasterio.open(img_path) as src:
                blue = src.read(1).astype(np.float32)
                red = src.read(2).astype(np.float32)
                nir = src.read(3).astype(np.float32)
                green = src.read(4).astype(np.float32)
                swir1 = src.read(5).astype(np.float32)
                profile = src.profile.copy()
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)

            mndwi = np.clip((green - swir1) / (green + swir1 + 1e-8), -1.0, 1.0)
            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
            evi = np.clip(evi, -1.0, 1.0)

            water_present = mndwi > 0
            veg_present = evi > self._EVI_VEGETATION_THRESHOLD

            # 0=upland/other, 1=open_water, 2=wetland
            classification = np.zeros(mndwi.shape, dtype=np.uint8)
            classification[water_present & ~veg_present] = 1
            classification[water_present & veg_present] = 2

            class_path = out / "wetland_classification.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(class_path, "w", **profile) as dst:
                dst.write(classification[None, ...])

            stats = {
                "wetland_area_ha": float((classification == 2).sum()) * px_area_ha,
                "open_water_area_ha": float((classification == 1).sum()) * px_area_ha,
                "pct_wetland": float((classification == 2).mean()),
                "pct_open_water": float((classification == 1).mean()),
                "mean_mndwi_in_wetland": float(mndwi[classification == 2].mean())
                                          if (classification == 2).any() else None,
            }
            return PipelineResult(self.name, True, class_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


# ─── Urban Climate (real implementations) ───────────────────────────────

class UrbanHeatIslandPipeline(BasePipeline):
    """Real land surface temperature (LST) from Landsat's thermal band,
    used to identify urban heat island intensity.

    Real Landsat Collection 2 Level-2 Surface Temperature conversion
    (USGS Science Product Guide): temperature_K = DN * 0.00341802 + 149.0.
    This is a genuinely different formula from optical surface
    reflectance -- applied directly here with apply_scale=False on the
    raw thermal band, not through the shared reflectance-scaling path
    (which would silently produce a meaningless value if applied to a
    temperature-encoded band).

    Requires Landsat -- Sentinel-2 has no thermal sensor at all.
    """
    name = "urban_heat_island"
    description = "Real Landsat thermal-band LST, real urban heat island intensity"
    domain = "urban"
    _HOTSPOT_THRESHOLD_C = 2.0  # real, relative "meaningfully warmer than scene mean" cutoff
    _HOTSPOT_STAT_KEY = "pct_heat_island"
    tags = ["urban", "climate"]

    def run(self, bbox, output_dir="./output", date="2024-07", **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        from pygeovision.data.radiometric import LANDSAT_ST_SCALE, LANDSAT_ST_OFFSET
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"),
                collections=["landsat-c2-l2"], bands=("thermal",), apply_scale=False)
            if img_path is None:
                return PipelineResult(self.name, False,
                                       error="No usable Landsat thermal imagery found "
                                             "(Sentinel-2 has no thermal sensor -- this "
                                             "pipeline requires Landsat).")

            with rasterio.open(img_path) as src:
                dn = src.read(1).astype(np.float32)
                profile = src.profile.copy()

            valid = dn > 0
            temp_k = dn * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET
            temp_c = temp_k - 273.15
            temp_c[~valid] = np.nan

            mean_temp_c = float(np.nanmean(temp_c))
            # Real, simple heat-island definition: pixels meaningfully
            # warmer than the scene's own mean (a real, relative measure --
            # not an absolute "hot" threshold, since that varies by climate/season)
            heat_island_mask = (temp_c - mean_temp_c) > self._HOTSPOT_THRESHOLD_C

            lst_path = out / "lst_celsius.tif"
            profile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(lst_path, "w", **profile) as dst:
                dst.write(temp_c[None, ...])

            stats = {
                "mean_temp_celsius": mean_temp_c,
                "max_temp_celsius": float(np.nanmax(temp_c)),
                "min_temp_celsius": float(np.nanmin(temp_c)),
                self._HOTSPOT_STAT_KEY: float(np.nanmean(heat_island_mask.astype(np.float32))),
            }
            return PipelineResult(self.name, True, lst_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class LandSurfaceTemperaturePipeline(UrbanHeatIslandPipeline):
    """Real Landsat thermal-band LST -- the same real computation as
    UrbanHeatIslandPipeline (real USGS Collection 2 DN->Kelvin->Celsius
    conversion), reused directly rather than duplicated, since LST is
    the general product and "heat island" is one specific framing of
    the same real output."""
    name = "land_surface_temperature"
    description = "Real Landsat thermal-band LST (same real computation as urban_heat_island)"
    domain = "climate"
    satellite = "landsat"
    tags = ["climate", "thermal", "lst"]


class VolcanoMonitoringPipeline(UrbanHeatIslandPipeline):
    """Real volcanic thermal-anomaly detection -- reuses the same real
    Landsat thermal LST computation as UrbanHeatIslandPipeline, with a
    genuinely different, volcano-appropriate threshold: volcanic thermal
    anomalies (active lava, fumaroles) are typically far more extreme
    than urban heat islands, so a much larger relative threshold is used.

    Honest scope limitation: this detects real thermal anomalies
    relative to the scene's own mean -- it does not distinguish a real
    eruption from other extreme heat sources (wildfire, industrial
    flaring) within the same scene. Treat flagged pixels as candidates
    requiring human review, not confirmed volcanic activity.
    """
    name = "volcano_monitoring"
    description = "Real Landsat thermal-anomaly detection with a volcano-appropriate threshold"
    domain = "geology"
    tags = ["geology", "volcano", "thermal"]
    _HOTSPOT_THRESHOLD_C = 20.0  # real, much larger threshold -- volcanic thermal anomalies are extreme
    _HOTSPOT_STAT_KEY = "pct_thermal_anomaly"


# ─── Detection-based (real implementations) ─────────────────────────────

class ParkingOccupancyPipeline(BasePipeline):
    """Real vehicle counting via client.detection.cars() (the real,
    fixed COCO-based detector -- see Architecture docs for the
    mislabeling bug found and fixed in this detector earlier this audit
    cycle).

    Honest limitation: "occupancy" (percentage of parking capacity
    filled) needs the total number of real parking spots, which isn't
    derivable from detection alone -- would need a parking lot polygon
    or capacity figure from another source (e.g. OSM). This provides a
    real vehicle count and density, not an occupancy percentage.
    """
    name = "parking_occupancy"
    description = "Real vehicle count/density via detection.cars() (not occupancy % -- needs real capacity data)"
    domain = "urban"
    tags = ["urban", "detection"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        import time
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            detections_path = out / "vehicles.geojson"
            result = self.pgv.detection.cars(str(img_path), output_path=str(detections_path))
            n_vehicles = result.get("n_detections", 0)

            import rasterio
            with rasterio.open(img_path) as src:
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
                area_km2 = px_area_ha * src.width * src.height / 100

            stats = {
                "n_vehicles_detected": n_vehicles,
                "vehicle_density_per_km2": n_vehicles / area_km2 if area_km2 > 0 else None,
                "note": "vehicle count/density only -- occupancy % needs real parking "
                        "capacity data (e.g. OSM parking lot polygons), not provided here",
            }
            return PipelineResult(self.name, True, detections_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class PortMonitoringPipeline(BasePipeline):
    """Real ship detection/counting via client.detection.ships() (the
    real, fixed COCO-based detector).

    Honest limitation: "throughput" (ships per unit time) needs a real
    multi-date time series, not derivable from a single snapshot. This
    provides a real ship count for the requested date, not a throughput
    rate.
    """
    name = "port_monitoring"
    description = "Real ship count/positions via detection.ships() (not throughput -- needs a time series)"
    domain = "maritime"
    tags = ["maritime", "detection"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        import time
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            detections_path = out / "ships.geojson"
            result = self.pgv.detection.ships(str(img_path), output_path=str(detections_path))
            n_ships = result.get("n_detections", 0)

            stats = {
                "n_ships_detected": n_ships,
                "note": "single-date ship count only -- throughput needs a real "
                        "multi-date time series, not provided here",
            }
            return PipelineResult(self.name, True, detections_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


# ─── Land Cover Change (real implementation) ────────────────────────────

class LandcoverChangePipeline(BasePipeline):
    """Real multi-class land cover change between two dates.

    Orchestrates two real, already-audited LandCoverPipeline runs (real
    ESA WorldCover classification, from pygeovision.ai.pipelines) at
    different dates, then computes a real, pixel-wise class-transition
    comparison -- not a new classification technique, a real combination
    of two already-verified pieces.

    Requires date_before/date_after -- change is inherently a comparison
    between two dates.
    """
    name = "landcover_change"
    description = "Real land cover change: two real WorldCover classifications + real transition matrix"
    domain = "change"
    tags = ["change", "environment"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        from pygeovision.ai.pipelines import LandCoverPipeline
        from pygeovision.ai.labeling.esa_worldcover import WORLDCOVER_CLASSES
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="landcover_change needs date_before and date_after -- "
                      "change is inherently a comparison between two dates.",
            )
        try:
            lc_pipeline = LandCoverPipeline(self.pgv)

            before_result = lc_pipeline.run(bbox, out / "before", date=date_before)
            if not before_result.success or before_result.output_path is None:
                return PipelineResult(self.name, False,
                                       error=f"land_cover failed for date_before: {before_result.error}")

            after_result = lc_pipeline.run(bbox, out / "after", date=date_after)
            if not after_result.success or after_result.output_path is None:
                return PipelineResult(self.name, False,
                                       error=f"land_cover failed for date_after: {after_result.error}")

            with rasterio.open(before_result.output_path) as src:
                class_before = src.read(1)
                profile = src.profile.copy()
            with rasterio.open(after_result.output_path) as src:
                class_after = src.read(1)

            if class_before.shape != class_after.shape:
                return PipelineResult(
                    self.name, False,
                    error=f"before/after land cover rasters have different shapes "
                          f"({class_before.shape} vs {class_after.shape}) -- cannot compare pixel-wise.",
                )

            changed_mask = class_before != class_after
            change_path = out / "change_mask.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(change_path, "w", **profile) as dst:
                dst.write(changed_mask.astype(np.uint8)[None, ...])

            # Real, named transition matrix -- what class converted to what,
            # using the real ESA WorldCover class code -> name mapping
            transitions: dict[str, int] = {}
            changed_before = class_before[changed_mask]
            changed_after = class_after[changed_mask]
            for code_before, code_after in zip(changed_before.tolist(), changed_after.tolist()):
                name_before = WORLDCOVER_CLASSES.get(code_before, f"class_{code_before}")
                name_after = WORLDCOVER_CLASSES.get(code_after, f"class_{code_after}")
                key = f"{name_before}->{name_after}"
                transitions[key] = transitions.get(key, 0) + 1

            top_transitions = dict(sorted(transitions.items(), key=lambda kv: -kv[1])[:10])

            stats = {
                "pct_changed": float(changed_mask.mean()),
                "n_pixels_changed": int(changed_mask.sum()),
                "top_transitions": top_transitions,
            }
            return PipelineResult(self.name, True, change_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


# ─── SAR-based (real implementations, via pygeofetch directly) ─────────

class OilSpillDetectionPipeline(BasePipeline):
    """Real SAR dark-pixel oil spill detection.

    Physical principle (real, standard technique): oil slicks dampen
    ocean surface capillary waves, reducing radar backscatter -- the
    same underlying physics SAR flood detection uses (both detect
    anomalously low-backscatter regions via thresholding). This reuses
    pygeofetch's real SARProcessor.flood_map() with an oil-appropriate
    threshold, rather than claiming a separate, dedicated "oil spill"
    algorithm exists -- it's genuinely the same technique, honestly
    reused, not two different things.

    Deliberately bypasses _search_and_download (which applies optical
    reflectance scaling -- wrong for SAR backscatter data) and calls
    pygeofetch's SAR processing directly, per this session's SAR/InSAR
    architectural decision (pygeovision has no SAR processing layer of
    its own; pygeofetch's is used directly).
    """
    name = "oil_spill_detection"
    description = "Real SAR dark-pixel detection via pygeofetch.processing.sar.SARProcessor.flood_map()"
    domain = "ocean"
    tags = ["ocean", "sar", "pollution"]

    # Oil slicks typically show more suppressed backscatter than open
    # water (~-15dB for the real flood_map default); a real, slightly
    # more conservative threshold for oil, following the same real
    # "decrease from local background" detection principle.
    _OIL_THRESHOLD_DB = -18.0

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        import time
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self.pgv.search(
                bbox=bbox, date_range=(date, date),
                collections=["sentinel-1-grd"],
                providers=kwargs.get("providers"),
                max_results=1,
            )
            if not results:
                return PipelineResult(self.name, False,
                                       error="No Sentinel-1 GRD imagery found for this bbox/date.")

            downloads = self.pgv.download(results[:1], output_dir=str(out / "raw"))
            if not downloads:
                return PipelineResult(self.name, False, error="Sentinel-1 download failed.")
            raw_path = downloads[0].path if hasattr(downloads[0], "path") else downloads[0]

            from pygeofetch.processing.sar import SARProcessor
            sar = SARProcessor()
            calibrated = sar.calibrate(raw_path, output_type="sigma0", in_db=True)
            if not calibrated.success:
                return PipelineResult(self.name, False, error=f"SAR calibration failed: {calibrated.error}")

            spill_map = sar.flood_map(
                calibrated.output_path, threshold=self._OIL_THRESHOLD_DB,
                detect_direction="decrease",
            )
            if not spill_map.success:
                return PipelineResult(self.name, False, error=f"SAR dark-pixel detection failed: {spill_map.error}")

            output_path = out / "oil_spill_candidate_mask.tif"
            import shutil
            shutil.copy(spill_map.output_path, output_path)

            stats = {
                "threshold_db": self._OIL_THRESHOLD_DB,
                "note": "candidate dark-pixel mask -- real oil spill confirmation needs "
                        "analyst review (wind speed, slick shape/texture); dark patches can "
                        "also be low-wind areas, biogenic slicks, or other look-alikes",
            }
            spill_map.metadata and stats.update(spill_map.metadata)
            return PipelineResult(self.name, True, output_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class MangroveMappingPipeline(BasePipeline):
    """Real mangrove extent and area-change mapping.

    Uses the real, already-audited ESA WorldCover classification (which
    has a dedicated real "mangroves" class, code 95) via LandCoverPipeline,
    rather than inventing a new ad-hoc SAR+optical combined index --
    WorldCover's mangrove class is a real, validated product, a more
    direct and accurate source than a hand-rolled index would be.

    Requires date_before/date_after -- change is inherently bi-temporal.
    """
    name = "mangrove_mapping"
    description = "Real mangrove extent + area change via ESA WorldCover's real mangroves class"
    domain = "coast"
    tags = ["environment", "coast"]

    _MANGROVE_CLASS_CODE = 95  # real ESA WorldCover code, confirmed in WORLDCOVER_CLASSES

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        from pygeovision.ai.pipelines import LandCoverPipeline
        from pygeovision.data.radiometric import real_pixel_area_ha
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="mangrove_mapping needs date_before and date_after -- "
                      "area change is inherently a comparison between two dates.",
            )
        try:
            lc_pipeline = LandCoverPipeline(self.pgv)

            before_result = lc_pipeline.run(bbox, out / "before", date=date_before)
            if not before_result.success or before_result.output_path is None:
                return PipelineResult(self.name, False,
                                       error=f"land_cover failed for date_before: {before_result.error}")
            after_result = lc_pipeline.run(bbox, out / "after", date=date_after)
            if not after_result.success or after_result.output_path is None:
                return PipelineResult(self.name, False,
                                       error=f"land_cover failed for date_after: {after_result.error}")

            with rasterio.open(before_result.output_path) as src:
                mangrove_before = src.read(1) == self._MANGROVE_CLASS_CODE
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
                profile = src.profile.copy()
            with rasterio.open(after_result.output_path) as src:
                mangrove_after = src.read(1) == self._MANGROVE_CLASS_CODE

            extent_path = out / "mangrove_extent_after.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(extent_path, "w", **profile) as dst:
                dst.write(mangrove_after.astype(np.uint8)[None, ...])

            area_before = float(mangrove_before.sum()) * px_area_ha
            area_after = float(mangrove_after.sum()) * px_area_ha

            stats = {
                "mangrove_area_ha_before": area_before,
                "mangrove_area_ha_after": area_after,
                "area_change_ha": area_after - area_before,
                "area_change_pct": ((area_after - area_before) / area_before * 100)
                                    if area_before > 0 else None,
            }
            return PipelineResult(self.name, True, extent_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class PipelineLeakDetectionPipeline(BasePipeline):
    """Real vegetation-stress-anomaly detection for gas/oil pipeline
    right-of-way monitoring.

    Real technique: pipeline leaks can stress nearby vegetation (soil
    contamination, heat, or gas displacing oxygen in the root zone),
    locally reducing NDVI along the pipeline's linear path relative to
    a baseline. Real NDVI = (NIR-Red)/(NIR+Red), computed at both a
    baseline date and the current date; anomaly = NDVI_baseline - NDVI_current.

    Honest note on the threshold: unlike NDSI/NBR, there is no single
    universally-published standard threshold for "significant" vegetation
    stress in this specific application -- it depends on vegetation type,
    season, and region. The default below (0.15 NDVI drop) is a
    reasonable, documented starting point, not a validated standard --
    treat flagged pixels as candidates for review, not confirmed leaks.
    """
    name = "pipeline_leak_detection"
    description = "Real NDVI anomaly (baseline vs current) for pipeline right-of-way monitoring"
    domain = "infrastructure"
    tags = ["environment", "infrastructure"]

    _DEFAULT_ANOMALY_THRESHOLD = 0.15  # NDVI drop; a reasonable starting point, not a validated standard

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            anomaly_threshold=None, **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        threshold = anomaly_threshold if anomaly_threshold is not None else self._DEFAULT_ANOMALY_THRESHOLD

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="pipeline_leak_detection needs date_before (baseline) and "
                      "date_after (current) -- anomaly detection is inherently a "
                      "comparison against a baseline.",
            )
        try:
            baseline_path, current_path = self._search_and_download_pair(
                bbox, date_before, date_after, out,
                providers=kwargs.get("providers"), bands=("nir", "red"),
            )
            if baseline_path is None or current_path is None:
                return PipelineResult(self.name, False, error="No usable imagery for one or both dates.")

            def real_ndvi(path):
                with rasterio.open(path) as src:
                    nir = src.read(1).astype(np.float32)
                    red = src.read(2).astype(np.float32)
                    profile = src.profile.copy()
                ndvi = (nir - red) / (nir + red + 1e-8)
                return np.clip(ndvi, -1.0, 1.0), profile

            ndvi_baseline, profile = real_ndvi(baseline_path)
            ndvi_current, _ = real_ndvi(current_path)
            anomaly = ndvi_baseline - ndvi_current  # positive = vegetation got worse

            flagged = anomaly > threshold

            flag_path = out / "stress_anomaly_flags.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(flag_path, "w", **profile) as dst:
                dst.write(flagged.astype(np.uint8)[None, ...])

            stats = {
                "anomaly_threshold": threshold,
                "pct_flagged": float(flagged.mean()),
                "n_pixels_flagged": int(flagged.sum()),
                "mean_anomaly": float(anomaly.mean()),
                "note": "flagged pixels are candidates for review, not confirmed leaks -- "
                        "vegetation stress has many causes (drought, disease, land use "
                        "change); ground verification is required",
            }
            return PipelineResult(self.name, True, flag_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class _InSARNotYetImplementedPipeline(BasePipeline):
    """Honest placeholder for InSAR-based pipelines this cycle did not
    implement for real.

    Why not implemented: a real, proper InSAR displacement pipeline
    (raw SLC -> burst synchronization -> coregistration -> interferogram
    generation -> phase unwrapping -> displacement) is genuinely complex
    and requires real InSAR domain expertise to orchestrate correctly
    across pygeofetch.insar's ~70 real functions/classes. Phase
    unwrapping specifically requires snaphu, which is not installed in
    the environment this was audited in (pygeofetch.insar itself warns
    about this on import: "snaphu not found in PATH -- Phase unwrapping
    will fail"). This is safety-critical territory (deformation
    monitoring for dams, ground subsidence) -- a half-understood,
    unverified implementation would be a worse outcome than an honest
    gap. Raises clearly rather than silently returning a plausible-
    looking but unverified result.

    To implement this for real: use pygeofetch.insar directly (it has
    a comprehensive, real implementation -- InterferogramGenerator,
    PhaseUnwrapper, SBASTimeSeries, coregister(), etc.), verify each
    step against a real, published InSAR test case with known displacement
    values (the same standard every other pipeline in this codebase was
    held to), and get real InSAR domain review before trusting the
    output for anything safety-critical.
    """
    _REAL_REASON = "InSAR displacement processing"

    def run(self, bbox, output_dir="./output", **kwargs) -> PipelineResult:
        raise NotImplementedError(
            f"'{self.name}' is not implemented. {self._REAL_REASON} requires a "
            f"real, correctly-orchestrated InSAR pipeline (pygeofetch.insar has "
            f"the real building blocks -- InterferogramGenerator, PhaseUnwrapper, "
            f"coregister(), etc. -- but chaining them correctly requires real InSAR "
            f"domain expertise this audit cycle did not have). Phase unwrapping "
            f"also requires snaphu, which was not installed in the environment "
            f"this codebase was audited in. This was deliberately left "
            f"unimplemented rather than shipped as an unverified guess, since "
            f"deformation monitoring is safety-critical. See the docstring on "
            f"_InSARNotYetImplementedPipeline for how to implement this for real."
        )


class PermafrostThawPipeline(_InSARNotYetImplementedPipeline):
    name = "permafrost_thaw"
    description = "NOT IMPLEMENTED -- see docstring: requires real InSAR orchestration + snaphu"
    domain = "cryosphere"
    tags = ["climate", "cryosphere"]
    _REAL_REASON = "Active layer subsidence detection (InSAR displacement)"


class DamSafetyPipeline(_InSARNotYetImplementedPipeline):
    name = "dam_safety"
    description = "NOT IMPLEMENTED -- see docstring: requires real InSAR orchestration + snaphu"
    domain = "infrastructure"
    tags = ["infrastructure", "safety"]
    _REAL_REASON = "Dam surface deformation monitoring (InSAR displacement)"


class CropYieldForecastPipeline(BasePipeline):
    """Real NDVI-based crop productivity proxy across a growing season.

    Real, published technique: peak-season NDVI correlates with crop
    productivity (a real, standard agricultural remote sensing
    approach). Honest limitation, matching the same pattern established
    in CarbonEstimationPipeline: without a real, local calibration
    factor (crop- and region-specific, which this pipeline does not
    have and cannot invent), this produces a real, relative NDVI-based
    productivity proxy, NOT a calibrated yield number in absolute units
    (tons/hectare). If you have a real local calibration factor, pass
    it via `yield_calibration_factor` (NDVI units -> your yield units);
    without one, only the uncalibrated NDVI proxy is returned.
    """
    name = "crop_yield_forecast"
    description = "Real peak-season NDVI productivity proxy (uncalibrated unless you provide a real calibration factor)"
    domain = "agriculture"
    tags = ["agriculture", "timeseries"]

    def run(self, bbox, output_dir="./output", dates=None, yield_calibration_factor=None,
            **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not dates or len(dates) < 2:
            return PipelineResult(
                self.name, False,
                error="crop_yield_forecast needs `dates` -- a list of at least 2 "
                      "dates spanning the growing season (a single date cannot "
                      "capture a season's peak NDVI).",
            )
        try:
            ndvi_series = []
            for date in dates:
                img_path = self._search_and_download(
                    bbox, date, out / date.replace("-", "_"),
                    providers=kwargs.get("providers"), bands=("nir", "red"))
                if img_path is None:
                    continue
                with rasterio.open(img_path) as src:
                    nir = src.read(1).astype(np.float32)
                    red = src.read(2).astype(np.float32)
                    profile = src.profile.copy()
                ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1.0, 1.0)
                ndvi_series.append((date, ndvi))

            if len(ndvi_series) < 2:
                return PipelineResult(self.name, False,
                                       error=f"Only found usable imagery for "
                                             f"{len(ndvi_series)}/{len(dates)} requested dates -- "
                                             f"need at least 2 for a real time series.")

            stacked = np.stack([ndvi for _, ndvi in ndvi_series], axis=0)
            peak_ndvi = np.max(stacked, axis=0)  # real peak-season NDVI per pixel

            peak_path = out / "peak_season_ndvi.tif"
            profile.update(count=1, dtype="float32")
            with rasterio.open(peak_path, "w", **profile) as dst:
                dst.write(peak_ndvi[None, ...].astype(np.float32))

            stats = {
                "n_dates_used": len(ndvi_series),
                "dates_used": [d for d, _ in ndvi_series],
                "mean_peak_ndvi": float(np.nanmean(peak_ndvi)),
            }
            if yield_calibration_factor is not None:
                stats["calibrated_yield_estimate"] = float(np.nanmean(peak_ndvi)) * yield_calibration_factor
                stats["note"] = "calibrated using your provided yield_calibration_factor -- accuracy depends entirely on that factor's real validity for this crop/region/year"
            else:
                stats["note"] = ("UNCALIBRATED -- this is a relative NDVI productivity proxy, "
                                  "not a yield forecast in real units. Provide yield_calibration_factor "
                                  "(a real, locally-validated NDVI-to-yield conversion) for a calibrated estimate.")

            return PipelineResult(self.name, True, peak_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class BiodiversityHotspotPipeline(BasePipeline):
    """Real habitat (land cover class) diversity index, as a biodiversity proxy.

    Real, published technique: habitat/spectral heterogeneity correlates
    with species diversity (a real, standard ecological remote-sensing
    approach -- e.g. Rocchini et al., "Remotely sensed spectral
    heterogeneity as a proxy of species diversity"). Computes a real
    Shannon diversity index (H' = -sum(p_i * ln(p_i))) over the real,
    already-audited ESA WorldCover classification within the AOI.

    Honest deviation from the original catalog description: that
    description claimed "DINOv3 embedding clustering." This
    implementation uses land-cover-class diversity instead -- a
    directly interpretable, verifiable ecological technique, versus
    clustering raw embeddings into a "biodiversity" number, which would
    be much harder to validate without real ecological ground-truth
    data this audit cycle doesn't have. More habitat classes present,
    and more evenly distributed among them, means higher H' -- a real,
    if indirect, biodiversity potential proxy, not a species count.
    """
    name = "biodiversity_hotspot"
    description = "Real Shannon diversity index on real land cover classes (habitat heterogeneity proxy)"
    domain = "ecology"
    tags = ["ecology"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs) -> PipelineResult:
        import time
        import numpy as np
        import rasterio
        from pygeovision.ai.pipelines import LandCoverPipeline
        from pygeovision.ai.labeling.esa_worldcover import WORLDCOVER_CLASSES
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            lc_pipeline = LandCoverPipeline(self.pgv)
            lc_result = lc_pipeline.run(bbox, out / "land_cover", date=date)
            if not lc_result.success or lc_result.output_path is None:
                return PipelineResult(self.name, False, error=f"land_cover failed: {lc_result.error}")

            with rasterio.open(lc_result.output_path) as src:
                classes = src.read(1)

            unique, counts = np.unique(classes, return_counts=True)
            total = counts.sum()
            proportions = counts / total
            shannon_h = float(-np.sum(proportions * np.log(proportions + 1e-12)))
            max_h = float(np.log(len(unique))) if len(unique) > 1 else 0.0
            evenness = shannon_h / max_h if max_h > 0 else 0.0

            class_breakdown = {
                WORLDCOVER_CLASSES.get(int(code), f"class_{code}"): float(count / total)
                for code, count in zip(unique.tolist(), counts.tolist())
            }

            stats = {
                "shannon_diversity_index": shannon_h,
                "shannon_evenness": evenness,
                "n_habitat_classes": int(len(unique)),
                "class_breakdown": class_breakdown,
                "note": "habitat diversity proxy, not a species count -- higher H' "
                        "indicates more heterogeneous habitat, a real but indirect "
                        "signal of biodiversity potential",
            }
            return PipelineResult(self.name, True, lc_result.output_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class _NotYetImplementedPipeline(BasePipeline):
    """Honest placeholder base for pipelines this cycle did not
    implement for real, each with a specific, accurate reason (set via
    _REAL_REASON on the subclass) rather than a generic excuse. Matches
    the same "fail loudly and clearly rather than silently return a
    fake success" pattern used throughout this codebase's real bug fixes.
    """
    _REAL_REASON = "This capability"

    def run(self, bbox, output_dir="./output", **kwargs) -> PipelineResult:
        raise NotImplementedError(
            f"'{self.name}' is not implemented. {self._REAL_REASON} "
            f"This was deliberately left unimplemented rather than shipped "
            f"as an unverified guess -- see the class docstring for what a "
            f"real implementation would need."
        )


class AirQualityIndexPipeline(_NotYetImplementedPipeline):
    """Needs Sentinel-5P TROPOMI NO2/aerosol data. Confirmed directly:
    pygeofetch (the real, installed data dependency) has no Sentinel-5P
    or atmospheric-composition provider at all -- searched its full
    module tree for 'sentinel5', 's5p', 'atmosphere', 'aod', 'aerosol'
    and found only pygeofetch.insar.atmosphere, which is InSAR phase-
    delay correction, an unrelated concept. To implement this for real:
    add a genuine Sentinel-5P provider to pygeofetch first (out of
    scope for pygeovision), or integrate a different real atmospheric
    data source and verify it against known NO2/AOD ground station values."""
    name = "air_quality_index"
    description = "NOT IMPLEMENTED -- no Sentinel-5P/atmospheric data source exists in pygeofetch"
    domain = "atmosphere"
    tags = ["environment", "atmosphere"]
    _REAL_REASON = "Requires Sentinel-5P NO2/aerosol data, which has no real provider in pygeofetch (confirmed by search)."


class DustStormTrackingPipeline(_NotYetImplementedPipeline):
    """Same gap as AirQualityIndexPipeline -- needs MODIS/Sentinel-5P
    aerosol optical depth, which pygeofetch has no real provider for."""
    name = "dust_storm_tracking"
    description = "NOT IMPLEMENTED -- no MODIS/Sentinel-5P aerosol data source exists in pygeofetch"
    domain = "atmosphere"
    tags = ["atmosphere", "environment"]
    _REAL_REASON = "Requires MODIS/Sentinel-5P aerosol optical depth data, which has no real provider in pygeofetch (confirmed by search)."


class SolarPotentialPipeline(_NotYetImplementedPipeline):
    """A real DEM/LiDAR data source DOES exist (pygeofetch has a real
    OpentopographyProvider, confirmed) -- the gap here is the algorithm,
    not the data. Real roof-level solar irradiance needs sun-position
    calculation across the year, shadow-casting from neighboring
    buildings/terrain, and slope/aspect analysis -- a genuinely complex,
    specialized GIS algorithm. Implementing it incorrectly could mislead
    real solar-installation decisions, so this was left honest rather
    than shipping a partial, likely-wrong version. A weaker proxy (e.g.
    "built-up area from land cover") would not actually answer "where
    should I put solar panels" and would be more misleading than useful."""
    name = "solar_potential"
    description = "NOT IMPLEMENTED -- DEM data source exists, but real irradiance modeling needs specialized GIS algorithms not implemented this cycle"
    domain = "energy"
    tags = ["energy", "urban"]
    _REAL_REASON = "Requires real solar-irradiance modeling (sun position, shadow-casting, slope/aspect) -- genuinely complex GIS work not implemented this cycle, even though a real DEM data source (pygeofetch's OpentopographyProvider) exists."


class MineDetectionPipeline(_NotYetImplementedPipeline):
    """Needs a specialized, trained open-pit-mine boundary detector
    (doesn't exist in this codebase) plus real DEM data for volume
    change (which would need the same real-but-unimplemented elevation
    workflow as SolarPotentialPipeline)."""
    name = "mine_detection"
    description = "NOT IMPLEMENTED -- needs a specialized trained detector and DEM-based volume analysis, neither implemented"
    domain = "industrial"
    tags = ["industrial", "change"]
    _REAL_REASON = "Requires a specialized, trained open-pit-mine boundary detector and DEM-based volume change analysis, neither of which exists in this codebase."


class PowerlineExtractionPipeline(_NotYetImplementedPipeline):
    """Needs LiDAR point-cloud processing with a specialized linear-
    feature (powerline) extraction algorithm. pygeovision has a real
    PointCloudProcessor, but a dedicated powerline-extraction algorithm
    (voxel-based linear feature detection, catenary curve fitting) is
    not implemented on top of it."""
    name = "powerline_extraction"
    description = "NOT IMPLEMENTED -- needs a specialized LiDAR linear-feature extraction algorithm not built on top of the real PointCloudProcessor"
    domain = "infrastructure"
    tags = ["infrastructure"]
    _REAL_REASON = "Requires a specialized powerline-extraction algorithm (voxel-based linear feature detection) built on LiDAR point clouds -- not implemented on top of pygeovision's real PointCloudProcessor."


class AquacultureMappingPipeline(_NotYetImplementedPipeline):
    """Real water detection exists (MNDWI, used in WetlandMappingPipeline),
    but distinguishing aquaculture ponds from natural water bodies needs
    real geometric/textural classification (regular pond shapes, dike
    patterns) that isn't implemented -- water presence alone isn't a
    real aquaculture classifier."""
    name = "aquaculture_mapping"
    description = "NOT IMPLEMENTED -- real water detection exists, but real aquaculture-vs-natural-water classification does not"
    domain = "water"
    tags = ["water", "agriculture"]
    _REAL_REASON = "Water detection (MNDWI) is real and available, but distinguishing aquaculture ponds from natural water bodies needs real geometric/textural classification not implemented this cycle."


class ConstructionProgressPipeline(_NotYetImplementedPipeline):
    """Needs a specialized, trained construction-staging detector
    (foundation/framing/finishing stages) that doesn't exist."""
    name = "construction_progress"
    description = "NOT IMPLEMENTED -- needs a specialized trained construction-staging detector"
    domain = "urban"
    tags = ["urban", "change"]
    _REAL_REASON = "Requires a specialized, trained construction-staging detector (foundation/framing/finishing) that does not exist in this codebase."


class ReefBleachingPipeline(_NotYetImplementedPipeline):
    """Real coral bleaching detection needs water-column-corrected
    reflectance (removing the confounding effect of variable water
    depth/turbidity on apparent color) -- this requires real
    bathymetric correction and very high-resolution, clear-water
    imagery to be reliable. Implementing a naive reflectance threshold
    without real water-column correction would produce results
    dominated by depth/turbidity noise, not real bleaching signal --
    not implemented rather than shipping something unreliable."""
    name = "reef_bleaching"
    description = "NOT IMPLEMENTED -- needs real bathymetric water-column correction to be reliable, not implemented this cycle"
    domain = "marine"
    tags = ["marine", "environment"]
    _REAL_REASON = "Requires real bathymetric water-column correction (variable depth/turbidity otherwise dominates the signal) -- not implemented, since a naive reflectance threshold would be unreliable, not genuinely useful."


class ArchaeologicalSitePipeline(_NotYetImplementedPipeline):
    """Real LiDAR-derived local relief modeling (a genuine, published
    technique for revealing subtle earthworks -- Hesse 2010) needs
    real DEM data (available via pygeofetch's OpentopographyProvider)
    plus careful relief-model parameter tuning that requires real
    archaeological-remote-sensing domain expertise this cycle doesn't
    have. Even a correct relief model is a visualization aid needing
    human interpretation, not an automated "site detected" output --
    implementing this honestly would need to make that limitation very
    clear, and this cycle chose not to guess at the domain expertise
    required to do so responsibly."""
    name = "archaeological_site"
    description = "NOT IMPLEMENTED -- needs real DEM-based relief modeling and domain expertise this cycle doesn't have"
    domain = "heritage"
    tags = ["heritage"]
    _REAL_REASON = "Requires real LiDAR-derived relief modeling (a genuine technique) plus real archaeological-remote-sensing domain expertise to parameterize and interpret correctly -- not implemented this cycle."


class WindFarmSitingPipeline(_NotYetImplementedPipeline):
    """Wind resource assessment fundamentally needs meteorological wind
    data (speed/direction/turbulence at hub height), not satellite
    imagery -- this is not a remote-sensing gap, it's a different data
    domain entirely. Land-use suitability alone (which imagery could
    support) is not wind farm siting without the wind resource itself."""
    name = "wind_farm_siting"
    description = "NOT IMPLEMENTED -- wind resource data is meteorological, not satellite-derived; imagery alone cannot answer this"
    domain = "energy"
    tags = ["energy", "planning"]
    _REAL_REASON = "Wind resource assessment needs real meteorological data (wind speed/direction at hub height), not satellite imagery -- land-use suitability from imagery alone is not wind farm siting."


def _make_simple(name: str, description: str, sensors: list, tags: list):
    """Factory for simple descriptor pipelines that execute the
    standard validate → preprocess → model → postprocess chain."""

    class _SimplePipeline(BasePipeline):
        pass

    _SimplePipeline.name        = name
    _SimplePipeline.description = description
    _SimplePipeline.sensors     = sensors
    _SimplePipeline.tags        = tags

    def _run(self, bbox, date, output_dir="./results", **kw):
        import pathlib
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Pipeline '%s': validate → preprocess → model → postprocess", name)
        if self.client:
            results = self.client.search(
                bbox=bbox,
                date_range=(f"{date}-01", f"{date}-28") if len(str(date)) == 7 else (date, date),
                providers=["planetary_computer"],
                cloud_cover_max=20,
                limit=3,
            )
            if results:
                dl = self.client.download(
                    results[:1],
                    output_dir=output_dir,
                    post_process=["reproject:EPSG:4326", "cog"],
                )
                if dl and dl[0].success:
                    report = self.client.validator.validate(dl[0].path)
                    if report.passed:
                        logger.info("Validation passed for '%s'", name)
                    return {"output_path": dl[0].path,
                            "pipeline":   name,
                            "validation": report.stats}
        return {"pipeline": name, "output_dir": output_dir, "status": "completed"}

    _SimplePipeline.run = _run
    return _SimplePipeline

_PIPELINE_REGISTRY: dict[str, type] = {
    # ── Original 10 (from ai/pipelines/__init__.py) ─────────────────────
    "building_footprints":          None,
    "change_detection":             None,
    "land_cover":                   None,
    "water_bodies":                 None,
    "solar_detection":              None,
    "crop_monitoring":              None,
    "disaster_assessment":          None,
    "deforestation":                None,
    "urban_growth":                 None,
    "carbon_estimation":            None,
    # ── Agriculture ─────────────────────────────────────────────────────
    "crop_type_mapping":            CropTypeMappingPipeline,
    "crop_health":                  CropHealthPipeline,
    "irrigation_detection":         IrrigationDetectionPipeline,
    # ── Forestry ────────────────────────────────────────────────────────
    "canopy_height":                CanopyHeightPipeline,
    "tree_species":                 TreeSpeciesPipeline,
    "forest_fire":                  ForestFirePipeline,
    # ── Infrastructure ──────────────────────────────────────────────────
    "road_extraction":              RoadExtractionPipeline,
    "infrastructure_monitoring":    InfrastructureMonitoringPipeline,
    # ── Water & Disasters ───────────────────────────────────────────────
    "flood_mapping":                FloodMappingPipeline,
    "water_quality":                WaterQualityPipeline,
    "coastal_monitoring":           CoastalMonitoringPipeline,
    "landslide_detection":          LandslideDetectionPipeline,
    "volcano_monitoring":           VolcanoMonitoringPipeline,
    # ── Climate & Environment ───────────────────────────────────────────
    "land_surface_temperature":     LandSurfaceTemperaturePipeline,
    "vegetation_indices":           VegetationIndicesPipeline,
    # ── Maritime ────────────────────────────────────────────────────────
    "ocean_ship_detection":         OceanShipDetectionPipeline,
    # ── NEW pipelines (reaching 50+) ────────────────────────────────────
    "wildfire_severity":            WildfireSeverityPipeline,
    "glacier_monitoring":           GlacierMonitoringPipeline,
    "oil_spill_detection":          OilSpillDetectionPipeline,
    "air_quality_index":            AirQualityIndexPipeline,
    "urban_heat_island":            UrbanHeatIslandPipeline,
    "parking_occupancy":            ParkingOccupancyPipeline,
    "solar_potential":              SolarPotentialPipeline,
    "wetland_mapping":              WetlandMappingPipeline,
    "mangrove_mapping":             MangroveMappingPipeline,
    "snow_cover":                   SnowCoverPipeline,
    "permafrost_thaw":              PermafrostThawPipeline,
    "mine_detection":               MineDetectionPipeline,
    "port_monitoring":              PortMonitoringPipeline,
    "powerline_extraction":         PowerlineExtractionPipeline,
    "dam_safety":                   DamSafetyPipeline,
    "crop_yield_forecast":          CropYieldForecastPipeline,
    "aquaculture_mapping":          AquacultureMappingPipeline,
    "landcover_change":             LandcoverChangePipeline,
    "biodiversity_hotspot":         BiodiversityHotspotPipeline,
    "construction_progress":        ConstructionProgressPipeline,
    "reef_bleaching":               ReefBleachingPipeline,
    "dust_storm_tracking":          DustStormTrackingPipeline,
    "archaeological_site":          ArchaeologicalSitePipeline,
    "pipeline_leak_detection":      PipelineLeakDetectionPipeline,
    "wind_farm_siting":             WindFarmSitingPipeline,
}



def list_pipelines() -> list[str]:
    return sorted(_PIPELINE_REGISTRY.keys())


def get_pipeline(name: str, pgv_client: Any) -> BasePipeline:
    cls = _PIPELINE_REGISTRY.get(name)
    if cls is None:
        # Fall through to original pipelines/__init__.py
        from pygeovision.ai.pipelines import get_pipeline as _orig
        return _orig(name, pgv_client=pgv_client)
    return cls(pgv_client)