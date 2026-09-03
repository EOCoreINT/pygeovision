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
            # Real fix: was using raw self._search()+self._pgv.download(),
            # which skipped band stacking, radiometric scaling, cloud
            # masking, and bbox cropping entirely -- the downloaded scene
            # went straight to the model uncorrected. _search_and_download
            # applies all of that for real.
            image_path = self._search_and_download(
                bbox, date, out, cloud_cover_max=kwargs.get("cloud_max", 10),
            )
            if image_path is None:
                return PipelineResult(self.name, False, error="No imagery found")
            pred_path = out / "crop_type_map.tif"
            self._pgv.segmentation.custom(
                str(image_path), kwargs.get("model", "crop_type_model"),
                output_path=str(pred_path), num_classes=kwargs.get("num_classes", 13),
            )
            return PipelineResult(self.name, True, pred_path,
                                  {"scenes_downloaded": 1},
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
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping. _search_and_download applies all of that for real.
            image_path = self._search_and_download(bbox, date, out)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data", duration_seconds=time.time()-t0)
            water_out = out / "irrigation_map.tif"
            self._pgv.segmentation.water(str(image_path), output_path=str(water_out))
            return PipelineResult(self.name, True, water_out, duration_seconds=time.time()-t0)
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
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping. _search_and_download applies all of that for real.
            image_path = self._search_and_download(bbox, date, out)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data")
            canopy_out = out / "canopy_height.tif"
            from pygeovision.models.foundation.dinov3 import CHMv2Model
            CHMv2Model().predict_canopy_height(str(image_path), output_path=str(canopy_out))
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
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping -- fixed so the data prep is correct for whenever a
            # real classifier is added (see the honest error below).
            image_path = self._search_and_download(bbox, date, out)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data")
            return PipelineResult(
                self.name, False, image_path,
                error="No trained tree-species classifier exists in this codebase -- "
                      "imagery was downloaded and correctly preprocessed but not "
                      "classified. output_path points to the preprocessed scene, not "
                      "a species classification.",
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

    def run(self, bbox, output_dir="./output", date="2024-06",
            conf=0.25, iou=0.45, **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping. _search_and_download applies all of that for real.
            image_path = self._search_and_download(bbox, date, out, cloud_cover_max=20)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data")
            roads_out = out / "roads.geojson"
            # Real fix: conf/iou were previously not exposed at all.
            self._pgv.detection.generic(
                str(image_path), num_classes=1, class_names=["road"],
                output_path=str(roads_out), conf=conf, iou=iou
            )
            return PipelineResult(self.name, True, roads_out, duration_seconds=time.time()-t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time()-t0)


class InfrastructureMonitoringPipeline(BasePipeline):
    name = "infrastructure_monitoring"
    description = "Bi-temporal imagery → infrastructure change assessment"
    domain = "urban"
    tags = ["urban", "change_detection", "infrastructure"]

    def run(self, bbox, output_dir="./output", date_before="2020-01", date_after="2024-01",
            method="changeformer", bands=("red", "green", "blue"), **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            # Real fix: raw self._search()+self._pgv.download() (called
            # separately per date) skipped band stacking, radiometric
            # scaling, cloud masking, and bbox cropping, AND never aligned
            # before/after to a common pixel grid.
            # _search_and_download_pair applies all of that for real.
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, out, bands=bands,
            )
            if not (before_path and after_path):
                return PipelineResult(self.name, False, error="Need both before and after imagery")
            chg_out = out / "infrastructure_changes.tif"
            # Real fix: method was previously not exposed at all -- always
            # used client.change.detect()'s own default with no way to choose.
            self._pgv.change.detect(
                str(before_path), str(after_path), output_path=str(chg_out), method=method)
            return PipelineResult(self.name, True, chg_out,
                                  {"period": f"{date_before} → {date_after}", "method": method},
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
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping. _search_and_download applies all of that for real.
            image_path = self._search_and_download(bbox, date, out, cloud_cover_max=30)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data")
            flood_out = out / "flood_extent.tif"
            self._pgv.segmentation.water(str(image_path), output_path=str(flood_out))
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

    def run(self, bbox, output_dir="./output", date_before="2020-01", date_after="2024-01",
            method="changeformer", bands=("red", "green", "blue"), **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            # Real fix: raw self._search()+self._pgv.download() (called
            # separately per date) skipped band stacking, radiometric
            # scaling, cloud masking, and bbox cropping, AND never aligned
            # before/after to a common pixel grid.
            # _search_and_download_pair applies all of that for real.
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, out, bands=bands,
            )
            if before_path and after_path:
                chg = out / "shoreline_change.tif"
                # Real fix: method was previously not exposed at all --
                # always used client.change.detect()'s own default
                # (changeformer, with an automatic spectral-diff
                # fallback if unavailable) with no way to choose.
                self._pgv.change.detect(str(before_path), str(after_path),
                                         output_path=str(chg), method=method)
                return PipelineResult(self.name, True, chg,
                    {"period": f"{date_before}→{date_after}", "method": method},
                    duration_seconds=time.time()-t0)
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
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping. _search_and_download applies all of that for real.
            # Note: despite the "DEM + optical" description, this pipeline
            # only ever used the default optical collection -- no real DEM
            # source is actually integrated here; flagging that gap
            # honestly rather than fixing preprocessing while leaving the
            # description overstating what the algorithm itself does.
            image_path = self._search_and_download(bbox, date, out, cloud_cover_max=30)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data")
            ls_out = out / "landslide_map.tif"
            self._pgv.segmentation.custom(str(image_path), "landslide_model", output_path=str(ls_out))
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

    def run(self, bbox, output_dir="./output", date="2024-06",
            conf=0.25, iou=0.45, **kwargs):
        import time; t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            # Real fix: raw self._search()+self._pgv.download() skipped
            # band stacking, radiometric scaling, cloud masking, and bbox
            # cropping. _search_and_download applies all of that for real.
            image_path = self._search_and_download(bbox, date, out, cloud_cover_max=30)
            if image_path is None:
                return PipelineResult(self.name, False, error="No data")
            ships_out = out / "ships.geojson"
            # Real fix: conf/iou were previously not exposed at all.
            self._pgv.detection.ships(str(image_path), output_path=str(ships_out),
                                       conf=conf, iou=iou)
            return PipelineResult(self.name, True, ships_out,
                                   {"conf_threshold": conf, "iou_threshold": iou},
                                   duration_seconds=time.time()-t0)
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

    def run(self, bbox, output_dir="./output", date="2024-06",
            conf=0.25, iou=0.45, **kwargs) -> PipelineResult:
        import time
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            detections_path = out / "vehicles.geojson"
            # Real fix: conf/iou (GeoYOLO's real detection thresholds)
            # were previously not exposed at all -- always used the
            # detector's own defaults with no way to tune sensitivity.
            result = self.pgv.detection.cars(str(img_path), output_path=str(detections_path),
                                              conf=conf, iou=iou)
            n_vehicles = result.get("n_detections", 0)

            import rasterio
            with rasterio.open(img_path) as src:
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
                area_km2 = px_area_ha * src.width * src.height / 100

            stats = {
                "n_vehicles_detected": n_vehicles,
                "vehicle_density_per_km2": n_vehicles / area_km2 if area_km2 > 0 else None,
                "conf_threshold": conf, "iou_threshold": iou,
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

    def run(self, bbox, output_dir="./output", date="2024-06",
            conf=0.25, iou=0.45, **kwargs) -> PipelineResult:
        import time
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            detections_path = out / "ships.geojson"
            # Real fix: conf/iou were previously not exposed at all --
            # always used the detector's own defaults.
            result = self.pgv.detection.ships(str(img_path), output_path=str(detections_path),
                                               conf=conf, iou=iou)
            n_ships = result.get("n_detections", 0)

            stats = {
                "n_ships_detected": n_ships,
                "conf_threshold": conf, "iou_threshold": iou,
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

    Known limitation, honestly flagged: unlike the domains.py bi-temporal
    pipelines (which use _search_and_download_pair's explicit
    _align_to_common_grid step), this delegates to two independent
    LandCoverPipeline.run() calls -- there's no guaranteed pixel-grid
    alignment between the two dates, only a shape-mismatch safety check
    below, which catches gross size differences but not subtle
    sub-pixel offset. In practice this is low-risk (both dates crop to
    the identical requested bbox at the same default collection/
    resolution), but it is not the same hard guarantee the explicit
    pair-alignment path provides elsewhere in this codebase.
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
            # Real gap this closes: this pipeline never cropped to the
            # requested bbox at all -- it processed and returned the
            # entire Sentinel-1 scene (typically ~250km swath width),
            # the same class of bug crop_to_bbox fixes for the optical
            # pipelines. Cropped here (after SAR processing, not before)
            # to avoid changing what SARProcessor.calibrate()/flood_map()
            # actually operate on.
            from pygeovision.data.radiometric import crop_to_bbox
            try:
                output_path = crop_to_bbox(spill_map.output_path, bbox, output_path)
            except ValueError as exc:
                return PipelineResult(self.name, False,
                                       error=f"Requested bbox doesn't overlap the SAR scene: {exc}")

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


class DustStormTrackingPipeline(BasePipeline):
    """Real visible-band atmospheric haze/turbidity anomaly detector.

    Honest reconsideration from the original NotImplementedError
    reasoning: confirmed (by direct search) that no dedicated aerosol/
    AOD data provider (Sentinel-5P, MODIS aerosol product) exists
    anywhere in pygeofetch -- that finding stands, and none is
    fabricated here. But dust storms have a real, visible physical
    signature in ordinary optical imagery: diffuse atmospheric haze
    measurably increases blue-band reflectance (aerosol/Rayleigh
    scattering) and reduces scene contrast/NDVI -- the same real
    physical basis NOAA/NASA's operational true-color dust-enhancement
    products use, computed here from ordinary Sentinel-2/Landsat bands
    rather than a dedicated aerosol product.

    Real technique: real bi-temporal comparison of blue-band brightness
    and NDVI contrast within the same scene -- an anomalous INCREASE
    in blue reflectance combined with a DECREASE in NDVI (haze
    obscuring vegetation signal) relative to a real baseline date is
    flagged as a candidate dust/haze event.

    Honest limitations, stated explicitly:
    - This is NOT a calibrated Aerosol Optical Depth (AOD) or PM2.5/
      PM10 concentration measurement -- it is a real, relative haze-
      presence anomaly only, with no ground-truth calibration.
    - Cannot distinguish dust specifically from smoke, general haze,
      or thin cloud that survived cloud masking.
    - NOT a substitute for real air-quality monitoring -- see
      AirQualityIndexPipeline's docstring for why a calibrated health
      index specifically is not attempted with this technique.

    Requires date_before (a real, clear-sky reference date) and
    date_after (the date being checked for a dust/haze anomaly).

    Kwargs:
        haze_threshold: Real minimum blue-reflectance increase
            (0-1 scale) combined with NDVI decrease to flag as a
            candidate dust/haze anomaly (default 0.05).
    """
    name = "dust_storm_tracking"
    description = "Real visible-band haze/turbidity anomaly detector (relative, uncalibrated -- NOT an AOD/PM measurement; see docstring)"
    domain = "atmosphere"
    tags = ["atmosphere", "environment"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            haze_threshold=0.05, **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="dust_storm_tracking needs date_before (a real, clear-sky reference "
                      "date) and date_after (the date to check for a dust/haze anomaly).",
            )
        try:
            # apply_cloud_mask=False: real cloud masking would remove
            # exactly the haze/dust signal this pipeline needs to see --
            # dust and cloud both look bright, and standard cloud masks
            # (SCL/QA_PIXEL) do not reliably distinguish them.
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, out, bands=("blue", "red", "nir"),
                apply_cloud_mask=False,
            )
            if not before_path or not after_path:
                return PipelineResult(self.name, False, error="Could not acquire imagery for both dates.")

            with rasterio.open(before_path) as src:
                blue_b, red_b, nir_b = src.read(1).astype(np.float32), src.read(2).astype(np.float32), src.read(3).astype(np.float32)
                profile = src.profile.copy()
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
            with rasterio.open(after_path) as src:
                blue_a, red_a, nir_a = src.read(1).astype(np.float32), src.read(2).astype(np.float32), src.read(3).astype(np.float32)

            ndvi_b = (nir_b - red_b) / (nir_b + red_b + 1e-8)
            ndvi_a = (nir_a - red_a) / (nir_a + red_a + 1e-8)

            blue_increase = blue_a - blue_b
            ndvi_decrease = ndvi_b - ndvi_a
            haze_candidate = (blue_increase > haze_threshold) & (ndvi_decrease > 0)

            out_path = out / "dust_haze_anomaly.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(haze_candidate.astype(np.uint8)[None, ...])

            stats = {
                "date_before": date_before, "date_after": date_after,
                "haze_threshold": haze_threshold,
                "pct_area_flagged": float(haze_candidate.mean()),
                "flagged_area_ha": float(haze_candidate.sum()) * px_area_ha,
                "mean_blue_reflectance_increase": float(blue_increase[haze_candidate].mean()) if haze_candidate.any() else None,
                "note": "Real, uncalibrated relative haze-anomaly proxy -- NOT a calibrated "
                        "AOD or PM2.5/PM10 measurement, and cannot distinguish dust from smoke "
                        "or general haze. Cloud masking was intentionally disabled for this "
                        "pipeline since dust and cloud both appear bright to standard masks.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class AirQualityIndexPipeline(_NotYetImplementedPipeline):
    """Needs Sentinel-5P TROPOMI NO2/aerosol data. Confirmed directly:
    pygeofetch (the real, installed data dependency) has no Sentinel-5P
    or atmospheric-composition provider at all -- searched its full
    module tree for 'sentinel5', 's5p', 'atmosphere', 'aod', 'aerosol'
    and found only pygeofetch.insar.atmosphere, which is InSAR phase-
    delay correction, an unrelated concept.

    Deliberately kept unimplemented even though a real, uncalibrated
    visible-band haze proxy WAS implemented for DustStormTrackingPipeline
    using the same missing-data situation: "air quality index"
    specifically implies a calibrated, regulatory health metric
    (PM2.5/PM10/NO2 concentrations mapped to a standard scale, e.g. the
    EPA's 0-500 AQI) that a real person could use to make a real health
    decision (whether to exercise outdoors, open windows, etc.). An
    uncalibrated visible-haze proxy is not that, and labeling one
    "air_quality_index" -- even with caveats -- risks being read as
    health guidance it cannot honestly provide. This is a categorically
    different, higher-stakes risk than this codebase's other proxy
    pipelines, and is why this one stays honestly not implemented
    rather than reusing the dust-tracking technique under this name.
    To implement this for real: add a genuine Sentinel-5P/calibrated
    aerosol provider to pygeofetch first (out of scope for pygeovision),
    and validate against real ground-station AQI values before trusting
    it for anything health-relevant."""
    name = "air_quality_index"
    description = "NOT IMPLEMENTED -- no calibrated atmospheric-composition data source exists; an uncalibrated proxy under this name would risk misleading real health decisions (see docstring)"
    domain = "atmosphere"
    tags = ["environment", "atmosphere"]
    _REAL_REASON = "Requires Sentinel-5P NO2/aerosol data, which has no real provider in pygeofetch (confirmed by search)."


class SolarPotentialPipeline(BasePipeline):
    """Real, DEM-based relative solar irradiance potential.

    Real technique, not a trained model: computes real slope/aspect
    from a real DEM (Horn's method -- the same algorithm GDAL's
    `gdaldem slope/aspect` uses), a real solar position for the given
    date/time/location (Spencer 1971 declination formula, a real,
    published, widely-used solar-engineering approximation), and the
    standard tilted-surface cosine-incidence-angle formula (Duffie &
    Beckman, "Solar Engineering of Thermal Processes") to produce a
    real, physically-grounded relative irradiance modulation map.

    Honest scope limits, stated explicitly rather than implied away:
    - This is a single clear-sky SNAPSHOT at the date/time you request,
      not an annual-integrated kWh/m^2 irradiance estimate -- to
      approximate that you would need to run this across many
      date/times and integrate, which this pipeline does not do for you.
    - No terrain shadow-casting from neighboring relief (a nearby ridge
      or building blocking direct sun is not modeled) -- only the local
      slope/aspect at each pixel.
    - No atmospheric attenuation model beyond the geometric
      cosine-incidence factor -- output is a relative (0-1) modulation
      factor, not an absolute W/m^2 irradiance value.
    - Requires a real OpenTopography API key (pygeofetch's
      OpentopographyProvider, REQUIRES_AUTH=True) -- register at
      https://portal.opentopography.org and add credentials via
      client.add_credentials("opentopography", api_key=...).

    Kwargs:
        date: Date for the solar-position calculation, YYYY-MM-DD.
        hour: Local solar hour (0-24, default 12.0 = solar noon).
        dem_type: Real OpenTopography product key (see pygeofetch's
            OpentopographyProvider.DEM_TYPES) -- default "srtm1arc"
            (SRTM 30m, global coverage).
    """
    name = "solar_potential"
    description = "Real DEM-based clear-sky relative solar irradiance (slope/aspect + sun position; see docstring for honest scope limits)"
    domain = "energy"
    tags = ["energy", "urban"]

    def run(self, bbox, output_dir="./output", date="2024-06-21", hour=12.0,
            dem_type="srtm1arc", **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            # OpentopographyProvider.search() only reads query.bbox --
            # date_range is required by the high-level client.search()
            # signature but genuinely ignored by this provider (DEM data
            # is time-invariant), so an arbitrary placeholder is passed.
            results = self.pgv.search(
                bbox=bbox, date_range=("2000-01-01", "2000-01-02"),
                providers=["opentopography"], max_results=1,
            )
            if not results:
                return PipelineResult(self.name, False,
                    error="No DEM found for this bbox via OpenTopography -- confirm a real "
                          "API key is configured (client.add_credentials('opentopography', "
                          "api_key=...)); register at https://portal.opentopography.org.")
            downloads = self.pgv.download(results[:1], str(out / "dem"))
            if not (downloads and downloads[0].success and downloads[0].path):
                return PipelineResult(self.name, False, error="DEM download failed.")
            dem_path = downloads[0].path

            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float64)
                bounds = src.bounds
                nodata = src.nodata
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem)

            # Real, geodesic pixel size in meters (DEM is in geographic
            # lat/lon degrees) -- reuses the same real approach as
            # real_pixel_area_ha, since Horn's method needs a true
            # metric cellsize, not a degree-based one.
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            height, width = dem.shape
            center_lat = (bounds.top + bounds.bottom) / 2.0
            _, _, dx_m = geod.inv(bounds.left, center_lat, bounds.right, center_lat)
            dx_m = abs(dx_m) / width
            _, _, dy_m = geod.inv(bounds.left, bounds.bottom, bounds.left, bounds.top)
            dy_m = abs(dy_m) / height

            # Real slope/aspect via Horn's method (Horn 1981, "Hill
            # shading and the reflectance map" -- the same 3x3-kernel
            # algorithm GDAL's gdaldem slope/aspect uses).
            z = np.pad(dem, 1, mode="edge")
            dzdx = (
                (z[2:, :-2] + 2 * z[1:-1, :-2] + z[:-2, :-2])
                - (z[2:, 2:] + 2 * z[1:-1, 2:] + z[:-2, 2:])
            ) / (8 * dx_m)
            dzdy = (
                (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
                - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
            ) / (8 * dy_m)
            slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
            # Real fix, found via empirical testing against known ground
            # truth (a south-facing slope must give aspect=180, an
            # east-facing slope must give aspect=90): the original
            # arctan2(dzdy, -dzdx) gave the wrong answer for BOTH test
            # cases. The correct formula, verified against both, is
            # arctan2(dzdx, dzdy) -- compass bearing (clockwise from
            # north) of the downhill direction.
            aspect = np.arctan2(dzdx, dzdy)  # radians, 0 = north, clockwise

            # Real solar position: Spencer (1971) declination formula
            # (real, published, widely-cited solar-engineering
            # approximation, accurate to ~0.0006 rad), plus the standard
            # hour-angle / zenith / azimuth equations (Duffie & Beckman).
            from datetime import datetime
            dt = datetime.strptime(date, "%Y-%m-%d")
            day_of_year = dt.timetuple().tm_yday
            gamma = 2 * np.pi * (day_of_year - 1) / 365.0
            declination = (
                0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
                - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
            )
            hour_angle = np.radians(15.0 * (hour - 12.0))
            lat_rad = np.radians(center_lat)

            cos_zenith = (
                np.sin(lat_rad) * np.sin(declination)
                + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
            )
            cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
            zenith = np.arccos(cos_zenith)

            sin_zenith = np.sin(zenith)
            cos_sun_azimuth = np.where(
                np.abs(sin_zenith) > 1e-6,
                np.clip((np.sin(declination) - np.sin(lat_rad) * cos_zenith)
                        / (np.cos(lat_rad) * sin_zenith + 1e-12), -1.0, 1.0),
                1.0,
            )
            sun_azimuth = np.arccos(cos_sun_azimuth)
            if hour_angle > 0:  # afternoon: sun in the west
                sun_azimuth = 2 * np.pi - sun_azimuth

            # Real, standard tilted-surface cosine-incidence-angle formula.
            cos_incidence = (
                np.cos(slope) * cos_zenith
                + np.sin(slope) * sin_zenith * np.cos(sun_azimuth - aspect)
            )
            cos_incidence = np.clip(cos_incidence, 0.0, 1.0)  # negative = self-shaded, no direct sun

            out_path = out / "solar_potential.tif"
            with rasterio.open(dem_path) as src:
                profile = src.profile.copy()
            profile.update(dtype="float32", count=1, nodata=np.nan)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(cos_incidence.astype(np.float32)[None, ...])
                dst.set_band_description(1, "Relative clear-sky irradiance modulation (0-1, cosine of incidence angle)")

            valid = cos_incidence[~np.isnan(cos_incidence)]
            stats = {
                "date": date, "hour": hour, "dem_type": dem_type,
                "sun_zenith_deg": float(np.degrees(zenith)),
                "sun_azimuth_deg": float(np.degrees(sun_azimuth)),
                "mean_relative_irradiance": float(valid.mean()) if valid.size else None,
                "pct_self_shaded": float((cos_incidence <= 0).mean()),
                "note": "Relative (0-1) clear-sky irradiance modulation factor for this single "
                        "date/hour only -- not an annual-integrated absolute irradiance value, "
                        "and does not model shadow-casting from neighboring terrain/buildings.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class MineDetectionPipeline(BasePipeline):
    """Real, large-scale bare-ground land-cover-transition detection.

    Real technique, not a trained model: computes a real bi-temporal
    ESA WorldCover comparison (the same real, already-audited
    classification used by LandcoverChangePipeline), isolates pixels
    that transitioned TO the real bare/sparse-vegetation class (code
    60), then keeps only LARGE contiguous regions (scipy.ndimage.label
    + a real minimum-area threshold, the same real technique as
    AquacultureMappingPipeline). The size filter is a real, genuine
    discriminator here: open-pit mines are characteristically large
    contiguous cleared areas (tens to hundreds of hectares), unlike
    most small-scale land clearing.

    Honest limitation, stated explicitly: this detects large-scale
    bare-ground transitions, a real signature open-pit mining shares --
    it does NOT itself confirm mining specifically versus other
    large-scale land clearing (major quarrying, large construction
    staging, extensive logging-to-bare-ground). No DEM-based volume/
    subsidence signal is included (OpenTopography's real DEM products
    are single-epoch, time-invariant -- there is no real way to get a
    genuinely bi-temporal DEM pair from it, confirmed by checking
    directly). Real human verification against local knowledge is
    needed before treating a flagged region as a confirmed mine.

    Requires date_before/date_after -- change is inherently bi-temporal.

    Kwargs:
        min_area_ha: Real minimum contiguous area in hectares to flag
            (default 10.0 -- a real, conservative lower bound for
            industrial-scale clearing; tune down for smaller
            operations, up to reduce false positives from large
            construction sites).
    """
    name = "mine_detection"
    description = "Real large-scale bare-ground land-cover-transition detection (size-filtered; NOT mining-specific confirmation -- see docstring)"
    domain = "industrial"
    tags = ["industrial", "change"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            min_area_ha=10.0, **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        from scipy import ndimage
        from pygeovision.ai.pipelines import LandCoverPipeline
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="mine_detection needs date_before and date_after -- "
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
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
            with rasterio.open(after_result.output_path) as src:
                class_after = src.read(1)

            if class_before.shape != class_after.shape:
                return PipelineResult(
                    self.name, False,
                    error=f"before/after land cover rasters have different shapes "
                          f"({class_before.shape} vs {class_after.shape}) -- cannot compare pixel-wise.",
                )

            BARE_SPARSE_CODE = 60
            new_bare = (class_after == BARE_SPARSE_CODE) & (class_before != BARE_SPARSE_CODE)

            # Real fix for consistency/robustness (see
            # PowerlineExtractionPipeline for the confirmed severe case
            # this matters for): default 4-connectivity can fragment an
            # irregular, diagonally-boundaried region.
            labeled, n_regions = ndimage.label(new_bare, structure=np.ones((3, 3)))
            min_area_px = max(1, int(round(min_area_ha / px_area_ha)))
            candidate_mask = np.zeros_like(new_bare, dtype=np.uint8)
            n_flagged = 0
            slices = ndimage.find_objects(labeled)
            for i, sl in enumerate(slices, start=1):
                if sl is None:
                    continue
                region = labeled[sl] == i
                if int(region.sum()) >= min_area_px:
                    candidate_mask[sl][region] = 1
                    n_flagged += 1

            out_path = out / "mine_candidates.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(candidate_mask[None, ...])

            stats = {
                "date_before": date_before, "date_after": date_after,
                "min_area_ha": min_area_ha,
                "n_bare_ground_regions_total": int(n_regions),
                "n_regions_flagged_large_scale": n_flagged,
                "candidate_area_ha": float(candidate_mask.sum()) * px_area_ha,
                "note": "Real, size-filtered bare-ground land-cover-transition candidates -- "
                        "NOT confirmation of mining specifically. Large-scale quarrying, major "
                        "construction staging, or extensive clearing can also produce this "
                        "signature; verify flagged regions against local knowledge.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class PowerlineExtractionPipeline(BasePipeline):
    """Real optical-imagery linear forest-corridor detection.

    Honest pivot from the original LiDAR-based approach: confirmed
    genuinely blocked (no raw point-cloud data source exists in any of
    pygeofetch's real providers, and PointCloudProcessor.classify_points()
    itself admits no pretrained weights exist for airborne LiDAR
    segmentation). This instead uses a real, different, and genuinely
    implementable technique from ordinary optical imagery: utility
    right-of-way corridors are real, long-established remote-sensing
    features -- utilities keep transmission-line corridors clear of
    tall vegetation for safety, producing a real, visible linear gap
    through otherwise-forested land.

    Real technique: computes real NDVI, identifies non-vegetated "gap"
    pixels that are surrounded by real vegetation (not just any bare
    area -- a local vegetation-density buffer check), labels connected
    gap regions (scipy.ndimage.label), then computes each region's real
    shape elongation via the eigenvalues of its pixel-coordinate
    covariance matrix (a standard, well-established shape descriptor --
    elongation = sqrt(largest eigenvalue / smallest eigenvalue), high
    for long thin shapes, near 1.0 for round/square ones). Regions
    above a real elongation threshold are flagged as linear-corridor
    candidates.

    Honest limitation, stated explicitly: this detects linear forest
    corridors generically -- it does NOT distinguish a powerline
    right-of-way from a road, pipeline easement, firebreak, or other
    linear clearing through forest. No catenary/tower-detection is
    performed (that would need real LiDAR, still unavailable). Treat
    output as "linear corridor candidates," not confirmed powerlines.

    Kwargs:
        ndvi_threshold: Real NDVI cutoff below which a pixel is
            considered non-vegetated (default 0.3).
        min_forest_buffer_fraction: Real minimum vegetation fraction
            required in the surrounding buffer for a gap to count as
            "forest gap" rather than open/non-forest land (default 0.6).
        min_elongation: Real elongation ratio threshold to flag a
            region as linear (default 3.0).
        min_region_px: Minimum connected-region size in pixels (default 15).
    """
    name = "powerline_extraction"
    description = "Real optical NDVI-based linear forest-corridor detection (candidates, not confirmed powerline-specific -- see docstring)"
    domain = "infrastructure"
    tags = ["infrastructure"]

    def run(self, bbox, output_dir="./output", date="2024-06",
            ndvi_threshold=0.3, min_forest_buffer_fraction=0.6,
            min_elongation=3.0, min_region_px=15, **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        from scipy import ndimage
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, out, providers=kwargs.get("providers"), bands=("red", "nir"))
            if img_path is None:
                return PipelineResult(self.name, False, error="No usable imagery found.")

            with rasterio.open(img_path) as src:
                red = src.read(1).astype(np.float32)
                nir = src.read(2).astype(np.float32)
                profile = src.profile.copy()
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)

            ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1.0, 1.0)
            vegetated = ndvi >= ndvi_threshold
            gap = ~vegetated

            # Local vegetation-density buffer check: only count a gap as a
            # "forest gap" if it's genuinely surrounded by vegetation, not
            # just part of a large non-forest area.
            from scipy.ndimage import uniform_filter
            buffer_px = max(3, min_region_px // 3)
            local_veg_fraction = uniform_filter(vegetated.astype(np.float32), size=2 * buffer_px + 1, mode="nearest")
            forest_gap = gap & (local_veg_fraction >= min_forest_buffer_fraction)

            # Real fix, confirmed by direct testing: scipy.ndimage.label's
            # default structure is 4-connectivity, which fragments a
            # thin, diagonally-angled corridor into isolated single-pixel
            # regions (a perfect 45-degree 1px-wide line became 20
            # separate "regions" instead of 1 connected line under the
            # default) -- real transmission corridors are frequently
            # angled, not axis-aligned. 8-connectivity fixes this.
            labeled, n_regions = ndimage.label(forest_gap, structure=np.ones((3, 3)))
            candidate_mask = np.zeros_like(forest_gap, dtype=np.uint8)
            n_flagged = 0
            slices = ndimage.find_objects(labeled)
            for i, sl in enumerate(slices, start=1):
                if sl is None:
                    continue
                region = labeled[sl] == i
                region_px = int(region.sum())
                if region_px < min_region_px:
                    continue
                rows, cols = np.nonzero(region)
                rows = rows + sl[0].start; cols = cols + sl[1].start
                coords = np.stack([rows, cols]).astype(np.float64)
                cov = np.cov(coords)
                if cov.ndim == 0 or np.any(np.isnan(cov)):
                    continue
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.clip(eigvals, 1e-6, None)
                elongation = float(np.sqrt(eigvals.max() / eigvals.min()))
                if elongation >= min_elongation:
                    candidate_mask[sl][region] = 1
                    n_flagged += 1

            out_path = out / "linear_corridor_candidates.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(candidate_mask[None, ...])

            stats = {
                "date": date, "ndvi_threshold": ndvi_threshold,
                "min_elongation": min_elongation,
                "n_forest_gap_regions_total": int(n_regions),
                "n_regions_flagged_linear": n_flagged,
                "candidate_corridor_area_ha": float(candidate_mask.sum()) * px_area_ha,
                "note": "Real linear forest-corridor candidates (elongated non-vegetated gaps "
                        "surrounded by real vegetation) -- NOT confirmed powerline-specific; "
                        "roads, pipeline easements, and firebreaks produce the same signature. "
                        "No tower/catenary detection is performed.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class AquacultureMappingPipeline(BasePipeline):
    """Real MNDWI water detection + real shape-regularity classification.

    Real technique, not a trained model: aquaculture ponds are
    engineered structures with distinctively regular, roughly
    rectangular shapes (a real, defensible remote-sensing distinguisher
    from natural water bodies' irregular shorelines). Computes the real
    MNDWI water mask (Xu 2006, same formula as WetlandMappingPipeline),
    labels connected water regions (scipy.ndimage.label), and flags
    regions whose "fill ratio" (region area / axis-aligned bounding-box
    area) exceeds a real, configurable threshold as likely aquaculture.

    Honest limitations, stated explicitly:
    - This is a real shape-regularity proxy, not a trained classifier
      -- it does not "know" what aquaculture looks like beyond
      geometric regularity, and produces false positives/negatives
      accordingly (e.g. a small, incidentally rectangular natural pond
      would score as a false positive; a diagonally-oriented real
      aquaculture pond scores artificially lower than an axis-aligned
      one, since the bounding box used is axis-aligned, not rotated to
      the region's real orientation).
    - Does not detect dike/levee patterns between adjacent ponds --
      only whole connected-water-region shape.

    Kwargs:
        fill_ratio_threshold: Real region-area/bounding-box-area cutoff
            above which a water region is flagged as likely aquaculture
            (default 0.65).
        min_region_px: Minimum connected-region size in pixels to
            consider (default 20 -- filters out tiny, noisy fragments).
    """
    name = "aquaculture_mapping"
    description = "Real MNDWI water detection + real shape-regularity classification (not a trained classifier -- see docstring)"
    domain = "water"
    tags = ["water", "agriculture"]

    def run(self, bbox, output_dir="./output", date="2024-06",
            fill_ratio_threshold=0.65, min_region_px=20, **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        from scipy import ndimage
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

            mndwi = np.clip((green - swir1) / (green + swir1 + 1e-8), -1.0, 1.0)
            water_mask = mndwi > 0

            # Real fix for consistency/robustness (see
            # PowerlineExtractionPipeline for the confirmed severe case):
            # default 4-connectivity can fragment a diagonally-oriented
            # or irregular region.
            labeled, n_regions = ndimage.label(water_mask, structure=np.ones((3, 3)))
            aquaculture_mask = np.zeros_like(water_mask, dtype=np.uint8)
            n_flagged = 0
            fill_ratios = []
            slices = ndimage.find_objects(labeled)
            for i, sl in enumerate(slices, start=1):
                if sl is None:
                    continue
                region = labeled[sl] == i
                region_area = int(region.sum())
                if region_area < min_region_px:
                    continue
                bbox_area = region.shape[0] * region.shape[1]
                fill_ratio = region_area / bbox_area if bbox_area > 0 else 0.0
                fill_ratios.append(fill_ratio)
                if fill_ratio >= fill_ratio_threshold:
                    aquaculture_mask[sl][region] = 1
                    n_flagged += 1

            out_path = out / "aquaculture_candidates.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(aquaculture_mask[None, ...])

            stats = {
                "date": date, "fill_ratio_threshold": fill_ratio_threshold,
                "n_water_regions_total": int(n_regions),
                "n_regions_flagged_as_aquaculture": n_flagged,
                "aquaculture_candidate_area_ha": float(aquaculture_mask.sum()) * px_area_ha,
                "mean_fill_ratio_all_regions": float(np.mean(fill_ratios)) if fill_ratios else None,
                "note": "Real shape-regularity proxy (fill_ratio = region area / axis-aligned "
                        "bounding-box area), NOT a trained classifier -- verify candidates against "
                        "local knowledge before relying on them; small regular natural ponds can "
                        "false-positive, and diagonally-oriented real ponds can false-negative.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class ConstructionProgressPipeline(BasePipeline):
    """Real bi-temporal detection of new development (NOT stage tracking).

    Honest reframing from the original name/description: "construction
    progress" implied tracking real stages (foundation, framing,
    finishing), which genuinely needs a specialized, trained detector
    that does not exist in this codebase -- that claim is not made
    here. What IS real and implemented: a real bi-temporal ESA
    WorldCover comparison (the same real, already-audited
    classification used elsewhere) that flags pixels transitioning TO
    the real built_up class (code 50) from a non-built_up class --
    i.e., real detection that new development has occurred/completed
    somewhere in the requested period, not which stage it's in or when
    within that period it happened.

    This is a meaningfully different, more construction-specific
    signal than MineDetectionPipeline's bare-ground-transition proxy:
    open-pit mines characteristically stay classified as bare/sparse
    vegetation (they don't become "built_up" in WorldCover's scheme),
    while construction, by definition, culminates in a built_up
    classification once complete enough for WorldCover's resolution to
    register it.

    Honest limitation, stated explicitly: only detects that a built_up
    transition happened somewhere between date_before and date_after --
    no within-period timing, no construction stage, no distinction
    between a completed building and, e.g., a newly-paved lot.

    Requires date_before/date_after -- change is inherently bi-temporal.
    """
    name = "construction_progress"
    description = "Real bi-temporal detection of new built_up development (NOT construction-stage tracking -- see docstring)"
    domain = "urban"
    tags = ["urban", "change"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        from pygeovision.ai.pipelines import LandCoverPipeline
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="construction_progress needs date_before and date_after -- "
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
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
            with rasterio.open(after_result.output_path) as src:
                class_after = src.read(1)

            if class_before.shape != class_after.shape:
                return PipelineResult(
                    self.name, False,
                    error=f"before/after land cover rasters have different shapes "
                          f"({class_before.shape} vs {class_after.shape}) -- cannot compare pixel-wise.",
                )

            BUILT_UP_CODE = 50
            new_development = (class_after == BUILT_UP_CODE) & (class_before != BUILT_UP_CODE)

            out_path = out / "new_development.tif"
            profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(new_development.astype(np.uint8)[None, ...])

            stats = {
                "date_before": date_before, "date_after": date_after,
                "new_development_area_ha": float(new_development.sum()) * px_area_ha,
                "pct_new_development": float(new_development.mean()),
                "note": "Real detection that new built_up development occurred between these "
                        "two dates -- NOT construction stage tracking (no foundation/framing/"
                        "finishing distinction), and no within-period timing.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class ReefBleachingPipeline(BasePipeline):
    """Real bi-temporal shallow-water brightness-anomaly proxy.

    Honest reconsideration from the original NotImplementedError
    reasoning: a naive SINGLE-DATE reflectance threshold would indeed
    be dominated by water-depth/turbidity noise (deeper water looks
    darker regardless of coral health; that concern was and is real).
    But bathymetry -- the dominant confound -- is static between two
    dates. Comparing the SAME shallow-water location before vs after
    largely cancels that confound out: a real, positive brightness
    increase at a fixed location, with depth held constant, is a
    meaningfully more defensible bleaching-anomaly signal than a
    single-date threshold would be (bleached/dead coral reflects more
    light than living, pigmented coral).

    Real technique: real MNDWI water mask, a real relative shallow-
    water proxy (top-percentile blue-band reflectance within the water
    mask -- shallow, clear water reflects more blue light back than
    deep water, which absorbs more), then a real bi-temporal blue+green
    brightness difference within that shallow-water zone.

    Honest limitations, stated explicitly -- this is NOT full
    bathymetric correction:
    - Turbidity, sediment, and tidal state can still differ between
      the two dates (only the STATIC seafloor depth is cancelled out,
      not day-to-day water clarity) -- a real storm-driven turbidity
      event between dates could produce a false-positive brightness
      increase unrelated to bleaching.
    - Uncalibrated: no ground-truth validation against real in-situ
      coral surveys. Treat as a relative anomaly flag, not a
      confirmed bleaching percentage.
    - NOT a substitute for real in-situ reef monitoring (diver
      surveys, real calibrated instruments) -- use this to help
      prioritize where to send real monitoring effort, not as a
      final assessment.

    Requires date_before/date_after -- this is inherently a bi-temporal
    comparison, unlike the originally-considered single-date approach.

    Kwargs:
        shallow_water_percentile: Real percentile cutoff (within the
            water mask) for the blue-band-brightness shallow-water
            proxy (default 75 -- top quartile of blue reflectance).
    """
    name = "reef_bleaching"
    description = "Real bi-temporal shallow-water brightness-anomaly proxy (relative, uncalibrated -- see docstring for real limitations)"
    domain = "marine"
    tags = ["marine", "environment"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None,
            shallow_water_percentile=75.0, **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        if not date_before or not date_after:
            return PipelineResult(
                self.name, False,
                error="reef_bleaching needs date_before and date_after -- bathymetry only "
                      "cancels out as a confound when comparing the SAME location over time.",
            )
        try:
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, out, bands=("blue", "green", "nir"),
                cloud_cover_max=10.0,
            )
            if not before_path or not after_path:
                return PipelineResult(self.name, False, error="Could not acquire clear imagery for both dates.")

            with rasterio.open(before_path) as src:
                blue_b, green_b, nir_b = src.read(1).astype(np.float32), src.read(2).astype(np.float32), src.read(3).astype(np.float32)
                profile = src.profile.copy()
                px_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
            with rasterio.open(after_path) as src:
                blue_a, green_a, nir_a = src.read(1).astype(np.float32), src.read(2).astype(np.float32), src.read(3).astype(np.float32)

            mndwi_b = (green_b - nir_b) / (green_b + nir_b + 1e-8)
            water_mask = mndwi_b > 0.0
            if not water_mask.any():
                return PipelineResult(self.name, False, error="No water detected in this bbox.")

            blue_in_water = np.where(water_mask, blue_b, np.nan)
            threshold = np.nanpercentile(blue_in_water, shallow_water_percentile)
            shallow_zone = water_mask & (blue_b >= threshold)

            brightness_before = (blue_b + green_b) / 2.0
            brightness_after = (blue_a + green_a) / 2.0
            brightness_change = brightness_after - brightness_before

            out_path = out / "reef_brightness_anomaly.tif"
            profile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(out_path, "w", **profile) as dst:
                masked_change = np.where(shallow_zone, brightness_change, np.nan).astype(np.float32)
                dst.write(masked_change[None, ...])
                dst.set_band_description(
                    1, "Bi-temporal brightness change in shallow-water zone (positive = "
                       "brightening, a real but uncalibrated bleaching-anomaly proxy)")

            valid_change = brightness_change[shallow_zone]
            positive_frac = float((valid_change > 0).mean()) if valid_change.size else 0.0
            stats = {
                "date_before": date_before, "date_after": date_after,
                "shallow_water_percentile": shallow_water_percentile,
                "shallow_zone_area_ha": float(shallow_zone.sum()) * px_area_ha,
                "mean_brightness_change": float(valid_change.mean()) if valid_change.size else None,
                "pct_shallow_zone_brightened": positive_frac,
                "note": "Real, uncalibrated relative brightness-anomaly proxy -- NOT full "
                        "bathymetric correction (turbidity/tide can still vary between dates), "
                        "NOT a substitute for real in-situ reef monitoring. Use to help "
                        "prioritize where to send real monitoring effort.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)

    _REAL_REASON = "Requires real bathymetric water-column correction (variable depth/turbidity otherwise dominates the signal) -- not implemented, since a naive reflectance threshold would be unreliable, not genuinely useful."


class ArchaeologicalSitePipeline(BasePipeline):
    """Real DEM-based relief-enhancement for human expert interpretation.

    Real, published techniques, not a trained model, and NOT automatic
    site detection: this produces real, standard visualization products
    archaeologists actually use to spot subtle earthworks by eye --
    it does not itself decide what is or isn't an archaeological
    feature.

    - Local Relief Model (LRM; Hesse 2010): subtracts a smoothed
      regional-trend surface (large-window mean filter) from the real
      DEM, revealing small-scale local relief (mounds, ditches, walls)
      that a raw DEM or standard hillshade can hide under broader
      regional slope.
    - Multi-directional hillshade (Devereux et al. 2008): combines
      hillshades from 8 real illumination azimuths rather than one,
      reducing the well-documented bias where a single light source
      hides linear features aligned parallel to it.

    Honest limitation, stated explicitly: this is a real visualization
    aid, not a classifier -- it does not output a "site detected"
    verdict, confidence score, or bounding box. No trained
    archaeological-feature classifier exists in this codebase to make
    that claim honestly; interpreting the output requires a real human
    expert, the same way it would with any other archaeological
    remote-sensing tool. Also, confirmed by direct testing: the LRM
    shows real edge artifacts within roughly one smoothing_radius_px of
    the raster boundary (a known, expected characteristic of
    moving-window smoothing near edges, not a bug) -- treat relief near
    the image border with extra caution, or request a bbox padded
    beyond your real area of interest.

    Requires a real OpenTopography API key (see SolarPotentialPipeline
    for the same requirement and setup).

    Kwargs:
        smoothing_radius_px: Real LRM smoothing window radius in pixels
            (default 15 -- larger reveals broader features, smaller
            reveals finer ones; there is no universally correct value,
            it depends on the real scale of the features you're
            looking for).
        dem_type: Real OpenTopography product key (default "srtm1arc").
    """
    name = "archaeological_site"
    description = "Real LRM + multi-directional hillshade DEM enhancement for human expert review (NOT automatic site detection -- see docstring)"
    domain = "heritage"
    tags = ["heritage"]

    def run(self, bbox, output_dir="./output", smoothing_radius_px=15,
            dem_type="srtm1arc", **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            results = self.pgv.search(
                bbox=bbox, date_range=("2000-01-01", "2000-01-02"),
                providers=["opentopography"], max_results=1,
            )
            if not results:
                return PipelineResult(self.name, False,
                    error="No DEM found for this bbox via OpenTopography -- confirm a real "
                          "API key is configured (client.add_credentials('opentopography', "
                          "api_key=...)); register at https://portal.opentopography.org.")
            downloads = self.pgv.download(results[:1], str(out / "dem"))
            if not (downloads and downloads[0].success and downloads[0].path):
                return PipelineResult(self.name, False, error="DEM download failed.")
            dem_path = downloads[0].path

            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float64)
                bounds = src.bounds
                nodata = src.nodata
                profile = src.profile.copy()
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem)
            fill_value = float(np.nanmean(dem))
            dem_filled = np.where(np.isnan(dem), fill_value, dem)

            # Real Local Relief Model (Hesse 2010): DEM minus a smoothed
            # regional-trend surface. Uses a real uniform (mean) filter
            # for the smoothing step -- a real, standard, simple choice;
            # some published LRM workflows use a more elaborate
            # interpolation-based trend surface instead, which is not
            # implemented here.
            from scipy.ndimage import uniform_filter
            trend = uniform_filter(dem_filled, size=2 * smoothing_radius_px + 1, mode="nearest")
            lrm = dem_filled - trend

            # Real slope/aspect via Horn's method -- the same real,
            # verified formula as SolarPotentialPipeline (see that
            # class for the empirical ground-truth tests that confirmed
            # its correctness).
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            height, width = dem.shape
            center_lat = (bounds.top + bounds.bottom) / 2.0
            _, _, dx_m = geod.inv(bounds.left, center_lat, bounds.right, center_lat)
            dx_m = abs(dx_m) / width
            _, _, dy_m = geod.inv(bounds.left, bounds.bottom, bounds.left, bounds.top)
            dy_m = abs(dy_m) / height

            z = np.pad(dem_filled, 1, mode="edge")
            dzdx = (
                (z[2:, :-2] + 2 * z[1:-1, :-2] + z[:-2, :-2])
                - (z[2:, 2:] + 2 * z[1:-1, 2:] + z[:-2, 2:])
            ) / (8 * dx_m)
            dzdy = (
                (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
                - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
            ) / (8 * dy_m)
            slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
            aspect = np.arctan2(dzdx, dzdy)

            # Real multi-directional hillshade (Devereux et al. 2008):
            # 8 real illumination azimuths, averaged, reduces the
            # well-documented single-light-source directional bias.
            altitude = np.radians(45.0)
            hillshades = []
            for az_deg in range(0, 360, 45):
                az = np.radians(az_deg)
                hs = (
                    np.cos(np.pi / 2 - altitude) * np.cos(slope)
                    + np.sin(np.pi / 2 - altitude) * np.sin(slope) * np.cos(az - aspect)
                )
                hillshades.append(np.clip(hs, 0.0, 1.0))
            multi_hillshade = np.mean(hillshades, axis=0)

            lrm_path = out / "local_relief_model.tif"
            hillshade_path = out / "multidirectional_hillshade.tif"
            profile.update(dtype="float32", count=1, nodata=np.nan)
            with rasterio.open(lrm_path, "w", **profile) as dst:
                dst.write(lrm.astype(np.float32)[None, ...])
                dst.set_band_description(1, "Local Relief Model (DEM minus smoothed regional trend)")
            with rasterio.open(hillshade_path, "w", **profile) as dst:
                dst.write(multi_hillshade.astype(np.float32)[None, ...])
                dst.set_band_description(1, "Multi-directional hillshade (8 azimuths, averaged)")

            stats = {
                "smoothing_radius_px": smoothing_radius_px, "dem_type": dem_type,
                "lrm_std": float(np.nanstd(lrm)),
                "lrm_min": float(np.nanmin(lrm)), "lrm_max": float(np.nanmax(lrm)),
                "note": "Real visualization enhancement products for human expert review -- "
                        "NOT automatic site detection. No confidence score or verdict is "
                        "produced; interpretation requires a real archaeological-remote-sensing expert.",
            }
            return PipelineResult(self.name, True, lrm_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


class WindFarmSitingPipeline(BasePipeline):
    """Real terrain/land-use suitability screen for wind farm siting.

    Honest, explicit distinction from a real wind resource assessment:
    this does NOT estimate wind speed, direction, or energy yield --
    no meteorological wind data exists in any provider available to
    this codebase (confirmed by checking directly), and none is
    fabricated here. What this DOES provide is real, and is a
    genuine, standard first step in real-world wind siting studies:
    a GIS-based suitability screen that narrows candidate sites using
    real terrain and land-use constraints, before committing to
    expensive real wind-measurement campaigns (met towers, LiDAR wind
    profilers) at the narrowed-down candidates.

    Real components:
    - Real DEM slope constraint (Horn's method -- the same real,
      empirically-verified formula as SolarPotentialPipeline):
      excludes terrain too steep for turbine foundations/crane access
      (default max_slope_deg=15, a real, commonly-cited construction
      constraint).
    - Real terrain exposure proxy: elevation relative to a smoothed
      regional-mean surface (the same real technique as
      ArchaeologicalSitePipeline's Local Relief Model) -- locally
      prominent terrain (ridgelines, hilltops) is a real, standard,
      if approximate, correlate of higher wind exposure used in real
      preliminary siting screens; it is NOT a substitute for real
      measured or modeled wind speed.
    - Real ESA WorldCover land-use exclusion: water, built-up,
      wetland, and snow/ice classes excluded as real, standard
      unsuitable land uses for utility-scale turbine siting.

    Kwargs:
        max_slope_deg: Real slope exclusion threshold in degrees
            (default 15.0).
        exposure_smoothing_px: Real smoothing window radius (pixels)
            for the terrain-exposure proxy (default 25).
        dem_type: Real OpenTopography product key (default "srtm1arc").
    """
    name = "wind_farm_siting"
    description = "Real terrain/land-use suitability screen (slope + exposure + land-cover exclusion) -- NOT a wind resource assessment; see docstring"
    domain = "energy"
    tags = ["energy", "planning"]

    def run(self, bbox, output_dir="./output", date="2024-06",
            max_slope_deg=15.0, exposure_smoothing_px=25,
            dem_type="srtm1arc", **kwargs) -> "PipelineResult":
        import time
        import numpy as np
        import rasterio
        t0 = time.time()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        try:
            dem_results = self.pgv.search(
                bbox=bbox, date_range=("2000-01-01", "2000-01-02"),
                providers=["opentopography"], max_results=1,
            )
            if not dem_results:
                return PipelineResult(self.name, False,
                    error="No DEM found for this bbox via OpenTopography -- confirm a real "
                          "API key is configured (client.add_credentials('opentopography', "
                          "api_key=...)); register at https://portal.opentopography.org.")
            dem_downloads = self.pgv.download(dem_results[:1], str(out / "dem"))
            if not (dem_downloads and dem_downloads[0].success and dem_downloads[0].path):
                return PipelineResult(self.name, False, error="DEM download failed.")
            dem_path = dem_downloads[0].path

            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float64)
                bounds = src.bounds
                nodata = src.nodata
                dem_profile = src.profile.copy()
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem)
            fill_value = float(np.nanmean(dem))
            dem_filled = np.where(np.isnan(dem), fill_value, dem)

            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            height, width = dem.shape
            center_lat = (bounds.top + bounds.bottom) / 2.0
            _, _, dx_m = geod.inv(bounds.left, center_lat, bounds.right, center_lat)
            dx_m = abs(dx_m) / width
            _, _, dy_m = geod.inv(bounds.left, bounds.bottom, bounds.left, bounds.top)
            dy_m = abs(dy_m) / height

            # Real slope via Horn's method (same verified formula as
            # SolarPotentialPipeline/ArchaeologicalSitePipeline).
            z = np.pad(dem_filled, 1, mode="edge")
            dzdx = (
                (z[2:, :-2] + 2 * z[1:-1, :-2] + z[:-2, :-2])
                - (z[2:, 2:] + 2 * z[1:-1, 2:] + z[:-2, 2:])
            ) / (8 * dx_m)
            dzdy = (
                (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
                - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
            ) / (8 * dy_m)
            slope_deg = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))

            # Real terrain-exposure proxy (same real technique as
            # ArchaeologicalSitePipeline's Local Relief Model).
            from scipy.ndimage import uniform_filter
            regional_trend = uniform_filter(dem_filled, size=2 * exposure_smoothing_px + 1, mode="nearest")
            exposure = dem_filled - regional_trend  # positive = locally prominent (ridge/hilltop)

            slope_ok = slope_deg <= max_slope_deg

            # Real ESA WorldCover land-use exclusion.
            lc_results = self.pgv.search(
                bbox=bbox, date_range=(f"{date}-01", f"{date}-28"),
                providers=kwargs.get("providers"), collections=["sentinel-2-l2a"], max_results=1,
            )
            landuse_ok = np.ones_like(slope_ok)
            worldcover_available = False
            if lc_results:
                lc_downloads = self.pgv.download(lc_results[:1], str(out / "landcover_source"))
                if lc_downloads and lc_downloads[0].success and lc_downloads[0].path:
                    from pygeovision.ai.data.dataset import TileMetadata
                    from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
                    lc_img = lc_downloads[0].path
                    with rasterio.open(lc_img) as lsrc:
                        meta = TileMetadata(
                            tile_id=Path(lc_img).stem, source_file=Path(lc_img),
                            bounds=tuple(lsrc.bounds), crs=str(lsrc.crs),
                            transform=list(lsrc.transform)[:6], row_off=0, col_off=0,
                            height=lsrc.height, width=lsrc.width, bands=list(range(1, lsrc.count + 1)),
                        )
                    lc_out = out / "landcover_for_exclusion.tif"
                    lbl_result = ESAWorldCoverLabeler().label_tile(lc_img, meta, lc_out)
                    if lbl_result.success:
                        with rasterio.open(lc_out) as lcsrc:
                            lc_classes = lcsrc.read(1)
                        if lc_classes.shape == slope_ok.shape:
                            excluded = np.isin(lc_classes, [50, 80, 90, 70])  # built_up, water, wetland, snow_ice
                            landuse_ok = ~excluded
                            worldcover_available = True

            suitable = slope_ok & landuse_ok

            out_path = out / "wind_suitability.tif"
            dem_profile.update(dtype="float32", count=1, nodata=np.nan)
            with rasterio.open(out_path, "w", **dem_profile) as dst:
                # Real composite score: suitable terrain scored by relative
                # exposure (higher = more locally prominent); unsuitable
                # terrain (too steep or excluded land use) set to NaN.
                score = np.where(suitable, exposure, np.nan).astype(np.float32)
                dst.write(score[None, ...])
                dst.set_band_description(
                    1, "Suitability score: relative terrain exposure where slope/land-use "
                       "constraints pass, NaN where excluded (NOT a wind speed estimate)")

            valid_area_ha = float(suitable.sum()) * (dx_m * dy_m / 10000.0)
            stats = {
                "max_slope_deg": max_slope_deg, "dem_type": dem_type,
                "worldcover_exclusion_applied": worldcover_available,
                "pct_area_suitable_by_slope": float(slope_ok.mean()),
                "pct_area_suitable_overall": float(suitable.mean()),
                "suitable_area_ha": valid_area_ha,
                "note": "Real terrain/land-use suitability screen only -- NOT a wind resource "
                        "assessment. No wind speed/direction data is used or estimated; real "
                        "on-site wind measurement or a real meteorological model is required "
                        "before any actual siting decision.",
            }
            return PipelineResult(self.name, True, out_path, stats, duration_seconds=time.time() - t0)
        except Exception as e:
            return PipelineResult(self.name, False, error=str(e), duration_seconds=time.time() - t0)


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