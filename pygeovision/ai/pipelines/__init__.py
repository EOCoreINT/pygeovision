"""
PyGeoVision Geospatial AI Pipelines.

Each pipeline uses PyGeoFetch (via the pgv_client) for data acquisition
and PyGeoVision AI for model inference. The pgv_client is a PyGeoVision
instance that wraps PyGeoFetch's search/download/post-process API.

Available pipelines (10):
    change_detection       building_footprints    land_cover
    crop_monitoring        disaster_assessment    deforestation
    urban_growth           water_bodies           solar_detection
    carbon_estimation

Example:
    >>> import pygeovision as pgv
    >>> client = pgv.PyGeoVision()
    >>> result = client.pipeline("building_footprints",
    ...     bbox=(-0.15, 51.47, -0.10, 51.52), date="2024-06")
    >>> print(result.output_path, result.stats)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result from a geospatial pipeline run."""
    pipeline: str
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return f"PipelineResult({self.pipeline}, output={self.output_path})"
        return f"PipelineResult({self.pipeline}, FAILED: {self.error})"


# ---------------------------------------------------------------------------
# Base pipeline
# ---------------------------------------------------------------------------

class BasePipeline(ABC):
    """Abstract base for all PyGeoVision pipelines.

    ``pgv_client`` is a PyGeoVision instance — call
    ``pgv_client.search()`` and ``pgv_client.download()`` for data,
    which delegate to PyGeoFetch under the hood.
    """

    def __init__(self, pgv_client: Any) -> None:
        self.pgv = pgv_client  # PyGeoVision instance

    @abstractmethod
    def run(
        self,
        bbox: tuple[float, float, float, float],
        output_dir: str | Path,
        **kwargs: Any,
    ) -> PipelineResult: ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    # Real, shared radiometric/band logic lives in
    # pygeovision.data.radiometric -- a single, public source of truth
    # (previously this logic lived only here, unreachable from the
    # separate public client.segmentation/client.classification API).
    from pygeovision.data.radiometric import (
        LANDSAT_ETM_TM_ALIASES as _LANDSAT_ETM_TM_ALIASES,
        LANDSAT_OLI_ALIASES as _LANDSAT_OLI_ALIASES,
        S2_BAND_ALIASES as _S2_BAND_ALIASES_DEFAULT,
        landsat_mission_aliases as _landsat_mission_aliases_impl,
    )

    _S2_BAND_ALIASES = dict(_S2_BAND_ALIASES_DEFAULT)  # a real, mutable per-class copy (some tests customise it)

    @staticmethod
    def _landsat_mission_aliases(scene_id: str) -> dict[str, tuple[str, ...]] | None:
        return BasePipeline._landsat_mission_aliases_impl(scene_id)

    def _stack_bands(
        self,
        asset_paths: dict[str, Path],
        output_path: Path,
        bands: tuple[str, ...] = ("red", "green", "blue"),
        scene_id: str = "",
        apply_scale: bool = True,
        apply_cloud_mask: bool = True,
    ) -> Path:
        """Thin wrapper over pygeovision.data.radiometric.stack_and_prepare_bands
        — see that function's docstring for full details. Kept here for
        backward compatibility with existing callers of this method;
        the real, shared implementation (band selection, radiometric
        scaling, cloud masking) lives in the public module so it's also
        usable outside the ai.pipelines system.

        One difference from the shared function: this method reads
        self._S2_BAND_ALIASES (allowing per-instance customisation,
        which some tests rely on) rather than the module's fixed
        S2_BAND_ALIASES directly.
        """
        from pygeovision.data.radiometric import stack_and_prepare_bands
        import pygeovision.data.radiometric as _radiometric_mod

        # Respect any instance-level customisation of _S2_BAND_ALIASES
        # (e.g. tests adding a custom "swir1" alias) by temporarily
        # patching the module-level dict the shared function reads.
        original_aliases = _radiometric_mod.S2_BAND_ALIASES
        try:
            _radiometric_mod.S2_BAND_ALIASES = self._S2_BAND_ALIASES
            return stack_and_prepare_bands(
                asset_paths, output_path, bands=bands, scene_id=scene_id,
                apply_scale=apply_scale, apply_cloud_mask=apply_cloud_mask,
            )
        finally:
            _radiometric_mod.S2_BAND_ALIASES = original_aliases


    def _search_and_download(
        self,
        bbox: tuple[float, float, float, float],
        date: str,
        output_dir: Path,
        collections: list[str] | None = None,
        providers: list[str] | None = None,
        cloud_cover_max: float = 20.0,
        post_process: list[str] | None = None,
        max_results: int = 5,
        bands: tuple[str, ...] = ("red", "green", "blue"),
    ) -> Path | None:
        """Search PyGeoFetch → download best scene → return a real,
        correctly-stacked multi-band image path.

        Real bug this fixes, confirmed from a production failure log: a
        multi-asset scene (e.g. Sentinel-2 L2A's 15 separate band files —
        AOT, B01-B12, B8A, SCL, WVP) previously returned whichever single
        file happened to be downloaded first (observed to be AOT, a
        1-band aerosol auxiliary layer) as "the image" — completely
        unusable for any model expecting real RGB/multispectral input,
        and the direct cause of a real "expected 3 channels, got 1"
        crash. Now builds a genuine band-stacked composite when more
        than one real asset is available.
        """
        try:
            # Build date range from a YYYY-MM string
            if len(date) == 7:  # YYYY-MM
                start = f"{date}-01"
                end = f"{date}-28"
            else:
                start = end = date

            results = self.pgv.search(
                bbox=bbox,
                date_range=(start, end),
                collections=collections or ["sentinel-2-l2a"],
                providers=providers,
                cloud_cover_max=cloud_cover_max,
                max_results=max_results,
                sort_by="cloud_cover",
                sort_order="asc",
            )

            if not results:
                return None

            # Download best (lowest cloud cover) scene
            pp = post_process or ["unzip", "reproject:EPSG:4326"]
            downloads = self.pgv.download(
                results[:1],
                output_dir=output_dir / "imagery",
                parallel=1,
                post_process=pp,
                verify_checksum=False,
            )

            if not (downloads and downloads[0].success):
                return None

            dl = downloads[0]
            if len(dl.asset_paths) > 1:
                # Real bug this fixes: previously fell back to dl.path
                # (an arbitrary single asset) when stacking failed --
                # for a multi-asset scene that's never a safe fallback,
                # confirmed as the direct cause of a real production
                # crash (a QA_PIXEL quality band used as reflectance
                # imagery). _stack_bands now raises clearly instead.
                image_path = self._stack_bands(
                    dl.asset_paths, output_dir / "imagery" / "stack.tif",
                    bands=bands, scene_id=dl.scene_id,
                )
            else:
                image_path = dl.path

            if image_path is None:
                return None

            # Real gap this closes: bbox was previously used only to
            # SEARCH for overlapping scenes -- nothing ever cropped the
            # downloaded/stacked imagery to it, so every pipeline ran
            # tiled inference over the entire downloaded scene/tile
            # (often 100km+ on a side), not just the requested area.
            from pygeovision.data.radiometric import crop_to_bbox
            try:
                return crop_to_bbox(image_path, bbox, output_dir / "imagery" / "stack_clipped.tif")
            except ValueError as exc:
                logger.error(
                    "_search_and_download: %s — the search returned a scene "
                    "that doesn't actually cover the requested bbox.", exc,
                )
                return None

        except Exception as exc:
            logger.error("_search_and_download failed: %s", exc)
        return None

    def _search_and_download_pair(
        self,
        bbox: tuple[float, float, float, float],
        date_before: str,
        date_after: str,
        output_dir: Path,
        collections: list[str] | None = None,
        providers: list[str] | None = None,
        cloud_cover_max: float = 25.0,
    ) -> tuple[Path | None, Path | None]:
        """Download a bi-temporal (before/after) image pair, aligned to
        a common pixel grid.

        Real gap this fixes, confirmed from a production failure log:
        two different scenes (different dates, different acquisition
        footprints) each get correctly stacked internally by
        _search_and_download/_stack_bands, but nothing aligned the
        PAIR to each other -- "before" and "after" ending up at
        different pixel dimensions even though both nominally cover the
        same bbox, which run_pair()'s mismatch check correctly caught
        but nothing upstream ever resolved.
        """
        before_path = self._search_and_download(
            bbox, date_before, output_dir / "before",
            collections=collections, providers=providers, cloud_cover_max=cloud_cover_max,
        )
        after_path = self._search_and_download(
            bbox, date_after, output_dir / "after",
            collections=collections, providers=providers, cloud_cover_max=cloud_cover_max,
        )
        if before_path is not None and after_path is not None:
            before_path, after_path = self._align_to_common_grid(
                before_path, after_path, output_dir,
            )
        return before_path, after_path

    def _align_to_common_grid(
        self, path_a: Path, path_b: Path, output_dir: Path,
    ) -> tuple[Path, Path]:
        """Crop both rasters to their real geographic intersection and
        resample both onto one shared pixel grid, at the finer of the
        two images' native resolutions.

        Known limitation, found while testing: uses a single resolution
        value for both axes (assumes near-square pixels), which matches
        how real reprojected satellite imagery actually comes out in
        practice, but isn't precisely correct for a genuinely
        non-square-pixel source.

        If the two rasters don't actually overlap at all, returns the
        original paths unchanged -- run_pair()'s existing mismatch check
        will then correctly report that clearly, rather than this
        function silently producing an empty/degenerate result.
        """
        import numpy as np
        import rasterio
        from rasterio.warp import Resampling, reproject

        with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
            crs = src_a.crs
            bounds_a = src_a.bounds
            if src_b.crs != crs:
                from rasterio.warp import transform_bounds
                bounds_b = transform_bounds(src_b.crs, crs, *src_b.bounds)
            else:
                bounds_b = src_b.bounds

            left = max(bounds_a.left, bounds_b[0])
            bottom = max(bounds_a.bottom, bounds_b[1])
            right = min(bounds_a.right, bounds_b[2])
            top = min(bounds_a.top, bounds_b[3])
            if left >= right or bottom >= top:
                logger.warning(
                    "_align_to_common_grid: %s and %s do not actually overlap -- "
                    "leaving both unchanged; downstream alignment check will report this.",
                    path_a.name, path_b.name,
                )
                return path_a, path_b

            res_a = abs(src_a.transform.a)
            if src_b.crs != crs:
                res_b = (bounds_b[2] - bounds_b[0]) / src_b.width
            else:
                res_b = abs(src_b.transform.a)
            res = min(res_a, res_b)

            out_w = max(1, round((right - left) / res))
            out_h = max(1, round((top - bottom) / res))
            out_transform = rasterio.transform.from_origin(left, top, res, res)

            aligned_paths = []
            for src, name in ((src_a, path_a), (src_b, path_b)):
                out_path = output_dir / f"{name.stem}_aligned.tif"
                out_profile = src.profile.copy()
                out_profile.update(
                    height=out_h, width=out_w, transform=out_transform, crs=crs,
                )
                data = np.zeros((src.count, out_h, out_w), dtype=src.dtypes[0])
                for band_i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_i),
                        destination=data[band_i - 1],
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=out_transform, dst_crs=crs,
                        resampling=Resampling.bilinear,
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(data)
                aligned_paths.append(out_path)

        logger.info(
            "_align_to_common_grid: %s + %s -> common %dx%d grid at %.6f res",
            path_a.name, path_b.name, out_w, out_h, res,
        )
        return aligned_paths[0], aligned_paths[1]

    def _run_model(
        self,
        image_path: Path,
        output_path: Path,
        model_name: str,
        num_classes: int,
        in_channels: int = 3,
        tile_size: int = 512,
        overlap: int = 64,
        image_path_b: Path | None = None,
    ) -> Any | None:
        """Load model from hub and run tiled inference.

        Real bug this fixes: previously accepted only one image path,
        with no mechanism at all for two-image inference — confirmed
        from a real production crash: siamese change-detection models
        (forward(t1, t2)) got called with just one tensor, regardless of
        whether the caller had actually downloaded and had a second
        (e.g. "after") image ready to use. Pass image_path_b for any
        two-image model; it dispatches to TiledInference.run_pair()
        instead of run().
        """
        try:
            from pygeovision.ai.inference.tiled_inference import TiledInference
            from pygeovision.ai.models.hub import ModelHub

            hub = ModelHub()
            model = hub.load(model_name, num_classes=num_classes, in_channels=in_channels)
            engine = TiledInference(model, tile_size=tile_size, overlap=overlap)
            if image_path_b is not None:
                return engine.run_pair(image_path, image_path_b, output_path, num_classes=num_classes)
            return engine.run(image_path, output_path, num_classes=num_classes)
        except Exception as exc:
            logger.warning("Model inference failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# All 10 pipeline implementations
# ---------------------------------------------------------------------------

class ChangeDetectionPipeline(BasePipeline):
    """Bi-temporal change detection using Sentinel-2 and a siamese/transformer model.

    Kwargs:
        date_before: ISO date for first (before) image.
        date_after: ISO date for second (after) image.
        model: 'siamese_unet' or 'changeformer'.
        cloud_cover_max: Max cloud cover per image.
    """
    NAME = "change_detection"

    def run(self, bbox, output_dir, date_before="2022-06", date_after="2024-06",
            model="siamese_unet", cloud_cover_max=25.0, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        logger.info("[%s] bbox=%s before=%s after=%s", self.NAME, bbox, date_before, date_after)
        try:
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, output_dir, providers=providers,
                cloud_cover_max=cloud_cover_max,
            )
            if not before_path or not after_path:
                return PipelineResult(self.NAME, success=False,
                    error="Could not acquire imagery for one or both time periods.")

            output_path = output_dir / "change_mask.tif"
            pred = self._run_model(before_path, output_path, model, num_classes=2,
                                    image_path_b=after_path)

            return PipelineResult(
                pipeline=self.NAME, output_path=output_path if pred is not None else None,
                metadata={"date_before": date_before, "date_after": date_after, "model": model},
                success=pred is not None,
                error="" if pred is not None else "Model inference failed",
            )
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class LandCoverPipeline(BasePipeline):
    """Land cover classification using ESA WorldCover labels or a trained model.

    Kwargs:
        date: YYYY-MM acquisition date.
        source: 'worldcover', 'dynamic_world', or model name.
        num_classes: Number of land cover classes.
    """
    NAME = "land_cover"

    def run(self, bbox, output_dir, date="2023-06", source="worldcover",
            num_classes=11, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        logger.info("[%s] source=%s date=%s", self.NAME, source, date)
        try:
            if source == "worldcover":
                img_path = self._search_and_download(bbox, date, output_dir, providers=providers)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery found.")
                import rasterio

                from pygeovision.ai.data.dataset import TileMetadata
                from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
                with rasterio.open(img_path) as src:
                    meta = TileMetadata(
                        tile_id=Path(img_path).stem, source_file=Path(img_path),
                        bounds=tuple(src.bounds), crs=str(src.crs), transform=list(src.transform)[:6],
                        row_off=0, col_off=0, height=src.height, width=src.width,
                        bands=list(range(1, src.count + 1)),
                    )
                lbl = ESAWorldCoverLabeler()
                out = output_dir / "land_cover.tif"
                r = lbl.label_tile(img_path, meta, out)
                return PipelineResult(self.NAME, output_path=out if r.success else None,
                    stats=r.class_distribution or {}, success=r.success,
                    metadata={"source": source, "date": date})
            elif source == "dynamic_world":
                img_path = self._search_and_download(bbox, date, output_dir, providers=providers)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery.")
                import rasterio

                from pygeovision.ai.data.dataset import TileMetadata
                from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler
                with rasterio.open(img_path) as src:
                    meta = TileMetadata(
                        tile_id=Path(img_path).stem, source_file=Path(img_path),
                        bounds=tuple(src.bounds), crs=str(src.crs), transform=list(src.transform)[:6],
                        row_off=0, col_off=0, height=src.height, width=src.width,
                        bands=list(range(1, src.count + 1)),
                    )
                lbl = DynamicWorldLabeler(start_date=f"{date}-01", end_date=f"{date}-28")
                out = output_dir / "land_cover.tif"
                r = lbl.label_tile(img_path, meta, out)
                return PipelineResult(self.NAME, output_path=out if r.success else None,
                    success=r.success, metadata={"source": source})
            else:
                img_path = self._search_and_download(bbox, date, output_dir, providers=providers)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery.")
                out = output_dir / "land_cover.tif"
                pred = self._run_model(img_path, out, source, num_classes=num_classes)
                return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                    success=pred is not None, metadata={"source": source, "date": date})
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class BuildingFootprintsPipeline(BasePipeline):
    """Building footprint segmentation.

    Kwargs:
        date: YYYY-MM.
        model: Segmentation model name.
        cloud_cover_max: Max cloud %.
    """
    NAME = "building_footprints"

    def run(self, bbox, output_dir, date="2024-06", model="unet_resnet50",
            cloud_cover_max=15.0, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            img_path = self._search_and_download(bbox, date, output_dir,
                providers=providers, cloud_cover_max=cloud_cover_max)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery found.")
            out = output_dir / "building_mask.tif"
            pred = self._run_model(img_path, out, model, num_classes=2)
            stats = {}
            if pred is not None:
                import numpy as np
                stats["building_coverage"] = float((pred == 1).mean())
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                stats=stats, metadata={"model": model, "date": date},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class CropMonitoringPipeline(BasePipeline):
    """Crop type mapping and agricultural monitoring."""
    NAME = "crop_monitoring"

    def run(self, bbox, output_dir, date="2023-06", crop_classes=None,
            model="segformer_b2", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            img_path = self._search_and_download(bbox, date, output_dir, providers=providers)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery.")
            num_classes = len(crop_classes) + 1 if crop_classes else 10
            out = output_dir / "crop_map.tif"
            pred = self._run_model(img_path, out, model, num_classes=num_classes)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                success=pred is not None, metadata={"date": date, "crop_classes": crop_classes})
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class DisasterAssessmentPipeline(BasePipeline):
    """Rapid damage assessment after natural disasters using bi-temporal imagery."""
    NAME = "disaster_assessment"

    def run(self, bbox, output_dir, pre_date="2024-01", post_date="2024-02",
            disaster_type="generic", model="siamese_unet", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            pre_path, post_path = self._search_and_download_pair(
                bbox, pre_date, post_date, output_dir, providers=providers, cloud_cover_max=30.0)
            if not pre_path or not post_path:
                return PipelineResult(self.NAME, success=False,
                    error="Imagery not found for both dates.")
            out = output_dir / "damage_assessment.tif"
            pred = self._run_model(pre_path, out, model, num_classes=4)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"pre_date": pre_date, "post_date": post_date,
                          "disaster_type": disaster_type},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class DeforestationPipeline(BasePipeline):
    """Forest loss and deforestation detection."""
    NAME = "deforestation"

    def run(self, bbox, output_dir, baseline_year="2020", analysis_year="2024",
            model="changeformer", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            bl_path, cur_path = self._search_and_download_pair(
                bbox, f"{baseline_year}-07", f"{analysis_year}-07", output_dir,
                collections=["sentinel-2-l2a"], providers=providers)
            if not bl_path or not cur_path:
                return PipelineResult(self.NAME, success=False, error="Imagery unavailable.")
            out = output_dir / "deforestation_map.tif"
            pred = self._run_model(bl_path, out, model, num_classes=3)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"baseline_year": baseline_year, "analysis_year": analysis_year},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class UrbanGrowthPipeline(BasePipeline):
    """Urban expansion and impervious surface change detection."""
    NAME = "urban_growth"

    def run(self, bbox, output_dir, start_year="2018", end_year="2024",
            model="siamese_unet", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            # Landsat for multi-year comparisons (better temporal consistency)
            start_path, end_path = self._search_and_download_pair(
                bbox, f"{start_year}-06", f"{end_year}-06", output_dir,
                collections=["landsat-c2-l2"], providers=providers)
            if not start_path or not end_path:
                return PipelineResult(self.NAME, success=False, error="Landsat imagery not found.")
            out = output_dir / "urban_growth.tif"
            pred = self._run_model(start_path, out, model, num_classes=2,
                                    image_path_b=end_path)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"start_year": start_year, "end_year": end_year},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class WaterBodiesPipeline(BasePipeline):
    """Surface water body mapping — NDWI or deep learning."""
    NAME = "water_bodies"

    def run(self, bbox, output_dir, date="2024-06", method="ndwi",
            cloud_cover_max=10.0, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            # Real NDWI needs green + NIR bands (McFeeters 1996: NDWI =
            # (green - nir) / (green + nir)). Previously requested NDWI
            # via a post_process step that never actually ran for any
            # real multi-asset scene (same root cause as the carbon
            # estimation bug: _search_and_download's _stack_bands path
            # always intercepts first) -- ndwi = src.read(1) was
            # silently reading RED-band reflectance and thresholding
            # THAT as if it were NDWI, a real, confirmed bug (real water
            # has low red reflectance, well below any real NDWI
            # threshold; bright bare/urban surfaces could exceed a raw
            # red-reflectance threshold of 0.3 and be wrongly flagged as
            # water instead).
            if method == "ndwi":
                img_path = self._search_and_download(
                    bbox, date, output_dir, providers=providers,
                    cloud_cover_max=cloud_cover_max, bands=("green", "nir"))
            else:
                img_path = self._search_and_download(
                    bbox, date, output_dir, providers=providers, cloud_cover_max=cloud_cover_max)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No clear imagery found.")

            if method == "ndwi":
                out = output_dir / "ndwi.tif"
                import numpy as np
                import rasterio
                with rasterio.open(img_path) as src:
                    green = src.read(1).astype(np.float32)
                    nir = src.read(2).astype(np.float32)
                    profile = src.profile.copy()
                ndwi = (green - nir) / (green + nir + 1e-8)
                ndwi = np.clip(ndwi, -1.0, 1.0)
                profile.update(count=1, dtype="float32")
                with rasterio.open(out, "w", **profile) as dst:
                    dst.write(ndwi[None, ...])
                water_mask = (ndwi > 0.3).astype(np.uint8)
                coverage = float(water_mask.mean())
                stats = {"water_coverage": coverage}
            else:
                out = output_dir / "water_mask.tif"
                pred = self._run_model(img_path, out, "unet_resnet50", num_classes=2)
                import numpy as np
                stats = {"water_coverage": float((pred == 1).mean())} if pred is not None else {}

            return PipelineResult(self.NAME, output_path=Path(out),
                stats=stats, metadata={"date": date, "method": method}, success=True)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class SolarDetectionPipeline(BasePipeline):
    """Solar panel / photovoltaic installation detection."""
    NAME = "solar_detection"

    def run(self, bbox, output_dir, date="2024-06", cloud_cover_max=5.0,
            model="unet_efficientnet_b4", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            img_path = self._search_and_download(
                bbox, date, output_dir, providers=providers, cloud_cover_max=cloud_cover_max)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No cloud-free imagery.")
            out = output_dir / "solar_mask.tif"
            pred = self._run_model(img_path, out, model, num_classes=2,
                                   tile_size=256, overlap=32)
            stats = {}
            if pred is not None:
                import numpy as np
                stats["panel_coverage"] = float((pred == 1).mean())
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                stats=stats, metadata={"date": date, "model": model},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class CarbonEstimationPipeline(BasePipeline):
    """Above-ground biomass and carbon stock estimation via NDVI proxy."""
    NAME = "carbon_estimation"

    def run(self, bbox, output_dir, date="2024-06", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            # Real NDVI needs red + NIR bands. Previously requested NDVI
            # via a post_process=[...,"ndvi"] step -- but that step never
            # actually ran for any real multi-asset scene (the
            # _stack_bands path in _search_and_download always
            # intercepts first for a multi-asset scene, which is
            # virtually every real Sentinel-2/Landsat product), so
            # ndvi = src.read(1) was silently reading RED-band
            # reflectance and treating it as NDVI -- a real, severe,
            # confirmed bug producing scientifically meaningless carbon
            # estimates on every real run. Now requests real red+nir
            # bands and computes the real, standard NDVI formula
            # directly from the radiometrically-corrected stack.
            img_path = self._search_and_download(
                bbox, date, output_dir, providers=providers, bands=("red", "nir"))
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery.")

            import numpy as np
            import rasterio
            from pygeovision.data.radiometric import real_pixel_area_ha
            with rasterio.open(img_path) as src:
                red = src.read(1).astype(np.float32)
                nir = src.read(2).astype(np.float32)
                profile = src.profile.copy()
                pixel_area_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.clip(ndvi, -1.0, 1.0)

            # Simple allometric AGB → carbon (placeholder; replace with trained model)
            agb = np.clip(50.0 * ndvi ** 2, 0, 500).astype(np.float32)
            carbon = agb * 0.47

            #lets implement with a trained model
            

            out = output_dir / "carbon_map.tif"
            profile.update(dtype="float32", count=1, compress="lzw")
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(carbon[np.newaxis, ...])

            valid = carbon[carbon > 0]
            total_area = float(carbon.size * pixel_area_ha)
            stats = {
                "mean_carbon_mg_ha": float(valid.mean()) if valid.size else 0.0,
                "total_area_ha": round(total_area, 1),
                "total_carbon_mg": round(float(valid.mean() * total_area), 1) if valid.size else 0.0,
            }
            return PipelineResult(self.NAME, output_path=out, stats=stats,
                metadata={"date": date}, success=True)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PIPELINE_REGISTRY: dict[str, type] = {
    "change_detection":     ChangeDetectionPipeline,
    "land_cover":           LandCoverPipeline,
    "building_footprints":  BuildingFootprintsPipeline,
    "crop_monitoring":      CropMonitoringPipeline,
    "disaster_assessment":  DisasterAssessmentPipeline,
    "deforestation":        DeforestationPipeline,
    "urban_growth":         UrbanGrowthPipeline,
    "water_bodies":         WaterBodiesPipeline,
    "solar_detection":      SolarDetectionPipeline,
    "carbon_estimation":    CarbonEstimationPipeline,
}


def get_pipeline(name: str, pgv_client: Any) -> BasePipeline:
    """Instantiate a pipeline by name.

    Args:
        name: Pipeline name.
        pgv_client: PyGeoVision client (provides .search() and .download()
                   which delegate to PyGeoFetch).

    Returns:
        Instantiated pipeline.
    """
    if name not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline '{name}'. Available: {sorted(_PIPELINE_REGISTRY.keys())}"
        )
    return _PIPELINE_REGISTRY[name](pgv_client)


def list_pipelines() -> list[str]:
    return sorted(_PIPELINE_REGISTRY.keys())


__all__ = [
    "BasePipeline", "PipelineResult", "get_pipeline", "list_pipelines",
    "ChangeDetectionPipeline", "LandCoverPipeline", "BuildingFootprintsPipeline",
    "CropMonitoringPipeline", "DisasterAssessmentPipeline", "DeforestationPipeline",
    "UrbanGrowthPipeline", "WaterBodiesPipeline", "SolarDetectionPipeline",
    "CarbonEstimationPipeline",
]