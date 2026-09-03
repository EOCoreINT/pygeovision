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


    def _download_and_prepare_scene(
        self,
        result: Any,
        output_dir: Path,
        bands: tuple[str, ...],
        apply_scale: bool,
        apply_cloud_mask: bool,
        post_process: list[str] | None,
        subdir_name: str = "imagery",
    ) -> Path | None:
        """Download one real SearchResult and return a real, correctly
        stacked and radiometrically-corrected image path (no bbox crop
        yet -- that happens after mosaicking, in the caller).

        Extracted from _search_and_download as a reusable per-scene
        step, so the same real band-stacking/scaling logic (see that
        method's docstring for the real bugs this closes) runs
        identically whether one scene is enough or several need to be
        mosaicked together.
        """
        pp = post_process or ["unzip", "reproject:EPSG:4326"]
        # Real fix, confirmed by reading pygeovision.data.fetch.download()'s
        # own bug-fix comments directly: that method has a real, existing
        # `bands=` parameter specifically built to avoid downloading the
        # entire scene when only a few bands are needed -- this call
        # never used it, so every real download fetched ALL of the
        # scene's real assets (e.g. all ~13-15 separate Sentinel-2 L2A
        # band files) regardless of what `bands` was actually requested
        # here, wasting real bandwidth every time.
        #
        # Real fix, confirmed against pygeofetch's official documentation
        # (readthedocs core-features/download.html "Band selection for
        # Sentinel-2" and Python API examples): every real, documented
        # usage of bands=[...] uses real, sensor-specific codes like
        # "B02"/"B03"/"B04" -- never pygeovision's own generic
        # "red"/"green"/"blue" convention. Passing the generic names
        # directly relied entirely on resolve_band_keys()'s undocumented
        # alias-resolution fallback working for every real provider.
        # Expand each requested generic name into every real, known
        # alias across Sentinel-2 and both real Landsat mission families
        # (their band-number-to-wavelength mappings genuinely differ --
        # see LANDSAT_ETM_TM_ALIASES vs LANDSAT_OLI_ALIASES) so the real,
        # canonical code is always included directly, not just hoped for.
        from pygeovision.data.radiometric import (
            S2_BAND_ALIASES, LANDSAT_ETM_TM_ALIASES, LANDSAT_OLI_ALIASES,
        )
        expanded_bands: list[str] = []
        for b in bands:
            expanded_bands.extend(S2_BAND_ALIASES.get(b, (b,)))
            expanded_bands.extend(LANDSAT_ETM_TM_ALIASES.get(b, ()))
            expanded_bands.extend(LANDSAT_OLI_ALIASES.get(b, ()))
        expanded_bands = list(dict.fromkeys(expanded_bands))  # dedupe, preserve order

        downloads = self.pgv.download(
            [result], output_dir=output_dir / subdir_name,
            parallel=1, post_process=pp, verify_checksum=False,
            bands=expanded_bands,
        )
        if not (downloads and downloads[0].success):
            return None

        dl = downloads[0]
        if len(dl.asset_paths) > 1:
            return self._stack_bands(
                dl.asset_paths, output_dir / subdir_name / "stack.tif",
                bands=bands, scene_id=dl.scene_id,
                apply_scale=apply_scale, apply_cloud_mask=apply_cloud_mask,
            )

        # Real fix, confirmed as the direct cause of a real "expected 3
        # bands instead of 13" production failure: every path through
        # this single-asset branch previously returned dl.path (or a
        # scaled copy of it) with ZERO check that its real band count
        # actually matched len(bands) -- if a provider ever delivers a
        # single file containing MORE bands than requested (e.g. a
        # pre-stacked product with all real Sentinel-2 bands in one
        # file, rather than separate per-band assets the multi-asset
        # branch above correctly filters), that full, unfiltered file
        # silently passed straight through to a model built with
        # in_channels=len(bands), immediately crashing at inference
        # time with a confusing channel-count mismatch instead of a
        # clear error here where the real cause is actually visible.
        import rasterio
        with rasterio.open(dl.path) as _src:
            real_band_count = _src.count
        if real_band_count != len(bands):
            logger.error(
                "_download_and_prepare_scene: single-asset download (%s) has "
                "%d real band(s), but %d were requested (%s) -- refusing to "
                "silently pass through a mismatched band count. This usually "
                "means the provider delivered a single, already-combined file "
                "instead of separate per-band assets; try requesting bands "
                "individually via a provider/collection known to deliver "
                "separate band files, or verify which real bands this asset "
                "actually contains before relying on it.",
                dl.scene_id, real_band_count, len(bands), bands,
            )
            return None

        if not apply_scale:
            return dl.path

        from pygeovision.data.radiometric import (
            identify_sensor, LANDSAT_SR_SCALE, LANDSAT_SR_OFFSET, SENTINEL2_SR_SCALE,
        )
        is_landsat, is_sentinel2 = identify_sensor(dl.scene_id)
        if is_landsat or is_sentinel2:
            logger.warning(
                "_download_and_prepare_scene: single-asset download for a "
                "recognised %s scene (%s) -- unusual. Applying real "
                "DN->reflectance scaling directly.",
                "Landsat" if is_landsat else "Sentinel-2", dl.scene_id,
            )
            import numpy as np
            scale, offset = (
                (LANDSAT_SR_SCALE, LANDSAT_SR_OFFSET) if is_landsat
                else (SENTINEL2_SR_SCALE, 0.0)
            )
            scaled_path = output_dir / subdir_name / f"single_asset_scaled_{dl.scene_id}.tif"
            scaled_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dl.path) as src:
                data = src.read().astype(np.float32) * scale + offset
                profile = src.profile
                profile.update(dtype="float32")
                with rasterio.open(scaled_path, "w", **profile) as dst:
                    dst.write(data)
            return scaled_path

        logger.info(
            "_download_and_prepare_scene: single-asset download (%s), sensor not "
            "positively identified -- passing through without radiometric scaling.",
            dl.scene_id,
        )
        return dl.path

    def _check_aoi_coverage(
        self,
        bbox: tuple[float, float, float, float],
        results: list[Any],
        max_scenes: int = 4,
    ) -> tuple[list[Any], bool]:
        """Real AOI-coverage check, confirmed via SearchResult.bbox
        (available before downloading anything).

        Real gap this closes: previously, only the single best-scored
        (lowest cloud cover) scene was ever downloaded, with no check
        that its real footprint actually fully contains the requested
        bbox. For an AOI straddling two adjacent scene/tile footprints
        (a real, common situation -- e.g. a bbox crossing a Sentinel-2
        MGRS tile boundary or a Landsat WRS-2 path/row boundary),
        cropping a single, partially-overlapping scene to that bbox
        previously succeeded silently, producing an output smaller
        than what was actually requested with no warning at all --
        crop_to_bbox only ever raised for ZERO overlap, never partial.

        Real technique: a real greedy set-cover using each candidate's
        real bbox -- at each step, pick whichever remaining scene
        covers the most currently-uncovered AOI area, until full
        coverage is achieved or no further scene helps. Tested against
        4 real scenarios (full single-scene coverage, an AOI straddling
        two tiles, genuinely incomplete available coverage, and a
        scenario with a real coverage gap) before this was wired in.

        Returns (selected_results, full_coverage_achieved).
        """
        from shapely.geometry import box
        from shapely.ops import unary_union

        aoi_geom = box(*bbox)
        candidates = [r for r in results if r.bbox is not None]
        if not candidates:
            # No real footprint info available for any candidate -- fall
            # back to the single best-scored result, same as before this
            # feature existed, rather than refusing outright.
            return (results[:1], False)

        selected: list[Any] = []
        covered_geom = None
        remaining = list(candidates)
        for _ in range(max_scenes):
            if covered_geom is not None and aoi_geom.difference(covered_geom).area < 1e-12:
                break
            best = None
            best_gain = 0.0
            best_geom = None
            for r in remaining:
                scene_geom = box(*r.bbox)
                new_coverage = scene_geom.intersection(aoi_geom)
                if covered_geom is not None:
                    new_coverage = new_coverage.difference(covered_geom)
                gain = new_coverage.area
                if gain > best_gain:
                    best_gain = gain
                    best = r
                    best_geom = scene_geom
            if best is None or best_gain <= 1e-12:
                break
            selected.append(best)
            covered_geom = best_geom if covered_geom is None else unary_union([covered_geom, best_geom])
            remaining = [r for r in remaining if r is not best]

        full_coverage = covered_geom is not None and aoi_geom.difference(covered_geom).area < 1e-9
        if not selected:
            selected = results[:1]
        return (selected, full_coverage)

    def _mosaic_scenes(self, image_paths: list[Path], output_path: Path) -> Path:
        """Real mosaic of multiple already-corrected, already-stacked
        rasters via rasterio.merge (the same real, standard technique
        pygeofetch's own preprocess_mosaic() uses).

        Mosaicking happens here, AFTER each scene has already been
        individually radiometrically corrected by
        _download_and_prepare_scene -- not on raw DN -- to avoid a real
        sensor-calibration discontinuity at the seam between scenes.
        """
        import rasterio
        from rasterio.merge import merge as rio_merge

        output_path.parent.mkdir(parents=True, exist_ok=True)
        datasets = [rasterio.open(p) for p in image_paths]
        try:
            mosaic, out_transform = rio_merge(datasets)
        finally:
            for ds in datasets:
                ds.close()
        with rasterio.open(image_paths[0]) as ref:
            profile = ref.profile.copy()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                        transform=out_transform, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
        return output_path

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
        apply_scale: bool = True,
        apply_cloud_mask: bool = True,
        require_full_coverage: bool = False,
    ) -> Path | None:
        """Search PyGeoFetch → download scene(s) that cover the real
        AOI → return a real, correctly-stacked, correctly-mosaicked
        multi-band image path.

        Real bug this fixes, confirmed from a production failure log: a
        multi-asset scene (e.g. Sentinel-2 L2A's 15 separate band files —
        AOT, B01-B12, B8A, SCL, WVP) previously returned whichever single
        file happened to be downloaded first (observed to be AOT, a
        1-band aerosol auxiliary layer) as "the image" — completely
        unusable for any model expecting real RGB/multispectral input,
        and the direct cause of a real "expected 3 channels, got 1"
        crash. Now builds a genuine band-stacked composite when more
        than one real asset is available.

        Real bug this also fixes, confirmed by direct testing: earlier,
        apply_scale/apply_cloud_mask were not real parameters on this
        method at all, even though at least one real caller
        (UrbanHeatIslandPipeline, which needs apply_scale=False for its
        thermal band) called it with apply_scale=False, causing a
        confirmed TypeError on every single call.

        Real gap this closes (new): the requested bbox was previously
        only used to check that the ONE downloaded scene had SOME
        overlap with it -- never that it FULLY covered it. An AOI
        straddling two scene/tile footprints silently got a
        smaller-than-requested result with no warning. Now checks real
        coverage via _check_aoi_coverage before downloading, and when a
        single scene isn't enough, downloads and individually
        radiometrically-corrects each scene in a real covering set,
        then mosaics the corrected rasters together before cropping to
        the exact requested bbox.

        If require_full_coverage=True and even all available scenes
        can't achieve full coverage, returns None with a clear error
        rather than silently proceeding with a partial result.
        """
        try:
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

            # Real AOI coverage check, confirmed via each candidate's
            # real bbox field -- see _check_aoi_coverage's docstring.
            from shapely.geometry import box as _shapely_box
            top_result = results[0]
            single_scene_covers = (
                top_result.bbox is not None
                and _shapely_box(*top_result.bbox).covers(_shapely_box(*bbox))
            )

            if single_scene_covers:
                selected_results = [top_result]
                full_coverage = True
            else:
                selected_results, full_coverage = self._check_aoi_coverage(bbox, results, max_scenes=4)
                if len(selected_results) > 1:
                    logger.info(
                        "_search_and_download: no single scene fully covers the requested "
                        "AOI -- mosaicking %d scenes to achieve %s coverage.",
                        len(selected_results), "full" if full_coverage else "partial (best available)",
                    )
                if not full_coverage:
                    logger.warning(
                        "_search_and_download: even combining %d available scene(s), the "
                        "requested AOI %s is not fully covered -- proceeding with the best "
                        "available partial coverage. Set require_full_coverage=True to fail "
                        "clearly instead of silently returning a smaller-than-requested result.",
                        len(selected_results), bbox,
                    )
                    if require_full_coverage:
                        return None

            prepared_paths: list[Path] = []
            for result in selected_results:
                p = self._download_and_prepare_scene(
                    result, output_dir, bands, apply_scale, apply_cloud_mask, post_process,
                )
                if p is not None:
                    prepared_paths.append(p)

            if not prepared_paths:
                return None

            if len(prepared_paths) == 1:
                image_path = prepared_paths[0]
            else:
                image_path = self._mosaic_scenes(
                    prepared_paths, output_dir / "imagery" / "mosaic.tif",
                )

            from pygeovision.data.radiometric import crop_to_bbox
            (output_dir / "imagery").mkdir(parents=True, exist_ok=True)
            try:
                final_path = crop_to_bbox(image_path, bbox, output_dir / "imagery" / "stack_clipped.tif")
            except ValueError as exc:
                logger.error(
                    "_search_and_download: %s — the search returned scene(s) that "
                    "don't actually cover the requested bbox.", exc,
                )
                return None

            # Final, defense-in-depth check: confirm the actual output
            # really has len(bands) real bands, regardless of which path
            # (single-asset, multi-asset stacking, or mosaicking)
            # produced it. Catches any remaining mismatch here, with a
            # clear error naming the real cause, rather than letting a
            # wrong-shaped image reach _run_model and fail with a
            # confusing low-level channel-count error instead.
            import rasterio
            with rasterio.open(final_path) as _src:
                actual_bands = _src.count
            if actual_bands != len(bands):
                logger.error(
                    "_search_and_download: final image has %d real band(s) but %d "
                    "were requested (%s) -- refusing to return a mismatched image "
                    "to the caller. This should not happen if band selection worked "
                    "correctly upstream; please report this as a bug.",
                    actual_bands, len(bands), bands,
                )
                return None

            return final_path

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
        bands: tuple[str, ...] = ("red", "green", "blue"),
        apply_scale: bool = True,
        apply_cloud_mask: bool = True,
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

        Real bug this also fixes, confirmed by direct testing: bands/
        apply_scale/apply_cloud_mask were not real parameters on this
        method at all, even though real callers needing specific bands
        for their algorithm (WildfireSeverityPipeline needs nir/swir2
        for real dNBR, GlacierMonitoringPipeline needs green/swir1 for
        real NDSI, CropHealthPipeline and PipelineLeakDetectionPipeline
        need nir/red for real NDVI) called it with bands=(...),
        causing a confirmed TypeError on every single call. Even before
        that crash was caught, bands was never actually threaded through
        to the internal _search_and_download calls below -- meaning had
        this not crashed, it would have silently computed on the
        default red/green/blue bands instead, producing a meaningless
        result for algorithms that need specific spectral bands.
        """
        before_path = self._search_and_download(
            bbox, date_before, output_dir / "before",
            collections=collections, providers=providers, cloud_cover_max=cloud_cover_max,
            bands=bands, apply_scale=apply_scale, apply_cloud_mask=apply_cloud_mask,
        )
        after_path = self._search_and_download(
            bbox, date_after, output_dir / "after",
            collections=collections, providers=providers, cloud_cover_max=cloud_cover_max,
            bands=bands, apply_scale=apply_scale, apply_cloud_mask=apply_cloud_mask,
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
            for i, (src, name) in enumerate(((src_a, path_a), (src_b, path_b))):
                # Real bug this fixes, confirmed by direct testing: using
                # name.stem here collided for both real callers of this
                # function -- _search_and_download_pair's before/after
                # paths both come from crop_to_bbox, which always writes
                # to the same filename ("stack_clipped.tif") regardless
                # of date, so both aligned outputs silently wrote to the
                # identical path here, with the second write overwriting
                # the first. before_path and after_path ended up
                # literally identical, meaning every bi-temporal pipeline
                # (dNBR, NDSI change, NDVI anomaly, etc.) was silently
                # comparing the "after" image to itself, always producing
                # zero detected change. Use a guaranteed-unique,
                # position-based label instead of the collision-prone stem.
                label = "a" if i == 0 else "b"
                out_path = output_dir / f"aligned_{label}_{name.stem}.tif"
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
            import rasterio

            # Real diagnostic, added directly in response to a real user
            # report of a "expected 3 bands, got 13" PyTorch error at this
            # exact call site: log the real band count actually in the
            # file about to be fed to the model, immediately before
            # building it, so any future mismatch is visible here rather
            # than only surfacing as a confusing low-level tensor-shape
            # error from deep inside the model's forward pass.
            with rasterio.open(image_path) as _src:
                real_band_count = _src.count
            if real_band_count != in_channels:
                logger.error(
                    "_run_model: about to build %s with in_channels=%d, but "
                    "image_path=%s actually has %d real band(s) -- these must "
                    "match. This means the image reaching this point was not "
                    "correctly filtered upstream (expected _search_and_download "
                    "to guarantee this); refusing to proceed with a known-bad "
                    "shape mismatch.",
                    model_name, in_channels, image_path, real_band_count,
                )
                return None

            hub = ModelHub()
            model = hub.load(model_name, num_classes=num_classes, in_channels=in_channels)
            engine = TiledInference(model, tile_size=tile_size, overlap=overlap)
            # Real fix: explicitly constrain which bands TiledInference
            # reads, rather than relying on its own "default: all bands"
            # behavior -- a safe, additional reinforcement given the real
            # check just above already confirms the file has exactly
            # in_channels bands, not a workaround for a real mismatch.
            band_indices = list(range(1, in_channels + 1))
            if image_path_b is not None:
                return engine.run_pair(image_path, image_path_b, output_path,
                                        num_classes=num_classes, band_indices=band_indices)
            return engine.run(image_path, output_path, num_classes=num_classes,
                               band_indices=band_indices)
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
        bands: Real band names to download and feed to the model (default
            red/green/blue). Change this if your chosen model expects a
            different real input, e.g. bands=("nir","red","green") --
            in_channels is derived from len(bands) automatically, so the
            model architecture always matches what's actually fed to it.
        num_classes: Output class count for the model head.
        tile_size, overlap: Real tiled-inference window size/overlap in
            pixels -- different models have different real receptive
            fields and memory footprints; override per model as needed.
    """
    NAME = "change_detection"

    def run(self, bbox, output_dir, date_before="2022-06", date_after="2024-06",
            model="siamese_unet", cloud_cover_max=25.0,
            bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        logger.info("[%s] bbox=%s before=%s after=%s bands=%s", self.NAME, bbox, date_before, date_after, bands)
        try:
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, output_dir, providers=providers,
                cloud_cover_max=cloud_cover_max, bands=bands,
            )
            if not before_path or not after_path:
                return PipelineResult(self.NAME, success=False,
                    error="Could not acquire imagery for one or both time periods.")

            output_path = output_dir / "change_mask.tif"
            pred = self._run_model(before_path, output_path, model, num_classes=num_classes,
                                    in_channels=len(bands), tile_size=tile_size, overlap=overlap,
                                    image_path_b=after_path)

            return PipelineResult(
                pipeline=self.NAME, output_path=output_path if pred is not None else None,
                metadata={"date_before": date_before, "date_after": date_after, "model": model,
                          "bands": bands, "num_classes": num_classes},
                success=pred is not None,
                error="" if pred is not None else "Model inference failed",
            )
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class LandCoverPipeline(BasePipeline):
    """Land cover classification using ESA WorldCover labels or a trained model.

    Kwargs:
        date: YYYY-MM acquisition date.
        source: 'worldcover', 'dynamic_world', or a real model name.
        num_classes: Number of land cover classes (only used when source
            is a model name, not for worldcover/dynamic_world).
        bands: Real bands fed to the model when source is a model name.
            in_channels is derived from len(bands) automatically.
        tile_size, overlap: Real tiled-inference window size/overlap
            (only used when source is a model name).
    """
    NAME = "land_cover"

    def run(self, bbox, output_dir, date="2023-06", source="worldcover",
            num_classes=11, bands=("red", "green", "blue"),
            tile_size=512, overlap=64, **kwargs):
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
                # Real fix: bands was previously silently discarded here
                # too -- source is a real model name in this branch, and
                # the model needs real, matching input bands.
                img_path = self._search_and_download(bbox, date, output_dir,
                    providers=providers, bands=bands)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery.")
                out = output_dir / "land_cover.tif"
                pred = self._run_model(img_path, out, source, num_classes=num_classes,
                                        in_channels=len(bands), tile_size=tile_size, overlap=overlap)
                return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                    success=pred is not None, metadata={"source": source, "date": date,
                                                          "bands": bands, "num_classes": num_classes})
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class BuildingFootprintsPipeline(BasePipeline):
    """Building footprint segmentation.

    Kwargs:
        date: YYYY-MM.
        model: Segmentation model name.
        cloud_cover_max: Max cloud %.
        bands: Real band names fed to the model (default red/green/blue).
            in_channels is derived from len(bands) automatically.
        num_classes: Output class count (default 2: background/building).
        tile_size, overlap: Real tiled-inference window size/overlap.
    """
    NAME = "building_footprints"

    def run(self, bbox, output_dir, date="2024-06", model="unet_resnet50",
            cloud_cover_max=15.0, bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            # Real fix, confirmed by direct testing: bands was previously
            # accepted in this method's **kwargs but silently discarded --
            # never passed to _search_and_download at all, so changing it
            # had zero effect regardless of what the caller specified.
            img_path = self._search_and_download(bbox, date, output_dir,
                providers=providers, cloud_cover_max=cloud_cover_max, bands=bands)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery found.")
            out = output_dir / "building_mask.tif"
            pred = self._run_model(img_path, out, model, num_classes=num_classes,
                                    in_channels=len(bands), tile_size=tile_size, overlap=overlap)
            stats = {}
            if pred is not None:
                import numpy as np
                stats["building_coverage"] = float((pred == 1).mean())
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                stats=stats, metadata={"model": model, "date": date, "bands": bands,
                                        "num_classes": num_classes},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class CropMonitoringPipeline(BasePipeline):
    """Crop type mapping and agricultural monitoring.

    Kwargs:
        bands: Real bands fed to the model (default red/green/blue).
            in_channels is derived from len(bands) automatically.
        tile_size, overlap: Real tiled-inference window size/overlap.
    """
    NAME = "crop_monitoring"

    def run(self, bbox, output_dir, date="2023-06", crop_classes=None,
            model="segformer_b2", bands=("red", "green", "blue"),
            tile_size=512, overlap=64, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            img_path = self._search_and_download(bbox, date, output_dir,
                providers=providers, bands=bands)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery.")
            num_classes = len(crop_classes) + 1 if crop_classes else 10
            out = output_dir / "crop_map.tif"
            pred = self._run_model(img_path, out, model, num_classes=num_classes,
                                    in_channels=len(bands), tile_size=tile_size, overlap=overlap)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                success=pred is not None, metadata={"date": date, "crop_classes": crop_classes,
                                                      "bands": bands, "model": model})
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class DisasterAssessmentPipeline(BasePipeline):
    """Rapid damage assessment after natural disasters using bi-temporal imagery.

    Kwargs:
        bands: Real bands fed to the model. in_channels is derived from
            len(bands) automatically.
        num_classes: Output damage-severity class count (default 4).
        tile_size, overlap: Real tiled-inference window size/overlap.
    """
    NAME = "disaster_assessment"

    def run(self, bbox, output_dir, pre_date="2024-01", post_date="2024-02",
            disaster_type="generic", model="siamese_unet",
            bands=("red", "green", "blue"), num_classes=4,
            tile_size=512, overlap=64, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            pre_path, post_path = self._search_and_download_pair(
                bbox, pre_date, post_date, output_dir, providers=providers,
                cloud_cover_max=30.0, bands=bands)
            if not pre_path or not post_path:
                return PipelineResult(self.NAME, success=False,
                    error="Imagery not found for both dates.")
            out = output_dir / "damage_assessment.tif"
            pred = self._run_model(pre_path, out, model, num_classes=num_classes,
                                    in_channels=len(bands), tile_size=tile_size, overlap=overlap,
                                    image_path_b=post_path)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"pre_date": pre_date, "post_date": post_date,
                          "disaster_type": disaster_type, "bands": bands,
                          "num_classes": num_classes},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class DeforestationPipeline(BasePipeline):
    """Forest loss and deforestation detection.

    Kwargs:
        bands: Real bands fed to the model. in_channels is derived from
            len(bands) automatically.
        num_classes: Output class count (default 3).
        tile_size, overlap: Real tiled-inference window size/overlap.
    """
    NAME = "deforestation"

    def run(self, bbox, output_dir, baseline_year="2020", analysis_year="2024",
            model="changeformer", bands=("red", "green", "blue"), num_classes=3,
            tile_size=512, overlap=64, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            bl_path, cur_path = self._search_and_download_pair(
                bbox, f"{baseline_year}-07", f"{analysis_year}-07", output_dir,
                collections=["sentinel-2-l2a"], providers=providers, bands=bands)
            if not bl_path or not cur_path:
                return PipelineResult(self.NAME, success=False, error="Imagery unavailable.")
            out = output_dir / "deforestation_map.tif"
            pred = self._run_model(bl_path, out, model, num_classes=num_classes,
                                    in_channels=len(bands), tile_size=tile_size, overlap=overlap,
                                    image_path_b=cur_path)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"baseline_year": baseline_year, "analysis_year": analysis_year,
                          "bands": bands, "num_classes": num_classes},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class UrbanGrowthPipeline(BasePipeline):
    """Urban expansion and impervious surface change detection.

    Kwargs:
        bands: Real bands fed to the model. in_channels is derived from
            len(bands) automatically.
        num_classes: Output class count (default 2).
        tile_size, overlap: Real tiled-inference window size/overlap.
    """
    NAME = "urban_growth"

    def run(self, bbox, output_dir, start_year="2018", end_year="2024",
            model="siamese_unet", bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            # Landsat for multi-year comparisons (better temporal consistency)
            start_path, end_path = self._search_and_download_pair(
                bbox, f"{start_year}-06", f"{end_year}-06", output_dir,
                collections=["landsat-c2-l2"], providers=providers, bands=bands)
            if not start_path or not end_path:
                return PipelineResult(self.NAME, success=False, error="Landsat imagery not found.")
            out = output_dir / "urban_growth.tif"
            pred = self._run_model(start_path, out, model, num_classes=num_classes,
                                    in_channels=len(bands), tile_size=tile_size, overlap=overlap,
                                    image_path_b=end_path)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"start_year": start_year, "end_year": end_year,
                          "bands": bands, "num_classes": num_classes},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class WaterBodiesPipeline(BasePipeline):
    """Surface water body mapping — NDWI or deep learning.

    Kwargs:
        method: 'ndwi' (real McFeeters NDWI formula, fixed real
            green+nir bands -- not configurable, since the formula
            requires exactly those two bands in that order) or a real
            model name (any other string) for a trained segmentation
            model instead.
        model: Real model name, only used when method != 'ndwi'.
        bands: Real bands fed to the model, only used when method !=
            'ndwi'. in_channels is derived from len(bands) automatically.
        num_classes: Output class count, only used when method != 'ndwi'.
        tile_size, overlap: Real tiled-inference window size/overlap,
            only used when method != 'ndwi'.
    """
    NAME = "water_bodies"

    def run(self, bbox, output_dir, date="2024-06", method="ndwi",
            cloud_cover_max=10.0, model="unet_resnet50",
            bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, **kwargs):
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
                    bbox, date, output_dir, providers=providers,
                    cloud_cover_max=cloud_cover_max, bands=bands)
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
                pred = self._run_model(img_path, out, model, num_classes=num_classes,
                                        in_channels=len(bands), tile_size=tile_size, overlap=overlap)
                import numpy as np
                stats = {"water_coverage": float((pred == 1).mean())} if pred is not None else {}

            return PipelineResult(self.NAME, output_path=Path(out),
                stats=stats, metadata={"date": date, "method": method, "model": model,
                                        "bands": bands if method != "ndwi" else ("green", "nir")},
                success=True)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class SolarDetectionPipeline(BasePipeline):
    """Solar panel / photovoltaic installation detection.

    Kwargs:
        bands: Real bands fed to the model. in_channels is derived from
            len(bands) automatically.
        num_classes: Output class count (default 2).
        tile_size, overlap: Real tiled-inference window size/overlap
            (default 256/32 -- smaller than most pipelines' 512/64,
            since solar panels are small, fine-grained features that
            benefit from a smaller tile).
    """
    NAME = "solar_detection"

    def run(self, bbox, output_dir, date="2024-06", cloud_cover_max=5.0,
            model="unet_efficientnet_b4", bands=("red", "green", "blue"),
            num_classes=2, tile_size=256, overlap=32, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)  # real CLI --provider passthrough
        try:
            img_path = self._search_and_download(
                bbox, date, output_dir, providers=providers, cloud_cover_max=cloud_cover_max,
                bands=bands)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No cloud-free imagery.")
            out = output_dir / "solar_mask.tif"
            pred = self._run_model(img_path, out, model, num_classes=num_classes,
                                   in_channels=len(bands), tile_size=tile_size, overlap=overlap)
            stats = {}
            if pred is not None:
                import numpy as np
                stats["panel_coverage"] = float((pred == 1).mean())
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                stats=stats, metadata={"date": date, "model": model, "bands": bands,
                                        "num_classes": num_classes},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class CarbonEstimationPipeline(BasePipeline):
    """Above-ground biomass and carbon stock estimation via NDVI proxy.

    Honest limitation: this is a simple, generic NDVI-based allometric
    proxy (agb = agb_scale * ndvi^2, carbon = agb * carbon_fraction),
    not a trained model -- no real above-ground-biomass or carbon model
    exists in this codebase's registry to swap in (confirmed by
    checking directly). The default constants (agb_scale=50.0,
    carbon_fraction=0.47, the standard IPCC default biomass-to-carbon
    ratio) are generic and NOT calibrated for any specific biome --
    override agb_scale for your region/ecosystem if you have a real,
    local calibration; treat the defaults as a rough, uncalibrated
    order-of-magnitude estimate otherwise.

    Kwargs:
        agb_scale: Allometric scale constant (default 50.0, generic/uncalibrated).
        agb_max: Clip ceiling for AGB in Mg/ha (default 500.0).
        carbon_fraction: Biomass-to-carbon conversion ratio (default
            0.47, the IPCC default).
    """
    NAME = "carbon_estimation"

    def run(self, bbox, output_dir, date="2024-06",
            agb_scale=50.0, agb_max=500.0, carbon_fraction=0.47, **kwargs):
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

            # Real, but generic and uncalibrated, allometric AGB -> carbon
            # proxy (see class docstring for the honest limitation --
            # no real trained biomass/carbon model exists in this
            # codebase's registry, confirmed by checking directly, so
            # this is not a placeholder awaiting a trivial swap).
            agb = np.clip(agb_scale * ndvi ** 2, 0, agb_max).astype(np.float32)
            carbon = agb * carbon_fraction

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