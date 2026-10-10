"""
PyGeoVision geospatial pipelines.

A pipeline takes a bounding box and returns a finished raster plus
statistics: imagery comes from PyGeoFetch, the analysis is either a
published index/algorithm or a model you trained yourself.

Every pipeline is one of two kinds, and says so in ``kind``:

``"ready"``
    Computed directly from imagery or from a published reference product.
    Runs out of the box (given data access).
``"needs_weights"``
    Runs a neural network. PyGeoVision does not ship task weights, and an
    untrained network produces random output, so these refuse to run
    unless you pass ``checkpoint=`` (a ``.pth`` you trained with
    ``GeoTrainer``). They never fall back to random weights.

Use :func:`list_pipelines` / :func:`pipeline_kinds` to see the catalogue.

Example:
    >>> import pygeovision as pgv
    >>> client = pgv.PyGeoVision()
    >>> result = client.pipeline("water_bodies",
    ...     bbox=(-0.15, 51.47, -0.10, 51.52), date="2024-06")
    >>> print(result.output_path, result.stats)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result of one pipeline run (used by every pipeline in the catalogue)."""
    pipeline: str
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""
    duration_seconds: float = 0.0

    @property
    def name(self) -> str:
        """Alias of ``pipeline`` (kept so ``result.name`` also works)."""
        return self.pipeline

    def __str__(self) -> str:
        if self.success:
            return f"PipelineResult({self.pipeline}, output={self.output_path})"
        return f"PipelineResult({self.pipeline}, FAILED: {self.error})"


class ModelWeightsRequired(RuntimeError):
    """Raised instead of running a network that has no trained weights."""



# ---------------------------------------------------------------------------
# Base pipeline
# ---------------------------------------------------------------------------

class BasePipeline(ABC):
    """Abstract base for all PyGeoVision pipelines.

    ``pgv_client`` is a PyGeoVision instance — call
    ``pgv_client.search()`` and ``pgv_client.download()`` for data,
    which delegate to PyGeoFetch under the hood.
    """

    #: Registry name, set on every concrete pipeline.
    name: str = "base"
    #: One-line description shown in the catalogue.
    description: str = ""
    #: Topic area, used to group the documentation.
    domain: str = "other"
    #: ``"ready"`` (runs out of the box) or ``"needs_weights"`` (needs ``checkpoint=``).
    kind: str = "ready"
    tags: list[str] = []

    def __init__(self, pgv_client: Any) -> None:
        self.pgv = pgv_client  # PyGeoVision instance

    @property
    def NAME(self) -> str:  # noqa: N802 - kept for backward compatibility
        return self.name

    @abstractmethod
    def run(
        self,
        bbox: tuple[float, float, float, float],
        output_dir: str | Path,
        **kwargs: Any,
    ) -> PipelineResult: ...

    # ------------------------------------------------------------------
    # Shared result helpers
    # ------------------------------------------------------------------

    def _ok(self, output_path: Any, stats: dict[str, Any] | None = None,
            metadata: dict[str, Any] | None = None, t0: float | None = None) -> PipelineResult:
        import time
        return PipelineResult(
            pipeline=self.name, output_path=Path(output_path) if output_path else None,
            stats=stats or {}, metadata=metadata or {}, success=True,
            duration_seconds=(time.time() - t0) if t0 is not None else 0.0,
        )

    def _fail(self, error: str, t0: float | None = None) -> PipelineResult:
        import time
        return PipelineResult(
            pipeline=self.name, success=False, error=error,
            duration_seconds=(time.time() - t0) if t0 is not None else 0.0,
        )

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

    # ------------------------------------------------------------------
    # Neural-network inference -- only ever with user-supplied weights
    # ------------------------------------------------------------------

    def _load_trained_model(
        self, model_name: str, checkpoint: str | Path | None,
        num_classes: int, in_channels: int,
    ) -> Any:
        """Build ``model_name`` and load ``checkpoint`` into it, strictly.

        This is the only way a pipeline obtains a network. Without a
        checkpoint it raises :class:`ModelWeightsRequired` instead of
        building the architecture with random (or ImageNet-only) weights,
        because that would return a prediction that looks like a result
        and is not one. The load is ``strict=True`` so a checkpoint that
        does not match the architecture / class count / band count fails
        loudly rather than silently loading nothing.
        """
        if not checkpoint:
            raise ModelWeightsRequired(
                f"'{self.name}' runs a neural network and PyGeoVision does not ship trained "
                f"weights for it -- an untrained network returns random output. Pass "
                f"checkpoint='path/to/model.pth', trained with "
                f"pygeovision.ai.training.trainer.GeoTrainer on labelled chips of your own "
                f"area. It must match model={model_name!r}, num_classes={num_classes} and "
                f"{in_channels} input band(s)."
            )
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        import torch
        from pygeovision.ai.models.hub import ModelHub

        model = ModelHub().load(
            model_name, num_classes=num_classes, in_channels=in_channels, pretrained=False,
        )
        if getattr(model, "_pygeovision_fallback_used", False):
            raise RuntimeError(
                f"model {model_name!r} could not be built (the hub substituted "
                f"{getattr(model, '_pygeovision_actual_name', '?')!r}); refusing to load "
                f"weights into a different architecture."
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model", "network"):
                if key in state:
                    state = state[key]
                    break
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint {ckpt_path.name} does not match model {model_name!r} "
                f"(num_classes={num_classes}, in_channels={in_channels}): {str(exc)[:300]}"
            ) from exc
        model.eval()
        return model

    def _run_model(
        self,
        image_path: Path,
        output_path: Path,
        model: Any,
        num_classes: int,
        in_channels: int = 3,
        tile_size: int = 512,
        overlap: int = 64,
        image_path_b: Path | None = None,
        checkpoint: str | Path | None = None,
    ) -> Any:
        """Tiled inference with a trained network; returns the class map.

        ``model`` is a network already built by :meth:`_load_trained_model`,
        or a registry name together with ``checkpoint``. Raises (never
        returns None) on any failure, with the reason in the message.
        Pass ``image_path_b`` for two-image (siamese) models.
        """
        import rasterio

        from pygeovision.ai.inference.tiled_inference import TiledInference

        if isinstance(model, str):
            model = self._load_trained_model(model, checkpoint, num_classes, in_channels)
        with rasterio.open(image_path) as src:
            real_band_count = src.count
        if real_band_count != in_channels:
            raise ValueError(
                f"{image_path} has {real_band_count} band(s) but the model expects "
                f"{in_channels}; refusing to run with a mismatched band count."
            )
        engine = TiledInference(model, tile_size=tile_size, overlap=overlap)
        band_indices = list(range(1, in_channels + 1))
        if image_path_b is not None:
            return engine.run_pair(image_path, image_path_b, output_path,
                                   num_classes=num_classes, band_indices=band_indices)
        return engine.run(image_path, output_path, num_classes=num_classes,
                          band_indices=band_indices)

    # ------------------------------------------------------------------
    # A grid for products that need no imagery
    # ------------------------------------------------------------------

    @staticmethod
    def _template_raster(
        bbox: tuple[float, float, float, float], path: Path,
        resolution_m: float = 10.0, max_pixels: int = 100_000_000,
    ) -> Path:
        """Write an empty ~10 m EPSG:4326 raster covering ``bbox``.

        Reference products (WorldCover) only need a target grid, not an
        image, so they should not fail just because no scene was found.
        """
        import math

        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        west, south, east, north = bbox
        lat0 = (south + north) / 2.0
        dy = resolution_m / 110_574.0
        dx = resolution_m / (111_320.0 * max(math.cos(math.radians(lat0)), 1e-6))
        width = max(1, math.ceil((east - west) / dx))
        height = max(1, math.ceil((north - south) / dy))
        if width * height > max_pixels:
            raise ValueError(
                f"bbox is too large for one {resolution_m:g} m raster "
                f"({width}x{height} px > {max_pixels}); split it into tiles."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        transform = from_origin(west, north, (east - west) / width, (north - south) / height)
        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform, compress="lzw",
        ) as dst:
            dst.write(np.zeros((1, height, width), dtype=np.uint8))
        return path


def _month(value: Any, default_month: str = "07") -> str:
    """'2020' -> '2020-07'; 'YYYY-MM' / 'YYYY-MM-DD' unchanged."""
    s = str(value)
    return f"{s}-{default_month}" if len(s) == 4 else s


def _otsu_threshold(values: Any) -> float:
    """Otsu's threshold of a 1-D sample (256-bin histogram)."""
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or v.max() == v.min():
        return float(v.max()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0 * w0[-1]) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1.0
    return float(centers[int(np.argmax(between))])


# ---------------------------------------------------------------------------
# Pipelines that run out of the box
# ---------------------------------------------------------------------------

class LandCoverPipeline(BasePipeline):
    """Land cover for an area of interest.

    ``source="worldcover"`` (default) returns the **ESA WorldCover 2021
    10 m reference map** clipped to the bounding box. It is a published,
    validated product -- not a classification of a new image -- so it
    describes 2021 whatever ``date`` you pass (``date`` is ignored for this
    source, and the result says so). Do not use it to measure change.

    ``source="dynamic_world"`` labels the area from Google Dynamic World
    for the month in ``date`` (needs Earth Engine access).

    Any other ``source`` is treated as a model name and needs
    ``checkpoint=`` (your trained weights).
    """
    name = "land_cover"
    description = "ESA WorldCover 2021 reference map (or Dynamic World / your own model) for a bounding box"
    domain = "land"
    kind = "ready"
    tags = ["land", "reference-product"]

    def run(self, bbox, output_dir, date="2023-06", source="worldcover",
            num_classes=11, bands=("red", "green", "blue"),
            tile_size=512, overlap=64, checkpoint=None, **kwargs):
        import time
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        logger.info("[%s] source=%s date=%s", self.name, source, date)
        try:
            import rasterio

            from pygeovision.ai.data.dataset import TileMetadata

            def _meta(path):
                with rasterio.open(path) as src:
                    return TileMetadata(
                        tile_id=Path(path).stem, source_file=Path(path),
                        bounds=tuple(src.bounds), crs=str(src.crs), transform=list(src.transform)[:6],
                        row_off=0, col_off=0, height=src.height, width=src.width,
                        bands=list(range(1, src.count + 1)),
                    )

            out = output_dir / "land_cover.tif"
            if source == "worldcover":
                from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
                grid = self._template_raster(bbox, output_dir / "grid_template.tif")
                r = ESAWorldCoverLabeler(year=2021).label_tile(grid, _meta(grid), out)
                if not r.success:
                    return self._fail(r.error or "WorldCover labelling failed", t0)
                return self._ok(out, r.class_distribution or {},
                                {"source": source, "worldcover_year": 2021, "date_used": False,
                                 "note": "Static 2021 reference map; the date argument does not select imagery."}, t0)
            if source == "dynamic_world":
                img_path = self._search_and_download(bbox, date, output_dir, providers=providers)
                if not img_path:
                    return self._fail("No imagery found to define the output grid.", t0)
                from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler
                lbl = DynamicWorldLabeler(start_date=f"{date}-01", end_date=f"{date}-28")
                r = lbl.label_tile(img_path, _meta(img_path), out)
                if not r.success:
                    return self._fail(r.error or "Dynamic World labelling failed", t0)
                return self._ok(out, r.class_distribution or {}, {"source": source, "date": date}, t0)

            net = self._load_trained_model(source, checkpoint, num_classes, len(bands))
            img_path = self._search_and_download(bbox, date, output_dir, providers=providers, bands=bands)
            if not img_path:
                return self._fail("No imagery found.", t0)
            self._run_model(img_path, out, net, num_classes, len(bands), tile_size, overlap)
            return self._ok(out, {}, {"source": source, "date": date, "bands": bands,
                                      "num_classes": num_classes}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class WaterBodiesPipeline(BasePipeline):
    """Surface water from McFeeters' NDWI = (green - NIR) / (green + NIR).

    ``method="ndwi"`` (default) is a spectral index: no weights needed.
    Pixels with NDWI above ``ndwi_threshold`` (default 0.3, conservative;
    McFeeters' original cut is 0.0) are water. Cloud/shadow-masked pixels
    are written as nodata and excluded from ``water_coverage``.

    ``method="model"`` runs your own segmentation network
    (``model=`` + ``checkpoint=`` required).

    Outputs: ``ndwi.tif`` (continuous index, the returned ``output_path``)
    and ``water_mask.tif`` (1 = water, 0 = not water, 255 = nodata).
    """
    name = "water_bodies"
    description = "Surface water mask from NDWI (green, NIR); or your own trained model"
    domain = "water"
    kind = "ready"
    tags = ["water", "ndwi"]

    def run(self, bbox, output_dir, date="2024-06", method="ndwi",
            cloud_cover_max=10.0, ndwi_threshold=0.3, model="unet_resnet50",
            checkpoint=None, bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, **kwargs):
        import time

        from pygeovision.ai.pipelines._raster import (
            fraction, normalized_difference, read_stack, valid_mask, write_classes, write_float,
        )
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        try:
            if method not in ("ndwi", "model"):
                return self._fail(f"method must be 'ndwi' or 'model', got {method!r}.", t0)
            if method == "model":
                net = self._load_trained_model(model, checkpoint, num_classes, len(bands))
                img = self._search_and_download(bbox, date, output_dir, providers=providers,
                                                cloud_cover_max=cloud_cover_max, bands=bands)
                if not img:
                    return self._fail("No clear imagery found.", t0)
                out = output_dir / "water_mask.tif"
                pred = self._run_model(img, out, net, num_classes, len(bands), tile_size, overlap)
                return self._ok(out, {"water_coverage": float((pred == 1).mean())},
                                {"date": date, "method": method, "model": model}, t0)

            img = self._search_and_download(bbox, date, output_dir, providers=providers,
                                            cloud_cover_max=cloud_cover_max, bands=("green", "nir"))
            if not img:
                return self._fail("No clear imagery found.", t0)
            data, profile = read_stack(img)
            green, nir = data[0], data[1]
            ndwi = normalized_difference(green, nir)
            valid = valid_mask(ndwi)
            water = (ndwi > ndwi_threshold) & valid
            ndwi_path = write_float(output_dir / "ndwi.tif", ndwi, profile, ["NDWI (green-NIR)/(green+NIR)"])
            write_classes(output_dir / "water_mask.tif", water, valid, profile,
                          ["0=not water", "1=water", "255=nodata"])
            return self._ok(
                ndwi_path,
                {"water_coverage": fraction(water, valid), "valid_fraction": float(valid.mean()),
                 "ndwi_threshold": ndwi_threshold},
                {"date": date, "method": method, "bands": ("green", "nir"),
                 "mask_path": str(output_dir / "water_mask.tif")}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class ChangeDetectionPipeline(BasePipeline):
    """Bi-temporal change detection.

    ``method="cva"`` (default) is classical **change vector analysis**: the
    Euclidean length of the per-band reflectance difference between the two
    dates. A pixel is "changed" when that length exceeds
    ``max(Otsu threshold, min_magnitude)``; the floor stops Otsu from
    splitting pure sensor noise into two classes on an unchanged scene.
    No weights are needed. It detects *any* spectral change (including
    seasonal vegetation and illumination), so compare the same season, and
    treat the result as "spectrally changed", not "land use changed".
    Cloud-masked pixels in either date are nodata.

    ``method="model"`` runs your own siamese network (``model=`` +
    ``checkpoint=`` required).

    Outputs: ``change_mask.tif`` (1 changed, 0 unchanged, 255 nodata) and
    ``change_magnitude.tif``.
    """
    name = "change_detection"
    description = "Change vector analysis between two dates (Otsu threshold); or your own siamese model"
    domain = "change"
    kind = "ready"
    tags = ["change", "bitemporal"]

    def run(self, bbox, output_dir, date_before="2022-06", date_after="2024-06",
            method="cva", model="siamese_unet", checkpoint=None, cloud_cover_max=25.0,
            bands=None, num_classes=2, tile_size=512, overlap=64,
            min_magnitude=0.05, **kwargs):
        import time

        import numpy as np
        import rasterio

        from pygeovision.ai.pipelines._raster import (
            area_ha, fraction, read_stack, write_classes, write_float,
        )
        from pygeovision.data.radiometric import real_pixel_area_ha
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        try:
            if method not in ("cva", "model"):
                return self._fail(f"method must be 'cva' or 'model', got {method!r}.", t0)
            if bands is None:
                bands = ("blue", "green", "red", "nir") if method == "cva" else ("red", "green", "blue")
            net = None
            if method == "model":
                net = self._load_trained_model(model, checkpoint, num_classes, len(bands))
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, output_dir, providers=providers,
                cloud_cover_max=cloud_cover_max, bands=tuple(bands),
            )
            if not before_path or not after_path:
                return self._fail("Could not acquire imagery for one or both dates.", t0)

            meta = {"date_before": date_before, "date_after": date_after,
                    "method": method, "bands": tuple(bands)}
            if method == "model":
                out = output_dir / "change_mask.tif"
                pred = self._run_model(before_path, out, net, num_classes, len(bands),
                                       tile_size, overlap, image_path_b=after_path)
                meta["model"] = model
                return self._ok(out, {"change_fraction": float((pred > 0).mean())}, meta, t0)

            a, profile = read_stack(before_path)
            b, _ = read_stack(after_path)
            if a.shape != b.shape:
                return self._fail(f"before/after rasters differ in shape ({a.shape} vs {b.shape}).", t0)
            valid = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
            if not valid.any():
                return self._fail("No pixel is cloud-free in both dates.", t0)
            magnitude = np.sqrt(((b - a) ** 2).sum(axis=0)).astype(np.float32)
            magnitude[~valid] = np.nan
            otsu = _otsu_threshold(magnitude[valid])
            threshold = max(otsu, float(min_magnitude))
            changed = (magnitude > threshold) & valid

            mask_path = write_classes(output_dir / "change_mask.tif", changed, valid, profile,
                                      ["0=unchanged", "1=changed", "255=nodata"])
            write_float(output_dir / "change_magnitude.tif", magnitude, profile,
                        ["change vector magnitude (reflectance units)"])
            with rasterio.open(mask_path) as src:
                px_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)
            stats = {
                "change_fraction": fraction(changed, valid),
                "changed_area_ha": area_ha(changed, valid, px_ha),
                "threshold": threshold, "otsu_threshold": otsu, "min_magnitude": float(min_magnitude),
                "valid_fraction": float(valid.mean()),
            }
            return self._ok(mask_path, stats, meta, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


# ---------------------------------------------------------------------------
# Pipelines that need your trained weights
# ---------------------------------------------------------------------------

class _NetworkPipeline(BasePipeline):
    """Shared shape of the pipelines that only run a trained network."""
    kind = "needs_weights"


class BuildingFootprintsPipeline(_NetworkPipeline):
    """Building segmentation with a network you trained (``checkpoint=`` required).

    Output ``building_mask.tif``: class map (1 = building). ``stats``
    holds ``building_coverage``.
    """
    name = "building_footprints"
    description = "Building segmentation with your trained model (checkpoint required)"
    domain = "urban"
    tags = ["urban", "segmentation", "needs-weights"]

    def run(self, bbox, output_dir, date="2024-06", model="unet_resnet50", checkpoint=None,
            cloud_cover_max=15.0, bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, **kwargs):
        import time
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        try:
            net = self._load_trained_model(model, checkpoint, num_classes, len(bands))
            img = self._search_and_download(bbox, date, output_dir, providers=providers,
                                            cloud_cover_max=cloud_cover_max, bands=bands)
            if not img:
                return self._fail("No imagery found.", t0)
            out = output_dir / "building_mask.tif"
            pred = self._run_model(img, out, net, num_classes, len(bands), tile_size, overlap)
            return self._ok(out, {"building_coverage": float((pred == 1).mean())},
                            {"model": model, "date": date, "bands": bands, "num_classes": num_classes}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class CropMonitoringPipeline(_NetworkPipeline):
    """Crop-type mapping with a network you trained (``checkpoint=`` required).

    ``crop_classes`` is the list of crop names you trained on;
    ``num_classes`` is ``len(crop_classes) + 1`` (background) unless given.
    """
    name = "crop_monitoring"
    description = "Crop-type segmentation with your trained model (checkpoint required)"
    domain = "agriculture"
    tags = ["agriculture", "segmentation", "needs-weights"]

    def run(self, bbox, output_dir, date="2023-06", crop_classes=None, model="segformer_b2",
            checkpoint=None, bands=("red", "green", "blue"), num_classes=None,
            tile_size=512, overlap=64, **kwargs):
        import time
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        try:
            if num_classes is None:
                if not crop_classes:
                    return self._fail("Give crop_classes (your class names) or num_classes "
                                      "matching the trained checkpoint.", t0)
                num_classes = len(crop_classes) + 1
            net = self._load_trained_model(model, checkpoint, num_classes, len(bands))
            img = self._search_and_download(bbox, date, output_dir, providers=providers, bands=bands)
            if not img:
                return self._fail("No imagery found.", t0)
            out = output_dir / "crop_map.tif"
            self._run_model(img, out, net, num_classes, len(bands), tile_size, overlap)
            return self._ok(out, {}, {"date": date, "crop_classes": crop_classes, "bands": bands,
                                      "model": model, "num_classes": num_classes}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class SolarDetectionPipeline(_NetworkPipeline):
    """Solar-panel segmentation with a network you trained (``checkpoint=`` required).

    Panels are small, so the default tile is 256 px. At 10 m Sentinel-2
    resolution only utility-scale arrays are resolvable; train and run on
    higher-resolution imagery for rooftop panels.
    """
    name = "solar_detection"
    description = "Solar-panel segmentation with your trained model (checkpoint required)"
    domain = "energy"
    tags = ["energy", "segmentation", "needs-weights"]

    def run(self, bbox, output_dir, date="2024-06", cloud_cover_max=5.0,
            model="unet_efficientnet_b4", checkpoint=None, bands=("red", "green", "blue"),
            num_classes=2, tile_size=256, overlap=32, **kwargs):
        import time
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        try:
            net = self._load_trained_model(model, checkpoint, num_classes, len(bands))
            img = self._search_and_download(bbox, date, output_dir, providers=providers,
                                            cloud_cover_max=cloud_cover_max, bands=bands)
            if not img:
                return self._fail("No cloud-free imagery found.", t0)
            out = output_dir / "solar_mask.tif"
            pred = self._run_model(img, out, net, num_classes, len(bands), tile_size, overlap)
            return self._ok(out, {"panel_coverage": float((pred == 1).mean())},
                            {"date": date, "model": model, "bands": bands, "num_classes": num_classes}, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class _BitemporalNetworkPipeline(_NetworkPipeline):
    """Two-date siamese inference with a network you trained."""
    _default_model = "siamese_unet"
    _default_classes = 2
    _output_name = "change.tif"
    _collections: list[str] | None = None
    _cloud_max = 30.0

    def _run_pair(self, bbox, output_dir, date_before, date_after, model, checkpoint,
                  bands, num_classes, tile_size, overlap, kwargs, extra_meta):
        import time
        t0 = time.time()
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        providers = kwargs.pop("providers", None)
        try:
            net = self._load_trained_model(model, checkpoint, num_classes, len(bands))
            before, after = self._search_and_download_pair(
                bbox, date_before, date_after, output_dir, collections=self._collections,
                providers=providers, cloud_cover_max=self._cloud_max, bands=bands)
            if not before or not after:
                return self._fail("Imagery not found for both dates.", t0)
            out = output_dir / self._output_name
            pred = self._run_model(before, out, net, num_classes, len(bands), tile_size, overlap,
                                   image_path_b=after)
            meta = {"date_before": date_before, "date_after": date_after, "model": model,
                    "bands": bands, "num_classes": num_classes, **extra_meta}
            return self._ok(out, {"changed_fraction": float((pred > 0).mean())}, meta, t0)
        except Exception as exc:
            return self._fail(str(exc), t0)


class DisasterAssessmentPipeline(_BitemporalNetworkPipeline):
    """Damage assessment from a pre/post-event pair with your trained siamese network.

    Dates: ``date_before`` (pre-event) and ``date_after`` (post-event).
    ``num_classes`` is the number of damage classes you trained on.
    """
    name = "disaster_assessment"
    description = "Pre/post-event damage classes with your trained siamese model (checkpoint required)"
    domain = "disaster"
    tags = ["disaster", "change", "needs-weights"]
    _output_name = "damage_assessment.tif"

    def run(self, bbox, output_dir, date_before=None, date_after=None, disaster_type="generic",
            model="siamese_unet", checkpoint=None, bands=("red", "green", "blue"),
            num_classes=4, tile_size=512, overlap=64, pre_date=None, post_date=None, **kwargs):
        date_before = date_before or pre_date or "2024-01"
        date_after = date_after or post_date or "2024-02"
        return self._run_pair(bbox, output_dir, date_before, date_after, model, checkpoint,
                              bands, num_classes, tile_size, overlap, kwargs,
                              {"disaster_type": disaster_type})


class DeforestationPipeline(_BitemporalNetworkPipeline):
    """Forest-loss mapping with your trained change network.

    Dates may be a year ("2020", meaning July of that year) or YYYY-MM.
    """
    name = "deforestation"
    description = "Forest-loss classes between two dates with your trained model (checkpoint required)"
    domain = "forest"
    tags = ["forest", "change", "needs-weights"]
    _output_name = "deforestation_map.tif"
    _collections = ["sentinel-2-l2a"]

    def run(self, bbox, output_dir, date_before=None, date_after=None, model="changeformer",
            checkpoint=None, bands=("red", "green", "blue"), num_classes=3,
            tile_size=512, overlap=64, baseline_year=None, analysis_year=None, **kwargs):
        before = _month(date_before or baseline_year or "2020")
        after = _month(date_after or analysis_year or "2024")
        return self._run_pair(bbox, output_dir, before, after, model, checkpoint,
                              bands, num_classes, tile_size, overlap, kwargs, {})


class UrbanGrowthPipeline(_BitemporalNetworkPipeline):
    """Urban-expansion mapping from Landsat pairs with your trained change network.

    Dates may be a year ("2018", meaning June of that year) or YYYY-MM.
    """
    name = "urban_growth"
    description = "Urban expansion between two Landsat dates with your trained model (checkpoint required)"
    domain = "urban"
    tags = ["urban", "change", "needs-weights"]
    _output_name = "urban_growth.tif"
    _collections = ["landsat-c2-l2"]

    def run(self, bbox, output_dir, date_before=None, date_after=None, model="siamese_unet",
            checkpoint=None, bands=("red", "green", "blue"), num_classes=2,
            tile_size=512, overlap=64, start_year=None, end_year=None, **kwargs):
        before = _month(date_before or start_year or "2018", "06")
        after = _month(date_after or end_year or "2024", "06")
        return self._run_pair(bbox, output_dir, before, after, model, checkpoint,
                              bands, num_classes, tile_size, overlap, kwargs, {})


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

_CORE_PIPELINES: dict[str, type] = {
    "change_detection":    ChangeDetectionPipeline,
    "land_cover":          LandCoverPipeline,
    "building_footprints": BuildingFootprintsPipeline,
    "crop_monitoring":     CropMonitoringPipeline,
    "disaster_assessment": DisasterAssessmentPipeline,
    "deforestation":       DeforestationPipeline,
    "urban_growth":        UrbanGrowthPipeline,
    "water_bodies":        WaterBodiesPipeline,
    "solar_detection":     SolarDetectionPipeline,
}


def pipeline_catalogue() -> dict[str, type]:
    """Every pipeline (core + domain), keyed by name."""
    from pygeovision.ai.pipelines import domains
    return {**_CORE_PIPELINES, **domains.DOMAIN_PIPELINES}


def get_pipeline(name: str, pgv_client: Any) -> BasePipeline:
    """Instantiate a pipeline by name.

    Raises:
        ValueError: unknown name, or a pipeline that was removed (the
            message says why).
    """
    from pygeovision.ai.pipelines.removed import REMOVED_PIPELINES

    catalogue = pipeline_catalogue()
    if name in catalogue:
        return catalogue[name](pgv_client)
    if name in REMOVED_PIPELINES:
        raise ValueError(f"Pipeline '{name}' was removed: {REMOVED_PIPELINES[name]}")
    raise ValueError(f"Unknown pipeline '{name}'. Available: {sorted(catalogue)}")


def list_pipelines() -> list[str]:
    """Sorted names of every pipeline."""
    return sorted(pipeline_catalogue())


def pipeline_kinds() -> dict[str, str]:
    """``{name: "ready" | "needs_weights"}`` for every pipeline."""
    return {n: c.kind for n, c in sorted(pipeline_catalogue().items())}


__all__ = [
    "BasePipeline", "PipelineResult", "ModelWeightsRequired",
    "get_pipeline", "list_pipelines", "pipeline_kinds", "pipeline_catalogue",
    "ChangeDetectionPipeline", "LandCoverPipeline", "BuildingFootprintsPipeline",
    "CropMonitoringPipeline", "DisasterAssessmentPipeline", "DeforestationPipeline",
    "UrbanGrowthPipeline", "WaterBodiesPipeline", "SolarDetectionPipeline",
]
