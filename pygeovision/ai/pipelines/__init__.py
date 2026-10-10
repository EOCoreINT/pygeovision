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


class _IncompleteSceneError(Exception):
    """The downloaded files cannot provide the requested bands (usually a partial earlier download)."""


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
    #: map, vectors and reports written after the run (``pygeovision.reporting.Deliverables``), or None
    deliverables: Any = None

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

# Asset keys the STAC catalogues use for bands that have no Sentinel-2 equivalent (Landsat Collection 2 L2):
# "lwir11" is ST_B10 on Landsat 8/9, "lwir" is ST_B6 on Landsat 4/5/7.
_STAC_LANDSAT_KEYS: dict[str, tuple[str, ...]] = {"thermal": ("lwir11", "lwir")}


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
        self._coverage_log: list[dict[str, Any]] = []

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
        stats = dict(stats or {})
        metadata = dict(metadata or {})
        log = getattr(self, "_coverage_log", [])
        if log:
            # The smallest coverage of any input image: a result built from an
            # image that covers 23% of the bbox must say so.
            stats.setdefault("aoi_coverage_fraction", min(e["aoi_coverage_fraction"] for e in log))
            metadata.setdefault("input_coverage", list(log))
            self._coverage_log = []
        return PipelineResult(
            pipeline=self.name, output_path=Path(output_path) if output_path else None,
            stats=stats, metadata=metadata, success=True,
            duration_seconds=(time.time() - t0) if t0 is not None else 0.0,
        )

    def _fail(self, error: str, t0: float | None = None) -> PipelineResult:
        import time
        for e in getattr(self, "_coverage_log", []):
            if e.get("clear_fraction", 1.0) < self._MIN_CLEAR_FRACTION:
                error += (f" The downloaded scene(s) {', '.join(e.get('scenes_used', []))} are clear over only "
                          f"{100 * e['clear_fraction']:.0f}% of the requested area (the rest is cloud, shadow "
                          f"or outside the scene); try another date.")
                break
        self._coverage_log = []
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

        Runs identically whether one scene is enough or several need to be
        mosaicked together. If the files on disk turn out to be an
        incomplete earlier download that pygeofetch's resume logic took for
        finished (it matches on the scene id and trusts a single file), the
        scene is downloaded once more with resume off before giving up.
        """
        for fresh in (False, True):
            dl = self._download_scene_files(
                result, output_dir, bands, apply_cloud_mask, post_process, subdir_name, fresh=fresh,
            )
            if dl is None:
                return None
            try:
                return self._prepare_downloaded_scene(
                    dl, output_dir, bands, apply_scale, apply_cloud_mask, subdir_name,
                )
            except _IncompleteSceneError as exc:
                if fresh:
                    logger.error("_download_and_prepare_scene: %s", exc)
                    return None
                logger.warning(
                    "_download_and_prepare_scene: %s -- most likely an interrupted earlier "
                    "download that resume treated as complete; downloading it again without resume.",
                    exc,
                )
        return None

    def _download_scene_files(
        self,
        result: Any,
        output_dir: Path,
        bands: tuple[str, ...],
        apply_cloud_mask: bool,
        post_process: list[str] | None,
        subdir_name: str,
        fresh: bool = False,
    ) -> Any:
        """Download only the assets the pipeline needs; None on failure."""
        pp = post_process or ["unzip", "reproject:EPSG:4326"]
        # Request pygeofetch's real asset codes, not pygeovision's generic
        # names: expand each requested band into every known alias across
        # Sentinel-2 and both Landsat families (their band numbering differs,
        # see LANDSAT_ETM_TM_ALIASES vs LANDSAT_OLI_ALIASES).
        from pygeovision.data.radiometric import (
            S2_BAND_ALIASES, LANDSAT_ETM_TM_ALIASES, LANDSAT_OLI_ALIASES,
        )
        expanded_bands: list[str] = []
        for b in bands:
            if b in _STAC_LANDSAT_KEYS:
                # "thermal" must not be sent as such: pygeofetch resolves it to the SWIR asset ("swir16"),
                # so the thermal band was never downloaded. Ask for the catalogues' own thermal keys.
                expanded_bands.extend(_STAC_LANDSAT_KEYS[b])
            else:
                expanded_bands.extend(S2_BAND_ALIASES.get(b, (b,)))
            expanded_bands.extend(LANDSAT_ETM_TM_ALIASES.get(b, ()))
            expanded_bands.extend(LANDSAT_OLI_ALIASES.get(b, ()))
        if apply_cloud_mask:
            # The cloud mask is read from the scene classification layer
            # (Sentinel-2) or QA_PIXEL (Landsat). With the band filter on,
            # they are only downloaded if asked for; without them masking
            # is silently skipped.
            expanded_bands.extend(("SCL", "QA_PIXEL"))
        expanded_bands = list(dict.fromkeys(expanded_bands))  # dedupe, preserve order

        downloads = self.pgv.download(
            [result], output_dir=output_dir / subdir_name,
            parallel=1, post_process=pp, verify_checksum=False,
            bands=expanded_bands,
            **({"resume": False, "overwrite": True} if fresh else {}),
        )
        if not (downloads and downloads[0].success):
            return None
        return downloads[0]

    def _prepare_downloaded_scene(
        self,
        dl: Any,
        output_dir: Path,
        bands: tuple[str, ...],
        apply_scale: bool,
        apply_cloud_mask: bool,
        subdir_name: str,
    ) -> Path | None:
        """Stack, scale and cloud-mask one downloaded scene.

        Raises _IncompleteSceneError when the files on disk cannot hold the requested bands.
        """
        import re

        if len(dl.asset_paths) > 1:
            # One file per scene: every scene of a mosaic is stacked into the
            # same folder, so a fixed "stack.tif" was overwritten by the next
            # scene and the mosaic was built from the last scene only.
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dl.scene_id))[:120] or "scene"
            try:
                return self._stack_bands(
                    dl.asset_paths, output_dir / subdir_name / f"stack_{safe_id}.tif",
                    bands=bands, scene_id=dl.scene_id,
                    apply_scale=apply_scale, apply_cloud_mask=apply_cloud_mask,
                )
            except RuntimeError as exc:
                raise _IncompleteSceneError(str(exc)) from exc

        # Single file for the scene. A provider can deliver a pre-stacked file
        # with more or fewer bands than requested; passing it on would crash
        # the model later with a channel-count mismatch, so refuse it here.
        import rasterio
        with rasterio.open(dl.path) as _src:
            real_band_count = _src.count
        from pygeovision.data.radiometric import identify_sensor, dn_to_reflectance
        is_landsat, is_sentinel2 = identify_sensor(dl.scene_id)
        if real_band_count != len(bands):
            msg = (
                f"single-asset download ({dl.scene_id}) has {real_band_count} real band(s), "
                f"but {len(bands)} were requested ({bands}); refusing to silently pass through "
                f"a mismatched band count"
            )
            if is_landsat or is_sentinel2:
                raise _IncompleteSceneError(msg)     # these products ship one file per band
            logger.error("_download_and_prepare_scene: %s. The provider probably delivered one "
                         "already-combined file; use a collection that ships separate band files.", msg)
            return None

        if not apply_scale:
            return dl.path

        if is_landsat or is_sentinel2:
            logger.warning(
                "_download_and_prepare_scene: single-asset download for a "
                "recognised %s scene (%s) -- unusual. Applying real "
                "DN->reflectance scaling directly (no cloud mask: no QA layer in this file).",
                "Landsat" if is_landsat else "Sentinel-2", dl.scene_id,
            )
            from pygeovision.data._profile import writable_profile
            scaled_path = output_dir / subdir_name / f"single_asset_scaled_{dl.scene_id}.tif"
            scaled_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dl.path) as src:
                data = dn_to_reflectance(src.read(), dl.scene_id, is_landsat=is_landsat)
                profile = writable_profile(src.profile, dtype="float32", nodata=float("nan"))
                with rasterio.open(scaled_path, "w", **profile) as dst:
                    dst.write(data)
            return scaled_path

        logger.info(
            "_download_and_prepare_scene: single-asset download (%s), sensor not "
            "positively identified -- passing through without radiometric scaling.",
            dl.scene_id,
        )
        return dl.path

    #: Candidates fetched per search. The old cap of 5 was shared by every
    #: provider, and the same scene is listed by three of them, so five hits were
    #: often two scenes: an AOI over four tiles never saw one of them.
    _SEARCH_POOL = 100
    #: A scene must add at least this share of the AOI to be worth downloading
    #: (a neighbouring orbit's sliver is not worth another 1 GB scene).
    _MIN_GAIN_FRACTION = 0.005

    @staticmethod
    def _scene_key(result: Any) -> tuple:
        """Identity of a scene across providers: (tile or path/row, acquisition date)."""
        import re
        sid = str(getattr(result, "id", "") or "")
        m_date = re.search(r"(?<!\d)((?:19|20)\d{6})(?!\d)", sid)
        m_tile = re.search(r"_T?(\d{2}[A-Z]{3})(?:_|$)", sid)            # Sentinel-2 MGRS tile
        m_wrs = re.search(r"_L\d{1}[A-Z]{2}?_?(\d{6})_((?:19|20)\d{6})", sid) or \
            re.search(r"_(\d{6})_((?:19|20)\d{6})_", sid)                # Landsat path/row + date
        if m_tile and m_date:
            return ("s2", m_tile.group(1), m_date.group(1))
        if m_wrs:
            return ("wrs", m_wrs.group(1), m_wrs.group(2))
        bbox = getattr(result, "bbox", None)
        return ("bbox", tuple(round(v, 3) for v in bbox) if bbox else sid, str(getattr(result, "datetime", ""))[:10])

    # Providers whose asset links can be downloaded without an AWS account, tried first when several providers
    # list the same scene. The Landsat copies on aws_earth / element84 are requester-pays S3 objects and fail
    # without AWS credentials, so which listing came back first (a race between providers) decided whether a
    # run worked.
    _PREFERRED_PROVIDERS = ("planetary_computer", "copernicus", "usgs")

    def _dedupe_scenes(self, results: list[Any]) -> list[Any]:
        """Drop repeat listings of one scene (the same tile and date from several
        providers), keeping the best one. Scenes stay in their ranked order; among the listings of one
        scene a provider from ``_PREFERRED_PROVIDERS`` wins, otherwise the first. The other listings are
        remembered in ``self._alternates`` so a failed download can be retried from another provider."""
        groups: dict[Any, list[Any]] = {}
        for r in results:
            groups.setdefault(self._scene_key(r), []).append(r)

        def rank(r: Any) -> int:
            prov = str(getattr(r, "provider", ""))
            return self._PREFERRED_PROVIDERS.index(prov) if prov in self._PREFERRED_PROVIDERS else len(self._PREFERRED_PROVIDERS)

        out: list[Any] = []
        self._alternates = {}
        for listings in groups.values():
            ordered = sorted(listings, key=rank)           # stable: ties keep the search order
            out.append(ordered[0])
            self._alternates[id(ordered[0])] = ordered[1:]
        return out

    def _check_aoi_coverage(
        self,
        bbox: tuple[float, float, float, float],
        results: list[Any],
        max_scenes: int = 4,
        min_gain_fraction: float | None = None,
    ) -> tuple[list[Any], bool]:
        """Greedy set cover of the AOI by scene footprints (SearchResult.bbox,
        available before anything is downloaded).

        An AOI straddling tile or path/row boundaries needs several scenes;
        cropping only the best one used to succeed silently with a smaller
        output than requested. At each step the scene adding the most still
        uncovered AOI area is taken. Among scenes whose gain is within 5% of the
        best, the earliest in ``results`` wins, so with results sorted by cloud
        cover a slightly larger sliver does not beat a clearer scene. Scenes that
        would add less than ``min_gain_fraction`` of the AOI (default 0.5%) are
        not selected.

        Returns (selected_results, full_coverage_achieved).
        """
        from shapely.geometry import box
        from shapely.ops import unary_union

        aoi_geom = box(*bbox)
        min_gain = aoi_geom.area * (self._MIN_GAIN_FRACTION if min_gain_fraction is None else min_gain_fraction)
        candidates = [r for r in results if r.bbox is not None]
        if not candidates:
            # No footprint information for any candidate -- fall back to the
            # single best-scored result rather than refusing outright.
            return (results[:1], False)

        selected: list[Any] = []
        covered_geom = None
        remaining = list(candidates)
        for _ in range(max_scenes):
            if covered_geom is not None and aoi_geom.difference(covered_geom).area < 1e-12:
                break
            gains: list[tuple[Any, float, Any]] = []
            for r in remaining:
                scene_geom = box(*r.bbox)
                new_coverage = scene_geom.intersection(aoi_geom)
                if covered_geom is not None:
                    new_coverage = new_coverage.difference(covered_geom)
                gains.append((r, new_coverage.area, scene_geom))
            best_gain = max((g for _, g, _ in gains), default=0.0)
            if best_gain <= max(min_gain, 1e-12):
                break
            best, _, best_geom = next(t for t in gains if t[1] >= 0.95 * best_gain)
            selected.append(best)
            covered_geom = best_geom if covered_geom is None else unary_union([covered_geom, best_geom])
            remaining = [r for r in remaining if r is not best]

        full_coverage = covered_geom is not None and aoi_geom.difference(covered_geom).area < 1e-9
        if not selected:
            selected = results[:1]
        return (selected, full_coverage)

    def _footprint_coverage(self, bbox: tuple[float, float, float, float], results: list[Any]) -> float:
        """Share (0-1) of the AOI inside the union of the scenes' footprints."""
        from shapely.geometry import box
        from shapely.ops import unary_union

        aoi = box(*bbox)
        boxes = [box(*r.bbox) for r in results if getattr(r, "bbox", None) is not None]
        if not boxes:
            return 1.0 if results else 0.0
        return float(unary_union(boxes).intersection(aoi).area / aoi.area)

    def _pick_replacement(
        self,
        bbox: tuple[float, float, float, float],
        candidates: list[Any],
        keep: list[Any],
        exclude: set[int],
    ) -> Any | None:
        """The best untried scene for the part of the AOI the kept scenes leave uncovered."""
        from shapely.geometry import box
        from shapely.ops import unary_union

        aoi = box(*bbox)
        have = [box(*r.bbox) for r in keep if getattr(r, "bbox", None) is not None]
        uncovered = aoi.difference(unary_union(have)) if have else aoi
        if uncovered.is_empty:
            return None
        best, best_gain = None, aoi.area * self._MIN_GAIN_FRACTION
        for r in candidates:
            if id(r) in exclude or getattr(r, "bbox", None) is None:
                continue
            gain = box(*r.bbox).intersection(uncovered).area
            if gain > best_gain * 1.0000001 or (best is None and gain >= best_gain and gain > 0):
                best, best_gain = r, gain
        return best

    def _mosaic_scenes(self, image_paths: list[Path], output_path: Path) -> Path:
        """Real mosaic of multiple already-corrected, already-stacked
        rasters via rasterio.merge (the same real, standard technique
        pygeofetch's own preprocess_mosaic() uses).

        Mosaicking happens here, AFTER each scene has already been
        individually radiometrically corrected by
        _download_and_prepare_scene -- not on raw DN -- to avoid a real
        sensor-calibration discontinuity at the seam between scenes.
        """
        import numpy as np
        import rasterio
        from rasterio.merge import merge as rio_merge
        from pygeovision.data._profile import writable_profile

        if len({str(p) for p in image_paths}) != len(image_paths):
            raise ValueError(f"_mosaic_scenes: the same file was passed more than once: {image_paths}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        datasets = [rasterio.open(p) for p in image_paths]
        nodata = datasets[0].nodata
        if nodata is None and not np.issubdtype(np.dtype(datasets[0].dtypes[0]), np.floating):
            nodata = 0          # DN 0 is "no data"; without it the first scene's empty corner would hide the next scene
        try:
            mosaic, out_transform = rio_merge(datasets, nodata=nodata)
        finally:
            for ds in datasets:
                ds.close()
        with rasterio.open(image_paths[0]) as ref:
            profile = writable_profile(
                ref.profile, height=mosaic.shape[1], width=mosaic.shape[2],
                transform=out_transform, compress="lzw", nodata=nodata,
            )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
        return output_path

    #: AOI gaps smaller than this share are not worth another scene.
    _GAP_TOLERANCE = 0.002
    #: How many extra scenes may be added to close gaps in the data.
    _MAX_GAP_SCENES = 4

    @staticmethod
    def _aoi_grid(bbox: tuple[float, float, float, float], n: int = 600):
        """A coarse EPSG:4326 grid over the AOI, used to measure where imagery exists."""
        from rasterio.transform import from_bounds
        w, s_, e, n_ = bbox
        dx, dy = e - w, n_ - s_
        width = max(2, int(round(n * dx / max(dx, dy))))
        height = max(2, int(round(n * dy / max(dx, dy))))
        return from_bounds(w, s_, e, n_, width, height), (height, width)

    def _data_footprint_on_grid(self, path: Path, transform: Any, shape: tuple[int, int]):
        """Boolean array on the AOI grid: True where the image at ``path`` holds data.

        Uses the footprint raster written next to a stacked scene (so cloud-masked pixels still count as
        covered); for any other raster, pixels that are finite and non-zero in some band."""
        import numpy as np
        import rasterio
        from rasterio.warp import Resampling, reproject
        from pygeovision.data.radiometric import footprint_path

        out = np.zeros(shape, dtype="uint8")
        fp = footprint_path(path)
        if fp.is_file():
            with rasterio.open(fp) as src:
                reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest)
            return out.astype(bool)
        with rasterio.open(path) as src:
            f = max(1, int(np.ceil(max(src.height, src.width) / 4096)))
            data = src.read(out_shape=(src.count, max(1, src.height // f), max(1, src.width // f)),
                            resampling=Resampling.nearest)
            ok = (np.isfinite(data) & (data != 0)).any(axis=0).astype("uint8")
            reproject(ok, out, src_transform=src.transform * src.transform.scale(f, f), src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest)
        return out.astype(bool)

    def _aoi_data_gap(self, bbox, paths: list[Path]):
        """(gap mask, transform, covered fraction): the part of the AOI none of ``paths`` has data for."""
        import numpy as np
        transform, shape = self._aoi_grid(bbox)
        covered = np.zeros(shape, dtype=bool)
        for p in paths:
            try:
                covered |= self._data_footprint_on_grid(p, transform, shape)
            except Exception as exc:  # noqa: BLE001 - a footprint we cannot read just does not count as coverage
                logger.warning("_aoi_data_gap: could not read the data footprint of %s: %s", Path(p).name, exc)
        return ~covered, transform, float(covered.mean())

    def _pick_for_gap(self, gap, transform, candidates: list[Any], exclude: set[int]) -> Any | None:
        """The untried scene whose footprint covers most of the gap (earlier = clearer wins within 5%)."""
        import numpy as np
        h, w = gap.shape
        cols = np.arange(w)
        rows = np.arange(h)
        xs = transform.c + (cols + 0.5) * transform.a
        ys = transform.f + (rows + 0.5) * transform.e
        best, best_gain = None, max(self._GAP_TOLERANCE, self._MIN_GAIN_FRACTION) * gap.size
        for r in candidates:
            b = getattr(r, "bbox", None)
            if id(r) in exclude or b is None:
                continue
            inside = ((xs >= b[0]) & (xs <= b[2]))[None, :] & ((ys >= b[1]) & (ys <= b[3]))[:, None]
            gain = float((gap & inside).sum())
            if gain > best_gain * 1.05 or (best is None and gain >= best_gain):
                best, best_gain = r, gain
        return best

    def _prepare_with_alternates(self, result, output_dir, bands, apply_scale, apply_cloud_mask, post_process):
        """Download and prepare one scene; on failure try the same scene from the other providers listing it."""
        p = self._download_and_prepare_scene(result, output_dir, bands, apply_scale, apply_cloud_mask, post_process)
        for alt in ([] if p is not None else getattr(self, "_alternates", {}).get(id(result), [])):
            logger.warning(
                "_search_and_download: scene %s failed from provider %s; trying the same scene from %s.",
                getattr(result, "id", "?"), getattr(result, "provider", "?"), getattr(alt, "provider", "?"))
            p = self._download_and_prepare_scene(alt, output_dir, bands, apply_scale, apply_cloud_mask, post_process)
            if p is not None:
                break
        return p

    def _fill_aoi_gaps(self, bbox, candidates, prepared_paths, prepared_results, tried, failed_ids,
                       output_dir, bands, apply_scale, apply_cloud_mask, post_process) -> float:
        """Close the parts of the AOI the downloaded scenes hold no data for.

        Scene footprints are bounding boxes, but a scene is a tilted swath with empty corners, so a footprint
        that "covers" the AOI can still leave a gap. The data really present is measured on the prepared
        rasters; while a gap remains, the candidate that covers most of it is downloaded, prepared like the
        others (each scene is corrected on its own, the mosaic is built afterwards) and added. A scene that
        turns out to add nothing is dropped. Returns the share of the AOI the scenes have data for."""
        gap, transform, covered = self._aoi_data_gap(bbox, prepared_paths)
        added = 0
        while gap.mean() > self._GAP_TOLERANCE and added < self._MAX_GAP_SCENES:
            nxt = self._pick_for_gap(gap, transform, candidates, tried)
            if nxt is None:
                break
            tried.add(id(nxt))
            logger.info("_search_and_download: %.1f%% of the AOI has no imagery yet; adding scene %s.",
                        100 * gap.mean(), getattr(nxt, "id", "?"))
            p = self._prepare_with_alternates(nxt, output_dir, bands, apply_scale, apply_cloud_mask, post_process)
            if p is None:
                failed_ids.append(str(getattr(nxt, "id", "?")))
                continue
            new_gap, _, new_cov = self._aoi_data_gap(bbox, [*prepared_paths, p])
            if new_cov - covered < self._GAP_TOLERANCE / 2:
                logger.info("_search_and_download: scene %s adds no new coverage; not used.", getattr(nxt, "id", "?"))
                continue
            prepared_paths.append(p)
            prepared_results.append(nxt)
            gap, covered = new_gap, new_cov
            added += 1
        if gap.mean() > self._GAP_TOLERANCE:
            logger.warning("_search_and_download: %.1f%% of the AOI is not covered by any available scene.",
                           100 * gap.mean())
        return covered

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
        _skip_keys: frozenset = frozenset(),
        _attempt: int = 0,
    ) -> Path | None:
        """Search PyGeoFetch, download the scene(s) that cover the AOI and return
        one band-stacked, reflectance-scaled, cloud-masked image cropped to ``bbox``.

        * Only the requested bands (plus the cloud-mask layer) are downloaded.
        * The search looks at up to ``max(max_results, 100)`` candidates, drops
          repeat listings of one scene from several providers, and picks the
          fewest scenes that cover the AOI (a greedy set cover on footprints).
        * Each scene is corrected on its own, then the scenes are mosaicked.
        * A scene that cannot be downloaded or prepared is replaced by another
          scene covering the same part of the AOI when one exists.
        * How much of the AOI the result really covers is recorded and returned
          to the caller in the pipeline result (``aoi_coverage_fraction``).

        If ``require_full_coverage`` is True and the available scenes cannot cover
        the AOI, returns None instead of a smaller-than-requested result.
        """
        try:
            from pygeovision.data._dates import date_window
            start, end = date_window(date)

            # use_cache=False: cached search results are rebuilt from JSON and lose
            # the native satellite_data, so they were downloaded by a different,
            # degraded path (no engine, no band filter, no cloud-mask layer).
            results = self.pgv.search(
                bbox=bbox,
                date_range=(start, end),
                collections=collections or ["sentinel-2-l2a"],
                providers=providers,
                cloud_cover_max=cloud_cover_max,
                max_results=max(max_results, self._SEARCH_POOL),
                sort_by="cloud_cover",
                sort_order="asc",
                use_cache=False,
            )

            if _skip_keys:
                results = [r for r in results if self._scene_key(r) not in _skip_keys]
            if not results:
                return None
            candidates = self._dedupe_scenes(results)

            from shapely.geometry import box as _shapely_box
            top_result = candidates[0]
            single_scene_covers = (
                top_result.bbox is not None
                and _shapely_box(*top_result.bbox).covers(_shapely_box(*bbox))
            )

            if single_scene_covers:
                selected_results = [top_result]
                full_coverage = True
            else:
                selected_results, full_coverage = self._check_aoi_coverage(bbox, candidates, max_scenes=4)
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
            prepared_results: list[Any] = []
            failed_ids: list[str] = []
            queue = list(selected_results)
            tried: set[int] = set()
            max_attempts = len(selected_results) + 3
            while queue and len(tried) < max_attempts:
                result = queue.pop(0)
                tried.add(id(result))
                p = self._prepare_with_alternates(
                    result, output_dir, bands, apply_scale, apply_cloud_mask, post_process,
                )
                if p is not None:
                    prepared_paths.append(p)
                    prepared_results.append(result)
                    continue
                failed_ids.append(str(getattr(result, "id", "?")))
                logger.warning(
                    "_search_and_download: scene %s could not be prepared; looking for another "
                    "scene that covers its part of the AOI.", failed_ids[-1],
                )
                repl = self._pick_replacement(bbox, candidates, prepared_results + queue, tried)
                if repl is not None:
                    queue.append(repl)

            if not prepared_paths:
                return None

            # Footprints are bounding boxes; measure where the imagery really is and fetch more scenes for any gap.
            data_coverage = self._fill_aoi_gaps(
                bbox, candidates, prepared_paths, prepared_results, tried, failed_ids,
                output_dir, bands, apply_scale, apply_cloud_mask, post_process,
            )
            if require_full_coverage and data_coverage < 1 - self._GAP_TOLERANCE:
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

            # Final, defense-in-depth check: the output must have len(bands) bands
            # whichever path (single asset, stacking, mosaic) produced it.
            import rasterio
            with rasterio.open(final_path) as _src:
                actual_bands = _src.count
                _b = _src.bounds
                _crs = _src.crs
            if actual_bands != len(bands):
                logger.error(
                    "_search_and_download: final image has %d real band(s) but %d "
                    "were requested (%s) -- refusing to return a mismatched image "
                    "to the caller. This should not happen if band selection worked "
                    "correctly upstream; please report this as a bug.",
                    actual_bands, len(bands), bands,
                )
                return None

            clear = self._clear_fraction(final_path)
            self._record_coverage(
                label=output_dir.name, bbox=bbox, prepared=prepared_results,
                failed_ids=failed_ids, raster_bounds=(_b.left, _b.bottom, _b.right, _b.top), raster_crs=_crs,
                data_coverage=data_coverage, clear_fraction=clear,
            )
            if clear < self._MIN_CLEAR_FRACTION and _attempt < self._MAX_ALTERNATIVE_ROUNDS:
                # The best scene by scene-wide cloud cover can be cloudy exactly over the AOI (or empty there).
                # Try the other scenes of the date window before giving up.
                ours = self._coverage_log[-1]
                n_before = len(self._coverage_log)
                used = {self._scene_key(r) for r in prepared_results}
                logger.warning(
                    "_search_and_download: only %.0f%% of the AOI is clear (not cloud, shadow or empty) in %s; "
                    "looking for another scene in the date window.", 100 * clear,
                    ", ".join(str(getattr(r, "id", "?")) for r in prepared_results))
                alt = self._search_and_download(
                    bbox, date, output_dir, collections=collections, providers=providers,
                    cloud_cover_max=cloud_cover_max, post_process=post_process, max_results=max_results,
                    bands=bands, apply_scale=apply_scale, apply_cloud_mask=apply_cloud_mask,
                    require_full_coverage=require_full_coverage,
                    _skip_keys=frozenset(_skip_keys | used), _attempt=_attempt + 1)
                alt_clear = self._clear_fraction(alt) if alt is not None else -1.0
                if alt is not None and alt_clear > clear:
                    if ours in self._coverage_log:
                        self._coverage_log.remove(ours)
                    return alt
                del self._coverage_log[n_before:]
            return final_path

        except Exception as exc:
            logger.error("_search_and_download failed: %s", exc)
        return None

    #: Below this share of clear pixels over the AOI another scene is tried.
    _MIN_CLEAR_FRACTION = 0.05
    _MAX_ALTERNATIVE_ROUNDS = 3

    @staticmethod
    def _clear_fraction(path: Path | None) -> float:
        """Share of the pixels of a prepared image that hold data (finite and not 0 in some band)."""
        import numpy as np
        import rasterio
        if path is None:
            return 0.0
        try:
            with rasterio.open(path) as src:
                f = max(1, int(np.ceil(max(src.height, src.width) / 4096)))
                data = src.read(out_shape=(src.count, max(1, src.height // f), max(1, src.width // f)))
            ok = np.isfinite(data) & (data != 0)
            return float(ok.any(axis=0).mean())
        except Exception as exc:  # noqa: BLE001 - advisory only
            logger.warning("_clear_fraction: could not read %s: %s", path, exc)
            return 1.0

    def _record_coverage(
        self,
        label: str,
        bbox: tuple[float, float, float, float],
        prepared: list[Any],
        failed_ids: list[str],
        raster_bounds: tuple[float, float, float, float],
        raster_crs: Any,
        data_coverage: float | None = None,
        clear_fraction: float | None = None,
    ) -> None:
        """Remember how much of the requested bbox this image really covers."""
        from shapely.geometry import box

        footprint = self._footprint_coverage(bbox, prepared)
        extent = 1.0
        try:
            if raster_crs is not None and str(raster_crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
                from rasterio.warp import transform_bounds
                raster_bounds = transform_bounds(raster_crs, "EPSG:4326", *raster_bounds)
            aoi = box(*bbox)
            extent = float(box(*raster_bounds).intersection(aoi).area / aoi.area)
        except Exception as exc:  # noqa: BLE001 - coverage is advisory, never fail the run for it
            logger.debug("_record_coverage: raster extent check skipped: %s", exc)
        # The measured share of the AOI that holds imagery beats the footprint estimate (bounding boxes of
        # tilted scenes overstate coverage).
        fraction = max(0.0, min(1.0, extent, data_coverage if data_coverage is not None else footprint))
        entry = {
            "image": label,
            "aoi_coverage_fraction": round(fraction, 4),
            "scenes_used": [str(getattr(r, "id", "?")) for r in prepared],
            "scenes_failed": list(failed_ids),
        }
        if clear_fraction is not None:
            entry["clear_fraction"] = round(clear_fraction, 4)
        self._coverage_log.append(entry)
        if fraction < 0.999:
            logger.warning(
                "_search_and_download: the %s image covers only %.0f%% of the requested bbox %s.",
                label, 100 * fraction, bbox,
            )

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
                from pygeovision.data._profile import writable_profile
                out_profile = writable_profile(
                    src.profile, height=out_h, width=out_w, transform=out_transform, crs=crs,
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
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path} -- pass the path of a trained .pth file for "
                f"'{self.name}' (names such as buildings.pth in examples are placeholders; "
                f"`pygeovision channel list` shows which pipelines need your own weights)")

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
                from pygeovision.data._dates import date_window as _date_window
                lbl = DynamicWorldLabeler(start_date=_date_window(date)[0], end_date=_date_window(date)[1])
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