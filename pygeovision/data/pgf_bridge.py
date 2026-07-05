"""
pygeovision.data.pgf_bridge
===========================

Definitive bridge between PyGeoVision and PyGeoFetch.

Architecture
------------
PyGeoFetch v1.0 (installed) provides:
  • search()   — SearchQuery + BoundingBox → List[SatelliteData]
  • download() — SatelliteData + DownloadOptions → List[DownloadResult]
  • auth       — keyring-backed credential store
  • cache      — 1-hour search result cache

PyGeoFetch v2.0 (README / future) will add:
  • client.preprocess.*  — atmos, cloud_mask, clip, reproject, resample …
  • client.indices.*     — ndvi, evi, nbr, tct, pca …
  • client.post.*        — vectorize, smooth, zonal_stats, cog …
  • client.sar.*         — despeckle, calibrate, flood_map, coherence

PyGeoVision owns:
  • Preprocessor          — stack, clip, cloud-mask, normalise, resample (own rasterio impl)
  • SpectralIndices        — 22 indices (own numpy impl)
  • PostProcessor          — vectorise, sieve, accuracy, zonal_stats (own impl)
  • DataValidator          — mandatory validation gate before AI
  • GeoAI plugin (optional)

This bridge:
  1. Uses PyGeoFetch's real Python API for search/download (not CLI).
  2. Translates PyGeoFetch ``SatelliteData`` ↔ PyGeoVision ``SearchResult``.
  3. Translates PyGeoFetch ``DownloadResult`` ↔ PyGeoVision ``DownloadResult``.
  4. Runs the mandatory preprocessing chain after download and
     before any AI model.
  5. Is forward-compatible: when PyGeoFetch v2.0 ships,
     ``_pgf2_available()`` returns True and the native methods are used.

Usage::

    from pygeovision.data.pgf_bridge import PyGeoFetchBridge

    bridge = PyGeoFetchBridge()

    # Search (uses PyGeoFetch Python API)
    results = bridge.search(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        date_range=("2024-01-01", "2024-06-01"),
        providers=["planetary_computer"],
        cloud_cover_max=10,
        limit=5,
    )

    # Download (uses PyGeoFetch Python API)
    downloads = bridge.download(
        results,
        output_dir="./data/",
        parallel=4,
        bands=["B02","B03","B04","B08","B11","B12"],
        post_process=["reproject:EPSG:4326", "cog"],
    )

    # Preprocess ready for AI (uses PGV Preprocessor, validated)
    for dl in downloads:
        ready = bridge.prepare_for_ai(
            dl.path,
            bbox=(-74.1, 40.6, -73.7, 40.9),
            stack_bands=["B02","B03","B04","B08","B11","B12"],
            scl_path=None,
            normalise="scale_factor",
            model_type="segmentation",
        )
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyGeoFetch availability
# ---------------------------------------------------------------------------

def _pgf_available() -> bool:
    """True when PyGeoFetch v1.0+ Python API is importable."""
    try:
        from pygeofetch import PyGeoFetch       # noqa: F401
        return True
    except ImportError:
        return False


def _pgf2_available() -> bool:
    """True when PyGeoFetch v2.0+ processing engine is available."""
    try:
        from pygeofetch import PyGeoFetch
        c = PyGeoFetch()
        return hasattr(c, "preprocess") and hasattr(c.preprocess, "atmos")
    except Exception:
        return False


def _get_pgf_client():
    """Return a PyGeoFetch Python API client, or None."""
    try:
        from pygeofetch import PyGeoFetch
        return PyGeoFetch()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Type translation helpers
# ---------------------------------------------------------------------------

def _pgf_satellite_data_to_pgv(sd, provider_hint: str = "") -> "SearchResult":
    """Convert PyGeoFetch ``SatelliteData`` to PyGeoVision ``SearchResult``."""
    from pygeovision.data.fetch import SearchResult

    cloud = getattr(sd, "cloud_cover", None)
    sat   = getattr(sd, "satellite", None) or ""
    coll  = getattr(sd, "collection", None) or ""
    bbox  = getattr(sd, "bbox", None)
    props = getattr(sd, "properties", {}) or {}
    score = float(getattr(sd, "score", 0.0) or 0.0)
    prov  = getattr(sd, "provider", provider_hint)

    dt = getattr(sd, "datetime", None)
    if dt is None:
        dt = getattr(sd, "start_datetime", None)
    datetime_str = dt.isoformat() if dt else ""

    # SatelliteAsset has: key, href, roles, extra_fields (all required)
    # Convert to plain dicts with all needed fields for cache/download
    raw_assets = getattr(sd, "assets", {}) or {}
    assets = {}
    for k, v in raw_assets.items():
        if hasattr(v, "href"):
            # Native SatelliteAsset — preserve key + href + roles
            assets[k] = {
                "key":   getattr(v, "key", k),
                "href":  str(v.href),
                "roles": list(getattr(v, "roles", []) or []),
                "title": getattr(v, "title", None),
                "media_type": getattr(v, "media_type", None),
            }
        elif isinstance(v, dict):
            # Already a dict — ensure key field present
            assets[k] = {"key": k, **v}
        else:
            assets[k] = {"key": k, "href": str(v), "roles": []}

    result = SearchResult(
        id           = str(getattr(sd, "id", "")),
        provider     = str(prov),
        satellite    = str(sat),
        collection   = str(coll),
        datetime     = datetime_str,
        cloud_cover  = float(cloud) if cloud is not None else 0.0,
        bbox         = tuple(bbox) if bbox else None,
        score        = score,
        assets       = assets,
        properties   = props,
    )
    result.satellite_data = sd
    return result


def _pgf_download_result_to_pgv(dr) -> "DownloadResult":
    """Convert PyGeoFetch ``DownloadResult`` to PyGeoVision ``DownloadResult``."""
    from pygeovision.data.fetch import DownloadResult

    try:
        from pygeofetch.models.download_task import DownloadStatus
        status  = getattr(dr, "status", None)
        success = (status == DownloadStatus.COMPLETED) if status else False
    except ImportError:
        success = getattr(dr, "success", False)

    # Resolve output path — PyGeoFetch uses output_path and output_paths
    output_path  = getattr(dr, "output_path",  None)
    output_paths = list(getattr(dr, "output_paths", None) or [])
    if output_path is None and output_paths:
        output_path = output_paths[0]

    error_    = getattr(dr, "error", None)
    size_mb   = float(getattr(dr, "bytes_downloaded", 0) or 0) / 1024 / 1024
    duration  = float(getattr(dr, "duration_seconds", 0.0) or 0.0)

    # PGV DownloadResult fields: scene_id, provider, path, success,
    # bytes_downloaded, duration_seconds, checksum_verified, error
    return DownloadResult(
        scene_id          = str(getattr(dr, "data_id", "")),
        provider          = str(getattr(dr, "provider", "")),
        path              = pathlib.Path(output_path) if output_path else None,
        success           = success,
        bytes_downloaded  = int(getattr(dr, "bytes_downloaded", 0) or 0),
        duration_seconds  = duration,
        checksum_verified = bool(getattr(dr, "checksum_verified", False) or False),
        error             = str(error_) if error_ else "",
    )


# ---------------------------------------------------------------------------
# PyGeoFetchBridge
# ---------------------------------------------------------------------------

class PyGeoFetchBridge:
    """Definitive integration bridge between PyGeoVision and PyGeoFetch.

    All search and download operations go through PyGeoFetch's Python API
    when it is installed.  Preprocessing, spectral indices, and
    postprocessing use PyGeoFetch v2.0 natively when available, or
    PyGeoVision's own implementations as a validated fallback.

    Args:
        validator: :class:`DataValidator` instance used to validate
            data before and after each processing step.
        pgv_preprocessor: PyGeoVision :class:`Preprocessor` (fallback).
        pgv_indices: PyGeoVision :class:`SpectralIndices` (fallback).
        pgv_postprocessor: PyGeoVision :class:`PostProcessor` (fallback).
    """

    def __init__(
        self,
        validator=None,
        pgv_preprocessor=None,
        pgv_indices=None,
        pgv_postprocessor=None,
    ):
        self._pgf     = _get_pgf_client()
        self._pgf_v2  = _pgf2_available()
        self._v       = validator
        self._pre     = pgv_preprocessor
        self._idx     = pgv_indices
        self._post    = pgv_postprocessor

        if self._pgf:
            logger.info("PyGeoFetch v1.0 Python API active — using native search/download")
        else:
            logger.info("PyGeoFetch not installed — falling back to pystac-client")

        if self._pgf_v2:
            logger.info("PyGeoFetch v2.0 processing engine detected — using native preprocessing")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Optional[Tuple[str, str]] = None,
        providers: Optional[List[str]] = None,
        cloud_cover_max: float = 100.0,
        cloud_cover_min: float = 0.0,
        max_results: int = 100,
        limit: Optional[int] = None,
        sort_by: str = "cloud_cover",
        sort_ascending: bool = True,
        collections: Optional[List[str]] = None,
        satellites: Optional[List[str]] = None,
        cql2_filter: Optional[str] = None,
        use_cache: bool = True,
        on_provider_failure: str = "skip",
        timeout_seconds: int = 60,
    ) -> List["SearchResult"]:
        """Search satellite imagery via PyGeoFetch's Python API.

        Translates PyGeoVision's keyword-based interface to
        ``pygeofetch.SearchQuery`` and returns ``SearchResult`` objects.

        Args:
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` WGS84.
            date_range: ``(start_date, end_date)`` ISO strings.
            providers: List of provider IDs (e.g. ``["planetary_computer"]``).
            cloud_cover_max: Max cloud cover percent (0–100).
            cloud_cover_min: Min cloud cover percent (default 0).
            max_results: Maximum results per provider.
            limit: Alias for ``max_results``.
            sort_by: ``"cloud_cover"`` | ``"datetime"`` | ``"score"``.
            sort_ascending: Sort direction (True=ascending).
            collections: STAC collection filter.
            satellites: Satellite name filter.
            cql2_filter: CQL2 expression for advanced filtering.
            use_cache: Use PyGeoFetch's 1-hour search cache.
            on_provider_failure: ``"skip"`` | ``"abort"`` | ``"retry"``.
            timeout_seconds: Per-provider timeout.

        Returns:
            List of :class:`SearchResult` objects, sorted by score.

        Example::

            results = bridge.search(
                bbox=(-74.1, 40.6, -73.7, 40.9),
                date_range=("2024-06-01", "2024-06-30"),
                providers=["planetary_computer", "copernicus"],
                cloud_cover_max=10,
            )
        """
        if limit is not None:
            max_results = limit

        if self._pgf is None:
            logger.warning("PyGeoFetch not available — returning empty results")
            return []

        # Build pygeofetch SearchQuery
        from pygeofetch.models.search_query import SearchQuery, BoundingBox

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        pgf_bbox = BoundingBox.from_string(bbox_str)

        start_d = end_d = None
        if date_range:
            from datetime import date as _date
            try:
                start_d = _date.fromisoformat(str(date_range[0])[:10])
                end_d   = _date.fromisoformat(str(date_range[1])[:10])
            except (ValueError, TypeError):
                pass

        sort_field = sort_by if sort_by in ("cloud_cover", "datetime", "score") else "datetime"

        query = SearchQuery(
            bbox               = pgf_bbox,
            start_date         = start_d,
            end_date           = end_d,
            cloud_cover_min    = cloud_cover_min,
            cloud_cover_max    = cloud_cover_max,
            max_results        = max_results,
            sort_by            = sort_field,
            sort_ascending     = sort_ascending,
            collections        = collections or [],
            satellites         = satellites or [],
            sensors            = [],
            processing_levels  = [],
            providers          = providers or [],
            provider_filters   = {},
            ids                = [],
            cql2_filter        = cql2_filter,
            on_provider_failure= on_provider_failure,
            timeout_seconds    = timeout_seconds,
        )

        try:
            pgf_results = self._pgf.search(query, providers=providers, use_cache=use_cache)
        except Exception as e:
            logger.error("PyGeoFetch search failed: %s", e)
            return []

        results = [_pgf_satellite_data_to_pgv(sd) for sd in (pgf_results or [])]
        logger.info("PyGeoFetch search → %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        items: List["SearchResult"],
        output_dir: Union[str, pathlib.Path],
        parallel: int = 4,
        bands: Optional[List[str]] = None,
        post_process: Optional[List[str]] = None,
        verify_checksum: bool = True,
        resume: bool = True,
        retry_attempts: int = 3,
        bandwidth_limit_mb: Optional[float] = None,
        on_failure: str = "skip",
        overwrite: bool = False,
        notify_webhook: Optional[str] = None,
    ) -> List["DownloadResult"]:
        """Download satellite imagery via PyGeoFetch's Python API.

        Translates PyGeoVision's download interface to
        ``pygeofetch.DownloadOptions``, then translates
        ``pygeofetch.DownloadResult`` back.

        Args:
            items: List of :class:`SearchResult` (must have ``.satellite_data``).
            output_dir: Destination directory.
            parallel: Concurrent download workers.
            bands: Band names to download (e.g. ``["B02","B03","B04","B08"]``).
                ``None`` = all assets.
            post_process: PyGeoFetch post-process action strings,
                e.g. ``["reproject:EPSG:4326", "cog"]``.
            verify_checksum: MD5 checksum verification.
            resume: Resume interrupted downloads.
            retry_attempts: Max retry attempts.
            bandwidth_limit_mb: Throttle in MB/s (0 = unlimited).
            on_failure: ``"skip"`` | ``"abort"`` | ``"retry"``.
            overwrite: Overwrite existing files.
            notify_webhook: Slack / webhook URL for completion notification.

        Returns:
            List of :class:`DownloadResult`.

        Example::

            downloads = bridge.download(
                results[:3],
                output_dir="./data/",
                parallel=4,
                bands=["B02","B03","B04","B08","B11","B12"],
                post_process=["reproject:EPSG:4326", "cog"],
            )
        """
        if self._pgf is None:
            logger.warning("PyGeoFetch not available — cannot download")
            return []

        from pygeofetch.models.download_task import DownloadOptions, PostProcessAction

        # Build post-process actions
        pp_actions = []
        for step in (post_process or []):
            if ":" in step:
                action, param = step.split(":", 1)
                pp_actions.append(PostProcessAction(action=action, params={"value": param}))
            else:
                pp_actions.append(PostProcessAction(action=step, params={}))

        opts = DownloadOptions(
            parallel           = parallel,
            verify_checksum    = verify_checksum,
            resume             = resume,
            retry_attempts     = retry_attempts,
            bandwidth_limit_mbps = bandwidth_limit_mb or 0,
            on_failure         = on_failure,
            overwrite          = overwrite,
            notify_webhook     = notify_webhook,
            post_process       = pp_actions,
        )

        # Filter assets by band if requested
        sat_data_list = []
        for item in items:
            sd = getattr(item, "satellite_data", None)
            if sd is None:
                logger.warning("Item %s has no satellite_data — skipping", item.id)
                continue
            if bands:
                # Filter sd.assets to only requested bands
                if hasattr(sd, "assets") and sd.assets:
                    sd_copy = sd.model_copy(deep=True)
                    sd_copy.assets = {
                        k: v for k, v in sd.assets.items()
                        if k in bands or any(b.lower() in k.lower() for b in bands)
                    }
                    sat_data_list.append(sd_copy)
                else:
                    sat_data_list.append(sd)
            else:
                sat_data_list.append(sd)

        if not sat_data_list:
            logger.warning("No valid SatelliteData objects to download")
            return []

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            pgf_results = self._pgf.download(
                sat_data_list,
                destination=pathlib.Path(output_dir),
                options=opts,
            )
        except Exception as e:
            logger.error("PyGeoFetch download failed: %s", e)
            return []

        results = [_pgf_download_result_to_pgv(dr) for dr in (pgf_results or [])]
        success_n = sum(1 for r in results if r.success)
        logger.info("PyGeoFetch download: %d/%d succeeded", success_n, len(results))
        return results

    # ------------------------------------------------------------------
    # Auth passthrough
    # ------------------------------------------------------------------

    def add_credentials(self, provider: str, **kwargs) -> bool:
        """Add provider credentials via PyGeoFetch keyring."""
        if self._pgf is None:
            logger.warning("PyGeoFetch not available — credentials not stored")
            return False
        try:
            self._pgf.add_credentials(provider, **kwargs)
            logger.info("Credentials added for %s via PyGeoFetch", provider)
            return True
        except Exception as e:
            logger.error("add_credentials(%s) failed: %s", provider, e)
            return False

    # ------------------------------------------------------------------
    # prepare_for_ai: post-download preprocessing gate
    # ------------------------------------------------------------------

    def prepare_for_ai(
        self,
        input_path: str,
        *,
        # Stacking
        stack_bands: Optional[List[str]] = None,
        stack_dir: Optional[str] = None,
        # Spatial
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_crs: str = "EPSG:4326",
        clip_geojson: Optional[str] = None,
        # Cloud masking
        cloud_mask_path: Optional[str] = None,
        scl_path: Optional[str] = None,
        scl_keep_classes: Sequence[int] = (4, 5, 6),
        # Normalisation
        normalise: Optional[str] = "scale_factor",
        scale_factor: float = 10000.0,
        # Resampling
        resample_m: Optional[float] = None,
        # Validation
        model_type: str = "segmentation",
        output_path: Optional[str] = None,
        # v2.0 native preprocessing (used when available)
        use_pgf_preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Complete preprocessing pipeline: download output → AI-ready array.

        This is the **mandatory gate** between raw satellite data and AI
        models.  Every dataset passes through here before inference.

        Order of operations:
            1. Stack bands (if scene_dir contains individual band files)
            2. Clip to study area (bbox or polygon)
            3. Cloud / SCL masking
            4. Normalise pixel values
            5. Resample to target resolution
            6. DataValidator — auto-fix nulls, outliers, dtype
            7. Return AI-ready float32 numpy array

        When PyGeoFetch v2.0 is detected, native preprocessing is used
        for steps 1–5 (atmospheric correction, cloud mask, clip, etc.).
        Otherwise PyGeoVision's own Preprocessor is used.

        Args:
            input_path: Stacked GeoTIFF **or** scene directory.
            stack_bands: Band names to stack from ``input_path``/``stack_dir``.
            stack_dir: Alternative directory for band discovery.
            bbox: Clip bbox ``(minlon,minlat,maxlon,maxlat)`` WGS84.
            bbox_crs: CRS of the bbox (default EPSG:4326).
            clip_geojson: Path to GeoJSON polygon for clipping.
            cloud_mask_path: Binary cloud mask (1=cloud,0=clear).
            scl_path: Sentinel-2 SCL band for cloud/shadow masking.
            scl_keep_classes: SCL classes to keep (default veg+soil+water).
            normalise: ``"scale_factor"`` | ``"minmax"`` | ``"zscore"``
                | ``"percentile"`` | ``None``.
            scale_factor: Divisor for ``"scale_factor"`` (10000 for S2 L2A).
            resample_m: Target pixel size in metres.
            model_type: Model type hint for DataValidator.
            output_path: If set, saves the preprocessed file here.
            use_pgf_preprocess: Use PyGeoFetch v2.0 preprocessing when
                available (set False to always use PGV's implementation).

        Returns:
            Dict with:
              - ``"array"``       : float32 numpy array ``(C, H, W)``
              - ``"output_path"`` : path of the saved preprocessed file
              - ``"shape"``       : ``(C, H, W)``
              - ``"resolution_m"``: pixel size of the preprocessed file
              - ``"report"``      : :class:`ValidationReport`
              - ``"steps"``       : preprocessing steps applied

        Example::

            result = bridge.prepare_for_ai(
                "sentinel2_scene.tif",
                bbox=(-74.1, 40.6, -73.7, 40.9),
                scl_path="SCL.tif",
                normalise="scale_factor",
                model_type="segmentation",
                output_path="ready.tif",
            )
            arr = result["array"]   # float32 (C,H,W), validated, AI-ready
        """
        steps_applied = []

        # ── PyGeoFetch v2.0 native preprocessing (future-proof path) ────
        if use_pgf_preprocess and self._pgf_v2:
            return self._prepare_via_pgf_v2(
                input_path, stack_bands=stack_bands, bbox=bbox,
                bbox_crs=bbox_crs,                     # ← was missing
                clip_geojson=clip_geojson, cloud_mask_path=cloud_mask_path,
                scl_path=scl_path, normalise=normalise,
                scale_factor=scale_factor, resample_m=resample_m,
                model_type=model_type, output_path=output_path,
            )

        # ── PyGeoVision own Preprocessor (current path) ─────────────────
        if self._pre is None:
            from pygeovision.preprocess import Preprocessor
            self._pre = Preprocessor(validator=self._v)

        # Determine final output path
        if output_path is None:
            p = pathlib.Path(input_path if not pathlib.Path(input_path).is_dir()
                              else f"{input_path}/scene")
            output_path = str(p.parent / f"{p.stem}_ai_ready.tif")

        result = self._pre.pipeline(
            input_path   = input_path,
            output_path  = output_path,
            stack_bands  = stack_bands,
            stack_dir    = stack_dir,
            bbox         = bbox,
            bbox_crs     = bbox_crs,
            clip_geojson = clip_geojson,
            cloud_mask_path  = cloud_mask_path,
            scl_path         = scl_path,
            scl_keep_classes = scl_keep_classes,
            normalise        = normalise,
            scale_factor     = scale_factor,
            resample_m       = resample_m,
        )
        steps_applied = result.get("steps_applied", [])

        # ── Mandatory DataValidator gate ─────────────────────────────────
        arr = None
        report = None
        if self._v and pathlib.Path(output_path).exists():
            arr    = self._v.validate_for_inference(output_path, model_type=model_type)
            report = self._v.validate(output_path)
            steps_applied.append(f"validate({model_type})")

        return {
            "array":        arr,
            "output_path":  output_path,
            "shape":        result.get("shape"),
            "resolution_m": result.get("resolution_m"),
            "report":       report,
            "steps":        steps_applied,
        }

    def _prepare_via_pgf_v2(self, input_path, **kw) -> Dict[str, Any]:
        """Use PyGeoFetch v2.0 native preprocessing when available."""
        pgf_pre  = self._pgf.preprocess
        steps    = []
        current  = input_path

        if kw.get("bbox"):
            bbox      = kw["bbox"]
            bbox_crs  = kw.get("bbox_crs", "EPSG:4326")

            # ── Reproject bbox to match the raster's native CRS ──────────────
            # The scene may have been reprojected (e.g. to UTM) during download.
            # Feeding WGS84 degrees to a UTM scene → WindowError / no overlap.
            bbox_native = _reproject_bbox(bbox, bbox_crs, current)

            r       = pgf_pre.clip(current, bbox=bbox_native)
            current = str(r.output_path)
            steps.append(f"pgf.clip({bbox})")

        if kw.get("cloud_mask_path"):
            r = pgf_pre.cloud_mask(current, method="threshold")
            current = str(r.output_path)
            steps.append("pgf.cloud_mask")

        if kw.get("scl_path"):
            r = pgf_pre.cloud_mask(current, method="scl", scl_band=kw["scl_path"])
            current = str(r.output_path)
            steps.append("pgf.cloud_mask_scl")

        if kw.get("normalise") == "scale_factor":
            sf = kw.get("scale_factor", 10000.0)
            r  = pgf_pre.atmos(current, method="dos1")
            current = str(r.output_path)
            steps.append(f"pgf.atmos_dos1")

        if kw.get("resample_m"):
            r = pgf_pre.resample(current, resolution=int(kw["resample_m"]))
            current = str(r.output_path)
            steps.append(f"pgf.resample({kw['resample_m']}m)")

        op = kw.get("output_path") or current
        if op != current:
            import shutil
            shutil.copy2(current, op)

        arr    = None
        report = None
        if self._v:
            model_type = kw.get("model_type", "segmentation")
            arr    = self._v.validate_for_inference(op, model_type=model_type)
            report = self._v.validate(op)
            steps.append(f"validate({model_type})")

        return {
            "array": arr, "output_path": op,
            "shape": arr.shape if arr is not None else None,
            "resolution_m": None, "report": report, "steps": steps,
        }

    # ------------------------------------------------------------------
    # Preprocessing proxy (v2.0 native / PGV fallback)
    # ------------------------------------------------------------------

    def preprocess_atmos(self, input_path: str, method: str = "dos1", **kw) -> str:
        """Atmospheric correction.

        Uses PyGeoFetch v2.0 ``preprocess.atmos()`` when available.
        PyGeoVision does not implement full atmospheric correction —
        use ``normalise("scale_factor")`` as an approximation.

        Args:
            input_path: Input GeoTIFF.
            method: ``"dos1"`` (Dark Object Subtraction) | ``"sen2cor"``
                | ``"flaash"``.

        Returns:
            Output path.
        """
        if self._pgf_v2:
            r = self._pgf.preprocess.atmos(input_path, method=method, **kw)
            return str(r.output_path)
        # PGV approximation: scale-factor normalisation acts like DOS1 correction
        if self._pre:
            logger.info(
                "PyGeoFetch v2.0 not available — using scale_factor normalisation "
                "as DOS1 approximation"
            )
            return self._pre.normalise(input_path, method="scale_factor")
        return input_path

    def preprocess_cloud_mask(
        self, input_path: str, method: str = "scl",
        scl_band: Optional[str] = None, **kw
    ) -> str:
        """Cloud masking.

        Uses PyGeoFetch v2.0 ``preprocess.cloud_mask()`` when available.
        Falls back to PGV ``apply_scl()`` or ``apply_cloud_mask()``.
        """
        if self._pgf_v2:
            r = self._pgf.preprocess.cloud_mask(
                input_path, method=method, scl_band=scl_band, **kw)
            return str(r.output_path)
        if self._pre:
            if method == "scl" and scl_band:
                return self._pre.apply_scl(input_path, scl_path=scl_band)
            elif scl_band:
                return self._pre.apply_cloud_mask(input_path, mask_path=scl_band)
        return input_path

    def preprocess_clip(
        self, input_path: str,
        bbox: Optional[Tuple] = None,
        geometry: Optional[str] = None,
        output_path: Optional[str] = None,
        bbox_crs: str = "EPSG:4326",
        **kw
    ) -> str:
        """Clip to bounding box or polygon.

        The bbox is automatically reprojected to the raster's native CRS
        when they differ — so you can always supply WGS84 coordinates
        regardless of what CRS the scene was downloaded in.
        """
        if self._pgf_v2:
            if bbox is not None:
                # Reproject supplied bbox to the raster's native CRS
                bbox_native = _reproject_bbox(bbox, bbox_crs, input_path)
                r = self._pgf.preprocess.clip(input_path, bbox=bbox_native, **kw)
            elif geometry is not None:
                r = self._pgf.preprocess.clip(input_path, geometry=geometry, **kw)
            else:
                return input_path
            return str(r.output_path)
        if self._pre:
            if geometry:
                return self._pre.clip_to_polygon(input_path, geometry,
                                                   output_path=output_path)
            if bbox:
                return self._pre.clip_to_bbox(input_path, bbox,
                                               output_path=output_path,
                                               bbox_crs=bbox_crs)
        return input_path

    def preprocess_reproject(self, input_path: str, crs: str = "EPSG:4326", **kw) -> str:
        """Reproject to target CRS."""
        if self._pgf_v2:
            r = self._pgf.preprocess.reproject(input_path, crs=crs, **kw)
            return str(r.output_path)
        # Use rasterio fallback
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            from rasterio.crs import CRS
            import numpy as np
            out = _auto_suffix(input_path, f"_{crs.replace(':','')}")
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(input_path) as src:
                dst_crs = CRS.from_string(crs)
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                profile = src.profile.copy()
                profile.update(crs=dst_crs, transform=transform,
                               width=width, height=height, compress="lzw")
                with rasterio.open(out, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, i),
                                  destination=rasterio.band(dst, i),
                                  src_transform=src.transform,
                                  src_crs=src.crs,
                                  dst_transform=transform,
                                  dst_crs=dst_crs,
                                  resampling=Resampling.bilinear)
            return out
        except Exception as e:
            logger.warning("Reproject fallback failed: %s", e)
            return input_path

    def preprocess_resample(
        self, input_path: str, resolution: float,
        method: str = "bilinear", **kw
    ) -> str:
        """Resample to target resolution in metres."""
        if self._pgf_v2:
            r = self._pgf.preprocess.resample(
                input_path, resolution=int(resolution), method=method, **kw)
            return str(r.output_path)
        if self._pre:
            return self._pre.resample(input_path, resolution, resampling=method)
        return input_path

    def preprocess_pansharpen(
        self, pan: str, ms: str,
        method: str = "brovey", **kw
    ) -> str:
        """Pan-sharpening (brovey or IHS)."""
        if self._pgf_v2:
            r = self._pgf.preprocess.pansharpen(pan=pan, ms=ms, method=method, **kw)
            return str(r.output_path)
        logger.warning("Pan-sharpening requires PyGeoFetch v2.0 or GDAL pansharpen")
        return ms

    def preprocess_tile(
        self, input_path: str, tile_size: int = 512,
        overlap: int = 64, output_dir: Optional[str] = None, **kw
    ) -> List[str]:
        """Cut a scene into chips for training."""
        if self._pgf_v2:
            r = self._pgf.preprocess.tile(
                input_path, tile_size=tile_size, overlap=overlap, **kw)
            return [str(p) for p in (r.output_paths or [str(r.output_path)])]
        # PGV fallback: return the original scene (caller handles tiling via TiledInference)
        logger.info("tile() → TiledInference handles tiling at inference time")
        return [input_path]

    def preprocess_mosaic(self, inputs: List[str], method: str = "first", **kw) -> str:
        """Mosaic multiple scenes."""
        if self._pgf_v2:
            r = self._pgf.preprocess.mosaic(inputs, method=method, **kw)
            return str(r.output_path)
        try:
            import rasterio
            from rasterio.merge import merge as rio_merge
            datasets = [rasterio.open(p) for p in inputs]
            mosaic, transform = rio_merge(datasets)
            for ds in datasets: ds.close()
            out = _auto_suffix(inputs[0], "_mosaic")
            with rasterio.open(inputs[0]) as ref:
                profile = ref.profile.copy()
                profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                               transform=transform, compress="lzw")
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(mosaic)
            return out
        except Exception as e:
            logger.warning("mosaic fallback failed: %s", e)
            return inputs[0]

    def preprocess_composite(
        self, inputs: List[str], method: str = "median",
        output_path: Optional[str] = None, **kw
    ) -> str:
        """Temporal composite (median / mean / max / best-pixel)."""
        if self._pgf_v2:
            r = self._pgf.preprocess.composite(inputs, method=method, **kw)
            return str(r.output_path)
        try:
            import rasterio, numpy as np
            arrays = []
            for p in inputs:
                with rasterio.open(p) as src:
                    arrays.append(src.read().astype(np.float32))
            stack = np.stack(arrays, axis=0)           # (T, C, H, W)
            if method == "median":
                comp = np.median(stack, axis=0)
            elif method == "mean":
                comp = np.mean(stack, axis=0)
            elif method == "max":
                comp = np.max(stack, axis=0)
            else:
                comp = np.median(stack, axis=0)
            out = output_path or _auto_suffix(inputs[0], f"_{method}")
            with rasterio.open(inputs[0]) as ref:
                profile = ref.profile.copy()
                profile.update(dtype="float32", compress="lzw")
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(comp)
            return out
        except Exception as e:
            logger.warning("composite fallback failed: %s", e)
            return inputs[0]

    def preprocess_topo_correct(
        self, input_path: str, dem: str,
        method: str = "c_correction", **kw
    ) -> str:
        """Topographic correction using a DEM."""
        if self._pgf_v2:
            r = self._pgf.preprocess.topo_correct(
                input_path, dem=dem, method=method, **kw)
            return str(r.output_path)
        logger.warning("topo_correct() requires PyGeoFetch v2.0")
        return input_path

    def preprocess_cloud_fill(
        self, cloudy: str, time_series: List[str], **kw
    ) -> str:
        """Fill cloud gaps using a temporal series."""
        if self._pgf_v2:
            r = self._pgf.preprocess.cloud_fill(cloudy, time_series=time_series, **kw)
            return str(r.output_path)
        logger.warning("cloud_fill() requires PyGeoFetch v2.0")
        return cloudy

    # ------------------------------------------------------------------
    # Spectral indices proxy (v2.0 native / PGV fallback)
    # ------------------------------------------------------------------

    def index(self, name: str, output_path: Optional[str] = None, **band_paths) -> str:
        """Compute a spectral index.

        When PyGeoFetch v2.0 is available, delegates to
        ``client.indices.{name}()``.  Otherwise uses PyGeoVision's
        own :class:`SpectralIndices`.

        Args:
            name: Index name (``"ndvi"``, ``"evi"``, ``"nbr"`` etc.).
            output_path: Output GeoTIFF path.
            **band_paths: Band file paths — e.g.
                ``red="B04.tif"``, ``nir="B08.tif"``.

        Returns:
            Output path string.

        Example::

            out = bridge.index("ndvi", red="B04.tif", nir="B08.tif",
                               output_path="ndvi.tif")
        """
        if self._pgf_v2 and hasattr(self._pgf, "indices"):
            fn = getattr(self._pgf.indices, name, None)
            if fn:
                r = fn(**band_paths)
                return str(r.output_path)

        # PGV fallback — indices take a stacked GeoTIFF + band indices
        if self._idx:
            # Convert file paths to band indices when needed
            logger.info("Using PyGeoVision SpectralIndices for '%s'", name)
            fn = getattr(self._idx, name, None)
            if fn:
                source = next(iter(band_paths.values()), None)
                if source and output_path:
                    return fn(source, output_path=output_path) or output_path
        logger.warning("Cannot compute index '%s' — no backend available", name)
        return ""

    # ------------------------------------------------------------------
    # SAR proxy (v2.0 native only — no PGV fallback)
    # ------------------------------------------------------------------

    def sar_despeckle(
        self, input_path: str, filter: str = "enhanced_lee",
        window: int = 5, **kw
    ) -> str:
        """SAR speckle filtering."""
        if self._pgf_v2 and hasattr(self._pgf, "sar"):
            r = self._pgf.sar.despeckle(input_path, filter=filter, window=window, **kw)
            return str(r.output_path)
        logger.warning("sar.despeckle() requires PyGeoFetch v2.0")
        return input_path

    def sar_calibrate(
        self, input_path: str, output_type: str = "sigma0",
        in_db: bool = True, **kw
    ) -> str:
        """SAR radiometric calibration."""
        if self._pgf_v2 and hasattr(self._pgf, "sar"):
            r = self._pgf.sar.calibrate(
                input_path, output_type=output_type, in_db=in_db, **kw)
            return str(r.output_path)
        logger.warning("sar.calibrate() requires PyGeoFetch v2.0")
        return input_path

    def sar_flood_map(
        self, post_path: str, threshold: float = -15.0,
        reference: Optional[str] = None, **kw
    ) -> str:
        """SAR flood extent mapping."""
        if self._pgf_v2 and hasattr(self._pgf, "sar"):
            r = self._pgf.sar.flood_map(
                post_path, threshold=threshold, reference=reference, **kw)
            return str(r.output_path)
        logger.warning("sar.flood_map() requires PyGeoFetch v2.0")
        return post_path

    def sar_coherence(
        self, slc1: str, slc2: str, window: int = 7, **kw
    ) -> str:
        """InSAR coherence map."""
        if self._pgf_v2 and hasattr(self._pgf, "sar"):
            r = self._pgf.sar.coherence(slc1, slc2, window=window, **kw)
            return str(r.output_path)
        logger.warning("sar.coherence() requires PyGeoFetch v2.0")
        return slc1

    # ------------------------------------------------------------------
    # Pipeline builder proxy
    # ------------------------------------------------------------------

    def pipeline(self, name: str) -> Any:
        """Return PyGeoFetch's chainable pipeline builder (v2.0).

        When PyGeoFetch v1.0 is installed (no pipeline builder),
        returns a simple :class:`_SimpleChain` stub.

        Example::

            result = (
                bridge.pipeline("sentinel2-workflow")
                .atmos(method="dos1")
                .cloud_mask(method="scl", scl_band="SCL.tif")
                .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
                .ndvi(red="B04.tif", nir="B08.tif")
                .cog(compress="deflate")
                .run(input="scene.tif", output_dir="./processed/")
            )
        """
        if self._pgf_v2 and hasattr(self._pgf, "pipeline"):
            return self._pgf.pipeline(name)
        return _SimpleChain(name, self)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_process(
        self,
        inputs: List[str],
        chain: List[Tuple[str, Dict]],
        output_dir: str,
        parallel: int = 4,
    ) -> List[Dict]:
        """Run a preprocessing chain on multiple scenes in parallel.

        When PyGeoFetch v2.0 is available, delegates to
        ``client.batch_process()``.  Otherwise runs the chain
        sequentially using PyGeoVision's own implementations.

        Args:
            inputs: List of input GeoTIFF paths.
            chain: List of ``(operation_name, kwargs)`` tuples. E.g.::

                [
                    ("clip",      {"bbox": (-74.1, 40.6, -73.7, 40.9)}),
                    ("reproject", {"crs": "EPSG:4326"}),
                    ("normalise", {"method": "scale_factor"}),
                ]

            output_dir: Directory for processed outputs.
            parallel: Concurrent workers (used by PyGeoFetch v2.0).

        Returns:
            List of dicts with ``{"input", "output", "success", "error"}``.
        """
        if self._pgf_v2 and hasattr(self._pgf, "batch_process"):
            pgf_chain = [(op, kw) for op, kw in chain]
            return self._pgf.batch_process(inputs, pgf_chain,
                                            output_dir=output_dir, parallel=parallel)

        # Sequential fallback
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        for inp in inputs:
            current = inp
            error   = None
            try:
                for op_name, op_kw in chain:
                    if op_name == "clip":
                        current = self.preprocess_clip(current, **op_kw)
                    elif op_name == "reproject":
                        current = self.preprocess_reproject(current, **op_kw)
                    elif op_name == "resample":
                        current = self.preprocess_resample(
                            current, resolution=op_kw.get("resolution_m", 10.0))
                    elif op_name == "normalise" and self._pre:
                        current = self._pre.normalise(current, **op_kw)
                    elif op_name == "cloud_mask":
                        current = self.preprocess_cloud_mask(current, **op_kw)
                    elif op_name == "atmos":
                        current = self.preprocess_atmos(current, **op_kw)
                    elif op_name == "mosaic":
                        pass  # mosaic needs multiple inputs
                    elif op_name == "cog" and self._post:
                        current = self._post.to_cog(current)
                    else:
                        logger.warning("batch_process: unknown op '%s', skipping", op_name)
            except Exception as e:
                error = str(e)
                logger.error("batch_process error on %s at %s: %s", inp, op_name, e)

            results.append({
                "input":   inp,
                "output":  current,
                "success": error is None,
                "error":   error,
            })
        return results

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return bridge and PyGeoFetch status information."""
        info = {
            "pgf_available":    self._pgf is not None,
            "pgf_v2_available": self._pgf_v2,
        }
        if self._pgf:
            try:
                pgf_status = self._pgf.status()
                info["pgf_status"] = pgf_status
            except Exception:
                pass
        return info


# ---------------------------------------------------------------------------
# Simple chain stub (PyGeoFetch v1.0 fallback)
# ---------------------------------------------------------------------------

class _SimpleChain:
    """Minimal chainable pipeline stub when PyGeoFetch v2.0 is not available."""

    def __init__(self, name: str, bridge: PyGeoFetchBridge):
        self._name   = name
        self._bridge = bridge
        self._steps: List[Tuple[str, Dict]] = []

    def __getattr__(self, item: str):
        def _capture(**kw):
            self._steps.append((item, kw))
            return self
        return _capture

    def run(self, input: str, output_dir: str = "./processed/", **kw) -> Dict:
        logger.info("_SimpleChain: running %d steps on %s", len(self._steps), input)
        results = self._bridge.batch_process([input], self._steps, output_dir, parallel=1)
        return results[0] if results else {"input": input, "success": False}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_suffix(path: str, suffix: str) -> str:
    p = pathlib.Path(path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def _reproject_bbox(
    bbox: Tuple[float, float, float, float],
    bbox_crs: str,
    raster_path: str,
) -> Tuple[float, float, float, float]:
    """Reproject a bounding box to match the CRS of *raster_path*.

    Scenes are often downloaded in a projected CRS (e.g. UTM Zone 30N,
    EPSG:32630) while the user's bbox is always supplied in geographic
    WGS84 (EPSG:4326).  Passing WGS84 degrees to rasterio.mask on a
    UTM scene produces ``WindowError: Intersection is empty`` because
    degrees (-0.25 … -0.20) are nowhere near UTM metres (500 000 …
    600 000).

    This helper transparently reprojects the bbox so the clip always
    works regardless of what CRS the scene was downloaded in.

    Args:
        bbox:       ``(minx, miny, maxx, maxy)`` in ``bbox_crs``.
        bbox_crs:   CRS authority string of the supplied bbox
                    (e.g. ``"EPSG:4326"``).
        raster_path: Path to a GeoTIFF whose CRS is the target.

    Returns:
        ``(minx, miny, maxx, maxy)`` in the raster's native CRS.
    """
    try:
        import rasterio
        import rasterio.warp as rwarp
        from rasterio.crs import CRS

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        src_crs = CRS.from_user_input(bbox_crs)

        # Nothing to do when the CRS already matches
        if src_crs == raster_crs:
            return bbox

        minx, miny, maxx, maxy = bbox
        xs, ys = rwarp.transform(src_crs, raster_crs,
                                  [minx, maxx], [miny, maxy])
        reprojected = (min(xs), min(ys), max(xs), max(ys))
        logger.debug(
            "_reproject_bbox: %s [%s] → %s [%s]",
            bbox, bbox_crs,
            tuple(round(v, 1) for v in reprojected), raster_crs,
        )
        return reprojected

    except Exception as exc:
        logger.warning(
            "_reproject_bbox failed (%s) — returning original bbox unchanged. "
            "If the raster is in a projected CRS and the bbox is in degrees, "
            "clipping may fail with 'Input shapes do not overlap raster'.",
            exc,
        )
        return bbox
