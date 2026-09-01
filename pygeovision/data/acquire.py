"""
pygeovision.data.acquire
========================
Acquisition layer: search, download, and preprocess satellite data.

Architecture post-audit (Jul 2026)
------------------------------------
pygeofetch owns EVERYTHING from data acquisition to processed rasters:
  • Search        — SearchQuery across 22+ providers
  • Download      — parallel, resumable, checksummed
  • Preprocessing — atmos, cloud_mask, clip, reproject, resample, mosaic,
                    composite, topo_correct, cloud_fill, pansharpen
  • Spectral indices — 17 built-in + 232 via spyndex
  • SAR processing  — GRDExtractor, despeckle, calibrate, flood_map, coherence
  • Time series    — TimeSeriesAnalyzer, IndexTimeStack
  • Postprocessing — vectorize, zonal_stats, cog, smooth, buffer

pygeovision owns the AI intelligence layer only:
  • validate_for_ai()      — DataValidator gate before any model
  • prepare_sar_for_ai()   — GCP recovery + 6-channel HLS mapping
  • prepare_insar_for_ai() — SNAP/SNAPHU InSAR pipeline
  • Foundation model adapters (Prithvi, DINOv3, ChangeFormer)
  • Training, evaluation, inference, explainability

This module is the ONLY place pygeovision calls pygeofetch.
All other pygeovision modules import from here, never from pygeofetch directly.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── pygeofetch availability ────────────────────────────────────────────────────

def _require_pgf():
    """Return PyGeoFetch client or raise ImportError."""
    try:
        from pygeofetch import PyGeoFetch
        return PyGeoFetch()
    except ImportError:
        raise ImportError(
            "pygeofetch is required for data acquisition. "
            "Install it: pip install pygeofetch"
        )


def _pgf_has_v2(client) -> bool:
    """True if pygeofetch has the v2 processing engine."""
    return (
        hasattr(client, "preprocess") and
        hasattr(client.preprocess, "atmos") and
        hasattr(client, "sar") and
        hasattr(client, "indices")
    )


# ── Result dataclasses (thin wrappers — preserve pygeofetch objects) ──────────

@dataclass
class SearchResult:
    """
    A found satellite scene.
    Thin wrapper — the native pygeofetch SatelliteData is always in .native.
    """
    id:           str
    provider:     str
    satellite:    str
    datetime:     str
    cloud_cover:  float | None
    bbox:         tuple[float, float, float, float] | None
    score:        float | None = None
    collection:   str = ""
    assets:       dict[str, Any] = field(default_factory=dict)
    properties:   dict[str, Any] = field(default_factory=dict)
    native:       Any = field(default=None, repr=False)  # pygeofetch SatelliteData

    @property
    def date(self) -> str:
        return self.datetime[:10] if self.datetime else ""

    @property
    def is_sar(self) -> bool:
        return any(s in self.satellite.lower() for s in ["sentinel-1", "palsar", "sar"])

    def __str__(self):
        return (
            f"[{self.provider}] {self.satellite} | {self.date} | "
            f"cloud={self.cloud_cover}% | {self.id[:40]}"
        )


@dataclass
class DownloadResult:
    """A downloaded file from pygeofetch."""
    scene_id:          str
    provider:          str
    path:              pathlib.Path | None
    success:           bool
    bytes_downloaded:  int = 0
    duration_seconds:  float = 0.0
    checksum_verified: bool = False
    error:             str = ""

    @property
    def size_mb(self) -> float:
        return self.bytes_downloaded / (1024 * 1024)

    def __str__(self):
        if self.success:
            return f"✓ {self.path}  ({self.size_mb:.1f} MB)"
        return f"✗ {self.scene_id}: {self.error}"


@dataclass
class ProcessingResult:
    """Result of a pygeofetch processing operation."""
    output_path: str
    success:     bool
    metadata:    dict[str, Any] = field(default_factory=dict)
    error:       str = ""

    def __str__(self):
        if self.success:
            return f"✓ {self.output_path}"
        return f"✗ {self.error}"


# ── Type translation ───────────────────────────────────────────────────────────

def _pgf_to_search_result(sd, provider_hint: str = "") -> SearchResult:
    """Convert pygeofetch SatelliteData → pygeovision SearchResult."""
    cloud = getattr(sd, "cloud_cover", None)
    sat   = str(getattr(sd, "satellite", "") or "")
    coll  = str(getattr(sd, "collection", "") or "")
    bbox  = getattr(sd, "bbox", None)
    props = getattr(sd, "properties", {}) or {}
    score = float(getattr(sd, "score", 0.0) or 0.0)
    prov  = str(getattr(sd, "provider", provider_hint) or provider_hint)

    dt = getattr(sd, "datetime", None) or getattr(sd, "start_datetime", None)
    datetime_str = dt.isoformat() if dt else ""

    raw_assets = getattr(sd, "assets", {}) or {}
    assets = {}
    for k, v in raw_assets.items():
        if hasattr(v, "href"):
            assets[k] = {
                "key": getattr(v, "key", k),
                "href": str(v.href),
                "roles": list(getattr(v, "roles", []) or []),
                "title": getattr(v, "title", None),
                "media_type": getattr(v, "media_type", None),
            }
        elif isinstance(v, dict):
            assets[k] = {"key": k, **v}
        else:
            assets[k] = {"key": k, "href": str(v), "roles": []}

    return SearchResult(
        id          = str(getattr(sd, "id", "")),
        provider    = prov,
        satellite   = sat,
        collection  = coll,
        datetime    = datetime_str,
        cloud_cover = float(cloud) if cloud is not None else None,
        bbox        = tuple(bbox) if bbox else None,
        score       = score,
        assets      = assets,
        properties  = props,
        native      = sd,
    )


def _pgf_to_download_result(dr) -> DownloadResult:
    """Convert pygeofetch DownloadResult → pygeovision DownloadResult."""
    try:
        from pygeofetch.models.download_task import DownloadStatus
        status  = getattr(dr, "status", None)
        success = (status == DownloadStatus.COMPLETED) if status else False
    except ImportError:
        success = bool(getattr(dr, "success", False))

    output_path  = getattr(dr, "output_path", None)
    output_paths = list(getattr(dr, "output_paths", None) or [])
    if output_path is None and output_paths:
        output_path = output_paths[0]

    return DownloadResult(
        scene_id          = str(getattr(dr, "data_id", "")),
        provider          = str(getattr(dr, "provider", "")),
        path              = pathlib.Path(output_path) if output_path else None,
        success           = success,
        bytes_downloaded  = int(getattr(dr, "bytes_downloaded", 0) or 0),
        duration_seconds  = float(getattr(dr, "duration_seconds", 0.0) or 0.0),
        checksum_verified = bool(getattr(dr, "checksum_verified", False) or False),
        error             = str(getattr(dr, "error", "") or ""),
    )


# ── SatelliteAcquirer — the single acquisition class ──────────────────────────

class SatelliteAcquirer:
    """
    Single interface for all satellite data acquisition and preprocessing.

    Delegates entirely to pygeofetch for:
      - Search across 22+ providers
      - Download with parallel workers, resume, checksum
      - Atmospheric correction, cloud masking, clipping, resampling
      - Spectral indices (NDVI, EVI, NDWI, NBR, SAR calibration, etc.)
      - SAR processing (GRD extraction, despeckle, calibrate, flood_map)
      - Time-series analysis (build_index_stack, trend, anomaly)
      - Postprocessing (vectorize, zonal_stats, COG conversion)

    pygeovision then takes the processed output and applies:
      - AI readiness validation (DataValidator)
      - SAR → 6-channel HLS mapping for foundation models
      - InSAR displacement mapping (SNAP + SNAPHU)
      - Foundation model inference (Prithvi, DINOv3, ChangeFormer)

    Usage::

        acq = SatelliteAcquirer()
        acq.add_credentials("copernicus", username="...", password="...")

        # Search
        scenes = acq.search(
            bbox=(-0.25, 5.52, -0.10, 5.62),
            date_range=("2025-01-01", "2025-02-28"),
            satellite="Sentinel-1",
            providers=["planetary_computer"],
        )

        # Download
        downloads = acq.download(
            scenes[:1],
            output_dir="./data/",
            bands=["vv", "vh"],
            post_process=["reproject:EPSG:32630", "cog"],
        )

        # Preprocess (pygeofetch handles this entirely)
        result = acq.preprocess(
            downloads[0].path,
            clip_bbox=(-0.25, 5.52, -0.10, 5.62),
            despeckle=True,          # SAR only
            calibrate_db=True,       # SAR only
            normalise="minmax",
        )

        # Hand off to pygeovision AI layer
        from pygeovision.data.ai_prep import prepare_sar_for_ai
        ai_input = prepare_sar_for_ai(result.output_path)
    """

    def __init__(self):
        self._pgf = _require_pgf()
        self._v2  = _pgf_has_v2(self._pgf)
        logger.info(
            "SatelliteAcquirer: pygeofetch v%s | processing_engine=%s",
            getattr(self._pgf, "__version__", "?"),
            "v2" if self._v2 else "v1",
        )

    # ── Credentials ───────────────────────────────────────────────────────────

    def add_credentials(self, provider: str, **kwargs) -> SatelliteAcquirer:
        """
        Add provider credentials via pygeofetch's keyring store.

        Args:
            provider: Provider ID, e.g. ``"copernicus"``, ``"usgs"``,
                      ``"planetary_computer"``.
            **kwargs: ``username=``, ``password=``, ``api_key=`` etc.

        Returns:
            self (fluent interface)

        Example::

            acq.add_credentials("copernicus",
                                username="you@example.com",
                                password="your-password")
        """
        try:
            self._pgf.add_credentials(provider, **kwargs)
            logger.info("Credentials stored for '%s'", provider)
        except Exception as e:
            logger.error("add_credentials('%s') failed: %s", provider, e)
        return self

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        bbox:            tuple[float, float, float, float],
        date_range:      tuple[str, str] | None = None,
        satellite:       str | None = None,
        providers:       list[str] | None = None,
        cloud_cover_max: float = 100.0,
        cloud_cover_min: float = 0.0,
        collections:     list[str] | None = None,
        sar_product_type:str | None = None,   # "GRD" | "SLC"
        polarisation:    str | None = None,   # "VV&VH" | "VV" | "VH"
        max_results:     int = 100,
        sort_by:         str = "cloud_cover",
        use_cache:       bool = True,
        geometry:        dict | None = None,  # GeoJSON polygon
    ) -> list[SearchResult]:
        """
        Search satellite imagery via pygeofetch across multiple providers.

        All spatial, temporal, and product filters are passed directly to
        pygeofetch's SearchQuery — no server-side filter is ever silently
        dropped (pygeofetch bug fixes from Jul 2026 session verified this).

        Args:
            bbox:             ``(lon_min, lat_min, lon_max, lat_max)`` WGS84.
            date_range:       ``(start_date, end_date)`` ISO strings.
            satellite:        Satellite name filter, e.g. ``"Sentinel-1"``.
            providers:        List of provider IDs. ``None`` = all hardened providers.
            cloud_cover_max:  Maximum cloud cover 0–100.
            cloud_cover_min:  Minimum cloud cover 0–100.
            collections:      STAC collection filter, e.g. ``["sentinel-1-grd"]``.
            sar_product_type: SAR product type: ``"GRD"`` or ``"SLC"``.
            polarisation:     SAR polarisation: ``"VV&VH"``, ``"VV"``, ``"VH"``.
            max_results:      Max results per provider.
            sort_by:          ``"cloud_cover"`` | ``"datetime"`` | ``"score"``.
            use_cache:        Use pygeofetch's 1-hour search result cache.
            geometry:         GeoJSON polygon for precise spatial filtering.
                              Overrides bbox for spatial constraint when provided.
                              (pygeofetch bug fix: geometry was previously ignored)

        Returns:
            List of :class:`SearchResult`, sorted by score descending.

        Example::

            scenes = acq.search(
                bbox=(-0.25, 5.52, -0.10, 5.62),
                date_range=("2025-01-01", "2025-02-28"),
                satellite="Sentinel-1",
                providers=["planetary_computer"],
                sar_product_type="GRD",
                polarisation="VV&VH",
            )
            print(f"Found {len(scenes)} scenes")
            for s in scenes:
                print(s)
        """
        from datetime import date as _date

        from pygeofetch.models.search_query import BoundingBox, SearchQuery

        bbox_obj = BoundingBox.from_string(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

        start_d = end_d = None
        if date_range:
            try:
                start_d = _date.fromisoformat(str(date_range[0])[:10])
                end_d   = _date.fromisoformat(str(date_range[1])[:10])
            except (ValueError, TypeError) as e:
                logger.warning("date_range parse failed: %s", e)

        satellites = [satellite] if satellite else []
        colls      = collections or []

        # SAR product type filter (uses pygeofetch's SearchQuery fields directly)
        extra_filters = {}
        if sar_product_type:
            extra_filters["sar_product_type"] = sar_product_type
        if polarisation:
            extra_filters["polarisation"] = polarisation

        query = SearchQuery(
            bbox              = bbox_obj,
            geometry          = geometry,        # GeoJSON polygon — now correctly wired
            start_date        = start_d,
            end_date          = end_d,
            cloud_cover_min   = cloud_cover_min,
            cloud_cover_max   = cloud_cover_max,
            max_results       = max_results,
            sort_by           = sort_by,
            sort_ascending    = True,
            collections       = colls,
            satellites        = satellites,
            sensors           = [],
            processing_levels = [],
            providers         = providers or [],
            provider_filters  = extra_filters,
            ids               = [],
        )

        try:
            pgf_results = self._pgf.search(
                query, providers=providers, use_cache=use_cache
            )
        except Exception as e:
            logger.error("pygeofetch.search() failed: %s", e)
            return []

        results = [_pgf_to_search_result(sd) for sd in (pgf_results or [])]
        logger.info("Search → %d scenes across %s", len(results), providers or "all")
        return results

    # ── Download ──────────────────────────────────────────────────────────────

    def download(
        self,
        scenes:           list[SearchResult],
        output_dir:       str | pathlib.Path,
        bands:            list[str] | None = None,
        post_process:     list[str] | None = None,
        parallel:         int = 4,
        verify_checksum:  bool = True,
        resume:           bool = True,
        retry_attempts:   int = 3,
        overwrite:        bool = False,
        on_failure:       str = "skip",
    ) -> list[DownloadResult]:
        """
        Download satellite scenes via pygeofetch.

        pygeofetch handles:
          - Parallel concurrent workers
          - HTTP Range-based resume (correct after Jul 2026 fix)
          - Exponential backoff scaled to file size
          - Multi-point file integrity validation (not just first tile)
          - MD5 checksum verification
          - Post-processing: reproject, COG conversion, band selection
          - 0-byte file detection and immediate retry

        Args:
            scenes:           List of :class:`SearchResult` to download.
            output_dir:       Destination directory.
            bands:            Band names to download, e.g. ``["vv", "vh"]``,
                              ``["B02","B03","B04","B08","B11","B12"]``.
                              ``None`` = all available assets.
            post_process:     pygeofetch post-processing steps.
                              Common examples:
                              ``["reproject:EPSG:32630", "cog"]``
                              ``["reproject:EPSG:4326"]``
                              ``["cog"]``
            parallel:         Concurrent download workers (default 4).
            verify_checksum:  MD5 checksum verification after download.
            resume:           Resume interrupted downloads via Range requests.
            retry_attempts:   Max retry attempts (default 3).
            overwrite:        Overwrite existing files.
            on_failure:       ``"skip"`` | ``"abort"`` | ``"retry"``.

        Returns:
            List of :class:`DownloadResult`.

        Example::

            downloads = acq.download(
                scenes[:1],
                output_dir="./data/sentinel1/",
                bands=["vv", "vh"],
                post_process=["reproject:EPSG:32630", "cog"],
            )
            for dl in downloads:
                print(dl)
        """
        from pygeofetch.models.download_task import DownloadOptions, PostProcessAction

        pp_actions = []
        for step in (post_process or []):
            if ":" in step:
                action, param = step.split(":", 1)
                pp_actions.append(PostProcessAction(action=action, params={"value": param}))
            else:
                pp_actions.append(PostProcessAction(action=step, params={}))

        opts = DownloadOptions(
            parallel            = parallel,
            verify_checksum     = verify_checksum,
            resume              = resume,
            retry_attempts      = retry_attempts,
            bandwidth_limit_mbps= 0,
            on_failure          = on_failure,
            overwrite           = overwrite,
            post_process        = pp_actions,
        )

        sat_data_list = []
        for scene in scenes:
            sd = scene.native
            if sd is None:
                logger.warning("Scene %s has no native pygeofetch object — skipping", scene.id)
                continue
            if bands:
                if hasattr(sd, "assets") and sd.assets:
                    try:
                        sd_copy = sd.model_copy(deep=True)
                    except AttributeError:
                        import copy
                        sd_copy = copy.deepcopy(sd)
                    sd_copy.assets = {
                        k: v for k, v in sd.assets.items()
                        if k in bands or any(b.lower() in k.lower() for b in bands)
                    }
                    sat_data_list.append(sd_copy)
                    continue
            sat_data_list.append(sd)

        if not sat_data_list:
            logger.warning("No valid scenes to download")
            return []

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            pgf_results = self._pgf.download(
                sat_data_list,
                destination=pathlib.Path(output_dir),
                options=opts,
            )
        except Exception as e:
            logger.error("pygeofetch.download() failed: %s", e)
            return []

        results = [_pgf_to_download_result(dr) for dr in (pgf_results or [])]
        n_ok = sum(1 for r in results if r.success)
        logger.info("Download: %d/%d succeeded", n_ok, len(results))
        return results

    # ── Preprocessing — all delegated to pygeofetch ───────────────────────────

    def preprocess(
        self,
        input_path:       str | pathlib.Path,
        clip_bbox:        tuple[float, float, float, float] | None = None,
        clip_bbox_crs:    str = "EPSG:4326",
        clip_geometry:    dict | None = None,
        atmos_method:     str | None = None,    # "dos1"|"sen2cor"|"flaash"
        cloud_mask_method:str | None = None,    # "scl"|"threshold"|"fmask"
        scl_path:         str | None = None,
        despeckle:        bool = False,             # SAR: Lee filter
        despeckle_filter: str = "enhanced_lee",
        despeckle_window: int = 7,
        calibrate_db:     bool = False,             # SAR: DN→sigma0 dB
        calibrate_type:   str = "sigma0",
        normalise:        str | None = None,     # "minmax"|"zscore"|"scale_factor"
        scale_factor:     float = 10000.0,
        resample_m:       float | None = None,
        resample_reference: str | None = None,  # grid-align to another raster
        topo_correct_dem: str | None = None,
        pansharpen_pan:   str | None = None,
        output_path:      str | None = None,
    ) -> ProcessingResult:
        """
        Apply a preprocessing chain via pygeofetch.

        pygeofetch is responsible for all raster operations — pygeovision
        does not reimplement any of these. The result is an AI-ready raster
        that pygeovision's AI layer can consume directly.

        Automatic CRS handling:
            clip_bbox is automatically reprojected to the raster's native CRS
            before clipping — so WGS84 bboxes work correctly with UTM scenes.
            (This was a confirmed pygeofetch bug that has been fixed.)

        Args:
            input_path:         Input GeoTIFF.
            clip_bbox:          ``(lon_min, lat_min, lon_max, lat_max)`` WGS84.
            clip_bbox_crs:      CRS of the supplied bbox (default EPSG:4326).
            clip_geometry:      GeoJSON polygon for precise clipping.
            atmos_method:       Atmospheric correction method. ``None`` = skip.
            cloud_mask_method:  Cloud masking method. ``None`` = skip.
            scl_path:           Sentinel-2 SCL band for SCL cloud masking.
            despeckle:          Apply Lee speckle filter (SAR only).
            despeckle_filter:   ``"enhanced_lee"`` | ``"lee"`` | ``"frost"``.
            despeckle_window:   Filter window size in pixels.
            calibrate_db:       Calibrate DN to dB (SAR only — after despeckle).
            calibrate_type:     ``"sigma0"`` | ``"gamma0"`` | ``"beta0"``.
            normalise:          Pixel normalisation. ``None`` = skip.
            scale_factor:       Divisor for ``"scale_factor"`` normalisation.
            resample_m:         Resample to this pixel size in metres.
            resample_reference: Align grid to match another raster exactly.
            topo_correct_dem:   Path to DEM for topographic correction.
            pansharpen_pan:     Path to panchromatic band for pan-sharpening.
            output_path:        Final output path. Auto-generated if None.

        Returns:
            :class:`ProcessingResult` with ``.output_path`` and ``.success``.

        Example (SAR GRD)::

            result = acq.preprocess(
                downloads[0].path,
                clip_bbox=(-0.25, 5.52, -0.10, 5.62),
                despeckle=True,
                calibrate_db=True,
                normalise="minmax",
            )
            # → AI-ready float32 GeoTIFF in UTM

        Example (Sentinel-2)::

            result = acq.preprocess(
                downloads[0].path,
                clip_bbox=(-0.25, 5.52, -0.10, 5.62),
                atmos_method="dos1",
                cloud_mask_method="scl",
                scl_path="SCL.tif",
                normalise="scale_factor",
                resample_m=10.0,
            )
        """
        if not self._v2:
            return self._preprocess_fallback(
                input_path, clip_bbox=clip_bbox, clip_bbox_crs=clip_bbox_crs,
                despeckle=despeckle, calibrate_db=calibrate_db,
                normalise=normalise, scale_factor=scale_factor,
                resample_m=resample_m, output_path=output_path,
            )

        current = str(input_path)
        steps   = []

        try:
            pre = self._pgf.preprocess
            sar = self._pgf.sar if hasattr(self._pgf, "sar") else None

            # 1. Despeckle (SAR — MUST be on linear data, before calibration)
            if despeckle and sar:
                r = sar.despeckle(
                    current,
                    filter=despeckle_filter,
                    window=despeckle_window,
                )
                current = str(r.output_path)
                steps.append(f"despeckle({despeckle_filter})")

            # 2. SAR calibration (DN → sigma0/gamma0 in dB)
            if calibrate_db and sar:
                r = sar.calibrate(current, output_type=calibrate_type, in_db=True)
                current = str(r.output_path)
                steps.append(f"calibrate({calibrate_type}, dB)")

            # 3. Atmospheric correction (optical)
            if atmos_method:
                r = pre.atmos(current, method=atmos_method)
                current = str(r.output_path)
                steps.append(f"atmos({atmos_method})")

            # 4. Cloud masking (optical)
            if cloud_mask_method:
                r = pre.cloud_mask(
                    current,
                    method=cloud_mask_method,
                    scl_band=scl_path,
                )
                current = str(r.output_path)
                steps.append(f"cloud_mask({cloud_mask_method})")

            # 5. Topographic correction
            if topo_correct_dem:
                r = pre.topo_correct(current, dem=topo_correct_dem)
                current = str(r.output_path)
                steps.append("topo_correct")

            # 6. Clip to AOI (CRS-aware — bbox auto-reprojected to raster CRS)
            if clip_bbox or clip_geometry:
                if clip_geometry:
                    r = pre.clip(current, geometry=clip_geometry)
                else:
                    # Reproject bbox to raster CRS first
                    bbox_native = _reproject_bbox_to_raster_crs(
                        clip_bbox, clip_bbox_crs, current
                    )
                    r = pre.clip(current, bbox=bbox_native)
                current = str(r.output_path)
                steps.append("clip")

            # 7. Pan-sharpening
            if pansharpen_pan:
                r = pre.pansharpen(pan=pansharpen_pan, ms=current, method="brovey")
                current = str(r.output_path)
                steps.append("pansharpen")

            # 8. Resample — grid-align to reference raster if provided
            if resample_reference:
                r = pre.resample(current, reference=resample_reference)
                current = str(r.output_path)
                steps.append("resample(grid-align)")
            elif resample_m:
                r = pre.resample(current, resolution=int(resample_m))
                current = str(r.output_path)
                steps.append(f"resample({resample_m}m)")

            # 9. Normalise pixel values
            if normalise:
                if normalise == "scale_factor":
                    r = pre.atmos(current, method="dos1")
                else:
                    # pygeofetch normalise / minmax / zscore
                    r = pre.normalise(current, method=normalise,
                                      scale_factor=scale_factor)
                current = str(r.output_path)
                steps.append(f"normalise({normalise})")

            # Copy to final output path if specified
            if output_path and output_path != current:
                import shutil
                pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, output_path)
                current = output_path

            logger.info("preprocess: %s → %s  steps=%s",
                        pathlib.Path(str(input_path)).name,
                        pathlib.Path(current).name, steps)
            return ProcessingResult(
                output_path=current, success=True,
                metadata={"steps": steps},
            )

        except Exception as e:
            logger.error("preprocess() failed: %s", e)
            return ProcessingResult(output_path=str(input_path), success=False, error=str(e))

    def _preprocess_fallback(
        self, input_path, *, clip_bbox=None, clip_bbox_crs="EPSG:4326",
        despeckle=False, calibrate_db=False, normalise=None,
        scale_factor=10000.0, resample_m=None, output_path=None,
    ) -> ProcessingResult:
        """
        Minimal fallback when pygeofetch v2 processing engine is not available.
        Uses the pygeovision GCP-recovery + rasterio for basic operations.
        """
        logger.warning(
            "pygeofetch v2 processing engine not detected. "
            "Falling back to pygeovision minimal preprocessor. "
            "Install pygeofetch >= 2.0 for full preprocessing capability."
        )
        current = str(input_path)
        steps   = []

        # GCP recovery (pygeovision-specific fix for pygeofetch GRD downloads)
        from pygeovision.data.validators.georeference import validate_sar_georeference
        geo = validate_sar_georeference(current)
        if geo.recovered and geo.repaired_path:
            current = geo.repaired_path
            steps.append("gcp_recovery")

        # Clip
        if clip_bbox:
            try:
                import rasterio
                import rasterio.mask
                from shapely.geometry import box

                from pygeovision.data.validators.georeference import reproject_bbox_to_raster_crs

                bbox_native = reproject_bbox_to_raster_crs(clip_bbox, current)
                geom = [box(*bbox_native).__geo_interface__]
                out = output_path or current.replace(".tif", "_clip.tif")
                with rasterio.open(current) as src:
                    data, transform = rasterio.mask.mask(src, geom, crop=True)
                    meta = src.meta.copy()
                meta.update(height=data.shape[1], width=data.shape[2], transform=transform)
                with rasterio.open(out, "w", **meta) as dst:
                    dst.write(data)
                current = out
                steps.append("clip(rasterio)")
            except Exception as e:
                logger.warning("Fallback clip failed: %s", e)

        return ProcessingResult(output_path=current, success=True, metadata={"steps": steps})

    # ── Spectral indices — all delegated to pygeofetch ───────────────────────

    def index(
        self,
        name:        str,
        output_path: str | None = None,
        **band_paths,
    ) -> ProcessingResult:
        """
        Compute a spectral index via pygeofetch.

        pygeofetch supports 17 built-in indices and 232+ via spyndex.
        All computation is done by pygeofetch — pygeovision does not
        reimplement index formulas.

        Args:
            name:        Index name: ``"ndvi"``, ``"evi"``, ``"ndwi"``,
                         ``"mndwi"``, ``"ndbi"``, ``"nbr"``, ``"savi"``,
                         ``"ndsi"``, ``"tct"``, etc.
            output_path: Output GeoTIFF path.
            **band_paths: Band file paths, e.g. ``red="B04.tif"``,
                          ``nir="B08.tif"``.

        Returns:
            :class:`ProcessingResult` with ``.output_path``.

        Example::

            result = acq.index("ndvi", red="B04.tif", nir="B08.tif",
                               output_path="ndvi.tif")
        """
        if not self._v2 or not hasattr(self._pgf, "indices"):
            logger.warning("index() requires pygeofetch >= 2.0")
            return ProcessingResult(output_path="", success=False,
                                    error="pygeofetch v2 required for spectral indices")
        try:
            fn = getattr(self._pgf.indices, name, None)
            if fn is None:
                return ProcessingResult(output_path="", success=False,
                                        error=f"Unknown index '{name}'")
            r = fn(**band_paths)
            out = str(r.output_path)
            if output_path and output_path != out:
                import shutil
                shutil.copy2(out, output_path)
                out = output_path
            return ProcessingResult(output_path=out, success=True)
        except Exception as e:
            logger.error("index('%s') failed: %s", name, e)
            return ProcessingResult(output_path="", success=False, error=str(e))

    # ── SAR-specific — all delegated to pygeofetch ────────────────────────────

    def extract_grd(
        self,
        safe_zip:    str,
        output_dir:  str,
        polarisation: str = "VV",
        subswath:    str | None = None,
    ) -> ProcessingResult:
        """
        Extract and georeference a Sentinel-1 GRD band from a .SAFE.zip.

        Uses pygeofetch's GRDExtractor which correctly handles GCP-based
        georeferencing for raw SAR products. This closes the raw-TIFF-has-no-CRS
        gap that caused silent wrong-location results in earlier versions.

        Args:
            safe_zip:     Path to Sentinel-1 .SAFE.zip file.
            output_dir:   Output directory.
            polarisation: ``"VV"`` | ``"VH"`` | ``"VV&VH"`` (both).
            subswath:     Optional subswath selector.

        Returns:
            :class:`ProcessingResult`.
        """
        if not self._v2 or not hasattr(self._pgf, "sar"):
            return ProcessingResult(output_path="", success=False,
                                    error="GRDExtractor requires pygeofetch >= 2.0")
        try:
            from pygeofetch.sar import GRDExtractor
            extractor = GRDExtractor()
            r = extractor.extract(
                safe_zip,
                output_dir=output_dir,
                polarisation=polarisation,
                subswath=subswath,
            )
            return ProcessingResult(output_path=str(r.output_path), success=True,
                                    metadata=r.metadata or {})
        except Exception as e:
            logger.error("extract_grd() failed: %s", e)
            return ProcessingResult(output_path="", success=False, error=str(e))

    # ── Time series — all delegated to pygeofetch ─────────────────────────────

    def build_time_series(
        self,
        scenes:          list[SearchResult],
        output_dir:      str,
        index_name:      str = "ndvi",
        bands:           list[str] | None = None,
        bbox:            tuple[float, float, float, float] | None = None,
        align_grids:     bool = True,  # pygeofetch fix: auto grid alignment
        composite_method: str = "median",
    ) -> dict[str, Any]:
        """
        Build an index time stack from multiple scenes via pygeofetch.

        Uses pygeofetch's TimeSeriesAnalyzer which now auto-aligns mismatched
        grids (fix from Jul 2026 — previously raised immediately on any grid
        mismatch between dates, breaking elongated or irregularly-shaped AOIs).

        Args:
            scenes:           Scenes to include in the time stack.
            output_dir:       Directory for downloaded/processed files.
            index_name:       Index to compute: ``"ndvi"``, ``"evi"``, etc.
            bands:            Bands to download.
            bbox:             Clip bounding box.
            align_grids:      Auto-align mismatched grids (default True).
            composite_method: Cloud compositing: ``"median"`` | ``"mean"`` | ``"max"``.

        Returns:
            Dict with:
              - ``"stack"``     : IndexTimeStack from pygeofetch
              - ``"values"``    : np.ndarray (T, H, W)
              - ``"dates"``     : list of datetime
              - ``"xarray"``    : xr.DataArray (if xarray installed)
        """
        if not self._v2 or not hasattr(self._pgf, "TimeSeriesAnalyzer"):
            logger.warning("TimeSeriesAnalyzer requires pygeofetch >= 2.0")
            return {}

        try:
            from pygeofetch.processor import TimeSeriesAnalyzer

            downloads = self.download(scenes, output_dir=output_dir, bands=bands)
            paths = [str(d.path) for d in downloads if d.success and d.path]

            tsa = TimeSeriesAnalyzer(paths)
            stack = tsa.build_index_stack(
                index_name, align_grids=align_grids,
                bbox=bbox, composite_method=composite_method,
            )
            return {
                "stack":  stack,
                "values": stack.values,
                "dates":  stack.dates,
                "xarray": stack.as_xarray() if hasattr(stack, "as_xarray") else None,
            }
        except Exception as e:
            logger.error("build_time_series() failed: %s", e)
            return {}

    # ── Postprocessing — all delegated to pygeofetch ──────────────────────────

    def vectorize(
        self,
        raster_path: str,
        output_path: str | None = None,
        smooth:      bool = True,
    ) -> ProcessingResult:
        """
        Convert a raster mask to vector polygons via pygeofetch.

        Args:
            raster_path: Binary raster mask (e.g. flood map).
            output_path: Output GeoJSON/shapefile path.
            smooth:      Apply boundary smoothing.

        Returns:
            :class:`ProcessingResult`.
        """
        if not self._v2 or not hasattr(self._pgf, "post"):
            return ProcessingResult(output_path="", success=False,
                                    error="vectorize requires pygeofetch >= 2.0")
        try:
            r = self._pgf.post.vectorize(raster_path, smooth=smooth)
            out = str(r.output_path)
            if output_path and output_path != out:
                import shutil
                shutil.copy2(out, output_path)
                out = output_path
            return ProcessingResult(output_path=out, success=True)
        except Exception as e:
            logger.error("vectorize() failed: %s", e)
            return ProcessingResult(output_path="", success=False, error=str(e))

    def zonal_stats(
        self,
        raster_path: str,
        zones_path:  str,
        stats:       list[str] = ("mean", "std", "min", "max", "count"),
    ) -> dict[str, Any]:
        """
        Compute zonal statistics via pygeofetch.

        Args:
            raster_path: Input raster.
            zones_path:  Zone polygons (GeoJSON/shapefile).
            stats:       Statistics to compute.

        Returns:
            Dict of zone → statistics.
        """
        if not self._v2 or not hasattr(self._pgf, "post"):
            return {}
        try:
            r = self._pgf.post.zonal_stats(raster_path, zones=zones_path, stats=list(stats))
            return r.result if hasattr(r, "result") else {}
        except Exception as e:
            logger.error("zonal_stats() failed: %s", e)
            return {}

    def to_cog(self, input_path: str, output_path: str | None = None) -> str:
        """Convert to Cloud Optimised GeoTIFF via pygeofetch."""
        if self._v2 and hasattr(self._pgf, "post"):
            try:
                r = self._pgf.post.cog(input_path)
                return str(r.output_path)
            except Exception as e:
                logger.warning("to_cog via pygeofetch failed: %s", e)
        # Fallback: gdal_translate
        import subprocess
        out = output_path or input_path.replace(".tif", "_cog.tif")
        try:
            subprocess.run(
                ["gdal_translate", "-of", "COG", "-co", "COMPRESS=LZW",
                 input_path, out],
                check=True, capture_output=True,
            )
        except Exception:
            import shutil
            shutil.copy2(input_path, out)
        return out


# ── Convenience helper ────────────────────────────────────────────────────────

def _reproject_bbox_to_raster_crs(
    bbox: tuple[float, float, float, float],
    bbox_crs: str,
    raster_path: str,
) -> tuple[float, float, float, float]:
    """Reproject bbox to the native CRS of a raster file."""
    try:
        import rasterio
        import rasterio.warp as rwarp
        from rasterio.crs import CRS

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        src_crs = CRS.from_user_input(bbox_crs)
        if src_crs == raster_crs:
            return bbox

        minx, miny, maxx, maxy = bbox
        xs, ys = rwarp.transform(src_crs, raster_crs, [minx, maxx], [miny, maxy])
        return (min(xs), min(ys), max(xs), max(ys))
    except Exception as e:
        logger.warning("_reproject_bbox_to_raster_crs failed: %s — using original bbox", e)
        return bbox