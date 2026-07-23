"""
PyGeoVision — World-Class Geospatial AI Platform.

PyGeoVision is a fully self-contained geospatial AI platform built on two
native layers:

  🛰️  data        — Universal satellite data pipeline (22+ providers)
                    CLI: pygeofetch search/download/pipeline/auth/cache
                    Python: SatelliteFetcher (wraps pygeofetch CLI + pystac_client)

  🧠  AI layer    — Native AI for geospatial data (PyTorch, transformers, SMP)
                    Segmentation, detection, classification, change detection,
                    embeddings, SAM, Prithvi, cloud masking, ONNX, and more —
                    all implemented natively, no external AI platform required.

Architecture:
  client.data.*            → search, download, pipeline, auth, cache
  client.segmentation.*    → buildings, water, SAM, custom models
  client.detection.*       → generic/ships/cars object detection (YOLO)
  client.change.*          → bi-temporal change detection (ChangeFormer)
  client.classification.*  → scene classification, land cover
  client.pipeline()        → end-to-end: data → AI → output

Quick start:
    >>> import pygeovision as pgv
    >>>
    >>> client = pgv.PyGeoVision()
    >>>
    >>> # Add credentials (stored in system keyring via pygeofetch)
    >>> client.data.add_credentials("usgs", username="user", password="pass")
    >>> client.data.add_credentials("planet", api_key="PL_KEY")
    >>>
    >>> # Search 22+ satellite providers
    >>> results = client.search(
    ...     bbox=(-74.1, 40.6, -73.7, 40.9),
    ...     date_range=("2024-01-01", "2024-06-01"),
    ...     providers=["planetary_computer", "copernicus", "usgs"],
    ...     cloud_cover_max=15,
    ... )
    >>> print(f"Found {len(results)} scenes")
    >>>
    >>> # Download with post-processing
    >>> downloads = client.download(
    ...     results[:5],
    ...     output_dir="./data/",
    ...     post_process=["unzip", "reproject:EPSG:4326", "compress:lzw", "cog"],
    ... )
    >>>
    >>> # AI: segment buildings natively (SAM auto-segmentation)
    >>> masks = client.segmentation.buildings(
    ...     downloads[0].path,
    ...     output_path="buildings.tif",
    ... )
    >>>
    >>> # End-to-end pipeline: search → download → AI
    >>> result = client.pipeline(
    ...     "building_footprints",
    ...     bbox=(-74.1, 40.6, -73.7, 40.9),
    ...     date="2024-06",
    ... )
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pygeovision._version import __version__
from pygeovision.agent import GeoAgent  # autonomous geospatial AI agent
from pygeovision.ai.models.zoo import ModelSpec, ModelZoo, model_zoo
from pygeovision.ai.pipelines.domains import list_pipelines as list_all_pipelines
from pygeovision.core.config import PyGeoVisionConfig
from pygeovision.core.exceptions import (  # noqa: F401
    AIEngineError,
    AINotAvailableError,
    InferenceError,
    LabelingError,
    ModelNotFoundError,
    PipelineError,
    PyGeoVisionAuthError,
    PyGeoVisionConfigError,
    PyGeoVisionError,
    TrainingError,
)
from pygeovision.data.fetch import DownloadResult, SatelliteFetcher, SearchResult
from pygeovision.data.pipeline import DataPipeline
from pygeovision.datasets.registry import DatasetInfo, DatasetRegistry, dataset_registry

logger = logging.getLogger(__name__)

__all__ = [
    "PyGeoVision",
    "GeoAgent",
    "__version__",
    "PyGeoVisionError",
    "SatelliteFetcher",
    "SearchResult",
    "DownloadResult",
    "DataPipeline",
    "dataset_registry",
    "DatasetRegistry",
    "DatasetInfo",
    "model_zoo",
    "ModelZoo",
    "ModelSpec",
    "list_all_pipelines",
]


class PyGeoVision:
    """PyGeoVision — World-Class Geospatial AI Platform.

    Unified interface combining a satellite data layer (pygeofetch) with a
    fully native geospatial AI layer — segmentation, detection, change
    detection, classification, training, and inference — in one
    production-ready platform.

    Args:
        config_path: Path to PyGeoVision or pygeofetch config YAML.
        cache_dir: Override local cache directory.
        pygeofetch_cmd: Override the pygeofetch CLI command
            (e.g. 'PyGeoFetch' on some systems).
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Attributes:
        data: SatelliteFetcher — full pygeofetch Python API.
        segmentation, detection, change, classification: native AI layers.
        config: PyGeoVisionConfig.

    Example — complete end-to-end workflow::

        import pygeovision as pgv

        client = pgv.PyGeoVision()

        # Authenticate with providers
        client.data.add_credentials("usgs", username="user", password="pass")
        client.data.add_credentials("copernicus",
            client_id="my-id", client_secret="my-secret")
        client.data.add_credentials("planet", api_key="PL_KEY")

        # Search satellite data across 22+ providers
        results = client.search(
            bbox=(-0.15, 51.47, -0.10, 51.52),
            date_range=("2024-06-01", "2024-06-30"),
            providers=["planetary_computer", "copernicus"],
            cloud_cover_max=10,
        )

        # Download with post-processing (delegates to PyGeoFetch)
        downloads = client.download(
            results[:3],
            output_dir="./sentinel2/",
            parallel=4,
            post_process=["unzip", "reproject:EPSG:4326", "cog"],
        )

        # AI: segment buildings natively (SAM auto-segmentation)
        masks = client.segmentation.buildings(
            downloads[0].path,
            output_path="buildings.tif",
        )

        # End-to-end pipeline
        result = client.pipeline("building_footprints", bbox=..., date="2024-06")

        # YAML pipeline (delegates to PyGeoFetch pipeline run)
        client.data.run_pipeline("weekly-sentinel2.yaml")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        cache_dir: Path | None = None,
        pygeofetch_cmd: str = "pygeofetch",   # kept for backward compat, unused
        log_level: str = "INFO",
    ) -> None:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        self.config = PyGeoVisionConfig.load(config_path) if config_path else PyGeoVisionConfig()

        # Core data layer — wraps pygeofetch
        self.data = SatelliteFetcher(
            config_path=Path(config_path) if config_path else None,
            cache_dir=cache_dir,
        )

        self._ai_engine: Any | None = None

        # ── Phase 2+ Independent layers ─────────────────────────────────
        # All accessible directly on the client object.

        # Auto-labeling — 7+ sources (OSM, MS Buildings, Google, ESA, SAM …)
        from pygeovision.labeling import (
            ActiveLearner,
            AutoLabelPipeline,
            DynamicWorldLabeler,
            ESAWorldCoverLabeler,
            FoundationModelLabeler,
            GoogleBuildingsLabeler,
            LabelQualityAssessor,
            MicrosoftBuildingsLabeler,
            OSMLabeler,
            SAMAutoLabeler,
        )
        self.labeling = _LabelingClientProxy()

        # Geospatial losses — Dice, Focal, Tversky, Boundary, Lovász, OHEM
        from pygeovision.losses import (
            BoundaryAwareLoss,
            ClassBalancedCrossEntropy,
            ComboLoss,
            DiceLoss,
            FocalLoss,
            GeospatialMixedLoss,
            LovaszLoss,
            OhemCrossEntropy,
            TverskyLoss,
        )
        self.losses = _LossesClientProxy()

        # Advanced inference — Gaussian tiling, batch, streaming, ensemble
        from pygeovision.inference import BatchInferenceEngine, TiledInference
        self.inference = _InferenceClientProxy()

        # Explainability — GradCAM, uncertainty, SHAP, attention maps
        self.xai = _XAIClientProxy()

        # Monitoring — drift detection, performance tracking, alerts
        self.monitoring = _MonitoringClientProxy()

        # Edge deployment — NVIDIA Jetson, ONNX Runtime
        self.edge = _EdgeClientProxy()

        # Cloud deployment — AWS SageMaker, Azure ML, GCP Vertex AI
        self.cloud = _CloudClientProxy()

        # Advanced AI — few-shot, multi-task, AutoML, VLM, time series, 3D
        self.few_shot   = _FewShotClientProxy()
        self.multitask  = _MultiTaskClientProxy()
        self.automl     = _AutoMLClientProxy()
        self.vlm        = _VLMClientProxy()
        self.timeseries = _TimeSeriesClientProxy()
        self.pointcloud = _PointCloudClientProxy()

        # Segmentation, detection, change detection, classification —
        # native model layers (SAM, YOLO, ChangeFormer, CLIP / ESA WorldCover)
        self.segmentation  = _SegmentationClientProxy()
        self.detection     = _DetectionClientProxy()
        self.change        = _ChangeDetectionClientProxy()
        self.classification = _ClassificationClientProxy()

        # ── NEW: Data Validation + Full Preprocessing Stack ─────────────
        # DataValidator — mandatory before every model run
        from pygeovision.data.validator import DataValidator
        self.validator = DataValidator(mode="fix")

        # Preprocessor — 100+ spatial preprocessing operations
        from pygeovision.preprocess import Preprocessor
        self.preprocess = Preprocessor(validator=self.validator)

        # SpectralIndices — 22 validated indices (NDVI, EVI, NBR, TCT, PCA…)
        from pygeovision.data.indices import SpectralIndices
        self.indices = SpectralIndices(validator=self.validator)

        # PostProcessor — 20+ prediction postprocessing operations
        from pygeovision.data.postprocess import PostProcessor
        self.postprocess = PostProcessor(validator=self.validator)

        # ── PyGeoFetch Bridge — definitive integration layer ─────────────
        # Unified access to PyGeoFetch's search/download Python API,
        # preprocessing chain, spectral indices, SAR, and batch processing.
        from pygeovision.data.pgf_bridge import PyGeoFetchBridge
        self._pgf_bridge = PyGeoFetchBridge(
            validator       = self.validator,
            pgv_preprocessor= self.preprocess,
            pgv_indices     = self.indices,
            pgv_postprocessor= self.postprocess,
        )

        # ── SAR processing proxy (via PyGeoFetch v2.0 / bridge) ─────────
        self.sar = _SARProxy(self._pgf_bridge)

    # ------------------------------------------------------------------
    # Authentication (delegates to PyGeoFetch via self.data)
    # ------------------------------------------------------------------

    def add_credentials(
        self,
        provider: str,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> PyGeoVision:
        """Add satellite provider credentials (stored via pygeofetch keyring).

        Delegates to ``pygeofetch auth add PROVIDER ...`` for secure storage.

        Auth modes by provider:
            usgs, nasa_earthdata                   → username + password
            planet, opentopography, airbus_oneatlas → api_key
            copernicus, sentinel_hub, maxar_gbdx    → client_id + client_secret
            aws_earth, planetary_computer, element84 → no auth needed

        Args:
            provider: pygeofetch provider ID (22+ supported).
            username: For user/pass auth providers.
            password: For user/pass auth providers.
            api_key: For API key providers.
            client_id: For OAuth2 providers.
            client_secret: For OAuth2 providers.

        Returns:
            Self (for method chaining).

        Example:
            >>> client \\
            ...   .add_credentials("usgs", username="user", password="pass") \\
            ...   .add_credentials("planet", api_key="PL_KEY") \\
            ...   .add_credentials("copernicus",
            ...       client_id="my-id", client_secret="my-secret")
        """
        self.data.add_credentials(
            provider, username=username, password=password,
            api_key=api_key, client_id=client_id, client_secret=client_secret,
        )
        return self

    # ------------------------------------------------------------------
    # Study area boundary (real administrative polygon, not an assumed bbox)
    # ------------------------------------------------------------------

    def boundary(
        self,
        query: str,
        output_path: str | None = None,
        reference_area_km2: float | None = None,
        tolerance: float = 0.35,
        raise_on_mismatch: bool = True,
    ):
        """Fetch a real administrative boundary polygon for a study area by
        name, instead of hand-typing an assumed bounding box.

        Resolves `query` via OpenStreetMap Nominatim to its actual boundary
        geometry, computes its true area in the correct local UTM zone
        (auto-detected), and — if `reference_area_km2` is given —
        cross-checks the result against that known value so a mis-resolved
        query is caught rather than silently used.

        Args:
            query: Place name, e.g. "Accra Metropolitan District, Ghana".
            output_path: If given, save the resolved boundary GeoJSON here.
            reference_area_km2: Known reference area for validation (optional
                but recommended — catches wrong-entity resolution).
            tolerance: Allowed fractional deviation before treating the
                match as suspect.
            raise_on_mismatch: Raise on a failed reference check (default)
                vs. return the result anyway with `.validated=False`.

        Returns:
            AdminBoundary with `.geojson`, `.bbox`, `.area_km2`, `.utm_epsg`.

        Example::

            aoi = client.boundary(
                "Accra Metropolitan District, Greater Accra Region, Ghana",
                output_path="accra_boundary.geojson",
                reference_area_km2=60.0,
            )
            print(aoi.summary())
            results = client.search(bbox=aoi.bbox, date_range=("2024-01-01", "2024-06-30"))
        """
        from pygeovision.data.boundary import fetch_admin_boundary
        return fetch_admin_boundary(
            query, output_path=output_path,
            reference_area_km2=reference_area_km2,
            tolerance=tolerance, raise_on_mismatch=raise_on_mismatch,
        )

    def select_covering_scenes(self, results, bbox, max_scenes: int = 4):
        """Greedily select the fewest, lowest-cloud scenes whose combined
        footprint covers `bbox`, instead of assuming the top-ranked single
        scene is sufficient.

        Falls back to just the first `max_scenes` results if scenes don't
        carry footprint geometry (some providers omit it).

        Args:
            results: SearchResult list from `client.search()`, already
                sorted by whatever priority you want (e.g. cloud cover).
            bbox: (lon_min, lat_min, lon_max, lat_max) the selection must cover.
            max_scenes: Safety cap on how many scenes to select.

        Returns:
            The selected subset of `results`, in the same order.

        Example::

            results = client.search(bbox=aoi.bbox, date_range=(...),
                                     sort_by="cloud_cover", sort_order="asc")
            selected = client.select_covering_scenes(results, aoi.bbox)
            downloads = client.download(selected, output_dir="./data/", bands=BANDS)
        """
        from shapely.geometry import box as shp_box, shape as shp_shape

        aoi_box = shp_box(*bbox)
        selected, covered = [], None
        for r in results:
            geom = getattr(r, "geometry", None) or getattr(r, "footprint", None)
            fp = shp_shape(geom) if geom else None
            selected.append(r)
            covered = fp if covered is None else (covered.union(fp) if fp else covered)
            if covered is not None and covered.contains(aoi_box):
                break
            if len(selected) >= max_scenes:
                break
        return selected

    # ------------------------------------------------------------------
    # Search (delegates to PyGeoFetch via self.data)
    # ------------------------------------------------------------------

    def search(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        collections: list[str] | None = None,
        providers: list[str] | None = None,
        satellite: str | None = None,
        cloud_cover_max: float = 30.0,
        max_results: int = 100,
        limit: int | None = None,          # alias for max_results
        sort_by: str = "datetime",
        sort_order: str = "desc",
        processing_level: str | None = None,
        resolution_range: tuple[float, float] | None = None,
        cql2_filter: str | None = None,
        on_provider_failure: str = "skip",
        timeout: int = 120,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Search for satellite imagery across 22+ pygeofetch providers.

        Delegates to ``pygeofetch search run`` (CLI) with pystac_client
        as fallback for STAC providers.

        All 22 providers supported:
            Open access (no credentials):
                planetary_computer, aws_earth, element84, noaa_big_data,
                esa_scihub, jaxa_earth, isro_bhuvan, inpe_cbers,
                digitalglobe, geoserver_generic

            Requires credentials:
                usgs, copernicus, nasa_earthdata, nasa_earthdata_cloud,
                opentopography, planet, sentinel_hub, maxar_gbdx,
                airbus_oneatlas, alaska_satellite_facility,
                google_earth_engine, terrabotics

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
            date_range: (start_date, end_date) as 'YYYY-MM-DD'.
            collections: STAC collection IDs e.g. ['sentinel-2-l2a'].
            providers: pygeofetch provider IDs. Auto-selected from
                       collections/satellite if not specified.
            satellite: Shortcut name ('sentinel-2', 'landsat', 'planet',
                       'worldview', 'pleiades', 'dem', 'modis', etc.)
            cloud_cover_max: Max cloud cover % (0–100).
            max_results: Maximum scenes to return.
            sort_by: 'datetime', 'cloud_cover', 'score', 'satellite'.
            sort_order: 'asc' or 'desc'.
            processing_level: 'L2A', 'L1C', 'L1TP', etc.
            resolution_range: (min_m, max_m) spatial resolution filter.
            cql2_filter: CQL2 expression for advanced filtering.
            on_provider_failure: 'skip', 'abort', or 'retry'.
            timeout: HTTP timeout in seconds.
            use_cache: Use 1-hour result cache.

        Returns:
            List of SearchResult objects.

        Example:
            >>> # Open access — no credentials needed
            >>> results = client.search(
            ...     bbox=(-0.15, 51.47, -0.10, 51.52),
            ...     date_range=("2024-06-01", "2024-06-30"),
            ...     collections=["sentinel-2-l2a"],
            ...     cloud_cover_max=10,
            ... )
            >>> for r in results[:5]:
            ...     print(r)
        """
        # `limit` is a convenience alias for max_results (matches STAC convention)
        if limit is not None:
            max_results = limit
        return self.data.search(
            bbox=bbox,
            date_range=date_range,
            collections=collections,
            providers=providers,
            satellite=satellite,
            cloud_cover_max=cloud_cover_max,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            processing_level=processing_level,
            resolution_range=resolution_range,
            cql2_filter=cql2_filter,
            on_provider_failure=on_provider_failure,
            timeout=timeout,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Download (delegates to PyGeoFetch via self.data)
    # ------------------------------------------------------------------

    def download(
        self,
        items: list[SearchResult] | SearchResult,
        output_dir: str | Path = "./data",
        parallel: int = 4,
        verify_checksum: bool = True,
        resume: bool = True,
        retry_attempts: int = 5,
        post_process: list[str] | None = None,
        bandwidth_limit_mb: float | None = None,
        on_failure: str = "skip",
        overwrite: bool = False,
        notify_webhook: str | None = None,
        bands: list[str] | None = None,    # filter assets by band names
    ) -> list[DownloadResult]:
        """Download satellite scenes via pygeofetch.

        Delegates to ``pygeofetch download run`` for resilient, parallel
        downloading with full post-processing support.

        Post-processing actions (chained in order):
            unzip                    Extract ZIP/TAR archives
            reproject:EPSG:4326      Reproject to target CRS
            compress:lzw             Apply compression
            ndvi                     Compute NDVI
            ndwi                     Compute NDWI
            composite                Temporal composite
            atmospheric:sen2cor      Atmospheric correction
            clip:area.geojson        Clip to geometry
            resample:10              Resample to N metres
            cog                      Cloud Optimized GeoTIFF
            merge                    Merge overlapping scenes
            pan-sharpen              Pan-sharpen multispectral

        Args:
            items: SearchResult(s) from search().
            output_dir: Local download directory.
            parallel: Concurrent downloads (default 4).
            verify_checksum: SHA256 verify each file post-download.
            resume: Auto-resume interrupted downloads.
            retry_attempts: Max retries per file (exponential backoff).
            post_process: Processing chain (list of action strings).
            bandwidth_limit_mb: Download throttle in MB/s.
            on_failure: 'skip', 'abort', or 'retry'.
            overwrite: Overwrite existing files.
            notify_webhook: Slack/Teams webhook for completion notification.

        Returns:
            List of DownloadResult objects.

        Example:
            >>> results = client.search(bbox=..., date_range=...)
            >>> downloads = client.download(
            ...     results[:5],
            ...     output_dir="./sentinel2/",
            ...     parallel=4,
            ...     post_process=["unzip", "reproject:EPSG:4326", "compress:lzw", "cog"],
            ...     verify_checksum=True,
            ... )
            >>> for d in downloads:
            ...     print(d)
        """
        # If bands requested, filter assets on each SearchResult so that
        # only those band assets are passed to pygeofetch's download engine.
        if bands:
            for item in (items if isinstance(items, list) else [items]):
                if item.assets:
                    item.assets = {
                        k: v for k, v in item.assets.items()
                        if k in bands or any(b.lower() in k.lower() for b in bands)
                    }
        return self.data.download(
            items=items,
            output_dir=output_dir,
            parallel=parallel,
            verify_checksum=verify_checksum,
            resume=resume,
            retry_attempts=retry_attempts,
            post_process=post_process,
            bandwidth_limit_mb=bandwidth_limit_mb,
            on_failure=on_failure,
            overwrite=overwrite,
            notify_webhook=notify_webhook,
        )

    # ------------------------------------------------------------------
    # Pipelines (data + AI end-to-end)
    # ------------------------------------------------------------------

    def pipeline(
        self,
        pipeline_name: str,
        bbox: tuple[float, float, float, float],
        output_dir: str | Path = "./pipeline_output",
        **kwargs: Any,
    ) -> Any:
        """Run an end-to-end geospatial pipeline (data + AI).

        Downloads imagery via pygeofetch then runs a native AI model.

        Available pipelines:
            change_detection     Bi-temporal change detection
            land_cover           Global land cover (ESA WorldCover)
            building_footprints  Building segmentation (SAM)
            crop_monitoring      Crop type mapping
            disaster_assessment  Rapid damage assessment
            deforestation        Forest loss detection
            urban_growth         Urban expansion monitoring
            water_bodies         Surface water mapping (NDWI)
            solar_detection      Solar panel detection
            carbon_estimation    Biomass/carbon via NDVI

        Args:
            pipeline_name: Pipeline name from the list above.
            bbox: (min_lon, min_lat, max_lon, max_lat).
            output_dir: Output directory for results.
            **kwargs: Pipeline-specific arguments (date, date_before,
                      date_after, model, source, method, etc.).

        Returns:
            PipelineResult with output_path and stats.

        Example:
            >>> result = client.pipeline(
            ...     "building_footprints",
            ...     bbox=(-0.15, 51.47, -0.10, 51.52),
            ...     date="2024-06",
            ... )
            >>> print(f"Coverage: {result.stats['building_coverage']:.1%}")

            >>> result = client.pipeline(
            ...     "change_detection",
            ...     bbox=(-74.1, 40.6, -73.7, 40.9),
            ...     date_before="2020-01",
            ...     date_after="2024-01",
            ... )
        """
        from pygeovision.ai.pipelines import get_pipeline
        p = get_pipeline(pipeline_name, pgv_client=self)
        return p.run(bbox=bbox, output_dir=output_dir, **kwargs)

    def create_pipeline(
        self,
        name: str,
        description: str = "",
        schedule: str | None = None,
    ) -> DataPipeline:
        """Create a new pygeofetch YAML data pipeline programmatically.

        Example::

            pipeline = client.create_pipeline("weekly-sentinel2")
            pipeline.search(
                providers=["planetary_computer", "copernicus"],
                bbox=(-74.1, 40.6, -73.7, 40.9),
                date_range="last_7_days",
                cloud_cover="0-10",
            ).filter(
                "data.cloud_cover < 5"
            ).download(
                parallel=4,
                output="./raw/",
                post_process=["unzip", "reproject:EPSG:4326", "cog"],
            ).export(
                format="cloud_optimized_geotiff",
                destination="s3://my-bucket/",
            ).schedule("0 6 * * 1")
            pipeline.run()
        """
        return DataPipeline(name=name, description=description, schedule=schedule)

    def run_pipeline_yaml(
        self,
        pipeline_yaml: str | Path,
        step: str | None = None,
    ) -> dict[str, Any]:
        """Run a pygeofetch YAML pipeline file.

        Delegates to ``pygeofetch pipeline run FILE``.

        Example pipeline YAML::

            name: weekly-sentinel2-ndvi
            schedule: "0 6 * * 1"
            steps:
              - search:
                  providers: [copernicus, aws_earth, planetary_computer]
                  date_range: last_7_days
                  cloud_cover: 0-10
                  bbox: "-74.1,40.6,-73.7,40.9"
              - filter:
                  expression: "data.cloud_cover < 5"
              - download:
                  parallel: 4
                  output: ./raw/
                  verify_checksum: true
              - export:
                  format: cloud_optimized_geotiff
                  destination: s3://my-bucket/ndvi/

        Args:
            pipeline_yaml: Path to YAML pipeline file.
            step: Run only a specific named step.

        Returns:
            Dict with run summary.
        """
        return self.data.run_pipeline(pipeline_yaml, step=step)

    # ------------------------------------------------------------------
    # Provider management (delegates to PyGeoFetch)
    # ------------------------------------------------------------------

    def list_providers(
        self,
        auth_only: bool = False,
        open_only: bool = False,
        capabilities: list[str] | None = None,
    ) -> dict[str, dict]:
        """List all 22 pygeofetch satellite data providers.

        Args:
            auth_only: Only providers requiring authentication.
            open_only: Only open-access (no credentials needed) providers.
            capabilities: Filter by: 'sar', 'optical', 'stac', 'sub_meter'.

        Returns:
            Dict of provider_id → metadata (name, satellites, auth, etc.).
        """
        return self.data.list_providers(
            auth_only=auth_only, open_only=open_only, capabilities=capabilities
        )

    def test_provider(self, provider: str) -> bool:
        """Test connectivity to a pygeofetch provider."""
        return self.data.test_provider(provider)

    # ------------------------------------------------------------------
    # Cache management (delegates to PyGeoFetch)
    # ------------------------------------------------------------------

    def clear_cache(
        self,
        provider: str | None = None,
        older_than: str | None = None,
    ) -> None:
        """Clear pygeofetch search result cache.

        Args:
            provider: Clear only this provider's cache (None = all).
            older_than: Clear entries older than duration (e.g. '7d', '1h').
        """
        self.data.clear_cache(provider=provider, older_than=older_than)

    def cache_stats(self) -> dict[str, Any]:
        """Get pygeofetch cache statistics."""
        return self.data.cache_stats()

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return full PyGeoVision system status.

        Includes: pygeofetch version, native AI stack (torch, rasterio),
        registered AI models, and provider count.
        """
        info: dict[str, Any] = {
            "pygeovision_version": __version__,
            "python": platform.python_version(),
            "platform": platform.system(),
        }

        # pygeofetch status
        pf_status = self.data.status()
        info["pygeofetch"] = {
            "available": self.data._has_pygeofetch(),
            "version": self.data._pygeofetch_version(),
            "providers": 22,
            "open_providers": len([p for p in pf_status.get("open_providers", [])]),
        }

        # torch status
        try:
            import torch
            info["torch"] = {
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else (
                    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    else "cpu"
                ),
            }
            if torch.cuda.is_available():
                info["torch"]["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch"] = {"available": False}

        # rasterio status
        try:
            import rasterio
            info["rasterio"] = rasterio.__version__
        except ImportError:
            info["rasterio"] = None

        # geopandas status
        try:
            import geopandas
            info["geopandas"] = geopandas.__version__
        except ImportError:
            info["geopandas"] = None

        # AI model registry
        try:
            from pygeovision.ai.models.registry import registry
            info["registered_ai_models"] = len(registry)
        except Exception:
            info["registered_ai_models"] = 0

        return info

    def doctor(self) -> dict[str, Any]:
        """Run comprehensive diagnostics on the pygeofetch data layer."""
        return self.data.doctor()

    # ------------------------------------------------------------------
    # PyGeoFetch-integrated preprocessing gate
    # ------------------------------------------------------------------

    def prepare_for_ai(
        self,
        input_path: str,
        *,
        stack_bands: list[str] | None = None,
        stack_dir: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        bbox_crs: str = "EPSG:4326",
        clip_geojson: str | None = None,
        cloud_mask_path: str | None = None,
        scl_path: str | None = None,
        scl_keep_classes: tuple = (4, 5, 6),
        normalise: str | None = "scale_factor",
        scale_factor: float = 10000.0,
        resample_m: float | None = None,
        model_type: str = "segmentation",
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Complete preprocessing pipeline — raw scene → AI-ready array.

        This is the **mandatory gate** between satellite data and AI models.
        Routes through the :class:`PyGeoFetchBridge` which uses PyGeoFetch
        v2.0 preprocessing natively when available, falling back to
        PyGeoVision's own Preprocessor.

        Pipeline order:
            1. **Stack** — find + combine individual band files
            2. **Clip** — crop to study area (bbox or polygon)
            3. **Cloud mask** — apply cloud / SCL mask
            4. **Normalise** — scale pixel values to model range
            5. **Resample** — match model input resolution
            6. **Validate** — :class:`DataValidator` auto-fixes nulls,
               outliers, dtype, range

        Args:
            input_path: Stacked GeoTIFF **or** scene directory path.
            stack_bands: Band names to discover + stack
                (e.g. ``["B02","B03","B04","B08","B11","B12"]``).
            stack_dir: Alternative directory for band discovery.
            bbox: Clip bbox ``(minlon, minlat, maxlon, maxlat)`` WGS84.
            bbox_crs: CRS of the bbox (default EPSG:4326).
            clip_geojson: Path to GeoJSON polygon for clipping.
            cloud_mask_path: Binary cloud mask (1=cloud, 0=clear).
            scl_path: Sentinel-2 SCL band for cloud/shadow masking.
            scl_keep_classes: SCL classes to keep (default 4,5,6 =
                vegetation, bare soil, water).
            normalise: ``"scale_factor"`` | ``"minmax"`` | ``"zscore"``
                | ``"percentile"`` | ``None``.
            scale_factor: Divisor for ``"scale_factor"`` normalisation
                (10000 for Sentinel-2 L2A reflectance).
            resample_m: Target pixel size in metres.
            model_type: Hint for :class:`DataValidator` —
                ``"segmentation"``, ``"detection"``,
                ``"change_detection"``, ``"foundation"``, etc.
            output_path: Save preprocessed file here.

        Returns:
            Dict with:
              ``"array"``         — float32 numpy array ``(C, H, W)``
              ``"output_path"``   — path of the preprocessed file
              ``"shape"``         — ``(C, H, W)``
              ``"resolution_m"``  — pixel size in metres
              ``"report"``        — :class:`ValidationReport`
              ``"steps"``         — list of steps applied

        Example::

            result = client.prepare_for_ai(
                "./downloads/S2C_20240628/",
                stack_bands = ["B02","B03","B04","B08","B11","B12"],
                bbox        = (-74.1, 40.6, -73.7, 40.9),
                scl_path    = "./downloads/S2C_20240628/SCL.tif",
                normalise   = "scale_factor",
                model_type  = "segmentation",
                output_path = "ready.tif",
            )
            arr = result["array"]   # float32 (6, H, W), validated, AI-ready
        """
        return self._pgf_bridge.prepare_for_ai(
            input_path,
            stack_bands      = stack_bands,
            stack_dir        = stack_dir,
            bbox             = bbox,
            bbox_crs         = bbox_crs,
            clip_geojson     = clip_geojson,
            cloud_mask_path  = cloud_mask_path,
            scl_path         = scl_path,
            scl_keep_classes = scl_keep_classes,
            normalise        = normalise,
            scale_factor     = scale_factor,
            resample_m       = resample_m,
            model_type       = model_type,
            output_path      = output_path,
        )

    def pgf_pipeline(self, name: str) -> Any:
        """Return a PyGeoFetch chainable processing pipeline builder.

        Uses PyGeoFetch v2.0's native builder when available.
        Falls back to a PyGeoVision stub that routes each step through
        the :class:`PyGeoFetchBridge`.

        Example::

            result = (
                client.pgf_pipeline("sentinel2-ndvi")
                .atmos(method="dos1")
                .cloud_mask(method="scl", scl_band="SCL.tif")
                .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
                .reproject(crs="EPSG:4326")
                .ndvi(red="B04.tif", nir="B08.tif")
                .vectorize(threshold=0.3)
                .smooth(tolerance=0.5)
                .cog(compress="deflate")
                .run(input="scene.tif", output_dir="./processed/")
            )
        """
        return self._pgf_bridge.pipeline(name)

    def batch_process(
        self,
        inputs: list[str],
        chain: list[tuple[str, dict[str, Any]]],
        output_dir: str = "./processed/",
        parallel: int = 4,
    ) -> list[dict[str, Any]]:
        """Run a preprocessing chain on multiple scenes.

        Uses PyGeoFetch v2.0's parallel batch engine when available.
        Falls back to sequential processing via
        :class:`PyGeoFetchBridge`.

        Args:
            inputs: List of input GeoTIFF paths.
            chain: Ordered list of ``(operation, kwargs)`` tuples::

                [
                    ("clip",      {"bbox": (-74.1, 40.6, -73.7, 40.9)}),
                    ("reproject", {"crs": "EPSG:4326"}),
                    ("normalise", {"method": "scale_factor"}),
                    ("cog",       {}),
                ]

            output_dir: Output directory.
            parallel: Concurrent workers.

        Returns:
            List of ``{"input", "output", "success", "error"}`` dicts.

        Example::

            results = client.batch_process(
                inputs = ["scene1.tif", "scene2.tif", "scene3.tif"],
                chain  = [
                    ("clip",      {"bbox": (-74.1,40.6,-73.7,40.9)}),
                    ("normalise", {"method": "scale_factor"}),
                    ("cog",       {}),
                ],
                output_dir = "./processed/",
                parallel   = 4,
            )
            succeeded = [r for r in results if r["success"]]
            print(f"{len(succeeded)}/{len(results)} succeeded")
        """
        return self._pgf_bridge.batch_process(
            inputs, chain, output_dir=output_dir, parallel=parallel)

    def __repr__(self) -> str:
        pgf    = "✓" if self.data._has_pygeofetch() else "✗"
        pgf_v2 = "✓v2" if self._pgf_bridge._pgf_v2 else "✗"
        try:
            import torch  # noqa: F401
            ai_stack = "✓torch"
        except ImportError:
            ai_stack = "torch-not-installed"
        from pygeovision.ai.models.zoo import model_zoo
        from pygeovision.ai.pipelines.domains import list_pipelines
        from pygeovision.datasets.registry import dataset_registry
        return (
            f"PyGeoVision(v{__version__} | "
            f"pygeofetch={pgf} | pgf_v2={pgf_v2} | ai={ai_stack} | "
            f"datasets={len(dataset_registry)} | models={len(model_zoo)} | "
            f"pipelines={len(list_pipelines())} | "
            f"validator=✓ | preprocess=✓ | indices=22 | postprocess=✓ | "
            f"sar={'✓' if self._pgf_bridge._pgf_v2 else 'pgf_v2_pending'} | "
            f"labeling=7src | losses=10 | inference=4 | "
            f"xai=4 | monitoring=3 | cloud=3·aws·azure·gcp | "
            f"edge=ONNX+Jetson | vlm=CLIP+Moon | few_shot | timeseries | 3D)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2+ client proxy classes — thin facades on the new independent layers
# ─────────────────────────────────────────────────────────────────────────────

class _LabelingClientProxy:
    """client.labeling — 7+ auto-labeling sources."""
    def osm(self, bbox, categories=None, output_path="./labels/osm.tif", **kw):
        from pygeovision.labeling.osm import OSMLabeler
        return OSMLabeler().label(bbox, categories=categories, output_path=output_path, **kw)
    def microsoft_buildings(self, bbox, output_path="./labels/ms_buildings.tif", **kw):
        from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler
        return MicrosoftBuildingsLabeler().label(bbox, output_path=output_path, **kw)
    def google_buildings(self, bbox, output_path="./labels/google_buildings.tif", **kw):
        from pygeovision.labeling.buildings import GoogleBuildingsLabeler
        return GoogleBuildingsLabeler().label(bbox, output_path=output_path, **kw)
    def esa_worldcover(self, bbox, output_path="./labels/esa_worldcover.tif", **kw):
        from pygeovision.labeling.landcover import ESAWorldCoverLabeler
        return ESAWorldCoverLabeler().label(bbox, output_path=output_path, **kw)
    def dynamic_world(self, bbox, date_range=None, output_path="./labels/dynamic_world.tif", **kw):
        from pygeovision.labeling.landcover import DynamicWorldLabeler
        return DynamicWorldLabeler().label(bbox, date_range=date_range or ("2024-01-01","2024-12-31"),
                                            output_path=output_path, **kw)
    def sam_auto(self, image_path, output_path="./labels/sam.tif", **kw):
        from pygeovision.labeling.sam_auto import SAMAutoLabeler
        return SAMAutoLabeler().auto_label(image_path, output_path=output_path, **kw)
    def foundation(self, image_path, output_path="./labels/foundation.tif", **kw):
        from pygeovision.labeling.foundation import FoundationModelLabeler
        return FoundationModelLabeler().pseudo_label(image_path, output_path, **kw)
    def pipeline(self, bbox, sources=None, output_dir="./labels/", **kw):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        return AutoLabelPipeline(sources=sources).run(bbox, output_dir=output_dir, **kw)
    def quality(self, label_path):
        from pygeovision.labeling.quality import LabelQualityAssessor
        return LabelQualityAssessor().assess(label_path)
    def active_learner(self, strategy="entropy", budget=100):
        from pygeovision.labeling.active import ActiveLearner
        return ActiveLearner(strategy=strategy, budget=budget)
    def __repr__(self): return "LabelingLayer(osm|ms_buildings|google_buildings|esa_worldcover|dynamic_world|sam_auto|foundation|pipeline|quality|active)"


class _LossesClientProxy:
    """client.losses — geospatial-specific loss functions."""
    @property
    def dice(self): from pygeovision.losses.segmentation import DiceLoss; return DiceLoss()
    @property
    def focal(self): from pygeovision.losses.segmentation import FocalLoss; return FocalLoss()
    @property
    def tversky(self): from pygeovision.losses.segmentation import TverskyLoss; return TverskyLoss()
    @property
    def combo(self): from pygeovision.losses.segmentation import ComboLoss; return ComboLoss()
    @property
    def boundary(self): from pygeovision.losses.segmentation import BoundaryAwareLoss; return BoundaryAwareLoss()
    @property
    def lovasz(self): from pygeovision.losses.segmentation import LovaszLoss; return LovaszLoss()
    @property
    def ohem(self): from pygeovision.losses.segmentation import OhemCrossEntropy; return OhemCrossEntropy()
    @property
    def mixed(self): from pygeovision.losses.segmentation import GeospatialMixedLoss; return GeospatialMixedLoss()
    @property
    def ciou(self): from pygeovision.losses.detection import CIoULoss; return CIoULoss()
    @property
    def class_balanced(self): from pygeovision.losses.class_balance import ClassBalancedCrossEntropy; return ClassBalancedCrossEntropy()
    def get(self, name, **kw):
        MAP = {"dice": "DiceLoss", "focal": "FocalLoss", "tversky": "TverskyLoss",
               "combo": "ComboLoss", "boundary": "BoundaryAwareLoss", "lovasz": "LovaszLoss",
               "ohem": "OhemCrossEntropy", "mixed": "GeospatialMixedLoss"}
        if name not in MAP:
            raise ValueError(f"Loss '{name}' not found. Available: {list(MAP)}")
        mod = __import__("pygeovision.losses.segmentation", fromlist=[MAP[name]])
        return getattr(mod, MAP[name])(**kw)
    def __repr__(self): return "LossesLayer(dice|focal|tversky|combo|boundary|lovasz|ohem|mixed|ciou|class_balanced)"


class _InferenceClientProxy:
    """client.inference — advanced tiled/batch/streaming inference."""
    def tiled(self, model, chip_size=512, overlap=128, blend_mode="gaussian", **kw):
        from pygeovision.inference.tiled import TiledInference
        return TiledInference(model=model, chip_size=chip_size, overlap=overlap,
                               blend_mode=blend_mode, **kw)
    def batch(self, model, n_workers=4, **kw):
        from pygeovision.inference.batch import BatchInferenceEngine
        return BatchInferenceEngine(model=model, n_workers=n_workers, **kw)
    def streaming(self, model, chip_size=1024, **kw):
        from pygeovision.inference.stream import StreamingInference
        return StreamingInference(model=model, chip_size=chip_size, **kw)
    def ensemble(self, models, weights=None, fusion="mean", **kw):
        from pygeovision.inference.stream import EnsembleInference
        return EnsembleInference(models=models, weights=weights, fusion=fusion, **kw)
    def gaussian_blend(self, size=512, sigma_ratio=0.25):
        from pygeovision.inference.tiled import GaussianBlend
        return GaussianBlend.window(size, sigma_ratio)
    def __repr__(self): return "InferenceLayer(tiled|batch|streaming|ensemble|gaussian_blend)"


class _XAIClientProxy:
    """client.xai — explainability for geospatial models."""
    def gradcam(self, model, target_layer=None):
        from pygeovision.explainability.gradcam import GradCAM
        return GradCAM(model, target_layer)
    def gradcam_pp(self, model, target_layer=None):
        from pygeovision.explainability.gradcam import GradCAMPlusPlus
        return GradCAMPlusPlus(model, target_layer)
    def uncertainty(self, model, n_passes=20):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        return UncertaintyEstimator(model, n_passes)
    def attention(self, model):
        from pygeovision.explainability.attention import AttentionMapExtractor
        return AttentionMapExtractor(model)
    def shap(self, model):
        from pygeovision.explainability.shap_geo import GeospatialSHAP
        return GeospatialSHAP(model)
    def __repr__(self): return "XAILayer(gradcam|gradcam_pp|uncertainty|attention|shap)"


class _MonitoringClientProxy:
    """client.monitoring — drift detection and performance tracking."""
    def drift_detector(self, model=None):
        from pygeovision.monitoring.drift import DriftDetector
        return DriftDetector(model=model)
    def performance_tracker(self, model_name="model"):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        return ModelPerformanceTracker(model_name=model_name)
    def alert_manager(self, channels=None):
        from pygeovision.monitoring.alerts import AlertManager
        return AlertManager(channels=channels)
    def __repr__(self): return "MonitoringLayer(drift_detector|performance_tracker|alert_manager)"


class _EdgeClientProxy:
    """client.edge — edge deployment (Jetson, ONNX Runtime)."""
    def onnx_runtime(self, onnx_path, device="cpu"):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        return ONNXRuntimeInference(onnx_path, device=device)
    def export_onnx(self, model, output_path, input_shape=(1,4,512,512), **kw):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        return ONNXRuntimeInference.from_pytorch(model, output_path, input_shape, **kw)
    def jetson(self):
        from pygeovision.edge.jetson import JetsonDeployer
        return JetsonDeployer()
    def __repr__(self): return "EdgeLayer(onnx_runtime|export_onnx|jetson)"


class _CloudClientProxy:
    """client.cloud — cloud deployment (AWS, Azure, GCP)."""
    def aws(self, region="us-east-1", **kw):
        from pygeovision.cloud.deploy import AWSDeployer
        return AWSDeployer(region=region, **kw)
    def azure(self, **kw):
        from pygeovision.cloud.deploy import AzureDeployer
        return AzureDeployer(**kw)
    def gcp(self, project_id=None, region="us-central1"):
        from pygeovision.cloud.deploy import GCPDeployer
        return GCPDeployer(project_id=project_id, region=region)
    def deploy(self, provider, model_path, endpoint_name, **kw):
        from pygeovision.cloud.deploy import CloudDeployer
        return CloudDeployer.from_provider(provider, **kw).deploy(model_path, endpoint_name)
    def __repr__(self): return "CloudLayer(aws|azure|gcp|deploy)"


class _FewShotClientProxy:
    """client.few_shot — few-shot learning for geospatial classification."""
    def learner(self, backbone="dinov2-base", method="prototypical"):
        from pygeovision.advanced.few_shot import FewShotLearner
        return FewShotLearner(backbone=backbone, method=method)
    def __repr__(self): return "FewShotLayer(learner)"


class _MultiTaskClientProxy:
    """client.multitask — multi-task model training."""
    def model(self, backbone="resnet50", tasks=None, n_classes=None):
        from pygeovision.advanced.multitask import MultiTaskLearner
        return MultiTaskLearner(backbone=backbone, tasks=tasks, n_classes=n_classes)
    def __repr__(self): return "MultiTaskLayer(model)"


class _AutoMLClientProxy:
    """client.automl — automated hyperparameter optimisation."""
    def optimizer(self, metric="val_iou", n_trials=50, backend="optuna"):
        from pygeovision.advanced.automl import GeoAutoML
        return GeoAutoML(metric=metric, n_trials=n_trials, backend=backend)
    def __repr__(self): return "AutoMLLayer(optimizer)"


class _VLMClientProxy:
    """client.vlm — vision-language models (CLIP, Moondream)."""
    def clip(self, model="remoteclip-b32"):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        return CLIPGeo(model=model)
    def moondream(self):
        from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
        return MoondreamGeo()
    def retrieval(self, model="openclip-b32"):
        from pygeovision.advanced.vlm.retrieval import GeoImageRetrieval
        return GeoImageRetrieval(model=model)
    def zero_shot(self, image_path, categories, model="remoteclip-b32"):
        return self.clip(model).zero_shot(image_path, categories)
    def caption(self, image_path):
        return self.moondream().caption(image_path)
    def vqa(self, image_path, question):
        return self.moondream().vqa(image_path, question)
    def __repr__(self): return "VLMLayer(clip|moondream|retrieval|zero_shot|caption|vqa)"


class _TimeSeriesClientProxy:
    """client.timeseries — temporal analysis of satellite image stacks."""
    def analyzer(self, sensor="sentinel2"):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries(sensor=sensor)
    def ndvi_series(self, image_paths, date_strings=None, sensor="sentinel2"):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries(sensor).compute_index_series(image_paths, "ndvi", date_strings)
    def index_series(self, image_paths, index="ndvi", date_strings=None, sensor="sentinel2"):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries(sensor).compute_index_series(image_paths, index, date_strings)
    def __repr__(self): return "TimeSeriesLayer(analyzer|ndvi_series|index_series)"


class _PointCloudClientProxy:
    """client.pointcloud — 3D LiDAR and point cloud processing."""
    def processor(self):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        return PointCloudProcessor()
    def canopy_height_model(self, las_path, output_path, resolution=1.0):
        return self.processor().canopy_height_model(las_path, output_path, resolution)
    def __repr__(self): return "PointCloudLayer(processor|canopy_height_model)"


class _SegmentationClientProxy:
    """client.segmentation — raster segmentation (buildings, water, general SAM)."""
    def buildings(self, image_path, output_path="./output/buildings.tif", **kw):
        """Segment building footprints via SAM auto-segmentation."""
        from pygeovision.labeling.sam_auto import SAMAutoLabeler
        return SAMAutoLabeler().auto_label(image_path, output_path, **kw)
    def water(self, image_path, output_path="./output/water.tif", ndwi_threshold=0.0, **kw):
        """Extract water bodies via NDWI thresholding."""
        import numpy as np
        import rasterio
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            n_bands = src.count
            green_idx = 2 if n_bands >= 4 else 1
            nir_idx   = 4 if n_bands >= 4 else min(n_bands, 2)
            green = src.read(green_idx).astype(np.float32)
            nir   = src.read(nir_idx).astype(np.float32)
        ndwi = (green - nir) / (green + nir + 1e-8)
        water_mask = (ndwi > ndwi_threshold).astype(np.uint8)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(water_mask[np.newaxis])
        return {"output_path": output_path, "n_water_pixels": int(water_mask.sum()),
                "water_fraction": round(float(water_mask.mean()), 4),
                "method": "ndwi_threshold", "threshold": ndwi_threshold}
    def sam(self, image_path, output_path="./output/sam.tif", points_per_side=32, **kw):
        from pygeovision.labeling.sam_auto import SAMAutoLabeler
        return SAMAutoLabeler().auto_label(image_path, output_path,
                                            points_per_side=points_per_side, **kw)
    def custom(self, image_path, model, output_path="./output/pred.tif",
               chip_size=512, overlap=128, **kw):
        """Run any custom PyTorch segmentation model with tiled inference."""
        from pygeovision.inference.tiled import TiledInference
        return TiledInference(model=model, chip_size=chip_size, overlap=overlap,
                               **kw).infer(image_path, output_path)
    def __repr__(self): return "SegmentationLayer(buildings|water|sam|custom)"


class _DetectionClientProxy:
    """client.detection — object detection (generic, ships, cars) via native YOLO."""
    def generic(self, image_path, num_classes=5, class_names=None, output_path=None, **kw):
        from pygeovision.models.detection.yolo import GeoYOLO
        return GeoYOLO(num_classes=num_classes, class_names=class_names).detect(
            image_path, output_path=output_path, **kw)
    def ships(self, image_path, output_path=None, **kw):
        from pygeovision.models.detection.yolo import GeoYOLO
        return GeoYOLO(num_classes=1, class_names=["ship"]).detect(
            image_path, output_path=output_path, **kw)
    def cars(self, image_path, output_path=None, **kw):
        from pygeovision.models.detection.yolo import GeoYOLO
        return GeoYOLO(num_classes=1, class_names=["car"]).detect(
            image_path, output_path=output_path, **kw)
    def custom(self, image_path, model, output_path="./output/detections.tif", **kw):
        from pygeovision.inference.tiled import TiledInference
        return TiledInference(model=model, **kw).infer(image_path, output_path)
    def __repr__(self): return "DetectionLayer(generic|ships|cars|custom)"


class _ChangeDetectionClientProxy:
    """client.change — bi-temporal change detection (ChangeFormer, with a
    dependency-free spectral-diff fallback)."""
    def detect(self, before, after, output_path="./output/change.tif",
               method="changeformer", **kw):
        if method == "changeformer":
            try:
                from pygeovision.models.change_detection.changeformer import ChangeDetection
                cd = ChangeDetection(model_variant="changeformer", **kw)
                cd.build()
                return cd.detect(before, after, output_path=output_path)
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "ChangeFormer unavailable (%s) — using spectral-diff fallback", exc)
        return self._diff_change(before, after, output_path)
    def _diff_change(self, before, after, output_path):
        import numpy as np
        import rasterio
        with rasterio.open(before) as s1:
            img1 = s1.read().astype(np.float32)
            profile = s1.profile
        with rasterio.open(after) as s2:
            img2 = s2.read().astype(np.float32)
        if img1.shape != img2.shape:
            from scipy.ndimage import zoom as nd_zoom
            scale = (1.0, img1.shape[1] / img2.shape[1], img1.shape[2] / img2.shape[2])
            img2 = nd_zoom(img2, scale, order=1)
        diff = np.abs(img1 - img2).mean(axis=0)
        threshold = np.percentile(diff, 90)
        change_mask = (diff > threshold).astype(np.uint8)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(change_mask[np.newaxis])
        return {"output_path": output_path, "method": "spectral_diff",
                "change_fraction": round(float(change_mask.mean()), 4)}
    def __repr__(self): return "ChangeDetectionLayer(detect)"


class _ClassificationClientProxy:
    """client.classification — scene classification and land-cover mapping."""
    def scene(self, image_path, categories=None, **kw):
        """Zero-shot scene classification via CLIP."""
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        cats = categories or ["forest", "urban", "agriculture", "water", "barren"]
        return CLIPGeo().zero_shot(image_path, cats)
    def land_cover(self, image_path, year=2021, output_path="./output/land_cover.tif", **kw):
        """Land cover classification via ESA WorldCover, clipped to the image's extent."""
        import rasterio
        from rasterio.warp import transform_bounds

        from pygeovision.labeling.landcover import ESAWorldCoverLabeler
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return ESAWorldCoverLabeler(year=year).label(bbox, output_path=output_path, **kw)
    def __repr__(self): return "ClassificationLayer(scene|land_cover)"


# ─────────────────────────────────────────────────────────────────────────────
# SAR processing proxy — routes through PyGeoFetchBridge
# ─────────────────────────────────────────────────────────────────────────────

class _SARProxy:
    """``client.sar`` — SAR processing via PyGeoFetch v2.0 (forward-compatible).

    Methods delegate to PyGeoFetch v2.0's ``client.sar.*`` when available.
    A helpful warning is raised when PyGeoFetch v2.0 is not yet installed.

    Example::

        # Despeckle a Sentinel-1 scene
        out = client.sar.despeckle("sentinel1.tif", filter="enhanced_lee")

        # Radiometric calibration
        out = client.sar.calibrate("sentinel1.tif", output_type="sigma0", in_db=True)

        # Flood extent mapping
        out = client.sar.flood_map("post_flood.tif", threshold=-15.0,
                                    reference="pre_flood.tif")

        # InSAR coherence
        out = client.sar.coherence("slc_20240101.tif", "slc_20240113.tif", window=7)
    """

    def __init__(self, bridge: Any):
        self._b = bridge

    def despeckle(
        self,
        input_path: str,
        filter: str = "enhanced_lee",
        window: int = 5,
        **kwargs,
    ) -> str:
        """Speckle filter a SAR GeoTIFF.

        Args:
            input_path: Single-band SAR GeoTIFF (linear power or dB).
            filter: ``"lee"`` | ``"enhanced_lee"`` | ``"frost"`` | ``"gamma"``.
            window: Filter window size in pixels (odd number).

        Returns:
            Output path string.
        """
        return self._b.sar_despeckle(input_path, filter=filter, window=window, **kwargs)

    def calibrate(
        self,
        input_path: str,
        output_type: str = "sigma0",
        in_db: bool = True,
        **kwargs,
    ) -> str:
        """Radiometric calibration of raw SAR DN values.

        Args:
            input_path: Raw SAR digital-number GeoTIFF.
            output_type: ``"sigma0"`` (backscatter) | ``"gamma0"`` | ``"beta0"``.
            in_db: Output in dB scale (``10 * log10(linear)``).

        Returns:
            Output path string.
        """
        return self._b.sar_calibrate(
            input_path, output_type=output_type, in_db=in_db, **kwargs)

    def flood_map(
        self,
        post_path: str,
        threshold: float = -15.0,
        reference: str | None = None,
        **kwargs,
    ) -> str:
        """Map flood extent from post-event SAR imagery.

        Pixels below ``threshold`` dB are classified as water/flooded.
        When a ``reference`` pre-event image is provided, change-based
        detection is used instead of a fixed threshold.

        Args:
            post_path: Post-flood SAR GeoTIFF (calibrated dB).
            threshold: dB threshold below which pixels are flood
                (default -15.0 dB = open water).
            reference: Optional pre-flood SAR GeoTIFF for change-based
                detection.

        Returns:
            Output path of binary flood mask GeoTIFF.
        """
        return self._b.sar_flood_map(
            post_path, threshold=threshold, reference=reference, **kwargs)

    def coherence(
        self,
        slc1_path: str,
        slc2_path: str,
        window: int = 7,
        **kwargs,
    ) -> str:
        """Compute InSAR coherence between two co-registered SLC scenes.

        Args:
            slc1_path: First complex SLC GeoTIFF.
            slc2_path: Second complex SLC GeoTIFF.
            window: Estimation window size in pixels.

        Returns:
            Output path of coherence GeoTIFF (float32, [0, 1]).
        """
        return self._b.sar_coherence(slc1_path, slc2_path, window=window, **kwargs)

    def is_available(self) -> bool:
        """True when PyGeoFetch v2.0 SAR engine is available."""
        return self._b._pgf_v2 and hasattr(self._b._pgf, "sar")

    def __repr__(self) -> str:
        avail = "✓ PyGeoFetch v2.0" if self.is_available() else "⚠ requires PyGeoFetch v2.0"
        return f"SARProxy({avail} | despeckle|calibrate|flood_map|coherence)"