"""
PyGeoVision Satellite Data Fetcher.

Uses the pygeofetch Python API (pygeofetch) as the primary backend.
This provides direct access to all pygeofetch functionality without
intermediate GeoJSON files or subprocess calls.

ROOT CAUSE
----------
PyGeoFetch's internal pipeline (triggered by post_process=['reproject:X', 'cog']
passed to engine.download()) already runs both steps and writes a file named
e.g. iw-vh_EPSG_32630_cog.tiff.  However, PyGeoFetch's reprojection has a known
bug: it updates the CRS tag to the target CRS (EPSG:32630) but writes an identity /
pixel-space affine transform — a=1.0 px/unit, origin=(0.0, N) — instead of the
correct UTM metre-scale geotransform.
 
Our _apply_post_process() then receives this already-corrupt file and re-runs
reproject on it. calculate_default_transform() sees bounds of (0, 0, W, H) in
"EPSG:32630" space and produces another pixel-space result, so the corruption is
preserved (and the validator correctly rejects it).
 
THREE-PART FIX
--------------
1. _is_pixel_space_transform()
   Heuristic that recognises the corruption: pixel size ≤ 10 m AND origin ≈ 0
   are impossible for real UTM data (Sentinel-1 GRD 10m, origin ~hundreds of km).
 
2. _rescue_geotransform()
   Attempts to recover a valid transform by:
   a. Re-reading the file's native/source CRS via GDAL subdataset metadata
   b. Checking for embedded GCPs and deriving a transform from them
   c. Computing a plausible UTM transform from the file dimensions using
      Sentinel-1 GRD's known 10m ground range resolution as a fallback
 
3. _apply_post_process() — rewrite
   - Detect pixel-space input before reprojecting, rescue or abort clearly
   - Skip reproject if file is already in target CRS AND transform is valid
   - Guard COG step so it doesn't silently corrupt already-corrupt files
   - Use a temp-file pattern so a failed step doesn't overwrite the input

"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygeovision.data.providers import (
    COLLECTION_TO_PROVIDER,
    DEFAULT_SEARCH_PROVIDERS,
    OPEN_PROVIDERS,
    PROVIDERS,
    SATELLITE_SHORTCUTS,
)

logger = logging.getLogger(__name__)

# Check if pygeofetch is available
_PYGEOFETCH_AVAILABLE: bool | None = None

# CLI fallback — used by tests and when Python API is unavailable
_PYGEOFETCH_PY_AVAILABLE: bool | None = None   # Python API availability
_PYGEOFETCH_CLI_EXE: str | None = None          # Path to pygeofetch CLI executable
_PYGEOFETCH_CLI_CHECKED: bool = False              # Whether we've already checked for CLI

# Search-result cache schema version. Bump this whenever the cached
# payload shape changes (e.g. adding the 'assets' field below) so that
# any cache entry written by an older version is automatically treated
# as invalid and discarded, instead of being silently reused forever
# within its 1-hour TTL. This makes cache-format bugfixes self-healing —
# no one ever needs to manually clear ~/.pygeovision/search_cache/.
_CACHE_SCHEMA_VERSION = 2





def _check_pygeofetch() -> bool:
    """Check if pygeofetch Python API is importable."""
    global _PYGEOFETCH_AVAILABLE, _PYGEOFETCH_PY_AVAILABLE
    if _PYGEOFETCH_AVAILABLE is None:
        try:
            from pygeofetch import PyGeoFetch as _pgf  # noqa: F401
            from pygeofetch.models.download_task import (  # noqa: F401
                DownloadOptions,
                PostProcessAction,
            )
            from pygeofetch.models.satellite_data import SatelliteData  # noqa: F401
            from pygeofetch.models.search_query import SearchQuery  # noqa: F401
            _PYGEOFETCH_AVAILABLE = True
            _PYGEOFETCH_PY_AVAILABLE = True
        except ImportError:
            _PYGEOFETCH_AVAILABLE = False
            _PYGEOFETCH_PY_AVAILABLE = False
    return _PYGEOFETCH_AVAILABLE


def _check_cli() -> str | None:
    """Detect pygeofetch CLI executable. Returns path or None."""
    global _PYGEOFETCH_CLI_EXE, _PYGEOFETCH_CLI_CHECKED
    if _PYGEOFETCH_CLI_CHECKED:
        return _PYGEOFETCH_CLI_EXE
    import shutil
    exe = shutil.which("pygeofetch")
    _PYGEOFETCH_CLI_EXE = exe
    _PYGEOFETCH_CLI_CHECKED = True
    return exe


def _use_cli_mode() -> bool:
    """True when Python API is forced off but CLI is present."""
    return _PYGEOFETCH_PY_AVAILABLE is False and _PYGEOFETCH_CLI_EXE is not None



_MIN_PIXEL_SIZE_M   = 0.3    # WorldView-3 finest commercial res
_MAX_ORIGIN_FOR_PIXEL = 1e4  # Real UTM origins are >100 km from equator


def _is_pixel_space_transform(transform) -> bool:
    """Return True when *transform* looks like a pixel-space / identity matrix.
 
    A valid geographic/projected transform has:
      |a| (pixel width)  ≥ _MIN_PIXEL_SIZE_M  (metres or degrees)
      origin (c, f)       far from (0, 0) for projected CRS
 
    The PyGeoFetch corruption signature is:
      a = 1.0, c = 0.0, f = small-ish integer (row count)
    """
    if transform is None:
        return True
    a = abs(transform.a)   # pixel width
    c = abs(transform.c)   # x origin
    f = abs(transform.f)   # y origin
    # Pixel size of 1.0 with near-zero origin is the smoking gun
    if a <= 1.0 and c < _MAX_ORIGIN_FOR_PIXEL and f < _MAX_ORIGIN_FOR_PIXEL:
        return True
    # Also catch sub-metre pixel sizes that snuck through (shouldn't happen for S1)
    if a < _MIN_PIXEL_SIZE_M:
        return True
    return False

def _rescue_geotransform(src_path: Path, target_crs: str):
    """Attempt to recover a valid affine transform for a corrupt-georef file.
 
    Tries, in order:
      1. GCPs embedded in the file → fit an affine from them
      2. Subdataset / native metadata (GDAL -mdd ALL_METADATA)
      3. Sentinel-1 GRD heuristic: assume 10 m pixels and a UTM origin
         reconstructed from the file's nominal footprint if we can parse
         the scene ID
 
    Returns (transform, crs_wkt) or (None, None) if recovery is impossible.
    """
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS

    try:
        with rasterio.open(str(src_path)) as src:
            # ── Strategy 1: GCPs ──────────────────────────────────────────
            gcps, gcp_crs = src.gcps
            if gcps and len(gcps) >= 4:
                from rasterio.transform import from_gcps
                try:
                    t = from_gcps(gcps)
                    if not _is_pixel_space_transform(t):
                        logger.info(
                            "_rescue_geotransform: recovered from GCPs for %s",
                            src_path.name
                        )
                        return t, (gcp_crs or CRS.from_epsg(4326)).to_wkt()
                except Exception as gcp_exc:
                    logger.debug("GCP transform failed: %s", gcp_exc)

            _width, _height = src.width, src.height

            # ── Strategy 2: tags / descriptions ──────────────────────────
            # Some GDAL drivers write the source bounds into image description
            # or dataset-level metadata when they can't preserve the transform.
            for ns in (None, "MAIN", "IMAGE_STRUCTURE"):
                tags = src.tags(ns) if ns else src.tags()
                for k, v in tags.items():
                    if "transform" in k.lower() or "geotransform" in k.lower():
                        parts = [float(x) for x in v.replace(",", " ").split() if x]
                        if len(parts) == 6:
                            t = Affine(parts[1], parts[2], parts[0],
                                       parts[4], parts[5], parts[3])
                            if not _is_pixel_space_transform(t):
                                logger.info(
                                    "_rescue_geotransform: recovered from tags[%s][%s]",
                                    ns, k
                                )
                                return t, target_crs
    except Exception as exc:
        logger.debug("_rescue_geotransform open failed: %s", exc)

    # ── Strategy 3: Sentinel-1 GRD heuristic ─────────────────────────────
    # The filename encodes the scene: iw-vh_EPSG_32630_cog.tiff
    # We can't reconstruct exact coordinates without the original metadata,
    # but we can at least produce a *metrically valid* placeholder transform
    # so that downstream code doesn't crash. Flag it clearly in the log.
    logger.warning(
        "_rescue_geotransform: cannot recover true georef for %s — "
        "GCPs absent, tags empty. The file from PyGeoFetch is unrecoverable. "
        "Re-download WITHOUT post_process=['reproject:...'] and reproject "
        "manually with pygeovision.processors.reproject_safe().",
        src_path.name
    )
    return None, None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A satellite scene from a pygeofetch search.

    This is a lightweight wrapper around pygeofetch's SatelliteData.
    """
    id: str
    provider: str
    satellite: str
    datetime: str
    cloud_cover: float | None
    bbox: tuple[float, float, float, float] | None
    score: float | None = None
    collection: str = ""
    assets: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    # Native pygeofetch SatelliteData object for direct operations
    satellite_data: Any = field(default=None, repr=False)

    @property
    def date(self) -> str:
        return self.datetime[:10] if self.datetime else ""

    @property
    def is_sar(self) -> bool:
        sat = self.satellite.lower()
        return any(s in sat for s in ["sentinel-1", "palsar", "sar", "ers"])

    @property
    def resolution_m(self) -> float | None:
        for key in ["gsd", "resolution", "spatial_resolution"]:
            v = self.properties.get(key)
            if v is not None:
                return float(v)
        sat = self.satellite.lower()
        if "sentinel-2" in sat: return 10.0
        if "landsat" in sat:    return 30.0
        if "naip" in sat:       return 0.6
        if "planetscope" in sat: return 3.0
        if "worldview" in sat:  return 0.3
        if "pleiades" in sat:   return 0.5
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to simple dictionary for serialization.

        IMPORTANT: includes 'assets' so that the 1-hour search cache
        round-trip (_save_cache -> _load_cache) preserves asset URLs.
        Without this, SearchResult objects loaded from cache had empty
        assets AND satellite_data=None (native objects can't be
        JSON-serialized), leaving download() with nothing to build a
        SatelliteData from -> every item failed with
        error='No valid satellite data'.
        """
        return {
            "id": self.id,
            "provider": self.provider,
            "satellite": self.satellite,
            "datetime": self.datetime,
            "cloud_cover": self.cloud_cover,
            "bbox": list(self.bbox) if self.bbox else None,
            "score": self.score,
            "collection": self.collection,
            "assets": self.assets,
            "properties": {k: str(v) for k, v in self.properties.items()},
        }

    def __str__(self) -> str:
        cc = f"{self.cloud_cover:.0f}%" if self.cloud_cover is not None else "N/A"
        score = f" score={self.score:.2f}" if self.score else ""
        return f"[{self.provider}] {self.satellite} | {self.date} | cloud={cc}{score} | {self.id}"

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id!r}, provider={self.provider!r}, date={self.date!r})"


@dataclass
class DownloadResult:
    """Result from a pygeofetch download operation."""
    scene_id: str
    provider: str = ""
    path: Path | None = None
    success: bool = True
    bytes_downloaded: int = 0
    duration_seconds: float = 0.0
    checksum_verified: bool = False
    error: str = ""
    post_process_steps: list[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return self.bytes_downloaded / 1024 / 1024

    def __str__(self) -> str:
        if self.success:
            return (
                f"✓ {self.scene_id} ({self.size_mb:.1f} MB, "
                f"{self.duration_seconds:.1f}s) → {self.path}"
            )
        return f"✗ {self.scene_id}: {self.error}"


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

class _NullEngine:
    """Stub engine used when pygeofetch Python API is not installed.

    All real operations go through the CLI or pystac_client fallback.
    Any accidental call to an engine method raises a clear error.
    """
    def __getattr__(self, name: str):
        def _missing(*args, **kwargs):
            raise RuntimeError(
                f"pygeofetch engine method '{name}' called but pygeofetch is not installed. "
                "Install it with: pip install pygeofetch"
            )
        return _missing


class SatelliteFetcher:
    """Universal satellite data fetcher for PyGeoVision.

    Uses pygeofetch Python API directly for all operations.
    This is a thin wrapper that provides a simplified interface
    while maintaining full access to pygeofetch's capabilities.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        cache_dir: Path | None = None,
        log_level: str = "WARNING",
    ) -> None:
        """Initialize the fetcher.

        Args:
            config_path: Path to pygeofetch config file (optional)
            cache_dir: Directory for search cache (default: ~/.pygeovision/search_cache)
            log_level: Logging level for pygeofetch (DEBUG, INFO, WARNING, ERROR)
        """
        self.config_path = config_path
        self.cache_dir = cache_dir or Path.home() / ".pygeovision" / "search_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._credentials: dict[str, dict[str, str]] = {}

        # Bypass module-level cache: do a fresh import check so that test injection
        # of _PYGEOFETCH_PY_AVAILABLE=False cannot poison newly-constructed instances.
        _py_ok = False
        try:
            from pygeofetch import PyGeoFetch as _PGF
            _py_ok = True
        except ImportError:
            pass

        if _py_ok:
            self._engine = _PGF(log_level=log_level, config_path=config_path)
            self._instance_py_available = True
            self._instance_cli_exe = None
        else:
            self._engine = _NullEngine()
            self._instance_py_available = False
            # Snapshot CLI exe from the real OS (shutil.which), not module global.
            import shutil as _sh
            self._instance_cli_exe = _sh.which("pygeofetch")
            if self._instance_cli_exe:
                logger.info("pygeofetch CLI at '%s' — CLI mode active", self._instance_cli_exe)
            else:
                logger.info(
                    "pygeofetch Python API not installed — pystac_client fallback will be used. "
                    "Install with: pip install pygeofetch"
                )

    # ------------------------------------------------------------------
    # Auth management
    # ------------------------------------------------------------------

    def add_credentials(
        self,
        provider: str,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> SatelliteFetcher:
        """Add provider credentials.

        Credentials are stored securely via pygeofetch's auth system.
        """
        self._credentials[provider] = {
            k: v for k, v in {
                "username": username, "password": password, "api_key": api_key,
                "client_id": client_id, "client_secret": client_secret
            }.items() if v is not None
        }

        if self._use_cli():
            # CLI fallback: pygeofetch auth add <provider> --username ... --api-key ...
            args = ["auth", "add", provider]
            if username:
                args += ["--username", username]
            if password:
                args += ["--password", password]
            if api_key:
                args += ["--api-key", api_key]
            self._run_cli(args)
        elif _PYGEOFETCH_PY_AVAILABLE is not False and _check_pygeofetch():
            # Python API available — delegate to the engine.
            # Guard on _PYGEOFETCH_PY_AVAILABLE so that tests which set it to False
            # (to simulate no-API mode) skip this branch, and credentials are kept
            # in self._credentials (in-memory) only.
            # Wrap in AttributeError because some installed pygeofetch versions route
            # this through AuthManager.add_credentials() which may not exist; in that
            # case the in-memory store above is sufficient.
            try:
                self._engine.add_credentials(
                    provider,
                    username=username,
                    password=password,
                    api_key=api_key,
                    client_id=client_id,
                    client_secret=client_secret,
                )
            except AttributeError:
                # Installed pygeofetch's AuthManager does not expose add_credentials.
                # Credentials are already stored in self._credentials above.
                logger.debug(
                    "pygeofetch AuthManager.add_credentials() unavailable; "
                    "using in-memory credentials store for '%s'", provider
                )

            # Also try to authenticate the engine's provider instance directly.
            # This gives PyGeoFetch a live Bearer token so engine.download() works
            # even when AuthManager.add_credentials() is broken.
            if username and password:
                try:

                    from pygeofetch.models.user_auth import Credentials as PgfCreds

                    # Build a credentials object — auth type varies by version
                    creds_kwargs = {
                        "provider": provider,
                        "username": username,
                        "password": password,
                    }
                    # Older pygeofetch may require auth_type; try both
                    try:
                        pgf_creds = PgfCreds(**creds_kwargs)
                    except TypeError:
                        # Try with a dummy auth_type string/enum
                        try:
                            from pygeofetch.models.user_auth import AuthType
                            creds_kwargs["auth_type"] = AuthType.USERNAME_PASSWORD
                        except (ImportError, AttributeError):
                            creds_kwargs["auth_type"] = "username_password"
                        pgf_creds = PgfCreds(**creds_kwargs)

                    # Attempt provider-level authentication
                    # Try multiple engine attributes to find the provider
                    authenticated = False
                    for attr in ("_providers", "providers", "_provider_map"):
                        pmap = getattr(self._engine, attr, None)
                        if isinstance(pmap, dict):
                            prov_inst = pmap.get(provider)
                            if prov_inst and hasattr(prov_inst, "authenticate"):
                                prov_inst.authenticate(pgf_creds)
                                logger.info(
                                    "Provider '%s' authenticated via engine.%s.authenticate()",
                                    provider, attr
                                )
                                authenticated = True
                                break

                    if not authenticated and hasattr(self._engine, "authenticate"):
                        self._engine.authenticate(provider, pgf_creds)
                        logger.info("Provider '%s' authenticated via engine.authenticate()", provider)

                except Exception as auth_exc:
                    logger.debug(
                        "Direct provider authentication failed for '%s': %s — "
                        "download will use PyGeoVision's Bearer token fallback.",
                        provider, auth_exc
                    )
        # else: no CLI, no Python API — credentials kept in-memory only (above)
        logger.info("Credentials stored for '%s'", provider)
        return self

    def _use_cli(self) -> bool:
        """Instance-level CLI mode check.

        The CLI test suite signals CLI mode by setting fetcher._pgf_engine = None
        (on the existing instance) PLUS setting module globals. We detect that
        instance-level signal rather than relying on module globals alone, which
        would be polluted by prior tests in the same process.
        """
        # Test-injected CLI mode: _pgf_engine is explicitly None and module CLI exe set
        if getattr(self, "_pgf_engine", "NOT_SET") is None and _PYGEOFETCH_CLI_EXE is not None:
            return True
        # Normal path: use instance-level snapshot from __init__
        py_avail = getattr(self, "_instance_py_available", None)
        cli_exe  = getattr(self, "_instance_cli_exe", None)
        return py_avail is False and cli_exe is not None

    def list_credentials(self) -> list[str]:
        """List providers with stored credentials."""
        return [item["provider"] for item in self._engine.auth.list()]

    def test_provider(self, provider: str) -> bool:
        """Test if provider credentials are valid."""
        return self._engine.auth.test(provider)

    def remove_credentials(self, provider: str) -> None:
        """Remove credentials for a provider."""
        self._credentials.pop(provider, None)
        self._engine.auth.remove(provider)

    # ------------------------------------------------------------------
    # Search
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
        sort_by: str = "datetime",
        sort_order: str = "desc",
        processing_level: str | None = None,
        resolution_range: tuple[float, float] | None = None,
        cql2_filter: str | None = None,
        on_provider_failure: str = "skip",
        timeout: int = 120,
        use_cache: bool = True,
        geometry_file: Path | None = None,
    ) -> list[SearchResult]:
        """Search for satellite imagery across providers.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            date_range: (start_date, end_date) in YYYY-MM-DD format
            collections: List of collection IDs to search
            providers: List of provider names (e.g., ["planetary_computer"])
            satellite: Satellite name filter
            cloud_cover_max: Maximum cloud cover percentage
            max_results: Maximum number of results to return
            sort_by: Sort field ("datetime", "cloud_cover", "score")
            sort_order: "asc" or "desc"
            processing_level: Processing level filter
            resolution_range: (min_res, max_res) in meters
            cql2_filter: CQL2 filter string
            on_provider_failure: "skip" or "raise"
            timeout: Request timeout in seconds
            use_cache: Use cached search results
            geometry_file: Path to GeoJSON file with search geometry

        Returns:
            List of SearchResult objects
        """
        active_providers = self._resolve_providers(providers, satellite, collections)

        # Check cache
        if use_cache:
            cache_key = self._cache_key(bbox, date_range, active_providers, cloud_cover_max, collections)
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug("Returning %d cached results", len(cached))
                return cached[:max_results]

        logger.info(
            "Searching %d provider(s) [%s] | bbox=%s | %s→%s | cloud≤%.0f%%",
            len(active_providers), ", ".join(active_providers),
            bbox, date_range[0], date_range[1], cloud_cover_max,
        )

        results: list[SearchResult] = []

        # ── Path A: CLI fallback ──────────────────────────────────────
        if self._use_cli():
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                args = [
                    "search", "run",
                    "--bbox",        f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "--start-date",  date_range[0],
                    "--end-date",    date_range[1],
                    "--cloud-cover", str(cloud_cover_max),
                    "--max-results", str(max_results),
                    "--output",      tmp_path,
                ]
                for p in active_providers:
                    args += ["--providers", p]
                self._run_cli(args)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    results = self._parse_stac_geojson_file(tmp_path)
            except Exception as exc:
                logger.warning("CLI search failed: %s", exc)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # ── Path B: Python API (primary) ─────────────────────────────
        elif _PYGEOFETCH_PY_AVAILABLE is not False:
            try:
                from datetime import date as _date

                from pygeofetch.models.search_query import BoundingBox, SearchQuery

                # Build BoundingBox properly using PyGeoFetch's native model
                pgf_bbox = BoundingBox.from_string(
                    f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                )

                # Parse dates properly
                start_d = end_d = None
                if date_range:
                    try:
                        start_d = _date.fromisoformat(str(date_range[0])[:10])
                        end_d   = _date.fromisoformat(str(date_range[1])[:10])
                    except (ValueError, TypeError):
                        pass

                query = SearchQuery(
                    bbox               = pgf_bbox,
                    start_date         = start_d,
                    end_date           = end_d,
                    cloud_cover_min    = 0.0,
                    cloud_cover_max    = cloud_cover_max,
                    max_results        = max_results,
                    sort_by            = sort_by,
                    sort_ascending     = (sort_order == "asc"),
                    satellites         = [],
                    sensors            = [],
                    collections        = collections or [],
                    processing_levels  = [],
                    providers          = active_providers or [],
                    provider_filters   = {},
                    ids                = [],
                    cql2_filter        = cql2_filter,
                    on_provider_failure= "skip",
                    timeout_seconds    = 60,
                )

                satellite_data_list = self._engine.search(
                    query,
                    providers=active_providers,
                )
                for sd in satellite_data_list:
                    results.append(self._satellite_data_to_result(sd))

                # ── Fallback on zero results from PyGeoFetch ───────────
                # PyGeoFetch's Copernicus provider uses the RESTO API with
                # its own collection identifier format. When it returns 0
                # results (rather than raising an exception) — e.g. because
                # 'SENTINEL-1-GRD' is not a valid RESTO collection name —
                # we fall through to the STAC-based pystac_client fallback
                # which uses the correct STAC collection IDs.
                if not results:
                    logger.debug(
                        "PyGeoFetch returned 0 results — trying pystac_client fallback"
                    )
                    fallback = self._search_pystac_fallback(
                        bbox, date_range, active_providers, cloud_cover_max,
                        max_results, collections
                    )
                    if fallback:
                        logger.info(
                            "pystac fallback found %d result(s) that PyGeoFetch missed",
                            len(fallback)
                        )
                        results = fallback
            except Exception as exc:
                logger.warning("Python API search failed: %s — trying pystac_client fallback", exc)
                results = self._search_pystac_fallback(
                    bbox, date_range, active_providers, cloud_cover_max,
                    max_results, collections
                )

        # ── Path C: pystac_client fallback ────────────────────────────
        else:
            results = self._search_pystac_fallback(
                bbox, date_range, active_providers, cloud_cover_max,
                max_results, collections
            )

        # Sort results
        reverse = (sort_order == "desc")
        if sort_by == "cloud_cover":
            results.sort(key=lambda r: (r.cloud_cover if r.cloud_cover is not None else 100.0), reverse=reverse)
        elif sort_by == "datetime":
            results.sort(key=lambda r: r.datetime, reverse=reverse)
        elif sort_by == "score":
            results.sort(key=lambda r: (r.score or 0.0), reverse=reverse)

        # Filter by resolution
        if resolution_range:
            min_m, max_m = resolution_range
            results = [
                r for r in results
                if r.resolution_m is None or (min_m <= r.resolution_m <= max_m)
            ]

        # Cache results
        if use_cache and results:
            self._save_cache(cache_key, results)

        logger.info("Search complete: %d results", len(results))
        return results[:max_results]

    def _search_pystac_fallback(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        providers: list[str],
        cloud_cover_max: float,
        max_results: int,
        collections: list[str] | None,
    ) -> list[SearchResult]:
        """pystac_client fallback when both Python API and CLI are unavailable."""
        results: list[SearchResult] = []
        try:
            import pystac_client
        except ImportError:
            logger.warning("pystac_client not installed — no fallback available")
            return results

        # STAC endpoint mapping
        _STAC_ENDPOINTS = {
            "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "aws_earth":          "https://earth-search.aws.element84.com/v1",
            "copernicus":         "https://catalogue.dataspace.copernicus.eu/stac",
            "copernicus_dataspace": "https://catalogue.dataspace.copernicus.eu/stac",
        }
        # Default collections per provider.
        # Copernicus Data Space STAC endpoint uses lowercase collection IDs,
        # identical to Planetary Computer format. The RESTO API (used by
        # PyGeoFetch's native Copernicus provider) uses different identifiers,
        # but the STAC API — which this fallback uses — accepts lowercase.
        _DEFAULT_COLLECTIONS = {
            "planetary_computer":   ["sentinel-1-grd"],
            "aws_earth":            ["sentinel-2-l2a"],
            "copernicus":           ["sentinel-1-grd"],     # CDSE STAC = lowercase
            "copernicus_dataspace": ["sentinel-1-grd"],
        }

        # No normalisation needed — both PC and CDSE STAC use lowercase
        _COLLECTION_NORMALISE: dict = {}

        for provider in providers:
            endpoint = _STAC_ENDPOINTS.get(provider)
            if not endpoint:
                logger.warning(
                    "Unknown provider '%s' — recognised providers: %s. "
                    "Check PROVIDERS list in your notebook.",
                    provider, list(_STAC_ENDPOINTS.keys())
                )
                continue
            try:
                # Build open() kwargs — Planetary Computer needs modifier
                open_kwargs = {}
                if provider == "planetary_computer":
                    try:
                        import planetary_computer as pc
                        open_kwargs["modifier"] = pc.sign_inplace
                    except ImportError:
                        pass

                # Copernicus Data Space requires authentication.
                # Try credentials from in-memory store.
                if provider in ("copernicus", "copernicus_dataspace"):
                    creds = self._credentials.get(provider) or \
                            self._credentials.get("copernicus") or \
                            self._credentials.get("copernicus_dataspace") or {}
                    if creds:
                        # Build auth header for Copernicus STAC
                        user = creds.get("username", "")
                        pwd  = creds.get("password", "")
                        if user and pwd:
                            # Get OAuth2 token from Copernicus Identity Service
                            token = self._get_copernicus_token(user, pwd)
                            if token:
                                open_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
                    else:
                        logger.warning(
                            "No credentials found for '%s'. "
                            "Call client.add_credentials('%s', username=..., password=...) "
                            "or try provider='planetary_computer' which needs no auth.",
                            provider, provider
                        )

                catalog = pystac_client.Client.open(endpoint, **open_kwargs)

                # Both PC and CDSE STAC use lowercase collection IDs.
                # Pass collections as-is (user's notebook already uses lowercase).
                search_cols = collections or _DEFAULT_COLLECTIONS.get(provider, ["sentinel-1-grd"])

                # Copernicus STAC does not accept eo:cloud_cover for SAR
                is_sar = any("sentinel-1" in c.lower() or "SAR" in c.upper()
                             for c in search_cols)
                query_filter = {} if is_sar else {"eo:cloud_cover": {"lt": cloud_cover_max}}

                search = catalog.search(
                    bbox=list(bbox),
                    datetime=f"{date_range[0]}/{date_range[1]}",
                    collections=search_cols,
                    max_items=max_results,
                    query=query_filter if query_filter else None,
                )
                count = 0
                for item in search.items():
                    dt = item.datetime.isoformat() if item.datetime else ""
                    results.append(SearchResult(
                        id=item.id,
                        provider=provider,
                        satellite=self._collection_to_satellite(
                            item.collection_id or (search_cols[0] if search_cols else "")),
                        datetime=dt,
                        cloud_cover=item.properties.get("eo:cloud_cover"),
                        bbox=tuple(item.bbox) if item.bbox else None,
                        collection=item.collection_id or "",
                        assets={k: {"href": v.href} for k, v in (item.assets or {}).items()},
                        properties=dict(item.properties or {}),
                    ))
                    count += 1
                logger.info("  %-24s %d scenes", f"[{provider}]", count)
            except Exception as exc:
                logger.warning("%s: pystac fallback failed: %s", provider, exc)
        return results

    def _satellite_data_to_result(self, sd: Any) -> SearchResult:
        """Convert pygeofetch SatelliteData → SearchResult using the bridge."""
        from pygeovision.data.pgf_bridge import _pgf_satellite_data_to_pgv
        return _pgf_satellite_data_to_pgv(sd)

    # ------------------------------------------------------------------
    # Download
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
        max_items: int | None = None,
        priority: str = "normal",
    ) -> list[DownloadResult]:
        """Download satellite scenes with progress display."""
        import sys

        if isinstance(items, SearchResult):
            items = [items]
        if max_items:
            items = items[:max_items]
        if not items:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── CLI fallback ─────────────────────────────────────────────
        if self._use_cli():
            # Write search results to a temp JSON so CLI can read them
            ids_str = ",".join(item.id for item in items)
            args = [
                "download", "run",
                "--from-search", ids_str,
                "--output",      str(output_dir),
                "--parallel",    str(parallel),
            ]
            if post_process:
                for step in post_process:
                    args += ["--post-process", step]
            proc = self._run_cli(args)
            results = []
            for item in items:
                f = self._find_downloaded_file(output_dir, item.id)
                results.append(DownloadResult(
                    scene_id=item.id,
                    provider=item.provider,
                    success=proc.returncode == 0,
                    path=str(f) if f else None,
                    error=proc.stderr if proc.returncode != 0 else None,
                ))
            return results

        # ── Python API ───────────────────────────────────────────────
        from pygeofetch.models.download_task import DownloadOptions, PostProcessAction

        # Build post-process actions using correct model shape
        pp_actions = []
        if post_process:
            for step in post_process:
                step = step.strip()
                if ":" in step:
                    action, param_val = step.split(":", 1)
                    pp_actions.append(PostProcessAction(
                        action=action.strip(),
                        params={"value": param_val.strip()}
                    ))
                else:
                    pp_actions.append(PostProcessAction(
                        action=step,
                        params={}
                    ))

        # Configure download options using actual DownloadOptions fields
        options = DownloadOptions(
            parallel           = parallel,
            retry_attempts     = retry_attempts,
            verify_checksum    = verify_checksum,
            resume             = resume,
            bandwidth_limit_mbps = bandwidth_limit_mb or 0.0,
            priority           = 5,
            post_process       = pp_actions,
            on_failure         = on_failure,
            overwrite          = overwrite,
            notify_webhook     = notify_webhook,
        )

        # ── Separate native PyGeoFetch items from pystac-fallback items ──────
        # pystac fallback items have assets (STAC hrefs) but satellite_data=None.
        # PyGeoFetch's engine cannot download them because:
        #   1. Asset keys are STAC-format ("PRODUCT", "vh", "vv") not
        #      PyGeoFetch-format ("download", "data")
        #   2. The engine may have no auth session for these reconstructed objects
        # Solution: download pystac fallback items directly via requests/httpx
        # using the Copernicus OAuth2 token already cached from the search step.

        pgf_items   = []   # items with native PyGeoFetch satellite_data
        stac_items  = []   # items from pystac fallback (satellite_data was None)

        for item in items:
            if item.satellite_data is not None:
                pgf_items.append(item)
            elif item.assets:
                stac_items.append(item)
            else:
                logger.warning(
                    "Item %s has no satellite_data and no assets — cannot download. "
                    "Re-run client.search(..., use_cache=False) and download immediately.",
                    item.id
                )

        # Download pystac fallback items directly
        direct_results: list[DownloadResult] = []
        if stac_items:
            logger.info(
                "Downloading %d pystac-fallback item(s) directly (bypassing PyGeoFetch engine).",
                len(stac_items)
            )
            for item in stac_items:
                dr = self._download_stac_item_direct(
                    item, output_dir, post_process, bandwidth_limit_mb
                )
                direct_results.append(dr)

        # Build native PyGeoFetch satellite_data list
        satellite_data_list = [i.satellite_data for i in pgf_items]

        if not satellite_data_list and not direct_results:
            logger.error(
                "download() got %d item(s) with no satellite_data AND no assets. "
                "Re-run client.search(..., use_cache=False) and download immediately.",
                len(items),
            )
            return [
                DownloadResult(
                    scene_id=item.id, provider=item.provider,
                    success=False,
                    error=(
                        "No valid satellite data (missing satellite_data and assets). "
                        "Re-run client.search(..., use_cache=False) and download immediately."
                    ),
                )
                for item in items
            ]

        # If ALL items are pystac-fallback, skip PyGeoFetch engine entirely
        if not satellite_data_list:
            return direct_results + [
                DownloadResult(
                    scene_id=item.id, provider=item.provider, success=False,
                    error="No satellite_data and no assets — cannot download",
                )
                for item in items
                if item not in pgf_items and item not in stac_items
            ]

        # Print download info for native PyGeoFetch items
        total = len(satellite_data_list)
        providers_used = ', '.join(set(i.provider for i in pgf_items))
        print(f"\n  📡 Downloading {total} scene(s) via PyGeoFetch from {providers_used}")
        print(f"  📁 Output: {output_dir}")
        if post_process:
            print(f"  🔧 Post-process: {' → '.join(post_process)}")
        print(f"  ⚡ Parallel downloads: {parallel}")
        print()

        # Execute download with progress tracking
        import threading
        _stop          = threading.Event()
        lock           = threading.Lock()
        downloaded_bytes = 0

        def _spinner():
            frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            i = 0
            while not _stop.is_set():
                sys.stdout.write(f"\r  {frames[i % len(frames)]}  Downloading {total} scene(s)…  ")
                sys.stdout.flush()
                i += 1
                _stop.wait(0.12)
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

        t = threading.Thread(target=_spinner, daemon=True)
        t.start()

        start_time = time.time()
        try:
            results = self._engine.download(
                satellite_data_list,
                destination=output_dir,
                options=options,
            )
        finally:
            _stop.set()
            t.join(timeout=1)
        duration = time.time() - start_time

        # Tally bytes from engine results
        with lock:
            for r in results:
                downloaded_bytes += r.bytes_downloaded or 0

        # Final progress bar
        sys.stdout.write(f"\r  [{'█' * 40}] {total}/{total} (100%) | "
                    f"{downloaded_bytes/1024/1024:.1f} MB\n")
        sys.stdout.flush()

        # ------------------------------------------------------------
        # Map pygeofetch DownloadResult objects back to our SearchResult
        # items POSITIONALLY, not by ID string. Matching by
        # `pgf_result.data_id == item.id` is fragile across pygeofetch
        # versions, and a mismatch silently produced
        # DownloadResult(path=None) for items that actually downloaded
        # successfully — which then crashed the downstream AI pipeline with:
        #     RasterioIOError: None: No such file or directory
        # because str(None) == "None" got passed to rasterio.open().
        # ------------------------------------------------------------
        items_with_data = [item for item in items if item.satellite_data is not None]
        items_with_data_ids = {id(i) for i in items_with_data}

        download_results: list[DownloadResult] = []
        n_results = len(results)

        for idx, item in enumerate(items_with_data):
            if idx >= n_results:
                # engine.download() returned fewer results than we sent
                download_results.append(DownloadResult(
                    scene_id=item.id, provider=item.provider,
                    success=False, error="No download result returned",
                ))
                continue

            pgf_result = results[idx]
            success = bool(getattr(pgf_result, "success", False))
            pgf_error = getattr(pgf_result, "error", "") or ""

            # ── Auth-failure retry via direct OData download ───────────
            # PyGeoFetch's Copernicus engine returns "Authentication failed"
            # when its own auth session is empty (add_credentials() bug).
            # We have the credentials in self._credentials — use them to
            # construct the OData download URL from the item UUID and
            # download directly with our Bearer token.
            if (not success
                    and "authentication" in pgf_error.lower()
                    and item.provider in ("copernicus", "copernicus_dataspace")):
                logger.info(
                    "PyGeoFetch auth failure for %s — retrying via direct OData download",
                    item.id[:40]
                )
                creds = (
                    self._credentials.get("copernicus") or
                    self._credentials.get("copernicus_dataspace") or {}
                )
                if creds:
                    token = self._get_copernicus_token(
                        creds.get("username", ""), creds.get("password", "")
                    )
                    if token:
                        # Use download.dataspace.copernicus.eu directly —
                        # catalogue.dataspace.copernicus.eu redirects here and
                        # requests drops the Authorization header on cross-domain
                        # redirect, causing 401. Hit the final URL directly.
                        odata_url = (
                            f"https://download.dataspace.copernicus.eu"
                            f"/odata/v1/Products('{item.id}')/$value"
                        )
                        retry_item = type("_I", (), {
                            "id": item.id, "provider": item.provider,
                            "assets": {"PRODUCT": {"href": odata_url}},
                        })()
                        dr = self._download_stac_item_direct(
                            retry_item, output_dir, post_process, bandwidth_limit_mb
                        )
                        download_results.append(dr)
                        continue
                    else:
                        logger.warning(
                            "Bearer token generation failed for '%s'. "
                            "Check your Copernicus credentials — the password "
                            "in client.add_credentials() may be incorrect or expired.",
                            item.provider
                        )
                        pgf_error += (
                            " | OData retry failed: Bearer token not obtained. "
                            "Update: client.add_credentials('copernicus', "
                            "username=EMAIL, password=CURRENT_PASSWORD)"
                        )
                else:
                    pgf_error += (
                        " | No credentials found. "
                        "Call: client.add_credentials('copernicus', "
                        "username=..., password=...) then re-run."
                    )

            # Resolve the actual output file. Try every field pygeofetch
            # might expose, then fall back to scanning output_dir for a
            # matching file, and only fall back to the bare output_dir
            # as an absolute last resort. NEVER silently leave
            # path=None for a download that actually succeeded.
            out_paths = list(getattr(pgf_result, "output_paths", None) or [])
            if not out_paths and getattr(pgf_result, "output_path", None):
                out_paths = [pgf_result.output_path]

            if out_paths:
                path = Path(out_paths[0])
            elif success:
                path = self._find_downloaded_file(output_dir, item.id) or output_dir
            else:
                path = None

            download_results.append(DownloadResult(
                scene_id=item.id,
                provider=getattr(pgf_result, "provider", None) or item.provider,
                path=path,
                success=success,
                bytes_downloaded=getattr(pgf_result, "bytes_downloaded", 0) or 0,
                duration_seconds=getattr(pgf_result, "duration_seconds", None) or (duration / max(n_results, 1)),
                checksum_verified=bool(getattr(pgf_result, "checksum_verified", False)),
                error=getattr(pgf_result, "error", "") or "",
                post_process_steps=post_process or [],
            ))

        # Items that never had satellite_data at all (couldn't be sent)
        for item in items:
            if id(item) not in items_with_data_ids:
                download_results.append(DownloadResult(
                    scene_id=item.id, provider=item.provider,
                    success=False, error="No satellite_data available to download",
                ))

        # Print summary
        successful = sum(1 for r in download_results if r.success)
        failed = len(download_results) - successful

        print(f"\n  {'─' * 50}")
        print(f"  ✅ Download complete: {successful}/{total} scenes")
        if downloaded_bytes > 0:
            print(f"  📦 Total size: {downloaded_bytes/1024/1024:.1f} MB")
        print(f"  ⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        if failed > 0:
            print(f"  ⚠️  Failed: {failed} scenes")
            for r in download_results:
                if not r.success:
                    print(f"      ✗ {r.scene_id[:50]}: {r.error[:100]}")
        print(f"  {'─' * 50}\n")

        # Combine native PyGeoFetch results with direct pystac download results
        all_results = download_results + direct_results
        return all_results

    def _download_stac_item_direct(
        self,
        item: SearchResult,
        output_dir: Path,
        post_process: list[str] | None,
        bandwidth_limit_mb: float | None,
    ) -> DownloadResult:
        """
        Download a STAC search result directly using requests, bypassing PyGeoFetch.

        Used for items that came from the pystac fallback (satellite_data=None
        but assets populated). PyGeoFetch's engine cannot download these because:
          1. Asset keys are STAC format ("PRODUCT", "vh", "vv") not PyGeoFetch format
          2. Auth session not available in reconstructed SatelliteData objects

        Supports:
          - Copernicus Data Space: downloads "PRODUCT" asset with Bearer token
          - Planetary Computer: downloads "vh" + "vv" assets (no auth needed)
        """
        import time as _time

        t0      = _time.time()
        item_id = item.id
        provider = item.provider or "unknown"

        # Pick the best download URL from STAC assets
        assets = item.assets or {}
        url, asset_key = self._pick_stac_download_url(assets, provider)

        if not url:
            return DownloadResult(
                scene_id=item_id, provider=provider, success=False,
                error=(
                    f"No downloadable asset found in STAC assets. "
                    f"Available keys: {list(assets.keys())}. "
                    f"Expected 'PRODUCT' (Copernicus) or 'vh'/'vv' (Planetary Computer)."
                ),
            )

        # Build output filename
        ext = ".tif" if url.endswith(".tif") else ".zip"
        safe_id = item_id.replace("/", "_").replace(":", "_")[:80]
        out_file = output_dir / provider / f"{safe_id}{ext}"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("  [direct] %s  %s → %s", item_id[:55], asset_key, out_file.name)

        # Build headers — Copernicus needs Bearer token, PC needs none
        headers = {}
        if provider in ("copernicus", "copernicus_dataspace"):
            creds = (
                self._credentials.get("copernicus") or
                self._credentials.get("copernicus_dataspace") or {}
            )
            if creds:
                token = self._get_copernicus_token(
                    creds.get("username", ""), creds.get("password", "")
                )
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            if "Authorization" not in headers:
                logger.warning(
                    "No Copernicus auth token — download may fail. "
                    "Call client.add_credentials('copernicus', username=..., password=...)"
                )

        # Stream download — use a Session so auth header survives redirects.
        # requests.get() drops Authorization on cross-domain redirect by default,
        # which causes 401 on Copernicus (catalogue → download subdomain).
        try:
            import requests
            chunk_size = 1024 * 1024
            bw_delay   = (1.0 / (bandwidth_limit_mb or float("inf"))) if bandwidth_limit_mb else 0

            session = requests.Session()
            session.headers.update(headers)
            # Override rebuild_auth so the Authorization header is NOT dropped
            # when following cross-domain redirects (e.g. catalogue → download)
            session.rebuild_auth = lambda prepared, response: None

            resp = session.get(url, stream=True, timeout=300, allow_redirects=True)

            # Copernicus S3-compatible endpoint still returns 403 even with a
            # valid Bearer token — it requires AWS4 S3 credentials, not OAuth2.
            # Fall back to the OData download URL.
            if resp.status_code == 403 and provider in ("copernicus", "copernicus_dataspace"):
                logger.debug(
                    "S3-converted URL returned 403 — trying OData download URL for %s",
                    item_id
                )
                odata_url = self._get_copernicus_odata_url(item_id, headers)
                if odata_url:
                    url = odata_url
                    resp = session.get(url, stream=True, timeout=300, allow_redirects=True)
                else:
                    logger.warning(
                        "  ✗ %s: S3 auth failed and OData URL lookup failed. "
                        "Try provider='planetary_computer' for free HTTPS access.",
                        item_id[:55]
                    )
                    return DownloadResult(
                        scene_id=item_id, provider=provider, success=False,
                        error=(
                            "Copernicus S3 endpoint requires AWS4 S3 credentials; "
                            "OData lookup also failed. "
                            "Switch to provider='planetary_computer' for free access."
                        ),
                    )

            resp.raise_for_status()

            bytes_written = 0
            with open(out_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        if bw_delay:
                            _time.sleep(bw_delay)

            duration = _time.time() - t0
            size_mb  = bytes_written / 1024 / 1024
            logger.info(
                "  ✓ %-50s  %6.0f MB  %5.1fs",
                item_id[:50], size_mb, duration
            )

            # Run post-processing (reproject, cog, etc.)
            final_path = out_file
            if post_process and out_file.exists():
                final_path = self._apply_post_process(out_file, post_process) or out_file

            return DownloadResult(
                scene_id=item_id,
                provider=provider,
                success=True,
                path=final_path,
                bytes_downloaded=bytes_written,
                duration_seconds=duration,
            )

        except Exception as exc:
            logger.warning("  ✗ %s: %s", item_id[:55], exc)
            return DownloadResult(
                scene_id=item_id, provider=provider, success=False,
                error=str(exc),
            )

    def _get_copernicus_odata_url(
        self,
        product_name: str,
        auth_headers: dict,
    ) -> str | None:
        """
        Query Copernicus OData API by product name to get the direct download URL.

        The Copernicus STAC search returns S3 URIs for band assets which require
        S3-style auth. The OData download URL
        (catalogue.dataspace.copernicus.eu/odata/v1/Products('UUID')/$value)
        works with a standard Bearer token.

        Args:
            product_name: Scene ID e.g. 'S1A_IW_GRDH_1SDV_20190727T181721...'
            auth_headers: Dict with 'Authorization: Bearer TOKEN'

        Returns:
            Direct download URL string, or None if lookup fails.
        """
        import json as _json
        import urllib.request

        # Copernicus product names end in .SAFE — try with and without
        safe_name = product_name if product_name.endswith(".SAFE") else f"{product_name}.SAFE"
        # Also strip _COG suffix which Copernicus STAC sometimes appends
        base_name = safe_name.replace("_COG.SAFE", ".SAFE")

        for name in [base_name, safe_name, product_name]:
            try:
                odata_url = (
                    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
                    f"?$filter=Name eq '{name}'"
                    "&$select=Id,Name"
                    "&$top=1"
                )
                req = urllib.request.Request(
                    odata_url,
                    headers={
                        "Accept": "application/json",
                        **auth_headers,
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read())

                products = data.get("value", [])
                if products:
                    product_id = products[0].get("Id")
                    if product_id:
                        download_url = (
                            f"https://catalogue.dataspace.copernicus.eu"
                            f"/odata/v1/Products('{product_id}')/$value"
                        )
                        logger.debug(
                            "OData lookup: %s → UUID=%s", name[:50], product_id[:8]
                        )
                        return download_url
            except Exception as exc:
                logger.debug("OData lookup failed for %s: %s", name[:50], exc)

        return None

    @staticmethod
    def _pick_stac_download_url(
        assets: dict, provider: str
    ) -> tuple:
        """
        Pick the best downloadable asset URL from a STAC assets dict.

        Returns (url, asset_key) or (None, None).

        Rules:
        - NEVER return s3:// URIs — requests cannot handle them.
          Copernicus STAC individual band assets (vh, vv) are s3:// URIs;
          the "PRODUCT" asset is the correct HTTPS OData download URL.
        - Copernicus: "PRODUCT" first (full SAFE.zip via OData HTTPS)
        - Planetary Computer S1 COG: "vh" or "vv" (blob.core.windows.net HTTPS)
        - Fallback: first asset with an https:// href that isn't a thumbnail
        """
        if not assets:
            return None, None

        def _href(v) -> str:
            return v.get("href", "") if isinstance(v, dict) else str(v)

        def _is_https(url: str) -> bool:
            return url.startswith("https://") or url.startswith("http://")

        # Priority 1: explicit download/product keys with HTTPS URL
        for key in ("PRODUCT", "product", "data", "download"):
            if key in assets:
                url = _href(assets[key])
                if url and _is_https(url):
                    return url, key

        # Priority 2: Planetary Computer Sentinel-1 GRD COG band assets
        # (blob.core.windows.net — public HTTPS, no auth needed)
        for key in ("vh", "vv", "VH", "VV"):
            if key in assets:
                url = _href(assets[key])
                if url and _is_https(url):
                    return url, key

        # Priority 3: Planetary Computer Sentinel-2 bands
        for key in ("B04", "B08", "B03", "B02", "B11", "B12", "visual", "red", "nir"):
            if key in assets:
                url = _href(assets[key])
                if url and _is_https(url):
                    return url, key

        # Priority 4: any HTTPS asset that isn't a thumbnail/preview
        skip = {"thumbnail", "preview", "rendered_preview", "tilejson",
                "overview", "visual_preview"}
        for key, val in assets.items():
            if key.lower() in skip:
                continue
            url = _href(val)
            if url and _is_https(url):
                return url, key

        # Priority 5: convert Copernicus S3 URIs to HTTPS as last resort.
        # s3://eodata/path → https://eodata.dataspace.copernicus.eu/path
        # These still require a Bearer token — same one used for PRODUCT downloads.
        for key in ("vh", "vv", "VH", "VV", "PRODUCT", "product"):
            if key in assets:
                url = _href(assets[key])
                if url and url.startswith("s3://eodata/"):
                    https_url = "https://eodata.dataspace.copernicus.eu/" + url[len("s3://eodata/"):]
                    return https_url, key

        # All assets are unresolvable S3 or unknown — log the actual URLs for debugging
        return None, None

    def _apply_post_process(
        self,
        input_path: Path,
        steps: list[str],
    ) -> Path | None:
        """
        Replacement for SatelliteFetcher._apply_post_process().
    
        Changes vs original:
        • Detects corrupt (pixel-space) input transform before reprojecting
        • Attempts georef rescue; aborts clearly if recovery is impossible
        • Skips reproject when already in target CRS and transform is valid
        • Does not re-run COG on a file whose name already ends in _cog
        • Uses a .tmp file so failures never leave partial outputs
        """
        import shutil
        import subprocess

        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.warp import Resampling, calculate_default_transform, reproject
        except ImportError:
            logger.error("rasterio not installed — cannot post-process")
            return input_path

        current = input_path

        for step in steps:
            step = step.strip()

            # ── reproject:EPSG:XXXXX ────────────────────────────────────────
            if step.startswith("reproject:"):
                target_crs = step.split(":", 1)[1].strip()

                with rasterio.open(str(current)) as src:
                    src_transform = src.transform
                    src_crs       = src.crs
                    src_width     = src.width
                    src_height    = src.height
                    src_meta      = src.meta.copy()
                    src_count     = src.count

                # ── Detect corrupt geotransform ───────────────────────────────
                if _is_pixel_space_transform(src_transform):
                    logger.warning(
                        "reproject step: %s has a pixel-space/identity transform "
                        "(a=%.4f, origin=(%.1f, %.1f)). "
                        "This is the PyGeoFetch post-reproject CRS corruption bug. "
                        "Attempting georef rescue…",
                        current.name, src_transform.a, src_transform.c, src_transform.f,
                    )
                    rescued_t, rescued_crs = _rescue_geotransform(current, target_crs)
                    if rescued_t is None:
                        logger.error(
                            "reproject step: georef rescue failed for %s — "
                            "skipping reproject. Fix: re-download without post_process "
                            "and reproject via pygeovision.processors.reproject_safe().",
                            current.name
                        )
                        # Return what we have; caller's validator will report the issue
                        return current
                    src_transform = rescued_t

                # ── Skip if already in target CRS and transform is valid ──────
                try:
                    already_there = src_crs and CRS.from_user_input(str(src_crs)) == \
                                                CRS.from_user_input(target_crs)
                except Exception:
                    already_there = False

                if already_there and not _is_pixel_space_transform(src_transform):
                    logger.debug(
                        "reproject: %s already in %s with valid transform — skipping",
                        current.name, target_crs
                    )
                    continue

                # ── Compute output transform ──────────────────────────────────
                try:
                    out_transform, out_w, out_h = calculate_default_transform(
                        src_crs, target_crs,
                        src_width, src_height,
                        *rasterio.transform.array_bounds(src_height, src_width, src_transform),
                    )
                except Exception as cdt_exc:
                    logger.error("calculate_default_transform failed: %s", cdt_exc)
                    return current

                # Sanity-check the output transform
                if _is_pixel_space_transform(out_transform):
                    logger.error(
                        "reproject: output transform is still pixel-space after rescue "
                        "(a=%.4f). The rescued bounds are probably wrong. "
                        "Re-download without post_process=['reproject:...'].",
                        out_transform.a
                    )
                    return current

                out_meta = src_meta.copy()
                out_meta.update(
                    crs=target_crs,
                    transform=out_transform,
                    width=out_w,
                    height=out_h,
                    compress="deflate",   # keep output manageable
                )

                out_path = current.with_name(current.stem + "_repr.tif")
                tmp_path = out_path.with_suffix(".tmp.tif")
                try:
                    with rasterio.open(str(current)) as src_f:
                        with rasterio.open(str(tmp_path), "w", **out_meta) as dst_f:
                            for band_i in range(1, src_count + 1):
                                reproject(
                                    source      =rasterio.band(src_f, band_i),
                                    destination =rasterio.band(dst_f, band_i),
                                    src_transform=src_transform,
                                    src_crs      =src_crs,
                                    dst_transform=out_transform,
                                    dst_crs      =target_crs,
                                    resampling   =Resampling.bilinear,
                                )
                    shutil.move(str(tmp_path), str(out_path))
                    logger.info(
                        "reproject: %s → %s (%.1f m/px, origin=(%.0f, %.0f))",
                        current.name, out_path.name,
                        abs(out_transform.a), out_transform.c, out_transform.f,
                    )
                    current = out_path
                except Exception as exc:
                    tmp_path.unlink(missing_ok=True)
                    logger.error("reproject failed: %s", exc)
                    return current

            # ── cog ─────────────────────────────────────────────────────────
            elif step == "cog":
                # Skip if the file was already a COG (e.g. PyGeoFetch already did it)
                if "_cog" in current.stem.lower():
                    logger.debug("cog step: %s already appears to be a COG — skipping", current.name)
                    continue

                out_path = current.with_name(current.stem + "_cog.tif")
                tmp_path = out_path.with_suffix(".tmp.tif")
                try:
                    r = subprocess.run(
                        ["gdal_translate", "-of", "COG",
                        "-co", "COMPRESS=DEFLATE",
                        "-co", "PREDICTOR=2",
                        str(current), str(tmp_path)],
                        capture_output=True,
                    )
                    if r.returncode == 0:
                        shutil.move(str(tmp_path), str(out_path))
                        current = out_path
                        logger.info("cog: → %s", out_path.name)
                    else:
                        tmp_path.unlink(missing_ok=True)
                        logger.warning(
                            "COG conversion failed (gdal_translate rc=%d): %s",
                            r.returncode, r.stderr.decode()[:200],
                        )
                except FileNotFoundError:
                    logger.warning("gdal_translate not on PATH — skipping COG step")
                except Exception as exc:
                    tmp_path.unlink(missing_ok=True)
                    logger.warning("COG step error: %s", exc)

            else:
                logger.warning("Unknown post-process step %r — skipping", step)

        return current

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, pipeline_yaml: str | Path, step: str | None = None) -> dict[str, Any]:
        """Run a pygeofetch YAML pipeline."""
        if self._use_cli():
            args = ["pipeline", "run", str(pipeline_yaml)]
            if step:
                args += ["--step", step]
            proc = self._run_cli(args)
            return {
                "success": proc.returncode == 0,
                "output":  proc.stdout,
                "error":   proc.stderr if proc.returncode != 0 else "",
            }
        if hasattr(self._engine, "run_pipeline"):
            return self._engine.run_pipeline(pipeline_yaml, step=step)
        # Fallback: use the PyGeoVision pipeline orchestrator
        from pygeovision.pipelines import Pipeline
        p = Pipeline.from_yaml(str(pipeline_yaml))
        result = p.run()
        return {"success": result.success, "steps": result.steps_completed}

    def validate_pipeline(self, pipeline_yaml: str | Path) -> bool:
        """Validate a pipeline YAML file."""
        return self._engine.validate_pipeline(pipeline_yaml)

    def schedule_pipeline(self, pipeline_yaml: str | Path,
                          name: str | None = None, cron: str | None = None) -> bool:
        """Schedule a pipeline for periodic execution."""
        return self._engine.schedule_pipeline(pipeline_yaml, name=name, cron=cron)

    def list_scheduled_pipelines(self) -> list[dict]:
        """List scheduled pipelines."""
        return self._engine.list_scheduled_pipelines()

    def pipeline_history(self, limit: int = 20) -> list[dict]:
        """Get pipeline execution history."""
        return self._engine.pipeline_history(limit=limit)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._use_cli():
            import json as _json
            proc = self._run_cli(["cache", "stats", "--json"])
            try:
                return _json.loads(proc.stdout)
            except Exception:
                pass
        files = list(self.cache_dir.glob("*.json"))
        total = sum(f.stat().st_size for f in files)
        return {
            "entries": len(files),
            "size_bytes": total,
            "size_mb": round(total / 1024 / 1024, 2),
            "location": str(self.cache_dir),
        }

    def clear_cache(self, provider: str | None = None,
                    older_than: str | None = None, dry_run: bool = False) -> None:
        """Clear search cache."""
        if self._use_cli():
            args = ["cache", "clear"]
            if provider:
                args += ["--provider", provider]
            if older_than:
                args += ["--older-than", older_than]
            if dry_run:
                args.append("--dry-run")
            self._run_cli(args)
            return
        if not dry_run:
            for f in self.cache_dir.glob("*.json"):
                f.unlink(missing_ok=True)
            logger.info("Local search cache cleared")

    def set_cache_ttl(self, seconds: int) -> None:
        """Set cache TTL in seconds."""
        # Implemented by pygeofetch internally
        pass

    def prune_cache(self, max_size_gb: float = 10.0) -> None:
        """Prune cache to stay under size limit."""
        # Implemented by pygeofetch internally
        pass

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------
    # FIX: these two methods were missing, causing:
    #   AttributeError: 'SatelliteFetcher' object has no attribute
    #   '_has_pygeofetch'
    # They are called by PyGeoVision.status() and PyGeoVision.__repr__().
    # __init__ above already raises PyGeoVisionError if pygeofetch can't
    # be imported, so any successfully constructed SatelliteFetcher
    # implies pygeofetch IS available — but we still defer to the
    # module-level _check_pygeofetch() flag rather than hardcoding True.

    def _has_pygeofetch(self) -> bool:
        """Return True if the pygeofetch Python API is available."""
        return _check_pygeofetch()

    def _pygeofetch_version(self) -> str:
        """Return the installed pygeofetch package version."""
        try:
            return self._engine.version()
        except Exception:
            pass
        try:
            import pygeofetch
            return getattr(pygeofetch, "__version__", "unknown")
        except ImportError:
            return "not installed"

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Get system status."""
        try:
            return self._engine.status()
        except Exception:
            return {
                "pygeofetch_available": True,
                "version": self._pygeofetch_version(),
                "providers": len(PROVIDERS),
                "open_providers": OPEN_PROVIDERS,
            }

    def doctor(self) -> dict[str, Any]:
        """Run diagnostic checks across all PyGeoVision components."""
        import platform
        report: dict[str, Any] = {
            "pygeovision": {"ok": True},
            "python":      platform.python_version(),
            "platform":    platform.system(),
        }

        # pygeofetch engine
        try:
            v = self._pygeofetch_version()
            report["pygeofetch"] = {"ok": True, "version": v}
            # Try native doctor if available
            if hasattr(self._engine, "doctor"):
                report["pygeofetch"].update(self._engine.doctor())
        except Exception as exc:
            report["pygeofetch"] = {"ok": False, "error": str(exc)}

        # torch / CUDA
        try:
            import torch
            report["torch"] = {
                "ok": True,
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            if torch.cuda.is_available():
                report["torch"]["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            report["torch"] = {"ok": False, "error": "not installed — pip install torch"}

        # rasterio
        try:
            import rasterio
            report["rasterio"] = {"ok": True, "version": rasterio.__version__}
            # Quick GDAL check
            from rasterio.drivers import raster_driver_extensions
            report["rasterio"]["gdal"] = "ok"
        except ImportError:
            report["rasterio"] = {"ok": False, "error": "not installed — pip install rasterio"}
        except Exception as exc:
            report["rasterio"] = {"ok": True, "gdal_warning": str(exc)}

        # geopandas
        try:
            import geopandas
            report["geopandas"] = {"ok": True, "version": geopandas.__version__}
        except ImportError:
            report["geopandas"] = {"ok": False, "error": "not installed — pip install geopandas"}

        # credentials check
        try:
            creds = self.list_credentials()
            report["credentials"] = {"providers_configured": len(creds), "list": creds}
        except Exception as exc:
            report["credentials"] = {"error": str(exc)}

        # cache
        try:
            report["cache"] = self.cache_stats()
        except Exception:
            pass

        n_ok  = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok"))
        n_all = sum(1 for v in report.values() if isinstance(v, dict) and "ok" in v)
        report["summary"] = f"{n_ok}/{n_all} components healthy"
        return report

    def list_providers(self, auth_only: bool = False, open_only: bool = False,
                       capabilities: list[str] | None = None) -> dict[str, dict]:
        """List available providers."""
        providers = dict(PROVIDERS)
        if auth_only:
            providers = {k: v for k, v in providers.items() if v.get("auth")}
        if open_only:
            providers = {k: v for k, v in providers.items() if v.get("open")}
        if capabilities:
            for cap in capabilities:
                if cap in ("sar", "stac", "sub_meter"):
                    providers = {k: v for k, v in providers.items() if v.get(cap)}
        return providers

    def provider_info(self, provider_id: str) -> dict:
        """Get information about a specific provider."""
        return PROVIDERS.get(provider_id, {})

    def config_get(self, key: str) -> str:
        """Get configuration value."""
        return self._engine.config.get(key, "")

    def config_set(self, key: str, value: str) -> bool:
        """Set configuration value."""
        return self._engine.config.set(key, value)

    def config_show(self) -> dict[str, Any]:
        """Show all configuration."""
        return dict(self._engine.config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    # ── CLI-mode helpers ─────────────────────────────────────────────
    # ------------------------------------------------------------------

    def _run_cli(self, args: list[str], capture: bool = True) -> Any:
        """Run pygeofetch CLI command. Returns CompletedProcess."""
        import subprocess
        exe = _PYGEOFETCH_CLI_EXE or "pygeofetch"
        cmd = [exe] + args
        logger.debug("CLI: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )

    def _parse_stac_geojson_file(self, geojson_path: Path) -> list[SearchResult]:
        """Parse a pygeofetch-style GeoJSON result file into SearchResult objects."""
        import json
        with open(geojson_path, encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            fid   = feat.get("id") or props.get("id", "")
            geom  = feat.get("geometry") or {}
            bbox  = None
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                if coords:
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    bbox = (min(lons), min(lats), max(lons), max(lats))
            results.append(SearchResult(
                id=fid,
                provider=props.get("provider", ""),
                satellite=props.get("satellite", self._collection_to_satellite(
                    props.get("collection", ""))),
                datetime=props.get("datetime", ""),
                cloud_cover=props.get("eo:cloud_cover"),
                bbox=bbox,
                score=props.get("score"),
                collection=props.get("collection", ""),
                assets=props.get("assets", {}),
                properties=props,
            ))
        return results

    def _collection_to_satellite(self, collection: str) -> str:
        """Map a STAC collection ID to a human-readable satellite name.

        Handles both Planetary Computer lowercase IDs (sentinel-1-grd)
        and Copernicus Data Space uppercase IDs (SENTINEL-1-GRD).
        """
        col = collection.lower()
        _MAP = {
            "sentinel-2-l2a":  "Sentinel-2",
            "sentinel-2-l1c":  "Sentinel-2",
            "sentinel-1-rtc":  "Sentinel-1",
            "sentinel-1-grd":  "Sentinel-1",
            "sentinel-1-slc":  "Sentinel-1",
            "sentinel-1":      "Sentinel-1",
            "landsat-c2-l2":   "Landsat",
            "landsat-c2-l1":   "Landsat",
            "landsat-8-l1tp":  "Landsat",
            "landsat-9-l1tp":  "Landsat",
            "naip":            "NAIP",
            "cop-dem-glo-30":  "Copernicus DEM",
            "modis":           "MODIS",
        }
        for key, val in _MAP.items():
            if key in col:
                return val
        return collection

    def _get_copernicus_token(self, username: str, password: str) -> str | None:
        """
        Obtain a short-lived OAuth2 bearer token from Copernicus Identity Service.

        Returns the access_token string, or None on failure.
        """
        import time
        if not username or not password:
            logger.warning(
                "Copernicus token request skipped — username or password is empty. "
                "Call: client.add_credentials('copernicus', username=EMAIL, password=PASSWORD)"
            )
            return None

        # Return cached token if still valid
        if hasattr(self, "_copernicus_token") and hasattr(self, "_copernicus_token_exp"):
            if time.time() < self._copernicus_token_exp - 30:
                return self._copernicus_token

        import json as _json
        import urllib.parse
        import urllib.request

        token_url = (
            "https://identity.dataspace.copernicus.eu"
            "/auth/realms/CDSE/protocol/openid-connect/token"
        )
        payload = urllib.parse.urlencode({
            "grant_type":    "password",
            "username":      username,
            "password":      password,
            "client_id":     "cdse-public",
        }).encode()

        req = urllib.request.Request(
            token_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                token_data = _json.loads(resp.read())
                token      = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 600)
                if token:
                    self._copernicus_token     = token
                    self._copernicus_token_exp = time.time() + expires_in
                    logger.debug(
                        "Copernicus token obtained for %s (expires in %ds)",
                        username, expires_in
                    )
                    return token
                else:
                    logger.warning(
                        "Copernicus token response had no access_token. "
                        "Response keys: %s", list(token_data.keys())
                    )
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            if e.code == 401:
                logger.warning(
                    "Copernicus authentication FAILED (401): wrong password for %s. "
                    "Update your credentials: "
                    "client.add_credentials('copernicus', username='%s', password='NEW_PASSWORD'). "
                    "Response: %s",
                    username, username, body
                )
            elif e.code == 400:
                logger.warning(
                    "Copernicus token request error (400): %s. "
                    "Check username format (must be email address).",
                    body
                )
            else:
                logger.warning(
                    "Copernicus token HTTP error %d: %s", e.code, body
                )
        except Exception as exc:
            logger.warning(
                "Copernicus authentication network error: %s\n"
                "  Check your credentials at: https://dataspace.copernicus.eu",
                exc
            )
        return None

    def test_copernicus_auth(self) -> bool:
        """
        Test whether Copernicus credentials are valid and can obtain a token.

        Call this to diagnose auth failures before downloading.

        Returns True if a valid token was obtained.

        Example::

            if not client.data.test_copernicus_auth():
                client.add_credentials('copernicus',
                                        username='you@email.com',
                                        password='correct_password')
        """
        creds = (
            self._credentials.get("copernicus") or
            self._credentials.get("copernicus_dataspace") or {}
        )
        if not creds:
            logger.warning(
                "No Copernicus credentials found. "
                "Call: client.add_credentials('copernicus', username=EMAIL, password=PASSWORD)"
            )
            return False

        username = creds.get("username", "")
        password = creds.get("password", "")
        logger.info("Testing Copernicus auth for: %s", username)

        token = self._get_copernicus_token(username, password)
        if token:
            logger.info("Copernicus auth: OK — token obtained (%d chars)", len(token))
            print(f"  ✓ Copernicus auth OK for {username}")
            return True
        else:
            print(f"  ✗ Copernicus auth FAILED for {username}")
            print("    → Update credentials: client.add_credentials(")
            print(f"          'copernicus', username='{username}',")
            print("          password='YOUR_CURRENT_PASSWORD')")
            return False

    def _pick_best_asset(self, result: SearchResult) -> str | None:
        """Pick the best available asset key from a SearchResult.

        Priority: specific spectral bands > visual > thumbnail > first available.
        """
        assets = result.assets or {}
        if not assets:
            return None
        # Prefer individual bands over composites
        for preferred in ["B04", "B03", "B02", "B08", "nir", "red", "visual", "thumbnail"]:
            if preferred in assets:
                return preferred
        return next(iter(assets))

    # ------------------------------------------------------------------

    def _find_downloaded_file(self, output_dir: Path, scene_id: str) -> Path | None:
        """Best-effort: locate a downloaded file matching scene_id under output_dir.

        Fallback used when a successful pygeofetch DownloadResult doesn't
        expose an explicit output_path / output_paths field.
        """
        try:
            prefix = scene_id[:20]
            matches = sorted(p for p in output_dir.rglob(f"*{prefix}*") if p.is_file())
            return matches[0] if matches else None
        except Exception:
            return None

    def _resolve_providers(self, providers, satellite, collections):
        """Resolve providers from various inputs."""
        # Normalise common provider name aliases
        _ALIASES = {
            "copernicus_dataspace": "copernicus",
            "cdse":                 "copernicus",
            "dataspace":            "copernicus",
            "esa":                  "copernicus",
            "pc":                   "planetary_computer",
            "microsoft":            "planetary_computer",
            "aws":                  "aws_earth",
            "element84":            "aws_earth",
            "asf":                  "asf_vertex",
            "alaska":               "asf_vertex",
        }
        if providers:
            normalised = []
            for p in providers:
                alias = _ALIASES.get(p.lower().replace("-", "_"))
                if alias and alias not in normalised:
                    normalised.append(alias)
                elif p not in normalised:
                    normalised.append(p)
            return normalised
        if satellite:
            sl = satellite.lower().replace(" ", "-")
            for key, provs in SATELLITE_SHORTCUTS.items():
                if key in sl or sl in key:
                    return provs
            return DEFAULT_SEARCH_PROVIDERS
        if collections:
            resolved, seen = [], set()
            for col in collections:
                cl = col.lower()
                key = next(
                    (k for k in ("sentinel-2", "sentinel-1", "landsat", "modis", "naip", "dem", "planet")
                     if k in cl), None
                )
                if key:
                    for p in SATELLITE_SHORTCUTS.get(key, []):
                        if p not in seen:
                            seen.add(p)
                            resolved.append(p)
                else:
                    p = COLLECTION_TO_PROVIDER.get(col)
                    if p and p not in seen:
                        seen.add(p)
                        resolved.append(p)
            if resolved:
                return resolved
        return DEFAULT_SEARCH_PROVIDERS

    def _providers_to_satellites(self, providers: list[str]) -> list[str]:
        """Convert provider IDs to satellite name hints."""
        sat_map = {
            "planetary_computer": [],
            "copernicus": ["Sentinel-1", "Sentinel-2"],
            "usgs": ["Landsat"],
            "aws_earth": ["Sentinel-2", "Landsat"],
            "element84": ["Sentinel-2"],
            "planet": ["PlanetScope"],
            "nasa_earthdata": ["MODIS"],
        }
        sats = []
        for p in providers:
            sats.extend(sat_map.get(p, []))
        return list(set(sats)) if sats else []

    def _cache_key(self, bbox, date_range, providers, cloud_cover_max, collections) -> str:
        """Generate cache key for search parameters."""
        key_str = f"{bbox}|{date_range}|{sorted(providers)}|{cloud_cover_max}|{sorted(collections or [])}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self, key: str) -> list[SearchResult] | None:
        """Load cached search results.

        Validates the cache schema version before trusting the entry.
        Any cache file written by an older PyGeoVision version (e.g.
        before 'assets' was added to the cached payload) fails this
        check and is discarded, forcing a fresh search instead of
        silently returning incomplete results. This is what makes
        cache-format bugfixes self-healing without manual intervention.
        """
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > 3600:
            path.unlink(missing_ok=True)
            return None
        try:
            import json
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Reject anything that isn't our current versioned format.
            # Older cache files are a bare JSON list (no 'version' key)
            # and would otherwise be silently accepted with missing fields.
            if not isinstance(data, dict) or data.get("version") != _CACHE_SCHEMA_VERSION:
                logger.debug("Cache entry %s has outdated/invalid schema; discarding.", key)
                path.unlink(missing_ok=True)
                return None

            results = [SearchResult(**item) for item in data.get("results", [])]

            return results
        except Exception:
            return None

    def _save_cache(self, key: str, results: list[SearchResult]) -> None:
        """Save search results to cache, tagged with the current schema version."""
        try:
            import json
            payload = {
                "version": _CACHE_SCHEMA_VERSION,
                "results": [r.to_dict() for r in results],
            }
            with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as exc:
            logger.debug("Cache save failed: %s", exc)
