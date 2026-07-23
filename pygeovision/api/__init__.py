"""
PyGeoVision REST-style Python API client.

Thin wrapper over PyGeoVision for use in web applications,
notebooks, and microservices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class PyGeoVisionClient:
    """High-level API client wrapping PyGeoVision.

    Provides a clean, discoverable interface for notebooks and web apps.

    Args:
        api_key: Optional API key for authenticated PyGeoFetch providers.
        config_path: Path to PyGeoVision or PyGeoFetch config YAML.

    Example:
        >>> from pygeovision.api import PyGeoVisionClient
        >>> client = PyGeoVisionClient()
        >>>
        >>> # Add PyGeoFetch credentials
        >>> client.add_credentials("planet", api_key="PL_KEY")
        >>> client.add_credentials("usgs", username="user", password="pass")
        >>>
        >>> # Search via PyGeoFetch
        >>> results = client.search(
        ...     bbox=(-74.1, 40.6, -73.7, 40.9),
        ...     date_range=("2024-01-01", "2024-06-01"),
        ...     cloud_cover_max=15,
        ... )
        >>>
        >>> # Download and process
        >>> downloads = client.download(results[:3], output_dir="./data/",
        ...     post_process=["unzip", "reproject:EPSG:4326", "cog"])
        >>>
        >>> # Native AI inference
        >>> masks = client._pgv.segmentation.buildings(downloads[0].path)
        >>>
        >>> # End-to-end pipeline
        >>> result = client.run_pipeline("building_footprints",
        ...     bbox=(-74.1, 40.6, -73.7, 40.9), date="2024-06")
    """

    def __init__(
        self,
        api_key: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        from pygeovision import PyGeoVision
        self._pgv = PyGeoVision(config_path=config_path)
        self._ai_engine: Any | None = None
        if api_key:
            # Store as generic credential
            self._pgv.data._credentials["__api_key__"] = {"api_key": api_key}

    def add_credentials(self, provider: str, **kwargs: Any) -> PyGeoVisionClient:
        """Add PyGeoFetch provider credentials (stored in system keyring).

        Args:
            provider: PyGeoFetch provider ID (22+ supported).
            **kwargs: username, password, api_key, client_id, client_secret.

        Returns:
            Self for chaining.
        """
        self._pgv.add_credentials(provider, **kwargs)
        return self

    def search(self, bbox: tuple, date_range: tuple,
               collections: list[str] | None = None,
               providers: list[str] | None = None,
               satellite: str | None = None,
               cloud_cover_max: float = 30.0,
               **kwargs: Any) -> list[Any]:
        """Search satellite imagery via PyGeoFetch (22+ providers).

        Delegates to ``pygeofetch search run`` as primary backend.
        """
        return self._pgv.search(
            bbox=bbox, date_range=date_range, collections=collections,
            providers=providers, satellite=satellite,
            cloud_cover_max=cloud_cover_max, **kwargs)

    def download(self, items: Any, output_dir: str | Path = "./data",
                 parallel: int = 4,
                 post_process: list[str] | None = None,
                 **kwargs: Any) -> list[Any]:
        """Download satellite data via PyGeoFetch.

        Post-processing: unzip, reproject:EPSG:4326, compress:lzw,
                         ndvi, ndwi, cog, resample:10, clip:area.geojson
        """
        return self._pgv.download(
            items, output_dir=output_dir, parallel=parallel,
            post_process=post_process, **kwargs)

    def run_pipeline(self, pipeline_name: str, bbox: tuple,
                     output_dir: str | Path = "./output",
                     **kwargs: Any) -> Any:
        """Run an end-to-end geospatial AI pipeline (PyGeoFetch + native AI)."""
        return self._pgv.pipeline(pipeline_name, bbox=bbox, output_dir=output_dir, **kwargs)

    def run_yaml_pipeline(self, yaml_path: str | Path,
                          step: str | None = None) -> dict[str, Any]:
        """Run a PyGeoFetch YAML pipeline file."""
        return self._pgv.run_pipeline_yaml(yaml_path, step=step)

    def create_pipeline(self, name: str, **kwargs: Any) -> Any:
        """Create a PyGeoFetch data pipeline programmatically."""
        return self._pgv.create_pipeline(name, **kwargs)

    @property
    def ai(self) -> Any:
        """Access the native AI engine (model registry, training, inference) via client.ai."""
        from pygeovision.ai.engine import AIEngine
        if self._ai_engine is None:
            self._ai_engine = AIEngine(pgv_client=self._pgv)
        return self._ai_engine

    @property
    def data(self) -> Any:
        """Direct access to the PyGeoFetch SatelliteFetcher."""
        return self._pgv.data

    def status(self) -> dict[str, Any]:
        """Full system status: PyGeoVision + PyGeoFetch + native AI stack."""
        return self._pgv.status()

    def list_providers(self, **kwargs: Any) -> dict[str, Any]:
        """List all 22 PyGeoFetch satellite data providers."""
        return self._pgv.list_providers(**kwargs)
