"""
PyGeoVision Data Layer — Universal Satellite Data Access.

Uses pygeofetch CLI as primary data backend (22+ providers, auth, pipeline,
cache) with pystac_client + planetary_computer as Python fallback.
"""

from pygeovision.data.fetch import DownloadResult, SatelliteFetcher, SearchResult
from pygeovision.data.pipeline import DataPipeline, PipelineStep
from pygeovision.data.providers import PROVIDERS, SATELLITE_SHORTCUTS, STAC_PROVIDERS

__all__ = [
    "SatelliteFetcher",
    "SearchResult",
    "DownloadResult",
    "PROVIDERS",
    "SATELLITE_SHORTCUTS",
    "STAC_PROVIDERS",
    "DataPipeline",
    "PipelineStep",
]
