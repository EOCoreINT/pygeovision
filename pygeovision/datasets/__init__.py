"""
PyGeoVision Dataset Infrastructure — EarthNets Foundation (Phase 1 + 5).

500+ remote sensing datasets with unified metadata, search, analysis, and
standardised benchmark building following the EarthNets methodology.
"""
from pygeovision.datasets.analysis import DatasetAnalyzer
from pygeovision.datasets.benchmark import BenchmarkBuilder, BenchmarkConfig
from pygeovision.datasets.loader import DatasetLoader
from pygeovision.datasets.registry import DatasetInfo, DatasetRegistry, dataset_registry

__all__ = [
    "DatasetRegistry", "DatasetInfo", "dataset_registry",
    "DatasetLoader",
    "DatasetAnalyzer",
    "BenchmarkBuilder", "BenchmarkConfig",
]
