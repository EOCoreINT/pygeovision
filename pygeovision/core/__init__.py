"""PyGeoVision Core — configuration, exceptions, and engine."""

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


def _get_engine():
    """Lazy import PyGeoVisionEngine (requires pygeofetch dependencies)."""
    from pygeovision.core.engine import PyGeoVisionEngine
    return PyGeoVisionEngine
