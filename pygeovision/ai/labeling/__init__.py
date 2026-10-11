"""
Automatic labeling system for satellite imagery.

Provides multiple labeling strategies that generate ground truth masks
from authoritative geospatial datasets, foundation models, and human review.
"""

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler
from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
from pygeovision.ai.labeling.google_buildings import GoogleBuildingsLabeler
from pygeovision.ai.labeling.microsoft_buildings import MicrosoftBuildingsLabeler
from pygeovision.ai.labeling.osm_labeler import OSMLabeler

try:
    from pygeovision.ai.labeling.sam_labeler import SAMLabeler
except ImportError:
    SAMLabeler = None  # type: ignore[assignment, misc]

try:
    from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler
except ImportError:
    FoundationModelLabeler = None  # type: ignore[assignment, misc]

try:
    from pygeovision.ai.labeling.label_studio import LabelStudioLabeler as LabelStudioIntegration
except ImportError:
    LabelStudioIntegration = None


class LabelingOrchestrator:
    """Placeholder that never worked: it ran nothing and returned ``status: "completed"``.

    It now raises instead of reporting success. To combine label sources use
    :class:`pygeovision.labeling.pipeline.AutoLabelPipeline` (bbox in, fused raster out), or call the
    labelers' ``label_tiles`` yourself and merge the rasters.
    """

    def __init__(self, data_pipeline: object, config: object = None) -> None:
        self._pygeofetch = data_pipeline
        self._config = config

    def label(self, *args: object, **kwargs: object) -> dict:
        raise NotImplementedError(
            "LabelingOrchestrator.label() was a stub that returned status='completed' without labelling "
            "anything. Use pygeovision.labeling.pipeline.AutoLabelPipeline, or a labeler's label_tiles()."
        )


__all__ = [
    "BaseLabeler",
    "LabelingResult",
    "LabelingOrchestrator",
    "OSMLabeler",
    "MicrosoftBuildingsLabeler",
    "GoogleBuildingsLabeler",
    "ESAWorldCoverLabeler",
    "DynamicWorldLabeler",
    "SAMLabeler",
    "FoundationModelLabeler",
    "LabelStudioLabeler", "LabelStudioIntegration",
]
