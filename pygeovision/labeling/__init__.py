"""
PyGeoVision Auto-Labeling Layer — Phase 2 Independence.

Automated label generation from 7+ geospatial sources.
Fully native — no external AI-platform dependency.
"""
from pygeovision.labeling.active import ActiveLearner
from pygeovision.labeling.buildings import GoogleBuildingsLabeler, MicrosoftBuildingsLabeler
from pygeovision.labeling.foundation import FoundationModelLabeler
from pygeovision.labeling.landcover import DynamicWorldLabeler, ESAWorldCoverLabeler
from pygeovision.labeling.osm import OSMLabeler
from pygeovision.labeling.pipeline import AutoLabelPipeline
from pygeovision.labeling.quality import LabelQualityAssessor
from pygeovision.labeling.sam_auto import SAMAutoLabeler, SamGeoLabeler

__all__ = [
    "OSMLabeler", "MicrosoftBuildingsLabeler", "GoogleBuildingsLabeler",
    "ESAWorldCoverLabeler", "DynamicWorldLabeler",
    "SAMAutoLabeler", "SamGeoLabeler", "FoundationModelLabeler",
    "ActiveLearner", "LabelQualityAssessor", "AutoLabelPipeline",
]