"""
PyGeoVision Explainability Layer (G6) — XAI for geospatial models.
GradCAM, SHAP, attention maps, uncertainty maps.
Fully native — no external AI-platform dependency.
"""
from pygeovision.explainability.attention import AttentionMapExtractor
from pygeovision.explainability.gradcam import GradCAM, GradCAMPlusPlus
from pygeovision.explainability.shap_geo import GeospatialSHAP
from pygeovision.explainability.uncertainty import UncertaintyEstimator

__all__ = [
    "GradCAM", "GradCAMPlusPlus",
    "AttentionMapExtractor",
    "UncertaintyEstimator",
    "GeospatialSHAP",
]
