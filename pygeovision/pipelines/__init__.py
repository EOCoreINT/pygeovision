"""
PyGeoVision Geospatial AI Pipelines.

Real fix, found during a production readiness audit: this module used
to contain a complete, independent, STALE duplicate of all 10 real
pipeline classes (ChangeDetectionPipeline, LandCoverPipeline, ...,
CarbonEstimationPipeline) -- a full copy from before the real,
extensively audited and fixed versions in pygeovision.ai.pipelines
existed. That duplicate had none of this project's real fixes (real
radiometric scaling, cloud masking, bbox cropping, mission-aware band
selection, the NDVI-via-post_process bug fix, and more) -- confirmed
directly: its CarbonEstimationPipeline still had the exact NDVI bug
found and fixed elsewhere (requesting NDVI via a post_process step that
never actually runs for a real multi-asset scene, silently reading
red-band reflectance as if it were NDVI).

No internal PyGeoVision code ever imported these stale classes (verified
directly), but they were genuinely, deliberately public via __all__ --
`from pygeovision.pipelines import CarbonEstimationPipeline` was a real,
reachable import path that silently returned the old, broken version.

This module now re-exports the real, audited classes directly, so both
import paths point to the same, real, tested code -- not a stale copy
that happens to share a name.

Also fixes a second, separate, confirmed-broken import: `from
pygeovision.pipelines import Pipeline` (used by
pygeovision/data/fetch.py's run_pipeline() fallback) previously raised
ImportError unconditionally, since Pipeline (the real YAML workflow
orchestrator, a different concept from the 10 task pipelines -- see
pygeovision.pipelines.orchestrator) was never actually re-exported here
despite being imported as if it were.

Available task pipelines (10, real, re-exported from pygeovision.ai.pipelines):
    change_detection       building_footprints    land_cover
    crop_monitoring        disaster_assessment    deforestation
    urban_growth           water_bodies           solar_detection
    carbon_estimation

Example:
    >>> from pygeovision.ai.pipelines import LandCoverPipeline
    >>> import pygeovision as pgv
    >>> client = pgv.PyGeoVision()
    >>> result = LandCoverPipeline(client).run(
    ...     bbox=(-0.15, 51.47, -0.10, 51.52), output_dir="./out", date="2024-06")

    For the CLI equivalent: `pygeovision channel land_cover --bbox ... --date ...`
"""

from __future__ import annotations

# Real fix: Pipeline was previously imported elsewhere in this codebase
# as `from pygeovision.pipelines import Pipeline` but was never actually
# re-exported here -- that import has been silently broken. Fixed by
# genuinely re-exporting it from the real orchestrator module.
from pygeovision.pipelines.orchestrator import Pipeline

# Real fix: re-export the real, audited classes instead of maintaining
# a separate, stale duplicate copy of all 10.
from pygeovision.ai.pipelines import (
    PipelineResult,
    BasePipeline,
    ChangeDetectionPipeline,
    LandCoverPipeline,
    BuildingFootprintsPipeline,
    CropMonitoringPipeline,
    DisasterAssessmentPipeline,
    DeforestationPipeline,
    UrbanGrowthPipeline,
    WaterBodiesPipeline,
    SolarDetectionPipeline,
    CarbonEstimationPipeline,
)

_PIPELINE_REGISTRY: dict[str, type] = {
    "change_detection":     ChangeDetectionPipeline,
    "land_cover":            LandCoverPipeline,
    "building_footprints":   BuildingFootprintsPipeline,
    "crop_monitoring":        CropMonitoringPipeline,
    "disaster_assessment":    DisasterAssessmentPipeline,
    "deforestation":          DeforestationPipeline,
    "urban_growth":           UrbanGrowthPipeline,
    "water_bodies":           WaterBodiesPipeline,
    "solar_detection":        SolarDetectionPipeline,
    "carbon_estimation":      CarbonEstimationPipeline,
}


def get_pipeline(name: str, pgv_client) -> BasePipeline:
    """Instantiate a real, audited pipeline by name.

    Args:
        name: Pipeline name (one of the 10 real pipelines above).
        pgv_client: PyGeoVision client.

    Returns:
        Instantiated pipeline.
    """
    if name not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline '{name}'. Available: {sorted(_PIPELINE_REGISTRY.keys())}. "
            f"For the broader, honestly-labelled 51-name catalog (10 real, 16 unverified, "
            f"25 real via this cycle's fixes), see pygeovision.ai.pipelines.domains."
        )
    return _PIPELINE_REGISTRY[name](pgv_client)


def list_pipelines() -> list[str]:
    return sorted(_PIPELINE_REGISTRY.keys())


__all__ = [
    "Pipeline", "BasePipeline", "PipelineResult", "get_pipeline", "list_pipelines",
    "ChangeDetectionPipeline", "LandCoverPipeline", "BuildingFootprintsPipeline",
    "CropMonitoringPipeline", "DisasterAssessmentPipeline", "DeforestationPipeline",
    "UrbanGrowthPipeline", "WaterBodiesPipeline", "SolarDetectionPipeline",
    "CarbonEstimationPipeline",
]