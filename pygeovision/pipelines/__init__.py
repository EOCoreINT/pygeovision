"""
PyGeoVision pipelines (re-exports).

The pipeline implementations live in :mod:`pygeovision.ai.pipelines`
(core) and :mod:`pygeovision.ai.pipelines.domains`; this package exposes
the same objects and the YAML workflow orchestrator (:class:`Pipeline`,
a different concept from the task pipelines), so there is exactly one
registry.

Example:
    >>> from pygeovision.pipelines import get_pipeline, list_pipelines
    >>> import pygeovision as pgv
    >>> client = pgv.PyGeoVision()
    >>> get_pipeline("water_bodies", client).run(
    ...     bbox=(-0.15, 51.47, -0.10, 51.52), output_dir="./out", date="2024-06")

    CLI equivalent: ``pygeovision channel water_bodies --bbox ... --date ...``
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
    BasePipeline,
    BuildingFootprintsPipeline,
    ChangeDetectionPipeline,
    CropMonitoringPipeline,
    DeforestationPipeline,
    DisasterAssessmentPipeline,
    LandCoverPipeline,
    ModelWeightsRequired,
    PipelineResult,
    SolarDetectionPipeline,
    UrbanGrowthPipeline,
    WaterBodiesPipeline,
    get_pipeline,
    list_pipelines,
    pipeline_kinds,
)

__all__ = [
    "Pipeline", "BasePipeline", "PipelineResult", "ModelWeightsRequired",
    "get_pipeline", "list_pipelines", "pipeline_kinds",
    "ChangeDetectionPipeline", "LandCoverPipeline", "BuildingFootprintsPipeline",
    "CropMonitoringPipeline", "DisasterAssessmentPipeline", "DeforestationPipeline",
    "UrbanGrowthPipeline", "WaterBodiesPipeline", "SolarDetectionPipeline",
]
