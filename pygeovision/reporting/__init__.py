"""Deliverables for every pipeline run: a map, analysis-ready vectors and reports.

    from pygeovision.reporting import generate_deliverables
    d = generate_deliverables(result, bbox, "out/flood", parameters={"date_before": "2024-05"})
    d.reports["pdf"], d.maps, d.vectors["gpkg"]

``PyGeoVision.pipeline()`` and ``pygeovision channel`` call this automatically
(switch off with ``report=False`` / ``--no-report`` or ``PYGEOVISION_REPORT=0``).
Install the libraries with ``pip install "pygeovision[report]"``.
"""
from __future__ import annotations

from pygeovision.reporting.build import (
    DEFAULT_VECTOR_FORMATS,
    REPORT_FORMATS,
    Deliverables,
    ReportingUnavailable,
    generate_deliverables,
    regenerate,
)
from pygeovision.reporting.specs import SPECS, ReportSpec, spec_for
from pygeovision.reporting.vectors import VECTOR_FORMATS

__all__ = [
    "generate_deliverables", "regenerate", "Deliverables", "ReportingUnavailable", "REPORT_FORMATS",
    "VECTOR_FORMATS", "DEFAULT_VECTOR_FORMATS", "SPECS", "ReportSpec", "spec_for",
]
