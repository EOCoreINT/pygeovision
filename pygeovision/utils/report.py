"""
pygeovision.utils.report
========================
Structured JSON report builder.

Replaces the 15–30 line report-building + json.dump block repeated in
5 of 6 notebooks::

    report = {
        "project"   : "...",
        "generated" : datetime.datetime.utcnow().isoformat(),
        "results"   : {...},
    }
    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {OUTPUT_DIR}/report.json")

After::

    from pygeovision.utils import JsonReport

    report = (
        JsonReport("SMART-FOREST Carbon Digital Twin", OUTPUT_DIR / "report.json")
        .add("satellite",     "Sentinel-2")
        .add("model_cv_r2",  0.84)
        .add_section("results", {"mean_agb": 210.3, "total_co2e_Mt": 1.24})
        .save()          # writes file + prints summary
    )
"""
from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any


class JsonReport:
    """
    Fluent JSON report builder for pygeovision project notebooks.

    Accumulates key-value pairs and nested sections, auto-stamps
    ``generated_at``, and writes a tidy JSON file with ``save()``.

    Usage::

        from pygeovision.utils import JsonReport

        report = (
            JsonReport("Obuasi Forest Degradation", OUTPUT_DIR / "report.json")
            .add("satellite", "Sentinel-2")
            .add("baseline",  ("2019-01-01", "2019-03-31"))
            .add("current",   ("2024-01-01", "2024-03-31"))
            .add_section("results", {
                "Stable Forest"     : {"pct": 62.1, "area_km2": 59.6},
                "Mild Degradation"  : {"pct": 24.3, "area_km2": 23.3},
                "Severe Degradation": {"pct": 13.6, "area_km2": 13.1},
            })
            .save(print_summary=True)
        )
        path = report.path   # str path of written JSON file
    """

    def __init__(
        self,
        project:     str,
        output_path: str | pathlib.Path,
        study_area:  str | None = None,
    ):
        self._data: dict[str, Any] = {
            "project"      : project,
            "generated_at" : datetime.datetime.utcnow().isoformat() + "Z",
        }
        if study_area:
            self._data["study_area"] = study_area
        self._path = pathlib.Path(output_path)

    # ── Fluent builder ────────────────────────────────────────────────────────

    def add(self, key: str, value: Any) -> JsonReport:
        """Add a top-level key-value pair."""
        # Convert numpy scalars / arrays to plain Python types
        self._data[key] = _to_python(value)
        return self

    def add_section(self, key: str, section: dict[str, Any]) -> JsonReport:
        """Add a nested dict section."""
        self._data[key] = {k: _to_python(v) for k, v in section.items()}
        return self

    def add_satellites(self, *names: str) -> JsonReport:
        """Convenience: add ``"satellites"`` list."""
        return self.add("satellites", list(names))

    def add_model(self, name: str, **metrics) -> JsonReport:
        """Convenience: add ``"model"`` section with name + metrics."""
        self._data.setdefault("model", {})["name"] = name
        for k, v in metrics.items():
            self._data["model"][k] = _to_python(v)
        return self

    def add_recommendations(self, *items: str) -> JsonReport:
        """Convenience: add ``"recommendations"`` list."""
        return self.add("recommendations", list(items))

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, print_summary: bool = True) -> JsonReport:
        """
        Write the report to JSON and optionally print a summary.

        Returns:
            ``self`` — so you can chain: ``report.add(...).save()``.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

        if print_summary:
            print()
            print("═" * 55)
            print(self._data.get("project", "REPORT").upper())
            print("═" * 55)
            # Print top-level scalar fields
            for k, v in self._data.items():
                if isinstance(v, (str, int, float, bool)):
                    print(f"  {k:30s}: {v}")
            # Print sections (dicts / lists) as sub-tables
            for k, v in self._data.items():
                if isinstance(v, dict) and v:
                    print(f"\n  {k}:")
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            vals = "  ".join(f"{kk}={vv}" for kk, vv in sv.items())
                            print(f"    {sk:25s}: {vals}")
                        else:
                            print(f"    {sk:25s}: {sv}")
                elif isinstance(v, list) and v:
                    print(f"\n  {k}:")
                    for item in v:
                        print(f"    → {item}")
            print(f"\n  Saved: {self._path}")

        return self

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"JsonReport('{self._data.get('project')}', {self._path.name})"


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Convert numpy / pandas types to plain Python for JSON serialisation."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.bool_):     return bool(obj)
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, pd.Series):    return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj
