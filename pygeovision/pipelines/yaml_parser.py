"""YAML pipeline configuration parser and validator."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pygeovision.pipelines.orchestrator import Pipeline

logger = logging.getLogger(__name__)


REQUIRED_FIELDS = {"name", "steps"}
# Actions a YAML file can use. Anything else needs a Python function (Step(..., _fn=...)), which YAML
# cannot express, so it is rejected here instead of running as a silent no-op.
VALID_ACTIONS = {"search", "download", "infer", "export"}


class PipelineYAMLParser:
    """Parse and validate PyGeoVision YAML pipeline configurations.

    YAML format::

        name: building_extraction
        description: Extract building footprints from Sentinel-2
        schedule: "0 6 * * *"   # Run daily at 6am UTC
        steps:
          - name: search
            action: search
            params:
              bbox: [-74.1, 40.6, -73.7, 40.9]
              providers: [planetary_computer]
              date_range: ["2024-01-01", "2024-12-31"]
              cloud_cover_max: 15
          - name: download
            action: download
            depends_on: [search]
            params:
              output_dir: ./data/
              parallel: 4
          - name: infer
            action: infer
            depends_on: [download]
            params:
              model: unet-r50
              num_classes: 2
          - name: export
            action: export
            depends_on: [infer]
            params:
              format: geojson
              output_dir: ./results/

    Example::

        parser = PipelineYAMLParser()
        pipeline = parser.load("agriculture.yaml")
        parser.validate(pipeline)
    """

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load and parse a YAML pipeline file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        self.validate(config)
        return config

    def loads(self, yaml_str: str) -> dict[str, Any]:
        """Parse a YAML pipeline string."""
        try:
            import yaml
            config = yaml.safe_load(yaml_str)
        except ImportError:
            raise ImportError("pip install pyyaml")
        self.validate(config)
        return config

    def validate(self, config: dict) -> None:
        """Validate a pipeline configuration dict."""
        missing = REQUIRED_FIELDS - set(config.keys())
        if missing:
            raise ValueError(f"Pipeline config missing required fields: {missing}")

        steps = config.get("steps", [])
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        step_names = set()
        for step in steps:
            if "name" not in step:
                raise ValueError("Each step must have a 'name' field")
            if "action" not in step:
                raise ValueError(f"Step '{step['name']}' missing 'action' field")
            if step["action"] not in VALID_ACTIONS:
                raise ValueError(
                    f"Step '{step['name']}' has action '{step['action']}'; YAML pipelines support "
                    f"{sorted(VALID_ACTIONS)} (other behaviour needs a Python Step with a function)")
            step_names.add(step["name"])

        if len(step_names) != len(steps):
            seen: set[str] = set()
            dup = next(s["name"] for s in steps if s["name"] in seen or seen.add(s["name"]))
            raise ValueError(f"Step names must be unique; '{dup}' appears twice")

        # Check dependency references
        for step in steps:
            for dep in step.get("depends_on", []):
                if dep not in step_names:
                    raise ValueError(f"Step '{step['name']}' depends on unknown step '{dep}'")

        # A cycle (a -> b -> a) has no valid order; without this check the steps would run in an arbitrary one
        graph = {s["name"]: list(s.get("depends_on", [])) for s in steps}
        state: dict[str, int] = {}

        def visit(name: str, trail: list[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                cycle = " -> ".join([*trail[trail.index(name):], name])
                raise ValueError(f"Steps depend on each other in a circle: {cycle}")
            state[name] = 1
            for dep in graph[name]:
                visit(dep, [*trail, name])
            state[name] = 2

        for name in graph:
            visit(name, [])

    @staticmethod
    def apply_overrides(config: dict, overrides: list[str]) -> dict:
        """Apply ``step.param=value`` overrides (value parsed as JSON, else kept as a string).

        Example: ``["search.bbox=[15.0,45.1,15.1,45.2]", "infer.checkpoint=best.pth"]``.
        """
        import json
        by_name = {s["name"]: s for s in config.get("steps", [])}
        for item in overrides:
            key, sep, raw = item.partition("=")
            step_name, dot, param = key.partition(".")
            if not sep or not dot or not param:
                raise ValueError(f"override {item!r} must look like step.param=value")
            if step_name not in by_name:
                raise ValueError(f"override {item!r}: no step named {step_name!r} "
                                 f"(steps: {sorted(by_name)})")
            try:
                value = json.loads(raw)
            except ValueError:
                value = raw
            by_name[step_name].setdefault("params", {})[param] = value
        return config

    def from_dict(self, config: dict) -> Pipeline:
        from pygeovision.pipelines.orchestrator import Pipeline
        return Pipeline.from_config(config)

    def dump(self, config: dict, path: str | Path) -> None:
        """Save a pipeline config to YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
