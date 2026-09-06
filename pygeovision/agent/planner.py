"""
pygeovision.agent.planner
==========================
Task planner for the GeoAgent.

With an Anthropic API key:  uses Claude to decompose natural language
    geospatial queries into ordered tool call sequences.

Without an API key (demo mode):  uses an intent-understanding heuristic
    that explicitly reasons about SENSOR, TASK TYPE, and APPROACH TIER
    before choosing a tool sequence.  This covers the vast majority of
    real geospatial workflows without any external API calls.

Decision hierarchy (heuristic mode)
------------------------------------
1. TASK TYPE — what does the user actually want to produce?
      Flood / inundation mapping
      Building / structural damage assessment
      Change detection (generic temporal comparison)
      Land cover / land use classification
      Crop / agriculture analysis
      Forest / deforestation monitoring
      Wildfire / burn severity
      Spectral index computation (NDVI, NDWI, …)
      Object detection (buildings, roads, solar panels)
      Named pipeline (building_footprints, glacier_monitoring, …)

2. APPROACH TIER — within the chosen task, which model tier?
      (Currently selects zero-shot; upgrade path shown in rationale)
      zero_shot → fine_tune → benchmark → domain

The planner always returns a list of ``Step`` objects — the executor
doesn't care how they were generated.

Note: SAR/InSAR-specific processing tools and routing have been
removed from pygeovision — that processing is handled directly by
pygeofetch's own real SAR/InSAR modules (pygeofetch.sar / pygeofetch.insar).
SAR/InSAR data acquisition itself is unaffected; only pygeovision's
duplicate processing wrapper was removed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pygeovision.agent.planner")


# ── Step / Plan dataclasses ────────────────────────────────────────────────────

@dataclass
class Step:
    """One planned tool call."""
    step_idx:   int
    tool_name:  str
    args:       dict[str, Any] = field(default_factory=dict)
    depends_on: list[int]      = field(default_factory=list)
    rationale:  str            = ""

    def to_dict(self) -> dict:
        return {
            "step":       self.step_idx,
            "tool":       self.tool_name,
            "args":       self.args,
            "depends_on": self.depends_on,
            "rationale":  self.rationale,
        }

@dataclass
class Plan:
    """Ordered sequence of steps to fulfil a user request."""
    query:   str
    steps:   list[Step] = field(default_factory=list)
    planner: str        = "heuristic"
    sensor:  str        = "unknown"   # "sar" | "optical" | "mixed"
    task:    str        = "unknown"
    notes:   str        = ""

    def pretty(self) -> str:
        lines = [f"Plan ({self.planner}) for: {self.query!r}",
                 f"  sensor={self.sensor}  task={self.task}", ""]
        for s in self.steps:
            lines.append(f"  Step {s.step_idx}: {s.tool_name}")
            lines.append(f"    args:     {json.dumps(s.args, indent=6, default=str)}")
            if s.rationale:
                lines.append(f"    rationale: {s.rationale}")
        if self.notes:
            lines.append(f"\nNotes: {self.notes}")
        return "\n".join(lines)

# ── LLM planner (Anthropic Claude) ────────────────────────────────────────────

SYSTEM_PROMPT = """You are GeoPlanner, the task-planning brain of PyGeoVision, a geospatial AI platform.

Your role: decompose a natural-language geospatial query into an ordered sequence of
PyGeoVision tool calls that will produce the requested output (map, GeoJSON, GeoTIFF,
statistics, etc.).

Decision rules you MUST follow:
1. TASK ROUTING
   • flood / inundation → search → download → prepare_for_ai → prithvi_inference(task=flood_detection)
   • building DAMAGE / earthquake damage / structural collapse → change_detection
   • deforestation / forest change → change_detection or run_pipeline(forest_monitoring)
   • land cover / LULC / classification → prithvi_inference(task=land_cover)
   • crop type / crop mapping → prithvi_inference(task=crop_mapping)
   • burn scar / wildfire mapping → prithvi_inference(task=burn_scar)
   • NDVI / NDWI / NDBI / spectral index → compute_spectral_index
   • building footprints / extraction → run_pipeline(building_footprints)

2. Always end with postprocess if the user wants a map, GeoJSON, or downloadable output.
3. Use run_pipeline for standard named tasks when a pipeline matches well.
4. Output ONLY a valid JSON object with key "steps". No prose before or after.

Note: SAR/InSAR-specific processing tools (sar_preprocess, sar_flood_detection,
slc_insar) have been removed — that processing is handled directly by
pygeofetch's own real SAR/InSAR modules (pygeofetch.sar / pygeofetch.insar),
not through this planner. Route SAR/InSAR-flavored queries to a direct
pygeofetch call instead of emitting a step for any of these tool names.
7. Each step: {{"step": int, "tool": str, "args": {{}}, "depends_on": [int], "rationale": str}}

Available tools:
{{tools_schema}}
"""

def _build_tools_schema(tool_instances: dict) -> str:
    parts = []
    for name, tool in tool_instances.items():
        params = [
            f"  - {p['name']} ({p['type']}, {'required' if p.get('required') else 'optional'})"
            + (f": {p['description']}" if p.get('description') else "")
            for p in tool.parameters
        ]
        parts.append(f"• {name}: {tool.description}\n" + "\n".join(params))
    return "\n\n".join(parts)


class LLMPlanner:
    """Plans tool sequences using an LLM — Anthropic Claude or Groq.

    Groq is a genuinely different provider (fast open-weight model
    inference — llama-3.3-70b-versatile, gpt-oss, etc. — not a
    Claude-compatible wrapper), so this uses Groq's real
    chat.completions API with JSON mode for reliable structured output,
    not a shared code path pretending the two APIs are the same shape.

    Example::

        planner = LLMPlanner(tools, provider="groq", api_key="gsk_...")
        planner = LLMPlanner(tools, provider="anthropic")  # default
    """

    def __init__(self, tool_instances: dict, api_key: str | None = None,
                 model: str | None = None, provider: str = "anthropic") -> None:
        self._tools      = tool_instances
        self._provider   = provider
        if provider == "groq":
            self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
            self._model   = model or "llama-3.3-70b-versatile"
        else:
            self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._model   = model or "claude-sonnet-4-6"
        self._tools_text = _build_tools_schema(tool_instances)

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def plan(self, query: str, context: dict | None = None) -> Plan:
        if not self.available:
            env_var = "GROQ_API_KEY" if self._provider == "groq" else "ANTHROPIC_API_KEY"
            raise RuntimeError(f"No {env_var} — use HeuristicPlanner.")

        ctx_str = (f"\n\nContext: {json.dumps(context, indent=2)}" if context else "")

        if self._provider == "groq":
            raw = self._plan_groq(query, ctx_str)
        else:
            raw = self._plan_anthropic(query, ctx_str)

        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned malformed JSON: {exc}") from exc

        steps = [
            Step(
                step_idx=s.get("step", i),
                tool_name=s.get("tool", ""),
                args=s.get("args", {}),
                depends_on=s.get("depends_on", []),
                rationale=s.get("rationale", ""),
            )
            for i, s in enumerate(parsed.get("steps", []))
        ]
        return Plan(query=query, steps=steps, planner=f"{self._provider}/{self._model}",
                    notes=parsed.get("notes", ""))

    def _plan_anthropic(self, query: str, ctx_str: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        client  = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT.format(tools_schema=self._tools_text),
            messages=[{"role": "user", "content": query + ctx_str}],
        )
        return message.content[0].text.strip()

    def _plan_groq(self, query: str, ctx_str: str) -> str:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")
        client = Groq(api_key=self._api_key)
        completion = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(tools_schema=self._tools_text)},
                {"role": "user", "content": query + ctx_str},
            ],
            # JSON mode — Groq's recommended way to get reliably-parseable
            # structured output (the planner's response MUST be valid JSON
            # to build a Plan; without this, models occasionally wrap
            # output in prose despite being told not to).
            response_format={"type": "json_object"},
            temperature=0.2,
            max_completion_tokens=2048,
        )
        return completion.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic planner — intent-understanding approach
# ═══════════════════════════════════════════════════════════════════════════════

class HeuristicPlanner:
    """
    Intent-understanding heuristic planner.

    Unlike a simple keyword→template map, this planner explicitly reasons
    through two decisions in order:

        1. TASK     — what the user wants to produce (flood map, damage map, land cover…)
        2. APPROACH — tool sequence that delivers that task

    The agent thus gives different answers to:
        "Map flood extent using Sentinel-2"         → optical Prithvi pipeline
        "Map building damage after earthquake"      → change_detection (not flood!)

    Note: SAR/InSAR-specific routing (sensor choice, SAR-flavored task
    variants) has been removed — that processing is handled directly by
    pygeofetch's own real SAR/InSAR modules, not through this planner.
    """

    # ── Task signal words ──────────────────────────────────────────────────────
    TASK_SIGNALS: dict[str, set[str]] = {
        "flood":        {"flood", "inundation", "submerged", "waterlogged",
                         "floodwater", "inundated", "flooding"},
        "damage":       {"damage", "damaged", "destruction", "collapsed", "rubble",
                         "destroyed", "earthquake", "hurricane", "typhoon",
                         "structural", "building damage", "disaster damage",
                         "seismic", "tsunami"},
        "change":       {"change", "deforestation", "forest loss", "urban growth",
                         "expansion", "temporal", "before after", "pre post",
                         "bi-temporal", "bitemporal", "compare"},
        "land_cover":   {"land cover", "land use", "lulc", "classification",
                         "land type", "habitat", "ecosystem"},
        "crop":         {"crop", "cropland", "agriculture", "farm", "field",
                         "harvest", "irrigated", "sowing"},
        "forest":       {"forest", "deforestation", "tree cover", "biomass",
                         "canopy", "woodland", "timber"},
        "burn":         {"fire", "wildfire", "burn scar", "burned area",
                         "fire scar", "conflagration", "dnbr", "nbr"},
        "spectral":     {"ndvi", "ndwi", "ndbi", "evi", "savi",
                         "vegetation index", "water index", "spectral index",
                         "greenness", "moisture index"},
        "building":     {"building", "footprint", "structure", "urban extraction",
                         "building detection"},
        "oil_spill":    {"oil spill", "oil slick", "hydrocarbon", "petroleum",
                         "marine pollution", "dark patch sea"},
        "subsidence":   {"subsidence", "settlement", "sinking", "deformation",
                         "ground movement", "displacement", "uplift"},
        "glacier":      {"glacier", "ice", "snow", "cryosphere", "permafrost"},
        "ship":         {"ship", "vessel", "maritime", "boat", "ais", "shipping"},
        "solar":        {"solar panel", "pv", "photovoltaic"},
        "road":         {"road", "highway", "network", "street"},
    }

    def __init__(self, tool_instances: dict) -> None:
        self._tools = tool_instances

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(self, query: str, context: dict | None = None) -> Plan:
        ctx = context or {}
        ql  = query.lower()

        sensor  = self._infer_sensor(ql, ctx)
        task    = self._infer_task(ql)
        od      = ctx.get("output_dir", "./agent_output/").rstrip("/")
        bbox    = ctx.get("bbox", [0, 0, 1, 1])
        date    = ctx.get("date", "2024-06")
        bands   = ctx.get("bands", ["B02","B03","B04","B08","B11","B12"])
        ctx_sar = ctx.get("sar_path")

        logger.info("HeuristicPlanner: sensor=%s  task=%s  query=%r",
                    sensor, task, query[:60])

        steps = self._build_plan(sensor, task, od=od, bbox=bbox, date=date,
                                  bands=bands, ctx_sar=ctx_sar, query=ql)

        return Plan(
            query=query, steps=steps, planner="heuristic",
            sensor=sensor, task=task,
            notes=(
                "Heuristic plan — set ANTHROPIC_API_KEY for LLM-powered planning "
                "with better parameter inference and multi-step reasoning."
            ),
        )

    # ── Sensor inference ───────────────────────────────────────────────────────

    def _infer_sensor(self, ql: str, ctx: dict) -> str:
        """Sensor selection for planning purposes.

        SAR/InSAR-specific processing tools have been removed from
        pygeovision (that processing is handled directly by pygeofetch's
        own real SAR/InSAR modules — see pygeofetch.sar / pygeofetch.insar).
        This always resolves to the optical planning path now; SAR/InSAR
        data acquisition itself is unaffected and still available via
        the normal search/download tools.
        """
        return "optical"

    # ── Task inference ─────────────────────────────────────────────────────────

    def _infer_task(self, ql: str) -> str:
        """Score every task against the query and return the best match."""
        scores: dict[str, int] = {}
        for task, signals in self.TASK_SIGNALS.items():
            score = sum(1 for kw in signals if kw in ql)
            if score:
                scores[task] = score

        if not scores:
            return "generic"

        # Resolve ties: damage beats flood when earthquake/disaster words present
        best = max(scores, key=lambda t: (scores[t], list(self.TASK_SIGNALS).index(t)))

        # Explicit tie-breaking rules
        if "damage" in scores and "flood" in scores:
            if any(w in ql for w in ("earthquake", "building", "structural",
                                      "collapse", "destruction", "rubble")):
                best = "damage"
            elif any(w in ql for w in ("water", "inundation", "flooded area")):
                best = "flood"

        if "forest" in scores and "change" in scores:
            best = "change"  # deforestation = change detection

        return best

    # ── Plan builder ───────────────────────────────────────────────────────────

    def _build_plan(
        self, sensor: str, task: str, *,
        od: str, bbox: list, date: str, bands: list,
        ctx_sar: str | None, query: str,
    ) -> list[Step]:
        """Route to the correct tool sequence given task.

        SAR/InSAR-specific processing branches have been removed --
        that processing is handled directly by pygeofetch's own real
        SAR/InSAR modules. `sensor` is always "optical" now (see
        _infer_sensor); the parameter is kept for call-site compatibility.
        """
        if task == "flood":
            return self._plan_optical_flood(od, bbox, date, bands)
        elif task == "damage":
            return self._plan_optical_damage(od, bbox, date, bands)
        elif task == "change" or task == "forest":
            return self._plan_optical_change(od, bbox, date, task)
        elif task == "land_cover":
            return self._plan_optical_land_cover(od, bbox, date, bands)
        elif task == "crop":
            return self._plan_optical_crop(od, bbox, date, bands)
        elif task == "burn":
            return self._plan_optical_burn(od, bbox, date, bands)
        elif task == "spectral":
            index = self._extract_index(query)
            return self._plan_spectral_index(od, bbox, date, bands, index)
        elif task == "building":
            return self._plan_building_footprints(od, bbox, date)
        elif task == "glacier":
            return self._plan_glacier(od, bbox, date)
        elif task == "solar":
            return self._plan_solar(od, bbox, date)
        elif task == "road":
            return self._plan_road(od, bbox, date)
        elif task == "subsidence":
            # Real fix, confirmed by tracing this directly: "subsidence"
            # is a real, still-detected task keyword set (a leftover
            # from before InSAR support was removed from pygeovision's
            # scope), but had no real routing branch at all -- it
            # silently fell through to the land-cover default below,
            # meaning a real request for subsidence/deformation
            # monitoring would silently get an unrelated land-cover
            # plan with no indication the real request couldn't be
            # fulfilled. Raises clearly instead, consistent with the
            # same honest-failure pattern used elsewhere in this
            # codebase (e.g. CenterNet, LISAt) for real, out-of-scope
            # capability gaps.
            raise ValueError(
                "Ground subsidence/deformation monitoring requires real "
                "InSAR displacement processing, which is out of "
                "pygeovision's scope (removed entirely -- see the "
                "project roadmap). Use pygeofetch's own real InSAR "
                "module directly: pygeofetch.insar."
            )
        else:
            return self._plan_optical_land_cover(od, bbox, date, bands)

    # ── Optical plan templates ─────────────────────────────────────────────────

    def _optical_search_download_prepare(self, od, bbox, date, bands,
                                          model_type="foundation") -> list[Step]:
        dr = (f"{date}-01", f"{date}-30") if len(date) == 7 else (date, date)
        return [
            Step(0, "search_satellite_data", {
                "bbox": bbox, "date_range": dr, "cloud_cover_max": 15,
            }, rationale="Find cloud-free optical scene from open providers"),
            Step(1, "download_satellite_data", {
                "scenes": "$step_0_output", "output_dir": f"{od}/raw/",
                "bands": bands, "max_scenes": 1,
            }, depends_on=[0]),
            Step(2, "prepare_for_ai", {
                "input_path": "$step_1_output", "bbox": bbox,
                "bands": bands, "model_type": model_type,
                "output_path": f"{od}/prepared.tif",
            }, depends_on=[1]),
        ]

    def _plan_optical_flood(self, od, bbox, date, bands) -> list[Step]:
        steps = self._optical_search_download_prepare(od, bbox, date, bands)
        steps += [
            Step(3, "prithvi_inference", {
                "input_path": f"{od}/prepared.tif",
                "task": "flood_detection", "source": "sentinel2",
                "output_path": f"{od}/flood_mask.tif",
            }, depends_on=[2], rationale=(
                "Prithvi-EO-2.0 flood mask (cloud-free optical). "
                "For cloud-cover situations, use pygeofetch's SAR flood "
                "mapping directly (pygeofetch.processing.sar.SARProcessor)."
            )),
            self._postprocess_step(4, f"{od}/flood_mask.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[3]),
        ]
        return steps

    def _plan_optical_damage(self, od, bbox, date, bands) -> list[Step]:
        steps = self._optical_search_download_prepare(od, bbox, date, bands)
        steps += [
            Step(3, "change_detection", {
                "pre_path":    "$context_pre_optical_path",
                "post_path":   f"{od}/prepared.tif",
                "output_path": f"{od}/damage_map.tif",
                "num_classes": 4,
                "in_channels": len(bands),
            }, depends_on=[2], rationale=(
                "Optical 4-class damage (no damage / minor / moderate / severe) "
                "using ChangeFormer. For cloud-independent results, use "
                "pygeofetch's SAR processing directly instead."
            )),
            self._postprocess_step(4, f"{od}/damage_map.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[3]),
        ]
        return steps

    def _plan_optical_change(self, od, bbox, date, task) -> list[Step]:
        pipe = "forest_monitoring" if task == "forest" else "change_detection"
        return [
            Step(0, "run_pipeline", {
                "pipeline_name": pipe,
                "bbox": bbox, "date": date, "output_dir": od,
            }, rationale=f"Optimised {pipe} pipeline (search→download→ChangeFormer→postprocess)"),
        ]

    def _plan_optical_land_cover(self, od, bbox, date, bands) -> list[Step]:
        steps = self._optical_search_download_prepare(od, bbox, date, bands)
        steps += [
            Step(3, "prithvi_inference", {
                "input_path": f"{od}/prepared.tif",
                "task": "land_cover", "source": "sentinel2",
                "source_bands": bands,
                "output_path": f"{od}/land_cover.tif",
            }, depends_on=[2], rationale="Prithvi-EO-2.0 600M — 9-class ESA land cover"),
            self._postprocess_step(4, f"{od}/land_cover.tif", od,
                                   ["cog"], depends_on=[3]),
        ]
        return steps

    def _plan_optical_crop(self, od, bbox, date, bands) -> list[Step]:
        steps = self._optical_search_download_prepare(od, bbox, date, bands)
        steps += [
            Step(3, "prithvi_inference", {
                "input_path": f"{od}/prepared.tif",
                "task": "crop_mapping", "source": "sentinel2",
                "output_path": f"{od}/crop_map.tif",
            }, depends_on=[2], rationale="Prithvi-EO-2.0 crop type classification"),
            self._postprocess_step(4, f"{od}/crop_map.tif", od,
                                   ["cog"], depends_on=[3]),
        ]
        return steps

    def _plan_optical_burn(self, od, bbox, date, bands) -> list[Step]:
        steps = self._optical_search_download_prepare(od, bbox, date, bands)
        steps += [
            Step(3, "prithvi_inference", {
                "input_path": f"{od}/prepared.tif",
                "task": "burn_scar", "source": "sentinel2",
                "output_path": f"{od}/burn_scar.tif",
            }, depends_on=[2], rationale=(
                "Prithvi burn scar + dNBR severity. "
                "Also computes NBR for pre/post fire severity comparison."
            )),
            self._postprocess_step(4, f"{od}/burn_scar.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[3]),
        ]
        return steps

    def _plan_spectral_index(self, od, bbox, date, bands, index) -> list[Step]:
        steps = self._optical_search_download_prepare(od, bbox, date, bands,
                                                       model_type="segmentation")
        steps += [
            Step(3, "compute_spectral_index", {
                "input_path":  f"{od}/prepared.tif",
                "index":       index,
                "output_path": f"{od}/{index}.tif",
            }, depends_on=[2], rationale=f"Compute {index.upper()} from stacked bands"),
            self._postprocess_step(4, f"{od}/{index}.tif", od,
                                   ["cog"], depends_on=[3]),
        ]
        return steps

    def _plan_building_footprints(self, od, bbox, date) -> list[Step]:
        return [
            Step(0, "run_pipeline", {
                "pipeline_name": "building_footprints",
                "bbox": bbox, "date": date, "output_dir": od,
            }, rationale="Optimised building footprint pipeline: SegFormer-B2 → GeoJSON"),
        ]

    def _plan_glacier(self, od, bbox, date) -> list[Step]:
        return [
            Step(0, "run_pipeline", {
                "pipeline_name": "glacier_monitoring",
                "bbox": bbox, "date": date, "output_dir": od,
            }, rationale="NDSI time-series + retreat rate calculation"),
        ]

    def _plan_solar(self, od, bbox, date) -> list[Step]:
        return [
            Step(0, "run_pipeline", {
                "pipeline_name": "solar_detection",
                "bbox": bbox, "date": date, "output_dir": od,
            }, rationale="Solar panel detection + capacity estimation"),
        ]

    def _plan_road(self, od, bbox, date) -> list[Step]:
        return [
            Step(0, "run_pipeline", {
                "pipeline_name": "road_extraction",
                "bbox": bbox, "date": date, "output_dir": od,
            }, rationale="Road network extraction + vectorisation"),
        ]

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _postprocess_step(self, idx: int, input_path: str, od: str,
                           operations: list[str], depends_on: list[int]) -> Step:
        return Step(idx, "postprocess", {
            "input_path": input_path,
            "operations": operations,
            "output_dir": od,
            "min_pixels":  30,
            "min_area_m2": 500.0,
        }, depends_on=depends_on, rationale=(
            "Remove small noise patches (sieve), export as GeoJSON polygons "
            "(vectorise), and Cloud-Optimised GeoTIFF for web delivery (cog)."
        ))

    def _extract_index(self, ql: str) -> str:
        for idx in ("ndwi", "ndbi", "evi", "savi", "nbr", "mndwi", "ndvi"):
            if idx in ql:
                return idx
        return "ndvi"   # default