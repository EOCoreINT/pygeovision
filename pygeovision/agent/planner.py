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
1. SENSOR — is this a SAR or optical task?
      SAR signals: "sar", "sentinel-1", "radar", "cloud cover", "night",
                   "cloud-independent", "microwave"
      Optical signals (override): "sentinel-2", "landsat", "optical",
                   "multispectral", "clear sky", "cloud-free"

2. TASK TYPE — what does the user actually want to produce?
      Flood / inundation mapping
      Building / structural damage assessment
      Change detection (generic temporal comparison)
      Land cover / land use classification
      Crop / agriculture analysis
      Forest / deforestation monitoring
      Oil spill / marine pollution detection
      Subsidence / deformation monitoring
      Wildfire / burn severity
      Spectral index computation (NDVI, NDWI, …)
      Object detection (buildings, roads, ships, solar panels)
      Named pipeline (building_footprints, glacier_monitoring, …)

3. APPROACH TIER — within the chosen task, which model tier?
      (Currently selects zero-shot; upgrade path shown in rationale)
      zero_shot → fine_tune → spt (SAR) / benchmark → domain (optical)

The planner always returns a list of ``Step`` objects — the executor
doesn't care how they were generated.
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
1. SENSOR CHOICE
   • "SAR", "Sentinel-1", "S1C", "S1D", "radar", "cloud cover", "night",
     "cloud-independent", "insar", "interferogram", "slc" → prefer SAR tools
   • "Sentinel-2", "Landsat", "optical", "multispectral", "clear sky" → prefer optical tools
   • When neither is specified, choose SAR for disaster/emergency and optical otherwise.
   • Sentinel-1C and Sentinel-1D are the active constellation (2025+); treat identically to Sentinel-1.

2. TASK ROUTING
   • flood / inundation + SAR → sar_preprocess then sar_flood_detection
   • flood / inundation + optical → search → download → prepare_for_ai → prithvi_inference(task=flood_detection)
   • building DAMAGE / earthquake damage / structural collapse → change_detection (NOT sar_flood_detection)
   • deforestation / forest change → change_detection or run_pipeline(forest_monitoring)
   • land cover / LULC / classification → prithvi_inference(task=land_cover)
   • crop type / crop mapping → prithvi_inference(task=crop_mapping)
   • burn scar / wildfire mapping → prithvi_inference(task=burn_scar)
   • NDVI / NDWI / NDBI / spectral index → compute_spectral_index
   • building footprints / extraction → run_pipeline(building_footprints)
   • oil spill → sar_preprocess then sar_flood_detection (dark-pixel applies to oil too)
   • subsidence / deformation + millimetre / insar / slc / precise → slc_insar
   • subsidence / deformation (no precision keyword, quick result needed) → sar_preprocess then change_detection
   • InSAR / interferogram / phase unwrapping / LOS displacement / deformation rate → slc_insar

3. SLC InSAR rules (slc_insar tool)
   • slc_insar needs master_zip and slave_zip: use $context_master_slc_zip and $context_slave_slc_zip
   • slc_insar is ONE step — it produces the final displacement GeoTIFF directly
   • Do NOT chain sar_preprocess before slc_insar — SLC processing is entirely different
   • Emit slc_insar for: "millimetre", "mm precision", "insar", "interferogram",
     "slc", "phase", "los displacement", "deformation rate", "coherence"

4. Always end with postprocess if the user wants a map, GeoJSON, or downloadable output.
5. Use run_pipeline for standard named tasks when a pipeline matches well.
6. Output ONLY a valid JSON object with key "steps". No prose before or after.
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
    """Plans tool sequences using Claude."""

    def __init__(self, tool_instances: dict, api_key: str | None = None,
                 model: str = "claude-sonnet-4-6") -> None:
        self._tools      = tool_instances
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model      = model
        self._tools_text = _build_tools_schema(tool_instances)

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def plan(self, query: str, context: dict | None = None) -> Plan:
        if not self.available:
            raise RuntimeError("No ANTHROPIC_API_KEY — use HeuristicPlanner.")
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        ctx_str = (f"\n\nContext: {json.dumps(context, indent=2)}" if context else "")
        client  = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT.format(tools_schema=self._tools_text),
            messages=[{"role": "user", "content": query + ctx_str}],
        )
        raw = message.content[0].text.strip()
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
        return Plan(query=query, steps=steps, planner=f"claude/{self._model}",
                    notes=parsed.get("notes", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic planner — intent-understanding approach
# ═══════════════════════════════════════════════════════════════════════════════

class HeuristicPlanner:
    """
    Intent-understanding heuristic planner.

    Unlike a simple keyword→template map, this planner explicitly reasons
    through three decisions in order:

        1. SENSOR  — SAR vs optical (from explicit mentions or implicit task context)
        2. TASK    — what the user wants to produce (flood map, damage map, land cover…)
        3. APPROACH — tool sequence that delivers that task via the chosen sensor

    The agent thus gives different answers to:
        "Map flood extent — cloud cover is 100%"    → SAR pipeline
        "Map flood extent using Sentinel-2"         → optical Prithvi pipeline
        "Map building damage after earthquake"      → change_detection (not flood!)
        "Detect oil spill with Sentinel-1 at night" → SAR dark-pixel pipeline
    """

    # ── Sensor signal words ────────────────────────────────────────────────────
    SAR_SIGNALS = {
        "sar", "sentinel-1", "sentinel 1", "s1", "s1a", "s1b",
        # Sentinel-1C and 1D (new constellation, active from 2025)
        "s1c", "s1d", "sentinel-1c", "sentinel-1d",
        "sentinel 1c", "sentinel 1d", "sentinel-1 c", "sentinel-1 d",
        # SAR-specific terms
        "radar", "backscatter", "cloud-independent", "cloud independent",
        "cloud cover", "cloud-free not available", "no optical",
        "night", "microwave", "c-band", "iw mode", "grd", "slc",
        # InSAR-specific — queries with these almost always want SAR
        "insar", "interferogram", "phase unwrap", "coherence",
        "los displacement", "line of sight", "deformation rate",
    }
    OPTICAL_SIGNALS = {
        "sentinel-2", "sentinel 2", "s2", "landsat", "optical",
        "multispectral", "rgb", "clear sky", "cloud-free", "naip",
        "planet", "worldview", "maxar",
    }

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
        "insar":        {"insar", "interferogram", "phase unwrapping",
                         "slc interferometry", "millimetre", "mm precision",
                         "millimeter", "deformation rate", "surface displacement",
                         "los displacement", "line of sight displacement",
                         "subsidence rate", "uplift rate", "coherence map",
                         "displacement time series", "phase unwrap"},
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
        """Decide whether the task calls for SAR, optical, or mixed data."""
        sar_score     = sum(1 for kw in self.SAR_SIGNALS     if kw in ql)
        optical_score = sum(1 for kw in self.OPTICAL_SIGNALS if kw in ql)

        # Explicit file path in context → trust it
        if ctx.get("sar_path"):
            return "sar"
        if optical_score > sar_score:
            return "optical"
        if sar_score > 0:
            return "sar"

        # Implicit SAR for tasks where radar is clearly superior or standard
        SAR_NATIVE_TASKS = {
            "oil_spill", "subsidence", "ship",
        }
        task = self._infer_task(ql)
        if task in SAR_NATIVE_TASKS:
            return "sar"

        # Implicit SAR for disaster contexts when cloud cover is mentioned
        if any(w in ql for w in ("cloud", "night", "storm", "rain", "typhoon")):
            if any(w in ql for w in ("flood", "damage", "disaster", "emergency")):
                return "sar"

        # Implicit SAR for earthquake/disaster damage — SAR is standard for rapid
        # damage assessment because it works through clouds and at night
        if any(w in ql for w in ("earthquake", "seismic", "tsunami", "typhoon",
                                  "hurricane", "cyclone")) and task == "damage":
            return "sar"

        # Default to optical
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

        # insar beats subsidence when precision signals are present
        if "insar" in scores and "subsidence" in scores:
            best = "insar"

        # Anything with millimetre/InSAR keywords → insar task
        if any(w in ql for w in ("insar", "interferogram", "millimetre", "millimeter",
                                  "slc", "phase unwrap", "coherence map")):
            if "insar" in scores:
                best = "insar"

        return best

    # ── Plan builder ───────────────────────────────────────────────────────────

    def _build_plan(
        self, sensor: str, task: str, *,
        od: str, bbox: list, date: str, bands: list,
        ctx_sar: str | None, query: str,
    ) -> list[Step]:
        """Route to the correct tool sequence given sensor + task."""

        # ── SAR branch ─────────────────────────────────────────────────────────
        if sensor == "sar":
            if task == "insar":
                return self._plan_slc_insar(od, bbox)
            elif task == "flood":
                return self._plan_sar_flood(od, bbox, ctx_sar)
            elif task == "damage":
                return self._plan_sar_damage(od, bbox, ctx_sar)
            elif task == "change":
                return self._plan_sar_change(od, bbox, ctx_sar)
            elif task == "oil_spill":
                return self._plan_sar_oil_spill(od, bbox, ctx_sar)
            elif task == "subsidence":
                # Precision-aware routing:
                # "millimetre" / "insar" / "slc" → true SLC InSAR pipeline
                # "proxy" / "quick" / no precision keyword → GRD amplitude proxy
                needs_precision = any(w in query for w in (
                    "millimetre", "millimeter", "mm", "insar",
                    "interferogram", "slc", "precise", "precision",
                ))
                if needs_precision:
                    return self._plan_slc_insar(od, bbox)
                return self._plan_sar_subsidence(od, bbox, ctx_sar)
            elif task == "ship":
                return self._plan_sar_ship_detection(od, bbox, ctx_sar)
            else:
                return self._plan_sar_flood(od, bbox, ctx_sar)

        # ── Optical branch ─────────────────────────────────────────────────────
        else:
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
            else:
                return self._plan_optical_land_cover(od, bbox, date, bands)

    # ── SAR plan templates ─────────────────────────────────────────────────────

    def _sar_preprocess_step(self, od: str, bbox: list, ctx_sar: str | None) -> Step:
        return Step(0, "sar_preprocess", {
            "raw_path":    ctx_sar or "$context_sar_path",
            "output_path": f"{od}/sar_ready.tif",
            "bbox_wgs84":  bbox,
            "filter_type": "enhanced_lee",
        }, rationale=(
            "BUG 1/2/3-aware SAR pipeline: verify download → validate georeference "
            "→ despeckle on LINEAR data → dB → normalise → CRS-aware clip"
        ))

    def _plan_slc_insar(self, od: str, bbox: list) -> list[Step]:
        """
        Plan true SLC InSAR: requires pre-downloaded SLC .zip products.
        The planner emits placeholder paths that the user must populate
        via context before execution.
        """
        return [
            Step(0, "slc_insar", {
                "master_zip":          "$context_master_slc_zip",
                "slave_zip":           "$context_slave_slc_zip",
                "output_dir":          f"{od}/slc_insar/",
                "subswath":            "IW2",
                "bursts":              [1, 9],
                "polarisation":        "VV",
                "coherence_threshold": 0.3,
            }, rationale=(
                "True SLC InSAR: SNAP co-registration → interferogram → "
                "Goldstein filter → SNAPHU unwrapping → phase-to-displacement → "
                "terrain correction. Achieves millimetre-precision LOS displacement. "
                "Supply master_zip and slave_zip via context before running. "
                "Temporal baseline 6–24 days optimal for Sentinel-1 IW coherence."
            )),
        ]

    def _plan_sar_flood(self, od, bbox, ctx_sar) -> list[Step]:
        return [
            self._sar_preprocess_step(od, bbox, ctx_sar),
            Step(1, "sar_flood_detection", {
                "sar_ready_path": f"{od}/sar_ready.tif",
                "output_path":    f"{od}/flood_mask.tif",
                "mode":           "zero_shot",
            }, depends_on=[0], rationale=(
                "VH threshold flood mask (Prithvi zero-shot adapter). "
                "Upgrade: set mode='fine_tune' after Sen1Floods11 training."
            )),
            self._postprocess_step(2, f"{od}/flood_mask.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[1]),
        ]

    def _plan_sar_damage(self, od, bbox, ctx_sar) -> list[Step]:
        return [
            self._sar_preprocess_step(od, bbox, ctx_sar),
            Step(1, "change_detection", {
                "pre_path":    "$context_pre_sar_path",
                "post_path":   f"{od}/sar_ready.tif",
                "output_path": f"{od}/damage_map.tif",
                "num_classes": 4,
                "in_channels": 2,
            }, depends_on=[0], rationale=(
                "SAR amplitude change between pre-event and post-event "
                "for structural damage classification (4 classes: "
                "stable/minor/moderate/severe). SAR is cloud-independent — "
                "results within hours of a new Sentinel-1 pass."
            )),
            self._postprocess_step(2, f"{od}/damage_map.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[1]),
        ]

    def _plan_sar_change(self, od, bbox, ctx_sar) -> list[Step]:
        return [
            self._sar_preprocess_step(od, bbox, ctx_sar),
            Step(1, "change_detection", {
                "pre_path":    "$context_pre_sar_path",
                "post_path":   f"{od}/sar_ready.tif",
                "output_path": f"{od}/change_mask.tif",
                "num_classes": 2,
                "in_channels": 2,
            }, depends_on=[0], rationale="Bi-temporal SAR amplitude change detection"),
            self._postprocess_step(2, f"{od}/change_mask.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[1]),
        ]

    def _plan_sar_oil_spill(self, od, bbox, ctx_sar) -> list[Step]:
        return [
            self._sar_preprocess_step(od, bbox, ctx_sar),
            Step(1, "sar_flood_detection", {
                "sar_ready_path": f"{od}/sar_ready.tif",
                "output_path":    f"{od}/dark_patch_mask.tif",
                "mode":           "zero_shot",
            }, depends_on=[0], rationale=(
                "Oil slicks suppress SAR backscatter (specular reflection) "
                "— the same VH dark-pixel algorithm that detects open water "
                "also detects oil spills. Night and cloud-proof."
            )),
            self._postprocess_step(2, f"{od}/dark_patch_mask.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[1]),
        ]

    def _plan_sar_subsidence(self, od, bbox, ctx_sar) -> list[Step]:
        return [
            self._sar_preprocess_step(od, bbox, ctx_sar),
            Step(1, "change_detection", {
                "pre_path":    "$context_pre_sar_path",
                "post_path":   f"{od}/sar_ready.tif",
                "output_path": f"{od}/subsidence_proxy.tif",
                "num_classes": 4,
                "in_channels": 2,
            }, depends_on=[0], rationale=(
                "SAR backscatter change proxy for ground deformation. "
                "For centimetre-precision use SLC + pyroSAR + SNAP InSAR."
            )),
            self._postprocess_step(2, f"{od}/subsidence_proxy.tif", od,
                                   ["sieve","cog"], depends_on=[1]),
        ]

    def _plan_sar_ship_detection(self, od, bbox, ctx_sar) -> list[Step]:
        return [
            self._sar_preprocess_step(od, bbox, ctx_sar),
            Step(1, "sar_flood_detection", {
                "sar_ready_path": f"{od}/sar_ready.tif",
                "output_path":    f"{od}/bright_target_mask.tif",
                "mode":           "zero_shot",
            }, depends_on=[0], rationale=(
                "Ships appear as bright (high VV) targets on dark sea. "
                "Inverted VH threshold approximates vessel detection. "
                "Production ship detection: use DINOv3 SAR adapter."
            )),
            self._postprocess_step(2, f"{od}/bright_target_mask.tif", od,
                                   ["sieve","vectorise","cog"], depends_on=[1]),
        ]

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
                "For cloud-cover situations use SAR pipeline instead."
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
                "using ChangeFormer. Pair with SAR pipeline for cloud-independent results."
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
                "pipeline_name": "solar_panels",
                "bbox": bbox, "date": date, "output_dir": od,
            }, rationale="Solar panel detection + capacity estimation"),
        ]

    def _plan_road(self, od, bbox, date) -> list[Step]:
        return [
            Step(0, "run_pipeline", {
                "pipeline_name": "road_network",
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
