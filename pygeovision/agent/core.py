"""
pygeovision.agent.core
=======================
GeoAgent — Autonomous Geospatial AI Agent powered by PyGeoVision.

GeoAgent accepts natural-language geospatial queries and autonomously
executes the complete workflow: satellite data search → download →
preprocessing → AI inference → postprocessing → output.

It works in two modes:

  LLM mode  (ANTHROPIC_API_KEY set):
      Uses Claude to decompose queries into optimal tool sequences.
      Handles complex, ambiguous, or multi-step queries.

  Demo mode (no API key):
      Uses a keyword-matching heuristic planner.
      Covers ~80% of common geospatial workflows with no external calls.

Quick start::

    import pygeovision as pgv
    from pygeovision.agent import GeoAgent

    client = pgv.PyGeoVision()
    agent  = GeoAgent(client)

    # Set spatial context once
    agent.set_context(
        bbox=[-0.30, 5.50, -0.05, 5.70],  # Accra, Ghana
        date="2026-06",
        output_dir="./accra_results/",
    )

    # Natural language query
    result = agent.run("Map current flood inundation in the Odaw River Basin")
    print(result.summary())
    # → ./accra_results/flood_mask_cog.tif
    # → ./accra_results/flood_mask.geojson

    # Streaming (progress per step)
    for event in agent.stream("Detect building damage after the earthquake"):
        print(event)
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger("pygeovision.agent")


class GeoAgent:
    """
    Autonomous Geospatial AI Agent.

    Parameters
    ----------
    pgv_client : PyGeoVision
        An initialised ``PyGeoVision()`` client.
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
        If absent, uses the heuristic planner (no external calls).
    model : str
        Claude model to use for LLM planning.  Default: claude-sonnet-4-6.
    stop_on_failure : bool
        Stop execution when a step fails (default True).
    verbose : bool
        Print step progress to stdout.
    output_dir : str
        Default output directory for all agent runs.
    """

    def __init__(
        self,
        pgv_client: Any,
        *,
        api_key:         str | None = None,
        model:           str | None = None,
        provider:        str | None = None,   # "anthropic" | "groq"; auto-detected if None
        stop_on_failure: bool = True,
        verbose:         bool = True,
        output_dir:      str  = "./agent_output/",
    ) -> None:
        self._pgv    = pgv_client
        self._model  = model
        self._stop   = stop_on_failure
        self._verbose= verbose

        # ── Build tools ────────────────────────────────────────────────────────
        from pygeovision.agent.executor import PlanExecutor
        from pygeovision.agent.memory import GeoAgentMemory
        from pygeovision.agent.planner import HeuristicPlanner, LLMPlanner
        from pygeovision.agent.tools import build_tools

        self._tools = build_tools(pgv_client)

        # ── Planner (LLM when a key is available, heuristic otherwise) ────────
        # Provider auto-detection: explicit `provider` wins; otherwise use
        # whichever of ANTHROPIC_API_KEY/GROQ_API_KEY is actually set
        # (Anthropic first if both are present, for backwards compatibility
        # with existing behaviour).
        _anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _groq_key      = os.environ.get("GROQ_API_KEY", "")

        if provider == "groq" or (provider is None and not _anthropic_key and _groq_key):
            _provider = "groq"
            _key = api_key or _groq_key
            _default_model = "llama-3.3-70b-versatile"
        else:
            _provider = "anthropic"
            _key = api_key or _anthropic_key
            _default_model = "claude-sonnet-4-6"

        if _key:
            self._planner   = LLMPlanner(
                self._tools, api_key=_key, model=model or _default_model, provider=_provider,
            )
            self._plan_mode = f"llm/{_provider}"
            self._model = model or _default_model
        else:
            self._planner   = HeuristicPlanner(self._tools)
            self._plan_mode = "heuristic"
            if verbose:
                print(
                    "[GeoAgent] No ANTHROPIC_API_KEY found — using heuristic planner.\n"
                    "           Set ANTHROPIC_API_KEY for LLM-powered query understanding."
                )

        self._memory   = GeoAgentMemory()
        self._executor = PlanExecutor(
            tools=self._tools,
            on_step_start=self._on_step_start if verbose else None,
            on_step_done =self._on_step_done  if verbose else None,
            stop_on_failure=stop_on_failure,
        )
        self._memory.set_context(output_dir=output_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_context(self, **kwargs) -> GeoAgent:
        """
        Set spatial context for all subsequent queries.

        Common context keys::

            bbox       = [lon_min, lat_min, lon_max, lat_max]  # WGS84
            date       = "2026-06"                              # YYYY-MM or YYYY-MM-DD
            output_dir = "./my_results/"
            bands      = ["B02","B03","B04","B08","B11","B12"]  # Sentinel-2 default
            providers  = ["planetary_computer"]
        """
        self._memory.set_context(**kwargs)
        return self

    def bind(self, name: str, value: Any) -> GeoAgent:
        """Bind a named output (e.g. a file path) for reference in future queries."""
        self._memory.bind(name, value)
        return self

    def run(
        self,
        query: str,
        context_override: dict[str, Any] | None = None,
    ):
        """
        Execute a natural-language geospatial query end-to-end.

        Parameters
        ----------
        query : str
            Natural-language instruction, e.g.
            "Map flood extent in Accra after the June 2026 rains".
        context_override : dict | None
            One-shot context that overrides the session context for this
            query only (does not update the session).

        Returns
        -------
        ExecutionTrace
            Use ``.summary()`` for a text summary, ``.final_output`` for
            the primary output file path, and ``.to_dict()`` for JSON.
        """
        ctx = {**self._memory.context_for_planner(), **(context_override or {})}

        if self._verbose:
            mode_str = f"[{self._plan_mode}]"
            print(f"\n{'─'*60}")
            print(f"GeoAgent {mode_str}: {query}")
            print(f"{'─'*60}")

        # Plan
        plan = self._planner.plan(query, context=ctx)
        if self._verbose:
            print(f"Plan ({plan.planner}, {len(plan.steps)} steps):")
            for s in plan.steps:
                print(f"  {s.step_idx}. {s.tool_name}"
                      + (f"  ← {s.rationale}" if s.rationale else ""))

        # Execute
        trace = self._executor.execute(plan, context=ctx)

        # Record
        self._memory.record_turn(query, plan, trace)

        if self._verbose:
            print(f"\n{trace.summary()}")
            if trace.final_output:
                print(f"\n→ Final output: {trace.final_output}")

        return trace

    def stream(
        self,
        query: str,
        context_override: dict[str, Any] | None = None,
    ) -> Iterator[dict]:
        """
        Streaming version of ``run()``.  Yields event dicts for each step.

        Useful for building UIs or logging pipelines.

        Event types: "plan" | "step_start" | "step_done" | "execution_complete"

        Example::

            for event in agent.stream("Map flood extent"):
                if event["type"] == "step_done":
                    print(f"Step {event['step']}: {'✓' if event['success'] else '✗'}")
                elif event["type"] == "execution_complete":
                    print(f"Done: {event['final_output']}")
        """
        ctx  = {**self._memory.context_for_planner(), **(context_override or {})}
        plan = self._planner.plan(query, context=ctx)
        yield from self._executor.stream(plan, context=ctx)

    def tools(self) -> list[str]:
        """Return list of available tool names."""
        return list(self._tools.keys())

    def tool_schema(self, name: str) -> dict | None:
        """Return the schema for a specific tool."""
        t = self._tools.get(name)
        return t.schema() if t else None

    def all_schemas(self) -> list[dict]:
        """Return schemas for all tools (useful for documentation)."""
        return [t.schema() for t in self._tools.values()]

    def history(self) -> list[dict]:
        """Return the conversation history as a list of turn dicts."""
        return [t.to_dict() for t in self._memory.turns]

    def save_session(self, path: str) -> None:
        """Save the session state (context + history) to a JSON file."""
        self._memory.save(path)

    def load_session(self, path: str) -> None:
        """Restore a previously saved session."""
        self._memory.load(path)

    def reset(self) -> GeoAgent:
        """Clear conversation history and bindings (preserves spatial context)."""
        from pygeovision.agent.memory import GeoAgentMemory
        ctx = self._memory._spatial.copy()
        self._memory = GeoAgentMemory()
        self._memory._spatial = ctx
        return self

    # ── Internal callbacks ─────────────────────────────────────────────────────

    def _on_step_start(self, step_idx: int, tool_name: str, args: dict) -> None:
        print(f"  [{step_idx}] ▶  {tool_name} …")

    def _on_step_done(self, step_idx: int, result) -> None:
        status = "✓" if result.success else "✗"
        detail = (f"→ {result.output_path}" if result.output_path
                  else result.error[:80] if result.error else "")
        print(f"  [{step_idx}] {status}  {result.tool}  ({result.duration_s:.1f}s)  {detail}")

    def __repr__(self) -> str:
        return (
            f"GeoAgent(planner={self._plan_mode!r}, "
            f"tools={len(self._tools)}, "
            f"turns={len(self._memory.turns)})"
        )