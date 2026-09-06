"""
pygeovision.agent.memory
=========================
Conversation memory and spatial context tracker for the GeoAgent.

Stores:
  * Full turn history (query, plan, execution trace)
  * Spatial context (bbox, date, output paths, variable bindings)
  * Derived facts from previous executions (scene IDs, file paths, stats)

Keeps the conversation window bounded so the LLM planner doesn't
receive a prompt that's too long.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("pygeovision.agent.memory")


# ── Turn ──────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One user query + agent response cycle."""
    idx:       int
    query:     str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    plan_summary:  str = ""
    plan_steps:    int = 0
    planner_used:  str = ""
    success:       bool = False
    final_output:  Any  = None
    outputs:       list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn": self.idx, "query": self.query, "timestamp": self.timestamp,
            "plan_steps": self.plan_steps, "planner": self.planner_used,
            "success": self.success, "final_output": str(self.final_output or ""),
        }


# ── Memory ────────────────────────────────────────────────────────────────────

class GeoAgentMemory:
    """
    Persistent short-term memory for one GeoAgent session.

    Usage::

        mem = GeoAgentMemory()
        mem.set_context(bbox=[-0.3, 5.5, -0.05, 5.7], date="2026-06")
        mem.record_turn(query, plan, trace)
        ctx = mem.context_for_planner()   # → dict for the planner
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._turns:    list[Turn] = []
        self._max       = max_turns
        self._spatial:  dict[str, Any] = {}   # bbox, date, crs, output_dir
        self._bindings: dict[str, Any] = {}   # named outputs: "flood_mask" → path

    # ── Spatial context ────────────────────────────────────────────────────────

    def set_context(self, **kwargs) -> None:
        """Set / update spatial context (bbox, date, output_dir, crs, …)."""
        self._spatial.update(kwargs)
        logger.debug("Context updated: %s", list(kwargs.keys()))

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._spatial.get(key, default)

    def bind(self, name: str, value: Any) -> None:
        """Bind a named output so future steps can reference it by name."""
        self._bindings[name] = value
        logger.debug("Bound '%s' → %s", name, str(value)[:80])

    def resolve(self, name: str) -> Any:
        return self._bindings.get(name)

    # ── Turn history ───────────────────────────────────────────────────────────

    def record_turn(self, query: str, plan: Any, trace: Any) -> Turn:
        """Record a completed turn."""

        turn = Turn(
            idx=len(self._turns),
            query=query,
            plan_steps=len(getattr(plan, "steps", [])),
            planner_used=getattr(plan, "planner", "unknown"),
            success=getattr(trace, "success", False),
            final_output=getattr(trace, "final_output", None),
        )
        if hasattr(plan, "steps"):
            turn.plan_summary = " → ".join(s.tool_name for s in plan.steps)

        # Collect all output paths
        if hasattr(trace, "executions"):
            turn.outputs = [
                {"step": ex.step.step_idx, "tool": ex.step.tool_name,
                 "success": ex.result.success, "path": ex.result.output_path}
                for ex in trace.executions
            ]

        # Auto-register output paths into spatial context for follow-up queries
        if trace.final_output:
            self._spatial["last_output"] = trace.final_output
            # Auto-bind by task heuristic
            for ex in getattr(trace, "executions", []):
                if "flood" in str(ex.step.args.get("task", "")) and ex.result.output_path:
                    self.bind("flood_mask", ex.result.output_path)
                elif "land_cover" in str(ex.step.args.get("task","")) and ex.result.output_path:
                    self.bind("land_cover_map", ex.result.output_path)

        if len(self._turns) >= self._max:
            self._turns.pop(0)
        self._turns.append(turn)
        return turn

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def last_turn(self) -> Turn | None:
        return self._turns[-1] if self._turns else None

    # ── Context for planner ────────────────────────────────────────────────────

    def context_for_planner(self) -> dict:
        """
        Returns a compact context dict that is safe to pass to the LLM planner
        (stays within prompt budget — max ~500 tokens).
        """
        ctx = dict(self._spatial)

        # Add last 3 turns as compressed history
        recent = self._turns[-3:] if self._turns else []
        if recent:
            ctx["conversation_history"] = [t.to_dict() for t in recent]

        # Add named bindings
        if self._bindings:
            ctx["named_outputs"] = {k: str(v) for k, v in self._bindings.items()}

        return ctx

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save session to a JSON file."""
        data = {
            "spatial": {k: str(v) for k, v in self._spatial.items()},
            "bindings": {k: str(v) for k, v in self._bindings.items()},
            "turns": [t.to_dict() for t in self._turns],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Session saved to %s", path)

    def load(self, path: str) -> None:
        """Restore session from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self._spatial  = data.get("spatial", {})
        self._bindings = data.get("bindings", {})
        # Real fix, confirmed necessary by direct inspection: this
        # previously never restored self._turns at all, despite the
        # log message below claiming "(%d turns)" were loaded -- a
        # saved conversation history was silently lost on every reload.
        # Turn.to_dict() itself doesn't save plan_summary/outputs, so
        # those two fields honestly can't round-trip from a saved file
        # and are left at their dataclass defaults rather than silently
        # fabricated.
        self._turns = [
            Turn(
                idx=t.get("turn", i), query=t.get("query", ""),
                timestamp=t.get("timestamp", ""), plan_steps=t.get("plan_steps", 0),
                planner_used=t.get("planner", ""), success=t.get("success", False),
                final_output=t.get("final_output") or None,
            )
            for i, t in enumerate(data.get("turns", []))
        ]
        logger.info("Session loaded from %s  (%d turns)", path, len(self._turns))

    def __repr__(self) -> str:
        return (
            f"GeoAgentMemory(turns={len(self._turns)}, "
            f"bbox={self._spatial.get('bbox')}, "
            f"date={self._spatial.get('date')})"
        )
