"""
pygeovision.agent.executor
===========================
Executes a ``Plan`` produced by the planner, resolving step
dependencies, substituting output references between steps,
and streaming progress events.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from pygeovision.agent.planner import Plan, Step
from pygeovision.agent.tools import GeoTool, ToolResult

logger = logging.getLogger("pygeovision.agent.executor")


# ── Execution result ───────────────────────────────────────────────────────────

@dataclass
class StepExecution:
    step: Step
    result: ToolResult


@dataclass
class ExecutionTrace:
    plan:           Plan
    executions:     list[StepExecution] = field(default_factory=list)
    total_duration: float = 0.0
    success:        bool  = False
    final_output:   Any   = None

    def summary(self) -> str:
        lines = [
            f"Execution summary ({len(self.executions)} steps, "
            f"{self.total_duration:.1f}s, {'✓ success' if self.success else '✗ failed'})"
        ]
        for ex in self.executions:
            status = "✓" if ex.result.success else "✗"
            lines.append(
                f"  {status} Step {ex.step.step_idx}: {ex.step.tool_name} "
                f"({ex.result.duration_s:.1f}s)"
            )
            if not ex.result.success:
                lines.append(f"    Error: {ex.result.error}")
            elif ex.result.output_path:
                lines.append(f"    → {ex.result.output_path}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_duration_s": round(self.total_duration, 2),
            "final_output": self.final_output,
            "steps": [
                {**ex.step.to_dict(), **ex.result.to_dict()}
                for ex in self.executions
            ],
        }


# ── Executor ───────────────────────────────────────────────────────────────────

class PlanExecutor:
    """
    Executes a Plan step by step.

    Parameters
    ----------
    tools : dict[str, GeoTool]
        All available tools (from ``build_tools()``).
    on_step_start : callable | None
        Callback ``fn(step_idx, tool_name, args)`` called before each step.
    on_step_done  : callable | None
        Callback ``fn(step_idx, result: ToolResult)`` called after each step.
    stop_on_failure : bool
        If True (default), stop execution when any step fails.
    """

    def __init__(
        self,
        tools: dict[str, GeoTool],
        on_step_start: Callable | None = None,
        on_step_done:  Callable | None = None,
        stop_on_failure: bool = True,
    ) -> None:
        self._tools           = tools
        self._on_start        = on_step_start
        self._on_done         = on_step_done
        self._stop_on_failure = stop_on_failure

    def execute(self, plan: Plan, context: dict | None = None) -> ExecutionTrace:
        """Execute the plan sequentially, resolving output references."""
        trace   = ExecutionTrace(plan=plan)
        outputs: dict[int, ToolResult] = {}   # step_idx → result
        ctx     = context or {}
        t_total = time.perf_counter()

        for step in plan.steps:
            logger.info("Step %d: %s  args=%s", step.step_idx, step.tool_name,
                        json.dumps(step.args, default=str)[:200])

            # Resolve $step_N_output and $context_* placeholders in args
            resolved_args = self._resolve_args(step.args, outputs, ctx)

            # Emit start event
            if self._on_start:
                try:
                    self._on_start(step.step_idx, step.tool_name, resolved_args)
                except Exception as exc:
                    logger.warning("on_start callback raised (continuing execution): %s", exc)

            # Find and call the tool
            tool = self._tools.get(step.tool_name)
            if tool is None:
                result = ToolResult(
                    tool=step.tool_name, success=False,
                    error=f"Unknown tool: '{step.tool_name}'",
                )
            else:
                try:
                    result = tool.run(**resolved_args)
                except Exception as exc:
                    result = ToolResult(
                        tool=step.tool_name, success=False, error=str(exc),
                    )

            outputs[step.step_idx] = result
            trace.executions.append(StepExecution(step=step, result=result))

            # Emit done event
            if self._on_done:
                try:
                    self._on_done(step.step_idx, result)
                except Exception as exc:
                    logger.warning("on_done callback raised (continuing execution): %s", exc)

            if not result.success and self._stop_on_failure:
                logger.warning("Step %d failed — stopping plan execution.", step.step_idx)
                break

        trace.total_duration = time.perf_counter() - t_total
        trace.success        = all(ex.result.success for ex in trace.executions)

        # Final output = last successful output_path or output dict
        last = [ex for ex in trace.executions if ex.result.success]
        if last:
            ex = last[-1]
            trace.final_output = ex.result.output_path or ex.result.output

        logger.info("Execution complete: %s  %.1fs", "✓" if trace.success else "✗",
                    trace.total_duration)
        return trace

    def stream(self, plan: Plan, context: dict | None = None) -> Iterator[dict]:
        """
        Generator version of ``execute`` that yields one event dict per step.

        Events have type: "step_start" | "step_done" | "execution_complete"
        """
        outputs: dict[int, ToolResult] = {}
        ctx     = context or {}
        t_total = time.perf_counter()
        execs   = []

        yield {"type": "plan", "steps": len(plan.steps), "planner": plan.planner}

        for step in plan.steps:
            resolved_args = self._resolve_args(step.args, outputs, ctx)
            yield {
                "type": "step_start",
                "step": step.step_idx,
                "tool": step.tool_name,
                "rationale": step.rationale,
            }

            tool = self._tools.get(step.tool_name)
            if tool is None:
                result = ToolResult(
                    tool=step.tool_name, success=False,
                    error=f"Unknown tool: '{step.tool_name}'",
                )
            else:
                try:
                    result = tool.run(**resolved_args)
                except Exception as exc:
                    result = ToolResult(
                        tool=step.tool_name, success=False, error=str(exc),
                    )

            outputs[step.step_idx] = result
            execs.append(StepExecution(step=step, result=result))

            yield {
                "type":   "step_done",
                "step":   step.step_idx,
                "tool":   step.tool_name,
                "success": result.success,
                "output_path": result.output_path,
                "error":  result.error,
                "duration_s": round(result.duration_s, 2),
            }

            if not result.success and self._stop_on_failure:
                break

        total_ok = all(ex.result.success for ex in execs)
        last_out = next((ex.result.output_path or ex.result.output
                         for ex in reversed(execs) if ex.result.success), None)

        yield {
            "type":           "execution_complete",
            "success":        total_ok,
            "total_duration": round(time.perf_counter() - t_total, 2),
            "final_output":   last_out,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _resolve_args(
        self,
        args:    dict[str, Any],
        outputs: dict[int, ToolResult],
        ctx:     dict,
    ) -> dict[str, Any]:
        """
        Replace placeholder strings in args:
          "$step_N_output"       → output_path (or output) of step N
          "$step_N_output.key"   → specific key in step N's output dict
          "$context_key"         → value from the context dict
        """
        def resolve_value(v: Any) -> Any:
            if not isinstance(v, str):
                return v

            # $step_N_output.key
            m = re.match(r"^\$step_(\d+)_output(?:\.(\w+))?$", v)
            if m:
                idx = int(m.group(1))
                key = m.group(2)
                if idx in outputs:
                    res = outputs[idx]
                    base = res.output_path or res.output
                    if key and isinstance(base, dict):
                        return base.get(key, base)
                    return base
                return v

            # $context_key
            if v.startswith("$context_"):
                ctx_key = v[len("$context_"):]
                return ctx.get(ctx_key, v)

            return v

        import re

        def deep_resolve(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: deep_resolve(resolve_value(vv)) for k, vv in obj.items()}
            if isinstance(obj, list):
                return [deep_resolve(item) for item in obj]
            return resolve_value(obj)

        return deep_resolve(args)