"""
pygeovision.agent
==================
Autonomous Geospatial AI Agent for PyGeoVision.

Quick start::

    import pygeovision as pgv
    from pygeovision.agent import GeoAgent

    client = pgv.PyGeoVision()
    agent  = GeoAgent(client, output_dir="./results/")

    agent.set_context(bbox=[-0.30, 5.50, -0.05, 5.70], date="2026-06")
    trace = agent.run("Map current flood inundation in the Odaw River Basin")
    print(trace.final_output)   # → ./results/flood_mask.geojson

LLM planning (requires Anthropic API key)::

    import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    agent = GeoAgent(client)
    trace = agent.run("Compare deforestation in the Amazon 2020 vs 2024")

Heuristic planning (no API key — covers ~80% of common workflows)::

    agent = GeoAgent(client)  # auto-detects no key → heuristic mode
"""

from pygeovision.agent.core import GeoAgent
from pygeovision.agent.executor import ExecutionTrace, PlanExecutor, StepExecution
from pygeovision.agent.memory import GeoAgentMemory, Turn
from pygeovision.agent.planner import HeuristicPlanner, LLMPlanner, Plan, Step
from pygeovision.agent.tools import TOOL_REGISTRY, GeoTool, ToolResult, build_tools

__all__ = [
    "GeoAgent",
    "GeoTool", "ToolResult", "TOOL_REGISTRY", "build_tools",
    "Plan", "Step", "LLMPlanner", "HeuristicPlanner",
    "PlanExecutor", "ExecutionTrace", "StepExecution",
    "GeoAgentMemory", "Turn",
]
