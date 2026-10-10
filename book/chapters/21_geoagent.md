# Chapter 21: The AI Agent

## 21.1 What is the GeoAgent?

GeoAgent converts natural-language geospatial queries into complete pipelines —
sensor choice, task type, and tool sequence all decided from the problem description.

```python
import pygeovision as pgv
from pygeovision.agent import GeoAgent

client = pgv.PyGeoVision()
agent  = GeoAgent(client, output_dir="./results/", verbose=True)
agent.set_context(bbox=(-0.30, 5.50, -0.05, 5.70), date="2026-06")

trace = agent.run("Map current flood inundation in the Odaw River Basin")
print(trace.final_output)  # ./results/flood_mask.geojson
```

## 21.2 Decision Hierarchy

| Step | Decision | Signals |
|------|----------|---------|
| 1 | **Sensor** | "SAR"/"Sentinel-1"/"cloud" vs "Sentinel-2"/"optical" |
| 2 | **Task** | 14 types: flood, damage, oil_spill, land_cover, NDVI... |
| 3 | **Approach** | Sensor x Task -> exact tool sequence |

## 21.3 Different Problems, Different Plans

```python
# SAR + flood (cloud-independent)
agent.run("Map SAR flood — cloud cover 100%")
# Plan: sar_preprocess -> sar_flood_detection -> postprocess

# SAR + damage (earthquake keywords win)
agent.run("Detect building damage after earthquake using Sentinel-1")
# Plan: sar_preprocess -> change_detection(4-class) -> postprocess

# Optical + flood (explicit "Sentinel-2")
agent.run("Map flood using Sentinel-2 — clear sky today")
# Plan: search -> download -> prepare_for_ai -> prithvi(flood) -> postprocess

# Optical + land cover
agent.run("Classify land cover across Ethiopia with Prithvi")
# Plan: search -> download -> prepare_for_ai -> prithvi(land_cover) -> postprocess
```

## 21.4 Streaming

```python
for event in agent.stream("Compute NDVI for crop monitoring"):
    if event["type"] == "plan":
        print(f"Plan: {event['steps']} steps")
    elif event["type"] == "step_done":
        print(f"  [{event['step']}] {event['tool']} ({'OK' if event['success'] else 'FAIL'})")
    elif event["type"] == "execution_complete":
        print(f"Output: {event['final_output']}")
```

## 21.5 Heuristic vs LLM Planning

```python
# Heuristic (no API key needed) - covers ~80% of workflows
agent_h = GeoAgent(client, verbose=False)

# LLM (set ANTHROPIC_API_KEY) - handles complex/ambiguous queries
import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
agent_l = GeoAgent(client)  # auto-detects key -> uses claude-sonnet-4-6
```

## 21.6 Session Memory

```python
agent.set_context(bbox=BBOX, date="2026-06", output_dir="./accra/")
agent.bind("baseline_sar", "./sar/S1_dry_season.tif")

# Multi-turn
agent.run("Map flood extent using SAR")
agent.run("Now classify land cover for comparison")

# Persist
agent.save_session("./sessions/accra.json")
agent.load_session("./sessions/accra.json")
```

## 21.7 Available Tools

| Tool | Category | Function |
|------|----------|----------|
| search_satellite_data | data | 22+ provider search |
| download_satellite_data | data | Download + COG |
| prepare_for_ai | preprocessing | Stack/clip/normalise |
| sar_preprocess | SAR | S0-S9 pipeline |
| sar_flood_detection | SAR | Prithvi VH threshold |
| prithvi_inference | inference | Land cover/flood/crop/burn |
| change_detection | inference | ChangeFormer |
| compute_spectral_index | analysis | NDVI/NDWI/NDBI/EVI |
| postprocess | postprocessing | Sieve/vectorise/COG |
| run_pipeline | pipeline | Named pipelines |

## 21.8 Custom Tools

```python
from pygeovision.agent.tools import GeoTool, ToolResult, TOOL_REGISTRY

class PopulationRiskTool(GeoTool):
    name        = "population_at_risk"
    description = "Compute population at risk from flood mask + WorldPop"
    parameters  = [{"name":"flood_path","type":"str","required":True},
                    {"name":"pop_path",  "type":"str","required":True}]

    def run(self, flood_path, pop_path, **_):
        import rasterio, numpy as np
        with rasterio.open(flood_path) as s: mask = s.read(1)
        with rasterio.open(pop_path)   as s: pop  = s.read(1)
        at_risk = float((pop * mask).sum())
        return ToolResult(tool=self.name, success=True,
                           output={"population_at_risk": at_risk})

TOOL_REGISTRY["population_at_risk"] = PopulationRiskTool
```

## Summary

- GeoAgent: natural language -> sensor decision -> task decision -> tool sequence.
- Two planners: heuristic (zero dependencies) and LLM (Anthropic API key).
- Session memory persists spatial context across queries.
- Extend with custom tools by subclassing GeoTool.

## Exercises

1. Run the agent with 5 different queries. Verify sensor/task decisions.
2. Implement a custom tool that computes flooded road length.
3. Compare heuristic vs LLM plans for "assess storm damage in the coastal zone".
