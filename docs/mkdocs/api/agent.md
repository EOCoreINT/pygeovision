# GeoAgent API Reference

GeoAgent accepts natural-language geospatial queries and autonomously executes
the complete workflow — sensor choice, task type, and tool sequence all decided
from the problem description alone.

## Quick Start

```python
import pygeovision as pgv
from pygeovision.agent import GeoAgent

client = pgv.PyGeoVision()
agent  = GeoAgent(client, output_dir="./results/")
agent.set_context(bbox=[-0.30, 5.50, -0.05, 5.70], date="2026-06")
trace = agent.run("Map current flood inundation in the Odaw River Basin")
print(trace.final_output)   # ./results/flood_mask.geojson
```

## Decision Hierarchy

| Step | Decision | Signals | Effect |
|------|----------|---------|--------|
| 1 | **Sensor** | "SAR" / "Sentinel-1" vs "Sentinel-2" / "optical" | SAR or optical tools |
| 2 | **Task** | 14 task classes | Correct model |
| 3 | **Approach** | Sensor × Task | Exact tool sequence |

## Tools (10)

| Tool | Category | Description |
|------|----------|-------------|
| `search_satellite_data` | data | Search 22+ providers |
| `download_satellite_data` | data | Download + COG |
| `prepare_for_ai` | preprocessing | Stack / clip / normalise |
| `sar_preprocess` | SAR | Full S0-S9 pipeline (3 bug-fixes) |
| `sar_flood_detection` | SAR | Prithvi zero-shot VH threshold |
| `prithvi_inference` | inference | Land cover / flood / crop / burn |
| `change_detection` | inference | ChangeFormer + co-registration |
| `compute_spectral_index` | analysis | NDVI / NDWI / NDBI / EVI |
| `postprocess` | postprocessing | Sieve → vectorise → COG |
| `run_pipeline` | pipeline | Named end-to-end pipelines |
