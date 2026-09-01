# GeoAgent — Natural-Language Task Planning

`GeoAgent` accepts a natural-language geospatial query and executes a real
tool sequence to answer it — search, download, preprocess, run a model,
postprocess — deciding the sequence from the query text alone.

**Verification status**: real, tested this cycle (all of `agent/planner.py`,
`agent/tools.py`, and their SAR/InSAR routing removal were directly audited
and regression-tested — 974 passed, 0 failed). `GeoAgent.run()` itself
(the orchestration in `agent/core.py`, `agent/executor.py`) has not been
separately exercised against a real end-to-end query this cycle — the
planner and tool registry it depends on have been.

## Quick Start

```python
import pygeovision as pgv
from pygeovision.agent import GeoAgent

client = pgv.PyGeoVision()
agent  = GeoAgent(client, output_dir="./results/")
agent.set_context(bbox=[-0.30, 5.50, -0.05, 5.70], date="2026-06")
trace = agent.run("Map current flood extent in the Odaw River Basin")
print(trace.final_output)
```

## `GeoAgent(pgv_client, **kwargs)`

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pgv_client` | `PyGeoVision` | required | An initialised client |
| `api_key` | `str \| None` | `None` | Anthropic API key; falls back to `ANTHROPIC_API_KEY` env var. Without one, uses the heuristic (non-LLM) planner |
| `model` | `str \| None` | `None` | Claude model for LLM planning |
| `provider` | `str \| None` | `None` | `"anthropic"` \| `"groq"`, auto-detected if unset |
| `stop_on_failure` | `bool` | `True` | Stop execution when a step fails |
| `verbose` | `bool` | `True` | Print step progress |
| `output_dir` | `str` | `"./agent_output/"` | Default output location |

### `set_context(**kwargs) -> GeoAgent`

Sets spatial/temporal context for subsequent queries. Common keys:

```python
agent.set_context(
    bbox=[lon_min, lat_min, lon_max, lat_max],   # WGS84
    date="2026-06",                               # YYYY-MM or YYYY-MM-DD
    output_dir="./my_results/",
    bands=["B02", "B03", "B04", "B08", "B11", "B12"],
    providers=["planetary_computer"],
)
```

### `bind(name, value) -> GeoAgent`

Binds a named output (e.g. a file path) so a later query in the same session can reference it.

### `run(query, context_override=None) -> ExecutionTrace`

Executes a natural-language query end-to-end. `context_override` applies a one-shot context just for this call, without changing the session context. Returns an `ExecutionTrace` — use `.summary()` for a text summary, `.final_output` for the primary output path, `.to_dict()` for JSON.

## Decision Hierarchy (heuristic planner)

With no Anthropic API key, `HeuristicPlanner` reasons through two decisions:

1. **Task** — which of the supported task types the query maps to (flood, damage, change, land cover, crop, burn, spectral index, building footprints, and more)
2. **Approach** — the tool sequence that delivers that task

Note: an earlier version of this planner also made a sensor decision (SAR vs. optical) with SAR-specific tool variants. That routing has been removed along with pygeovision's own SAR/InSAR processing layer — see [Architecture](../architecture.md). All planning now routes through the optical path; for SAR/InSAR work, call [pygeofetch](https://pypi.org/project/pygeofetch/) directly rather than through the agent.

## Tools (8)

| Tool | Category | Description |
|---|---|---|
| `search_satellite_data` | data | Search across configured providers |
| `download_satellite_data` | data | Download + post-process (reproject, COG) |
| `prepare_for_ai` | preprocessing | Stack bands / clip / normalise for model input |
| `prithvi_inference` | inference | Land cover / flood / crop / burn via Prithvi-EO-2.0 |
| `change_detection` | inference | Bi-temporal change detection |
| `compute_spectral_index` | analysis | NDVI / NDWI / NDBI / EVI |
| `postprocess` | postprocessing | Sieve → vectorise → COG |
| `run_pipeline` | pipeline | Named end-to-end pipelines — see [Pipelines](pipelines.md) for which of the 51 registered names are actually verified |

To add a new tool: subclass `GeoTool` in `agent/tools.py`, then register it in `TOOL_REGISTRY` at the bottom of that file.
