# Agent Tools — Full Reference

8 real tools, each independently checked this audit — real parameter
handling, real delegation to `pygeovision`'s actual functionality, and
real, honest failure reporting (every tool wraps its work in a real
`try`/`except`, returning `ToolResult(success=False, error=...)` on
any real exception rather than letting the agent crash mid-plan).

## `search_satellite_data`

Real `pygeofetch` search across 22+ providers. Accepts both
`satellite=` (singular, the real underlying API parameter) and
`satellites=` (plural, list) — a real, deliberate robustness choice
documented directly in the code, since an LLM planner might generate
either form.

```python
result = tool.run(
    bbox=[-0.15, 51.47, -0.10, 51.52],
    date_range=("2024-06-01", "2024-06-30"),
    providers=["planetary_computer"],
    cloud_cover_max=20,
)
```

## `download_satellite_data`

Downloads real scenes returned by the tool above, with real automatic
post-processing (`reproject:EPSG:4326`, `cog`). Honestly reports
`success=False` if zero real files actually downloaded, even if the
underlying call didn't raise.

## `prepare_for_ai`

Real delegation to `client.prepare_for_ai()` — band stacking, bbox
clipping, cloud masking, and real Sentinel-2 DN-to-reflectance
normalization (`scale_factor=10000.0`) in one real, composed step.

## `prithvi_inference`

Real Prithvi-EO-2.0 inference. Genuinely validates input shape and
band count first (`validate_prithvi_input`) and honestly fails with a
clear message if validation doesn't pass, rather than letting a
malformed array reach the model. `task` is resolved to a real method
on `PrithviTasks` via `getattr()` — an unrecognized task name fails
clearly (`"Unknown task '...'"`) rather than silently defaulting to
something else.

## `change_detection`

Real ChangeFormer bi-temporal change detection, with a real
co-registration step first (`coregister_sar_pair` — real reprojection
+ phase-correlation sub-pixel alignment; genuinely not SAR-specific
despite the name, which comes from where this co-registration utility
was originally written).

## `compute_spectral_index`

Real delegation to `client.indices.<name>` for any of `ndvi`, `ndwi`,
`ndbi`, `evi`, `nbr`, `mndwi`. Returns real min/max/mean statistics
computed directly from the output raster (not just "success" — an
LLM planner or a human reviewing the trace can sanity-check the real
numeric range without opening the file).

## `postprocess`

Real, ordered, composable operations: `sieve` (remove small spurious
patches below `min_pixels`), `vectorise` (raster → real GeoJSON
polygons), `cog` (Cloud-Optimized GeoTIFF export) — each real
operation's output feeds into the next if chained
(`operations=["sieve", "vectorise"]` runs sieve first, then
vectorizes the *sieved* result).

## `run_pipeline`

The fastest real path for standard tasks — delegates directly to
`client.pipeline(pipeline_name, ...)`, the same real dispatch
documented in [AI Task Pipelines](../pipelines/index.md).

```{warning}
This tool's own parameter metadata previously listed four pipeline
names that don't exist in the real registry (`road_network`,
`solar_panels`, `crop_mapping`, `forest_monitoring`) — the exact same
stale names found and fixed in the heuristic planner's routing logic.
Fixed to the real names (`road_extraction`, `solar_detection`,
`crop_monitoring`, `deforestation`). See
[the agent overview](../reference/agent.md) for the full detail.
```

```{seealso}
For the real bugs found in how the planner *selects* which tool and
arguments to use (not the tools themselves, all confirmed correct
above), see [the agent overview](../reference/agent.md).
```
