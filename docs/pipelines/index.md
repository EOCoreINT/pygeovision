# AI Task Pipelines — Overview

49 real, working pipelines. **Two genuinely different real
implementations**, not one shared architecture — this matters enough
to say before anything else on this page:

```{important}
**10 pipelines** (`change_detection`, `land_cover`,
`building_footprints`, `crop_monitoring`, `disaster_assessment`,
`deforestation`, `urban_growth`, `water_bodies`, `solar_detection`,
`carbon_estimation`) are the only ones reachable through the CLI's
`channel` command, and they run a **separate, real implementation**
(`pygeovision.ai.pipelines`) with its own real helper methods,
real historical bug fixes, and a `PipelineResult.pipeline` field
(not `.name`). See
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md) for full,
dedicated detail — the shared-architecture description on the rest of
this page describes the **other 39**, Python-API-only pipelines in
`pygeovision.ai.pipelines.domains`.
```

The other 39 are organized into 9 domains below. Every one of *those*
shares the same real internal architecture, described once here rather
than repeated on every page.

## The shared pipeline lifecycle (the 39 Python-API-only pipelines)

Every one of these 39 pipelines' `run()` method follows the same four real stages:

1. **Search & AOI coverage** — finds real scenes covering your
   requested bbox via `pygeofetch`. If one scene doesn't fully cover
   the request, a real greedy set-cover selects the minimal set of
   scenes needed and mosaics them (see
   [AOI Coverage & Band Selection](../core-features/aoi-coverage-and-bands.md)).
2. **Radiometric preparation** — real per-scene atmospheric/radiometric
   correction happens *before* mosaicking, not after, to avoid a real
   sensor-calibration discontinuity at the seam between merged scenes.
3. **Model or band-math execution** — either a real model from the
   [registry](../core-features/model-registry.md), with a real,
   task-aware fallback if the requested one fails to build, or a real,
   documented spectral-index formula for pipelines that don't need a
   trained model at all (e.g. `vegetation_indices`).
4. **Result packaging** — always returns the same real
   `PipelineResult` shape:

```python
@dataclass
class PipelineResult:
    name: str
    success: bool
    output_path: Path | None = None
    stats: dict = field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0
```

`success=False` always means a real, specific check failed — never
network flakiness silently swallowed into a fabricated result. Read
`result.error` for what actually went wrong.

## Three real, deliberate absences

- **`permafrost_thaw`, `dam_safety`** — removed entirely. Both required
  real InSAR displacement processing, which is out of `pygeovision`'s
  scope (`pygeofetch` has its own, real, separate InSAR module for
  this — `pygeofetch.insar`).
- **`air_quality_index`** — never implemented. The same visible-haze
  proxy technique used for `dust_storm_tracking` would risk being read
  as calibrated health guidance under this specific name, and no real,
  calibrated atmospheric-composition data source exists in this
  codebase's real dependencies.

## The nine domains

```{toctree}
:maxdepth: 1

cli-reachable-pipelines
agriculture
forestry
infrastructure
water-and-disasters
urban
change-detection
cryosphere-and-climate
heritage-and-archaeology
environment
```

## A note on honesty markers throughout

Several pipeline pages carry explicit ⚠️ warnings. These aren't hedging
— they mark the real, specific difference between "this is a
calibrated measurement" and "this is a real, useful, but relative or
uncalibrated proxy." Both kinds of pipeline are genuinely useful; the
warnings exist so you know which kind you're looking at before you act
on a result.
