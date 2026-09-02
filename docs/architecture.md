# Architecture

This page describes what's actually verified about pygeovision's structure,
not an aspirational design document. Where something is confirmed real
and tested, that's stated plainly; where it isn't, that's stated too.

## The two-package split

PyGeoVision depends on [pygeofetch](https://pypi.org/project/pygeofetch/)
(a real, independent PyPI package, confirmed installed and inspected this
cycle — v2.6.2.1) for the data layer:

> **pygeofetch's job**: talk to every provider, know every file format,
> know the physics of every correction (radiometric, SAR, InSAR), and
> hand back either an analysis-ready result or an honest error.
>
> **PyGeoVision's job**: start where pygeofetch's output ends — AI
> inference, pipeline orchestration, labeling, and model-specific
> preprocessing.

This boundary is enforced in one direction more strictly than the other:
pygeovision should never re-implement signal processing pygeofetch
already does correctly. This cycle found and fixed a real violation of
that principle — see **SAR/InSAR removal** below.

## Package layout, by verification status

```
pygeovision/                    ~60,500 lines total across 25 modules
├── data/          11,600 lines  Thoroughly audited — real bugs found and fixed
├── ai/             14,450 lines  Thoroughly audited — real bugs found and fixed
│   ├── pipelines/               All 51 pipelines individually verified (see Pipelines page)
│   ├── models/                  Model registry/hub bridge, verified;
│   │                            121 models individually build/forward-pass tested
│   ├── labeling/                WorldCover verified; others not yet
│   └── inference/               Tiled inference — memory fixes verified
├── pipelines/          566 lines  Fixed this cycle — was a complete stale
│                                 duplicate module (see below); now a
│                                 genuine re-export of the real ai/pipelines classes
├── models/          9,190 lines  Detection (YOLO) verified this cycle;
│                                 rest not yet audited
├── agent/           2,330 lines  Planner/tools verified this cycle
│                                 (SAR/InSAR routing removed)
├── labeling/        2,480 lines  NOT audited — separate from ai/labeling/,
│                                 confirmed genuinely different code
├── cli/             2,290 lines  Only `channel` verified; ~25 other
│                                 command groups not yet audited
├── training/         2,240 lines  Not yet audited
├── inference/           700 lines  NOT audited — separate from
│                                 ai/inference/, confirmed different code,
│                                 different defaults (overlap=128 vs 64)
├── viz/             1,620 lines  Signatures confirmed real; rendering
│                                 output not verified
├── ... (15 more modules)         Not yet audited this cycle:
│                                 advanced, api, benchmark, cloud, core,
│                                 datasets, edge, enterprise,
│                                 explainability, losses, monitoring,
│                                 preprocess, serving, utils
```

## A real, confirmed pattern: duplicate parallel implementations

This audit repeatedly found the *same capability implemented twice*,
under different module paths, with only one side actually fixed and
tested. Confirmed instances:

| Capability | Audited, fixed version | Separate version |
|---|---|---|
| 10 core pipelines + all 41 domain pipelines | `pygeovision.ai.pipelines.*` | `pygeovision.pipelines.*` — **fixed this cycle**. Was a complete, independent, stale duplicate with pre-fix bugs still present (confirmed: the old NDVI-via-post_process bug, and the area-calculation bug, both still there). Deliberately public via `__all__` — `from pygeovision.pipelines import CarbonEstimationPipeline` was a real, reachable import silently returning the broken version, even though no internal code used it. Rewritten as a genuine re-export layer; both paths now point to the literal same class object (`is`, confirmed, not just `==`). |
| Tiled inference | `pygeovision.ai.inference.tiled_inference.TiledInference` (memory-efficient windowed reads, auto batch sizing) | `pygeovision.inference.tiled.TiledInference` (used by `client.inference.tiled()`, `client.detection.custom`) — different defaults confirm it's genuinely separate code, not a re-export. **Not fixed.** |
| ESA WorldCover labeling | `pygeovision.ai.labeling.esa_worldcover.ESAWorldCoverLabeler` (3 real bugs fixed: nodata handling, class-name mapping, tile-origin int/float promotion) | `pygeovision.labeling.landcover.ESAWorldCoverLabeler` (used by `client.labeling.esa_worldcover()`) — a genuinely separate class definition. **Not fixed.** |
| AI pipeline execution | `pygeovision channel <name>` CLI → `ai.pipelines.*Pipeline` classes | `client.pipeline(name)` — **not the same thing at all**; this is PyGeoFetch's chainable data-processing builder, not an AI pipeline runner |
| Model catalogs | `pygeovision.ai.models.registry.registry` (14 "native" models, tried first) | `pygeovision.models.registry.model_registry` (121 names, the real fallback) AND `pygeovision.ai.models.zoo.model_zoo` (98 `ModelSpec` entries, what `repr(client)` reports) — three overlapping catalogs, different naming conventions, not consolidated |

A comprehensive sweep (every module mirroring an `ai/` submodule name,
every one of 223 modules' imports, every one of 362 unique function-
level imports across the codebase) found no other instance at the
severity of the `pygeovision.pipelines` case — the others checked
(`models/architectures/*`, `monitoring/`, `training/`, `labeling/osm.py`)
are genuinely different, coexisting implementations, not abandoned
duplicates, though most remain individually unaudited.

**Practical rule**: before assuming a fix applies everywhere, grep for
the class/function name across the whole tree. If it appears in more
than one file, check whether they're the same object (`is`) or two
separate definitions.

## SAR/InSAR removal — the reasoning

PyGeoVision previously had its own SAR/InSAR processing layer
(`client.sar`, plus `pygeovision.insar` in an earlier snapshot). This was
removed entirely this cycle, for a specific, verified reason: pygeofetch's
real, installed `SARProcessor` has exactly the four methods
(`calibrate`, `coherence`, `despeckle`, `flood_map`) pygeovision's
wrapper delegated to — confirmed by inspecting the actual installed
package, not assumed. Pygeofetch also has a comprehensive real `insar.*`
suite (interferogram generation, unwrapping, coregistration, timeseries,
PS selection, offset tracking) that pygeovision never had an equivalent
of.

Removed:
- `client.sar` (the `_SARProxy` class and its construction)
- `pygeovision.data.processors.sar` (557 lines — including some real,
  documented bug fixes around georeference corruption and WGS84/UTM
  clip mismatches that pygeofetch's simpler processor doesn't cover;
  removed per explicit instruction, not because they were wrong)
- Three agent tools (`SARPreprocessTool`, `SARFloodTool`, `SLCInSARTool`)
  and all planner routing to them
- 8 test files covering a `pygeovision.insar` module that, at the time
  of this removal, did not exist in the delivered codebase at all

For SAR/InSAR work now, call [pygeofetch](https://pypi.org/project/pygeofetch/)
directly — see [PyGeoVision Client](api/pygeovision.md) for a real,
verified example. `oil_spill_detection` does exactly this (calls
`pygeofetch.processing.sar.SARProcessor` directly, real and tested).

## The pipeline catalog — every one of 51 resolved

`pygeovision.ai.pipelines.domains.list_pipelines()` reports 51
pipelines. Every one has been individually resolved this cycle:

- **10 original pipelines** — thoroughly audited, tested, and fixed
  (real radiometric scaling, cloud masking, bbox cropping, bi-temporal
  grid alignment).
- **13 real implementations** (were generic stubs) — real dNBR, NDSI,
  MNDWI, real Landsat thermal LST, real NDVI/NDCI/EVI computed
  directly, real bi-temporal WorldCover comparisons, real SAR dark-pixel
  detection via pygeofetch directly. Each verified against
  hand-calculated expected values.
- **12 honest, specific `NotImplementedError`s** (were generic stubs)
  — each with a genuinely distinct, individually-verified reason: no
  real data source exists (confirmed by search, not assumed), the data
  exists but the algorithm is too specialized to trust yet, or the
  need is fundamentally outside satellite imagery (real meteorological
  data for wind farm siting, for instance).
- **16 pipelines with a confirmed, now-fixed bug pattern** — real,
  dedicated classes that had a bare `except Exception:` silently
  substituting a wrong result (raw unprocessed imagery, or the output
  *directory itself*) while still claiming `success=True`. All 9
  instances of this pattern found and fixed — errors now propagate
  correctly. 4 more had a related but different bug: claimed specific
  indices (`["NDVI", "NDWI"]`) were computed via a `post_process`
  mechanism that never actually ran — converted to real, direct
  computation. 4 more returned misleading `success=True` despite an
  honest-sounding note admitting no real work happened — 3 became real
  subclasses reusing already-verified computations
  (`LandSurfaceTemperaturePipeline`/`ForestFirePipeline`/
  `VolcanoMonitoringPipeline` reuse `UrbanHeatIslandPipeline`'s/
  `WildfireSeverityPipeline`'s real logic), 1 now honestly returns
  `success=False`.

See [Pipelines](api/pipelines.md) for the full, named breakdown.

## Two more real bugs found and fixed this cycle

**Area/geodesic calculation** — a pattern like
`abs(src.res[0] * src.res[1]) / 10000` assumes raster resolution is in
meters. The real, production pipeline always outputs `EPSG:4326`
(degrees). This made area/density values wrong by roughly **10 billion
times** — confirmed directly: a real test scene produced "4.7 billion
vehicles per km²" before the fix. Found in 5 places (including
`CarbonEstimationPipeline`, one of the original 10 "gold standard"
pipelines) by grepping for the exact buggy pattern after finding it
once. Fixed with a real, geodesically-accurate helper
(`pygeovision.data.radiometric.real_pixel_area_ha`, using `pyproj.Geod`).

**Model device placement (`.to()`)** — `hub.load()` and the top-level
`get_model()` fallback both unconditionally assumed every model
supports `.to(device)`. Real wrapper classes (`CLIPGeo`, `TesseraGeo`,
`AlphaEarthGeo`, `MoondreamGeo`) either didn't implement it, or
genuinely have no local device-bound model at all (`TesseraGeo`/
`AlphaEarthGeo` are Google Earth Engine query services, not local
neural networks). Fixed both call sites — found independently, in two
separate rounds, since the first fix alone didn't resolve every
affected model.

## What "verified" means in this documentation

Consistently across this documentation:

- **Verified / audited / tested this cycle** — the specific claim was
  checked against the real, installed code (reading real signatures,
  or running real code against real or synthetic data with a
  hand-calculated expected result), and where a bug was found, it was
  fixed and regression-tested.
- **Not yet verified** — the code exists and may well work correctly,
  but this audit cycle did not independently confirm it. Treat it the
  way you'd treat any third-party code you haven't tested yourself.
- **Removed** — deleted from the codebase this cycle, with the reason
  stated.

This standard is applied unevenly on purpose: the data layer, the full
pipeline catalog (both `ai/pipelines/` and the `pygeovision.pipelines`
fix), the model registry's build/forward-pass behavior, and object
detection received this level of scrutiny this cycle. Most of the CLI,
the serving/edge/cloud/training layers, and roughly a third of the
model registry (network-blocked from full verification in this
sandbox) did not yet.
