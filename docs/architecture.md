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
│   ├── pipelines/               10 real pipelines (see Pipelines page)
│   ├── models/                  Model registry/hub bridge, verified
│   ├── labeling/                WorldCover verified; others not yet
│   └── inference/               Tiled inference — memory fixes verified
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
├── ... (16 more modules)         Not yet audited this cycle:
│                                 advanced, api, benchmark, cloud, core,
│                                 datasets, edge, enterprise,
│                                 explainability, losses, monitoring,
│                                 pipelines, preprocess, serving, utils
```

## A real, confirmed pattern: duplicate parallel implementations

This audit repeatedly found the *same capability implemented twice*,
under different module paths, with only one side actually fixed and
tested. Confirmed instances:

| Capability | Audited, fixed version | Separate, unaudited version |
|---|---|---|
| Tiled inference | `pygeovision.ai.inference.tiled_inference.TiledInference` (memory-efficient windowed reads, auto batch sizing) | `pygeovision.inference.tiled.TiledInference` (used by `client.inference.tiled()` and `client.detection.custom`) — different defaults confirm it's genuinely separate code, not a re-export |
| ESA WorldCover labeling | `pygeovision.ai.labeling.esa_worldcover.ESAWorldCoverLabeler` (3 real bugs fixed: nodata handling, class-name mapping, tile-origin int/float promotion) | `pygeovision.labeling.landcover.ESAWorldCoverLabeler` (used by `client.labeling.esa_worldcover()`) — a genuinely separate class definition |
| AI pipeline execution | `pygeovision channel <name>` CLI → `ai.pipelines.*Pipeline` classes | `client.pipeline(name)` — **not the same thing at all**; this is PyGeoFetch's chainable data-processing builder, not an AI pipeline runner |

**Practical implication**: fixes made to one side of a pair do not apply
to the other. If you're using `client.labeling`/`client.inference`
directly rather than the `channel` CLI or `ai.pipelines` classes, treat
that code path as unaudited even where a same-named sibling has been
fixed.

## SAR/InSAR removal — the reasoning

PyGeoVision previously had its own SAR/InSAR processing layer
(`client.sar`, plus `pygeovision.insar` in an earlier snapshot). This was
removed entirely this cycle, for a specific, verified reason: pygeofetch's
real, installed `SARProcessor` has exactly the four methods
(`calibrate`, `coherence`, `despeckle`, `flood_map`) that pygeovision's
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
verified example.

## The pipeline catalog — a confirmed, significant gap

`pygeovision.ai.pipelines.domains.list_pipelines()` reports 51 pipelines.
Tracing every one to its real implementation:

- **10 real, task-specific pipelines** — thoroughly audited, tested, and
  fixed this cycle (real radiometric scaling, cloud masking, bbox
  cropping, bi-temporal grid alignment).
- **16 pipelines with dedicated classes** — real code, task-specific
  logic, not yet independently verified. One (`crop_type_mapping`) has a
  confirmed bug: it silently returns the raw, unprocessed input image as
  the "result" if its default model isn't registered, with
  `success=True` and no visible error.
- **26 pipelines built from a generic factory** (`_make_simple`) that
  only searches, downloads one scene, and validates it — despite
  specific-sounding descriptions ("SAR oil slick detection via adaptive
  backscatter threshold," "dNBR burn severity mapping"), these run no
  task-specific model or algorithm at all.

See [Pipelines](api/pipelines.md) for the full, named breakdown.

## What "verified" means in this documentation

Consistently across this documentation:

- **Verified / audited / tested this cycle** — the specific claim was
  checked against the real, installed code (reading real signatures,
  or running real code against real or synthetic data), and where a bug
  was found, it was fixed and regression-tested.
- **Not yet verified** — the code exists and may well work correctly,
  but this audit cycle did not independently confirm it. Treat it the
  way you'd treat any third-party code you haven't tested yourself.
- **Removed** — deleted from the codebase this cycle, with the reason
  stated.

This standard is applied unevenly on purpose: roughly 26,000 of the
package's ~60,500 lines (`data/`, `ai/`) received this level of scrutiny
this cycle. The rest did not yet.
