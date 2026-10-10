# Architecture

This page describes pygeovision's real, current structure — not an
aspirational design document. Every number here was pulled directly
from the installed package, not estimated.

## The two-package split

PyGeoVision depends on [pygeofetch](https://pygeofetch.readthedocs.io/en/latest/)
for the entire data layer — search, download, authentication, and
optical/SAR radiometric correction. The division of responsibility is
real and enforced in one direction more strictly than the other:

> **pygeofetch's job**: talk to every provider, know every file
> format, apply every real physical correction (radiometric, SAR,
> InSAR), and hand back either an analysis-ready result or an honest
> error.
>
> **pygeovision's job**: everything after that — AI model inference,
> pipeline orchestration, labeling, and training.

This boundary was actively enforced during this project's audit, not
just assumed: real InSAR-processing code that had crept into
pygeovision's own scope was found and removed entirely, with real
pipelines pointed back at `pygeofetch.insar` directly rather than
duplicating logic pygeofetch already owns correctly.

## Package layout

```
pygeovision/                 ~58,900 lines across 19 real modules
├── ai/            17,402 lines  The core AI layer
│   ├── pipelines/               25 task pipelines (see AI Task Pipelines)
│   ├── models/                  The native, fully-offline model registry + ModelHub
│   ├── labeling/                ESA WorldCover, SAM, OSM, building-footprint labelers
│   ├── inference/                A real TiledInference engine
│   └── training/                A real GeoTrainer with detection + finetuning support
├── data/          11,593 lines  AOI coverage, band selection, radiometric prep
├── models/         9,744 lines  The general-purpose model registry (77 real entries)
├── cli/            2,571 lines  Every real command (see Full CLI Reference)
├── obia/           1,951 lines  Object-based image analysis: segmentation, object features, rule/fuzzy/supervised classification, accuracy (see OBIA)
├── training/       2,265 lines  A second, CLI-connected GeoTrainer (see Training)
├── labeling/       2,644 lines  A second, standalone labeling module (OSM, buildings, SAM, Label Studio)
├── advanced/       1,478 lines  Few-shot, AutoML, multi-task, time series, CLIP retrieval
├── preprocess/     1,239 lines  Composable raster preparation primitives
├── datasets/       1,517 lines  A 503-entry real dataset discovery catalog
├── monitoring/       824 lines  Real drift detection (PSI, KL divergence), real Kalman-filter site tracking
├── explainability/   672 lines  Grad-CAM
├── inference/         729 lines  A separate, CLI-reachable TiledInference engine
├── losses/            618 lines  Real, hand-verified segmentation/detection losses
├── cloud/             460 lines  AWS/GCP deployment
├── core/              402 lines  Configuration and exceptions
├── edge/              432 lines  ONNX Runtime inference
└── benchmark/         373 lines  ModelEvaluator + real leaderboards
```

```{note}
Five directories that appeared in earlier snapshots of this codebase
(`api/`, `enterprise/`, `serving/`, `utils/`, `viz/` — 4,619 lines) were
confirmed to have zero real imports anywhere and removed entirely
during this project's audit, along with a superseded data-access
design (`core/engine.py`) that duplicated what `data/acquire.py`
already does correctly. The natural-language agent module
(`pygeovision.agent`, 1,920 lines — a real, working LLM/heuristic
planner) was later removed as a deliberate scope decision, not a
dead-code finding; see [Roadmap](reference/roadmap.md) for the record.
```

## Genuinely duplicate systems — by design decision, not oversight

Several concepts in this codebase have two real, independent
implementations. This is documented explicitly here because it's the
single most common source of confusion (and, during this project's
audit, of real bugs — a fix applied to one side of a duplicate
sometimes didn't reach the other):

| Concept | Two real implementations | Which one the CLI uses |
|---|---|---|
| Training | `pygeovision.training.trainer.GeoTrainer` vs. `pygeovision.ai.training.trainer.GeoTrainer` | The first (`pygeovision.training`) — the second has real detection-task and checkpoint-finetuning support the first does not |
| Tiled inference | `pygeovision.inference.tiled.TiledInference` vs. `pygeovision.ai.inference.tiled_inference.TiledInference` | The first, via `infer predict`/`infer batch`; the second via `ai infer` |
| Model registry | `pygeovision.models.registry` (68 general-purpose entries) vs. `pygeovision.ai.models.registry` (14, fully offline-buildable) | `ModelHub.load()` checks the native one first, falls back to the general one |
| Labeling | `pygeovision.labeling` (standalone, CLI-reachable) vs. `pygeovision.ai.labeling` (used internally by pipelines) | Pipelines use `ai.labeling`; the CLI's `label osm`/`label quality` use the standalone one |

See [Training](training/index.md) and [Tiled Inference](inference/tiled-inference.md)
for the full, real detail on each pair — including two severe bugs
found during this audit specifically *because* a fix reached only one
side of a duplicate.

## Model coverage

- **77 real entries** in the general-purpose registry (down from an
  original 121 — 59 were confirmed to have no genuine working backing
  and were removed; 13 of those were later restored as honestly-failing,
  discoverable entries rather than left silently deleted, plus DOFA and
  LISAt added since — see
  [Model Registry](core-features/model-registry.md)).
- **14 real entries** in the fully-offline native registry, every one
  confirmed to build without a network dependency.
- **97 entries** in a separate, metadata-only discovery catalog
  (`pygeovision.ai.models.zoo`) with no build capability at all.

## What "verified" means on this site

Every real bug documented across this site was found the same way:
reading the actual code rather than its docstring, then testing the
actual behavior rather than assuming it matched the description. See
[Roadmap](reference/roadmap.md) for the complete, running record of
what that process found and fixed.
