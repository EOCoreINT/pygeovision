# pygeovision documentation

**The AI intelligence layer for geospatial data.** One CLI, one Python API,
49 real task pipelines, 77 real model architectures across two registries —
segmentation, detection, classification, change detection, and foundation
models, built directly on top of [pygeofetch](https://pygeofetch.readthedocs.io/en/latest/)
for search and download.

```{note}
This documentation is being rebuilt from source, following an extensive
audit of the installed package. That audit found and fixed a number of
real, previously-shipped bugs — a default learning-rate schedule that
silently oscillated 50x instead of decaying once, a band-count mismatch
that could feed the wrong number of channels into a model, several
labeling functions that reported success with zero real results found,
and roughly 60 model-registry entries with no real, working backing
(including some cases where a real, different model was silently loaded
under a name that didn't match it). All of that is fixed in the current
release; pages here are checked against `pygeovision`'s source directly,
and note the real, current status of anything that isn't fully resolved.
```

## Who this is for

- Geospatial engineers who need real, working model inference and
  training on satellite imagery without hand-building the preprocessing,
  band-selection, and tiling logic themselves
- ML researchers who want a real, honest model registry — every entry
  either has a genuine, verified path to real weights, or says clearly
  that it doesn't
- Teams building production remote-sensing pipelines who need real
  fallback behavior (a model that fails to load falls back to a real,
  verified alternative for the same task, not silence or a crash)

## Quick links

- [Quick Start (5 Minutes)](getting-started/quickstart.md) — install to your first real inference result
- [AI Task Pipelines](core-features/pipelines.md) — all 49 real pipelines, organized by domain
- [Model Registry](core-features/model-registry.md) — 77 real models across two registries, with an honest accounting of what was removed and why
- [Training & Finetuning](training/index.md) — real segmentation and object-detection training, real checkpoint-based finetuning
- [Full CLI Reference](reference/cli.md) — every real command group

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Core Features
:hidden:

core-features/pipelines
core-features/model-registry
core-features/aoi-coverage-and-bands
core-features/labeling
```

```{toctree}
:maxdepth: 2
:caption: Training & Finetuning
:hidden:

training/index
```

```{toctree}
:maxdepth: 2
:caption: Inference
:hidden:

inference/tiled-inference
inference/onnx-export
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/python-api
reference/cli
reference/pipeline-catalog
reference/model-catalog
reference/datasets
reference/model-evaluation
reference/agent
reference/advanced
reference/roadmap
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

contributing
```
