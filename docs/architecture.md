# Architecture

How the package is organised, where each documented feature lives in the source, and which parts exist twice. Read it if you contribute,
if you import classes directly, or if a stack trace sends you into an unfamiliar folder. The counts below were taken from this release
and are reproducible with the commands at the end of the page.

## Two packages, one boundary

PyGeoVision depends on [pygeofetch](https://pygeofetch.readthedocs.io/en/latest/) for the **data layer** and does everything after it.

| | pygeofetch | PyGeoVision |
|---|---|---|
| Job | Talk to the 22 providers, know every file format, authenticate, correct radiometry (optical and SAR) | Choose scenes for an analysis, run it, label, train, infer, explain, monitor, report |
| Result | Files that are analysis-ready, or an honest error | Rasters, vectors, tables and reports with the numbers needed to judge them |
| You use it for | `pygeofetch search / download` on its own | `pygeovision channel …`, `client.pipeline(…)`, `GeoTrainer`, … |

The boundary is enforced in one direction: PyGeoVision calls pygeofetch; pygeofetch knows nothing of PyGeoVision. SAR/InSAR processing is
pygeofetch's job (`pygeofetch.insar`), and PyGeoVision's SAR flood pipelines use its radiometrically corrected scenes rather than repeating that work.
See [Data](data/index.md) for what each side does in practice.

## Package layout

About 240 Python files. Lines are physical lines including comments and docstrings.

| Package | Lines | Role | Documented in |
|---|---:|---|---|
| `ai/` | 18 600 | The AI layer: the 41 pipelines, training, tiled inference, labelers, native model registry | [Pipelines](pipelines/index.md), [Training](training/index.md), [Labeling](labeling/index.md) |
| `data/` | 12 800 | Scene choice, download, radiometry, reference products (WorldCover, JRC …), indices, post-processing, validation | [Data](data/index.md) |
| `models/` | 11 100 | The general model registry (79 entries), foundation-model wrappers, VLM, embeddings | [Models](models/index.md) |
| `reporting/` | 4 400 | Map, vector and PDF/Word/Excel deliverables | [Reporting](reporting/index.md) |
| `labeling/` | 3 500 | Standalone labeling module used by the CLI (OSM, buildings, WorldCover, SAM, quality, active learning) | [Labeling](labeling/index.md) |
| `cli/` | 3 100 | The 25 command groups and 92 commands | [CLI reference](reference/cli/index.md) |
| `advanced/` | 2 100 | Few-shot, multi-task, AutoML, vision-language, time series, point clouds | [Few-shot](classification/few-shot.md), [AutoML](experiments/automl.md), [Time series](timeseries/index.md), [Point clouds](lidar/index.md) |
| `obia/` | 2 000 | Object-based image analysis | [OBIA](obia/index.md) |
| `training/` | 1 700 | Callbacks, checkpoints, data, HPO, metrics, optimisers (the trainer itself is in `ai/training`) | [Training](training/index.md) |
| `datasets/` | 1 600 | A discovery catalogue of about 460 public datasets | [Datasets](datasets/index.md) |
| `preprocess/` | 1 300 | Composable raster preparation | [Preprocessing](data/preprocessing.md) |
| `explainability/` | 800 | Grad-CAM, uncertainty, SHAP, attention | [Explainability](explainability/index.md) |
| `inference/` | 850 | Tiled, batch and streaming inference used by the CLI | [Inference](inference/index.md) |
| `monitoring/` | 900 | Drift, performance, alerts, site tracking | [Monitoring](monitoring/index.md) |
| `pipelines/` | 800 | The YAML workflow engine | [YAML workflows](pipelines/yaml-workflows.md) |
| `losses/` | 700 | Segmentation, detection and class-balance losses | [Losses](training/losses.md) |
| `cloud/`, `edge/` | 600, 550 | AWS/Azure/GCP deployment; ONNX/Jetson | [Edge and cloud](deployment/index.md) |
| `benchmark/` | 450 | `ModelEvaluator`, leaderboards | [Benchmark](experiments/benchmark.md) |
| `core/` | 400 | Configuration and exceptions | [Config](reference/config.md), [Exceptions](reference/exceptions.md) |

### What a pipeline run touches

```text
pygeovision channel NAME --bbox … --date …
  cli/main.py              parse arguments, refuse removed / unknown names (exit 2)
  ai/pipelines/…           look the pipeline up; check checkpoint BEFORE any download (needs-weights ones)
  data/…                   search providers (pygeofetch), choose covering scenes, download, mosaic,
                           mask cloud (SCL), reproject to one UTM grid        -> rasters in memory
  the pipeline's own run() the documented method: an index, terrain, radar threshold, reference product …
  ai/pipelines/…           write rasters + run_summary.json
  reporting/…              map, vectors, PDF/Word/Excel (never turns an analysis into a failure)
  PipelineResult           success / error / stats / metadata / deliverables
```

## Systems that exist twice

A concept implemented in two places is the biggest source of "I fixed it and nothing changed". These pairs are real in this release
(checked by importing both and comparing the classes). When you import directly, pick the one in the **CLI** column to behave like the command line.

| Concept | CLI uses | The other one | Relationship |
|---|---|---|---|
| Trainer | `pygeovision.ai.training.trainer.GeoTrainer` | `pygeovision.training.GeoTrainer` | **The same class.** `pygeovision/training/trainer.py` re-exports it (the two diverging copies were merged) |
| Tiled inference | `pygeovision.inference.tiled.TiledInference` | `pygeovision.ai.inference.tiled_inference.TiledInference` | Different classes. The pipelines use the `ai` one; every CLI command uses the first |
| Model registry | `pygeovision.models.registry` (79 entries; `get_model`) | `pygeovision.ai.models.registry` (14 names; `ModelRegistry`) | Different purposes: architectures you can build vs. names your checkpoint is loaded into. A third, descriptive list is the zoo (96). See [model registries](models/registry.md) |
| Labeling | `pygeovision.labeling` (`label osm`, `label quality`) | `pygeovision.ai.labeling` (used inside pipelines) | Different classes with the same job; the foundation-model labelers use different models (DINOv2 clustering vs. Prithvi/RemoteCLIP) |
| Drift detection | `pygeovision.monitoring.drift.DriftDetector` | `pygeovision.ai.monitoring.DriftDetector` | Different classes |
| Post-processing | `pygeovision.data.postprocess.PostProcessor` (`client.postprocess`, CLI) | `pygeovision.ai.inference.postprocessing.PostProcessor` | Different classes |
| Data validation | `pygeovision.data.validator.DataValidator` (`client.validator`) | `pygeovision.data.ai_prep.DataValidator` | Different classes |

```{admonition} Rule when contributing
:class: important
A fix to one side of a pair **must be checked on the other**. Several serious bugs in this project's history (a dead augmentation pipeline,
a missing detection metric, a broken checkpoint resume) existed because a fix reached one copy. The trainer pair was merged for exactly this reason. Prefer
merging a pair (re-export) over fixing both.
```

## Conventions that hold across the package

These are enforced by tests, so you can rely on them.

- **A failed pretrained load raises** `ModelLoadError`. Nothing is substituted. Random weights appear only with an explicit `allow_random_init=True`,
  and such models carry `_pygeovision_is_random_init`.
- **A "needs weights" pipeline refuses to run without a checkpoint**, before it downloads anything (`ModelWeightsRequired`).
- **Multiband input** goes through `models/_adapt.py`; registry models are tensor-in, tensor-out (`models/_wrappers.py`).
- **Dense outputs stay registered with the raster.** Tiled or padded input, never resize-and-crop. Cell `(i, j)` of a feature grid covers exactly
  pixels `[i·p, (i+1)·p) × [j·p, (j+1)·p)`.
- **Nodata is excluded**, not counted as a class: from statistics, clustering and training labels, and written as the nodata value of
  the output (255 in class rasters, NaN in continuous ones). Class `0` is background.
- **Pipelines return failures; they do not raise them.** `PipelineResult(success=False, error=…)` always has a specific reason ([errors](reference/exceptions.md)).
- **Labeler contract.** `label_tile(tile_path, tile_metadata, output_path)` returns a `LabelingResult` (`tile_id, label_path, confidence, source,
  class_distribution, skipped, error, metadata`). Failures are returned, never raised, and a failed result has `label_path=None`. Vector-source labelers
  (OSM, Microsoft Buildings) use the geometry-only `label_tile(tile, output_dir, task)`; `BaseLabeler.label_tiles` adapts both.
- **Single source of truth** for shared code: CLIP/RemoteCLIP in `models/vlm/clip.py`, Moondream in `models/vlm/moondream.py` (the `advanced/vlm/*`
  and `models/embeddings.py` modules re-export them), Prithvi in `models/foundation/prithvi.py`, DINOv3/CHMv2 in `models/foundation/dinov3.py`, the
  shared image loader for vision-language models in `models/vlm/_image.py`.

## Three model lists

| List | Entries | Purpose | Browse |
|---|---:|---|---|
| Native registry | 79 | Architectures to build and train (`get_model`) | `pygeovision models list` |
| Legacy engine registry | 14 | Names that `infer`, `ai` and `channel` load **your** checkpoint into | `pygeovision ai-models list` |
| Model zoo | 96 | A descriptive catalogue; nothing is built from it | `pygeovision zoo list` |

The full table, with what each entry needs, is the generated [model catalogue](models/catalog.md).

## Reproduce these numbers

```bash
find pygeovision -name "*.py" | wc -l                                    # files
for d in pygeovision/*/; do echo "$d $(find $d -name '*.py' -exec cat {} + | wc -l)"; done
pygeovision channel list                                                 # pipelines and their kind
pygeovision models summary                                               # registry counts
```

## What was removed, and why

Five packages with no import anywhere (`api/`, `enterprise/`, `serving/`, `utils/`, `viz/`), a superseded data-access design, the natural-language agent
(a deliberate scope decision), and 21 pipelines that could not produce a correct result ([list](pipelines/removed.md)). The dated record is in the [roadmap](project/roadmap.md).
