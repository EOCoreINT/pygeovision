# Errors and exceptions

PyGeoVision reports problems in four different ways, depending on where you are. Knowing which one to expect saves you from
catching the wrong thing.

| Where | How a problem shows up | What to do |
|---|---|---|
| `pygeovision channel …` (pipelines) | `✗` line with the reason, **exit code 1**; unknown or removed pipeline: exit code 2 | Read the message; most name the fix ([troubleshooting](../getting-started/troubleshooting.md)) |
| `client.pipeline(…)` in Python | **No exception.** `result.success` is `False` and `result.error` holds the reason | `if not result.success: …` ([reading results](../pipelines/results.md)) |
| Other CLI groups | `Error: …` and a non-zero exit code (click errors) | Fix the argument named in the message |
| Library classes (`GeoTrainer`, `TiledInference`, `GeoTimeSeries`, …) | A normal Python exception, usually `ValueError`, `FileNotFoundError`, `ImportError` or `RuntimeError` | Catch the built-in, or let it surface: the message is written to say what to change |

```{admonition} Pipelines return failures; they do not raise them
:class: note
The one exception inside the pipeline layer is `ModelWeightsRequired`, raised **before anything is downloaded** when a pipeline that
runs a neural network is given no `checkpoint`. In the CLI this becomes an ordinary error message.
```

## What is actually raised

A count taken from the source of this release (every `raise` statement in `pygeovision/`):

| Exception | Raise sites | Typical meaning |
|---|---:|---|
| `ValueError` | ~325 | A bad argument: unknown option value, shape mismatch, empty input, out-of-range number |
| `ImportError` | ~112 | An optional package is missing; the message names it and the `pip install` |
| `RuntimeError` | ~42 | Something failed at run time: a download, a model build, a step |
| `click.ClickException` and friends | ~45 | CLI-level errors with a clean message and a non-zero exit |
| `NotImplementedError` | ~26 | A registry entry that is listed but cannot be built, or an option that is not supported; the message says why |
| `FileNotFoundError` | ~13 | A path that does not exist |
| `KeyError`, `TypeError` | ~19 | Unknown names; wrong argument types |

Use this to decide what to catch: for a library call, `except (ValueError, FileNotFoundError, ImportError, RuntimeError)` covers
nearly every expected failure, and `NotImplementedError` covers "this model is listed only".

## PyGeoVision's own exception classes

`pygeovision.core.exceptions` defines a hierarchy rooted at `PyGeoVisionError`. Its `str()` includes an optional `details` text
(`message — details`).

```{important}
Only some of these are raised by the code today. The others are **reserved names**: you can catch them without error, but nothing
raises them, so `except TrainingError:` will not see a failed training run. The table marks which is which. Catch the base
class `PyGeoVisionError` for anything that is raised, and the built-ins above for everything else.
```

| Class | Parent | Raised today? | Raised by / meaning |
|---|---|---|---|
| `PyGeoVisionError` | `Exception` | base | Root of the hierarchy |
| `PyGeoVisionConfigError` | `PyGeoVisionError` | reserved | Invalid configuration (config errors currently surface as pydantic `ValidationError`) |
| `PyGeoVisionAuthError` | `PyGeoVisionError` | reserved | Provider authentication ([authentication](../data/authentication.md)) |
| `AIEngineError` | `PyGeoVisionError` | base | Root of the AI-layer errors |
| `AINotAvailableError` | `AIEngineError` | reserved | AI extras not installed (currently an `ImportError`) |
| `GPUError`, `OutOfMemoryError` | `AIEngineError` | reserved | Device errors (PyTorch's own `torch.OutOfMemoryError` is what you see) |
| `DatasetError` | `AIEngineError` | **yes** | `ai/data/dataloader.py` |
| `TilingError` | `DatasetError` | **yes** | The tiling engine (`ai/data/tiling.py`) |
| `PreprocessingError` | `DatasetError` | **yes** | Image preprocessing |
| `AugmentationError` | `DatasetError` | reserved | |
| `LabelingError` | `AIEngineError` | **yes** | The AI labelers (SAM, foundation labeler, …) |
| `OSMLabelingError` | `LabelingError` | **yes** | OpenStreetMap labeling |
| `BuildingFootprintError` | `LabelingError` | **yes** | Microsoft building footprints could not be retrieved |
| `SAMLabelingError`, `FoundationModelLabelingError` | `LabelingError` | reserved | |
| `ModelError` | `AIEngineError` | base | Root of model errors |
| `ModelLoadError` | `ModelError` | **yes** | A pretrained model failed to load. **Nothing is substituted**; pass `allow_random_init=True` only if you accept random weights ([models](../models/index.md)) |
| `ModelNotFoundError`, `ModelDownloadError`, `ModelExportError`, `ArchitectureError` | `ModelError` | reserved | (registry misses raise `KeyError`/`ValueError`) |
| `TrainingError`, `CheckpointError`, `DistributedTrainingError`, `LossError` | `AIEngineError` | reserved | (training raises `ValueError`/`RuntimeError`) |
| `InferenceError`, `TiledInferenceError`, `PostProcessingError`, `VectorizationError` | `AIEngineError` | reserved | |
| `PipelineError`, `PipelineConfigError`, `PipelineStepError` | `AIEngineError` | reserved | (pipelines return `PipelineResult(success=False)`) |
| `MonitoringError`, `DriftDetectionError` | `AIEngineError` | reserved | |
| `ExperimentError` | `AIEngineError` | reserved | |

## Exceptions defined next to the code that raises them

| Class | Parent | Where | Meaning and what to do |
|---|---|---|---|
| `ModelWeightsRequired` | `RuntimeError` | `pygeovision.ai.pipelines` | A "needs weights" pipeline was run without `checkpoint=`. Train a model and pass it ([training](../training/index.md)) |
| `PrithviLoadError` | `ModelLoadError`, `RuntimeError` | `models/foundation/prithvi.py` | The Prithvi code or weights could not be loaded; the message gives the path/Hub id to supply |
| `ReferenceDataError` | `RuntimeError` | `data/reference.py` | A reference product (WorldCover, JRC, Dynamic World …) could not be read for the box |
| `FootprintSourceError` | `RuntimeError` | `data/footprints.py` | A building-footprint source failed or has no coverage |
| `ValidationError` | `ValueError` | `data/validator.py` | `report.raise_if_failed()` found a problem ([validation](../data/validation.md)) |
| `StepError` | `RuntimeError` | `pipelines/steps.py` | A step of a [YAML pipeline](../pipelines/yaml-workflows.md) failed |
| `ReportingUnavailable` | `RuntimeError` | `reporting/build.py` | A reporting dependency is missing; the analysis itself is not affected |

## Writing handlers

```python
from pygeovision import PyGeoVision
from pygeovision.core.exceptions import PyGeoVisionError, ModelLoadError
from pygeovision.models import get_model

client = PyGeoVision()

# 1. Pipelines: inspect the result, do not wrap in try/except.
result = client.pipeline("water_bodies", bbox=(-0.15, 51.47, -0.10, 51.52), date="2024-06", output_dir="./run")
if not result.success:
    print("failed:", result.error)

# 2. Library calls: built-ins plus the package base class.
try:
    model = get_model("prithvi_eo_2_0", num_classes=2, in_channels=6, pretrained=True)
except ModelLoadError as exc:       # no network, gated repo, bad weights: nothing was substituted
    print(exc)
except PyGeoVisionError as exc:     # any other exception of the package's own hierarchy
    print(exc)
except (ValueError, ImportError, NotImplementedError) as exc:
    print(exc)
```

## What to do when…

| You see | Meaning | Action |
|---|---|---|
| `ImportError: … pip install …` | An optional package is not installed | Install the named package; [installation](../getting-started/installation.md) lists the extras |
| `ModelLoadError` | A pretrained model could not be loaded | Check the network and licence/gating of the Hub repository; or supply a local path. Do not set `allow_random_init=True` unless you want random weights |
| `NotImplementedError: … listed only` | The model is in the catalogue for discovery but cannot be built | Pick a buildable entry from the [catalogue](../models/catalog.md) |
| `ModelWeightsRequired` | The pipeline runs your network | Pass `--checkpoint` ([pipelines](../pipelines/index.md)) |
| `ValueError` with a number or shape in it | An argument does not fit the data | The message names the argument; compare with the page of the tool |
| `result.success` is `False` | The analysis could not be done honestly (no imagery, too much cloud …) | `result.error`; widen the dates, see the pipeline page |
