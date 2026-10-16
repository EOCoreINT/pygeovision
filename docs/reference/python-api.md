# Python API

Everything the command line does is available from Python, and some things (training, custom models, the advanced tools) are
**only** available there. This page maps the `PyGeoVision` client to the guide pages that explain each part. The **signatures and docstrings of every
public class and function** are on the pages listed under [The API pages](#the-api-pages), one per area of the package.

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
```

```{admonition} Two ways in
:class: tip
- **The client** (`PyGeoVision()`): one object with the layers below as attributes. Convenient in a notebook or a script; heavy parts load only
  when you first use them.
- **The classes**: import and use directly (`from pygeovision.training import GeoTrainer`). Use this in libraries and tests.

Both reach the same code. The client's proxies add a few shortcuts (for example `client.pipeline(name, …)`).
```

## Map of the client

| Attribute / method | What it is | Explained in |
|---|---|---|
| `client.search(…)`, `client.download(…)`, `client.data.*` | Find and fetch imagery from the 22 providers | [Data](../data/index.md) |
| `client.pipeline(name, bbox=…, …)` | Run one of the 41 pipelines; returns a `PipelineResult` | [Pipelines](../pipelines/index.md), [results](../pipelines/results.md) |
| `client.create_pipeline`, `client.run_pipeline_yaml` | Your own data → AI workflow in YAML | [YAML pipelines](../pipelines/yaml-workflows.md) |
| `client.validator` | `DataValidator`: check a raster before a model sees it | [Validation](../data/validation.md) |
| `client.preprocess` | `Preprocessor`: stack, clip, mosaic, mask, normalise, resample | [Preprocessing](../data/preprocessing.md) |
| `client.indices` | `SpectralIndices`: 21 indices and band maths | [Indices](../data/indices.md) |
| `client.postprocess` | `PostProcessor`: vectorise, sieve, smooth, zonal stats, accuracy, export | [Post-processing](../data/postprocessing.md) |
| `client.labeling` | OSM, buildings, WorldCover, Dynamic World, SAM, foundation labelers; quality; active learning | [Labeling](../labeling/index.md) |
| `client.losses` | Segmentation and detection losses | [Losses](../training/losses.md) |
| `client.segmentation`, `.detection`, `.change`, `.classification` | Run a model on a file you already have | [Inference](../inference/index.md) |
| `client.inference` | Tiled, batch, streaming and ensemble inference | [Inference](../inference/index.md) |
| `client.xai` | Grad-CAM, uncertainty, SHAP, attention | [Explainability](../explainability/index.md) |
| `client.monitoring` | Drift, performance tracking, alerts | [Monitoring](../monitoring/index.md) |
| `client.edge`, `client.cloud` | ONNX / Jetson, AWS / Azure / GCP | [Edge and cloud](../deployment/index.md) |
| `client.few_shot`, `.multitask`, `.automl`, `.vlm`, `.timeseries`, `.pointcloud` | The advanced tools | [Few-shot](../classification/few-shot.md), [AutoML](../experiments/automl.md) |
| `client.status()`, `client.doctor()` | What is installed and what works | [Installation](../getting-started/installation.md) |
| `pgv.model_zoo`, `pgv.dataset_registry` | Metadata catalogues | [Models](../models/catalog.md), [Datasets](../datasets/index.md) |

Objects not on the client: the trainer (`pygeovision.training`), the model registry (`pygeovision.models.get_model`),
object-based analysis (`pygeovision.obia`), the benchmark evaluator (`pygeovision.benchmark`) and the report builder (`pygeovision.reporting`).

## The API pages

Each page below is generated from the source code (`tools/generate_api_docs.py`): a table of what the module defines, and then the
signature and docstring of each class and method. A test fails when a public name is missing from every page, so this list cannot
quietly fall behind the package.

```{include} api/_index_table.md
```

## How to read a signature

```python
GeoTrainer(model, config, *, callbacks=None, device=None)
```

- Arguments before the `*` are positional or by name; those after it are by name only.
- `x=None` means the argument is optional and `None` selects the default behaviour described in its docstring.
- A method listed with `->` returns that type. `PipelineResult` and `BenchmarkResult` are dataclass-style records; read their fields
  by attribute.
- Names starting with an underscore are private and may change without notice. The client's layer classes
  (`_SegmentationClientProxy` and the like) are the one exception: they are private *names* for public *attributes* (`client.segmentation`),
  so they are documented under [The client](api/client.md).

## Duplicate classes by design

Several concepts exist twice (trainer, tiled inference, drift detector, labelers). The pages above for [training](api/training.md),
[inference](api/inference.md) and [monitoring](api/monitoring.md) are the classes the **command line** uses; the other implementation
is on [the legacy engine page](api/engine.md). Why it is kept is described in
[Architecture](../architecture.md#systems-that-exist-twice).
