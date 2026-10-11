# Python API

Everything the command line does is available from Python, and some things (training, custom models, the advanced tools) are
**only** available there. This page maps the `PyGeoVision` client to the pages that explain each part, and then gives the signatures
and docstrings of the main classes, rendered from the source.

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
| `client.pipeline(name, bbox=…, …)` | Run one of the 40 pipelines; returns a `PipelineResult` | [Pipelines](../pipelines/index.md), [results](../pipelines/results.md) |
| `client.create_pipeline`, `client.run_pipeline_yaml` | Your own data → AI workflow in YAML | [YAML pipelines](../pipelines/yaml-workflows.md) |
| `client.validator` | `DataValidator`: check a raster before a model sees it | [Validation](../data/validation.md) |
| `client.preprocess` | `Preprocessor`: stack, clip, mosaic, mask, normalise, resample | [Preprocessing](../data/preprocessing.md) |
| `client.indices` | `SpectralIndices`: 21 indices and band maths | [Indices](../data/indices.md) |
| `client.postprocess` | `PostProcessor`: vectorise, sieve, smooth, zonal stats, accuracy, export | [Post-processing](../data/postprocessing.md) |
| `client.labeling` | OSM, buildings, WorldCover, Dynamic World, SAM, foundation labelers; quality; active learning | [Labeling](../labeling/index.md) |
| `client.losses` | Segmentation and detection losses | [Losses](../training/losses.md) |
| `client.segmentation`, `.detection`, `.change`, `.classification` | Run a model on a file you already have | [Direct inference](../cli/direct-inference.md) |
| `client.inference` | Tiled, batch, streaming and ensemble inference | [Inference](../inference/index.md) |
| `client.xai` | Grad-CAM, uncertainty, SHAP, attention | [Explainability](../explainability/index.md) |
| `client.monitoring` | Drift, performance tracking, alerts | [Monitoring](../monitoring/index.md) |
| `client.edge`, `client.cloud` | ONNX / Jetson, AWS / Azure / GCP | [Edge and cloud](../edge-cloud/index.md) |
| `client.few_shot`, `.multitask`, `.automl`, `.vlm`, `.timeseries`, `.pointcloud` | The advanced tools | [Advanced](../advanced/index.md) |
| `client.status()`, `client.doctor()` | What is installed and what works | [Installation](../getting-started/installation.md) |
| `pgv.model_zoo`, `pgv.dataset_registry` | Metadata catalogues | [Models](../models/catalog.md), [Datasets](../datasets/index.md) |

Objects not on the client: the trainer (`pygeovision.training`), the model registry (`pygeovision.models.get_model`),
object-based analysis (`pygeovision.obia`), the benchmark evaluator (`pygeovision.benchmark`) and the report builder (`pygeovision.reporting`).

## The client

```{eval-rst}
.. autoclass:: pygeovision.PyGeoVision
   :members: search, download, pipeline, add_credentials, boundary, select_covering_scenes, list_providers, test_provider, clear_cache, cache_stats, status, doctor, prepare_for_ai, create_pipeline, run_pipeline_yaml
   :no-index:
```

## Pipelines

```{eval-rst}
.. autoclass:: pygeovision.ai.pipelines.PipelineResult
   :members:
   :no-index:

.. autoclass:: pygeovision.ai.pipelines.BasePipeline
   :members: run, description, name, kind
   :no-index:

.. autoclass:: pygeovision.ai.pipelines.ModelWeightsRequired
   :no-index:
```

## Models and training

```{eval-rst}
.. autofunction:: pygeovision.models.get_model
   :no-index:

.. autofunction:: pygeovision.models.list_models
   :no-index:

.. autoclass:: pygeovision.training.TrainingConfig
   :no-index:

.. autoclass:: pygeovision.training.GeoTrainer
   :members: fit
   :no-index:
```

The meaning of every `TrainingConfig` field is in [The training configuration](../training/config.md).

## Inference

```{eval-rst}
.. autoclass:: pygeovision.inference.TiledInference
   :members:
   :no-index:

.. autoclass:: pygeovision.inference.BatchInferenceEngine
   :members:
   :no-index:
```

## Data preparation and post-processing

```{eval-rst}
.. autoclass:: pygeovision.data.validator.DataValidator
   :members:
   :no-index:

.. autoclass:: pygeovision.preprocess.Preprocessor
   :members:
   :no-index:

.. autoclass:: pygeovision.data.indices.SpectralIndices
   :members:
   :no-index:

.. autoclass:: pygeovision.data.postprocess.PostProcessor
   :members:
   :no-index:
```

## Explainability and monitoring

```{eval-rst}
.. autoclass:: pygeovision.explainability.GradCAM
   :members:
   :no-index:

.. autoclass:: pygeovision.explainability.UncertaintyEstimator
   :members:
   :no-index:

.. autoclass:: pygeovision.monitoring.DriftDetector
   :members:
   :no-index:
```

## Advanced tools

```{eval-rst}
.. autoclass:: pygeovision.advanced.automl.GeoAutoML
   :members:
   :no-index:

.. autoclass:: pygeovision.advanced.few_shot.FewShotLearner
   :members:
   :no-index:

.. autoclass:: pygeovision.advanced.multitask.MultiTaskLearner
   :members:
   :no-index:

.. autoclass:: pygeovision.advanced.timeseries.GeoTimeSeries
   :members:
   :no-index:

.. autoclass:: pygeovision.advanced.pointcloud.PointCloudProcessor
   :members:
   :no-index:
```

## Duplicate classes by design

Several concepts exist twice (trainer, tiled inference, drift detector, labelers). The classes above are the ones the **command line** uses.
The other implementation, and why it is kept, is described in [Architecture](../architecture.md#systems-that-exist-twice).
