# Deploying models: edge and cloud

A trained `.pth` file is not yet something a field laptop, a drone computer or a web service can run. Deployment is the step that
takes a model out of the training environment and puts it where it will be used. This section covers the ways PyGeoVision does that,
and says for each how far it has been verified, because the answer differs a lot from one path to the next.

```{figure} ../_static/img/tree-deploy-chooser.svg
:alt: Choosing where to run a trained model: scenes of your own use tiled inference; an app needing a web address uses a cloud endpoint on AWS, Azure or Google; a machine in the field uses ONNX Runtime, or TensorRT on an NVIDIA Jetson
:width: 100%
:class: pgv-figure

Start with who will call the model. Most projects never need a deployment at all.
```

## Do you need to deploy?

Often not. If **you** run the model on scenes, [tiled inference](../inference/index.md) on the machine you trained on (or a rented virtual
machine) is the simplest and most accurate route: no conversion, no endpoint, nothing to keep alive. Deploy only when somebody
or something **else** has to call the model, or when the target machine cannot carry PyTorch.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Edge: ONNX
:link: edge
:link-type: doc
**A portable file and a small runtime.** Export, check against PyTorch, run on CPU, GPU or Apple silicon, run on a whole GeoTIFF, benchmark.
:::

:::{grid-item-card} NVIDIA Jetson
:link: jetson
:link-type: doc
**A TensorRT engine for one device.** The ONNX export and `trtexec` command are verified; the engine build is not.
:::

:::{grid-item-card} Cloud overview
:link: cloud
:link-type: doc
**Why and when to use an endpoint**, what it costs, and a table of every error you are likely to meet.
:::

:::{grid-item-card} AWS SageMaker
:link: sagemaker
:link-type: doc
**The most complete cloud path.** Packs an ONNX file with a tested request handler, deploys, scales, cleans up.
:::

:::{grid-item-card} Azure ML
:link: azure
:link-type: doc
**Registers and creates the endpoint**; you add the scoring script (provided and tested) and an environment.
:::

:::{grid-item-card} Google Vertex AI
:link: vertex
:link-type: doc
**Uploads and deploys**; you add a custom serving container (server provided and tested).
:::
::::

## The paths side by side

| You want to run the model… | Use | Page | Verified here? |
|---|---|---|---|
| On any machine without PyTorch (CPU server, laptop, a Raspberry-class board) | Export to **ONNX**, run with **ONNX Runtime** | [Edge](edge.md#onnx-export-and-runtime) | **Yes**: export, numeric parity with PyTorch, tiled GeoTIFF inference and timings were run |
| On an NVIDIA **Jetson** | ONNX, then a **TensorRT** engine | [Jetson](jetson.md) | ONNX export and the `trtexec` command: yes. TensorRT build and timing: **no** (no TensorRT or GPU available) |
| As an endpoint on **AWS** | `AWSDeployer` | [SageMaker](sagemaker.md) | Archive and request handler run end to end with ONNX Runtime. **No live AWS call** |
| As an endpoint on **Azure** | `AzureDeployer` plus a scoring script | [Azure ML](azure.md) | Scoring script run and compared with PyTorch. **No live Azure call**; the SDK steps are a sketch |
| As an endpoint on **Google Cloud** | `GCPDeployer` plus a custom container | [Vertex AI](vertex.md) | Server run over real HTTP and compared with PyTorch. **No live Google call**; the image build is a sketch |

## The same path on every route

Whichever route you take, the work has the same four stages, and most deployment failures are in the first two.

| Stage | What you do | The check |
|---|---|---|
| 1. Export | Turn the trained network into an ONNX file at the chip size you will serve (`pygeovision edge export-onnx`) | The export opens in ONNX Runtime |
| 2. Compare | Run the **same** chips through the original and the export | Largest logit difference around 1e-5, and the same class at (almost) every pixel ([how](edge.md#check-the-export)) |
| 3. Wrap | Add the part that speaks to the outside: an ONNX Runtime session, a SageMaker archive, a scoring script, a container | The wrapper returns what the original returns, on a request built the way the caller will build it |
| 4. Watch | Benchmark on the target, then watch the model on real data ([monitoring](../monitoring/index.md)) | Latency on the target hardware; drift after weeks in use |

## One request, three languages

The three clouds share a JSON contract so a client written once works against any of them (SageMaker and Azure exactly; Vertex
wraps the same arrays in `instances`/`predictions`):

```python
import numpy as np

chip = np.random.rand(4, 128, 128).astype("float32")        # scaled exactly as in training
body = {"inputs": chip.tolist(), "argmax": True}             # SageMaker and Azure ML
body = {"instances": [chip.tolist()], "parameters": {"argmax": True}}   # Vertex AI custom container
```

The reply carries `predictions`: logits `(batch, classes, H, W)`, or class ids `(batch, H, W)` when `argmax` is set. Ask for class ids
unless you need probabilities: the logits for one 256 × 256 chip already weigh 4 MB of JSON ([payload sizes](sagemaker.md#payload-limits)).

## Command line

| Command | What it does |
|---|---|
| `pygeovision edge export-onnx MODEL_NAME --checkpoint best.pth -o model.onnx` | Export a registry model to ONNX ([options](edge.md#export)) |
| `pygeovision edge benchmark-onnx model.onnx --runs 30` | Time the ONNX model |
| `pygeovision cloud deploy-aws model.onnx ENDPOINT --region …` | Deploy to SageMaker |
| `pygeovision cloud deploy-gcp model.onnx ENDPOINT --project …` | Deploy to Vertex AI (see its [limits](vertex.md#why-a-bare-onnx-file-is-not-served)) |

All options are in the [command-line reference](../reference/cli/index.md). Azure has no command; use `AzureDeployer` from Python.

```{admonition} Read this before you deploy anything
:class: important
1. **An exported model is not a better model.** Export only changes the file format. Always compare the exported model's
   output with the original on a few real chips before trusting it ([how](edge.md#check-the-export)).
2. **The pre- and post-processing are not in the file.** Scaling, band order, tiling and class names must be reproduced
   exactly as in training ([Inference](../inference/index.md#matching-the-input-to-training)).
3. **Cloud endpoints cost money while they exist.** Delete them when you are done ([`delete_endpoint`](sagemaker.md#cleaning-up)).
4. **Latency numbers are for the machine that measured them.** Benchmark on the target hardware.
5. **A cloud call that has never been run is an experiment.** Nothing here was run against a live cloud service; start with the smallest
   instance and read the provider's logs.
```

## Related

- [Exporting and shrinking a model (training)](../training/export.md#exporting-and-shrinking-a-model)
- [Benchmarking](../experiments/benchmark.md), [Installation extras](../getting-started/installation.md)
- [Python API: `pygeovision.edge` and `pygeovision.cloud`](../reference/api/deployment.md)
