# Google Cloud Vertex AI

Vertex AI serves a model behind an **endpoint** on machines Google runs. `GCPDeployer` uploads the model to Cloud Storage, registers it
in Vertex AI, and deploys it. As with Azure, the shipped call stops short for a bare ONNX file: Google's prebuilt PyTorch container
does not serve it. This page says what the deployer does, why that is not enough, and gives a small **custom container** that does
serve an ONNX segmentation model (the server code is tested locally).

```{admonition} Status, stated plainly
:class: warning
- **Verified here (no Google account needed):** the server script on this page, run against an exported ONNX model over real HTTP
  and compared with PyTorch (`tests/test_docs_deploy_examples.py`); the `GCPDeployer` argument handling and error paths.
- **Not verified:** any call to a live Google Cloud service, and building or pushing the container image. The Dockerfile and the
  `Model.upload` call are sketches to adapt.
- A bare `.onnx` file deployed with the shipped `deploy` is uploaded but **not served** by the prebuilt PyTorch container.
```

```{figure} ../_static/img/flow-vertex.svg
:alt: Vertex AI deployment: export and check ONNX, build a serving container, upload the model to Cloud Storage, register it with the container, create an endpoint, deploy, test, delete
:width: 100%
:class: pgv-figure

The deployer does the upload, register and deploy; the serving container is yours.
```

## What you need

| Need | Detail |
|---|---|
| A Google Cloud project with billing and the Vertex AI API enabled | `project_id=` or `GCP_PROJECT_ID` |
| A Cloud Storage bucket that **already exists** | `gcs_bucket=` or `GCS_BUCKET`; the default name is `<project>-pgv` |
| Credentials | Application Default Credentials: `gcloud auth application-default login` |
| Quota for the machine and accelerator | GPUs such as `NVIDIA_TESLA_T4` need quota in the region |
| The packages | `pip install google-cloud-aiplatform google-cloud-storage` (part of `pygeovision[cloud]`) |
| An ONNX file you have **checked** against the original | [Edge deployment: check the export](edge.md#check-the-export) |

## The shipped deployer

```python
from pygeovision.cloud import GCPDeployer

dep = GCPDeployer(project_id="my-project", region="europe-west4")
result = dep.deploy("model.onnx", "pgv-buildings", machine_type="n1-standard-4",
                    accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1, gcs_bucket="my-bucket")
```

```console
$ pygeovision cloud deploy-gcp model.onnx pgv-buildings --project my-project --region europe-west4
```

The command prints the whole result dictionary and exits with status 1 when `success` is false.

| Argument | Default | Meaning |
|---|---|---|
| `model_path` | | The file to upload |
| `endpoint_name` | | Used as the model's and the endpoint's display name, and as the Cloud Storage folder |
| `machine_type` | `n1-standard-4` | The virtual machine type |
| `accelerator_type` | `NVIDIA_TESLA_T4` | The GPU type, used only when `accelerator_count > 0` |
| `accelerator_count` | 1 | `0` selects the CPU container and no GPU |
| `gcs_bucket` | `GCS_BUCKET` or `<project>-pgv` | **Must already exist** |

In order, it: uploads the file to `gs://<bucket>/pygeovision/models/<endpoint name>/` (one folder per endpoint, so two models
never share an artifact folder), uploads a Vertex AI **model** that points at that folder and at Google's prebuilt PyTorch serving
container (GPU or CPU build by `accelerator_count`), creates an **endpoint**, and deploys the model to it with 1 to 5 replicas.
It returns `{"success": True, "endpoint_name", "provider": "gcp_vertex", "endpoint_id", "model_gcs_uri"}`, or
`{"success": False, "error": ...}`; it never raises.

The other methods: `predict(endpoint_name, array)` (finds the endpoint by display name and sends `instances=[array.tolist()]`),
`list_endpoints()` (display names; an empty list with a log warning means the call failed) and `delete_endpoint(name)` (undeploys
everything, then deletes).

## Why a bare ONNX file is not served

The prebuilt PyTorch container expects a **TorchServe model archive** with a handler, not an `.onnx` file. The shipped call uploads the
file as it is, and the container finds nothing it can load. There are two ways forward: package a TorchServe archive in the format
Google documents for that container, or **serve it yourself with a custom container**, which is the more direct path for ONNX.

## A custom container for ONNX

Vertex AI starts your container and talks to it over HTTP: a health route and a predict route on the port in `AIP_HTTP_PORT`. The
server below follows that contract using only the standard library, `numpy` and `onnxruntime`:

```{literalinclude} ../_examples/deploy/vertex_server.py
:language: python
:caption: vertex_server.py
:linenos:
```

{download}`Download vertex_server.py <../_examples/deploy/vertex_server.py>`

Tested here over real HTTP with a small exported model: `GET /health` gives `200 {"status": "ok"}`; `POST /predict` with
`{"instances": [<chip>], "parameters": {"argmax": true}}` returns class ids identical to PyTorch's; a malformed body returns
`400` with the error text, and the server keeps running.

A minimal image (a sketch):

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir onnxruntime numpy
COPY vertex_server.py /app/vertex_server.py
COPY model.onnx /models/model.onnx
ENV MODEL_PATH=/models/model.onnx
CMD ["python", "/app/vertex_server.py"]
```

Build it, push it to Artifact Registry (`docker build`, `docker push`, or `gcloud builds submit`), and register the model with that image
instead of the prebuilt one:

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="europe-west4")
model = aiplatform.Model.upload(
    display_name="pgv-buildings",
    serving_container_image_uri="europe-west4-docker.pkg.dev/my-project/pgv/onnx-server:1",
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_ports=[8080],
)
endpoint = aiplatform.Endpoint.create(display_name="pgv-buildings")
model.deploy(endpoint=endpoint, machine_type="n1-standard-4", min_replica_count=1, max_replica_count=2)
```

For a GPU, use a base image with CUDA and `onnxruntime-gpu`, and pass `accelerator_type` and `accelerator_count` to `deploy`. The
model is baked into the image here, so no Cloud Storage upload is needed; to keep the model out of the image, upload it as
`GCPDeployer` does and point `artifact_uri` at the folder (Vertex then exposes it to the container; set `MODEL_PATH` accordingly).

## Calling it

```python
import numpy as np
chip = np.random.rand(4, 128, 128).astype("float32")          # scaled exactly as in training
predictions = dep.predict("pgv-buildings", chip)               # the endpoint's predictions list
```

With the server above `predict` sends `instances=[chip]` and the reply is `{"predictions": [<logits for each instance>]}`; the SDK
returns the list. Ask for class ids by sending `parameters={"argmax": True}` through the SDK yourself (`endpoint.predict(instances=...,
parameters=...)`): the deployer's `predict` does not pass parameters, and the logits reply is large ([payload sizes](sagemaker.md#payload-limits)).

## Cost and cleaning up

An endpoint bills for its machines **every hour it exists**.

```python
dep.list_endpoints()
dep.delete_endpoint("pgv-buildings")      # undeploys and deletes; {'success': True, 'deleted': 'pgv-buildings'}
```

The registered model, the Cloud Storage files and the container image remain; remove them in the console or with `gcloud` when you no
longer need them.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `{"success": False, "error": "pip install google-cloud-aiplatform google-cloud-storage"}` | SDK missing | Install the extras |
| `404 ... bucket does not exist` | The bucket must already exist | Create it, or pass `gcs_bucket=` |
| `403` / permission errors | The identity lacks Storage or Vertex AI roles | Grant `Storage Object Admin` on the bucket and `Vertex AI User` on the project |
| The endpoint is up but prediction errors | The prebuilt container cannot serve a bare `.onnx` | Use the custom container above |
| The container restarts in a loop | The server crashed at start (model path wrong) | Run the image locally: `docker run -p 8080:8080 ...` and `curl localhost:8080/health` |
| `Quota ... NVIDIA_TESLA_T4` exceeded | No GPU quota in the region | Request quota or set `accelerator_count=0` |
| `list_endpoints()` returns `[]` and the log warns | The call failed | Fix credentials, project or region |

## Related

- [Cloud overview](cloud.md), [SageMaker](sagemaker.md), [Azure ML](azure.md), [Edge: ONNX export and check](edge.md).
