# Cloud deployment: SageMaker, Azure ML, Vertex AI

**Package:** `pygeovision.cloud` · **Classes:** `AWSDeployer`, `AzureDeployer`, `GCPDeployer` (all subclasses of `CloudDeployer`),
helper `build_sagemaker_archive` · **CLI:** `pygeovision cloud deploy-aws`, `pygeovision cloud deploy-gcp` ·
**Install:** `pip install "pygeovision[cloud]"`

```{admonition} Honest status
:class: warning
These classes wrap each provider's SDK. **No call to a live AWS, Azure or Google Cloud service has been made from PyGeoVision's
test environment** (it needs accounts, credentials and money). What *has* been verified offline: the SageMaker archive and its request
handler (run end to end with ONNX Runtime), the argument handling, the request building and the error paths, with the SDKs
replaced by test doubles. Treat a first deployment as an experiment, watch the provider's console, and expect to adapt the
Azure and Vertex AI paths (see each section). Of the three, the AWS path is the most complete.
```

## Why

An endpoint is a model behind a web address: send an image chip, get the prediction back, no PyGeoVision on the caller's
side. It suits an application (a web map that classifies the chip you click on, a mobile app) and sharing a model with people who
cannot run it. It does **not** suit mapping whole scenes: each request carries one small chip, you pay for the machine
while it runs, and the stitching of tiles is still your job. For scene-sized work, run [tiled inference](../inference/index.md) on a
cloud virtual machine instead, or the [ONNX engine](edge.md) on your own hardware.

## What you need first

- A model exported to ONNX and **checked** against the original ([how](edge.md#check-the-export)), exported at the chip size you will send.
  Use a small chip: see [payload limits](#payload-limits).
- An account with billing, credentials the provider's SDK can find, and the permissions below.
- An idea of the bill: a GPU instance such as `ml.g4dn.xlarge` costs by the hour **for as long as the endpoint exists**, used or not.

## AWS SageMaker

### Configure

| Setting | Argument | Environment variable |
|---|---|---|
| Region | `region="us-east-1"` | |
| S3 bucket (must exist, in the same region) | `bucket="my-bucket"` | `AWS_S3_BUCKET` |
| Execution role for the endpoint | `role_arn="arn:aws:iam::…:role/…"` | `AWS_SAGEMAKER_ROLE_ARN` |
| Credentials | | the usual AWS chain (`aws configure`, environment, instance role) |

`AWSDeployer(...).generate_iam_policy()` prints a policy for the identity that **runs** the deployment: S3 upload, creating,
describing, invoking and deleting the endpoint, `iam:PassRole` for the execution role, and the auto-scaling calls. The execution role is a
different identity: it needs to read the bucket, pull the container from ECR and write CloudWatch logs (AWS's managed
`AmazonSageMakerFullAccess` policy covers this). Tighten the `Resource` entries before production use.

### Deploy

```python
from pygeovision.cloud import AWSDeployer

dep = AWSDeployer(region="eu-central-1", bucket="my-pgv-bucket", role_arn="arn:aws:iam::123456789012:role/pgv-sagemaker")
result = dep.deploy("model.onnx", "pgv-buildings-v1",
                    instance_type="ml.g4dn.xlarge",      # a GPU family; ml.c5.xlarge etc. select the CPU container
                    min_instances=1, max_instances=3, auto_scale=True)
```

```console
$ pygeovision cloud deploy-aws model.onnx pgv-buildings-v1 --region eu-central-1 --instance ml.c5.xlarge
```

(The command reads the bucket and role from the two environment variables.)

What happens, in order:

1. The settings are checked. A missing bucket or role gives `{"success": False, "error": "AWSDeployer needs bucket (AWS_S3_BUCKET) and role_arn (AWS_SAGEMAKER_ROLE_ARN)"}` *before* anything is uploaded.
2. An `.onnx` file is **packed into a SageMaker `model.tar.gz`** with a request handler and a `requirements.txt` (below). A file that
   already ends in `.tar.gz` is uploaded unchanged: use that to bring your own handler or a non-ONNX model.
3. The archive is uploaded to `s3://<bucket>/pygeovision/models/`.
4. A model and endpoint are created on AWS's PyTorch inference container (`pytorch-inference:2.1`, GPU or CPU build chosen from the
   instance *family*: `g`, `p`, `inf`, `trn` are GPU; `c5`, `m5` and the like are CPU), and the call waits until the endpoint is in service. **This takes minutes.**
5. If `auto_scale` and `max_instances > min_instances`, a target-tracking policy (70 invocations per instance per minute, scale-out cool-down 60 s,
   scale-in 300 s) is registered. A failure here is only a warning in the log.

The result on success: `success`, `endpoint_name`, `provider`, `instance_type`, `model_s3_uri`, `endpoint_url`, and `autoscaling` when set.
On any failure `success` is `False` with an `error` string; `deploy` does not raise. The account-level container registry for the
image (`763104351884`) is the one AWS publishes for most commercial regions; some regions (for example China and GovCloud) use others,
and there the call fails with an image-not-found error.

### The packed archive and request format

```python
from pygeovision.cloud import build_sagemaker_archive
build_sagemaker_archive("model.onnx")            # → model.tar.gz
```

```text
segformer-b0.onnx          15077994 bytes
code/                      the handler directory SageMaker looks in
code/inference.py              1178 bytes   model_fn / input_fn / predict_fn / output_fn
code/requirements.txt            18 bytes   onnxruntime, numpy (installed by the container at start-up)
```

The handler loads the first `.onnx` file with ONNX Runtime (CUDA if available, else CPU) and speaks JSON. The four handler functions
were run here in the order SageMaker calls them (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) with ONNX Runtime, and the
output matched PyTorch to 1e-5; what has *not* been tested is the container itself.

| Request body | Response |
|---|---|
| `{"inputs": <(C,H,W) or (B,C,H,W) nested list>}` | `{"predictions": <logits, (B, classes, H, W)>}` |
| `{"inputs": …, "argmax": true}` | `{"predictions": <class ids, (B, H, W)>}` |

```python
import numpy as np
chip = np.random.rand(4, 128, 128).astype("float32")           # scaled exactly as in training
out = dep.predict("pgv-buildings-v1", chip, argmax=True)       # {"predictions": [[[0, 0, 1, ...]]]}
classes = np.asarray(out["predictions"])[0]                    # (128, 128)
```

`predict` raises boto3's own errors (credentials, unknown endpoint, request too large). Only a missing SDK comes back as `{"error": "pip install boto3"}`.

### Payload limits

SageMaker real-time endpoints accept roughly 6 MB per request and return roughly 6 MB (check AWS's current limits). The JSON
text of a float array is large. Measured with a 4-band chip and a 3-class SegFormer-B0:

| Chip | Request | Response, logits | Response, `argmax=True` |
|---|---:|---:|---:|
| 64 × 64 | 0.33 MB | 0.27 MB | 0.01 MB |
| 128 × 128 | 1.33 MB | 1.09 MB | 0.05 MB |
| 256 × 256 | 5.32 MB | 4.36 MB | 0.20 MB |
| 512 × 512 | 21 MB | 17 MB | too large to send |

So: send chips of **128 × 128, or at most 256 × 256**, ask for `argmax` unless you need probabilities (six classes would
roughly double the logits response), and tile larger images yourself ([tiled inference](../inference/index.md) can drive a
function that calls the endpoint). Requests also time out after about a minute.

### Cleaning up

```python
dep.list_endpoints()                    # ['pgv-buildings-v1'], in-service endpoints only
dep.delete_endpoint("pgv-buildings-v1") # {'success': True, 'deleted': 'pgv-buildings-v1'}
```

`delete_endpoint` deletes the **endpoint** (and so stops the hourly charge). SageMaker's *model* and *endpoint configuration*
objects and the S3 archive stay behind (they cost little or nothing but clutter the console); remove them in the console or with
`aws sagemaker delete-model` / `delete-endpoint-config` / `aws s3 rm`. If `list_endpoints` cannot reach AWS (no credentials) it
returns `[]` and logs a warning: **an empty list from a failed call does not mean there are no endpoints**, so read the log.

## Azure Machine Learning

```python
from pygeovision.cloud import AzureDeployer

dep = AzureDeployer(subscription_id="…", resource_group="…", workspace_name="pgv-workspace", region="westeurope")
dep.deploy("model.onnx", "pgv-buildings", vm_size="Standard_NC4as_T4_v3", instance_count=1)
dep.predict("pgv-buildings", chip)
dep.list_endpoints(); dep.delete_endpoint("pgv-buildings")
```

The settings can come from `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP` and `AZURE_ML_WORKSPACE`; credentials come from
`DefaultAzureCredential` (`az login`, environment variables, managed identity). `deploy` registers the model, creates a managed online
endpoint with key authentication and one deployment named `default`, and returns `success`, `endpoint_name`, `provider`, `scoring_uri`, `vm_size`.

**Limitation.** The deployment is created *without* an environment or scoring script. Azure ML can serve some model types without code
(MLflow models, for example) but a bare ONNX file generally needs a scoring script and environment, so expect the deployment step to be
rejected. To use it with a custom model, extend `deploy` with an `Environment` and a `CodeConfiguration`. `predict` writes the JSON request
to a temporary file (which the SDK's `invoke` requires; this used to pass the JSON text as the file name and fail) and removes it afterwards.

## Google Cloud Vertex AI

```python
from pygeovision.cloud import GCPDeployer

dep = GCPDeployer(project_id="my-project", region="europe-west4")
dep.deploy("model.onnx", "pgv-buildings", machine_type="n1-standard-4",
           accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1, gcs_bucket="my-bucket")
```

```console
$ pygeovision cloud deploy-gcp model.onnx pgv-buildings --project my-project --region europe-west4
```

Settings: `project_id` (or `GCP_PROJECT_ID`), the bucket from `gcs_bucket` or `GCS_BUCKET` (default `<project>-pgv`; **it must already exist**).
Credentials: Application Default Credentials (`gcloud auth application-default login`). The file is uploaded to
`gs://<bucket>/pygeovision/models/<endpoint name>/` (one folder per endpoint, so two models never share an artifact folder), then
uploaded as a Vertex AI model on Google's prebuilt PyTorch container, deployed to a new endpoint with 1 to 5 replicas. With
`accelerator_count=0` the CPU container is used. The CLI command prints the whole result dict, and exits with status 1 on failure.

**Limitation.** The prebuilt PyTorch container serves models in its own archive/handler format; a bare `.onnx` file is uploaded as is
and is not served by it. Supply a model archive in the format that container documents, or a custom container, and change the
`serving_container_image_uri` accordingly. `predict` sends `instances=[array.tolist()]` to the endpoint whose display name matches.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `AWSDeployer needs bucket (AWS_S3_BUCKET) and role_arn …` | Not configured | Pass `bucket=` and `role_arn=` or set the variables |
| `{"success": False, "error": "pip install boto3 sagemaker"}` | SDK missing | `pip install "pygeovision[cloud]"` |
| `model file not found: …` | Wrong path | Check it |
| `expected an .onnx file …` (from `build_sagemaker_archive`) | A `.pt` or other file | Export to ONNX, or build your own `model.tar.gz` and pass that |
| `AccessDenied` / `not authorized to perform iam:PassRole` | Missing permission on your identity | Attach the policy from `generate_iam_policy()` |
| Endpoint stuck in `Creating`, then `Failed` | Container could not start: wrong role permissions, no image in this region, handler error | Open the endpoint's CloudWatch log in the AWS console; the failure reason is there |
| `ModelError … 413` / payload too large | The request or response exceeds the limit | Smaller chips, `argmax=True` ([limits](#payload-limits)) |
| `Got invalid dimensions for input` in the endpoint log | The ONNX file was exported at another size | Send that size, or export with `dynamic_spatial=True` ([edge](edge.md#export)) |
| Predictions are garbage but the endpoint works | The chip was not scaled as in training | Apply the same scaling before sending |
| `list_endpoints()` returns `[]` and the log has a warning | The call failed (credentials, region) | Fix the credentials/region; an empty list is not proof |
| Azure: deployment rejected / no scoring script | The `deploy` call defines none | [Extend it](#azure-machine-learning) |
| Vertex AI: endpoint up but prediction errors | The container cannot serve a bare `.onnx` | [Use a matching container or archive](#google-cloud-vertex-ai) |
| You are still being charged | Endpoint still running | `delete_endpoint(...)`, then check the console for leftovers |

```{seealso}
[Edge deployment](edge.md) · [Inference](../inference/index.md) · [Installation extras](../getting-started/installation.md) ·
[Monitoring a deployed model](../monitoring/index.md)
```
