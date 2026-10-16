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
  Use a small chip: see [payload limits](sagemaker.md#payload-limits).
- An account with billing, credentials the provider's SDK can find, and the permissions below.
- An idea of the bill: a GPU instance such as `ml.g4dn.xlarge` costs by the hour **for as long as the endpoint exists**, used or not.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `AWSDeployer needs bucket (AWS_S3_BUCKET) and role_arn …` | Not configured | Pass `bucket=` and `role_arn=` or set the variables |
| `{"success": False, "error": "pip install boto3 sagemaker"}` | SDK missing | `pip install "pygeovision[cloud]"` |
| `model file not found: …` | Wrong path | Check it |
| `expected an .onnx file …` (from `build_sagemaker_archive`) | A `.pt` or other file | Export to ONNX, or build your own `model.tar.gz` and pass that |
| `AccessDenied` / `not authorized to perform iam:PassRole` | Missing permission on your identity | Attach the policy from `generate_iam_policy()` |
| Endpoint stuck in `Creating`, then `Failed` | Container could not start: wrong role permissions, no image in this region, handler error | Open the endpoint's CloudWatch log in the AWS console; the failure reason is there |
| `ModelError … 413` / payload too large | The request or response exceeds the limit | Smaller chips, `argmax=True` ([limits](sagemaker.md#payload-limits)) |
| `Got invalid dimensions for input` in the endpoint log | The ONNX file was exported at another size | Send that size, or export with `dynamic_spatial=True` ([edge](edge.md#export)) |
| Predictions are garbage but the endpoint works | The chip was not scaled as in training | Apply the same scaling before sending |
| `list_endpoints()` returns `[]` and the log has a warning | The call failed (credentials, region) | Fix the credentials/region; an empty list is not proof |
| Azure: deployment rejected / no scoring script | The `deploy` call defines none | [Extend it](azure.md#azure-machine-learning) |
| Vertex AI: endpoint up but prediction errors | The container cannot serve a bare `.onnx` | [Use a matching container or archive](vertex.md#google-cloud-vertex-ai) |
| You are still being charged | Endpoint still running | `delete_endpoint(...)`, then check the console for leftovers |

```{seealso}
[Edge deployment](edge.md) · [Inference](../inference/index.md) · [Installation extras](../getting-started/installation.md) ·
[Monitoring a deployed model](../monitoring/index.md)
```
