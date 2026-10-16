# AWS SageMaker

## Configure

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

## Deploy

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

## The packed archive and request format

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

## Payload limits

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

## Cleaning up

```python
dep.list_endpoints()                    # ['pgv-buildings-v1'], in-service endpoints only
dep.delete_endpoint("pgv-buildings-v1") # {'success': True, 'deleted': 'pgv-buildings-v1'}
```

`delete_endpoint` deletes the **endpoint** (and so stops the hourly charge). SageMaker's *model* and *endpoint configuration*
objects and the S3 archive stay behind (they cost little or nothing but clutter the console); remove them in the console or with
`aws sagemaker delete-model` / `delete-endpoint-config` / `aws s3 rm`. If `list_endpoints` cannot reach AWS (no credentials) it
returns `[]` and logs a warning: **an empty list from a failed call does not mean there are no endpoints**, so read the log.
