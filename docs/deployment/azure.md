# Azure Machine Learning

Azure ML serves a model as a **managed online endpoint**: a web address that runs your scoring code on virtual machines Azure looks
after. `AzureDeployer` registers the model and creates the endpoint. Read the status box first: this is the least complete of the
three cloud paths, and the page tells you exactly what to add to make it work for an ONNX segmentation model.

```{admonition} Status, stated plainly
:class: warning
- **Verified here (no Azure account needed):** the scoring script on this page, run against an exported ONNX model and compared
  with PyTorch (`tests/test_docs_deploy_examples.py`); the argument handling and the error paths of `AzureDeployer`.
- **Not verified:** any call to a live Azure service. The SDK snippets for the environment and the code configuration are a
  sketch to adapt, not tested output. Watch the endpoint's logs in Azure ML studio on the first deployment.
- `AzureDeployer.deploy` as shipped creates the deployment **without** an environment or scoring script, so a bare `.onnx` file is
  most likely rejected. The steps below close that gap.
```

```{figure} ../_static/img/flow-azure.svg
:alt: Azure ML deployment: export and check ONNX, write a scoring script, register the model, create the endpoint, create the deployment with an environment and the script, test, delete
:width: 100%
:class: pgv-figure

The shipped `deploy` covers the middle steps; the scoring script and environment are yours.
```

## What you need

| Need | Detail |
|---|---|
| An Azure subscription with billing, a resource group, and an Azure ML **workspace** | `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE`, or pass them as arguments |
| Credentials | `DefaultAzureCredential`: `az login`, environment variables, or a managed identity |
| Quota for the VM size | GPU sizes (the default `Standard_NC4as_T4_v3`) usually need a quota request; a CPU size such as `Standard_DS3_v2` works for small models |
| The packages | `pip install azure-ai-ml azure-identity` (part of `pip install "pygeovision[cloud]"`) |
| An ONNX file you have **checked** against the original | [Edge deployment: check the export](edge.md#check-the-export) |

## The shipped deployer

```python
from pygeovision.cloud import AzureDeployer

dep = AzureDeployer(subscription_id="…", resource_group="…", workspace_name="pgv-workspace", region="westeurope")
result = dep.deploy("model.onnx", "pgv-buildings", vm_size="Standard_NC4as_T4_v3", instance_count=1)
```

| Argument of `deploy` | Default | Meaning |
|---|---|---|
| `model_path` | | The model file or folder to register |
| `endpoint_name` | | The endpoint (and registered model) name; letters, digits and dashes |
| `vm_size` | `Standard_NC4as_T4_v3` | The virtual machine size |
| `instance_count` | 1 | Number of machines |

In order, it: registers the model (type `custom_model`), creates a managed online endpoint with **key** authentication, and creates
one deployment named `default` on it. It returns a dictionary, and **never raises**:

| Result | Meaning |
|---|---|
| `{"success": True, "endpoint_name", "provider": "azure_ml", "scoring_uri", "vm_size"}` | Created |
| `{"success": False, "error": "pip install azure-ai-ml azure-identity"}` | SDK missing |
| `{"success": False, "error": "<Azure's message>"}` | Anything else: credentials, quota, a deployment Azure rejected |

The other methods: `predict(endpoint_name, array)` (sends `{"inputs": array.tolist()}` through the SDK's `invoke`, using a temporary
request file that is removed afterwards), `list_endpoints()` (names; **an empty list with a warning in the log means the call
failed, not that there are none**) and `delete_endpoint(name)`.

## Making it serve an ONNX model

A deployment needs two things the shipped call does not give: a **scoring script** (what to do with a request) and an
**environment** (the packages to run it with). The script below speaks the same JSON as the SageMaker handler, so a client works
against both.

```{literalinclude} ../_examples/deploy/azure_score.py
:language: python
:caption: azure_score.py
:linenos:
```

{download}`Download azure_score.py <../_examples/deploy/azure_score.py>`

`init()` runs once when the container starts and loads the first `.onnx` file under `AZUREML_MODEL_DIR` (Azure puts the registered
model there). `run(raw_data)` runs for each request. Tested here with a small exported model:

```text
request  : {"inputs": <(4, 16, 16) chip>}               -> predictions (1, 3, 16, 16), equal to PyTorch within 1e-5
request  : {"inputs": <chip>, "argmax": true}            -> class ids   (1, 16, 16)
```

Then create the deployment yourself with the code configuration and an environment. This is a sketch against the `azure-ai-ml`
SDK (version 2) that you must adapt and test:

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (CodeConfiguration, Environment, ManagedOnlineDeployment,
                                  ManagedOnlineEndpoint, Model)
from azure.identity import DefaultAzureCredential

ml = MLClient(DefaultAzureCredential(), "<subscription>", "<resource-group>", "<workspace>")

model = ml.models.create_or_update(Model(path="model.onnx", name="pgv-buildings", type="custom_model"))
ml.online_endpoints.begin_create_or_update(ManagedOnlineEndpoint(name="pgv-buildings", auth_mode="key")).result()

env = Environment(                                   # needs onnxruntime and numpy
    name="pgv-onnx-env",
    image="<a base image from Azure's list of inference images>",
    conda_file="conda.yml",                          # dependencies: python, pip: [onnxruntime, numpy, azureml-inference-server-http]
)
ml.online_deployments.begin_create_or_update(ManagedOnlineDeployment(
    name="default", endpoint_name="pgv-buildings", model=model, environment=env,
    code_configuration=CodeConfiguration(code="./score_dir", scoring_script="azure_score.py"),
    instance_type="Standard_DS3_v2", instance_count=1,
)).result()
ml.online_endpoints.begin_create_or_update(...)      # then route 100% of traffic to "default" (see the Azure ML docs)
```

The base image name, the `conda.yml` contents and the traffic setting follow Azure's own documentation for custom
deployments, which changes faster than this page; take them from there. The script and the JSON contract are the parts that are
specific to PyGeoVision.

## Calling it

```python
import numpy as np
chip = np.random.rand(4, 128, 128).astype("float32")          # scaled exactly as in training
reply = dep.predict("pgv-buildings", chip)                    # the SDK returns the response text
```

`predict` sends `{"inputs": ...}` and returns what the SDK returns (a JSON string). Parse it with `json.loads` and read
`["predictions"]`. Chips of 128 × 128 are comfortable; the request size grows fast ([payload sizes](sagemaker.md#payload-limits)).

## Cost and cleaning up

A managed endpoint bills for its machines **every hour it exists**, used or not.

```python
dep.list_endpoints()
dep.delete_endpoint("pgv-buildings")        # {'success': True, 'deleted': 'pgv-buildings'}
```

Deleting the endpoint stops the machines. The registered model (and the environment you created) remain in the workspace; they
cost little, and you can delete them in Azure ML studio.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `{"success": False, "error": "pip install azure-ai-ml azure-identity"}` | SDK missing | `pip install "pygeovision[cloud]"` |
| `DefaultAzureCredential failed to retrieve a token` | No login | `az login`, or set the service principal environment variables |
| Deployment rejected about missing environment or scoring script | The shipped `deploy` defines none | Create the deployment yourself, as above |
| `QuotaExceeded` / `not enough quota` | The VM size has no quota in the region | Request quota, or use a CPU size, or another region |
| The deployment starts and then fails the health check | A package is missing in the environment, or `init()` raised | Read the deployment logs in studio; test the script locally first |
| `list_endpoints()` returns `[]` and the log shows a warning | The call failed | Fix credentials, workspace name or region |
| Predictions are garbage | Chip not scaled as in training | Apply the same scaling before sending ([Inference](../inference/index.md#matching-the-input-to-training)) |

## Related

- [Cloud overview](cloud.md), [SageMaker](sagemaker.md), [Vertex AI](vertex.md), [Edge: ONNX export and check](edge.md).
- [`pygeovision.cloud` in the Python API](../reference/api/deployment.md).
