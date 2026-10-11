"""
Cloud deployment utilities (F4) — AWS SageMaker, Azure ML, GCP Vertex AI.

Thin wrappers over each provider's SDK that upload a model artifact and create a
real-time endpoint. They need the provider SDK, credentials and (usually) a billable
account, so they are not exercised against live services in PyGeoVision's own tests:
treat them as a starting point and read the provider's console/logs when a deployment
does not come up. Every ``deploy`` returns a dict with ``success`` and, on failure, ``error``.
"""
from __future__ import annotations

import json
import logging
import os
import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Handler the SageMaker PyTorch inference container calls (model_fn / input_fn / predict_fn /
# output_fn). It serves an ONNX file with onnxruntime and speaks JSON: {"inputs": [[[...]]]}
# in, {"predictions": [...]} out.
SAGEMAKER_INFERENCE_PY = '''"""Serve an ONNX model with onnxruntime inside the SageMaker PyTorch container.

Request:  {"inputs": <(C,H,W) or (B,C,H,W) nested list>, "argmax": false}
Response: {"predictions": <logits (B,classes,H,W), or class ids (B,H,W) when "argmax" is true>}
"""
import json
import os

import numpy as np
import onnxruntime as ort


def model_fn(model_dir):
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".onnx"))
    if not files:
        raise FileNotFoundError("no .onnx file in " + model_dir)
    wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
    providers = [p for p in wanted if p in ort.get_available_providers()]
    return ort.InferenceSession(os.path.join(model_dir, files[0]), providers=providers)


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError("unsupported content type: " + str(content_type))
    payload = json.loads(request_body)
    argmax = bool(payload.get("argmax", False)) if isinstance(payload, dict) else False
    array = np.asarray(payload["inputs"] if isinstance(payload, dict) else payload, dtype=np.float32)
    return (array[None] if array.ndim == 3 else array), argmax


def predict_fn(data, session):
    array, argmax = data
    out = session.run(None, {session.get_inputs()[0].name: array})[0]
    return out.argmax(axis=1) if argmax else out


def output_fn(prediction, accept="application/json"):
    return json.dumps({"predictions": prediction.tolist()}), "application/json"
'''


def build_sagemaker_archive(model_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Pack an ``.onnx`` file into the ``model.tar.gz`` layout SageMaker's PyTorch container serves.

    The archive holds the model at its root plus ``code/inference.py`` (the handler above) and
    ``code/requirements.txt`` (``onnxruntime``, installed by the container at start-up).
    Returns the archive path (default: next to the model, ``<name>.tar.gz``).
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if model_path.suffix != ".onnx":
        raise ValueError(f"expected an .onnx file, got {model_path.name!r}; for other formats "
                         "build your own model.tar.gz with a code/inference.py handler")
    out = Path(output_path) if output_path else model_path.with_suffix(".tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / "code"
        code.mkdir()
        (code / "inference.py").write_text(SAGEMAKER_INFERENCE_PY)
        (code / "requirements.txt").write_text("onnxruntime\nnumpy\n")
        with tarfile.open(out, "w:gz") as tar:
            tar.add(model_path, arcname=model_path.name)
            tar.add(code, arcname="code")
    return out


class CloudDeployer(ABC):
    """Abstract base class for cloud deployment; use :meth:`from_provider` or a concrete subclass."""

    def __init__(self, provider: str, region: str | None = None) -> None:
        self.provider = provider
        self.region = region

    @abstractmethod
    def deploy(self, model_path: str, endpoint_name: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def delete_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_endpoints(self) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def from_provider(provider: str, **kwargs) -> CloudDeployer:
        providers = {"aws": AWSDeployer, "azure": AzureDeployer, "gcp": GCPDeployer}
        if provider not in providers:
            raise ValueError(f"provider must be one of {list(providers)}")
        return providers[provider](**kwargs)


class AWSDeployer(CloudDeployer):
    """Deploy to AWS SageMaker (F4).

    Example::

        deployer = AWSDeployer(region="us-east-1")
        result = deployer.deploy(
            model_path="./models/buildings.onnx",
            endpoint_name="pgv-buildings-v1",
            instance_type="ml.g4dn.xlarge",
        )
        prediction = deployer.predict("pgv-buildings-v1", image_array)
    """

    def __init__(self, region: str = "us-east-1",
                 role_arn: str | None = None,
                 bucket: str | None = None) -> None:
        super().__init__("aws", region)
        self.role_arn = role_arn or os.environ.get("AWS_SAGEMAKER_ROLE_ARN", "")
        self.bucket   = bucket   or os.environ.get("AWS_S3_BUCKET", "")
        self._client = None

    def _get_client(self):
        if self._client: return self._client
        try:
            import boto3
            self._client = boto3.client("sagemaker", region_name=self.region)
            return self._client
        except ImportError:
            raise ImportError("pip install boto3")

    def deploy(
        self,
        model_path: str | Path,
        endpoint_name: str,
        instance_type: str = "ml.g4dn.xlarge",
        framework: str = "onnx",
        min_instances: int = 1,
        max_instances: int = 3,
        auto_scale: bool = True,
    ) -> dict[str, Any]:
        """Upload a model to S3 and create a SageMaker real-time endpoint.

        ``model_path`` is either a ready SageMaker ``model.tar.gz`` (with your own
        ``code/inference.py``) or an ``.onnx`` file, which is packed with
        :func:`build_sagemaker_archive` first. Needs ``bucket`` (or ``AWS_S3_BUCKET``) and
        ``role_arn`` (or ``AWS_SAGEMAKER_ROLE_ARN``). Creating the endpoint bills until
        :meth:`delete_endpoint` is called. Returns ``{"success": False, "error": ...}`` rather than raising.
        """
        try:
            import boto3
            import sagemaker
            from sagemaker.model import Model

            missing = [n for n, v in (("bucket (AWS_S3_BUCKET)", self.bucket),
                                      ("role_arn (AWS_SAGEMAKER_ROLE_ARN)", self.role_arn)) if not v]
            if missing:
                return {"success": False, "error": "AWSDeployer needs " + " and ".join(missing)}

            model_path = Path(model_path)
            if not model_path.is_file():
                return {"success": False, "error": f"model file not found: {model_path}"}
            if not model_path.name.endswith(".tar.gz"):
                model_path = build_sagemaker_archive(model_path, Path(tempfile.mkdtemp()) / (model_path.stem + ".tar.gz"))

            # Upload model to S3
            s3 = boto3.client("s3", region_name=self.region)
            model_key = f"pygeovision/models/{Path(model_path).name}"
            s3.upload_file(str(model_path), self.bucket, model_key)
            model_s3_uri = f"s3://{self.bucket}/{model_key}"
            logger.info("Model uploaded: %s", model_s3_uri)

            # Create SageMaker model
            sm = sagemaker.Session()
            model = Model(
                model_data=model_s3_uri,
                role=self.role_arn,
                image_uri=self._get_inference_image(framework, instance_type),
                sagemaker_session=sm,
            )

            # Deploy endpoint
            model.deploy(
                initial_instance_count=min_instances,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )

            result = {
                "success": True,
                "endpoint_name": endpoint_name,
                "provider": "aws_sagemaker",
                "instance_type": instance_type,
                "model_s3_uri": model_s3_uri,
                "endpoint_url": f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            }

            if auto_scale and max_instances > min_instances:
                self._setup_autoscaling(endpoint_name, min_instances, max_instances)
                result["autoscaling"] = {"min": min_instances, "max": max_instances}

            return result

        except ImportError:
            return {"success": False, "error": "pip install boto3 sagemaker"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _get_inference_image(self, framework: str, instance_type: str) -> str:
        """Get the appropriate SageMaker inference container URI."""
        # Determine GPU vs CPU from the actual instance FAMILY (the
        # segment between "ml." and the size, e.g. "g4dn" in
        # "ml.g4dn.xlarge") — not a naive substring search. The previous
        # check ("g" in instance_type) matched on the SIZE suffix instead
        # (every "large"/"xlarge"/"2xlarge" contains the letter "g"),
        # misclassifying common CPU-only instances like "ml.c5.xlarge" and
        # "ml.m5.large" as needing the GPU container image.
        gpu_families = ("g", "p", "inf", "trn")  # g4dn/g5, p2/p3/p4d, inf1/inf2, trn1
        parts = instance_type.split(".")
        family = parts[1] if len(parts) >= 3 else ""
        gpu = "gpu" if family.startswith(gpu_families) else "cpu"
        images = {
            "onnx": f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-inference:2.1-{gpu}-py310",
            "pytorch": f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-inference:2.1-{gpu}-py310",
            "tensorflow": f"763104351884.dkr.ecr.{self.region}.amazonaws.com/tensorflow-inference:2.13-{gpu}",
        }
        return images.get(framework, images["pytorch"])

    def _setup_autoscaling(self, endpoint_name: str, min_cap: int, max_cap: int) -> None:
        try:
            import boto3
            asg = boto3.client("application-autoscaling", region_name=self.region)
            resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
            asg.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                MinCapacity=min_cap, MaxCapacity=max_cap,
            )
            asg.put_scaling_policy(
                PolicyName=f"{endpoint_name}-autoscale",
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": 70.0,
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                    },
                    "ScaleOutCooldown": 60, "ScaleInCooldown": 300,
                },
            )
            logger.info("Auto-scaling configured: %d-%d instances", min_cap, max_cap)
        except Exception as exc:
            logger.warning("Auto-scaling setup failed: %s", exc)

    def predict(self, endpoint_name: str, input_data: Any, argmax: bool = False) -> Any:
        """Send one chip (or a batch) to the endpoint and return the decoded JSON response.

        For an endpoint made by :meth:`deploy` from an ``.onnx`` file the response is
        ``{"predictions": ...}``: logits ``(B, classes, H, W)``, or class ids ``(B, H, W)`` with
        ``argmax=True`` (about 20-30 times smaller; use it to stay under SageMaker's 6 MB response
        limit). A NumPy array is sent as ``{"inputs": array.tolist(), "argmax": ...}``; any other
        object is sent as JSON unchanged. Service errors (bad credentials, unknown endpoint, payload
        too large) are raised by boto3; only a missing SDK is returned as ``{"error": ...}``.
        """
        try:
            import json as _json

            import boto3
            import numpy as np
            runtime = boto3.client("sagemaker-runtime", region_name=self.region)
            if isinstance(input_data, np.ndarray):
                payload = _json.dumps({"inputs": input_data.tolist(), "argmax": bool(argmax)})
            else:
                payload = _json.dumps(input_data)
            resp = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload,
            )
            return _json.loads(resp["Body"].read())
        except ImportError:
            return {"error": "pip install boto3"}

    def delete_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        try:
            import boto3
            sm = boto3.client("sagemaker", region_name=self.region)
            sm.delete_endpoint(EndpointName=endpoint_name)
            return {"success": True, "deleted": endpoint_name}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_endpoints(self) -> list[str]:
        try:
            resp = self._get_client().list_endpoints(StatusEquals="InService")
            return [e["EndpointName"] for e in resp.get("Endpoints", [])]
        except Exception as exc:
            logger.warning("could not list SageMaker endpoints (an empty list does NOT mean there are none): %s", exc)
            return []

    def generate_iam_policy(self) -> str:
        """IAM policy JSON for the *user or role that runs* ``deploy`` / ``predict`` / ``delete_endpoint``.

        Covers the S3 upload, creating, describing, invoking and deleting the endpoint, passing the
        execution role to SageMaker, and the auto-scaling calls. The SageMaker *execution role*
        (``role_arn``) is a different identity and needs its own permissions (read the bucket, pull the
        container image from ECR, write CloudWatch logs). Tighten the ``Resource`` entries for production.
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                 "Resource": [f"arn:aws:s3:::{self.bucket}/*", f"arn:aws:s3:::{self.bucket}"]},
                {"Effect": "Allow", "Action": [
                    "sagemaker:CreateModel", "sagemaker:CreateEndpointConfig", "sagemaker:CreateEndpoint",
                    "sagemaker:DescribeModel", "sagemaker:DescribeEndpointConfig", "sagemaker:DescribeEndpoint",
                    "sagemaker:InvokeEndpoint", "sagemaker:ListEndpoints",
                    "sagemaker:DeleteEndpoint", "sagemaker:DeleteEndpointConfig", "sagemaker:DeleteModel"],
                 "Resource": "*"},
                {"Effect": "Allow", "Action": "iam:PassRole",
                 "Resource": self.role_arn or "*",
                 "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
                {"Effect": "Allow", "Action": [
                    "application-autoscaling:RegisterScalableTarget",
                    "application-autoscaling:PutScalingPolicy",
                    "application-autoscaling:DescribeScalableTargets",
                    "iam:CreateServiceLinkedRole"],
                 "Resource": "*"},
            ]
        }
        return json.dumps(policy, indent=2)


class AzureDeployer(CloudDeployer):
    """Deploy to Azure Machine Learning (F4).

    Example::

        deployer = AzureDeployer(
            subscription_id="...", resource_group="...", workspace_name="pgv-workspace"
        )
        result = deployer.deploy("model.onnx", "pgv-buildings",
                                  vm_size="Standard_NC4as_T4_v3")
    """

    def __init__(
        self,
        subscription_id: str | None = None,
        resource_group: str | None = None,
        workspace_name: str | None = None,
        region: str = "eastus",
    ) -> None:
        super().__init__("azure", region)
        self.subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        self.resource_group  = resource_group  or os.environ.get("AZURE_RESOURCE_GROUP", "")
        self.workspace_name  = workspace_name  or os.environ.get("AZURE_ML_WORKSPACE", "")

    def deploy(
        self,
        model_path: str | Path,
        endpoint_name: str,
        vm_size: str = "Standard_NC4as_T4_v3",
        instance_count: int = 1,
    ) -> dict[str, Any]:
        """Register ``model_path`` and create a managed online endpoint with one deployment named ``default``.

        The deployment is created **without** an environment or scoring script, so Azure ML can only serve
        models it understands without code (for example MLflow models); a bare ONNX file will most likely
        be rejected at the deployment step. Extend the deployment with an ``Environment`` and
        ``CodeConfiguration`` for a custom model. Endpoints bill until deleted.
        """
        try:
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import (
                CodeConfiguration,
                Environment,
                ManagedOnlineDeployment,
                ManagedOnlineEndpoint,
            )
            from azure.ai.ml.entities import Model as AzModel
            from azure.identity import DefaultAzureCredential

            ml = MLClient(DefaultAzureCredential(),
                           self.subscription_id, self.resource_group, self.workspace_name)

            # Register model
            model = ml.models.create_or_update(
                AzModel(path=str(model_path), name=endpoint_name,
                         type="custom_model", description="PyGeoVision model")
            )
            logger.info("Azure model registered: %s v%s", model.name, model.version)

            # Create endpoint
            endpoint = ml.online_endpoints.begin_create_or_update(
                ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
            ).result()

            # Create deployment
            ml.online_deployments.begin_create_or_update(
                ManagedOnlineDeployment(
                    name="default",
                    endpoint_name=endpoint_name,
                    model=model,
                    instance_type=vm_size,
                    instance_count=instance_count,
                )
            ).result()

            return {
                "success": True,
                "endpoint_name": endpoint_name,
                "provider": "azure_ml",
                "scoring_uri": endpoint.scoring_uri,
                "vm_size": vm_size,
            }
        except ImportError:
            return {"success": False, "error": "pip install azure-ai-ml azure-identity"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        try:
            import json as _json

            import numpy as np
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            ml = MLClient(DefaultAzureCredential(), self.subscription_id,
                           self.resource_group, self.workspace_name)
            payload = {"inputs": input_data.tolist() if hasattr(input_data, "tolist") else input_data}
            with tempfile.TemporaryDirectory() as tmp:                 # invoke() wants a file path
                request = Path(tmp) / "request.json"
                request.write_text(_json.dumps(payload))
                return ml.online_endpoints.invoke(endpoint_name=endpoint_name, request_file=str(request))
        except ImportError:
            return {"error": "pip install azure-ai-ml"}

    def delete_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            ml = MLClient(DefaultAzureCredential(), self.subscription_id,
                           self.resource_group, self.workspace_name)
            ml.online_endpoints.begin_delete(name=endpoint_name).result()
            return {"success": True, "deleted": endpoint_name}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_endpoints(self) -> list[str]:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            ml = MLClient(DefaultAzureCredential(), self.subscription_id,
                           self.resource_group, self.workspace_name)
            return [e.name for e in ml.online_endpoints.list()]
        except Exception as exc:
            logger.warning("could not list Azure ML endpoints (an empty list does NOT mean there are none): %s", exc)
            return []


class GCPDeployer(CloudDeployer):
    """Deploy to Google Cloud Vertex AI (F4).

    Example::

        deployer = GCPDeployer(project_id="my-project", region="us-central1")
        result = deployer.deploy("model.onnx", "pgv-buildings",
                                  machine_type="n1-standard-4-nvidia-tesla-t4")
    """

    def __init__(self, project_id: str | None = None, region: str = "us-central1") -> None:
        super().__init__("gcp", region)
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")

    def deploy(
        self,
        model_path: str | Path,
        endpoint_name: str,
        machine_type: str = "n1-standard-4",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        gcs_bucket: str | None = None,
    ) -> dict[str, Any]:
        """Upload ``model_path`` to ``gs://<bucket>/pygeovision/models/<endpoint_name>/`` and deploy it on Vertex AI.

        ``gcs_bucket`` (or ``GCS_BUCKET``, default ``<project_id>-pgv``) must already exist. The model is
        served by Google's prebuilt PyTorch container, which expects a model archive/handler in the format
        that container documents; a bare ``.onnx`` file is uploaded as is, so it is not served unless you
        supply a matching container or archive. ``min_replica_count`` is 1 and ``max_replica_count`` 5.
        Endpoints bill until deleted.
        """
        try:
            from google.cloud import aiplatform, storage

            aiplatform.init(project=self.project_id, location=self.region)

            # Upload to GCS
            bucket = gcs_bucket or os.environ.get("GCS_BUCKET", f"{self.project_id}-pgv")
            gcs_client = storage.Client(project=self.project_id)
            gcs_bucket_obj = gcs_client.bucket(bucket)
            prefix = f"pygeovision/models/{endpoint_name}"           # one folder per endpoint
            blob = gcs_bucket_obj.blob(f"{prefix}/{Path(model_path).name}")
            blob.upload_from_filename(str(model_path))
            model_gcs_uri = f"gs://{bucket}/{prefix}/"
            logger.info("Model uploaded: %s", model_gcs_uri)

            # Upload model to Vertex AI
            container_image = (
                "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest"
                if accelerator_count > 0
                else "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest"
            )
            model = aiplatform.Model.upload(
                display_name=endpoint_name,
                artifact_uri=model_gcs_uri,
                serving_container_image_uri=container_image,
            )

            # Create endpoint and deploy
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            model.deploy(
                endpoint=endpoint,
                machine_type=machine_type,
                accelerator_type=accelerator_type if accelerator_count > 0 else None,
                accelerator_count=accelerator_count if accelerator_count > 0 else None,
                min_replica_count=1,
                max_replica_count=5,
            )

            return {
                "success": True,
                "endpoint_name": endpoint_name,
                "provider": "gcp_vertex",
                "endpoint_id": endpoint.name,
                "model_gcs_uri": model_gcs_uri,
            }
        except ImportError:
            return {"success": False, "error": "pip install google-cloud-aiplatform google-cloud-storage"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
            if not endpoints:
                return {"error": f"Endpoint {endpoint_name!r} not found"}
            endpoint = endpoints[0]
            payload = input_data.tolist() if hasattr(input_data, "tolist") else input_data
            result = endpoint.predict(instances=[payload])
            return result.predictions
        except ImportError:
            return {"error": "pip install google-cloud-aiplatform"}

    def delete_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
            for ep in endpoints:
                ep.undeploy_all()
                ep.delete()
            return {"success": True, "deleted": endpoint_name}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_endpoints(self) -> list[str]:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            return [e.display_name for e in aiplatform.Endpoint.list()]
        except Exception as exc:
            logger.warning("could not list Vertex AI endpoints (an empty list does NOT mean there are none): %s", exc)
            return []
