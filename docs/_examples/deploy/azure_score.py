"""A scoring script for an Azure ML managed online endpoint that serves an ONNX segmentation model.

Azure ML calls init() once when the container starts and run() for every request.
Request body : {"inputs": <(C, H, W) or (B, C, H, W) nested list>, "argmax": false}
Response     : {"predictions": <logits (B, classes, H, W), or class ids (B, H, W) when "argmax" is true>}
It speaks the same JSON as the SageMaker handler of PyGeoVision (pygeovision.cloud.SAGEMAKER_INFERENCE_PY).
"""
import json
import os

import numpy as np
import onnxruntime as ort

session = None


def init():
    """Load the first .onnx file found in the registered model's folder."""
    global session
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    found = [os.path.join(root, f) for root, _, files in os.walk(model_dir) for f in files if f.endswith(".onnx")]
    if not found:
        raise FileNotFoundError(f"no .onnx file under {model_dir}")
    wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
    session = ort.InferenceSession(sorted(found)[0], providers=[p for p in wanted if p in ort.get_available_providers()])


def run(raw_data):
    """Score one request (a JSON string) and return a JSON-serialisable dictionary."""
    payload = json.loads(raw_data)
    argmax = bool(payload.get("argmax", False)) if isinstance(payload, dict) else False
    array = np.asarray(payload["inputs"] if isinstance(payload, dict) else payload, dtype=np.float32)
    if array.ndim == 3:                                   # one chip: add the batch axis
        array = array[None]
    out = session.run(None, {session.get_inputs()[0].name: array})[0]
    return {"predictions": (out.argmax(axis=1) if argmax else out).tolist()}
