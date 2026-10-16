"""A custom serving container entry point for Vertex AI that serves an ONNX segmentation model.

Vertex AI starts the container and talks to it over HTTP on the port in $AIP_HTTP_PORT:
  GET  $AIP_HEALTH_ROUTE   -> 200 when the model is loaded
  POST $AIP_PREDICT_ROUTE  -> body {"instances": [<(C, H, W) nested list>, ...], "parameters": {"argmax": false}}
                              reply {"predictions": [<logits (classes, H, W) or class ids (H, W)>, ...]}
The model file is read from $AIP_STORAGE_URI's local copy, or from $MODEL_PATH when you bake it into the image.
Standard library plus numpy and onnxruntime only, so the image stays small.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import onnxruntime as ort

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

_wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
SESSION = ort.InferenceSession(MODEL_PATH, providers=[p for p in _wanted if p in ort.get_available_providers()])
INPUT_NAME = SESSION.get_inputs()[0].name


def predict(body: dict) -> dict:
    argmax = bool(body.get("parameters", {}).get("argmax", False))
    batch = np.asarray(body["instances"], dtype=np.float32)          # (B, C, H, W)
    out = SESSION.run(None, {INPUT_NAME: batch})[0]
    return {"predictions": (out.argmax(axis=1) if argmax else out).tolist()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - http.server's naming
        self._send(200, {"status": "ok"}) if self.path == HEALTH_ROUTE else self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != PREDICT_ROUTE:
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            self._send(200, predict(json.loads(self.rfile.read(length))))
        except Exception as exc:  # noqa: BLE001 - report any bad request as a 400, never crash the server
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", int(os.environ.get("AIP_HTTP_PORT", "8080"))), Handler).serve_forever()
