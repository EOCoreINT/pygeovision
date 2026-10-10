"""Tests for pygeovision.serving — FastAPI inference server.

Uses FastAPI's TestClient (no real network/model required); model-backed
inference paths are exercised through the honest "no model loaded" note
path since loading a real ONNX/torch model isn't available in this
environment. What matters here is that routing, request/response schemas,
and the request-counting/batch logic actually work — this module had
real, previously-unverified bugs (nested-function forward-ref resolution
breaking every endpoint; batch/metrics/websocket being non-functional
stubs) found and fixed during a stub audit.
"""
import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed — pip install fastapi")

from fastapi.testclient import TestClient

from pygeovision.serving.api import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthAndMeta:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200


class TestModelRegistry:
    def test_register_and_list(self, client):
        r = client.post("/models/register", json={
            "name": "seg1", "task": "segmentation", "num_classes": 2,
        })
        assert r.status_code == 200
        assert r.json()["registered"] == "seg1"

        r = client.get("/models")
        assert r.status_code == 200
        names = [m["name"] for m in r.json()["models"]]
        assert "seg1" in names

    def test_deregister(self, client):
        client.post("/models/register", json={
            "name": "temp", "task": "segmentation", "num_classes": 2,
        })
        r = client.delete("/models/temp")
        assert r.status_code == 200
        r = client.get("/models")
        assert "temp" not in [m["name"] for m in r.json()["models"]]


class TestPredict:
    def test_predict_no_model_returns_honest_note(self, client):
        """With no model registered, /predict must succeed with an honest
        'no model loaded' note rather than crashing or faking a result."""
        r = client.post("/predict", json={
            "image_url": "https://example.com/scene.tif", "model_name": "default",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert "note" in body["statistics"]

    def test_predict_unregistered_named_model_404s(self, client):
        r = client.post("/predict", json={
            "image_url": "https://example.com/scene.tif", "model_name": "nonexistent",
        })
        assert r.status_code == 404


class TestBatchPredict:
    def test_batch_returns_real_per_item_results(self, client):
        """Regression test: /predict/batch used to unconditionally return
        {"status": "queued", ...} and do nothing. It must now actually
        process every URL and return one result per image."""
        r = client.post("/predict/batch", json={
            "image_urls": ["https://example.com/a.tif", "https://example.com/b.tif"],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["n_images"] == 2
        assert len(body["results"]) == 2
        assert all("image_url" in r for r in body["results"])

    def test_batch_unregistered_named_model_404s(self, client):
        r = client.post("/predict/batch", json={
            "image_urls": ["https://example.com/a.tif"], "model_name": "nonexistent",
        })
        assert r.status_code == 404


class TestMetrics:
    def test_requests_total_increments(self, client):
        """Regression test: requests_total used to be hardcoded to 0."""
        r0 = client.get("/metrics")
        assert r0.json()["requests_total"] == 0

        client.post("/predict", json={"image_url": "https://example.com/x.tif"})
        r1 = client.get("/metrics")
        assert r1.json()["requests_total"] == 1

        client.post("/predict/batch", json={
            "image_urls": ["https://example.com/a.tif", "https://example.com/b.tif"],
        })
        r2 = client.get("/metrics")
        assert r2.json()["requests_total"] == 3
