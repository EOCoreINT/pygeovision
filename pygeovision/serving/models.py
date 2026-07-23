"""Pydantic request/response models for the inference API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback dataclass
    class BaseModel:
        pass

    def Field(default=None, **kw):
        return default


class PredictRequest(BaseModel):
    """Request model for single image prediction."""
    image_b64: str | None = Field(None, description="Base64-encoded image bytes")
    image_url: str | None = Field(None, description="Accessible image URL")
    model_name: str = Field("default", description="Registered model name")
    task: str = Field("segmentation", description="Inference task")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    chip_size: int = Field(512, ge=32, le=2048)
    overlap: int = Field(64, ge=0, le=512)
    return_probabilities: bool = Field(False)
    output_format: str = Field("geotiff", description="geotiff|png|json")
    extra: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""
    image_urls: list[str] = Field(..., min_length=1, max_length=100)
    model_name: str = "default"
    task: str = "segmentation"
    confidence_threshold: float = 0.5
    async_mode: bool = Field(True, description="Process asynchronously")


class PredictResponse(BaseModel):
    """Response model for prediction results."""
    success: bool
    model_name: str
    task: str
    n_classes: int | None = None
    output_url: str | None = None
    output_b64: str | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    inference_time_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    error: str | None = None


class ModelInfo(BaseModel):
    """Model registration information."""
    name: str
    task: str
    num_classes: int
    in_channels: int = 4
    description: str = ""
    version: str = "1.0.0"
    onnx_path: str | None = None
    pytorch_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    gpu_available: bool
    memory_mb: float
    uptime_s: float


# `from __future__ import annotations` (PEP 563) makes every annotation in
# this module a deferred string. Pydantic v2 must re-resolve those forward
# references against the fully-populated module namespace before the models
# are usable — without this, FastAPI's schema generation fails with an
# opaque "not fully defined" error the first time any endpoint using these
# models is actually invoked.
for _model in (PredictRequest, BatchPredictRequest, PredictResponse, ModelInfo, HealthResponse):
    if hasattr(_model, "model_rebuild"):
        _model.model_rebuild()
