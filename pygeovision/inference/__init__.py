"""
PyGeoVision Advanced Inference Engine (B1-B6).
Tiled inference with Gaussian blending, batch processing, memory-efficient streaming.
Fully native — no external AI-platform dependency.
"""
from pygeovision.inference.batch import BatchInferenceEngine
from pygeovision.inference.stream import EnsembleInference, StreamingInference
from pygeovision.inference.tiled import GaussianBlend, TiledInference

__all__ = [
    "TiledInference", "GaussianBlend",
    "BatchInferenceEngine", "StreamingInference", "EnsembleInference",
]
