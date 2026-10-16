"""Per-pixel embeddings: classify from a few labelled points, and find pixels that look like a reference pixel.

The functions take a plain array ``(channels, height, width)``, which is what ``TesseraGeo.embeddings_for_bbox`` returns under
the key ``"embedding"``. Nothing here needs a network, so the same code runs on a stand-in array in the tests.
"""
from __future__ import annotations

import numpy as np


def classify_pixels(embedding: np.ndarray, points: list[tuple[int, int]], labels: list[int]) -> np.ndarray:
    """Train a logistic regression on the embedding vectors at labelled (row, col) points; predict every pixel.

    Returns a ``(height, width)`` array of class ids; pixels whose embedding is not finite get ``-1``.
    """
    from sklearn.linear_model import LogisticRegression

    c, h, w = embedding.shape
    rows, cols = np.asarray(points).T
    x_train = embedding[:, rows, cols].T                       # (n_points, channels)
    if not np.isfinite(x_train).all():
        raise ValueError("some labelled points fall on pixels without an embedding (NaN); move or drop them")
    model = LogisticRegression(max_iter=1000).fit(x_train, labels)

    flat = embedding.reshape(c, -1).T                          # (pixels, channels)
    valid = np.isfinite(flat).all(axis=1)
    out = np.full(h * w, -1, dtype=np.int32)
    out[valid] = model.predict(flat[valid])
    return out.reshape(h, w)


def similarity_map(embedding: np.ndarray, row: int, col: int) -> np.ndarray:
    """Cosine similarity of every pixel's embedding to the one at (row, col). Values in [-1, 1]; NaN where there is none."""
    c, h, w = embedding.shape
    flat = embedding.reshape(c, -1).T.astype(np.float64)
    ref = flat[row * w + col]
    norms = np.linalg.norm(flat, axis=1) * np.linalg.norm(ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = flat @ ref / norms
    return sim.reshape(h, w).astype(np.float32)
