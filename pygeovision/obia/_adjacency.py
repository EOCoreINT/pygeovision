"""Neighbourhood helpers shared by segmentation, features and accuracy code.

All functions work on a 2-D integer label raster and use 4-connectivity
(a pixel's neighbours are the pixels above, below, left and right).
Border length is counted in *pixel edges*: two objects that touch along a
straight line of 10 pixels share a border of length 10.
"""
from __future__ import annotations

import numpy as np


def border_pairs(labels: np.ndarray):
    """Shared border lengths between touching objects.

    Returns three equal-length arrays ``(a, b, length)`` with ``a < b``:
    object ``a`` and object ``b`` share ``length`` pixel edges.
    """
    labels = np.asarray(labels)
    left, right = labels[:, :-1], labels[:, 1:]
    up, down = labels[:-1, :], labels[1:, :]
    m1, m2 = left != right, up != down
    a = np.concatenate([left[m1], up[m2]]).astype(np.int64)
    b = np.concatenate([right[m1], down[m2]]).astype(np.int64)
    if a.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e.copy(), e.copy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    base = int(hi.max()) + 1
    uniq, counts = np.unique(lo * base + hi, return_counts=True)
    return uniq // base, uniq % base, counts.astype(np.int64)


def image_edge_lengths(labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Number of pixel edges each object has on the image frame.

    ``n_labels`` is the size of the returned array (index = label id).
    """
    labels = np.asarray(labels)
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    return np.bincount(edge.astype(np.int64), minlength=n_labels).astype(np.int64)


def adjacency_dict(labels: np.ndarray) -> dict[int, dict[int, int]]:
    """``{object: {neighbour: shared_border_length}}`` for every object."""
    a, b, ln = border_pairs(labels)
    adj: dict[int, dict[int, int]] = {int(i): {} for i in np.unique(labels)}
    for x, y, l in zip(a.tolist(), b.tolist(), ln.tolist()):
        adj[x][y] = l
        adj[y][x] = l
    return adj
