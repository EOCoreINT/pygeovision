"""Top-down segmentation: start from the whole image and *split*.

Region merging (:mod:`pygeovision.obia.merging`) builds objects from the bottom up. The three algorithms here work
the other way round or classify first:

``quadtree``
    Split a block into four until every band inside it is homogeneous (standard deviation <= ``scale``).
    Homogeneous areas become a few large blocks, detailed areas many small ones.
``multi_threshold``
    Cut one layer (a band, brightness, NDVI, NDWI ...) into classes with thresholds - given, or found with
    multi-Otsu - and let every connected patch of one class be an object.
``contrast_split``
    Repeatedly split every object into a *dark* and a *bright* part. The threshold is the one that maximises the
    **edge contrast**: the mean absolute difference between neighbouring pixels that lie on opposite sides of it.

All three take a ``(rows, cols)`` layer or a ``(bands, rows, cols)`` image and return an *unconnected-id* label
array; :func:`pygeovision.obia.segment` makes the ids connected and applies the validity mask.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------------- quadtree


def quadtree(image, scale: float = 10.0, min_size: int = 1, max_size: int | None = None, band_weights=None):
    """Quadtree segmentation.

    A block is split into four quarters (rows split at ``h // 2``, columns at ``w // 2``, so any image size works)
    while the standard deviation of *any* band inside it exceeds ``scale`` - a block is homogeneous only when every
    band is. Splitting stops when a block can no longer be halved without a side getting shorter than
    ``min_size``. Blocks larger than ``max_size`` (rows or columns) are split regardless of homogeneity.

    Parameters
    ----------
    image : (bands, rows, cols) or (rows, cols) array.
    scale : largest tolerated population standard deviation, in the units of ``image`` (0-255 after rescaling).
    min_size : smallest allowed block side in pixels (>= 1).
    max_size : optional largest allowed block side in pixels.
    band_weights : bands with weight 0 are ignored by the homogeneity test (default: all bands count).

    Returns
    -------
    (rows, cols) int32 array, one id per leaf block (1..n), so ids are *not* unique per connected object: two
    neighbouring blocks stay separate objects.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = img[None]
    if img.ndim != 3:
        raise ValueError("image must be (bands, rows, cols) or (rows, cols)")
    if scale < 0:
        raise ValueError("scale must be >= 0")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if max_size is not None and max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    B, H, W = img.shape
    use = np.ones(B, bool) if band_weights is None else np.asarray(band_weights, float) > 0
    if use.shape != (B,) or not use.any():
        raise ValueError("band_weights must give one non-negative weight per band, at least one > 0")
    img = img[use]

    # Integral images of the centred values (centring keeps the variance numerically exact for large images).
    c = img - img.mean(axis=(1, 2), keepdims=True)
    S = np.zeros((c.shape[0], H + 1, W + 1))
    SS = np.zeros_like(S)
    S[:, 1:, 1:] = c.cumsum(1).cumsum(2)
    SS[:, 1:, 1:] = (c * c).cumsum(1).cumsum(2)

    def box(T, r0, r1, c0, c1):
        return T[:, r1, c1] - T[:, r0, c1] - T[:, r1, c0] + T[:, r0, c0]

    r0 = np.array([0])
    r1 = np.array([H])
    c0 = np.array([0])
    c1 = np.array([W])
    leaves = []
    while r0.size:
        h, w = r1 - r0, c1 - c0
        can_r = (h >= 2) & (h // 2 >= min_size)
        can_c = (w >= 2) & (w // 2 >= min_size)
        n = (h * w).astype(np.float64)
        mean = box(S, r0, r1, c0, c1) / n
        var = np.maximum(box(SS, r0, r1, c0, c1) / n - mean * mean, 0.0)
        # tiny relative slack: float error must not make an exactly-at-threshold block "heterogeneous"
        hetero = (np.sqrt(var).max(axis=0) > scale * (1.0 + 1e-12) + 1e-9)
        if max_size is not None:
            hetero |= (h > max_size) | (w > max_size)
        split = hetero & (can_r | can_c)
        leaves.append((r0[~split], r1[~split], c0[~split], c1[~split]))
        r0, r1, c0, c1, can_r, can_c = (a[split] for a in (r0, r1, c0, c1, can_r, can_c))
        rm = np.where(can_r, r0 + (r1 - r0) // 2, r1)
        cm = np.where(can_c, c0 + (c1 - c0) // 2, c1)
        # children: (top|all rows) x (left|all cols), then the remaining halves where a split happened
        nr0 = [r0, r0, rm, rm]
        nr1 = [rm, rm, r1, r1]
        nc0 = [c0, cm, c0, cm]
        nc1 = [cm, c1, cm, c1]
        keep = [np.ones_like(can_r), can_c, can_r, can_r & can_c]
        r0 = np.concatenate([a[k] for a, k in zip(nr0, keep)])
        r1 = np.concatenate([a[k] for a, k in zip(nr1, keep)])
        c0 = np.concatenate([a[k] for a, k in zip(nc0, keep)])
        c1 = np.concatenate([a[k] for a, k in zip(nc1, keep)])

    out = np.zeros((H, W), dtype=np.int32)
    k = 0
    for a, b, cc, d in leaves:
        for i in range(a.size):
            k += 1
            out[a[i] : b[i], cc[i] : d[i]] = k
    return out


# ------------------------------------------------------------------------------------------------ multi-threshold


def multi_threshold(layer, thresholds=None, n_classes: int = 3, valid=None):
    """Segment by cutting one layer into classes.

    Parameters
    ----------
    layer : (rows, cols) float array.
    thresholds : increasing cut values, or ``None`` to find ``n_classes - 1`` of them with multi-Otsu.
        A pixel falls in class ``k`` when ``thresholds[k-1] <= value < thresholds[k]`` (values equal to a threshold
        belong to the upper class).
    n_classes : number of classes for multi-Otsu (>= 2).
    valid : optional mask of pixels used to find the Otsu thresholds.

    Returns
    -------
    (labels, thresholds): labels are the connected patches of each class (ids 1..n, 4-connectivity); the
    thresholds are returned as a tuple of floats.
    """
    from scipy import ndimage as ndi

    arr = np.asarray(layer, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("layer must be (rows, cols)")
    if thresholds is None:
        from skimage.filters import threshold_multiotsu

        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        vals = arr if valid is None else arr[valid]
        if np.unique(vals).size < n_classes:
            raise ValueError(f"layer has fewer than {n_classes} distinct values; cannot find {n_classes} classes")
        thresholds = threshold_multiotsu(vals, classes=int(n_classes))
    thr = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    if thr.size == 0 or (np.diff(thr) <= 0).any():
        raise ValueError("thresholds must be a non-empty, strictly increasing sequence")
    cls = np.digitize(arr, thr)  # right=False: value == threshold -> upper class
    out = np.zeros(arr.shape, dtype=np.int32)
    nxt = 0
    for k in range(thr.size + 1):
        comp, n = ndi.label(cls == k)  # default structure = 4-connectivity
        sel = comp > 0
        out[sel] = comp[sel] + nxt
        nxt += n
    return out, tuple(float(t) for t in thr)


# -------------------------------------------------------------------------------------------------- contrast split


def _pairs(parent: np.ndarray, layer: np.ndarray):
    """Horizontally / vertically adjacent pixel pairs lying in the same parent: (parent, low, high)."""
    ps, los, his = [], [], []
    for sl_a, sl_b in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pa, pb = parent[sl_a], parent[sl_b]
        keep = (pa == pb) & (pa > 0)
        va, vb = layer[sl_a][keep], layer[sl_b][keep]
        ps.append(pa[keep])
        los.append(np.minimum(va, vb))
        his.append(np.maximum(va, vb))
    return np.concatenate(ps), np.concatenate(los), np.concatenate(his)


def contrast_split(layer, levels: int = 2, steps: int = 32, min_contrast: float = 0.0, min_size: int = 1, init=None):
    """Contrast split segmentation: divide objects into dark and bright parts, level by level.

    For every object, ``steps`` candidate thresholds are spread evenly strictly between its lowest and highest
    value: ``t_k = lo + (hi - lo) * k / (steps + 1)``, ``k = 1..steps``. A threshold ``t`` splits the object into
    the pixels ``<= t`` (dark) and ``> t`` (bright); its **edge contrast** is the mean of ``|a - b|`` over all
    4-neighbour pixel pairs inside the object with ``a <= t < b``. The threshold with the largest edge contrast is
    used (the lowest on ties), provided each side has at least ``min_size`` pixels and the contrast is at least
    ``min_contrast``. The dark and bright parts then become connected objects and the next level starts from
    those.

    Parameters
    ----------
    layer : (rows, cols) float array.
    levels : number of splitting rounds (>= 1). After ``L`` rounds an object has been cut at most ``L`` times.
    steps : candidate thresholds per object (>= 1).
    min_contrast : do not split when even the best edge contrast (in layer units) is below this.
    min_size : each of the dark and bright parts must have at least this many pixels.
    init : optional (rows, cols) labels to start from instead of the whole image (0 = excluded).

    Returns
    -------
    (rows, cols) int32 labels, connected objects numbered from 1.
    """
    arr = np.asarray(layer, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("layer must be (rows, cols)")
    if levels < 1 or steps < 1 or min_size < 1:
        raise ValueError("levels, steps and min_size must be >= 1")
    if init is None:
        parent = np.ones(arr.shape, dtype=np.int64)
    else:
        parent = np.asarray(init).astype(np.int64)
        if parent.shape != arr.shape:
            raise ValueError("init shape does not match the layer")
    parent = _connected(parent)

    for _ in range(levels):
        n = int(parent.max())
        if n == 0:
            break
        flat_p = parent.ravel()
        inside = flat_p > 0
        lo = np.full(n + 1, np.inf)
        hi = np.full(n + 1, -np.inf)
        np.minimum.at(lo, flat_p[inside], arr.ravel()[inside])
        np.maximum.at(hi, flat_p[inside], arr.ravel()[inside])
        lo[0] = hi[0] = 0.0  # id 0 is 'outside'; keep the arithmetic finite
        pp, plo, phi = _pairs(parent, arr)
        pdiff = phi - plo

        best_c = np.full(n + 1, -np.inf)
        best_t = np.full(n + 1, np.nan)
        for k in range(1, steps + 1):
            t = lo + (hi - lo) * (k / (steps + 1))  # per-parent threshold
            m = (plo <= t[pp]) & (t[pp] < phi)
            cnt = np.bincount(pp[m], minlength=n + 1)
            tot = np.bincount(pp[m], weights=pdiff[m], minlength=n + 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                contrast = np.where(cnt > 0, tot / np.maximum(cnt, 1), -np.inf)
            n_dark = np.bincount(flat_p[inside], weights=(arr.ravel()[inside] <= t[flat_p[inside]]), minlength=n + 1)
            size = np.bincount(flat_p[inside], minlength=n + 1)
            ok = (n_dark >= min_size) & (size - n_dark >= min_size) & (contrast >= min_contrast) & (cnt > 0)
            better = ok & (contrast > best_c + 1e-12)  # strict: lowest threshold wins ties
            best_c[better] = contrast[better]
            best_t[better] = t[better]

        split = np.isfinite(best_t)
        if not split.any():
            break
        side = np.zeros(arr.shape, dtype=np.int64)
        has = parent > 0
        side[has] = arr[has] > np.where(split, best_t, np.inf)[parent[has]]
        # parents that were not split keep side 0, so their ids stay whole
        parent = _connected(np.where(has, parent * 2 + side, 0))
    return parent.astype(np.int32)


def _connected(lab: np.ndarray) -> np.ndarray:
    """Relabel so each id is one 4-connected patch (ids 1..n), 0 stays 0."""
    from scipy import ndimage as ndi

    lab = np.asarray(lab, dtype=np.int64)
    out = np.zeros(lab.shape, dtype=np.int64)
    nxt = 0
    ids = np.unique(lab)
    ids = ids[ids > 0]
    comp = np.zeros(lab.shape, dtype=np.int64)
    # label each id inside its bounding box (cost ~ pixels, not ids x pixels)
    comp_ids = np.zeros(int(ids.max()) + 1 if ids.size else 1, dtype=np.int64)
    comp_ids[ids] = np.arange(1, ids.size + 1)
    dense = np.where(lab > 0, comp_ids[lab], 0)
    for v, sl in enumerate(ndi.find_objects(dense), start=1):
        if sl is None:
            continue
        sub = dense[sl] == v
        c, k = ndi.label(sub)
        view = out[sl]
        view[sub] = c[sub] + nxt
        nxt += k
    return out
