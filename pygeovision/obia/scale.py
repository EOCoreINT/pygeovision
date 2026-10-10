"""Choosing a scale parameter on evidence.

There is no universally right scale: it depends on the size of the objects you
care about.  Two complementary tools, both driven by a single merging run (merging
only ever joins objects, so one pass can be paused at each scale):

* :func:`scale_diagnostics` - *no reference needed.*  Tracks mean within-object
  variability (local variance, LV) as scale grows and its rate of change (ROC-LV).
  Peaks in ROC-LV mark scales where objects stop being homogeneous parts and become
  heterogeneous wholes - candidate scales (Drăguţ et al. 2010, the idea behind ESP).
  A heuristic, not a verdict: always look at the objects.
* :func:`scale_vs_reference` - *needs reference objects.*  Computes the
  segmentation-quality score D (0 = perfect) at every scale so you can see the
  trade-off between over- and under-segmentation and pick the minimum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .accuracy import segmentation_quality
from .io import as_raster_data, rescale_bands
from .merging import RegionMerger
from .segmentation import _slic


def _merger(image, rescale, color_weight, compactness_weight, init, init_segments, band_weights,
            band_names=None, transform=None, crs=None):
    rd = as_raster_data(image, band_names, transform, crs)
    img = rescale_bands(rd.image) if rescale else rd.image.astype(np.float64)
    H, W = img.shape[1:]
    if init == "auto":
        init = "pixels" if H * W <= 10_000 else "superpixels"
    init_labels = (np.arange(H * W).reshape(H, W) if init == "pixels"
                   else _slic(img, init_segments or max(H * W // 16, 50), 0.1, 0.0))
    return img, RegionMerger(img, init_labels, color_weight, compactness_weight, band_weights)


def _local_variance(img: np.ndarray, labels: np.ndarray) -> float:
    """Mean over objects and bands of the within-object standard deviation."""
    flat = labels.ravel()
    n = np.bincount(flat).astype(float)
    ok = n > 0
    lvs = []
    for b in range(img.shape[0]):
        v = img[b].ravel()
        m = np.bincount(flat, weights=v)[ok] / n[ok]
        sq = np.bincount(flat, weights=v * v)[ok] / n[ok]
        lvs.append(np.sqrt(np.maximum(sq - m * m, 0.0)).mean())
    return float(np.mean(lvs))


def scale_diagnostics(image, scales, *, rescale=True, color_weight=0.9, compactness_weight=0.5,
                      init="auto", init_segments=None, band_weights=None, **kw) -> pd.DataFrame:
    """Object count, LV and ROC-LV at each scale.

    Returns a DataFrame indexed by scale with ``n_objects``, ``lv``, ``roc_lv``
    (percent change in LV from the previous scale) and ``candidate`` (True at local
    maxima of ROC-LV).
    """
    scales = sorted(float(s) for s in scales)
    if len(scales) < 3:
        raise ValueError("give at least 3 scales")
    img, m = _merger(image, rescale, color_weight, compactness_weight, init, init_segments,
                     band_weights, **kw)
    rows = []
    for s in scales:
        m.run(s)
        lab = m.labels()
        rows.append({"scale": s, "n_objects": int(lab.max()), "lv": _local_variance(img, lab)})
    df = pd.DataFrame(rows).set_index("scale")
    df["roc_lv"] = df["lv"].pct_change() * 100.0
    r = df["roc_lv"].to_numpy()
    cand = np.zeros(len(r), dtype=bool)
    for i in range(1, len(r) - 1):
        if np.isfinite(r[i - 1]) and np.isfinite(r[i]) and np.isfinite(r[i + 1]):
            cand[i] = r[i] > r[i - 1] and r[i] > r[i + 1]
    df["candidate"] = cand
    return df


def scale_vs_reference(image, scales, reference_labels, *, rescale=True, color_weight=0.9,
                       compactness_weight=0.5, init="auto", init_segments=None,
                       band_weights=None, **kw) -> pd.DataFrame:
    """Segmentation quality at each scale against reference objects.

    Columns: ``n_objects``, ``mean_over_seg``, ``mean_under_seg``, ``mean_D``,
    ``mean_QR``.  The scale with the smallest ``mean_D`` is the best compromise for
    these reference objects.
    """
    scales = sorted(float(s) for s in scales)
    _, m = _merger(image, rescale, color_weight, compactness_weight, init, init_segments,
                   band_weights, **kw)
    rows = []
    for s in scales:
        m.run(s)
        lab = m.labels()
        _, summ = segmentation_quality(lab, reference_labels)
        rows.append({"scale": s, "n_objects": int(lab.max()),
                     "mean_over_seg": summ["mean_over_seg"], "mean_under_seg": summ["mean_under_seg"],
                     "mean_D": summ["mean_D"], "mean_QR": summ["mean_QR"]})
    return pd.DataFrame(rows).set_index("scale")
