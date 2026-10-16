"""Accuracy assessment for objects.

Two different questions, two sets of tools:

**How good is the classification?**  :func:`confusion_report` compares reference
and predicted classes per *object*, optionally weighting each object by its area
(so a 5 ha field counts more than a 5 m² shed).  :func:`assess_classification`
gets the reference classes from polygons or points.  :func:`pixel_report` gives the
pixel-wise equivalent so the two views can be compared in class.

**How good is the segmentation?**  :func:`segmentation_quality` matches every
reference object to the segment overlapping it most and reports

* ``over_seg``  = 1 - |R∩S| / |R|   (0 = reference fully inside one segment)
* ``under_seg`` = 1 - |R∩S| / |S|   (0 = segment fully inside the reference)
* ``D``         = sqrt((over² + under²) / 2)   (0 = perfect; Clinton et al. 2010)
* ``QR``        = |R∩S| / |R∪S|   (1 = perfect; the Jaccard index)
* ``AFI``       = (|R| - |S|) / |R|   (>0: segment smaller than reference)

where R is a reference object and S its best-matching segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _report_from_matrix(cm: pd.DataFrame) -> dict:
    m = cm.to_numpy(dtype=float)
    total = m.sum()
    diag = np.diag(m)
    row, col = m.sum(1), m.sum(0)
    oa = diag.sum() / total if total else float("nan")
    pe = (row * col).sum() / (total ** 2) if total else float("nan")
    kappa = (oa - pe) / (1 - pe) if total and pe < 1 else (1.0 if oa == 1 else float("nan"))
    with np.errstate(divide="ignore", invalid="ignore"):
        prod = np.where(row > 0, diag / row, np.nan)        # recall: omission = 1 - this
        user = np.where(col > 0, diag / col, np.nan)        # precision: commission = 1 - this
        f1 = np.where((prod + user) > 0, 2 * prod * user / (prod + user), np.nan)
    per = pd.DataFrame({"producers_accuracy": prod, "users_accuracy": user, "f1": f1,
                        "reference_total": row, "predicted_total": col}, index=cm.index)
    return {"confusion": cm, "overall_accuracy": float(oa), "kappa": float(kappa),
            "per_class": per, "total": float(total)}


def confusion_report(reference: pd.Series, predicted: pd.Series, weights: pd.Series | None = None,
                     labels: list | None = None) -> dict:
    """Confusion matrix and accuracy measures for objects.

    ``reference``/``predicted`` are class names indexed by object id; only ids
    present in both are used.  With ``weights`` (e.g. object area) every cell
    sums weights instead of counting objects.  Rows = reference, columns =
    predicted.  ``overall_accuracy`` and ``kappa`` are weighted if weights are
    given.
    """
    ref = pd.Series(reference).dropna()
    pred = pd.Series(predicted)
    idx = ref.index.intersection(pred.index)
    if len(idx) == 0:
        raise ValueError("reference and predicted share no object ids")
    ref, pred = ref.loc[idx].astype(str), pred.loc[idx].astype(str)
    w = np.ones(len(idx)) if weights is None else pd.Series(weights).reindex(idx).to_numpy(float)
    if np.isnan(w).any():
        raise ValueError("weights are missing for some assessed objects")
    cls = labels if labels is not None else sorted(set(ref) | set(pred))
    cm = pd.DataFrame(0.0, index=cls, columns=cls)
    for r, p, wi in zip(ref, pred, w):
        if r in cm.index and p in cm.columns:
            cm.loc[r, p] += wi
    rep = _report_from_matrix(cm)
    rep["n_objects"] = int(len(idx))
    rep["weighted"] = weights is not None
    return rep


def assess_classification(result, labels, reference_gdf, class_column: str, transform=None,
                          weights: pd.Series | None = None, min_coverage: float | None = None, crs=None) -> dict:
    """Assess a :class:`ClassificationResult` against reference polygons/points.

    The reference class of an object is the class of the reference geometry that
    covers most of it (see :func:`~pygeovision.obia.classify.samples_from_geometries`).
    Only objects the reference reaches are assessed - report ``n_objects``
    alongside the accuracies. Pass ``crs`` (the image's coordinate system) when the reference is in another one
    and it is reprojected for you.
    """
    from .classify import samples_from_geometries
    ref = samples_from_geometries(labels, reference_gdf, class_column, transform, min_coverage, crs=crs)
    if ref.empty:
        raise ValueError("no object is covered by the reference geometries")
    return confusion_report(ref, result.classes, weights)


def pixel_report(predicted: np.ndarray, reference: np.ndarray, class_names: list[str]) -> dict:
    """Pixel-wise confusion for two code rasters (codes 1..K, 0 = ignore)."""
    p, r = np.asarray(predicted).ravel(), np.asarray(reference).ravel()
    keep = r > 0
    K = len(class_names)
    idx = r[keep].astype(int) - 1
    jdx = np.clip(p[keep].astype(int) - 1, -1, K - 1)
    cm = np.zeros((K, K + 1))
    np.add.at(cm, (idx, np.where(jdx < 0, K, jdx)), 1)
    # unclassified (code 0) pixels count as errors but have no column of their own
    unclassified = cm[:, K]
    rep = _report_from_matrix(pd.DataFrame(cm[:, :K], index=class_names, columns=class_names))
    rep["total"] = float(keep.sum())
    rep["unclassified_pixels"] = int(unclassified.sum())
    if rep["total"]:
        rep["overall_accuracy"] = float(np.trace(cm[:, :K]) / rep["total"])
    return rep


def segmentation_quality(seg_labels, ref_labels):
    """Compare a segmentation with reference objects.

    ``ref_labels``: integer raster, 0 = no reference, 1..n = one id per reference
    object (see :func:`~pygeovision.obia.vector.rasterize_geometries`).
    Returns ``(per_reference DataFrame, summary dict)``.
    """
    seg = np.asarray(seg_labels).astype(np.int64).ravel()
    ref = np.asarray(ref_labels).astype(np.int64).ravel()
    if seg.shape != ref.shape:
        raise ValueError("segmentation and reference rasters differ in shape")
    nseg = int(seg.max()) + 1
    seg_area = np.bincount(seg, minlength=nseg)
    m = ref > 0
    if not m.any():
        raise ValueError("reference raster contains no reference objects")
    ref_area = np.bincount(ref[m])
    key = ref[m] * nseg + seg[m]
    uniq, inter = np.unique(key, return_counts=True)
    df = pd.DataFrame({"ref": uniq // nseg, "seg": uniq % nseg, "inter": inter})
    df = df[df["seg"] > 0]                      # label 0 is "no data" (cloud), not a segment that could match
    if df.empty:
        raise ValueError("no reference object overlaps a segment (is the reference under cloud or on another grid?)")
    best = df.sort_values(["ref", "inter"], ascending=[True, False]).drop_duplicates("ref").set_index("ref")
    parts = df.assign(frac_in_ref=df["inter"] / seg_area[df["seg"].to_numpy()])
    n_parts = parts[parts["frac_in_ref"] >= 0.5].groupby("ref").size()
    out = pd.DataFrame(index=best.index)
    R = ref_area[best.index.to_numpy()].astype(float)
    S = seg_area[best["seg"].to_numpy()].astype(float)
    I = best["inter"].to_numpy().astype(float)
    out["best_segment"] = best["seg"]
    out["ref_area_px"], out["seg_area_px"], out["overlap_px"] = R, S, I
    out["over_seg"] = 1.0 - I / R
    out["under_seg"] = 1.0 - I / S
    out["D"] = np.sqrt((out["over_seg"] ** 2 + out["under_seg"] ** 2) / 2.0)
    out["QR"] = I / (R + S - I)
    out["AFI"] = (R - S) / R
    out["n_parts"] = n_parts.reindex(out.index).fillna(0).astype(int)
    out.index.name = "ref_id"
    summary = {"n_reference": int(len(out)),
               "mean_over_seg": float(out["over_seg"].mean()),
               "mean_under_seg": float(out["under_seg"].mean()),
               "mean_D": float(out["D"].mean()), "median_D": float(out["D"].median()),
               "mean_QR": float(out["QR"].mean()),
               "n_segments_total": int(len(np.unique(seg[seg > 0])))}
    return out, summary
