"""Object features: the table every OBIA classifier works on.

:func:`extract_features` turns a label raster + image into a pandas
``DataFrame`` with one row per object (index = object id) and these groups of
columns, named so they can be used directly in rule strings such as
``"mean_nir > 120 and compactness > 0.6"``:

spectral   ``mean_<layer>``, ``std_<layer>``, ``min_<layer>``, ``max_<layer>``,
           ``brightness``
geometry   ``area_px`` (``area_m2``), ``perimeter_px`` (pixel edges),
           ``compactness`` (4πA/P²), ``shape_index`` (P/4√A),
           ``elongation``, ``solidity``, ``extent``, ``orientation``
texture    ``glcm_<property>_<layer>`` (optional, per-object GLCM)
context    ``n_neighbors``, ``rel_border_image_edge``,
           ``diff_to_neighbors_<layer>``

Class-dependent context (``rel_border_<class>``) and hierarchy features
(``parent_*``, sub-object fractions) need a first classification or a second
segmentation level, so they have their own functions below.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ._adjacency import border_pairs, image_edge_lengths
from .io import RasterData

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation")


def clean_name(name: str) -> str:
    """Make a layer/class name safe to use inside a pandas expression."""
    s = re.sub(r"\W+", "_", str(name).strip()).strip("_")
    return ("_" + s) if s and s[0].isdigit() else (s or "layer")


def _layers(image, band_names, extra_layers):
    if isinstance(image, RasterData):
        arr, names = image.image, image.band_names
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        names = list(band_names) if band_names else [f"b{i + 1}" for i in range(arr.shape[0])]
    layers = {clean_name(n): arr[i].astype(np.float64) for i, n in enumerate(names)}
    n_bands = len(layers)
    for k, v in (extra_layers or {}).items():
        v = np.asarray(v, dtype=np.float64)
        if v.shape != arr.shape[1:]:
            raise ValueError(f"extra layer {k!r} has shape {v.shape}, expected {arr.shape[1:]}")
        layers[clean_name(k)] = v
    return layers, n_bands


def glcm_texture(labels, layer, levels: int = 16, distance: int = 1,
                 min_pixels: int = 9, ids=None, prefix: str = "glcm") -> pd.DataFrame:
    """Per-object Haralick GLCM texture.

    The layer is quantised to ``levels`` grey levels using global 2-98 %
    percentiles.  For each object only pixel pairs where *both* pixels belong
    to the object are counted (pixels outside are masked out, not treated as
    a grey level), pooled over four directions at the given distance, then
    normalised.  Objects smaller than ``min_pixels`` get NaN.
    """
    from skimage.feature import graycomatrix

    lab = np.asarray(labels)
    lo, hi = np.percentile(layer, [2, 98])
    q = np.zeros(layer.shape, dtype=np.uint8) if hi <= lo else \
        np.clip(((layer - lo) / (hi - lo)) * levels, 0, levels - 1).astype(np.uint8)
    ids = np.unique(lab[lab > 0]) if ids is None else np.asarray(ids)
    slices = ndi.find_objects(lab.astype(np.int64))
    i_idx, j_idx = np.indices((levels, levels))
    diff = (i_idx - j_idx).astype(np.float64)
    rows = {}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for oid in ids:
        sl = slices[oid - 1]
        mask = lab[sl] == oid
        if mask.sum() < min_pixels:
            rows[oid] = dict.fromkeys(GLCM_PROPS, np.nan)
            continue
        crop = np.where(mask, q[sl], levels).astype(np.uint8)
        g = graycomatrix(crop, [distance], angles, levels=levels + 1, symmetric=True)
        g = g[:levels, :levels, 0, :].sum(axis=2).astype(np.float64)
        tot = g.sum()
        if tot == 0:
            rows[oid] = dict.fromkeys(GLCM_PROPS, np.nan)
            continue
        p = g / tot
        pi, pj = p.sum(1), p.sum(0)
        gi = np.arange(levels)
        mi, mj = (gi * pi).sum(), (gi * pj).sum()
        si = np.sqrt(((gi - mi) ** 2 * pi).sum())
        sj = np.sqrt(((gi - mj) ** 2 * pj).sum())
        corr = 1.0 if si * sj < 1e-12 else float(((i_idx - mi) * (j_idx - mj) * p).sum() / (si * sj))
        nz = p > 0
        rows[oid] = {
            "contrast": float((p * diff ** 2).sum()),
            "dissimilarity": float((p * np.abs(diff)).sum()),
            "homogeneity": float((p / (1.0 + diff ** 2)).sum()),
            "energy": float(np.sqrt((p ** 2).sum())),
            "entropy": float(-(p[nz] * np.log2(p[nz])).sum()),
            "correlation": corr,
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "obj_id"
    return df.add_prefix(f"{prefix}_")


def extract_features(labels, image, band_names=None, *, extra_layers=None,
                     transform=None, pixel_area=None, spectral=True, geometry=True,
                     texture_layers=None, texture_levels=16, context=True) -> pd.DataFrame:
    """Build the object feature table.

    Parameters
    ----------
    labels : (rows, cols) int raster of object ids 1..n.
    image : :class:`RasterData` or (bands, rows, cols) array.
    extra_layers : ``{name: 2-D array}`` of derived layers (NDVI, nDSM, slope...)
        that get the same spectral/context statistics as the bands.
    transform / pixel_area : give either to also get ``area_m2`` (map units²)
        and ``centroid_x/y``.  ``image.transform`` is used if available.
    texture_layers : names of layers to compute GLCM texture for.
    """
    from skimage.measure import regionprops_table

    lab = np.asarray(labels).astype(np.int64)
    if lab.ndim != 2:
        raise ValueError("labels must be 2-D")
    layers, n_bands = _layers(image, band_names, extra_layers)
    if transform is None and isinstance(image, RasterData):
        transform = image.transform
    if pixel_area is None and transform is not None:
        pixel_area = abs(transform.a * transform.e)

    ids = np.unique(lab[lab > 0])
    n_max = int(lab.max())
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n_max + 1).astype(np.float64)[ids]
    out = pd.DataFrame(index=pd.Index(ids, name="obj_id"))

    means = {}
    if spectral or context:
        for name, arr in layers.items():
            means[name] = np.bincount(flat, weights=arr.ravel(), minlength=n_max + 1)[ids] / area
    if spectral:
        band_cols = []
        for name, arr in layers.items():
            m = means[name]
            sq = np.bincount(flat, weights=(arr * arr).ravel(), minlength=n_max + 1)[ids] / area
            out[f"mean_{name}"] = m
            out[f"std_{name}"] = np.sqrt(np.maximum(sq - m * m, 0.0))
            out[f"min_{name}"] = ndi.minimum(arr, lab, index=ids)
            out[f"max_{name}"] = ndi.maximum(arr, lab, index=ids)
        band_cols = [f"mean_{n}" for n in list(layers)[:n_bands]]
        out["brightness"] = out[band_cols].mean(axis=1)

    if geometry:
        props = regionprops_table(
            lab.astype(np.int32),
            properties=("label", "area", "solidity", "extent",
                        "orientation", "major_axis_length", "minor_axis_length",
                        "centroid"))
        g = pd.DataFrame(props).set_index("label").reindex(ids)
        g.index.name = "obj_id"
        out["area_px"] = g["area"]
        if pixel_area:
            out["area_m2"] = g["area"] * pixel_area
        # Exact boundary length in pixel edges (4-connected), image frame included.
        # Same unit as the border lengths used in context features and merging.
        pa, pb, pl = border_pairs(lab)
        per_full = (np.bincount(pa, weights=pl, minlength=n_max + 1)
                    + np.bincount(pb, weights=pl, minlength=n_max + 1)
                    + image_edge_lengths(lab, n_max + 1))
        per = pd.Series(per_full[ids].astype(np.float64), index=out.index).clip(lower=1.0)
        out["perimeter_px"] = per
        # 4*pi*A/P^2.  Continuous disc = 1; a pixel-aligned square = pi/4 = 0.785;
        # staircase edges of raster discs lower it further.  Lower = more ragged/elongated.
        out["compactness"] = 4.0 * np.pi * g["area"] / (per ** 2)
        # P / (4*sqrt(A)): exactly 1.0 for a square, larger for ragged or elongated
        # shapes (the "shape index" used in eCognition).
        out["shape_index"] = per / (4.0 * np.sqrt(g["area"]))
        out["elongation"] = g["major_axis_length"] / g["minor_axis_length"].clip(lower=1.0)
        out["solidity"] = g["solidity"]
        out["extent"] = g["extent"]
        out["orientation"] = np.degrees(g["orientation"])
        out["centroid_row"] = g["centroid-0"]
        out["centroid_col"] = g["centroid-1"]
        if transform is not None:
            xs, ys = transform * (g["centroid-1"].to_numpy() + 0.5,
                                  g["centroid-0"].to_numpy() + 0.5)
            out["centroid_x"], out["centroid_y"] = xs, ys

    if texture_layers:
        for name in texture_layers:
            key = clean_name(name)
            if key not in layers:
                raise KeyError(f"texture layer {name!r} not found; have {list(layers)}")
            tex = glcm_texture(lab, layers[key], levels=texture_levels, ids=ids)
            tex.columns = [c + f"_{key}" for c in tex.columns]
            out = out.join(tex)

    if context:
        a, b, ln = border_pairs(lab)
        edge = image_edge_lengths(lab, n_max + 1)[ids].astype(np.float64)
        total = np.bincount(a, weights=ln, minlength=n_max + 1) \
            + np.bincount(b, weights=ln, minlength=n_max + 1)
        total_all = total[ids] + edge
        nn = np.bincount(a, minlength=n_max + 1) + np.bincount(b, minlength=n_max + 1)
        out["n_neighbors"] = nn[ids]
        out["rel_border_image_edge"] = np.where(total_all > 0, edge / np.maximum(total_all, 1), 0.0)
        for name in layers:
            m_full = np.zeros(n_max + 1)
            m_full[ids] = means[name]
            d = np.abs(m_full[a] - m_full[b]) * ln
            acc = np.bincount(a, weights=d, minlength=n_max + 1) \
                + np.bincount(b, weights=d, minlength=n_max + 1)
            out[f"diff_to_neighbors_{name}"] = np.where(total[ids] > 0, acc[ids] / np.maximum(total[ids], 1), 0.0)
    return out


def class_context(labels, classes: pd.Series, class_names=None) -> pd.DataFrame:
    """Neighbourhood features that depend on a (first) classification.

    For every class ``c``: ``rel_border_<c>`` = share of an object's border
    that touches objects of class ``c`` and ``n_nb_<c>`` = number of such
    neighbours.  Typical use: re-classify "building" candidates that sit
    beside "road" objects, or objects almost surrounded by "water".
    """
    lab = np.asarray(labels).astype(np.int64)
    n_max = int(lab.max())
    cls = pd.Series(classes).reindex(range(n_max + 1))
    names = list(class_names) if class_names is not None else sorted(c for c in cls.dropna().unique())
    a, b, ln = border_pairs(lab)
    ids = np.unique(lab[lab > 0])
    total = np.bincount(a, weights=ln, minlength=n_max + 1) + np.bincount(b, weights=ln, minlength=n_max + 1)
    out = pd.DataFrame(index=pd.Index(ids, name="obj_id"))
    ca, cb = cls.to_numpy()[a], cls.to_numpy()[b]
    for c in names:
        key = clean_name(c)
        shared = (np.bincount(a, weights=ln * (cb == c), minlength=n_max + 1)
                  + np.bincount(b, weights=ln * (ca == c), minlength=n_max + 1))
        cnt = (np.bincount(a, weights=(cb == c).astype(float), minlength=n_max + 1)
               + np.bincount(b, weights=(ca == c).astype(float), minlength=n_max + 1))
        out[f"rel_border_{key}"] = np.where(total[ids] > 0, shared[ids] / np.maximum(total[ids], 1), 0.0)
        out[f"n_nb_{key}"] = cnt[ids].astype(int)
    return out


def parent_map(fine_labels, coarse_labels) -> pd.Series:
    """Super-object of every fine object (the coarse object covering most of it)."""
    fine = np.asarray(fine_labels).astype(np.int64).ravel()
    coarse = np.asarray(coarse_labels).astype(np.int64).ravel()
    base = int(coarse.max()) + 1
    uniq, counts = np.unique(fine * base + coarse, return_counts=True)
    f, c = uniq // base, uniq % base
    df = pd.DataFrame({"fine": f, "coarse": c, "n": counts})
    best = df.sort_values(["fine", "n"], ascending=[True, False]).drop_duplicates("fine")
    s = best.set_index("fine")["coarse"]
    s.index.name = "obj_id"
    return s[s.index > 0]


def super_object_features(fine_labels, coarse_labels, image, band_names=None,
                          extra_layers=None) -> pd.DataFrame:
    """Features describing how a fine object relates to its super-object.

    ``parent_id``, ``parent_area_px``, ``rel_area_of_parent`` and, per layer,
    ``diff_to_parent_<layer>`` (fine mean minus parent mean).  Works with any
    two labelings; with :func:`segment_levels` the coarse level is an exact
    union of fine objects.
    """
    fine = np.asarray(fine_labels).astype(np.int64)
    coarse = np.asarray(coarse_labels).astype(np.int64)
    pm = parent_map(fine, coarse)
    layers, _ = _layers(image, band_names, extra_layers)
    ids = pm.index.to_numpy()
    fa = np.bincount(fine.ravel(), minlength=int(fine.max()) + 1)[ids].astype(float)
    ca = np.bincount(coarse.ravel(), minlength=int(coarse.max()) + 1).astype(float)
    out = pd.DataFrame({"parent_id": pm.to_numpy()}, index=pm.index)
    out["parent_area_px"] = ca[pm.to_numpy()]
    out["rel_area_of_parent"] = fa / ca[pm.to_numpy()]
    for name, arr in layers.items():
        fm = np.bincount(fine.ravel(), weights=arr.ravel(), minlength=int(fine.max()) + 1)[ids] / fa
        cm = (np.bincount(coarse.ravel(), weights=arr.ravel(), minlength=int(coarse.max()) + 1) / np.maximum(ca, 1))
        out[f"diff_to_parent_{name}"] = fm - cm[pm.to_numpy()]
    return out


def sub_object_fraction(fine_classes: pd.Series, fine_area: pd.Series,
                        parents: pd.Series, class_name) -> pd.Series:
    """Area fraction of each super-object covered by sub-objects of one class."""
    df = pd.DataFrame({"cls": fine_classes, "area": fine_area, "parent": parents}).dropna(subset=["parent"])
    tot = df.groupby("parent")["area"].sum()
    hit = df[df["cls"] == class_name].groupby("parent")["area"].sum()
    frac = (hit.reindex(tot.index).fillna(0.0) / tot)
    frac.index.name = "parent_id"
    return frac


def spectral_layers(image, band_names=None) -> dict:
    """Common derived layers, built from whichever of these bands exist.

    Looks for bands named ``red``, ``green``, ``nir`` (case-insensitive) and returns
    ``ndvi = (nir-red)/(nir+red)`` and ``ndwi = (green-nir)/(green+nir)`` where
    possible.  Pass the result as ``extra_layers=`` to :func:`extract_features`.
    """
    layers, _ = _layers(image, band_names, None)
    low = {k.lower(): v for k, v in layers.items()}
    out = {}
    eps = 1e-9
    if "nir" in low and "red" in low:
        out["ndvi"] = (low["nir"] - low["red"]) / (low["nir"] + low["red"] + eps)
    if "nir" in low and "green" in low:
        out["ndwi"] = (low["green"] - low["nir"]) / (low["green"] + low["nir"] + eps)
    return out
