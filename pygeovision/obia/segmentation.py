"""Image segmentation: create image objects from a raster.

One entry point, :func:`segment`, with several algorithms so students can
compare how different methods carve up the same scene:

=============== =========================================== ==================
method          idea                                        main parameter(s)
=============== =========================================== ==================
``multires``    region merging, Baatz & Schäpe criterion    ``scale``,
                (see :mod:`pygeovision.obia.merging`)       ``color_weight``
``slic``        k-means clustering in colour+space          ``n_segments``
``felzenszwalb``graph-based merging (Felzenszwalb 2004)     ``scale``
``quickshift``  mode seeking in colour+space                ``max_dist``
``watershed``   flooding of the gradient image from seeds   ``spacing``
``chessboard``  regular square tiles (a baseline)           ``size``
``quadtree``    recursive 4-way split until homogeneous     ``scale`` (max std)
``multi_threshold`` classes of one layer, connected patches ``n_classes`` /
                (multi-Otsu or given thresholds)            ``thresholds``
``contrast_split`` repeated dark / bright split at the      ``levels``,
                max-edge-contrast threshold                 ``min_contrast``
``spectral_difference`` merge neighbours with similar means ``max_difference``
=============== =========================================== ==================

Note that ``scale`` means something different for ``multires`` and for
``felzenszwalb``: it is a good exercise to show students why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .io import RasterData, as_raster_data, rescale_bands, to_uint8, valid_mask
from .merging import RegionMerger, SpectralDifferenceMerger
from .splitting import contrast_split, multi_threshold, quadtree

METHODS = (
    "multires",
    "slic",
    "felzenszwalb",
    "quickshift",
    "watershed",
    "chessboard",
    "quadtree",
    "multi_threshold",
    "contrast_split",
    "spectral_difference",
)


@dataclass
class SegmentationResult:
    """Image objects as an integer raster (ids 1..n) plus provenance."""

    labels: np.ndarray
    method: str
    params: dict = field(default_factory=dict)
    raster: RasterData | None = None
    valid: np.ndarray | None = (
        None  # pixels that held data; label 0 marks the others (cloud, nodata)
    )

    @property
    def n_objects(self) -> int:
        return int(self.labels.max()) if self.labels.size else 0

    @property
    def transform(self):
        return None if self.raster is None else self.raster.transform

    @property
    def crs(self):
        return None if self.raster is None else self.raster.crs

    def to_vector(self, attributes=None):
        """Objects as a GeoDataFrame (one row per object)."""
        from .vector import labels_to_gdf

        return labels_to_gdf(self.labels, self.transform, self.crs, attributes)

    def boundaries(self) -> np.ndarray:
        """Boolean mask of pixels lying on an object boundary."""
        from skimage.segmentation import find_boundaries

        return find_boundaries(self.labels, mode="inner") & (self.labels > 0)

    def overlay(self, bands=(0, 1, 2), color=(255, 255, 0)) -> np.ndarray:
        """uint8 RGB preview of three bands with object outlines drawn on."""
        if self.raster is None:
            raise ValueError("no image attached to this result")
        img = to_uint8(rescale_bands(self.raster.image[list(bands)]))
        rgb = np.moveaxis(img, 0, -1).copy()
        rgb[self.boundaries()] = color
        return rgb

    def save(self, path: str):
        """Write the label raster as a GeoTIFF."""
        from affine import Affine
        import rasterio

        h, w = self.labels.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="int32",
            crs=self.crs,
            transform=self.transform or Affine.identity(),
        ) as dst:
            dst.write(self.labels.astype(np.int32), 1)
        return path


def _hwc01(img: np.ndarray) -> np.ndarray:
    """(bands,rows,cols) 0-255 -> (rows,cols,bands) float in 0-1."""
    return np.moveaxis(img / 255.0, 0, -1)


def _slic(img, n_segments, compactness, sigma):
    from skimage.segmentation import slic

    lab = slic(
        _hwc01(img),
        n_segments=int(n_segments),
        compactness=compactness,
        sigma=sigma,
        channel_axis=-1,
        convert2lab=False,
        start_label=1,
        enforce_connectivity=True,
    )
    return lab


def _watershed(img, spacing, compactness):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    grad = np.zeros(img.shape[1:])
    for b in range(img.shape[0]):
        g = ndi.gaussian_gradient_magnitude(img[b], sigma=1.0)
        grad += g * g
    grad = np.sqrt(grad)
    h, w = grad.shape
    markers = np.zeros((h, w), dtype=np.int32)
    rr = np.arange(spacing // 2, h, spacing)
    cc = np.arange(spacing // 2, w, spacing)
    k = 1
    for r in rr:
        for c in cc:
            markers[r, c] = k
            k += 1
    return watershed(grad, markers=markers, compactness=compactness)


def _chessboard(shape, size):
    h, w = shape
    rows = np.arange(h) // size
    cols = np.arange(w) // size
    ncol = int(cols.max()) + 1
    return (rows[:, None] * ncol + cols[None, :] + 1).astype(np.int32)


def _relabel(lab: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Consecutive ids 1..n, with every id spatially connected; pixels outside ``valid`` get 0.

    Each id is labelled inside its own bounding box, so the cost follows the number of pixels, not
    (number of objects x number of pixels): a segmentation with tens of thousands of objects on a real scene
    takes seconds.
    """
    from scipy import ndimage as ndi

    src = np.asarray(lab).astype(np.int64)
    # Compact the ids to 1..k (find_objects skips 0 and sizes its answer by the largest id);
    # pixels outside ``valid`` become 0 and stay out.
    if valid is None:
        _, inv = np.unique(src, return_inverse=True)
        src = inv.reshape(src.shape).astype(np.int64) + 1
    else:
        _, inv = np.unique(src[valid], return_inverse=True)
        compact = np.zeros(src.shape, dtype=np.int64)
        compact[valid] = inv.astype(np.int64) + 1
        src = compact
    out = np.zeros(src.shape, dtype=np.int32)
    nxt = 1
    for v, sl in enumerate(ndi.find_objects(src), start=1):
        if sl is None:
            continue
        sub = src[sl] == v
        comp, n = ndi.label(sub)
        view = out[sl]
        view[sub] = comp[sub] + (nxt - 1)
        nxt += n
    return out


def _initial_objects(img, init, init_segments):
    """Starting objects for the merging methods: single pixels, SLIC superpixels or a given label array."""
    H, W = img.shape[1:]
    if isinstance(init, np.ndarray):
        return init, "array"
    if init == "auto":
        init = "pixels" if H * W <= 10_000 else "superpixels"
    if init == "pixels":
        return np.arange(H * W).reshape(H, W), init
    if init == "superpixels":
        return _slic(img, init_segments or max(H * W // 16, 50), 0.1, 0.0), init
    raise ValueError("init must be 'auto', 'pixels', 'superpixels' or an array")


def _layer(rd: RasterData, img: np.ndarray, valid: np.ndarray, layer) -> np.ndarray:
    """One 2-D layer for the threshold methods: a band (index or name), ``"brightness"``, ``"ndvi"`` or ``"ndwi"``."""
    from scipy import ndimage as ndi

    from .features import spectral_layers

    if isinstance(layer, str) and layer.lower() in ("ndvi", "ndwi"):
        derived = spectral_layers(rd.image, rd.band_names)
        if layer.lower() not in derived:
            raise ValueError(f"{layer!r} needs bands named red/nir (ndvi) or green/nir (ndwi); have {rd.band_names}")
        arr = np.asarray(derived[layer.lower()], dtype=np.float64)
        ok = valid & np.isfinite(arr)
        if not ok.all():
            idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
            arr = arr[idx[0], idx[1]]
        return arr
    if isinstance(layer, str) and layer.lower() == "brightness":
        return img.mean(axis=0)
    if isinstance(layer, str):
        names = [str(n).lower() for n in rd.band_names]
        if layer.lower() not in names:
            raise ValueError(f"unknown layer {layer!r}; bands are {rd.band_names}, or use 'brightness', 'ndvi', 'ndwi'")
        return img[names.index(layer.lower())]
    k = int(layer)
    if not -img.shape[0] <= k < img.shape[0]:
        raise ValueError(f"band index {k} out of range for {img.shape[0]} bands")
    return img[k]


def _prepare(rd: RasterData, rescale: bool):
    """Image for the algorithms, plus the mask of pixels that hold data.

    Invalid pixels (cloud-masked, nodata) cannot be segmented: the algorithms either refuse them or, worse, let them
    poison every distance. They are filled with the value of the nearest valid pixel so the algorithm sees a continuous
    image, and the caller sets their label to 0 afterwards.
    """
    from scipy import ndimage as ndi

    valid = valid_mask(rd)
    if not valid.any():
        raise ValueError("the image has no valid pixels (every pixel is NaN or nodata)")
    img = rescale_bands(rd.image) if rescale else rd.image.astype(np.float64)
    if not valid.all():
        idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        img = img[:, idx[0], idx[1]]
    return img, valid


def segment(
    image,
    method: str = "multires",
    *,
    band_names=None,
    transform=None,
    crs=None,
    rescale: bool = True,
    **params,
) -> SegmentationResult:
    """Segment an image into objects.

    Parameters
    ----------
    image : path, :class:`RasterData` or (bands, rows, cols) array.
    method : one of :data:`METHODS`.
    rescale : stretch each band to 0-255 first (recommended; makes scale-like
        parameters comparable between images).
    **params : algorithm parameters, see the table in the module docstring.
        ``multires``: ``scale=30, color_weight=0.9, compactness_weight=0.5,
        init="auto"|"pixels"|"superpixels"|ndarray, init_segments=None,
        band_weights=None``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    rd = as_raster_data(image, band_names, transform, crs)
    img, valid = _prepare(rd, rescale)
    H, W = img.shape[1:]

    if method == "multires":
        p: dict[str, Any] = {
            "scale": 30.0,
            "color_weight": 0.9,
            "compactness_weight": 0.5,
            "init": "auto",
            "init_segments": None,
            "band_weights": None,
            "strategy": "best_first",
        }
        p.update(params)
        init_labels, init = _initial_objects(img, p["init"], p["init_segments"])
        m = RegionMerger(
            img, init_labels, p["color_weight"], p["compactness_weight"], p["band_weights"]
        ).run(p["scale"], p["strategy"])
        lab = m.labels()
        params = {**{k: v for k, v in p.items() if k != "init"}, "init": init}
    elif method == "spectral_difference":
        p = {
            "max_difference": 10.0,
            "metric": "max",
            "init": "auto",
            "init_segments": None,
            "band_weights": None,
            "strategy": "best_first",
        }
        p.update(params)
        init_labels, init = _initial_objects(img, p["init"], p["init_segments"])
        m = SpectralDifferenceMerger(img, init_labels, p["band_weights"], p["metric"]).run(
            p["max_difference"], p["strategy"]
        )
        lab = m.labels()
        params = {**{k: v for k, v in p.items() if k != "init"}, "init": init}
    elif method == "quadtree":
        p = {"scale": 10.0, "min_size": 1, "max_size": None, "band_weights": None}
        p.update(params)
        lab = quadtree(img, p["scale"], int(p["min_size"]), p["max_size"], p["band_weights"])
        params = p
    elif method == "multi_threshold":
        p = {"layer": "brightness", "thresholds": None, "n_classes": 3}
        p.update(params)
        lay = _layer(rd, img, valid, p["layer"])
        lab, found = multi_threshold(lay, p["thresholds"], int(p["n_classes"]), valid)
        params = {**p, "thresholds": list(found)}
    elif method == "contrast_split":
        p = {"layer": "brightness", "levels": 2, "steps": 32, "min_contrast": 0.0, "min_size": 1}
        p.update(params)
        lay = _layer(rd, img, valid, p["layer"])
        lab = contrast_split(lay, int(p["levels"]), int(p["steps"]), p["min_contrast"], int(p["min_size"]))
        params = p
    elif method == "slic":
        p = {"n_segments": 500, "compactness": 0.1, "sigma": 1.0}
        p.update(params)
        lab = _slic(img, p["n_segments"], p["compactness"], p["sigma"])
        params = p
    elif method == "felzenszwalb":
        from skimage.segmentation import felzenszwalb

        p = {"scale": 100.0, "sigma": 0.8, "min_size": 20}
        p.update(params)
        import warnings

        with warnings.catch_warnings():
            # skimage warns that a 4-band image might be a 3-D volume; channel_axis
            # is passed explicitly so the multichannel reading is intended.
            warnings.simplefilter("ignore", RuntimeWarning)
            lab = (
                felzenszwalb(
                    _hwc01(img),
                    scale=p["scale"],
                    sigma=p["sigma"],
                    min_size=int(p["min_size"]),
                    channel_axis=-1,
                )
                + 1
            )
        params = p
    elif method == "quickshift":
        from skimage.segmentation import quickshift

        p = {"kernel_size": 3, "max_dist": 6.0, "ratio": 0.5, "sigma": 0.0}
        p.update(params)
        lab = (
            quickshift(
                _hwc01(img),
                kernel_size=p["kernel_size"],
                max_dist=p["max_dist"],
                ratio=p["ratio"],
                sigma=p["sigma"],
                convert2lab=False,
                channel_axis=-1,
            )
            + 1
        )
        params = p
    elif method == "watershed":
        p = {"spacing": 10, "compactness": 0.0}
        p.update(params)
        lab = _watershed(img, int(p["spacing"]), p["compactness"])
        params = p
    else:  # chessboard
        p = {"size": 16}
        p.update(params)
        lab = _chessboard((H, W), int(p["size"]))
        params = p

    return SegmentationResult(
        _relabel(lab, valid), method, params, rd, None if valid.all() else valid
    )


def from_labels(
    labels, image=None, *, band_names=None, transform=None, crs=None, method="external"
) -> SegmentationResult:
    """Wrap an existing label raster (e.g. SAM masks, eCognition export) as image objects.

    Disconnected pieces sharing an id are split into separate objects and ids
    are renumbered 1..n.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("labels must be 2-D")
    rd = None if image is None else as_raster_data(image, band_names, transform, crs)
    if rd is not None and rd.shape != lab.shape:
        raise ValueError(f"image {rd.shape} and labels {lab.shape} differ in shape")
    return SegmentationResult(_relabel(lab), method, {}, rd)


def segment_levels(
    image, scales, *, band_names=None, transform=None, crs=None, rescale=True, **params
) -> dict[float, SegmentationResult]:
    """Nested multi-level segmentation from ONE merging run.

    Because region merging only ever joins objects, each coarser level is
    exactly a union of the finer one - a true object hierarchy, which is what
    super-/sub-object features need.  Returns ``{scale: SegmentationResult}``
    in ascending scale order.
    """
    scales = sorted(float(s) for s in scales)
    rd = as_raster_data(image, band_names, transform, crs)
    img, valid = _prepare(rd, rescale)
    H, W = img.shape[1:]
    p = {
        "color_weight": 0.9,
        "compactness_weight": 0.5,
        "init": "auto",
        "init_segments": None,
        "band_weights": None,
    }
    p.update(params)
    init = p["init"]
    if init == "auto":
        init = "pixels" if H * W <= 10_000 else "superpixels"
    init_labels = (
        np.arange(H * W).reshape(H, W)
        if init == "pixels"
        else _slic(img, p["init_segments"] or max(H * W // 16, 50), 0.1, 0.0)
    )
    m = RegionMerger(
        img, init_labels, p["color_weight"], p["compactness_weight"], p["band_weights"]
    )
    out = {}
    for s in scales:
        m.run(s)
        q = dict(p, scale=s)
        out[s] = SegmentationResult(
            _relabel(m.labels(), valid), "multires", q, rd, None if valid.all() else valid
        )
    return out
