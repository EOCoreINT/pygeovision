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
=============== =========================================== ==================

Note that ``scale`` means something different for ``multires`` and for
``felzenszwalb``: it is a good exercise to show students why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .io import RasterData, as_raster_data, rescale_bands
from .merging import RegionMerger

METHODS = ("multires", "slic", "felzenszwalb", "quickshift", "watershed", "chessboard")


@dataclass
class SegmentationResult:
    """Image objects as an integer raster (ids 1..n) plus provenance."""

    labels: np.ndarray
    method: str
    params: dict = field(default_factory=dict)
    raster: RasterData | None = None

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
        return find_boundaries(self.labels, mode="inner")

    def overlay(self, bands=(0, 1, 2), color=(255, 255, 0)) -> np.ndarray:
        """uint8 RGB preview of three bands with object outlines drawn on."""
        if self.raster is None:
            raise ValueError("no image attached to this result")
        img = rescale_bands(self.raster.image[list(bands)]).astype(np.uint8)
        rgb = np.moveaxis(img, 0, -1).copy()
        rgb[self.boundaries()] = color
        return rgb

    def save(self, path: str):
        """Write the label raster as a GeoTIFF."""
        import rasterio
        from affine import Affine
        h, w = self.labels.shape
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="int32", crs=self.crs,
                           transform=self.transform or Affine.identity()) as dst:
            dst.write(self.labels.astype(np.int32), 1)
        return path


def _hwc01(img: np.ndarray) -> np.ndarray:
    """(bands,rows,cols) 0-255 -> (rows,cols,bands) float in 0-1."""
    return np.moveaxis(img / 255.0, 0, -1)


def _slic(img, n_segments, compactness, sigma):
    from skimage.segmentation import slic
    lab = slic(_hwc01(img), n_segments=int(n_segments), compactness=compactness,
               sigma=sigma, channel_axis=-1, convert2lab=False, start_label=1,
               enforce_connectivity=True)
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


def _relabel(lab: np.ndarray) -> np.ndarray:
    """Consecutive ids 1..n, with every id spatially connected."""
    from scipy import ndimage as ndi
    out = np.zeros(lab.shape, dtype=np.int32)
    nxt = 1
    for v in np.unique(lab):
        comp, n = ndi.label(lab == v)
        out[comp > 0] = comp[comp > 0] + (nxt - 1)
        nxt += n
    return out


def segment(image, method: str = "multires", *, band_names=None, transform=None,
            crs=None, rescale: bool = True, **params) -> SegmentationResult:
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
    img = rescale_bands(rd.image) if rescale else rd.image.astype(np.float64)
    H, W = img.shape[1:]

    if method == "multires":
        p = dict(scale=30.0, color_weight=0.9, compactness_weight=0.5,
                 init="auto", init_segments=None, band_weights=None)
        p.update(params)
        init = p["init"]
        if isinstance(init, np.ndarray):
            init_labels = init
        else:
            if init == "auto":
                init = "pixels" if H * W <= 10_000 else "superpixels"
            if init == "pixels":
                init_labels = np.arange(H * W).reshape(H, W)
            elif init == "superpixels":
                nseg = p["init_segments"] or max(H * W // 16, 50)
                init_labels = _slic(img, nseg, 0.1, 0.0)
            else:
                raise ValueError("init must be 'auto', 'pixels', 'superpixels' or an array")
        m = RegionMerger(img, init_labels, p["color_weight"], p["compactness_weight"],
                         p["band_weights"]).run(p["scale"])
        lab = m.labels()
        used = {k: v for k, v in p.items() if k != "init"}
        used["init"] = init if not isinstance(init, np.ndarray) else "array"
        params = used
    elif method == "slic":
        p = dict(n_segments=500, compactness=0.1, sigma=1.0)
        p.update(params)
        lab = _slic(img, p["n_segments"], p["compactness"], p["sigma"])
        params = p
    elif method == "felzenszwalb":
        from skimage.segmentation import felzenszwalb
        p = dict(scale=100.0, sigma=0.8, min_size=20)
        p.update(params)
        import warnings
        with warnings.catch_warnings():
            # skimage warns that a 4-band image might be a 3-D volume; channel_axis
            # is passed explicitly so the multichannel reading is intended.
            warnings.simplefilter("ignore", RuntimeWarning)
            lab = felzenszwalb(_hwc01(img), scale=p["scale"], sigma=p["sigma"],
                               min_size=int(p["min_size"]), channel_axis=-1) + 1
        params = p
    elif method == "quickshift":
        from skimage.segmentation import quickshift
        p = dict(kernel_size=3, max_dist=6.0, ratio=0.5, sigma=0.0)
        p.update(params)
        lab = quickshift(_hwc01(img), kernel_size=p["kernel_size"], max_dist=p["max_dist"],
                         ratio=p["ratio"], sigma=p["sigma"], convert2lab=False,
                         channel_axis=-1) + 1
        params = p
    elif method == "watershed":
        p = dict(spacing=10, compactness=0.0)
        p.update(params)
        lab = _watershed(img, int(p["spacing"]), p["compactness"])
        params = p
    else:  # chessboard
        p = dict(size=16)
        p.update(params)
        lab = _chessboard((H, W), int(p["size"]))
        params = p

    return SegmentationResult(_relabel(lab), method, params, rd)


def from_labels(labels, image=None, *, band_names=None, transform=None, crs=None,
                method="external") -> SegmentationResult:
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


def segment_levels(image, scales, *, band_names=None, transform=None, crs=None,
                   rescale=True, **params) -> dict[float, SegmentationResult]:
    """Nested multi-level segmentation from ONE merging run.

    Because region merging only ever joins objects, each coarser level is
    exactly a union of the finer one - a true object hierarchy, which is what
    super-/sub-object features need.  Returns ``{scale: SegmentationResult}``
    in ascending scale order.
    """
    scales = sorted(float(s) for s in scales)
    rd = as_raster_data(image, band_names, transform, crs)
    img = rescale_bands(rd.image) if rescale else rd.image.astype(np.float64)
    H, W = img.shape[1:]
    p = dict(color_weight=0.9, compactness_weight=0.5, init="auto",
             init_segments=None, band_weights=None)
    p.update(params)
    init = p["init"]
    if init == "auto":
        init = "pixels" if H * W <= 10_000 else "superpixels"
    init_labels = (np.arange(H * W).reshape(H, W) if init == "pixels"
                   else _slic(img, p["init_segments"] or max(H * W // 16, 50), 0.1, 0.0))
    m = RegionMerger(img, init_labels, p["color_weight"], p["compactness_weight"],
                     p["band_weights"])
    out = {}
    for s in scales:
        m.run(s)
        q = dict(p, scale=s)
        out[s] = SegmentationResult(m.labels(), "multires", q, rd)
    return out
