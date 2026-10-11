"""Turn unnamed segments (SAM instances, foundation-model clusters) into class labels.

SAM and DINOv2 clustering give *regions* -- ``7`` means "this blob", not "building". This module names them
by looking at an existing class raster (OSM, WorldCover, a hand-labelled area) laid over the same pixels:
every segment takes the class that most of its pixels already have there, and segments where no class
holds enough of the pixels are marked unlabelled (255) instead of being forced into a class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

UNLABELLED = 255


def assign_classes(
    segments_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    min_share: float = 0.5,
    ignore_segment_ids: tuple[int, ...] = (0, 255),
    outside_value: int = 0,
    unlabelled_value: int = UNLABELLED,
    reference_nodata: int | None = None,
) -> dict[str, Any]:
    """Label each segment with the majority class of ``reference_path`` under it.

    Args:
        segments_path: Raster of segment ids (SAM ``auto_label`` output, DINOv2 cluster map, ...).
        reference_path: Class raster to learn from. It is resampled (nearest-neighbour) onto the segment
            raster's grid when the grids differ. Its own nodata (or ``reference_nodata``) is not voted.
        output_path: ``uint8`` class raster, on the segment raster's grid.
        min_share: Fraction of a segment's *voting* pixels the winning class must reach. Below it the
            segment becomes ``unlabelled_value`` (default 255, ignored by training).
        ignore_segment_ids: Ids that are "no segment" (SAM uses 0, the cluster labeller 255 for no data).
            These pixels get ``outside_value`` (0 = background) -- or 255 for id 255 / nodata.
        outside_value: Class written where there is no segment.

    Returns:
        ``success``, ``output_path``, ``n_segments``, ``n_unlabelled``, ``mapping`` (segment id -> class
        or ``unlabelled_value``), ``class_pixels``.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    if not 0.0 < min_share <= 1.0:
        raise ValueError("min_share must be in (0, 1]")
    with rasterio.open(str(segments_path)) as seg_src:
        seg = seg_src.read(1)
        profile = seg_src.profile
        seg_nodata = seg_src.nodata
        h, w, tf, crs = seg_src.height, seg_src.width, seg_src.transform, seg_src.crs
    with rasterio.open(str(reference_path)) as ref_src:
        ref_nd = reference_nodata if reference_nodata is not None else ref_src.nodata
        if ref_src.shape == (h, w) and ref_src.transform == tf and ref_src.crs == crs:
            ref = ref_src.read(1)
        else:
            ref = np.full((h, w), 255, dtype=np.uint8 if ref_src.dtypes[0] == "uint8" else ref_src.dtypes[0])
            reproject(rasterio.band(ref_src, 1), ref, src_transform=ref_src.transform, src_crs=ref_src.crs,
                      src_nodata=ref_src.nodata, dst_transform=tf, dst_crs=crs, dst_nodata=255,
                      resampling=Resampling.nearest)
            ref_nd = 255 if ref_nd is None else ref_nd
    ref = ref.astype(np.int64)
    voting = np.ones(ref.shape, bool) if ref_nd is None else ref != int(ref_nd)

    skip = np.isin(seg, list(ignore_segment_ids))
    if seg_nodata is not None:
        skip |= seg == seg_nodata
    ids, inv = np.unique(seg[~skip], return_inverse=True)
    out = np.full((h, w), outside_value, dtype=np.uint8)
    out[(seg == unlabelled_value) & skip] = unlabelled_value if unlabelled_value in ignore_segment_ids else outside_value

    mapping: dict[int, int] = {}
    if ids.size:
        r_vals = ref[~skip]
        v_mask = voting[~skip]
        n_cls = int(max(r_vals.max(), 0)) + 1
        key = inv[v_mask] * n_cls + r_vals[v_mask]
        hist = np.bincount(key, minlength=ids.size * n_cls).reshape(ids.size, n_cls)
        total = hist.sum(axis=1)
        best = hist.argmax(axis=1)
        share = np.where(total > 0, hist.max(axis=1) / np.maximum(total, 1), 0.0)
        cls = np.where(share >= min_share, best, unlabelled_value).astype(np.uint8)
        mapping = {int(i): int(c) for i, c in zip(ids, cls)}
        out[~skip] = cls[inv]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=unlabelled_value, compress="lzw")
    for k in ("blockxsize", "blockysize", "tiled"):
        profile.pop(k, None)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(out[np.newaxis])
    vals, counts = np.unique(out, return_counts=True)
    return {
        "success": True,
        "output_path": str(output_path),
        "n_segments": int(ids.size),
        "n_unlabelled": int(sum(1 for c in mapping.values() if c == unlabelled_value)),
        "mapping": mapping,
        "class_pixels": {int(v): int(c) for v, c in zip(vals, counts)},
    }
