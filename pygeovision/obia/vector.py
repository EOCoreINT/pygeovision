"""Raster <-> vector conversion for image objects and reference data."""
from __future__ import annotations

import numpy as np


def _identity():
    from affine import Affine
    return Affine.identity()


def labels_to_gdf(labels: np.ndarray, transform=None, crs=None,
                  attributes=None, id_column: str = "obj_id"):
    """One polygon (or multipolygon) per image object.

    ``attributes`` is an optional DataFrame indexed by object id (e.g. the
    feature table or a classification) whose columns are joined on.
    Without a ``transform`` coordinates are pixel units (row grows with y).
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union

    lab = np.asarray(labels).astype(np.int32)
    tr = transform if transform is not None else _identity()
    parts: dict[int, list] = {}
    for geom, val in shapes(lab, mask=lab > 0, transform=tr, connectivity=4):
        parts.setdefault(int(val), []).append(shp_shape(geom))
    ids = sorted(parts)
    geoms = [parts[i][0] if len(parts[i]) == 1 else unary_union(parts[i]) for i in ids]
    gdf = gpd.GeoDataFrame({id_column: ids}, geometry=geoms, crs=crs)
    if attributes is not None:
        attr = attributes.copy()
        attr.index.name = id_column
        gdf = gdf.merge(attr.reset_index(), on=id_column, how="left")
    return gdf


def rasterize_geometries(gdf, shape: tuple[int, int], transform,
                         value_column: str | None = None):
    """Burn polygons/points into a raster.

    Returns ``(raster, codes)``.  If ``value_column`` is given its distinct
    values are mapped to codes 1..K (``codes`` = ``{value: code}``); otherwise
    each feature gets its own code 1..N and ``codes`` maps row-position->code.
    Background is 0.  Where features overlap, later rows overwrite earlier ones.
    """
    from rasterio.features import rasterize

    if len(gdf) == 0:
        return np.zeros(shape, dtype=np.int32), {}
    tr = transform if transform is not None else _identity()
    if value_column is None:
        codes = {i: i + 1 for i in range(len(gdf))}
        vals = [codes[i] for i in range(len(gdf))]
    else:
        uniq = list(dict.fromkeys(gdf[value_column].tolist()))
        codes = {v: i + 1 for i, v in enumerate(uniq)}
        vals = [codes[v] for v in gdf[value_column]]
    shapes_ = [(g, v) for g, v in zip(gdf.geometry, vals) if g is not None and not g.is_empty]
    ras = rasterize(shapes_, out_shape=shape, transform=tr, fill=0,
                    dtype="int32", all_touched=False)
    return ras, codes
