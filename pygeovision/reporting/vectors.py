"""Analysis-ready vector exports: polygons with areas and class attributes.

Rasters are polygonised per connected patch (4-connectivity) and written, in
EPSG:4326, to a GeoPackage (one file, several layers, opens in QGIS/ArcGIS),
GeoJSON (web maps, presentations, GitHub) and/or Shapefile (legacy GIS), with
the same attributes in every format. Areas are measured in an equal-area
projection (EPSG:6933), so they are comparable between places.

If a raster would produce more polygons than ``max_polygons`` (speckle on a
large area), tiny patches are merged into their neighbours with a sieve filter
of growing size until it fits; the layer's ``info`` says what was done, and the
raster tables in the report are unaffected (they count pixels, not polygons).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.reporting.palettes import ClassDef, categorical

logger = logging.getLogger(__name__)

EQUAL_AREA_CRS = "EPSG:6933"
DEFAULT_MAX_POLYGONS = 300_000
_DISSOLVE_LIMIT = 200_000
VECTOR_FORMATS = ("gpkg", "geojson", "shp", "kml")
#: attribute names <= 10 characters so the Shapefile keeps them intact
ATTRS = ("patch_id", "class_id", "class_name", "color_hex", "area_ha", "area_m2", "pct_aoi", "cx", "cy", "pipeline", "period")


@dataclass
class VectorLayer:
    name: str
    gdf: Any                               # geopandas.GeoDataFrame in EPSG:4326 (may be empty)
    classes: Any = None                    # dissolved per-class GeoDataFrame (None if too large)
    source_layer: str = ""
    note: str = ""
    sieve_pixels: int = 0
    raw_patches: int = 0

    @property
    def n_features(self) -> int:
        return int(len(self.gdf))


def _count_patches(mask_by_value: Iterable[np.ndarray]) -> int:
    from scipy import ndimage
    total = 0
    for m in mask_by_value:
        _, n = ndimage.label(m)          # default structure = 4-connectivity
        total += int(n)
    return total


def polygonise(
    arr: np.ndarray, transform: Any, crs: Any, defs: Sequence[ClassDef], *, keep: Sequence[int] | None,
    skip: Sequence[int] = (), name: str, source_layer: str, aoi_area_ha: float, pipeline: str, period: str,
    max_polygons: int = DEFAULT_MAX_POLYGONS,
) -> VectorLayer:
    """Polygonise class array ``arr`` (NaN or negative = invalid).

    ``arr`` holds integer class values (float with NaN is accepted).
    """
    import geopandas as gpd
    from rasterio.features import shapes, sieve
    from shapely.geometry import shape as shp_shape

    a = np.asarray(arr)
    valid = np.isfinite(a) & (a >= 0) if np.issubdtype(a.dtype, np.floating) else (a >= 0)
    ints = np.where(valid, a, 0).astype(np.int32)
    present = sorted(int(v) for v in np.unique(ints[valid])) if valid.any() else []
    if keep is None:
        keep_vals = [v for v in present if v not in set(skip)]
    else:
        keep_vals = [v for v in keep if v in present]
    empty_cols = ["patch_id", "class_id", "class_name", "color_hex", "area_ha", "area_m2", "pct_aoi", "cx", "cy", "pipeline", "period", "geometry"]
    if not keep_vals:
        return VectorLayer(name, gpd.GeoDataFrame({c: [] for c in empty_cols}, geometry="geometry", crs="EPSG:4326"),
                           source_layer=source_layer, note="no pixels of the selected classes")
    if max(present) > 254:
        raise ValueError(f"class values above 254 cannot be polygonised (found {max(present)})")

    work = ints.astype(np.uint8)
    keep_mask = np.isin(ints, keep_vals) & valid
    raw = _count_patches(keep_mask & (ints == v) for v in keep_vals)
    min_px = 0
    note = ""
    if raw > max_polygons:
        min_px = 4
        while True:
            cand = sieve(work, min_px, connectivity=4, mask=valid.astype(np.uint8))
            n = _count_patches((np.isin(cand, keep_vals) & valid) & (cand == v) for v in keep_vals)
            if n <= max_polygons or min_px >= 1 << 20:
                work = cand
                break
            min_px *= 2
        keep_mask = np.isin(work, keep_vals) & valid
        note = (f"{raw:,} raw patches exceeded the {max_polygons:,}-polygon limit; patches smaller than "
                f"{min_px} pixels were merged into their neighbours.")
        logger.warning("%s: %s", name, note)

    geoms, vals = [], []
    for geom, val in shapes(work, mask=keep_mask.astype(np.uint8), transform=transform, connectivity=4):
        geoms.append(shp_shape(geom))
        vals.append(int(val))
    gdf = gpd.GeoDataFrame({"class_id": vals}, geometry=geoms, crs=crs)
    by_val = {d.value: d for d in defs}
    names = [by_val[v].name if v in by_val else f"Class {v}" for v in gdf["class_id"]]
    colors = [by_val[v].color if v in by_val else categorical(v) for v in gdf["class_id"]]
    gdf["class_name"] = names
    gdf["color_hex"] = colors
    gdf["area_m2"] = gdf.to_crs(EQUAL_AREA_CRS).area.round(1)
    gdf["area_ha"] = (gdf["area_m2"] / 1e4).round(4)
    gdf["pct_aoi"] = (gdf["area_ha"] / aoi_area_ha * 100.0).round(5) if aoi_area_ha > 0 else np.nan
    gdf = gdf.to_crs("EPSG:4326")
    cent = gdf.geometry.representative_point()
    gdf["cx"] = cent.x.round(6)
    gdf["cy"] = cent.y.round(6)
    gdf["pipeline"] = pipeline
    gdf["period"] = period
    gdf = gdf.sort_values("area_ha", ascending=False).reset_index(drop=True)
    gdf["patch_id"] = np.arange(1, len(gdf) + 1)
    gdf = gdf[[c for c in empty_cols if c in gdf.columns]]

    classes = None
    if len(gdf) <= _DISSOLVE_LIMIT:
        d = gdf.dissolve(by="class_id", aggfunc={"class_name": "first", "color_hex": "first", "area_ha": "sum",
                                                 "pipeline": "first", "period": "first"}).reset_index()
        d["n_patches"] = d["class_id"].map(gdf.groupby("class_id").size()).astype(int)
        d["area_ha"] = d["area_ha"].round(4)
        d["pct_aoi"] = (d["area_ha"] / aoi_area_ha * 100.0).round(5) if aoi_area_ha > 0 else np.nan
        classes = d
    return VectorLayer(name, gdf, classes, source_layer, note, min_px, raw)


def load_vector_file(name: str, path: Path, *, pipeline: str, period: str, aoi_area_ha: float,
                     max_features: int = DEFAULT_MAX_POLYGONS, layer: str | None = None) -> VectorLayer:
    """A vector layer that a pipeline already wrote (buildings, roads, streams, fire points) as a report deliverable.

    Geometries are kept as they are -- nothing is rasterised or merged. Attributes the exports expect are added
    (``patch_id``, ``class_id``/``class_name`` when missing, ``area_ha`` and ``pct_aoi`` for polygons). A layer with
    more than ``max_features`` features is not exported (the pipeline's own file still has all of them); the returned
    layer is empty and its ``note`` says why.
    """
    import geopandas as gpd

    path = Path(path)
    gdf = gpd.read_file(path, layer=layer or name)
    if len(gdf) > max_features:
        return VectorLayer(name, gdf.iloc[0:0].copy(), None, name,
                           f"{len(gdf):,} features exceed the {max_features:,}-feature limit for the report exports; "
                           f"all of them are in {path.name}.")
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf.set_crs("EPSG:4326")
    if len(gdf) and gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        area_m2 = gdf.to_crs(EQUAL_AREA_CRS).area
        gdf["area_ha"] = (area_m2 / 1e4).round(4)
        gdf["pct_aoi"] = (gdf["area_ha"] / aoi_area_ha * 100.0).round(5) if aoi_area_ha > 0 else np.nan
    if "class_id" not in gdf.columns:
        gdf["class_id"] = 0
    if "class_name" not in gdf.columns:
        gdf["class_name"] = name
    if len(gdf):
        cent = gdf.geometry.representative_point()
        gdf["cx"] = cent.x.round(6)
        gdf["cy"] = cent.y.round(6)
    gdf["pipeline"] = pipeline
    gdf["period"] = period
    gdf = gdf.reset_index(drop=True)
    gdf.insert(0, "patch_id", np.arange(1, len(gdf) + 1))
    return VectorLayer(name, gdf, None, name, "", 0, len(gdf))


def aoi_frame(bbox: tuple[float, float, float, float], attrs: dict[str, Any]) -> Any:
    """A one-row GeoDataFrame with the requested area of interest."""
    import geopandas as gpd
    from shapely.geometry import box
    clean = {k[:10]: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in attrs.items()}
    return gpd.GeoDataFrame([clean], geometry=[box(*bbox)], crs="EPSG:4326")


def _kml_available() -> bool:
    try:
        import pyogrio
        drivers = pyogrio.list_drivers()
    except Exception as exc:  # noqa: BLE001 - capability probe
        logger.debug("pyogrio driver probe failed: %s", exc)
        return False
    return any(k in drivers and "w" in drivers[k] for k in ("LIBKML", "KML"))


def write_vectors(layers: list[VectorLayer], aoi: Any, out_dir: Path, pipeline: str,
                  formats: Sequence[str]) -> tuple[dict[str, list[Path]], list[str]]:
    """Write ``layers`` (+ the AOI) in each requested format under ``out_dir``.

    Returns ``({format: [paths]}, warnings)``. A format that cannot be written is
    reported in ``warnings`` and does not stop the others.
    """
    written: dict[str, list[Path]] = {}
    warnings: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nonempty = [lyr for lyr in layers if lyr.n_features]
    for lyr in layers:
        if not lyr.n_features:
            warnings.append(f"vector layer '{lyr.name}' has no features ({lyr.note or 'nothing to polygonise'}).")

    for fmt in formats:
        try:
            if fmt == "gpkg":
                p = out_dir / f"{pipeline}.gpkg"
                if p.exists():
                    p.unlink()
                aoi.to_file(p, layer="aoi", driver="GPKG", engine="pyogrio")
                for lyr in nonempty:
                    lyr.gdf.to_file(p, layer=lyr.name, driver="GPKG", engine="pyogrio")
                    if lyr.classes is not None:
                        lyr.classes.to_file(p, layer=f"{lyr.name}_classes", driver="GPKG", engine="pyogrio")
                written["gpkg"] = [p]
            elif fmt == "geojson":
                d = out_dir / "geojson"
                d.mkdir(exist_ok=True)
                paths = []
                for name, g in [("aoi", aoi)] + [(lyr.name, lyr.gdf) for lyr in nonempty]:
                    p = d / f"{pipeline}_{name}.geojson"
                    if p.exists():
                        p.unlink()
                    g.to_file(p, driver="GeoJSON", engine="pyogrio", COORDINATE_PRECISION=6)
                    paths.append(p)
                written["geojson"] = paths
            elif fmt == "shp":
                d = out_dir / "shapefile"
                d.mkdir(exist_ok=True)
                paths = []
                for name, g in [("aoi", aoi)] + [(lyr.name, lyr.gdf) for lyr in nonempty]:
                    p = d / f"{pipeline}_{name}.shp"
                    g.to_file(p, driver="ESRI Shapefile", engine="pyogrio", encoding="utf-8")
                    paths.append(p)
                written["shp"] = paths
            elif fmt == "kml":
                if not _kml_available():
                    warnings.append("KML export skipped: this GDAL build has no KML writer.")
                    continue
                d = out_dir / "kml"
                d.mkdir(exist_ok=True)
                paths = []
                import pyogrio
                driver = "LIBKML" if "LIBKML" in pyogrio.list_drivers() else "KML"
                for name, g in [("aoi", aoi)] + [(lyr.name, lyr.gdf) for lyr in nonempty]:
                    p = d / f"{pipeline}_{name}.kml"
                    g.drop(columns=[c for c in ("cx", "cy") if c in g.columns]).to_file(p, driver=driver, engine="pyogrio")
                    paths.append(p)
                written["kml"] = paths
            else:
                warnings.append(f"unknown vector format '{fmt}' (choose from {', '.join(VECTOR_FORMATS)}).")
        except Exception as exc:  # noqa: BLE001 - one format failing must not lose the others
            logger.warning("vector export '%s' failed: %s", fmt, exc, exc_info=True)
            warnings.append(f"vector export '{fmt}' failed: {exc}")
    return written, warnings
