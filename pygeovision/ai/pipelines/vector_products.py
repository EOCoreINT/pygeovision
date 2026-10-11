"""Pipelines whose primary product is a vector layer from a published source: buildings and roads.

Both write a GeoPackage of the real geometries (so nothing is rasterised and lost) plus a raster that shows *how
much* there is per cell, which is what a map of a large area can actually display:

* ``building_footprints`` -- Microsoft ML footprints, Google Open Buildings or OpenStreetMap
  (``pygeovision.data.footprints``); with ``checkpoint=`` it instead runs a segmentation network you trained.
* ``road_network``        -- OpenStreetMap highways, length by class and road density.

These are *published datasets*, not detections on your imagery: they are only as complete and as current as the
source, which the stats and the report say.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import BasePipeline, PipelineResult
from pygeovision.ai.pipelines._raster import write_float
from pygeovision.data import footprints as fp
from pygeovision.data.reference import make_grid, utm_crs_for

logger = logging.getLogger(__name__)

BUILDING_SOURCES = {
    "microsoft": "Microsoft Global ML Building Footprints: machine-learning detections on Bing/Maxar/Vexcel imagery of "
                 "various dates (about 1% false positives reported by the project); building heights where available.",
    "google": "Google Open Buildings v3: machine-learning detections on 50 cm satellite imagery (mostly 2022-2023) with a "
              "per-building confidence; the dataset keeps detections of 0.65 and above.",
    "osm": "OpenStreetMap: volunteered data. Completeness varies a great deal between places -- good in many towns, "
           "thin in rural areas; buildings mapped as multipolygon relations are not included.",
}

#: OpenStreetMap ``highway`` values grouped for reporting (definitions: wiki.openstreetmap.org/wiki/Key:highway)
HIGHWAY_GROUPS = {
    "major roads": ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"),
    "local roads": ("tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service", "road"),
    "tracks and paths": ("track", "path", "footway", "cycleway", "bridleway", "steps", "pedestrian"),
}
NOT_BUILT_YET = ("proposed", "construction", "razed", "abandoned", "planned", "platform")


def _utm(bbox) -> str:
    return utm_crs_for((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _aoi_km2(bbox) -> float:
    import geopandas as gpd
    from shapely.geometry import box
    return float(gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(_utm(bbox)).area.iloc[0]) / 1e6


def _cells_inside(grid, bbox) -> np.ndarray:
    """True for grid cells whose centre lies inside the bounding box."""
    from pyproj import Transformer
    cols, rows = np.meshgrid(np.arange(grid.width) + 0.5, np.arange(grid.height) + 0.5)
    x = grid.transform.c + grid.transform.a * cols
    y = grid.transform.f + grid.transform.e * rows
    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(x, y)
    return (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])


def _cell_index(grid, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col = np.floor((x - grid.transform.c) / grid.transform.a).astype(int)
    row = np.floor((y - grid.transform.f) / grid.transform.e).astype(int)
    ok = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)
    return row, col, ok


def _clip(gdf, bbox):
    from shapely.geometry import box
    aoi = box(*bbox)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.intersects(aoi)].copy()
    gdf["geometry"] = gdf.geometry.intersection(aoi)
    return gdf[~gdf.geometry.is_empty].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Buildings
# ═══════════════════════════════════════════════════════════════════════════

def run_building_reference(pipeline: BasePipeline, bbox, out: Path, source: str, cell_m: float, min_confidence: float,
                           cache_dir: Any, session: Any, t0: float) -> PipelineResult:
    """The ``building_footprints`` pipeline when no network checkpoint is given."""
    try:
        if source not in BUILDING_SOURCES:
            return pipeline._fail(f"source must be one of {sorted(BUILDING_SOURCES)}, got {source!r}.", t0)
        if cell_m <= 0:
            return pipeline._fail("cell_m must be positive.", t0)
        if source == "microsoft":
            raw = fp.microsoft_footprints(bbox, cache_dir=cache_dir, session=session)
        elif source == "google":
            raw = fp.google_footprints(bbox, min_confidence=min_confidence, cache_dir=cache_dir, session=session)
        else:
            raw = fp.osm_features(bbox, "building", session=session)
        gdf = _clip(raw, bbox)
        utm = _utm(bbox)
        grid = make_grid(bbox, cell_m, crs=utm)
        g_utm = gdf.to_crs(utm)
        gdf["area_m2"] = g_utm.area.round(2).to_numpy()
        gdf["source"] = source
        gpkg = out / "building_footprints.gpkg"
        if len(gdf):
            gdf.to_file(gpkg, layer="buildings", driver="GPKG")
        cent = g_utm.geometry.centroid
        row, col, ok = _cell_index(grid, cent.x.to_numpy(), cent.y.to_numpy())
        area_sum = np.zeros((grid.height, grid.width), "float64")
        count = np.zeros((grid.height, grid.width), "float64")
        if ok.any():
            np.add.at(area_sum, (row[ok], col[ok]), gdf["area_m2"].to_numpy()[ok])
            np.add.at(count, (row[ok], col[ok]), 1.0)
        inside = _cells_inside(grid, bbox)
        cover = np.where(inside, np.clip(area_sum / (cell_m * cell_m), 0.0, 1.0), np.nan).astype("float32")
        path = write_float(out / "building_density.tif", cover, grid.profile(),
                           [f"share of each {cell_m:g} m cell covered by building footprints"])
        write_float(out / "building_count.tif", np.where(inside, count, np.nan).astype("float32"), grid.profile(),
                    [f"buildings per {cell_m:g} m cell (by centroid)"])
        aoi_km2 = _aoi_km2(bbox)
        areas = gdf["area_m2"].to_numpy() if len(gdf) else np.array([])
        stats: dict[str, Any] = {
            "source": source, "source_note": BUILDING_SOURCES[source], "building_count": int(len(gdf)),
            "building_area_ha": float(areas.sum()) / 1e4, "building_coverage": float(areas.sum()) / (aoi_km2 * 1e6),
            "buildings_per_km2": len(gdf) / aoi_km2, "mean_building_area_m2": float(areas.mean()) if len(areas) else None,
            "median_building_area_m2": float(np.median(areas)) if len(areas) else None, "aoi_area_km2": aoi_km2,
            "cell_m": cell_m, "max_cell_coverage": float(np.nanmax(cover)) if inside.any() else None,
            "footprints_file": str(gpkg) if len(gdf) else None,
        }
        if source == "google":
            stats["min_confidence"] = min_confidence
        if "height_m" in gdf.columns:
            h = gdf["height_m"].dropna()
            stats["buildings_with_height"] = int(len(h))
            stats["mean_height_m"] = float(h.mean()) if len(h) else None
        meta = {"date": "n/a (published dataset)", "crs": utm}
        if len(gdf):
            meta["vector_files"] = [{"name": "buildings", "path": str(Path(gpkg).resolve())}]
        return pipeline._ok(path, stats, meta, t0)
    except (fp.FootprintSourceError, ValueError) as exc:
        return pipeline._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Roads
# ═══════════════════════════════════════════════════════════════════════════

def _group_of(tag: str | None) -> str:
    for g, tags in HIGHWAY_GROUPS.items():
        if tag in tags:
            return g
    return "other"


def length_per_cell(lines_utm, grid, step_m: float) -> np.ndarray:
    """Metres of line in each grid cell. Lines are cut every ``step_m`` and each piece goes to the cell of its midpoint."""
    import shapely
    acc = np.zeros((grid.height, grid.width), "float64")
    if len(lines_utm) == 0:
        return acc
    geoms = shapely.segmentize(np.asarray(lines_utm.geometry.values), max_segment_length=step_m)
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    same = idx[1:] == idx[:-1]
    a, b = coords[:-1][same], coords[1:][same]
    mid, length = (a + b) / 2.0, np.hypot(*(b - a).T)
    row, col, ok = _cell_index(grid, mid[:, 0], mid[:, 1])
    np.add.at(acc, (row[ok], col[ok]), length[ok])
    return acc


class RoadNetworkPipeline(BasePipeline):
    """Road network of a bounding box from OpenStreetMap, with length by class and road density.

    Output ``road_density.tif`` is kilometres of road per square kilometre in cells of ``cell_m`` metres;
    ``road_network.gpkg`` holds every way with its ``highway`` tag, name, surface and length in metres (clipped to the
    box). ``exclude`` drops ways that are not roads yet (proposed, under construction) or no longer.

    This is a map of what OpenStreetMap volunteers have drawn, not a detection from imagery: coverage of minor roads
    and tracks in rural areas is often incomplete, so compare densities between places only where mapping effort is
    similar. Use it to train a road segmentation model (rasterise the lines as labels), then run that model with the
    ``road_extraction`` pipeline.
    """
    name = "road_network"
    description = "Road network from OpenStreetMap: length by road class, road density and a GeoPackage of every road"
    domain = "infrastructure"
    tags = ["roads", "osm", "network", "reference-product"]

    def run(self, bbox, output_dir="./output", cell_m=500.0, exclude=NOT_BUILT_YET, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if cell_m <= 0:
                return self._fail("cell_m must be positive.", t0)
            raw = fp.osm_features(bbox, "highway", session=kwargs.get("session"))
            raw = raw[~raw["highway"].isin(list(exclude))].reset_index(drop=True)
            gdf = _clip(raw, bbox)
            utm = _utm(bbox)
            grid = make_grid(bbox, cell_m, crs=utm)
            lines = gdf.to_crs(utm)
            gdf["length_m"] = lines.length.round(1).to_numpy()
            gdf["group"] = [_group_of(t) for t in gdf["highway"]]
            gpkg = out / "road_network.gpkg"
            if len(gdf):
                gdf.to_file(gpkg, layer="roads", driver="GPKG")
            acc = length_per_cell(lines, grid, max(cell_m / 5.0, 10.0))
            inside = _cells_inside(grid, bbox)
            dens = np.where(inside, acc / 1000.0 / (cell_m * cell_m / 1e6), np.nan).astype("float32")
            path = write_float(out / "road_density.tif", dens, grid.profile(), [f"km of road per km2, {cell_m:g} m cells"])
            aoi_km2 = _aoi_km2(bbox)
            total_km = float(gdf["length_m"].sum()) / 1000.0
            by_tag = gdf.groupby("highway")["length_m"].sum().sort_values(ascending=False) / 1000.0
            by_group = gdf.groupby("group")["length_m"].sum() / 1000.0
            stats = {
                "total_length_km": total_km, "road_density_km_per_km2": total_km / aoi_km2, "segments": int(len(gdf)),
                "km_by_highway": {k: round(float(v), 3) for k, v in by_tag.items()},
                "km_by_group": {k: round(float(v), 3) for k, v in by_group.items()},
                "unnamed_share_of_length": (float(gdf.loc[gdf["name"].isna(), "length_m"].sum()) / 1000.0 / total_km) if total_km else None,
                "aoi_area_km2": aoi_km2, "cell_m": cell_m, "excluded_tags": list(exclude),
                "product_note": "OpenStreetMap highway ways (volunteered data); completeness of minor roads and tracks varies by place.",
                "network_file": str(gpkg) if len(gdf) else None,
            }
            meta = {"date": "n/a (published dataset)", "crs": utm}
            if len(gdf):
                meta["vector_files"] = [{"name": "roads", "path": str(Path(gpkg).resolve())}]
            return self._ok(path, stats, meta, t0)
        except (fp.FootprintSourceError, ValueError) as exc:
            return self._fail(str(exc), t0)


VECTOR_PIPELINES: dict[str, type] = {RoadNetworkPipeline.name: RoadNetworkPipeline}