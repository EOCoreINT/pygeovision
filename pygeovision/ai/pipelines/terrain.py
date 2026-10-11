"""Terrain pipelines: slope and ruggedness, and drainage (flow accumulation, streams, HAND).

Both read a DEM -- by default the Copernicus GLO-30 DEM from the Planetary Computer, which needs no account or key;
``dem_source="copernicus_aws"`` reads the same product from AWS Open Data (also the automatic fallback if the
Planetary Computer read fails); ``dem_source="opentopography"`` uses an OpenTopography product instead (needs the
key the other DEM pipelines use) -- resample it to a local-UTM grid so that distances and slopes are in metres, and compute standard terrain measures.
Nothing is calibrated and no threshold decides what is "hazardous": the outputs describe the ground surface.

The Copernicus DEM is a *surface* model: it includes tree canopy and buildings, which raises elevation there and
distorts slope and drainage in forest and towns.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import PipelineResult
from pygeovision.ai.pipelines import _hydrology as hy
from pygeovision.ai.pipelines._raster import write_classes, write_float
from pygeovision.ai.pipelines.domains import _DemPipeline
from pygeovision.data.reference import (
    GridRaster,
    PlanetaryComputer,
    ReferenceDataError,
    TargetGrid,
    copernicus_dem_aws_urls,
    make_grid,
    read_to_grid,
)

logger = logging.getLogger(__name__)

DEM_NOTE = ("Copernicus GLO-30 digital surface model (ESA/Airbus, 30 m): heights include vegetation and buildings, "
            "and its vertical accuracy is a few metres, so small features and gentle slopes are uncertain.")


class _TerrainBase(_DemPipeline):
    collection = "cop-dem-glo-30"

    def _pc(self) -> PlanetaryComputer:
        pc = getattr(self, "pc", None)
        if pc is None:
            pc = self.pc = PlanetaryComputer()
        return pc

    def _elevation(self, bbox, out: Path, dem_source: str, dem_type: str, resolution_m: float, pad: int = 0,
                   grid: TargetGrid | None = None) -> GridRaster:
        """DEM on a local-UTM grid. ``pad`` extra cells are read on every side (for neighbourhood operators).

        ``grid`` reads onto an existing grid (another pipeline's) instead of making one.
        """
        if resolution_m <= 0:
            raise ReferenceDataError("resolution_m must be positive.")
        grid = grid or make_grid(bbox, resolution_m)
        if pad:
            from affine import Affine
            t = grid.transform
            grid = TargetGrid(crs=grid.crs, transform=Affine(t.a, 0.0, t.c - pad * grid.res, 0.0, t.e, t.f + pad * grid.res),
                              width=grid.width + 2 * pad, height=grid.height + 2 * pad, res=grid.res)
        if dem_source == "copernicus":
            pc = self._pc()
            items = pc.search(self.collection, bbox)
            if not items:
                raise ReferenceDataError("no Copernicus DEM tile covers this area (it is open sea, or outside the product).")
            urls = pc.asset_urls(self.collection, items, "data")
            if not urls:
                raise ReferenceDataError("the Copernicus DEM tiles for this area have no 'data' asset")
            try:
                r = read_to_grid(urls, grid, resampling="bilinear", dtype="float32")
            except ReferenceDataError as exc:
                # The same Copernicus GLO-30 product is published, unchanged, on AWS Open Data; use it rather than fail.
                logger.warning("Planetary Computer DEM read failed (%s); falling back to the AWS Open Data copy of the "
                               "same Copernicus GLO-30 DEM.", exc)
                try:
                    r = read_to_grid(copernicus_dem_aws_urls(bbox), grid, resampling="bilinear", dtype="float32",
                                     skip_missing=True)
                except ReferenceDataError as exc2:
                    raise ReferenceDataError(f"{exc} | the AWS Open Data fallback also failed: {exc2}") from exc2
        elif dem_source == "copernicus_aws":
            r = read_to_grid(copernicus_dem_aws_urls(bbox), grid, resampling="bilinear", dtype="float32",
                             skip_missing=True)
        elif dem_source == "opentopography":
            path, *_ = self._fetch_dem(bbox, out, dem_type)
            r = read_to_grid([str(path)], grid, resampling="bilinear", dtype="float32")
        else:
            raise ReferenceDataError("dem_source must be 'copernicus', 'copernicus_aws' or 'opentopography'.")
        if not r.valid.any():
            raise ReferenceDataError("the DEM holds no data for this area.")
        return r

    @staticmethod
    def _dem_host(r: GridRaster) -> str:
        """Host the DEM pixels actually came from (shows whether the Planetary Computer or the AWS copy was used)."""
        from urllib.parse import urlparse
        return (urlparse(r.sources[0]).netloc or "local file") if r.sources else "unknown"


def hand_from_dem(dem: np.ndarray, valid: np.ndarray, res: float, stream_threshold_km2: float = 1.0,
                  max_cells: int = 6_000_000) -> np.ndarray:
    """Height Above Nearest Drainage (m) of every cell: Priority-Flood conditioning, D8 routing, streams where the
    contributing area reaches ``stream_threshold_km2``. NaN where the DEM is invalid or no stream is downstream."""
    hy.check_size(dem.shape, int(max_cells))
    z = np.where(valid, dem, np.nan).astype("float64")
    filled = hy.priority_flood_epsilon(z)
    target = hy.d8_targets(filled, res)
    acc = hy.flow_accumulation(filled, target) * (res * res) / 1e6
    stream = (acc >= stream_threshold_km2) & valid
    return np.where(valid, hy.height_above_drainage(filled, target, stream.ravel()), np.nan)


def _tri(z: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index (Riley et al. 1999): root of the summed squared differences to the 8 neighbours (m)."""
    p = np.pad(z, 1, mode="edge")
    acc = np.zeros(z.shape)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                acc += (p[1 + dr:1 + dr + z.shape[0], 1 + dc:1 + dc + z.shape[1]] - z) ** 2
    return np.sqrt(acc)


class TerrainAnalysisPipeline(_TerrainBase):
    """Elevation, slope, aspect, hillshade and ruggedness of a bounding box from a DEM.

    Slope and aspect are Horn's 3x3 method (as ``gdaldem``); aspect is the compass bearing, clockwise from north,
    that the slope faces (downhill direction), and is NaN on exactly flat cells. The hillshade lights the terrain
    from azimuth ``sun_azimuth_deg`` and elevation ``sun_elevation_deg`` (315 and 45 by convention) and is a picture,
    not an irradiance (see ``solar_potential``). TRI is the Terrain Ruggedness Index of Riley et al. (1999).
    ``slope_thresholds_deg`` set which ``pct_area_slope_gt_<x>deg`` shares are reported.

    Outputs: ``elevation.tif``, ``slope_deg.tif``, ``aspect_deg.tif``, ``hillshade.tif``, ``tri_m.tif``.
    """
    name = "terrain_analysis"
    description = "Elevation, slope, aspect, hillshade and ruggedness (TRI) from the Copernicus 30 m DEM"
    domain = "terrain"
    tags = ["dem", "slope", "aspect", "terrain"]

    def run(self, bbox, output_dir="./output", dem_source="copernicus", dem_type="cop30", resolution_m=30.0,
            slope_thresholds_deg=(5, 15, 30), sun_azimuth_deg=315.0, sun_elevation_deg=45.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            r = self._elevation(bbox, out, dem_source, dem_type, float(resolution_m), pad=1)
            res = r.grid.res
            inner = (slice(1, -1), slice(1, -1))               # the one-cell margin only feeds the neighbourhood maths
            valid = r.valid[inner]
            if not valid.any():
                raise ReferenceDataError("the DEM holds no data for this area.")
            z = self._fill_nan_nearest(np.where(r.valid, r.data, np.nan).astype("float64"))
            slope, aspect = self._horn(z, res, res)
            slope_deg = np.degrees(slope)
            aspect_deg = np.degrees(aspect) % 360.0
            aspect_deg[slope == 0] = np.nan
            zen = math.radians(90.0 - sun_elevation_deg)
            illum = (math.cos(zen) * np.cos(slope)
                     + math.sin(zen) * np.sin(slope) * np.cos(math.radians(sun_azimuth_deg) - aspect))
            hill = np.clip(illum, 0.0, 1.0)
            tri = _tri(z)
            z, slope, slope_deg, aspect, aspect_deg, hill, tri = (a[inner] for a in (z, slope, slope_deg, aspect, aspect_deg, hill, tri))

            def masked(a):
                return np.where(valid, a, np.nan).astype("float32")
            prof = make_grid(bbox, res).profile()
            write_float(out / "elevation.tif", masked(z), prof, ["elevation, m"])
            path = write_float(out / "slope_deg.tif", masked(slope_deg), prof, ["slope, degrees"])
            write_float(out / "aspect_deg.tif", masked(aspect_deg), prof, ["downhill bearing, degrees clockwise from north"])
            write_float(out / "hillshade.tif", masked(hill), prof, ["hillshade 0-1"])
            write_float(out / "tri_m.tif", masked(tri), prof, ["terrain ruggedness index, m"])
            sv = slope_deg[valid]
            stats: dict[str, Any] = {
                "min_elevation_m": float(z[valid].min()), "max_elevation_m": float(z[valid].max()),
                "mean_elevation_m": float(z[valid].mean()), "relief_m": float(z[valid].max() - z[valid].min()),
                "mean_slope_deg": float(sv.mean()), "median_slope_deg": float(np.median(sv)),
                "p95_slope_deg": float(np.percentile(sv, 95)), "max_slope_deg": float(sv.max()),
                "mean_tri_m": float(tri[valid].mean()), "valid_fraction": float(valid.mean()),
                "resolution_m": res, "dem_source": dem_source, "dem_host": self._dem_host(r),
                "sun_azimuth_deg": sun_azimuth_deg,
                "sun_elevation_deg": sun_elevation_deg, "product_note": DEM_NOTE if dem_source.startswith("copernicus") else
                f"OpenTopography product {dem_type}.",
            }
            for x in slope_thresholds_deg:
                stats[f"pct_area_slope_gt_{x:g}deg"] = float((sv > x).mean())
            return self._ok(path, stats, {"crs": r.grid.crs, "date": "n/a (elevation model)"}, t0)
        except (ReferenceDataError, ValueError, RuntimeError) as exc:
            return self._fail(str(exc), t0)


class DrainageAnalysisPipeline(_TerrainBase):
    """Flow accumulation, stream network (Strahler order) and Height Above Nearest Drainage from a DEM.

    The DEM is conditioned with Priority-Flood (depressions filled), water is routed D8 (to the steepest downhill
    neighbour) and ``flow_accumulation_km2.tif`` is the area draining through each cell. Cells draining at least
    ``stream_threshold_km2`` form the stream network (``stream_order.tif``, Strahler order; ``stream_network.gpkg``
    holds the lines). ``hand_m.tif`` is each cell's height above the stream cell it drains into; the shares of the
    area within ``hand_thresholds_m`` of the drainage are reported.

    HAND describes how high the ground stands above the channel -- low HAND is a *necessary* condition for river
    flooding, not a flood map: it knows nothing of rainfall, discharge, channel capacity or defences. The stream
    threshold is your choice of how much catchment makes a channel; re-run with another value, or threshold
    ``flow_accumulation_km2.tif`` yourself. Contributing areas are truncated at the edge of the box, and the DEM is a
    surface model (canopy and buildings included).
    """
    name = "drainage_analysis"
    description = "Flow accumulation, stream network with Strahler order, and Height Above Nearest Drainage (HAND) from a DEM"
    domain = "terrain"
    tags = ["dem", "hydrology", "streams", "hand", "flood-susceptibility"]

    def run(self, bbox, output_dir="./output", dem_source="copernicus", dem_type="cop30", resolution_m=30.0,
            stream_threshold_km2=1.0, hand_thresholds_m=(5, 10, 15), max_cells=6_000_000, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if stream_threshold_km2 <= 0:
                raise ValueError("stream_threshold_km2 must be positive.")
            grid = make_grid(bbox, float(resolution_m))
            hy.check_size((grid.height, grid.width), int(max_cells))
            r = self._elevation(bbox, out, dem_source, dem_type, float(resolution_m))
            dem = np.where(r.valid, r.data, np.nan).astype("float64")
            res = r.grid.res
            logger.info("drainage analysis on %d x %d cells", dem.shape[1], dem.shape[0])
            filled = hy.priority_flood_epsilon(dem)
            target = hy.d8_targets(filled, res)
            acc = hy.flow_accumulation(filled, target)
            acc_km2 = acc * (res * res) / 1e6
            valid = r.valid
            stream = (acc_km2 >= stream_threshold_km2) & valid
            order_hl = np.argsort(-np.nan_to_num(filled.ravel(), nan=-np.inf), kind="stable")
            so = hy.strahler_order(target, stream.ravel(), order_hl).reshape(dem.shape)
            hand = hy.height_above_drainage(filled, target, stream.ravel())
            prof = r.grid.profile()
            write_float(out / "flow_accumulation_km2.tif", np.where(valid, acc_km2, np.nan).astype("float32"), prof,
                        ["contributing area, km2"])
            path = write_classes(out / "stream_order.tif", so.astype(np.uint8), valid, prof,
                                 ["0=no stream; n=Strahler order; 255=nodata"])
            write_float(out / "hand_m.tif", np.where(valid, hand, np.nan).astype("float32"), prof,
                        ["height above nearest drainage, m"])
            gpkg = self._write_streams(out / "stream_network.gpkg", target, so, r.grid)
            length_km = hy.stream_length_m(target, stream.ravel(), dem.shape[1], res) / 1000.0
            area_km2 = float(valid.sum()) * res * res / 1e6
            hv = hand[valid & np.isfinite(hand)]
            depth = (filled - dem)[valid]
            stats: dict[str, Any] = {
                "stream_threshold_km2": stream_threshold_km2, "stream_length_km": length_km,
                "drainage_density_km_per_km2": length_km / area_km2 if area_km2 else None,
                "max_strahler_order": int(so.max()), "stream_cells": int(stream.sum()),
                "largest_contributing_area_km2": float(np.nanmax(acc_km2)),
                "mean_hand_m": float(hv.mean()) if hv.size else None,
                "hand_defined_fraction": float(hv.size / max(int(valid.sum()), 1)),
                "cells_filled": int((depth > 1e-6).sum()), "max_fill_depth_m": float(depth.max()),
                "valid_fraction": float(valid.mean()), "resolution_m": res, "dem_source": dem_source,
                "dem_host": self._dem_host(r), "area_km2": area_km2,
                "product_note": DEM_NOTE if dem_source.startswith("copernicus") else f"OpenTopography product {dem_type}.",
            }
            for x in hand_thresholds_m:
                stats[f"pct_area_hand_le_{x:g}m"] = float((hv <= x).sum() / max(int(valid.sum()), 1))
            meta: dict[str, Any] = {"crs": r.grid.crs, "date": "n/a (elevation model)"}
            if gpkg is not None:
                meta["vector_files"] = [{"name": "streams", "path": str(Path(gpkg).resolve())}]
            return self._ok(path, stats, meta, t0)
        except (ReferenceDataError, ValueError, RuntimeError) as exc:
            return self._fail(str(exc), t0)

    @staticmethod
    def _write_streams(path: Path, target: np.ndarray, order: np.ndarray, grid) -> Path | None:
        import geopandas as gpd
        from shapely.geometry import LineString
        flat_order = order.ravel()
        idx = np.nonzero((flat_order > 0) & (target >= 0))[0]
        if idx.size == 0:
            return None
        w = grid.width
        t = target[idx]
        x0 = grid.transform.c + grid.transform.a * ((idx % w) + 0.5)
        y0 = grid.transform.f + grid.transform.e * ((idx // w) + 0.5)
        x1 = grid.transform.c + grid.transform.a * ((t % w) + 0.5)
        y1 = grid.transform.f + grid.transform.e * ((t // w) + 0.5)
        lines = [LineString([(a, b), (c, d)]) for a, b, c, d in zip(x0, y0, x1, y1)]
        gdf = gpd.GeoDataFrame({"strahler": flat_order[idx].astype(int)}, geometry=lines, crs=grid.crs)
        # one multi-line feature per Strahler order keeps the file small and is what a map legend wants
        gdf = gdf.dissolve(by="strahler").reset_index()
        gdf["length_km"] = (gdf.length / 1000.0).round(3)
        gdf.to_crs("EPSG:4326").to_file(path, layer="streams", driver="GPKG")
        return path


TERRAIN_PIPELINES: dict[str, type] = {c.name: c for c in (TerrainAnalysisPipeline, DrainageAnalysisPipeline)}
