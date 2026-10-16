"""Small checks the case-study pages use to read a PyGeoVision report.

Each function takes numbers (or a file the pipeline wrote) and returns numbers. None of them needs a network.
``tests/test_case_studies.py`` runs every one against the figures printed in the reports.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def class_areas(pixels: Mapping[str, int], pixel_ha: float) -> dict[str, tuple[float, float]]:
    """Area in hectares and share in per cent of every class, from pixel counts and the area of one pixel."""
    total = sum(pixels.values())
    return {name: (round(n * pixel_ha, 1), round(100 * n / total, 1)) for name, n in pixels.items()}


def flood_summary(pixels: Mapping[str, int], pixel_m: float) -> dict[str, float]:
    """Re-derive the headline figures of a ``multisensor_flood_mapping`` report from its class pixel counts.

    ``pixels`` has the keys ``not_flooded``, ``radar_optical``, ``radar_only``, ``optical_only``, ``disputed`` and
    ``reference_water`` (classes 0 to 5). Reference water is excluded from the land the flooded share is taken of.
    """
    ha = pixel_m * pixel_m / 1e4
    flooded = (pixels["radar_optical"] + pixels["radar_only"] + pixels["optical_only"]) * ha
    land = (sum(pixels.values()) - pixels["reference_water"]) * ha
    return {
        "flooded_ha": flooded,
        "flooded_upper_ha": flooded + pixels["disputed"] * ha,
        "land_ha": land,
        "flooded_share_of_land_pct": 100 * flooded / land,
        "optical_only_share_of_flooded_pct": 100 * pixels["optical_only"] * ha / flooded,
        "confirmed_share_of_flooded_pct": 100 * pixels["radar_optical"] * ha / flooded,
    }


def rowwise_area_ha(mask: np.ndarray, west: float, south: float, east: float, north: float) -> float:
    """Area in hectares of the True pixels of a raster in EPSG:4326, counting each row at its own latitude.

    A degree of longitude is shorter the further you are from the equator, so a pixel at the top of a box is smaller than one at
    the bottom. Multiplying a pixel count by one average pixel area ignores that; this sums row by row.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    rows, cols = mask.shape
    dx, dy = (east - west) / cols, (north - south) / rows
    per_row = mask.sum(axis=1)
    total = 0.0
    for i in np.nonzero(per_row)[0]:
        top, bottom = north - dy * i, north - dy * (i + 1)
        cell, _ = geod.polygon_area_perimeter([west, west + dx, west + dx, west], [top, top, bottom, bottom])
        total += abs(cell) * int(per_row[i])
    return total / 1e4


def shannon(counts: Sequence[int]) -> tuple[float, float]:
    """Shannon index H' = -sum(p ln p) and evenness H' / ln(classes present), from class pixel counts."""
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    h = -sum(c / total * math.log(c / total) for c in counts)
    return h, (h / math.log(len(counts)) if len(counts) > 1 else 0.0)


def hand_fits(bbox: tuple[float, float, float, float], resolution_m: float, limit: int = 6_000_000) -> tuple[int, bool]:
    """Cells of the output grid for ``bbox`` at ``resolution_m`` (the grid the pipeline builds), and whether the terrain
    stage accepts it. The terrain stage refuses a grid of more than ``limit`` cells."""
    from pygeovision.ai.pipelines.reference_products import make_grid

    grid = make_grid(bbox, float(resolution_m), max_pixels=10 ** 13)
    cells = grid.width * grid.height
    return cells, cells <= limit


def ndci_undefined(path: str) -> dict[str, float]:
    """Count the water pixels of a ``water_quality.tif`` that have a red reflectance but no NDCI.

    Band 1 is NDCI and band 2 is red reflectance, both only inside the water. NDCI is empty where red and red-edge reflectance
    add up to zero, so those pixels are water pixels without an indicator.
    """
    import rasterio

    with rasterio.open(path) as src:
        ndci, red = src.read(1), src.read(2)
    water = np.isfinite(red)
    missing = water & ~np.isfinite(ndci)
    return {
        "water_pixels": int(water.sum()),
        "ndci_pixels": int((water & np.isfinite(ndci)).sum()),
        "missing": int(missing.sum()),
        "missing_share_pct": 100 * float(missing.sum()) / max(int(water.sum()), 1),
        "red_exactly_zero": int((water & (red == 0)).sum()),
    }


def vector_area_by_class(gpkg: str, layer: str) -> dict[str, float]:
    """Total ``area_ha`` of each ``class_name`` in a vector layer a pipeline wrote (``deliverables/vectors/*.gpkg``)."""
    import geopandas as gpd

    gdf = gpd.read_file(gpkg, layer=layer)
    return {str(k): round(float(v), 1) for k, v in gdf.groupby("class_name")["area_ha"].sum().items()}


def drainage_density(stream_km: float, grid_km2: float) -> float:
    """Stream length per square kilometre of the analysed grid (not of the requested box)."""
    return stream_km / grid_km2


def classify_dnbr(dnbr: np.ndarray, thresholds: Sequence[float] = (0.10, 0.27, 0.66)) -> np.ndarray:
    """Burn-severity classes from a dNBR raster, the way ``wildfire_severity`` assigns them.

    0 unburned, 1 low, 2 moderate, 3 high (a value on a threshold takes the higher class); 255 where dNBR is not finite.
    Pass other ``thresholds`` to try a local calibration on the ``dnbr.tif`` the pipeline wrote.
    """
    classes = np.zeros(dnbr.shape, dtype=np.uint8)
    for k, t in enumerate(thresholds, start=1):
        classes[dnbr >= t] = k
    classes[~np.isfinite(dnbr)] = 255
    return classes


def flagged_by_land_cover(flag_path: str, land_cover_path: str) -> dict[int, int]:
    """Count the flagged pixels of a ``crop_health_anomaly.tif`` (class 1) in each class of a ``land_cover.tif``.

    The two rasters usually have different grids, so the land cover is resampled onto the flag raster's grid with nearest
    neighbour first. Returns ``{land-cover value: flagged pixels}``; 255 is flagged pixels the land cover does not cover.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(flag_path) as flag_src, rasterio.open(land_cover_path) as cover_src:
        flag = flag_src.read(1)
        cover = np.full(flag.shape, 255, dtype="uint8")
        reproject(rasterio.band(cover_src, 1), cover, dst_transform=flag_src.transform, dst_crs=flag_src.crs,
                  resampling=Resampling.nearest, dst_nodata=255)
    values, counts = np.unique(cover[flag == 1], return_counts=True)
    return {int(v): int(n) for v, n in zip(values, counts)}
