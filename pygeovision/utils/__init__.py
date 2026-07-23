"""
pygeovision.utils
=================
Common utilities extracted from repeated patterns across all pygeovision
project notebooks. Import once, use everywhere.

    from pygeovision.utils import (
        aoi_bbox,            # GeoJSON AOI → (lon_min, lat_min, lon_max, lat_max)
        save_aoi_geojson,    # write AOI as FeatureCollection GeoJSON
        best_scene,          # pick lowest-cloud-cover scene from search results
        download_ok,         # download + return first successful path
        save_raster,         # write ndarray → GeoTIFF from a reference raster
        DarkFigure,          # dark-themed matplotlib figure context manager
        show_panel,          # imshow panel with white colorbar + optional note
        JsonReport,          # accumulate + write structured JSON reports
        band_ndvi,           # NDVI from stacked raster
        band_evi,            # EVI from stacked raster
        band_mndwi,          # MNDWI from stacked raster (shoreline)
        download_s1,         # Sentinel-1 download + GCP recovery in one call
        preprocess_s2,       # Sentinel-2 clip + normalise in one call
        fetch_kp,            # NOAA SWPC Kp index (live JSON)
        fetch_f107,          # NOAA SWPC F10.7 (live JSON)
        s4_to_position_error,# S4 → GNSS positioning error (Aquino 2009)
    )
"""
from pygeovision.utils.acquire import best_scene, download_ok, download_s1, preprocess_s2
from pygeovision.utils.geo import aoi_bbox, save_aoi_geojson
from pygeovision.utils.raster import band_evi, band_mndwi, band_ndvi, save_raster
from pygeovision.utils.report import JsonReport
from pygeovision.utils.space_weather import fetch_f107, fetch_kp, s4_to_position_error
from pygeovision.utils.viz import DarkFigure, show_panel

__all__ = [
    # Geometry
    "aoi_bbox", "save_aoi_geojson",
    # Acquisition
    "best_scene", "download_ok", "download_s1", "preprocess_s2",
    # Raster
    "save_raster", "band_ndvi", "band_evi", "band_mndwi",
    # Visualisation
    "DarkFigure", "show_panel",
    # Reporting
    "JsonReport",
    # Space weather
    "fetch_kp", "fetch_f107", "s4_to_position_error",
]
