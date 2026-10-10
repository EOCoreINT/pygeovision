"""Pipelines that existed in earlier builds and were removed before launch.

A pipeline is removed when it cannot return a correct result today. The
test applied to every one of them was: *does the output mean what the
name says, computed from real inputs, with no invented constant or
untrained network standing in for calibration or training?*

Asking for one of these through ``get_pipeline`` / ``client.pipeline`` /
``pygeovision channel`` raises an error that quotes the reason below, so
nobody gets a silent "unknown pipeline". Each reason also says what would
be needed to bring the pipeline back.
"""
from __future__ import annotations

_UNTRAINED = (
    "It ran a neural network with no trained weights, so the output was random "
    "(two identical runs disagreed on ~38% of pixels). "
)
_COCO = (
    "It ran an everyday-photo (COCO) detector on 10 m satellite pixels, where "
    "a ship spans a few pixels and a car is smaller than one, so detections "
    "were not meaningful. Needs a detector trained on high-resolution imagery. "
)
_STATIC_WC = (
    "It compared two land-cover maps that were both the same static ESA "
    "WorldCover 2021 product, so it reported 0% change whatever the imagery "
    "showed. Needs a date-specific classifier (trained model) to bring back. "
)

REMOVED_PIPELINES: dict[str, str] = {
    # -- never implemented ------------------------------------------------
    "air_quality_index": (
        "It was never implemented (needs Sentinel-5P NO2/aerosol data, which no "
        "PyGeoFetch provider offers). An uncalibrated proxy under this name could "
        "be read as health guidance."
    ),
    "tree_species": (
        "It always returned failure: no species classifier exists. Train one and "
        "run it with your own weights."
    ),
    # -- untrained network ------------------------------------------------
    "crop_type_mapping": _UNTRAINED + "Use crop_monitoring with your own checkpoint.",
    "road_extraction": _UNTRAINED + "Train a road-segmentation model with GeoTrainer to bring it back.",
    "landslide_detection": (
        _UNTRAINED + "Its description also claimed a DEM input that was never used."
    ),
    "canopy_height": (
        "CHMv2Model builds an untrained decoder (no canopy-height weights exist in "
        "the package), so heights were random; the pipeline also reported success "
        "when the model returned an error."
    ),
    # -- output is not what the name says --------------------------------
    "irrigation_detection": "It returned a water mask (NDWI), which is not irrigation. Use water_bodies.",
    "infrastructure_monitoring": (
        "A generic change detector backed by an untrained model, with a fallback that "
        "always flagged exactly 10% of pixels. Use change_detection."
    ),
    "forest_fire": (
        "A duplicate of wildfire_severity (dNBR burn severity); it never detected "
        "active fire. Use wildfire_severity."
    ),
    "carbon_estimation": (
        "Carbon was 50 x NDVI^2 x 0.47 with an invented constant; NDVI saturates in "
        "dense vegetation, so the numbers were not carbon estimates. Needs "
        "field-biomass calibration or a biomass product."
    ),
    "crop_yield_forecast": (
        "It produced a peak-NDVI map, not a yield; the 'forecast' was mean NDVI times "
        "a number you typed. Use vegetation_indices."
    ),
    "pipeline_leak_detection": (
        "The crop_health NDVI-drop computation under another name; vegetation stress "
        "is not evidence of a leak."
    ),
    # -- heuristics that cannot be validated -----------------------------
    "dust_storm_tracking": (
        "A blue-reflectance/NDVI heuristic with an arbitrary threshold that cannot "
        "tell dust from cloud, smoke, snow or harvest. Needs aerosol data "
        "(Sentinel-5P / MODIS AOD)."
    ),
    "reef_bleaching": (
        "Brightness change over shallow water is dominated by tide, turbidity and sun "
        "glint, not bleaching; it was never validated against reef surveys."
    ),
    "powerline_extraction": (
        "It flagged any elongated gap in forest (roads, firebreaks, pipelines) at "
        "10 m, where lines and towers are not visible."
    ),
    "aquaculture_mapping": (
        "It flagged any rectangular water body by bounding-box fill ratio; "
        "reservoirs and lakes were false positives and it was never validated."
    ),
    "oil_spill_detection": (
        "A fixed -18 dB threshold marks every calm water surface (low wind, lakes, "
        "rivers) as a candidate; it cannot be validated without labelled SAR slicks "
        "and wind data."
    ),
    # -- detectors with the wrong input resolution ------------------------
    "ocean_ship_detection": _COCO,
    "port_monitoring": _COCO,
    "parking_occupancy": _COCO,
    # -- built on a static reference map ----------------------------------
    "landcover_change": _STATIC_WC,
    "mine_detection": _STATIC_WC,
    "construction_progress": _STATIC_WC,
    "mangrove_mapping": _STATIC_WC,
}
