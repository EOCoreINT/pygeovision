"""What each pipeline's report says: layers, metrics, method and caveats.

One :class:`ReportSpec` per pipeline. The report writers are generic; everything
that depends on *which* analysis was run lives here, in plain data plus a few
headline sentences. Texts state only what the pipeline actually computes (see
the pipeline docstrings) -- thresholds, formulas, and what the result is *not*.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pygeovision.reporting.palettes import (
    DYNAMIC_WORLD,
    NO_CHANGE,
    WORLDCOVER,
    ClassDef,
    categorical,
)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Binning:
    """Turn a continuous raster into classes (for the class table and for vectors).

    kind:
      ``fixed``     inner ``edges`` (ascending); ``names`` and ``colors`` have len(edges)+1 entries
      ``equal``     ``n`` equal-width classes between the 2nd and 98th percentile
      ``quantile``  ``n`` classes with equal pixel counts
      ``std``       three classes around the mean: below -k sd, within, above +k sd
    """
    kind: str
    edges: tuple[float, ...] = ()
    names: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    n: int = 0
    k: float = 1.0
    unit: str = ""
    cmap: str = "viridis"
    keep: tuple[int, ...] | None = None  # bin indices to polygonise (None = all)


@dataclass(frozen=True)
class VectorSpec:
    """Which part of a layer becomes polygons."""
    name: str                                # layer / file name
    keep: tuple[int, ...] | None = None      # class values to polygonise (None = every valid class)
    skip: tuple[int, ...] = ()               # values never polygonised (e.g. background)


@dataclass(frozen=True)
class Layer:
    key: str
    file: str                                # relative to the pipeline output directory
    title: str
    kind: str                                # "classes" | "continuous"
    classes: tuple[ClassDef, ...] = ()
    classes_hint: str = ""                   # "landcover" | "crops" | "user" -> resolved at run time
    band: int = 1
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    center: float | None = None              # diverging colour map centre
    label: str = ""                          # colour-bar / axis label
    binning: Binning | None = None
    vector: VectorSpec | None = None
    primary: bool = False
    optional: bool = False
    muted: tuple[int, ...] = ()              # class values drawn in the quiet "nothing here" colour


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    fmt: str = "text"
    note: str = ""


@dataclass(frozen=True)
class ReportSpec:
    pipeline: str
    title: str
    question: str
    method: tuple[str, ...]
    layers: tuple[Layer, ...]
    metrics: tuple[Metric, ...]
    headline: Callable[[Any], list[str]]
    reading: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    chart: str = "auto"                      # "auto" | "timeseries" | "before_after"
    multi_date: bool = False


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

R_NDWI = "McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing 17(7), 1425-1432."
R_MNDWI = "Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing 27(14), 3025-3033."
R_NBR = "Key, C.H. & Benson, N.C. (2006). Landscape Assessment (LA). In: FIREMON: Fire Effects Monitoring and Inventory System, USDA Forest Service RMRS-GTR-164-CD."
R_NDSI = "Hall, D.K., Riggs, G.A. & Salomonson, V.V. (1995). Development of methods for mapping global snow cover using Moderate Resolution Imaging Spectroradiometer data. Remote Sensing of Environment 54(2), 127-140."
R_EVI = "Huete, A. et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment 83(1-2), 195-213."
R_NDCI = "Mishra, S. & Mishra, D.R. (2012). Normalized difference chlorophyll index: a novel model for remote estimation of chlorophyll-a concentration in turbid productive waters. Remote Sensing of Environment 117, 394-406."
R_LRM = "Hesse, R. (2010). LiDAR-derived Local Relief Models - a new tool for archaeological prospection. Archaeological Prospection 17(2), 67-72."
R_MDH = "Devereux, B.J., Amable, G.S. & Crow, P. (2008). Visualisation of LiDAR terrain models for archaeological feature detection. Antiquity 82(316), 470-479."
R_HORN = "Horn, B.K.P. (1981). Hill shading and the reflectance map. Proceedings of the IEEE 69(1), 14-47."
R_OTSU = "Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics 9(1), 62-66."
R_WC = "Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. doi:10.5281/zenodo.7254221."
R_DW = "Brown, C.F. et al. (2022). Dynamic World, near real-time global 10 m land use land cover mapping. Scientific Data 9, 251."
R_SHANNON = "Shannon, C.E. (1948). A mathematical theory of communication. Bell System Technical Journal 27, 379-423."
R_SUN = "Duffie, J.A. & Beckman, W.A. Solar Engineering of Thermal Processes (Wiley); Spencer, J.W. (1971). Fourier series representation of the position of the sun. Search 2(5), 172."
R_LST = "USGS. Landsat Collection 2 Level-2 Science Product Guide (LSDS-1619)."
R_S2 = "ESA. Sentinel-2 Level-2A surface reflectance (Copernicus programme)."

_USER_CLASSES_NOTE = ("Class meanings come from the labels your network was trained on; this report cannot know them. "
                      "Pass class_names (CLI: --class-names) to name them.")
_NETWORK_LIMIT = ("This map is the output of a network you trained. The report does not measure its accuracy: "
                  "validate it against reference data before quoting the figures.")

# ---------------------------------------------------------------------------
# Headline sentences (each receives the run context ``c``)
# ---------------------------------------------------------------------------


def _h_crop_health(c):
    s = c.s
    return [f"NDVI fell by more than {c.num(s.get('anomaly_threshold'), 2)} on {c.f('frac', s.get('pct_flagged'))} "
            f"of the valid area ({c.f('ha', c.ha('anomaly', 1))}) between {c.m.get('date_before')} and {c.m.get('date_after')}.",
            f"Mean NDVI went from {c.num(s.get('mean_ndvi_before'), 2)} to {c.num(s.get('mean_ndvi_after'), 2)}."]


def _h_veg_indices(c):
    rows = c.timeseries
    if not rows:
        return ["No date produced usable imagery."]
    best = max(rows, key=lambda r: r.get("mean_ndvi") if r.get("mean_ndvi") is not None else -9)
    worst = min(rows, key=lambda r: r.get("mean_ndvi") if r.get("mean_ndvi") is not None else 9)
    out = [f"{len(rows)} of {len(c.m.get('dates') or rows)} requested dates had usable imagery."]
    if best.get("mean_ndvi") is not None:
        out.append(f"Mean NDVI peaked at {c.num(best['mean_ndvi'], 2)} ({best['date']}) and was lowest at "
                   f"{c.num(worst['mean_ndvi'], 2)} ({worst['date']}).")
    return out


def _h_water_quality(c):
    s = c.s
    return [f"{c.f('frac', s.get('pct_water'))} of valid pixels are water (NDWI > 0).",
            f"Inside the water, mean NDCI is {c.num(s.get('mean_ndci_in_water'), 3)} and mean red reflectance "
            f"{c.num(s.get('mean_red_reflectance_in_water'), 3)} -- relative indicators, not concentrations."]


def _h_flood(c):
    s = c.s
    return [f"{c.f('ha', s.get('flooded_area_ha'))} ({c.f('frac', s.get('flooded_fraction'))} of the valid area) is water on "
            f"{c.m.get('date_after')} that was not water on {c.m.get('date_before')}.",
            f"A further {c.f('ha', s.get('permanent_water_area_ha'))} is permanent water (water on both dates)."]


def _h_coastal(c):
    s = c.s
    net = s.get("net_water_change_ha")
    direction = "net gain of water" if (net or 0) > 0 else "net loss of water" if (net or 0) < 0 else "no net change"
    return [f"{c.f('ha', s.get('land_to_water_ha'))} changed from land to water and {c.f('ha', s.get('water_to_land_ha'))} "
            f"from water to land between {c.m.get('date_before')} and {c.m.get('date_after')}.",
            f"Net change: {c.f('ha', abs(net) if net is not None else None)} ({direction})."]


def _h_wetland(c):
    s = c.s
    return [f"Wet (inundated) vegetation covers {c.f('ha', s.get('wet_vegetation_area_ha'))} "
            f"({c.f('frac', s.get('pct_wet_vegetation'))}) and open water {c.f('ha', s.get('open_water_area_ha'))} "
            f"({c.f('frac', s.get('pct_open_water'))}) on {c.m.get('date')}."]


def _h_fire(c):
    s = c.s
    return [f"{c.f('ha', s.get('burned_area_ha'))} burned between {c.m.get('date_before')} and {c.m.get('date_after')} "
            f"(dNBR >= 0.10 on {c.f('frac', None if s.get('pct_unburned') is None else 1 - s['pct_unburned'])} of the valid area).",
            f"Severity split: {c.f('frac', s.get('pct_low_severity'))} low, {c.f('frac', s.get('pct_moderate_severity'))} moderate, "
            f"{c.f('frac', s.get('pct_high_severity'))} high. Mean dNBR {c.num(s.get('mean_dnbr'), 3)}."]


def _h_snow(c):
    s = c.s
    return [f"Snow or ice covers {c.f('ha', s.get('snow_area_ha'))} ({c.f('frac', s.get('snow_cover_fraction'))} of the valid area) on {c.m.get('date')}."]


def _h_glacier(c):
    s = c.s
    chg = s.get("area_change_ha")
    word = "grew" if (chg or 0) > 0 else "shrank" if (chg or 0) < 0 else "did not change"
    pct = s.get("area_change_pct")
    tail = f" ({pct:+.1f} %)" if pct is not None else ""
    return [f"Snow/ice extent on commonly cloud-free pixels {word}: {c.f('ha', s.get('area_ha_before'))} on {c.m.get('date_before')} "
            f"to {c.f('ha', s.get('area_ha_after'))} on {c.m.get('date_after')}{tail}."]


def _h_thermal(label):
    def f(c):
        s = c.s
        key = next((k for k in ("pct_heat_island", "pct_above_threshold", "pct_thermal_anomaly") if k in s), None)
        return [f"Land surface temperature on {c.m.get('date')} averages {c.f('c', s.get('mean_temp_celsius'))} "
                f"(range {c.f('c', s.get('min_temp_celsius'))} to {c.f('c', s.get('max_temp_celsius'))}).",
                f"{c.f('frac', s.get(key) if key else None)} of valid pixels are more than {c.num(s.get('threshold_c'), 1)} °C above the "
                f"scene mean ({label})."]
    return f


def _h_biodiversity(c):
    s = c.s
    return [f"The area contains {c.f('int', s.get('n_habitat_classes'))} land-cover classes; regional Shannon index "
            f"{c.num(s.get('region_shannon_index'), 2)}, evenness {c.num(s.get('region_evenness'), 2)}.",
            f"Hotspots (local diversity >= {c.num(s.get('hotspot_threshold'), 2)}) cover {c.f('frac', s.get('hotspot_fraction'))} of the area."]


def _h_solar_potential(c):
    s = c.s
    return [f"At {s.get('hour')} h on {s.get('date')} the sun is {c.num(s.get('sun_zenith_deg'), 1)}° from the zenith "
            f"(azimuth {c.num(s.get('sun_azimuth_deg'), 0)}°). Mean relative irradiance on the terrain is "
            f"{c.num(s.get('mean_relative_irradiance'), 2)} (1 = sun directly overhead of a surface facing it).",
            f"{c.f('frac', s.get('pct_self_shaded'))} of cells face away from the sun at that moment."]


def _h_archaeology(c):
    s = c.s
    return [f"The Local Relief Model has a standard deviation of {c.num(s.get('lrm_std_m'), 2)} m "
            f"(range {c.num(s.get('lrm_min_m'), 1)} to {c.num(s.get('lrm_max_m'), 1)} m).",
            "The layers are for expert interpretation; the pipeline does not decide that a site is present."]


def _h_wind(c):
    s = c.s
    return [f"{c.f('ha', s.get('suitable_area_ha'))} ({c.f('frac', s.get('pct_area_suitable_overall'))} of the area) passes the screen "
            f"(slope <= {c.num(s.get('max_slope_deg'), 0)}°"
            f"{', land-use exclusions applied' if s.get('landcover_exclusion_applied') else ', terrain only'}).",
            "This is a siting screen, not a wind-resource or energy-yield estimate."]


def _h_land_cover(c):
    rows = c.class_rows("land_cover")
    if not rows:
        return ["No land-cover classes were found in the output."]
    top = sorted(rows, key=lambda r: -(r["area_ha"] or 0))[:3]
    txt = ", ".join(f"{r['name']} ({c.f('frac', r['share'])})" for r in top)
    return [f"The area has {len(rows)} land-cover classes; the largest are {txt}."]


def _h_water_bodies(c):
    s = c.s
    return [f"Surface water covers {c.f('frac', s.get('water_coverage'))} of the valid area ({c.f('ha', c.ha('water', 1))}) on {c.m.get('date')}."]


def _h_change(c):
    s = c.s
    if "changed_area_ha" in s:
        return [f"{c.f('frac', s.get('change_fraction'))} of the valid area ({c.f('ha', s.get('changed_area_ha'))}) is spectrally changed "
                f"between {c.m.get('date_before')} and {c.m.get('date_after')} (magnitude > {c.num(s.get('threshold'), 3)})."]
    return [f"{c.f('frac', s.get('change_fraction'))} of pixels are labelled as changed by the network."]


def _h_network_mask(noun, key):
    def f(c):
        s = c.s
        return [f"The network labels {c.f('frac', s.get(key))} of the area as {noun} ({c.f('ha', c.ha('mask', 1))}) on {c.m.get('date')}."]
    return f


def _h_crop_monitoring(c):
    rows = [r for r in c.class_rows("crops") if r["value"] != 0]
    if not rows:
        return ["The network labelled every pixel as background."]
    top = sorted(rows, key=lambda r: -(r["area_ha"] or 0))[:3]
    return ["Largest mapped classes: " + ", ".join(f"{r['name']} ({c.f('ha', r['area_ha'])})" for r in top) + "."]


def _h_network_change(c):
    s = c.s
    changed = sum((r["area_ha"] or 0) for r in c.class_rows("change") if r["value"] != 0)
    return [f"The network labels {c.f('frac', s.get('changed_fraction'))} of the area as changed ({c.f('ha', changed)}) "
            f"between {c.m.get('date_before')} and {c.m.get('date_after')}."]


# ---------------------------------------------------------------------------
# Class legends
# ---------------------------------------------------------------------------

def _cd(*items):
    return tuple(ClassDef(v, n, col) for v, n, col in items)


_WATER_BLUE = "#1f78b4"
_FIRE = ("#d9dee3", "#fed976", "#fd8d3c", "#bd0026")

# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------

_SPEC_LIST: list[ReportSpec] = [
    ReportSpec(
        "crop_health", "Crop Health Anomaly (NDVI drop)",
        "Where did vegetation greenness fall between two dates?",
        ("NDVI = (NIR - red) / (NIR + red) is computed for both dates from surface reflectance.",
         "A pixel is flagged when NDVI(before) - NDVI(after) exceeds the anomaly threshold (default 0.15).",
         "Pixels that are cloud- or shadow-masked on either date are nodata; every percentage and area is over valid pixels only."),
        (Layer("anomaly", "crop_health_anomaly.tif", "NDVI drop anomaly", "classes",
               classes=_cd((0, "Not flagged", NO_CHANGE), (1, "NDVI dropped", "#d7301f")), muted=(0,),
               vector=VectorSpec("ndvi_drop", keep=(1,)), primary=True),
         Layer("ndvi_change", "ndvi_change.tif", "NDVI change (after - before)", "continuous",
               cmap="RdYlGn", vmin=-0.6, vmax=0.6, center=0.0, label="NDVI change")),
        (Metric("pct_flagged", "Share of valid area flagged", "frac"), Metric("anomaly_threshold", "NDVI-drop threshold", "num2"),
         Metric("mean_ndvi_before", "Mean NDVI, before", "num3"), Metric("mean_ndvi_after", "Mean NDVI, after", "num3"),
         Metric("valid_fraction", "Valid (cloud-free on both dates) share of AOI", "frac")),
        _h_crop_health,
        reading=("The flag marks any vegetation that browned or was cleared. Harvest, mowing, phenology and clearing look the same.",),
        limitations=("It is not restricted to crops: mask to your field boundaries before reading it as crop stress.",
                     "Choose two dates in the same phenological stage; seasonal change otherwise dominates the signal."),
        references=(R_EVI, R_S2)),

    ReportSpec(
        "vegetation_indices", "Vegetation Indices Time Series (NDVI, EVI, NDWI)",
        "How do vegetation greenness and moisture indices change across the requested dates?",
        ("For every requested date the best cloud-free scene(s) are mosaicked and three indices are written to <date>/indices.tif.",
         "NDVI = (NIR - red)/(NIR + red). EVI = 2.5 (NIR - red)/(NIR + 6 red - 7.5 blue + 1). NDWI = (green - NIR)/(green + NIR).",
         "Each date's mean is taken over its valid (cloud-free) pixels."),
        (),   # layers are built from the dates at run time
        (Metric("n_dates_used", "Dates with usable imagery", "int"),),
        _h_veg_indices,
        reading=("Compare means across dates only when the valid share is similar; a date with heavy cloud is less representative.",),
        limitations=("Scene means hide spatial pattern; use the per-date rasters and the vegetation-class polygons for that.",
                     "Vegetation classes in the polygons are NDVI bands (<0.1, 0.1-0.3, 0.3-0.6, >=0.6), a generic scheme, not a crop or ecosystem classification."),
        references=(R_EVI, R_S2), chart="timeseries", multi_date=True),

    ReportSpec(
        "water_quality", "Water Quality Indicators (NDCI and red reflectance)",
        "Where is the water, and how does a relative chlorophyll indicator vary inside it?",
        ("Water is NDWI = (green - NIR)/(green + NIR) > 0.",
         "Inside the water, NDCI = (red-edge 1 - red)/(red-edge 1 + red) is a relative chlorophyll-a indicator; red reflectance is a relative turbidity indicator.",
         "Needs Sentinel-2 (red-edge band). Output bands: 1 = NDCI, 2 = red reflectance."),
        (Layer("ndci", "water_quality.tif", "NDCI inside water (relative chlorophyll)", "continuous",
               cmap="YlGnBu", label="NDCI", primary=True,
               binning=Binning("quantile", n=3, names=("Lower third of NDCI", "Middle third of NDCI", "Upper third of NDCI"),
                               cmap="YlGnBu"),
               vector=VectorSpec("ndci_zones")),
         Layer("turbidity", "water_quality.tif", "Red reflectance inside water (relative turbidity)", "continuous",
               band=2, cmap="YlOrBr", label="red reflectance")),
        (Metric("pct_water", "Water share of valid area", "frac"), Metric("mean_ndci_in_water", "Mean NDCI in water", "num3"),
         Metric("mean_red_reflectance_in_water", "Mean red reflectance in water", "num3"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_water_quality,
        reading=("Zones are terciles of NDCI *within this water body*: they show where the indicator is relatively higher, not a quality grade.",),
        limitations=("NDCI and red reflectance are relative. Converting to chlorophyll-a (ug/l) or turbidity (NTU) needs local water samples.",
                     "Shallow water, bottom reflectance and sun glint also change these indices."),
        references=(R_NDCI, R_NDWI, R_S2)),

    ReportSpec(
        "flood_mapping", "Flood Extent Mapping (optical, MNDWI)",
        "Which areas are newly flooded compared with a dry reference date?",
        ("Water = MNDWI = (green - SWIR1)/(green + SWIR1) above the threshold (default 0) on each date.",
         "Flooded = water on the flood date that was not water on the dry reference; permanent water = water on both dates.",
         "Cloud-masked pixels on either date are nodata, never 'dry'."),
        (Layer("flood", "flood_extent.tif", "Flood extent", "classes",
               classes=_cd((0, "Dry", NO_CHANGE), (1, "Flooded", "#e6550d"), (2, "Permanent water", _WATER_BLUE)), muted=(0,),
               vector=VectorSpec("flood_extent", keep=(1, 2)), primary=True),),
        (Metric("flooded_area_ha", "Flooded area", "ha"), Metric("permanent_water_area_ha", "Permanent water", "ha"),
         Metric("flooded_fraction", "Flooded share of valid area", "frac"), Metric("mndwi_threshold", "MNDWI threshold", "num2"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_flood,
        reading=("Pixels hidden by cloud on either date cannot be assessed; a flood under cloud is not shown.",),
        limitations=("Optical imagery cannot see through cloud; the largest floods often coincide with the cloudiest days. Radar (Sentinel-1) is the usual complement.",
                     "Shadow, wet soil and dark surfaces can resemble water at MNDWI 0; check urban and mountain areas."),
        references=(R_MNDWI, R_S2)),

    ReportSpec(
        "coastal_monitoring", "Coastal Change (shoreline retreat and advance)",
        "Where did land turn to water, or water to land, between two dates?",
        ("Land/water masks come from MNDWI = (green - SWIR1)/(green + SWIR1) > threshold (default 0) on each date.",
         "Classes: land on both dates, water on both, land to water (retreat or inundation), water to land (advance).",
         "Cloud-masked pixels on either date are nodata."),
        (Layer("shore", "shoreline_change.tif", "Shoreline change", "classes",
               classes=_cd((0, "Land (both dates)", "#d2b48c"), (1, "Water (both dates)", "#4a90c2"),
                           (2, "Land to water", "#e31a1c"), (3, "Water to land", "#33a02c")), muted=(0, 1),
               vector=VectorSpec("shoreline_change", keep=(2, 3)), primary=True),),
        (Metric("land_to_water_ha", "Land to water", "ha"), Metric("water_to_land_ha", "Water to land", "ha"),
         Metric("net_water_change_ha", "Net water change", "ha"), Metric("mndwi_threshold", "MNDWI threshold", "num2"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_coastal,
        limitations=("A difference in tidal state between the two images appears as shoreline change. Use dates at the same tide.",
                     "Pixel size (10-30 m) limits the smallest change that can be mapped."),
        references=(R_MNDWI, R_S2)),

    ReportSpec(
        "wetland_mapping", "Wetland Mapping (open water vs inundated vegetation)",
        "Where is there open water, and where is vegetation standing in water?",
        ("Water = MNDWI > 0. Vegetation = EVI > 0.1.",
         "Water without vegetation = open water; water with vegetation = wet (inundated) vegetation."),
        (Layer("wetland", "wetland_classification.tif", "Wetland classification", "classes",
               classes=_cd((0, "Other", NO_CHANGE), (1, "Open water", _WATER_BLUE), (2, "Wet vegetation", "#1b9e77")), muted=(0,),
               vector=VectorSpec("wetland_classes", keep=(1, 2)), primary=True),),
        (Metric("wet_vegetation_area_ha", "Wet vegetation area", "ha"), Metric("open_water_area_ha", "Open water area", "ha"),
         Metric("pct_wet_vegetation", "Wet vegetation share", "frac"), Metric("pct_open_water", "Open water share", "frac"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_wetland,
        limitations=("It finds wetlands that are wet on the acquisition date. Dry-season wetlands and saturated soils are not detected.",),
        references=(R_MNDWI, R_EVI, R_S2)),

    ReportSpec(
        "wildfire_severity", "Wildfire Burn Severity (dNBR)",
        "How much burned, and how severely?",
        ("NBR = (NIR - SWIR2)/(NIR + SWIR2). dNBR = NBR(before) - NBR(after).",
         "Classes (simplified USGS/MTBS thresholds): unburned < 0.10; low 0.10-0.27; moderate 0.27-0.66; high >= 0.66.",
         "Cloud-masked pixels are nodata, not 'unburned'."),
        (Layer("severity", "severity_class.tif", "Burn severity", "classes",
               classes=_cd((0, "Unburned", "#d9dee3"), (1, "Low", "#fed976"), (2, "Moderate", "#fd8d3c"), (3, "High", "#bd0026")),
               muted=(0,), vector=VectorSpec("burn_severity", keep=(1, 2, 3)), primary=True),
         Layer("dnbr", "dnbr.tif", "dNBR", "continuous", cmap="RdYlGn_r", vmin=-0.5, vmax=1.3, label="dNBR")),
        (Metric("burned_area_ha", "Burned area (low + moderate + high)", "ha"), Metric("mean_dnbr", "Mean dNBR", "num3"),
         Metric("pct_unburned", "Unburned share", "frac"), Metric("pct_low_severity", "Low severity share", "frac"),
         Metric("pct_moderate_severity", "Moderate severity share", "frac"), Metric("pct_high_severity", "High severity share", "frac"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_fire,
        reading=("Choose pre- and post-fire dates close to the fire and in the same season; otherwise vegetation phenology adds to dNBR.",),
        limitations=("Thresholds are generic; field-calibrated composite burn index values give local accuracy.",
                     "Smoke, cloud and burn scars from earlier fires can bias the result."),
        references=(R_NBR, R_S2)),

    ReportSpec(
        "snow_cover", "Snow and Ice Cover (NDSI)",
        "Where is there snow or ice on the chosen date?",
        ("NDSI = (green - SWIR1)/(green + SWIR1); snow/ice where NDSI > 0.4 (the MODIS MOD10 threshold).",
         "Cloud-masked pixels are nodata and excluded from percentages."),
        (Layer("snow", "snow_extent.tif", "Snow / ice extent", "classes",
               classes=_cd((0, "No snow/ice", "#c9b99b"), (1, "Snow / ice", "#7fb3ff")), muted=(0,),
               vector=VectorSpec("snow_extent", keep=(1,)), primary=True),),
        (Metric("snow_area_ha", "Snow / ice area", "ha"), Metric("snow_cover_fraction", "Snow / ice share of valid area", "frac"),
         Metric("ndsi_threshold", "NDSI threshold", "num2"), Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_snow,
        limitations=("Extent only: snow water equivalent cannot be derived from optical imagery.",
                     "Cloud and snow can look alike in some bands; the cloud mask removes most, not all, confusion."),
        references=(R_NDSI, R_S2)),

    ReportSpec(
        "glacier_monitoring", "Glacier / Snow-Ice Extent Change (NDSI)",
        "Did snow and ice extent grow or shrink between two dates?",
        ("Snow/ice = NDSI > 0.4 on each date, evaluated only where both dates are cloud-free so that cloud cannot look like retreat.",
         "The map shows the extent on the later date; areas for both dates are in the statistics."),
        (Layer("ice", "glacier_extent_after.tif", "Snow / ice extent (later date)", "classes",
               classes=_cd((0, "No snow/ice", "#c9b99b"), (1, "Snow / ice", "#7fb3ff")), muted=(0,),
               vector=VectorSpec("glacier_extent_after", keep=(1,)), primary=True),),
        (Metric("area_ha_before", "Snow / ice area, before", "ha"), Metric("area_ha_after", "Snow / ice area, after", "ha"),
         Metric("area_change_ha", "Change", "ha"), Metric("area_change_pct", "Change (%)", "num1"),
         Metric("ndsi_threshold", "NDSI threshold", "num2"), Metric("valid_fraction", "Commonly valid share of AOI", "frac")),
        _h_glacier,
        reading=("Pick two dates at the end of the melt season so that seasonal snow does not hide the glacier signal.",),
        limitations=("NDSI maps seasonal snow and misses debris-covered ice, so it is not a glacier outline.",),
        references=(R_NDSI, R_S2), chart="before_after"),

    ReportSpec(
        "urban_heat_island", "Urban Heat Island (Landsat land surface temperature)",
        "How hot is the surface, and which areas are hotter than the scene average?",
        ("Land surface temperature (K) = DN x 0.00341802 + 149.0 from the Landsat Collection 2 Level-2 surface temperature band, shown in deg C.",
         "'Hot' pixels are more than the threshold (default 2 deg C) above the scene mean -- a relative measure, not urban-minus-rural."),
        (Layer("lst", "lst_celsius.tif", "Land surface temperature", "continuous", cmap="inferno", label="deg C",
               binning=Binning("equal", n=5, unit="°C", cmap="inferno"), vector=VectorSpec("temperature_zones"), primary=True),
         Layer("hot", "heat_island_mask.tif", "Hot areas (scene mean + threshold)", "classes",
               classes=_cd((0, "Not hot", NO_CHANGE), (1, "Hot", "#b30000")), muted=(0,),
               vector=VectorSpec("heat_island", keep=(1,)), optional=True)),
        (Metric("mean_temp_celsius", "Mean temperature", "c"), Metric("min_temp_celsius", "Minimum", "c"),
         Metric("max_temp_celsius", "Maximum", "c"), Metric("pct_heat_island", "Hot share of valid area", "frac"),
         Metric("threshold_c", "Threshold above scene mean (deg C)", "num1"), Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_thermal("urban heat island candidates"),
        limitations=("Needs Landsat (Sentinel-2 has no thermal sensor); the thermal band is ~100 m resampled to 30 m.",
                     "Surface temperature is not air temperature, and one overpass (about 10:30 local time) is not the daily maximum."),
        references=(R_LST,)),

    ReportSpec(
        "land_surface_temperature", "Land Surface Temperature (Landsat)",
        "What is the land surface temperature across the area?",
        ("Land surface temperature (K) = DN x 0.00341802 + 149.0 from the Landsat Collection 2 Level-2 surface temperature band, shown in deg C.",
         "'Above threshold' pixels are more than the threshold above the scene mean."),
        (Layer("lst", "lst_celsius.tif", "Land surface temperature", "continuous", cmap="inferno", label="deg C",
               binning=Binning("equal", n=5, unit="°C", cmap="inferno"), vector=VectorSpec("temperature_zones"), primary=True),
         Layer("hot", "above_threshold_mask.tif", "Above scene mean + threshold", "classes",
               classes=_cd((0, "Below threshold", NO_CHANGE), (1, "Above threshold", "#b30000")), muted=(0,),
               vector=VectorSpec("above_threshold", keep=(1,)), optional=True)),
        (Metric("mean_temp_celsius", "Mean temperature", "c"), Metric("min_temp_celsius", "Minimum", "c"),
         Metric("max_temp_celsius", "Maximum", "c"), Metric("pct_above_threshold", "Above-threshold share of valid area", "frac"),
         Metric("threshold_c", "Threshold above scene mean (deg C)", "num1"), Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_thermal("above the threshold"),
        limitations=("Needs Landsat. Surface temperature is not air temperature; one overpass is not the daily cycle.",),
        references=(R_LST,)),

    ReportSpec(
        "volcano_monitoring", "Thermal Anomaly Screening (volcano monitoring)",
        "Which pixels are much hotter than the surrounding scene?",
        ("Land surface temperature from Landsat Collection 2 Level-2 (K = DN x 0.00341802 + 149.0).",
         "An anomaly is a pixel more than the threshold (default 20 deg C) above the scene mean."),
        (Layer("lst", "lst_celsius.tif", "Land surface temperature", "continuous", cmap="inferno", label="deg C",
               binning=Binning("equal", n=5, unit="°C", cmap="inferno"), vector=VectorSpec("temperature_zones"), primary=True),
         Layer("hot", "thermal_anomaly_mask.tif", "Thermal anomaly candidates", "classes",
               classes=_cd((0, "Normal", NO_CHANGE), (1, "Thermal anomaly", "#b30000")), muted=(0,),
               vector=VectorSpec("thermal_anomaly", keep=(1,)), optional=True)),
        (Metric("mean_temp_celsius", "Mean temperature", "c"), Metric("max_temp_celsius", "Maximum", "c"),
         Metric("pct_thermal_anomaly", "Anomalous share of valid area", "frac"),
         Metric("threshold_c", "Threshold above scene mean (deg C)", "num1"), Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_thermal("thermal anomaly candidates"),
        limitations=("An anomaly is a candidate only: lava, fumaroles, wildfire and industrial heat look the same.",
                     "Landsat's thermal band saturates over active lava; use it for persistent warm ground, not eruption detection."),
        references=(R_LST,)),

    ReportSpec(
        "biodiversity_hotspot", "Habitat Diversity Hotspots (Shannon index)",
        "Where is habitat most heterogeneous?",
        ("Land cover comes from the ESA WorldCover 2021 map. The Shannon index H' = -sum(p ln p) of class proportions is computed in a moving window (default 21 x 21 pixels = 210 m).",
         "Hotspots are the top decile (default) of local H'. Habitat heterogeneity is an indirect proxy for biodiversity potential."),
        (Layer("shannon", "shannon_diversity.tif", "Local Shannon diversity (H')", "continuous", cmap="viridis", label="H' (ln units)", primary=True),
         Layer("hotspot", "hotspot_mask.tif", "Diversity hotspots", "classes",
               classes=_cd((0, "Other", NO_CHANGE), (1, "Hotspot", "#d95f02")), muted=(0,),
               vector=VectorSpec("diversity_hotspots", keep=(1,)), optional=True),
         Layer("land_cover", "land_cover/land_cover.tif", "ESA WorldCover 2021", "classes", classes=WORLDCOVER,
               vector=VectorSpec("land_cover"), optional=True)),
        (Metric("n_habitat_classes", "Habitat classes present", "int"), Metric("region_shannon_index", "Regional Shannon index", "num3"),
         Metric("region_evenness", "Regional evenness", "num3"), Metric("mean_local_shannon", "Mean local H'", "num3"),
         Metric("max_local_shannon", "Maximum local H'", "num3"), Metric("hotspot_threshold", "Hotspot threshold (H')", "num3"),
         Metric("hotspot_fraction", "Hotspot share", "frac"), Metric("window_px", "Window (pixels)", "int")),
        _h_biodiversity,
        limitations=("Heterogeneity is not a species count. WorldCover is the static 2021 product, so this describes 2021 whatever date is passed.",),
        references=(R_WC, R_SHANNON)),

    ReportSpec(
        "solar_potential", "Solar Potential (terrain illumination)",
        "How much does terrain slope and aspect help or hinder sunlight at a given date and hour?",
        ("Slope and aspect come from the DEM (Horn's method). The sun position follows the standard declination and hour-angle equations.",
         "The output is the cosine of the sun's incidence angle on each cell (0-1), for one instant."),
        (Layer("solar", "solar_potential.tif", "Relative clear-sky irradiance (0-1)", "continuous", cmap="YlOrRd", vmin=0.0, vmax=1.0,
               label="relative irradiance", primary=True,
               binning=Binning("fixed", edges=(0.25, 0.5, 0.75),
                               names=("Very low (< 0.25)", "Low (0.25-0.5)", "Moderate (0.5-0.75)", "High (>= 0.75)"),
                               colors=("#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c")),
               vector=VectorSpec("irradiance_classes")),),
        (Metric("mean_relative_irradiance", "Mean relative irradiance", "num3"), Metric("pct_self_shaded", "Cells facing away from the sun", "frac"),
         Metric("sun_zenith_deg", "Sun zenith angle", "deg"), Metric("sun_azimuth_deg", "Sun azimuth", "deg"),
         Metric("date", "Date"), Metric("hour", "Hour (solar time)", "num1"), Metric("dem_type", "DEM product")),
        _h_solar_potential,
        limitations=("A single-instant relative factor: no terrain shadowing by neighbouring ridges, no atmosphere, no clouds, and not kWh/m2.",
                     "Annual yield needs a time-integrated model with measured or modelled irradiance."),
        references=(R_SUN, R_HORN)),

    ReportSpec(
        "archaeological_site", "Archaeological Relief Visualisation",
        "Which landscape features stand out in the terrain for an expert to inspect?",
        ("A Local Relief Model (DEM minus a smoothed regional trend) highlights small rises and hollows (Hesse 2010).",
         "A multi-directional hillshade averages illumination from eight azimuths (Devereux et al. 2008)."),
        (Layer("lrm", "local_relief_model.tif", "Local Relief Model (m)", "continuous", cmap="RdBu_r", center=0.0, label="relief (m)",
               primary=True,
               binning=Binning("std", k=1.0, names=("Hollow (< -1 sd)", "Level (within 1 sd)", "Rise (> +1 sd)"),
                               colors=("#2166ac", "#e0e0e0", "#b2182b"), keep=(0, 2)),
               vector=VectorSpec("relief_anomalies")),
         Layer("hillshade", "multidirectional_hillshade.tif", "Multi-directional hillshade", "continuous", cmap="gray", vmin=0.0, vmax=1.0,
               label="illumination", optional=True)),
        (Metric("lrm_std_m", "LRM standard deviation", "num2"), Metric("lrm_min_m", "LRM minimum", "m"), Metric("lrm_max_m", "LRM maximum", "m"),
         Metric("smoothing_radius_px", "Smoothing radius (pixels)", "int"), Metric("dem_type", "DEM product")),
        _h_archaeology,
        limitations=("There is no 'site found' output. The 'rise' and 'hollow' polygons are simply cells more than 1 standard deviation from the local trend.",
                     "A 30 m DEM shows landscape-scale earthworks only; use LiDAR-derived DTMs for small mounds and walls.",
                     "The LRM is unreliable within the smoothing radius of the border."),
        references=(R_LRM, R_MDH, R_HORN)),

    ReportSpec(
        "wind_farm_siting", "Wind-Farm Siting Screen",
        "Which ground passes basic slope and land-use constraints, and which of it is topographically prominent?",
        ("Cells steeper than the slope limit (default 15 deg) are excluded; with land-cover exclusion on, so are built-up, snow/ice, water, wetland and mangrove cells (WorldCover 2021).",
         "Remaining cells are scored by terrain exposure: elevation minus a regional mean. Higher prominence tends to be windier."),
        (Layer("wind", "wind_suitability.tif", "Terrain exposure on allowed ground (m)", "continuous", cmap="YlGnBu", label="exposure (m)",
               primary=True,
               binning=Binning("quantile", n=3, names=("Lower exposure third", "Middle exposure third", "Upper exposure third"), cmap="YlGnBu"),
               vector=VectorSpec("suitable_ground")),),
        (Metric("suitable_area_ha", "Suitable area", "ha"), Metric("pct_area_suitable_overall", "Suitable share (all constraints)", "frac"),
         Metric("pct_area_suitable_by_slope", "Suitable share (slope only)", "frac"), Metric("max_slope_deg", "Slope limit", "deg"),
         Metric("landcover_exclusion_applied", "Land-cover exclusion applied", "bool"), Metric("dem_type", "DEM product")),
        _h_wind,
        limitations=("It does not estimate wind speed, direction or energy yield; that needs measurements or a wind model.",
                     "Grid access, roads, noise set-backs, ecology and ownership are not considered."),
        references=(R_WC, R_HORN)),

    ReportSpec(
        "land_cover", "Land Cover",
        "What land-cover classes occur in the area, and how much of each?",
        ("source = worldcover: the ESA WorldCover 2021 10 m reference map is clipped to the area. It is a published product, not a new classification, so it describes 2021 whatever date is passed.",
         "source = dynamic_world: Google Dynamic World labels for the requested month. Any other source runs your own trained model."),
        (Layer("land_cover", "land_cover.tif", "Land cover", "classes", classes_hint="landcover",
               vector=VectorSpec("land_cover", skip=(0,)), primary=True),),
        (Metric("source", "Source"), Metric("worldcover_year", "WorldCover year", "int")),
        _h_land_cover,
        limitations=("WorldCover 2020 and 2021 are different algorithm versions and must not be compared for change.",
                     "Class areas are summed from 10 m pixels; accuracy of the product is ~75% overall (see the WorldCover validation report)."),
        references=(R_WC, R_DW)),

    ReportSpec(
        "water_bodies", "Surface Water (NDWI)",
        "Where is open surface water?",
        ("NDWI = (green - NIR)/(green + NIR); water where NDWI > threshold (default 0.3, conservative; McFeeters' original cut is 0).",
         "Cloud- and shadow-masked pixels are nodata and excluded from the water share."),
        (Layer("water", "water_mask.tif", "Water mask", "classes",
               classes=_cd((0, "Not water", "#e4dcc8"), (1, "Water", _WATER_BLUE)), muted=(0,),
               vector=VectorSpec("water_bodies", keep=(1,)), primary=True),
         Layer("ndwi", "ndwi.tif", "NDWI", "continuous", cmap="BrBG", vmin=-1, vmax=1, center=0.0, label="NDWI", optional=True)),
        (Metric("water_coverage", "Water share of valid area", "frac"), Metric("ndwi_threshold", "NDWI threshold", "num2"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_water_bodies,
        limitations=("Shadows, burn scars and dark roofs can resemble water; narrow channels below pixel size are missed.",),
        references=(R_NDWI, R_S2)),

    ReportSpec(
        "change_detection", "Change Detection (change vector analysis)",
        "Where did the surface spectrally change between two dates?",
        ("Change vector analysis: the Euclidean length of the per-band reflectance difference between the two dates.",
         "A pixel is 'changed' when the magnitude exceeds max(Otsu threshold, minimum magnitude). The floor stops Otsu splitting pure sensor noise on an unchanged scene."),
        (Layer("change", "change_mask.tif", "Change mask", "classes",
               classes=_cd((0, "Unchanged", NO_CHANGE), (1, "Changed", "#d7301f")), muted=(0,),
               vector=VectorSpec("changed_areas", keep=(1,)), primary=True),
         Layer("magnitude", "change_magnitude.tif", "Change magnitude", "continuous", cmap="magma", label="reflectance units", optional=True)),
        (Metric("change_fraction", "Changed share of valid area", "frac"), Metric("changed_area_ha", "Changed area", "ha"),
         Metric("threshold", "Threshold used", "num3"), Metric("otsu_threshold", "Otsu threshold", "num3"),
         Metric("min_magnitude", "Minimum magnitude", "num3"), Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_change,
        reading=("'Changed' means spectrally changed. It includes seasonal vegetation, illumination and moisture, so compare the same season.",),
        limitations=("It does not say what changed. Interpret with the imagery, or use a trained change network for classes.",),
        references=(R_OTSU, R_S2)),

    ReportSpec(
        "building_footprints", "Building Footprints (trained network)",
        "Where does your network see buildings?",
        ("Tiled inference with a segmentation network you trained (checkpoint required); class 1 = building.",),
        (Layer("mask", "building_mask.tif", "Building mask", "classes",
               classes=_cd((0, "Background", NO_CHANGE), (1, "Building", "#c2185b")), muted=(0,),
               vector=VectorSpec("buildings", keep=(1,)), primary=True),),
        (Metric("building_coverage", "Building share of area", "frac"),),
        _h_network_mask("building", "building_coverage"),
        limitations=(_NETWORK_LIMIT, "At 10 m Sentinel-2 resolution individual houses are not resolvable; use higher-resolution imagery for footprints."),
        references=()),

    ReportSpec(
        "crop_monitoring", "Crop-Type Map (trained network)",
        "Which crop does your network assign to each pixel?",
        ("Tiled inference with a segmentation network you trained (checkpoint required). Class 0 is background; classes 1..n follow the crop list you trained on.",),
        (Layer("crops", "crop_map.tif", "Crop map", "classes", classes_hint="crops",
               vector=VectorSpec("crop_fields", skip=(0,)), primary=True),),
        (Metric("num_classes", "Number of classes", "int"),),
        _h_crop_monitoring,
        limitations=(_NETWORK_LIMIT, _USER_CLASSES_NOTE),
        references=()),

    ReportSpec(
        "solar_detection", "Solar Panel Detection (trained network)",
        "Where does your network see solar panels?",
        ("Tiled inference with a segmentation network you trained (checkpoint required); class 1 = panel.",),
        (Layer("mask", "solar_mask.tif", "Solar panel mask", "classes",
               classes=_cd((0, "Background", NO_CHANGE), (1, "Solar panel", "#f9a825")), muted=(0,),
               vector=VectorSpec("solar_panels", keep=(1,)), primary=True),),
        (Metric("panel_coverage", "Panel share of area", "frac"),),
        _h_network_mask("solar panel", "panel_coverage"),
        limitations=(_NETWORK_LIMIT, "At 10 m resolution only utility-scale arrays are resolvable."),
        references=()),

    ReportSpec(
        "disaster_assessment", "Disaster Damage Assessment (trained network)",
        "Which areas does your network label as damaged between the pre- and post-event images?",
        ("Two-image (siamese) inference with a network you trained (checkpoint required). Class 0 is no change; higher classes are the damage classes you trained on.",),
        (Layer("change", "damage_assessment.tif", "Damage classes", "classes", classes_hint="user",
               vector=VectorSpec("damage", skip=(0,)), primary=True),),
        (Metric("changed_fraction", "Changed share of area", "frac"),),
        _h_network_change,
        limitations=(_NETWORK_LIMIT, _USER_CLASSES_NOTE),
        references=()),

    ReportSpec(
        "deforestation", "Deforestation / Forest Loss (trained network)",
        "Where does your network see forest change between two dates?",
        ("Two-image inference with a change network you trained (checkpoint required). Class 0 is no change; other classes are those you trained on.",),
        (Layer("change", "deforestation_map.tif", "Forest-change classes", "classes", classes_hint="user",
               vector=VectorSpec("forest_change", skip=(0,)), primary=True),),
        (Metric("changed_fraction", "Changed share of area", "frac"),),
        _h_network_change,
        limitations=(_NETWORK_LIMIT, _USER_CLASSES_NOTE),
        references=(R_S2,)),

    ReportSpec(
        "urban_growth", "Urban Growth (trained network)",
        "Where does your network see urban change between two Landsat dates?",
        ("Two-image inference with a change network you trained (checkpoint required). Class 0 is no change; other classes are those you trained on.",),
        (Layer("change", "urban_growth.tif", "Urban-change classes", "classes", classes_hint="user",
               vector=VectorSpec("urban_growth", skip=(0,)), primary=True),),
        (Metric("changed_fraction", "Changed share of area", "frac"),),
        _h_network_change,
        limitations=(_NETWORK_LIMIT, _USER_CLASSES_NOTE),
        references=()),
]

SPECS: dict[str, ReportSpec] = {s.pipeline: s for s in _SPEC_LIST}


def generic_spec(pipeline: str, output_file: str | None) -> ReportSpec:
    """A spec for a pipeline that has none (for example a plug-in pipeline)."""
    layers: tuple[Layer, ...] = ()
    if output_file:
        layers = (Layer("output", output_file, "Pipeline output", "classes", classes_hint="user", primary=True,
                        vector=VectorSpec("output", skip=()), optional=True),)
    return ReportSpec(
        pipeline, pipeline.replace("_", " ").title(), "Results of this pipeline run.", ("See the pipeline documentation.",),
        layers, (), lambda c: [], limitations=("This pipeline has no dedicated report template; statistics are listed as produced.",))


def spec_for(pipeline: str, output_file: str | None = None) -> ReportSpec:
    return SPECS.get(pipeline) or generic_spec(pipeline, output_file)


def landcover_classes(source: str | None) -> tuple[ClassDef, ...]:
    return DYNAMIC_WORLD if source == "dynamic_world" else WORLDCOVER


def user_class(value: int, names: Sequence[str] | dict[int, str] | None, base0: str) -> ClassDef:
    """Name/colour for a class of a user-trained network."""
    if isinstance(names, dict):
        name = names.get(value)
    elif names and value < len(names):
        name = names[value]
    else:
        name = None
    if name is None:
        name = base0 if value == 0 else f"Class {value}"
    return ClassDef(value, str(name), NO_CHANGE if value == 0 else categorical(value - 1))