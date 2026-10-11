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
class TableSpec:
    """A table built from one entry of ``result.stats``.

    The entry is either a list of dicts (``columns`` = (key, header, format code) for each column) or a plain
    ``{name: number}`` dict (two columns: ``key_header``, ``value_header`` formatted with ``value_fmt``).
    """
    stat: str
    title: str
    columns: tuple[tuple[str, str, str], ...] = ()
    key_header: str = "Item"
    value_header: str = "Value"
    value_fmt: str = "num2"
    max_rows: int = 40
    sheet: str = ""


@dataclass(frozen=True)
class ChartSpec:
    """A bar / line chart of one entry of ``result.stats`` (a list of dicts, or a ``{label: number}`` dict).

    ``x`` is the dict key (or keys, joined with " -> ") holding each row's label and ``y`` the key holding its value;
    for a plain dict both are ignored.
    """
    stat: str
    title: str
    ylabel: str
    y: str = ""
    x: str | tuple[str, ...] = "date"
    kind: str = "bar"                        # "bar" | "barh" | "line"
    fmt: str = "{:,.1f}"


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
    tables: tuple[TableSpec, ...] = ()       # extra tables from list / dict statistics
    charts: tuple[ChartSpec, ...] = ()       # when given, these replace the automatic class-area chart
    extra_layers: Callable[[dict[str, Any], dict[str, Any]], Sequence[Layer]] | None = None   # layers that depend on the run


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
R_TWELE = "Twele, A., Cao, W., Plank, S. & Martinis, S. (2016). Sentinel-1-based flood mapping: a fully automated processing chain. International Journal of Remote Sensing 37(13), 2990-3004."
R_S1RTC = "Catalyst / ESA. Sentinel-1 Radiometrically Terrain Corrected (RTC) backscatter, Microsoft Planetary Computer (sentinel-1-rtc); contains modified Copernicus Sentinel data."
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


def _h_sar_flood(c):
    s = c.s
    return [f"{c.f('ha', s.get('flooded_area_ha'))} ({c.f('frac', s.get('flooded_fraction'))} of the valid area) is open water in the "
            f"flood window ({c.m.get('date_after')}) that was not water in the reference window ({c.m.get('date_before')}); "
            f"{c.f('ha', s.get('permanent_water_area_ha'))} is water in both.",
            f"Sentinel-1 {str(s.get('polarization', '')).upper()}, {s.get('orbit_state')} orbit {s.get('relative_orbit')}: "
            f"{s.get('scenes_reference')} reference and {s.get('scenes_flood')} flood scene(s); water below "
            f"{c.f('num1', s.get('water_db'))} dB ({s.get('water_db_source')}), at least {c.f('num1', s.get('change_db'))} dB darker than the reference."]


def _h_multisensor(c):
    s = c.s
    flooded, land = s.get("flooded_area_ha") or 0.0, s.get("land_area_ha") or 0.0
    used = ", ".join(s.get("sensors_used") or []) or "none"
    if land and flooded / land < 0.0005:
        first = (f"No significant open-water flooding was detected between {c.m.get('date_before')} and {c.m.get('date_after')}: "
                 f"{c.f('ha', flooded)} of {c.f('ha', land)} of land. Floodwater between buildings or under vegetation, "
                 f"which radar and optical imagery both miss, cannot be ruled out.")
    else:
        first = (f"{c.f('ha', flooded)} ({c.f('frac', s.get('flooded_fraction'))} of the land) is flooded between "
                 f"{c.m.get('date_before')} and {c.m.get('date_after')}: {c.f('ha', s.get('flooded_confirmed_ha'))} confirmed by "
                 f"radar and optical, {c.f('ha', s.get('flooded_radar_only_ha'))} radar only, "
                 f"{c.f('ha', s.get('flooded_optical_only_ha'))} optical only. A further {c.f('ha', s.get('flooded_disputed_ha'))} "
                 f"is disputed (radar flooded, a clear optical view says dry).")
    lines = [first,
             f"{c.f('ha', s.get('permanent_water_area_ha'))} was already water in the reference window (sea, lakes, rivers) and is "
             f"excluded from the flooded area and from the percentages.",
             f"Data used: {used}. Optical imagery gave a clear view of {c.f('frac', s.get('optical_coverage'))} of the land."]
    if s.get("radar_optical_f1_reference") is not None:
        lines.append(f"Radar and optical agree on where the water was before the event (F1 {c.num(s.get('radar_optical_f1_reference'), 2)}); "
                     f"radar water threshold {c.f('num1', s.get('water_db'))} dB ({s.get('water_db_source')}).")
    if s.get("hand_applied") and (s.get("excluded_by_hand_ha") or 0) > 0:
        lines.append(f"{c.f('ha', s.get('excluded_by_hand_ha'))} of apparent flooding was removed because it lies more than "
                     f"{c.f('num1', s.get('max_hand_m'))} m above the nearest drainage.")
    if s.get("data_notes"):
        lines.append(f"Data gaps: {s.get('data_notes')}.")
    return lines


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
# Headlines for the reference-product, terrain, vector and time-series pipelines
# ---------------------------------------------------------------------------

def _h_buildings(c):
    s = c.s
    if "building_count" not in s:
        return _h_network_mask("building", "building_coverage")(c)
    src = {"microsoft": "Microsoft ML Building Footprints", "google": "Google Open Buildings v3",
           "osm": "OpenStreetMap"}.get(s.get("source"), str(s.get("source")))
    return [f"{c.f('int', s.get('building_count'))} buildings from {src} fall inside the area, covering {c.f('ha', s.get('building_area_ha'))} "
            f"({c.f('frac', s.get('building_coverage'))} of the area; {c.num(s.get('buildings_per_km2'), 0)} per km²).",
            "These are the dataset's polygons, not detections on your own imagery."]


def _h_roads(c):
    s = c.s
    groups = s.get("km_by_group") or {}
    top = max(groups.items(), key=lambda kv: kv[1])[0] if groups else None
    out = [f"OpenStreetMap holds {c.num(s.get('total_length_km'), 1)} km of roads and paths in the area "
           f"({c.num(s.get('road_density_km_per_km2'), 2)} km per km²)."]
    if top:
        out.append(f"The longest group is {top} ({c.num(groups[top], 1)} km).")
    return out


def _h_lc_change(c):
    s = c.s
    out = [f"{c.f('ha', s.get('changed_area_ha'))} ({c.f('frac', s.get('change_fraction'))} of the valid area) is mapped as a different "
           f"land-cover class in {s.get('year_after')} than in {s.get('year_before')}."]
    t = s.get("transitions") or []
    if t:
        out.append(f"The largest change is {t[0]['from']} to {t[0]['to']}: {c.f('ha', t[0]['area_ha'])}.")
    return out


def _h_urban_expansion(c):
    s = c.s
    out = [f"Built-up area went from {c.f('ha', s.get('built_area_ha_before'))} in {s.get('year_before')} to "
           f"{c.f('ha', s.get('built_area_ha_after'))} in {s.get('year_after')}: gross gain {c.f('ha', s.get('gain_ha'))}, "
           f"gross loss {c.f('ha', s.get('loss_ha'))}, net {c.f('ha', s.get('net_change_ha'))}."]
    if s.get("annual_growth_rate") is not None:
        out.append(f"Compound annual growth of the built-up area: {c.f('frac', s.get('annual_growth_rate'))} per year.")
    return out


def _h_forest_change(c):
    s = c.s
    out = [f"Forest went from {c.f('ha', s.get('forest_area_ha_before'))} in {s.get('year_before')} to "
           f"{c.f('ha', s.get('forest_area_ha_after'))} in {s.get('year_after')}: loss {c.f('ha', s.get('forest_loss_ha'))}, "
           f"gain {c.f('ha', s.get('forest_gain_ha'))}."]
    if s.get("loss_share_of_initial_forest") is not None:
        out.append(f"The loss is {c.f('frac', s.get('loss_share_of_initial_forest'))} of the {s.get('year_before')} forest "
                   f"({c.f('ha', s.get('loss_ha_per_year'))} per year).")
    return out


def _h_water_history(c):
    s = c.s
    return [f"{c.f('ha', s.get('ever_water_ha'))} ({c.f('frac', s.get('ever_water_share'))} of the valid area) was water at some time "
            f"between 1984 and 2020.",
            f"By the end of 2020: {c.f('ha', s.get('permanent_now_ha'))} permanent and {c.f('ha', s.get('seasonal_now_ha'))} seasonal water; "
            f"{c.f('ha', s.get('lost_ha'))} of water had been lost and {c.f('ha', s.get('ephemeral_ha'))} was ephemeral."]


def _h_et(c):
    s = c.s
    return [f"Mean evapotranspiration in {s.get('year')} is {c.num(s.get('mean_et_mm'), 0)} mm per year "
            f"(range {c.num(s.get('min_et_mm'), 0)} to {c.num(s.get('max_et_mm'), 0)} mm).",
            f"Potential evapotranspiration averages {c.num(s.get('mean_pet_mm'), 0)} mm; the ET/PET ratio is {c.num(s.get('mean_et_pet_ratio'), 2)}. "
            f"Over the area that is {c.num(s.get('total_et_volume_million_m3'), 2)} million m³ of water."]


def _h_active_fire(c):
    s = c.s
    n = s.get("detections_pixel_days") or 0
    if not n:
        return [f"No fire pixel at '{s.get('min_confidence')}' confidence or above was detected between {s.get('start')} and {s.get('end')} "
                f"({s.get('days_with_data')} days with data)."]
    return [f"{c.f('int', n)} fire detections (pixel-days) on {c.f('int', s.get('pixels_with_fire'))} different 1 km pixels between "
            f"{s.get('start')} and {s.get('end')}.",
            f"A pixel burned for up to {s.get('max_days_with_fire_in_a_pixel')} day(s); the strongest fire radiative power is "
            f"{c.num(s.get('max_frp_mw'), 1)} MW."]


def _h_terrain(c):
    s = c.s
    return [f"Elevation runs from {c.f('m', s.get('min_elevation_m'))} to {c.f('m', s.get('max_elevation_m'))} "
            f"(relief {c.f('m', s.get('relief_m'))}, mean {c.f('m', s.get('mean_elevation_m'))}).",
            f"Mean slope is {c.f('deg', s.get('mean_slope_deg'))}; the steepest 5 % of cells exceed {c.f('deg', s.get('p95_slope_deg'))}."]


def _h_drainage(c):
    s = c.s
    out = [f"Streams draining at least {c.num(s.get('stream_threshold_km2'), 2)} km² total {c.num(s.get('stream_length_km'), 1)} km "
           f"(density {c.num(s.get('drainage_density_km_per_km2'), 2)} km per km²), up to Strahler order {s.get('max_strahler_order')}."]
    hands = sorted((k for k in s if k.startswith("pct_area_hand_le_")), key=lambda k: float(k.removeprefix("pct_area_hand_le_").removesuffix("m")))
    if hands:
        out.append("Share of the area within a given height above the nearest stream: " + ", ".join(
            f"{c.f('frac', s[k])} within {k.removeprefix('pct_area_hand_le_')}" for k in hands) + ".")
    return out


def _h_veg_anomaly(c):
    s = c.s
    return [f"NDVI in {c.m.get('date')} is on average {c.num(s.get('mean_anomaly'), 3)} from the {len(s.get('baseline_years_used') or [])}-year "
            f"baseline mean ({c.num(s.get('mean_ndvi_target'), 2)} against {c.num(s.get('mean_ndvi_baseline'), 2)}).",
            f"{c.f('frac', s.get('pct_below_normal'))} of the observed area is more than one standard deviation below normal and "
            f"{c.f('frac', s.get('pct_above_normal'))} above; mean Vegetation Condition Index {c.num(s.get('mean_vci'), 0)} of 100."]


def _h_water_series(c):
    s = c.s
    return [f"{s.get('n_dates_usable')} of {s.get('n_dates_requested')} dates were observed well enough to use. Observed water area ranged "
            f"from {c.f('ha', s.get('min_water_area_ha'))} to {c.f('ha', s.get('max_water_area_ha'))}.",
            f"{c.f('ha', s.get('always_water_ha'))} was water on every date it was seen and {c.f('ha', s.get('variable_water_ha'))} "
            f"was water on some dates only."]


# ---------------------------------------------------------------------------
# Class legends for the new products
# ---------------------------------------------------------------------------

_LC_CHANGE_YEAR = _cd((1, "Water", "#1A5BAB"), (2, "Trees", "#358221"), (4, "Flooded vegetation", "#87D19E"), (5, "Crops", "#FFDB5C"),
                      (7, "Built area", "#ED022A"), (8, "Bare ground", "#EDE9E4"), (9, "Snow and ice", "#F2FAFF"),
                      (10, "Clouds", "#C8C8C8"), (11, "Rangeland", "#C6AD8D"))
_GSW_CLASSES = _cd((0, "Never water", NO_CHANGE), (1, "Permanent", "#08306b"), (2, "New permanent", "#2171b5"), (3, "Lost permanent", "#d94801"),
                   (4, "Seasonal", "#6baed6"), (5, "New seasonal", "#9ecae1"), (6, "Lost seasonal", "#fdae6b"),
                   (7, "Seasonal to permanent", "#4292c6"), (8, "Permanent to seasonal", "#c6dbef"),
                   (9, "Ephemeral permanent", "#fdd0a2"), (10, "Ephemeral seasonal", "#fee6ce"))
_STREAM_ORDER = _cd((0, "No stream", NO_CHANGE), (1, "Order 1", "#c6dbef"), (2, "Order 2", "#9ecae1"), (3, "Order 3", "#6baed6"),
                    (4, "Order 4", "#4292c6"), (5, "Order 5", "#2171b5"), (6, "Order 6", "#08519c"), (7, "Order 7", "#08306b"),
                    (8, "Order 8", "#041f4a"))
_ANOMALY_CLASSES = _cd((0, "Much below normal (z <= -2)", "#8c510a"), (1, "Below normal (-2 < z <= -1)", "#d8b365"),
                       (2, "Near normal (|z| < 1)", NO_CHANGE), (3, "Above normal (1 <= z < 2)", "#5ab4ac"),
                       (4, "Much above normal (z >= 2)", "#01665e"))

R_IO = "Karra, K. et al. (2021). Global land use / land cover with Sentinel-2 and deep learning. IGARSS 2021, 4704-4707. (Impact Observatory / Esri 10 m annual land cover, io-lulc-annual-v02, Microsoft Planetary Computer.)"
R_FNF = "Shimada, M. et al. (2014). New global forest/non-forest maps from ALOS PALSAR data (2007-2010). Remote Sensing of Environment 155, 13-31. (JAXA global PALSAR-2/PALSAR forest/non-forest map, Microsoft Planetary Computer.)"
R_GSW = "Pekel, J.-F., Cottam, A., Gorelick, N. & Belward, A.S. (2016). High-resolution mapping of global surface water and its long-term changes. Nature 540, 418-422."
R_MOD16 = "Running, S., Mu, Q., Zhao, M. & Moreno, A. MODIS/Terra Net Evapotranspiration Gap-Filled Yearly L4 Global 500 m SIN Grid V061 (MOD16A3GF). NASA LP DAAC."
R_FIRE = "Giglio, L., Schroeder, W. & Justice, C.O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. Remote Sensing of Environment 178, 31-41."
R_COPDEM = "European Space Agency / Airbus. Copernicus DEM GLO-30 (doi:10.5270/ESA-c5d3d65)."
R_TRI = "Riley, S.J., DeGloria, S.D. & Elliot, R. (1999). A terrain ruggedness index that quantifies topographic heterogeneity. Intermountain Journal of Sciences 5(1-4), 23-27."
R_FLOOD = "Barnes, R., Lehman, C. & Mulla, D. (2014). Priority-flood: an optimal depression-filling and watershed-labeling algorithm for digital elevation models. Computers & Geosciences 62, 117-127."
R_D8 = "O'Callaghan, J.F. & Mark, D.M. (1984). The extraction of drainage networks from digital elevation data. Computer Vision, Graphics, and Image Processing 28(3), 323-344."
R_STRAHLER = "Strahler, A.N. (1957). Quantitative analysis of watershed geomorphology. Transactions of the American Geophysical Union 38(6), 913-920."
R_HAND = "Rennó, C.D. et al. (2008). HAND, a new terrain descriptor using SRTM-DEM: mapping terra-firme rainforest environments in Amazonia. Remote Sensing of Environment 112(9), 3469-3481."
R_VCI = "Kogan, F.N. (1995). Application of vegetation index and brightness temperature for drought detection. Advances in Space Research 15(11), 91-100."
R_MSFT = "Microsoft. Global ML Building Footprints (github.com/microsoft/GlobalMLBuildingFootprints), Open Data Commons Open Database License."
R_GOOGLE = "Sirko, W. et al. (2021). Continental-scale building detection from high resolution satellite imagery. arXiv:2107.12283. (Google Open Buildings v3.)"
R_OSM = "OpenStreetMap contributors. Data (c) OpenStreetMap, available under the Open Database License (openstreetmap.org/copyright)."


def _lc_year_layers(stats, meta):
    out = []
    for key, tag in (("year_before", "before"), ("year_after", "after")):
        y = stats.get(key)
        if y is not None:
            out.append(Layer(f"landcover_{tag}", f"landcover_{y}.tif", f"Land cover {y}", "classes", classes=_LC_CHANGE_YEAR,
                             optional=True))
    return out


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
        "sar_flood_mapping", "Flood Extent Mapping (Sentinel-1 radar)",
        "Which open-water areas are newly flooded compared with a dry reference, as seen by radar through cloud?",
        ("Only Sentinel-1 scenes of one orbit track (same direction and relative orbit) present in both windows are compared, so the viewing geometry is the same.",
         "Each window is the per-pixel median, in dB, of its terrain-corrected backscatter scenes (a 3x3 mean filter is applied to each scene first), which suppresses speckle.",
         "Water = backscatter below the threshold in dB. Flooded = water in the flood window, not water in the reference, and at least the stated number of dB darker; permanent water = water in both.",
         "Flooded patches smaller than the minimum patch size are removed. Pixels without a valid measurement in either window are nodata, never 'dry'."),
        (Layer("flood", "flood_extent.tif", "Flood extent", "classes",
               classes=_cd((0, "Dry", NO_CHANGE), (1, "Flooded", "#e6550d"), (2, "Permanent water", _WATER_BLUE)), muted=(0,),
               vector=VectorSpec("flood_extent", keep=(1, 2)), primary=True),
         Layer("change", "change_db.tif", "Backscatter change, flood minus reference (dB)", "continuous", cmap="RdBu", vmin=-10, vmax=10,
               center=0.0, label="dB", optional=True)),
        (Metric("flooded_area_ha", "Flooded area", "ha"), Metric("permanent_water_area_ha", "Permanent water", "ha"),
         Metric("flooded_fraction", "Flooded share of valid area", "frac"), Metric("water_db", "Water threshold (dB)", "num1"),
         Metric("change_db", "Required darkening (dB)", "num1"), Metric("scenes_reference", "Reference scenes", "int"),
         Metric("scenes_flood", "Flood-window scenes", "int"), Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_sar_flood,
        reading=("The flood layer shows open floodwater seen by radar on the dates of the scenes used, not the maximum extent of the event.",
                 "Compare the water threshold with the Otsu estimate in the statistics: if they differ a lot, the conventional default may not suit this scene."),
        limitations=("Floodwater between buildings and under dense vegetation is bright to radar (double bounce) and is not detected, so urban and forest flooding is under-mapped; use this for open-water flooding such as river plains and fields.",
                     "Water that had receded before the overpass is not seen, and a flood that peaked between two passes is under-estimated.",
                     "Wind-roughened water is bright and missed; wet soil, smooth tarmac and radar shadow can be dark. Requiring the pixel to be darker than the same-track reference removes shadow and permanent dark surfaces but not everything.",
                     "The default water threshold is a conventional value per polarisation, not a calibration to this scene. There is no ground truth here, so accuracy is not assessed."),
        references=(R_TWELE, R_OTSU, R_S1RTC)),

    ReportSpec(
        "multisensor_flood_mapping", "Flood Extent Mapping (radar + optical + terrain)",
        "Where is there newly flooded ground, and how many independent sensors agree?",
        ("Radar: Sentinel-1 scenes of one orbit track are compared between a dry reference window and the flood window (per-pixel median in dB; water below the threshold; at least the stated number of dB darker).",
         "Optical: Sentinel-2 MNDWI = (green - SWIR1)/(green + SWIR1) is computed per scene with cloud, shadow and cirrus removed using the Scene Classification Layer, then the per-pixel median of the clear observations is taken for each window; water = MNDWI above the threshold.",
         "The sensors inform each other: the radar water threshold is calibrated on the optical water map of the reference window (so both describe the same dry-period water), the radar change is measured after removing the average brightness offset between the two windows, and a clear optical view of the flood window confirms or contradicts each radar flood pixel.",
         "The sensors are not blended into one score. Each pixel is classed by what each sensor says: confirmed (both), radar only (optical had no clear view), optical only, or disputed (radar flooded, optical clear and dry).",
         "Terrain: flooding on ground higher than the stated height above the nearest drainage (HAND, from the Copernicus DEM) is removed. Water already present in the reference window is shown separately.",
         "Small patches are removed; pixels no sensor could observe are nodata, never 'dry'."),
        (Layer("flood", "flood_extent.tif", "Flood extent by sensor agreement", "classes",
               classes=_cd((0, "Not flooded", NO_CHANGE), (1, "Flooded: radar + optical", "#b2182b"), (2, "Flooded: radar only", "#ef8a62"),
                           (3, "Flooded: optical only", "#fdb863"), (4, "Disputed", "#8073ac"), (5, "Water in reference window", _WATER_BLUE)),
               muted=(0, 5), vector=VectorSpec("flood_extent", keep=(1, 2, 3, 4)), primary=True),
         Layer("change", "change_db.tif", "Radar backscatter change, flood minus reference (dB)", "continuous", cmap="RdBu", vmin=-10, vmax=10,
               center=0.0, label="dB", optional=True),
         Layer("mndwi", "mndwi_after.tif", "Optical MNDWI in the flood window", "continuous", cmap="BrBG", vmin=-1, vmax=1,
               center=0.0, label="MNDWI", optional=True),
         Layer("hand", "hand_m.tif", "Height above nearest drainage (m)", "continuous", cmap="viridis_r", vmin=0, vmax=30,
               label="m", optional=True)),
        (Metric("flooded_area_ha", "Flooded area (classes 1-3)", "ha"), Metric("flooded_fraction", "Flooded share of land", "frac"),
         Metric("flooded_confirmed_ha", "Confirmed by radar and optical", "ha"),
         Metric("flooded_radar_only_ha", "Radar only", "ha"), Metric("flooded_optical_only_ha", "Optical only", "ha"),
         Metric("flooded_disputed_ha", "Disputed", "ha"), Metric("permanent_water_area_ha", "Water in reference window", "ha"),
         Metric("excluded_by_hand_ha", "Removed by terrain (HAND)", "ha"), Metric("optical_coverage", "Share of land with a clear optical view", "frac"),
         Metric("radar_optical_f1_reference", "Radar-optical water agreement before the event (F1)", "num2"),
         Metric("water_db", "Radar water threshold (dB)", "num1"), Metric("radiometric_offset_db", "Radar offset between windows (dB)", "num2"),
         Metric("valid_fraction", "Valid share of AOI", "frac")),
        _h_multisensor,
        reading=("Trust 'radar + optical' most. 'Radar only' is expected where the optical view was cloud-covered; 'disputed' areas are where the sensors contradict each other and should be checked.",
                 "If 'Data used' lacks sentinel-2 or the terrain, the result is radar-based only and the data gaps line says why."),
        limitations=("Floodwater between buildings and under vegetation is bright to radar and dark-to-bright in optical imagery; urban flooding is under-mapped by both. Use local reports or high-resolution imagery for towns.",
                     "Optical imagery cannot see through cloud, and the cloudiest dates are often the flood dates. 'Optical only' and 'confirmed' therefore describe the clear days only.",
                     "Water that receded before the acquisitions is not seen; a peak between passes is under-estimated.",
                     "The water thresholds (radar dB, MNDWI 0) are conventional values, not calibrated to this scene, and there is no ground truth here, so accuracy is not assessed."),
        references=(R_TWELE, R_MNDWI, R_HAND, R_COPDEM, R_S1RTC, R_S2)),

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
        "building_footprints", "Building Footprints",
        "Where are the buildings in the area, how many, and how much ground do they cover?",
        ("Published footprints (default): the polygons of Microsoft's Global ML Building Footprints, Google's Open Buildings v3 or OpenStreetMap are read for the box, clipped to it and measured in the local UTM zone. Nothing is detected on your imagery.",
         "building_density.tif is the share of each cell covered by building area (a building counts in the cell holding its centroid); building_count.tif counts buildings per cell.",
         "With a checkpoint the pipeline instead runs a segmentation network you trained (class 1 = building) and writes building_mask.tif."),
        (Layer("density", "building_density.tif", "Share of each cell covered by buildings", "continuous", cmap="YlOrRd", vmin=0, label="share of cell",
               primary=True, optional=True),
         Layer("count", "building_count.tif", "Buildings per cell", "continuous", cmap="Purples", vmin=0, label="buildings", optional=True),
         Layer("mask", "building_mask.tif", "Building mask (trained network)", "classes",
               classes=_cd((0, "Background", NO_CHANGE), (1, "Building", "#c2185b")), muted=(0,),
               vector=VectorSpec("buildings", keep=(1,)), optional=True)),
        (Metric("building_count", "Buildings", "int"), Metric("building_area_ha", "Total footprint area", "ha"),
         Metric("building_coverage", "Share of the area covered by buildings", "frac"), Metric("buildings_per_km2", "Buildings per km²", "num1"),
         Metric("mean_building_area_m2", "Mean footprint (m²)", "num1"), Metric("median_building_area_m2", "Median footprint (m²)", "num1"),
         Metric("max_cell_coverage", "Highest cell coverage", "frac"), Metric("aoi_area_km2", "Area (km²)", "num2")),
        _h_buildings,
        reading=("Dataset footprints differ in date: a recent building may be missing and a demolished one still present.",),
        limitations=("Published footprints are only as complete, accurate and current as their source; compare the three sources if it matters.",
                     "A building_mask.tif result (only when you pass a checkpoint) is the output of a network you trained and has not been checked against reference data; at 10 m individual houses are not resolvable."),
        references=(R_MSFT, R_GOOGLE, R_OSM)),

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

_NEW_SPECS: list[ReportSpec] = [
    ReportSpec(
        "landcover_change", "Land-Cover Change (annual 10 m land cover)",
        "Where does the annual 10 m land-cover map differ between two years, and from which class to which?",
        ("The Impact Observatory / Esri annual land-cover product (2017-2023, nine classes) is read for both years from the "
         "Microsoft Planetary Computer, windowed to the area and resampled (nearest neighbour) onto one UTM grid.",
         "A pixel is 'changed' when its class differs between the years. Pixels that are no-data or cloud in either year are nodata.",
         "`min_change_ha` can remove changed patches smaller than a given area; the transition table lists the largest from-to pairs."),
        (Layer("change", "change_mask.tif", "Land-cover change", "classes",
               classes=_cd((0, "Unchanged", NO_CHANGE), (1, "Changed class", "#d7301f")), muted=(0,),
               vector=VectorSpec("landcover_change", keep=(1,)), primary=True),),
        (Metric("changed_area_ha", "Changed area", "ha"), Metric("change_fraction", "Share of valid area changed", "frac"),
         Metric("changed_in_patches_of_3px_or_less", "Changed area in patches of 3 pixels or fewer", "frac",
                "a high value points to classifier noise rather than real change"),
         Metric("valid_fraction", "Valid share of the area", "frac"), Metric("resolution_m", "Grid resolution (m)", "num1")),
        _h_lc_change,
        reading=("Each year is classified independently, so a pixel can change class because of classifier disagreement. Large, compact patches are more trustworthy than scattered single pixels.",),
        limitations=("Not an accuracy assessment: the product is a global model output and its errors differ by class and region.",
                     "Short-lived land cover (seasonal flooding, fallow, burn scars) changes class between years without a lasting change on the ground.",
                     "Reported areas are pixel counts on the output grid, not survey measurements."),
        references=(R_IO,),
        tables=(TableSpec("transitions", "Largest class changes", (("from", "From", "text"), ("to", "To", "text"), ("area_ha", "Area (ha)", "num1")),
                          sheet="Transitions"),
                TableSpec("area_ha_before", "Area by class, first year", key_header="Class", value_header="Area (ha)", value_fmt="num1", sheet="Area before"),
                TableSpec("area_ha_after", "Area by class, second year", key_header="Class", value_header="Area (ha)", value_fmt="num1", sheet="Area after")),
        charts=(ChartSpec("transitions", "Largest class changes (hectares)", "Area (ha)", y="area_ha", x=("from", "to"), kind="barh", fmt="{:,.2f}"),),
        extra_layers=_lc_year_layers),

    ReportSpec(
        "urban_expansion", "Urban Expansion (annual 10 m land cover)",
        "How much built-up area appeared or disappeared between two years?",
        ("Built area is class 7 of the annual 10 m land-cover product (Impact Observatory / Esri, 2017-2023) on both dates.",
         "A pixel is 'new built' when it is built in the second year only, 'lost' when built in the first year only. Gross gain and loss are reported separately because each year is classified on its own.",
         "Pixels that are no-data or cloud in either year are nodata; percentages are over valid pixels."),
        (Layer("built", "built_area_change.tif", "Built-up area change", "classes",
               classes=_cd((0, "Not built", NO_CHANGE), (1, "Built in both years", "#8d99a6"), (2, "New built area", "#d7301f"),
                           (3, "Built area lost", "#2b83ba")), muted=(0,),
               vector=VectorSpec("built_change", keep=(2, 3)), primary=True),),
        (Metric("built_area_ha_before", "Built area, first year", "ha"), Metric("built_area_ha_after", "Built area, second year", "ha"),
         Metric("gain_ha", "Gross gain", "ha"), Metric("loss_ha", "Gross loss", "ha"), Metric("net_change_ha", "Net change", "ha"),
         Metric("built_share_before", "Built share, first year", "frac"), Metric("built_share_after", "Built share, second year", "frac"),
         Metric("annual_growth_rate", "Compound annual growth of built area", "frac"), Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_urban_expansion,
        reading=("'Built area lost' is more often classifier disagreement (roofs under trees, dust, construction sites) than demolition. Read the gross figures, not only the net.",),
        limitations=("At 10 m, individual houses and narrow streets are not resolved; small settlements are under-mapped.",
                     "This is a map product's change, not a validated change inventory.",),
        references=(R_IO,)),

    ReportSpec(
        "forest_change", "Forest Change (JAXA PALSAR forest / non-forest)",
        "How much forest was lost or gained between two years from the radar-based forest map?",
        ("The JAXA global PALSAR forest / non-forest map (25 m, 2015-2020) is read for both years from the Microsoft Planetary Computer and resampled onto one UTM grid.",
         "'Forest' is class 1 (canopy cover above 90 %) by default, or classes 1 and 2 (above 10 %) with include_open_forest.",
         "Pixels that are water or no-data in either year are nodata. Loss is forest then, non-forest now; gain is the reverse."),
        (Layer("forest", "forest_change.tif", "Forest change", "classes",
               classes=_cd((0, "Non-forest", NO_CHANGE), (1, "Forest in both years", "#2f6f3e"), (2, "Forest loss", "#d55e00"),
                           (3, "Forest gain", "#0072b2")), muted=(0, 1),
               vector=VectorSpec("forest_change", keep=(2, 3)), primary=True),),
        (Metric("forest_area_ha_before", "Forest, first year", "ha"), Metric("forest_area_ha_after", "Forest, second year", "ha"),
         Metric("forest_loss_ha", "Forest loss", "ha"), Metric("forest_gain_ha", "Forest gain", "ha"), Metric("net_change_ha", "Net change", "ha"),
         Metric("loss_ha_per_year", "Loss per year", "ha"), Metric("loss_share_of_initial_forest", "Loss as share of first-year forest", "frac"),
         Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_forest_change,
        reading=("Radar backscatter responds to structure and moisture, so plantations count as forest and flooded or very dry conditions can move pixels between classes. Treat small, scattered changes as noise.",),
        limitations=("Each year is mapped separately: gains and losses include mapping disagreement, not only change on the ground.",
                     "Covers 2015-2020 only; it says nothing about later forest loss.",
                     "It cannot tell the cause of loss (logging, fire, farming, natural dieback)."),
        references=(R_FNF,)),

    ReportSpec(
        "surface_water_dynamics", "Surface Water History 1984-2020 (JRC Global Surface Water)",
        "Where has surface water been permanent, seasonal, new or lost over the Landsat record?",
        ("JRC Global Surface Water v1.3 (30 m, built from every usable Landsat 5/7/8 scene from March 1984 to December 2020) is read from the Microsoft Planetary Computer.",
         "The transitions layer compares the first and last years of the record; occurrence is the percentage of valid observations that showed water; seasonality is the number of months with water in 2020.",
         "Class groups: permanent now = permanent, new permanent, seasonal to permanent; seasonal now = seasonal, new seasonal, permanent to seasonal; lost = lost permanent, lost seasonal; ephemeral = ephemeral permanent and seasonal."),
        (Layer("transitions", "water_transitions.tif", "Water transitions 1984-2020", "classes", classes=_GSW_CLASSES, muted=(0,),
               vector=VectorSpec("water_transitions", skip=(0,)), primary=True),
         Layer("occurrence", "water_occurrence.tif", "Water occurrence (% of observations)", "continuous", cmap="Blues", vmin=0, vmax=100,
               label="% of observations", optional=True),
         Layer("seasonality", "water_seasonality.tif", "Months with water in 2020", "continuous", cmap="YlGnBu", vmin=0, vmax=12,
               label="months", optional=True)),
        (Metric("ever_water_ha", "Water at some time, 1984-2020", "ha"), Metric("ever_water_share", "Share of valid area ever water", "frac"),
         Metric("permanent_now_ha", "Permanent water now", "ha"), Metric("seasonal_now_ha", "Seasonal water now", "ha"),
         Metric("lost_ha", "Lost water", "ha"), Metric("ephemeral_ha", "Ephemeral water", "ha"), Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_water_history,
        reading=("Occurrence near 100 % marks permanent water; low occurrence marks floodplain or shoreline that is wet only sometimes.",),
        limitations=("The record ends in December 2020: later change is not shown. Use water_bodies for a current map and flood_mapping for an event.",
                     "Observation counts differ by place (clouds, Landsat 7 stripes before 2003 coverage), so occurrence is less certain where few scenes were usable.",),
        references=(R_GSW,)),

    ReportSpec(
        "evapotranspiration", "Annual Evapotranspiration (MODIS MOD16A3GF)",
        "How much water does the land surface return to the atmosphere in a year, compared with the potential?",
        ("MOD16A3GF (500 m, gap-filled yearly Penman-Monteith ET and PET) is read from the Microsoft Planetary Computer for the requested year.",
         "Stored values are scaled by 0.1 to kg m-2 per year (= mm per year); fill and land-class codes (open water, urban, barren, snow) are nodata, never zero.",
         "ET/PET is computed where both are valid and PET is above zero. The volume is mean ET depth times the valid area."),
        (Layer("et", "et_mm_per_year.tif", "Evapotranspiration (mm per year)", "continuous", cmap="YlGnBu", label="mm per year", primary=True),
         Layer("pet", "pet_mm_per_year.tif", "Potential evapotranspiration (mm per year)", "continuous", cmap="YlOrBr", label="mm per year", optional=True),
         Layer("ratio", "et_pet_ratio.tif", "ET / PET", "continuous", cmap="RdYlBu", vmin=0, vmax=1.2, label="ET / PET", optional=True)),
        (Metric("mean_et_mm", "Mean ET (mm per year)", "num1"), Metric("mean_pet_mm", "Mean PET (mm per year)", "num1"),
         Metric("mean_et_pet_ratio", "Mean ET / PET", "num2"), Metric("total_et_volume_million_m3", "Total ET volume (million m³)", "num2"),
         Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_et,
        reading=("ET/PET near 1 means water is not limiting; low values mean water-limited land (dry season, drylands, sparse cover).",),
        limitations=("500 m pixels average everything inside them: small fields, wetlands and urban pockets are blended.",
                     "A modelled product driven by weather reanalysis and MODIS vegetation; it is not measured flux and has no uncertainty layer here.",
                     "Open water and built-up pixels have no estimate, so the volume excludes them."),
        references=(R_MOD16,)),

    ReportSpec(
        "active_fire", "Active Fire Detections (MODIS MOD14A1)",
        "Where and on how many days did MODIS detect fire in a period?",
        ("Daily MOD14A1 granules (1 km) are read from the Microsoft Planetary Computer for the period.",
         "A pixel-day is a fire detection when its fire mask is 7 (low), 8 (nominal) or 9 (high confidence) and at least min_confidence; cloud, water and unknown pixels are never counted as 'no fire'.",
         "fire_days.tif counts detection days per pixel; observed_days.tif counts days the pixel was seen clear; max_frp_mw.tif is the strongest fire radiative power (MW)."),
        (Layer("fire_days", "fire_days.tif", "Days with a fire detection", "continuous", cmap="YlOrRd", vmin=0, label="days", primary=True),
         Layer("observed", "observed_days.tif", "Days observed clear", "continuous", cmap="Greys", vmin=0, label="days", optional=True),
         Layer("frp", "max_frp_mw.tif", "Maximum fire radiative power (MW)", "continuous", cmap="magma_r", vmin=0, label="MW", optional=True)),
        (Metric("detections_pixel_days", "Fire detections (pixel-days)", "int"), Metric("pixels_with_fire", "Pixels with at least one detection", "int"),
         Metric("max_days_with_fire_in_a_pixel", "Most detection days in one pixel", "int"), Metric("max_frp_mw", "Strongest FRP (MW)", "num1"),
         Metric("days_with_data", "Days with data", "int"), Metric("mean_observed_days", "Mean days observed clear per pixel", "num1")),
        _h_active_fire,
        reading=("A detection is a hot 1 km pixel at overpass time, not a burned area. Use wildfire_severity (dNBR) for burn extent.",),
        limitations=("Fires smaller or cooler than the sensor threshold, or under cloud and thick smoke, are missed; absence is not evidence of no fire.",
                     "Two overpasses a day at most (Terra and Aqua are separate products; this uses Terra MOD14A1), so short fires between overpasses are missed.",
                     "Agricultural burning, gas flares and hot bare ground can also trigger detections."),
        references=(R_FIRE,),
        tables=(TableSpec("per_day_detections", "Detections per day", key_header="Date", value_header="Fire pixels", value_fmt="int", sheet="Per day"),),
        charts=(ChartSpec("per_day_detections", "Fire pixels per day", "Fire pixels", kind="bar", fmt="{:,.0f}"),)),

    ReportSpec(
        "terrain_analysis", "Terrain Analysis (Copernicus DEM)",
        "What do elevation, slope, aspect and ruggedness look like in the area?",
        ("Elevation comes from the Copernicus GLO-30 DEM (or an OpenTopography product), warped onto a metric grid.",
         "Slope and aspect use Horn's 3x3 method; aspect is the compass bearing of the downhill direction. The hillshade is lit from the stated sun azimuth and elevation.",
         "TRI is the root of the summed squared differences between a cell and its eight neighbours."),
        (Layer("slope", "slope_deg.tif", "Slope (degrees)", "continuous", cmap="YlOrBr", vmin=0, label="degrees", primary=True),
         Layer("elevation", "elevation.tif", "Elevation (m)", "continuous", cmap="terrain", label="m", optional=True),
         Layer("hillshade", "hillshade.tif", "Hillshade", "continuous", cmap="gray", vmin=0, vmax=1, label="illumination", optional=True),
         Layer("aspect", "aspect_deg.tif", "Aspect (bearing of the downhill direction)", "continuous", cmap="twilight", vmin=0, vmax=360, label="degrees", optional=True),
         Layer("tri", "tri_m.tif", "Terrain ruggedness index (m)", "continuous", cmap="magma_r", vmin=0, label="m", optional=True)),
        (Metric("min_elevation_m", "Lowest elevation", "m"), Metric("max_elevation_m", "Highest elevation", "m"),
         Metric("mean_elevation_m", "Mean elevation", "m"), Metric("relief_m", "Relief", "m"),
         Metric("mean_slope_deg", "Mean slope", "deg"), Metric("p95_slope_deg", "95th percentile slope", "deg"),
         Metric("mean_tri_m", "Mean ruggedness (TRI)", "m"), Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_terrain,
        reading=("Slope is the steepest rate of change over a 3 x 3 cell window, so it is smoother than the true ground at fine scale.",),
        limitations=("The DEM is a surface model: tree canopy and buildings are included in the heights.",
                     "Slope and ruggedness depend on the grid resolution; compare areas only at the same resolution."),
        references=(R_COPDEM, R_HORN, R_TRI)),

    ReportSpec(
        "drainage_analysis", "Drainage Analysis (flow accumulation, streams, HAND)",
        "Where does water collect, which cells form streams, and how high does the ground stand above them?",
        ("The DEM is conditioned with Priority-Flood (depressions filled, flats resolved); water is routed D8 to the steepest downhill neighbour; flow accumulation is the draining area through each cell.",
         "Cells draining at least the stream threshold are streams, numbered by Strahler order.",
         "HAND (Height Above the Nearest Drainage) is each cell's elevation above the stream cell it drains into."),
        (Layer("hand", "hand_m.tif", "Height above nearest drainage (m)", "continuous", cmap="Blues_r", vmin=0, vmax=30, label="m", primary=True),
         Layer("streams", "stream_order.tif", "Stream network by Strahler order", "classes", classes=_STREAM_ORDER, muted=(0,)),
         Layer("flow", "flow_accumulation_km2.tif", "Contributing area (km²)", "continuous", cmap="PuBu", vmin=0, label="km²", optional=True)),
        (Metric("stream_length_km", "Stream length (km)", "num1"), Metric("drainage_density_km_per_km2", "Drainage density (km per km²)", "num2"),
         Metric("max_strahler_order", "Highest Strahler order", "int"), Metric("mean_hand_m", "Mean HAND", "m"),
         Metric("largest_contributing_area_km2", "Largest contributing area (km²)", "num2"), Metric("cells_filled", "Cells raised by depression filling", "int"),
         Metric("max_fill_depth_m", "Deepest fill", "m"), Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_drainage,
        reading=("Low HAND is a necessary condition for river flooding, not a flood map: it knows nothing of rainfall, discharge, channel capacity or defences.",),
        limitations=("Contributing areas are cut at the edge of the box, so streams near the boundary are under-estimated; use a box that includes the whole catchment where it matters.",
                     "The stream threshold is a choice. Channels, culverts and embankments finer than the DEM resolution are not represented.",
                     "A surface model (canopy, buildings) can block or divert modelled flow."),
        references=(R_COPDEM, R_FLOOD, R_D8, R_STRAHLER, R_HAND)),

    ReportSpec(
        "vegetation_anomaly", "Vegetation Anomaly (NDVI against the same month in earlier years)",
        "Is vegetation greener or browner than usual for this time of year?",
        ("NDVI is computed for the requested month and for the same month in each baseline year, each from the best cloud-masked Sentinel-2 / Landsat mosaic; every year is aligned to the grid of the first.",
         "Anomaly = NDVI - baseline mean; z-score = anomaly / baseline standard deviation (sample, at least 3 years); VCI = 100 (NDVI - min) / (max - min) over the baseline and the target.",
         "A pixel needs at least three cloud-free baseline years. Cloud-masked pixels are nodata, never 'normal'."),
        (Layer("z", "ndvi_zscore.tif", "NDVI z-score against the baseline", "continuous", cmap="BrBG", vmin=-3, vmax=3, center=0.0, label="z-score", primary=True),
         Layer("class", "anomaly_class.tif", "Anomaly class", "classes", classes=_ANOMALY_CLASSES, muted=(2,)),
         Layer("anomaly", "ndvi_anomaly.tif", "NDVI anomaly (NDVI minus baseline mean)", "continuous", cmap="BrBG", vmin=-0.4, vmax=0.4, center=0.0,
               label="NDVI", optional=True),
         Layer("vci", "vci.tif", "Vegetation Condition Index", "continuous", cmap="RdYlGn", vmin=0, vmax=100, label="VCI", optional=True)),
        (Metric("mean_ndvi_target", "Mean NDVI, target month", "num3"), Metric("mean_ndvi_baseline", "Mean NDVI, baseline", "num3"),
         Metric("mean_anomaly", "Mean anomaly", "num3"), Metric("mean_vci", "Mean VCI", "num1"),
         Metric("pct_below_normal", "Share more than 1 sd below normal", "frac"), Metric("pct_above_normal", "Share more than 1 sd above normal", "frac"),
         Metric("min_baseline_observations", "Fewest baseline years at a valid pixel", "int"), Metric("valid_fraction", "Valid share of the area", "frac")),
        _h_veg_anomaly,
        reading=("Low values accompany drought, but also late planting, harvest, grazing, burning, flooding, land-cover change and cloud residue.",),
        limitations=("A vegetation anomaly is not a drought declaration: the pipeline sees no rainfall, soil moisture or crop type.",
                     "With few baseline years the baseline mean and standard deviation are themselves uncertain; z-scores are most meaningful with 5 or more years.",
                     "Different sensors across years can shift NDVI slightly."),
        references=(R_VCI, R_S2)),

    ReportSpec(
        "water_extent_timeseries", "Water Extent Through Time (MNDWI)",
        "How much open water is there on each date, and how often was each pixel water?",
        ("On each date water is MNDWI = (green - SWIR1) / (green + SWIR1) above the threshold (default 0), computed on cloud-free pixels only.",
         "Water frequency is, per pixel, the share of its cloud-free observations in which it was water (needs at least min_observations).",
         "A date observed on fewer than min_valid_fraction of the area is listed but left out of the frequency map."),
        (Layer("frequency", "water_frequency.tif", "Water frequency (share of cloud-free observations)", "continuous", cmap="Blues", vmin=0, vmax=1,
               label="share of observations", primary=True),
         Layer("observations", "valid_observations.tif", "Cloud-free observations", "continuous", cmap="Greys", vmin=0, label="dates", optional=True)),
        (Metric("n_dates_requested", "Dates requested", "int"), Metric("n_dates_usable", "Dates used", "int"),
         Metric("always_water_ha", "Water on every observed date", "ha"), Metric("variable_water_ha", "Water on some dates only", "ha"),
         Metric("ever_water_ha", "Water on at least one date", "ha"), Metric("min_water_area_ha", "Smallest observed water area", "ha"),
         Metric("max_water_area_ha", "Largest observed water area", "ha")),
        _h_water_series,
        reading=("Compare dates through the water fraction of the observed pixels, not the raw area: a cloudier date sees less ground and so less water.",),
        limitations=("Shadow, wet soil, burn scars and dark pavement can be mistaken for water, as in any single-index mask.",
                     "Water under cloud on a date is unknown, not dry; it contributes nothing to that date.",),
        references=(R_MNDWI, R_S2),
        tables=(TableSpec("series", "Water area by date",
                          (("date", "Date", "text"), ("usable", "Used", "bool"), ("valid_fraction", "Observed share", "frac"),
                           ("water_area_ha", "Water area (ha)", "num1"), ("water_fraction_of_valid", "Water share of observed", "frac"),
                           ("note", "Note", "text")), sheet="Timeseries"),),
        charts=(ChartSpec("series", "Observed water area by date", "Water area (ha)", y="water_area_ha", x="date", kind="line"),
                ChartSpec("series", "Water share of the observed pixels", "Share of observed pixels", y="water_fraction_of_valid", x="date",
                          kind="line", fmt="{:.1%}")),
        multi_date=False),

    ReportSpec(
        "road_network", "Road Network (OpenStreetMap)",
        "How many kilometres of road are mapped in the area, of which kinds, and where are they dense?",
        ("Every OpenStreetMap way tagged 'highway' inside the box is downloaded through the Overpass API, clipped to the box, and measured in the local UTM zone.",
         "Ways that are proposed, under construction, razed or abandoned are left out unless exclude says otherwise.",
         "Road density is kilometres of road per square kilometre in square cells; lines are cut every cell/5 metres and each piece is assigned to the cell holding its midpoint."),
        (Layer("density", "road_density.tif", "Road density (km per km²)", "continuous", cmap="YlOrBr", vmin=0, label="km per km²", primary=True),),
        (Metric("total_length_km", "Total road length (km)", "num1"), Metric("road_density_km_per_km2", "Road density (km per km²)", "num2"),
         Metric("segments", "Mapped ways", "int"), Metric("unnamed_share_of_length", "Share of length without a name", "frac"),
         Metric("aoi_area_km2", "Area (km²)", "num2")),
        _h_roads,
        reading=("Density is in cells of cell_m metres; very high values in a single cell can be a junction or dual carriageway rather than a dense grid.",),
        limitations=("This is what OpenStreetMap volunteers have drawn, not a detection from imagery. Minor roads and tracks in rural areas are often incomplete, so compare densities between places only where mapping effort is similar.",
                     "Each OSM way counts once: a divided road mapped as two ways is counted twice, which is how its carriageways are mapped but exceeds the centreline length.",
                     "Road class comes from the OSM 'highway' tag, which is a mapping convention and not a measured width or surface."),
        references=(R_OSM,),
        tables=(TableSpec("km_by_highway", "Road length by OpenStreetMap highway class", key_header="Highway class", value_header="Length (km)",
                          value_fmt="num2", sheet="Roads by class"),
                TableSpec("km_by_group", "Road length by group", key_header="Group", value_header="Length (km)", value_fmt="num2", sheet="Roads by group")),
        charts=(ChartSpec("km_by_highway", "Road length by highway class (km)", "Length (km)", kind="barh", fmt="{:,.1f} km"),)),

    ReportSpec(
        "road_extraction", "Road Extraction (trained network)",
        "Where does your network see roads?",
        ("Tiled inference with a segmentation network you trained (checkpoint required); class 1 = road.",),
        (Layer("mask", "road_mask.tif", "Road mask", "classes",
               classes=_cd((0, "Background", NO_CHANGE), (1, "Road", "#7b3294")), muted=(0,),
               vector=VectorSpec("roads", keep=(1,)), primary=True),),
        (Metric("road_coverage", "Road share of area", "frac"),),
        _h_network_mask("road", "road_coverage"),
        limitations=(_NETWORK_LIMIT, "Roads are only a few pixels wide at 10 m; expect gaps and merged parallel roads. Train and run on the same kind of imagery.",),
        references=()),

    ReportSpec(
        "landslide_detection", "Landslide Detection (trained network)",
        "Where does your network see landslide scars between a before and an after image?",
        ("Two-image (siamese) inference with a change network you trained (checkpoint required). Class 0 is no change; other classes are those you trained on.",),
        (Layer("change", "landslide_map.tif", "Landslide classes", "classes", classes_hint="user",
               vector=VectorSpec("landslides", skip=(0,)), primary=True),),
        (Metric("changed_fraction", "Changed share of area", "frac"),),
        _h_network_change,
        limitations=(_NETWORK_LIMIT, _USER_CLASSES_NOTE, "Cloud, shadow and seasonal change between the two images also alter the pixels; the network only sees the two images."),
        references=()),
]

_SPEC_LIST.extend(_NEW_SPECS)

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