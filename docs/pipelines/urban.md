# Urban

Ten real pipelines, including `wildfire_severity` (filed here since it
covers non-forest burns too) and the two pipelines with the most
explicit honest-scope warnings in the whole catalog.

```{note}
`land_cover` and `urban_growth` are two of the 10 CLI-reachable
pipelines with a separate, real implementation — see
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md) for the
real, correct detail (including `land_cover`'s real three-way
worldcover/dynamic_world/model dispatch, and `urban_growth`'s real,
deliberate use of Landsat rather than Sentinel-2).
```

## `urban_heat_island` and `land_surface_temperature`

**What they do:** Real land-surface-temperature (LST) mapping from
Landsat's thermal band — literally the same real implementation under
two names, not two techniques (`LandSurfaceTemperaturePipeline` is a
direct Python subclass of `UrbanHeatIslandPipeline` with no overridden
logic — the same pattern as `forest_fire`/`wildfire_severity` in
[Forestry](forestry.md)).

**How it actually works:** Retrieves the real thermal band (requires
Landsat specifically — Sentinel-2 has no thermal sensor at all), and
applies the real, published USGS Collection 2 Level-2 Surface
Temperature conversion directly: `temperature_K = DN × 0.00341802 +
149.0`.

```python
from pygeovision.ai.pipelines.domains import UrbanHeatIslandPipeline

result = UrbanHeatIslandPipeline(client).run(bbox=(...), date="2024-07", output_dir="./output")
print(result.stats)
# {"mean_temp_celsius": ..., "max_temp_celsius": ..., "min_temp_celsius": ...,
#  "pct_heat_island": ...}  # fraction of pixels >2°C warmer than the scene mean
```

```{warning}
Corrected from an earlier version of this page, which claimed a real
NDVI-derived emissivity correction was applied. Checked directly: no
emissivity correction of any kind exists in the real implementation —
it's the direct DN→Kelvin→Celsius conversion above, nothing more. The
real stats keys are also different from what was previously shown
(`mean_temp_celsius`/`pct_heat_island`, not `mean_lst_c`/
`hotspot_area_ha`/`urban_rural_delta_c`).
```

`pct_heat_island` uses a real, relative definition — pixels more than
2°C warmer than the scene's own mean — deliberately relative rather
than an absolute "hot" threshold, since what counts as hot varies by
climate and season.

## `solar_potential`

**What it does:** Real clear-sky solar irradiance estimation from
terrain — **not** rooftop detection (see
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md)'s
`solar_detection` for that).

**How it actually works:** A real, from-scratch DEM analysis, verified
against independent physics. Automatically fetches a real DEM from
OpenTopography for the requested bbox. Computes real slope and aspect
via Horn's method (a real, published 3×3-kernel gradient estimator —
the same algorithm GDAL's `gdaldem slope/aspect` uses), real solar
position (zenith and azimuth) via Spencer's 1971 declination equations
for the given date/time, then the standard tilted-surface
cosine-incidence-angle formula (Duffie & Beckman, *Solar Engineering
of Thermal Processes*). Verified: on a flat synthetic DEM, the output
exactly matches `cos(solar zenith angle)` as it must by definition; at
winter solstice, south-facing terrain scores higher than flat, which
scores higher than north-facing — matching real, independently-known
physical expectation for the northern hemisphere.

```python
from pygeovision.ai.pipelines.domains import SolarPotentialPipeline

result = SolarPotentialPipeline(client).run(
    bbox=(...), date="2024-06-21", hour=12.0,
    dem_type="srtm1arc", output_dir="./output",
)
print(result.stats)
# {"date": ..., "hour": 12.0, "dem_type": "srtm1arc",
#  "sun_zenith_deg": ..., "sun_azimuth_deg": ...,
#  "mean_relative_irradiance": ..., "pct_self_shaded": ..., "note": "..."}
```

```{important}
**Requires a real OpenTopography API key** — same requirement as
`archaeological_site` in [Heritage & Archaeology](heritage-and-archaeology.md).
```

```{warning}
Corrected from an earlier version of this page: there is no real
`dem_path=` parameter — this pipeline fetches a DEM automatically, it
doesn't accept a user-supplied file. It also previously claimed a
"tilted-surface irradiance model combining direct and diffuse
components" — checked directly against the source: there is no
atmospheric attenuation or diffuse-irradiance modeling of any kind,
only the geometric cosine-incidence factor. `mean_relative_irradiance`
is a real 0–1 relative modulation factor, not kWh/m² — there's no
`mean_irradiance_kwh_m2`/`high_potential_area_ha` in the real output.
```

```{warning}
This is a single, clear-sky snapshot for the given date and time — not
an annual-integrated yield estimate, and it does **not** model
shadow-casting from neighboring terrain or buildings. Treat the output
as a relative site-comparison tool, not a bankable energy-yield figure.
```

```{note}
`wildfire_severity` (and its identical-implementation sibling
`forest_fire`) is documented in full in
[Forestry](forestry.md) — filed
there since the two names share one real implementation, worth
explaining together rather than split across two pages. It's
mentioned here because it also covers non-forest burns (grassland,
shrubland).
```

## `mine_detection`

**What it does:** Flags large-scale bare-ground clearing —
**not** confirmed mining activity specifically.

**How it actually works:** A real, size-filtered bi-temporal
bare-ground land-cover transition — mines are characteristically
large, contiguous cleared areas, so this filters transitions by a
real, configurable minimum area (default 10 hectares) to reject small,
unrelated clearings (a new building pad, a felled tree).

```python
result = MineDetectionPipeline(client).run(
    bbox=(...), date_before="2015", date_after="2024",
    output_dir="./output", min_area_ha=10.0,
)
```

**Verification:** tested against synthetic ground truth — a 16-hectare
synthetic cleared patch was correctly flagged; single-pixel scatter
noise was correctly not flagged.

```{warning}
Detects large-scale clearing generically. A quarry, a large
construction site, and a genuine mine all produce the same real
signature — this does not itself confirm mining activity.
```

## `construction_progress`

**What it does:** Detects that new development happened —
**not** what stage of construction it's in.

**How it actually works:** A real, specific bi-temporal land-cover
transition check, restricted to class 50 (built-up) in the real ESA
WorldCover taxonomy — deliberately narrower than the generic
bare-ground check `mine_detection` uses above, since construction
sites transition specifically *to* the built-up class, not just to
bare ground.

```python
result = ConstructionProgressPipeline(client).run(
    bbox=(...), date_before="2023-01", date_after="2024-01",
    output_dir="./output",
)
# result.stats: {"new_construction_area_ha": ..., "n_sites": ...}
```

## `landcover_change`

**What it does:** Real, general bi-temporal land-cover comparison
across all 11 real ESA WorldCover classes, not just built-up.

```python
from pygeovision.ai.pipelines.domains import LandcoverChangePipeline

result = LandcoverChangePipeline(client).run(
    bbox=(...), date_before="2015", date_after="2024", output_dir="./output",
)
print(result.stats)
# {"pct_changed": ..., "n_pixels_changed": ...,
#  "top_transitions": {"tree_cover->built_up": 4821, ...}}  # top 10 only, by frequency
```

```{note}
Corrected from an earlier version of this page: the real stats key is
`top_transitions` (the top 10 real class-to-class transitions by
pixel count, not a full transition matrix), plus `pct_changed` and
`n_pixels_changed`.
```

**Honest limitation:** documented directly in the pipeline's own
docstring — unlike the bi-temporal pipelines that use
`_search_and_download_pair`'s explicit pixel-grid alignment step, this
pipeline runs two independent `LandCoverPipeline` calls with only a
shape-mismatch safety check, not a guaranteed sub-pixel alignment. In
practice low-risk (both dates crop to the identical requested bbox at
the same default resolution), but not the same hard guarantee the
explicit pair-alignment path provides elsewhere in this codebase.

## `wind_farm_siting`

**What it does:** A real terrain/land-use suitability screen —
**not** a wind resource assessment.

**How it actually works:** Three real, combined constraints: a real
DEM slope limit via Horn's method (excludes terrain too steep for
turbine foundations/crane access), a real terrain-exposure proxy
(elevation relative to a smoothed regional-mean surface — the same
real technique as `archaeological_site`'s Local Relief Model;
ridgelines and hilltops score higher), and real ESA WorldCover
land-use exclusion (water, built-up, wetland, and snow/ice classes
excluded).

```python
from pygeovision.ai.pipelines.domains import WindFarmSitingPipeline

result = WindFarmSitingPipeline(client).run(
    bbox=(...), output_dir="./output", max_slope_deg=10.0,
)
print(result.stats)
# {"max_slope_deg": 10.0, "dem_type": "srtm1arc",
#  "worldcover_exclusion_applied": True, "pct_area_suitable_by_slope": ...,
#  "pct_area_suitable_overall": ..., "suitable_area_ha": ..., "note": "..."}
```

```{warning}
Corrected from an earlier version of this page: there is no real
`dem_path=` parameter — a DEM is fetched automatically via
OpenTopography (same real requirement as `solar_potential` above).
```

```{note}
A real, honest fallback worth knowing: if the real WorldCover lookup
fails for any reason, land-use exclusion is silently skipped (all
terrain treated as land-use-suitable) rather than failing the whole
pipeline — check the real `worldcover_exclusion_applied` stat to know
whether land-use exclusion actually ran for your result.
```

```{warning}
No real wind-speed or wind-resource data source exists anywhere in
this codebase. "Suitable" here means "not excluded by terrain/land-use
constraints" — it says nothing about whether the wind resource at a
site is actually strong enough to be viable. A real wind resource
assessment (e.g. from a real reanalysis or met-mast dataset) is a
separate, necessary step this pipeline does not perform.
```

## `dust_storm_tracking`

**What it does:** A real, uncalibrated visible-band atmospheric
haze-anomaly proxy — **not** a real Aerosol Optical Depth (AOD),
PM2.5, or PM10 measurement.

**How it actually works:** A real bi-temporal comparison of blue-band
reflectance (dust scatters blue light, increasing apparent
brightness) combined with a real NDVI decrease (dust obscures the
land surface, suppressing the apparent vegetation signal). Cloud
masking is deliberately disabled for this pipeline (`apply_cloud_mask
=False`) — a real, intentional choice, since dust plumes are
frequently misclassified as cloud by standard cloud-masking
algorithms and would otherwise be stripped from the input before the
pipeline ever sees them.

```python
result = DustStormTrackingPipeline(client).run(
    bbox=(...), date_before="2024-06-01", date_after="2024-06-03",
    output_dir="./output",
)
```

```{warning}
Real search was done for a genuine atmospheric-composition data
source before building this pipeline; `pygeofetch` (the real,
installed data dependency) has no Sentinel-5P or similar provider at
all. `air_quality_index` (see [Roadmap](../reference/roadmap.md)) was
deliberately left unimplemented for this same reason, rather than
present this uncalibrated proxy as calibrated air-quality data under
a health-relevant name.
```
