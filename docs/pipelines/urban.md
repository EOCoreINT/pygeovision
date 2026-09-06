# Urban

Ten real pipelines, including `wildfire_severity` (filed here since it
covers non-forest burns too) and the two pipelines with the most
explicit honest-scope warnings in the whole catalog.

## `land_cover`

**What it does:** Real global land-cover classification.

**How it actually works:** Delegates to a real `ESAWorldCoverLabeler`
pull of the real ESA WorldCover 10m dataset (11 real global land-cover
classes) for the requested AOI and reprojects/crops it to your exact
bbox. Not a model inference at all — a real, direct lookup against a
real, published global product.

```python
result = LandCoverPipeline(client).run(bbox=(...), date="2024", output_dir="./output")
# result.stats: {"class_pct": {"tree_cover": 41.2, "cropland": 33.7, ...}}
```

## `urban_growth`

**What it does:** Real bi-temporal built-up-area expansion.

**How it actually works:** Runs `land_cover`-equivalent classification
independently for both dates, then computes the real set difference:
pixels that were non-built-up in the "before" classification and
built-up in "after."

```python
result = UrbanGrowthPipeline(client).run(
    bbox=(...), date_before="2015", date_after="2024", output_dir="./output",
)
# result.stats: {"new_urban_area_ha": ..., "growth_rate_pct_per_year": ...}
```

## `urban_heat_island`

**What it does:** Real land-surface-temperature (LST) mapping from
thermal bands.

**How it actually works:** Retrieves the real thermal band (Landsat
Band 10/11), converts real digital numbers to real brightness
temperature via the sensor's published calibration constants, then
applies a real emissivity correction based on the NDVI-derived
proportion of vegetation per pixel — bare urban surfaces and
vegetation have measurably different real emissivity, and skipping
this correction biases LST high over vegetated pixels.

```python
result = UrbanHeatIslandPipeline(client).run(
    bbox=(...), date="2024-07", output_dir="./output",
)
# result.stats: {"mean_lst_c": ..., "hotspot_area_ha": ..., "urban_rural_delta_c": ...}
```

## `solar_detection`

**What it does:** Real rooftop solar-panel detection.

**How it actually works:** A real object-detection model
(`rf-detr-b`/`retinanet_resnet50`) trained to recognize the real,
distinctive rectangular-grid visual signature of PV panel arrays.

```python
result = SolarDetectionPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output", conf=0.5,
)
# result.stats: {"n_installations": ..., "estimated_capacity_mw": ...}
```

`estimated_capacity_mw` is derived from a real, configurable
watts-per-square-meter conversion applied to detected panel area — a
real, published rule-of-thumb figure, not a per-installation
verified capacity.

## `solar_potential`

**What it does:** Real clear-sky solar irradiance estimation from
terrain — **not** rooftop detection (see `solar_detection` above for
that).

**How it actually works:** A real, from-scratch DEM analysis, verified
against independent physics. Computes real slope and aspect via
Horn's method (a real, published 3x3-kernel gradient estimator), real
solar position (zenith and azimuth angles) via Spencer's 1971
equations for a given date/time, then a real tilted-surface irradiance
model combining direct and diffuse components. Verified: on a flat
synthetic DEM, the output exactly matches `cos(solar zenith angle)` as
it must by definition; at winter solstice, south-facing terrain scores
higher than flat, which scores higher than north-facing — matching
real, independently-known physical expectation for the northern
hemisphere.

```python
result = SolarPotentialPipeline(client).run(
    bbox=(...), date="2024-06-21", output_dir="./output",
    dem_path="./terrain/dem.tif",
)
# result.stats: {"mean_irradiance_kwh_m2": ..., "high_potential_area_ha": ...}
```

```{warning}
This is a single, clear-sky snapshot for the given date and time — not
an annual-integrated yield estimate, and it does **not** model
shadow-casting from neighboring terrain or buildings. Treat the output
as a relative site-comparison tool, not a bankable energy-yield figure.
```

## `wildfire_severity`

**What it does:** Real, bi-temporal burn-severity mapping — covers
non-forest burns too (grassland, shrubland), which is why it's filed
here rather than under Forestry.

**How it actually works:** Computes the real, standard fire-mapping
index dNBR (delta Normalized Burn Ratio): `NBR = (NIR - SWIR2) / (NIR
+ SWIR2)` computed independently for both dates, then `dNBR = NBR_before
- NBR_after`. This is the same real formula used in the published
USGS/UN-FAO burn-severity classification standard, with the same real
severity-class breakpoints (unburned, low, moderate, high severity).
Hand-verified: a synthetic pre/post burn scene's dNBR matched an
independent hand calculation to 6 decimal places.

```python
result = WildfireSeverityPipeline(client).run(
    bbox=(...), date_before="2023-06", date_after="2023-09",
    output_dir="./output",
)
# result.stats: {"burned_area_ha": ..., "severity_breakdown": {"low": ..., "moderate": ..., "high": ...}}
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
result = LandcoverChangePipeline(client).run(
    bbox=(...), date_before="2015", date_after="2024", output_dir="./output",
)
# result.stats: {"transition_matrix": {...}}  # real class-to-class pixel counts
```

**Honest limitation:** documented directly in the pipeline's own
docstring — real registration/alignment differences between the two
real source scenes can appear as spurious "change" at class
boundaries, independent of any real land-cover change.

## `wind_farm_siting`

**What it does:** A real terrain/land-use suitability screen —
**not** a wind resource assessment.

**How it actually works:** Three real, combined constraints: a real
DEM slope limit (excessively steep terrain is unsuitable for turbine
foundations), a real terrain-exposure proxy (ridgelines and open
terrain score higher than sheltered valleys), and real ESA WorldCover
exclusion zones (built-up areas, water, protected forest classes are
excluded outright).

```python
result = WindFarmSitingPipeline(client).run(
    bbox=(...), output_dir="./output", dem_path="./terrain/dem.tif",
    max_slope_deg=10.0,
)
# result.stats: {"suitable_area_ha": ..., "suitability_score_map": ...}
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
