# Water and floods

Water is where PyGeoVision has the most tools, and where picking the wrong one gives the most convincing wrong map. The sections
below follow the questions in the order projects ask them: **where is the water now**, **how did it change over a year or
four decades**, **what flooded in one event**, and **where could it flood in principle**.

```{admonition} Pick the sensor for the weather, not the other way round
:class: important
Optical water indices fail under cloud, and the cloudiest days are often the flood days. Radar sees through cloud but is
blind to floodwater among buildings and under dense vegetation. The three flood pipelines exist because of this; the
[comparison table](../pipelines/choosing.md) says which to choose.
```

## 1. Where is the water now?

```bash
pygeovision channel water_bodies \
    --bbox -0.15 51.47 -0.10 51.52 \
    --date 2024-06 \
    --param ndwi_threshold=0.2 \
    --output ./london_water
```

**Expect.** `water_mask.tif` (0 not water, 1 water, 255 no data), `ndwi.tif`, and `water_coverage` (share of *valid* pixels) with
`valid_fraction`. NDWI (green vs NIR) separates open water well and confuses **shadow, dark roofs and wet asphalt** with it.

**When it looks wrong:** water coverage jumps up in a hilly or built-up scene → look at the mask over steep north-facing slopes and
city centres; raise `ndwi_threshold`, or switch to the multi-sensor pipeline which uses terrain to reject impossible water.

## 2. How did a lake or reservoir change through the year?

```bash
pygeovision channel water_extent_timeseries \
    --bbox 0.0 6.0 0.5 6.5 \
    --dates 2024-01,2024-03,2024-05,2024-07,2024-09,2024-11 \
    --output ./volta_lake_2024
```

**Expect.** Area per date (`min/max_water_area_ha`, `always_water_ha`, `ever_water_ha`), `water_frequency.tif` (share of
cloud-free observations that were water) and `valid_observations.tif`. Dates with too little clear coverage are dropped (see
`n_dates_usable` against `n_dates_requested`, and `min_valid_fraction`).

**Read the frequency map with the observation count.** A pixel that is water on 2 of 2 clear looks 100 % wet; look at
`valid_observations.tif` before treating frequency as a property of the place.

## 3. What has water done over four decades?

```bash
pygeovision channel surface_water_dynamics \
    --bbox -0.30 6.20 0.30 6.60 \
    --output ./volta_water_history
```

JRC Global Surface Water (Landsat, 1984-2020) gives `water_transitions.tif` (permanent, new, lost, seasonal …),
`water_occurrence.tif` (% of observations) and `water_seasonality.tif`. It is a published product, so it needs
no imagery search, and it ends in 2020: to see *recent* change, compare with `water_bodies` for the present year and name the product
change in your caption.

## 4. What flooded in this event?

Decide with three questions:

1. **Was the sky clear on the flood day?** Yes → `flood_mapping` (optical, simple, explainable). No → radar.
2. **Do you need a number you can defend?** → `multisensor_flood_mapping`: it keeps each sensor's evidence visible and gives a lower bound
   (confirmed) and an upper bound (including disputed).
3. **Is the area hilly?** The multi-sensor pipeline removes flooding on ground more than `max_hand_m` above the nearest channel.

```bash
# optical, clear day
pygeovision channel flood_mapping \
    --bbox -0.70 5.20 -0.20 5.60 \
    --date-before 2024-02 --date-after 2024-06 \
    --param cloud_cover_max=30 \
    --output ./flood_optical

# radar, any weather — give RANGES that contain a Sentinel-1 pass
pygeovision channel sar_flood_mapping \
    --bbox -0.55 5.45 -0.15 5.75 \
    --date-before 2024-01-05/2024-01-31 --date-after 2024-06-10/2024-06-25 \
    --output ./flood_sar

# radar + optical + terrain, with agreement classes
pygeovision channel multisensor_flood_mapping \
    --bbox -0.55 5.45 -0.15 5.75 \
    --date-before 2024-01-05/2024-01-31 --date-after 2024-06-10/2024-06-25 \
    --output ./flood_multi
```

**Reading the multi-sensor result.** `flood_extent.tif` classes tell you how much to trust each pixel:

| Class | Meaning | Treat as |
|---|---|---|
| Flooded: radar + optical | Both sensors agree | Confirmed (lower bound) |
| Radar only | Radar says flooded and optical had no clear view | Probable; this is the cloud case |
| Optical only | Optical says flooded and radar did not flag | Check: shallow or vegetated flooding |
| Disputed | Radar flooded, a clear optical view says dry | Check first: shadow, smooth surfaces, wind |
| Water in reference window | Already water before the event | Not flood |

The headline `flooded_area_ha` is classes 1+2+3; `flooded_confirmed_ha` is the lower bound; `flooded_upper_ha` adds the disputed class.
Report all three, plus `optical_coverage` (how much of the flood window optical imagery could see).

```{admonition} If you keep getting "no data" in the wet season
:class: tip
That is nearly always the date window, not the code. A Sentinel-1 window must contain a pass (6-12 day repeat, so give 10-14 days),
and an optical window must contain a clear view. Give ranges, not single days; see the error that lists the nearest acquisitions, and
the [troubleshooting page](../getting-started/troubleshooting.md).
```

**What none of them measure:** depth, duration, damage, floods that receded before the satellite passed, or water between buildings.

## 5. Where could it flood in principle?

```bash
pygeovision channel drainage_analysis \
    --bbox -0.35 5.55 -0.10 5.75 \
    --param stream_threshold_km2=2.0 \
    --param hand_thresholds_m=5,10,15 \
    --output ./accra_drainage
```

`hand_m.tif` (height above the nearest drainage), `stream_order.tif` and `flow_accumulation_km2.tif`. Low HAND means ground close to a
channel; it is a **terrain** statement, not a flood prediction. The drainage density depends strongly on your `stream_threshold_km2`, so quote the value.

## 6. Quality and wetlands

| Question | Pipeline | Caution |
|---|---|---|
| Is the water greener or muddier? | `water_quality` (NDCI chlorophyll, red-reflectance turbidity) | Relative indicators; not concentrations without in-situ calibration |
| Open water vs flooded vegetation | `wetland_mapping` (MNDWI + EVI) | A season-dependent map; compare wet and dry dates |
| Is the shoreline moving? | `coastal_monitoring` | Tide state differs between scenes; compare the same tidal stage or say you did not |

## Case study: the 2023 Volta floods

`case_studies/ghana_volta_flood/` in the repository is a complete project, outside the 41 pipelines, that maps the 2023
Akosombo-spillage floods in the Lower Volta: Sentinel-2 epochs → spectral "teacher" labels → block-split chips → fine-tuned Prithvi-EO
→ noisy-student round → wall-to-wall maps → **independent accuracy sample** → district exposure → report.

It is the pattern to copy when a pipeline is not enough. Its important choices:

- **Labels are automatic and noisy, so the model is never judged against them.** It is judged on a stratified random sample that a human
  labels without seeing the map (Olofsson et al. 2014 estimators), which gives an *area-adjusted* flood area with a confidence interval.
- **Chips are split by geographic block** with a buffer, so train and test never share neighbouring pixels.
- **Competing maps are scored on the same points**: the first model, a U-Net baseline, the teacher, and a naive "water now, not before"
  map, so "does the foundation model earn its keep" is a measurement.

```{admonition} What has and has not been verified
:class: warning
The project's 65 tests check every stage and the estimators on a **synthetic** floodplain with known truth. **No accuracy figure for real
imagery has been produced here and none should be quoted** before someone runs it and labels the validation sample. The README
lists the known real-data risk (dark water flagged as cloud shadow) and the limits (optical cannot see under vegetation or cloud, so
the mapped area is a lower bound).
```

For a flood *without* a GPU or training, the `multisensor_flood_mapping` pipeline above is the supported route.
