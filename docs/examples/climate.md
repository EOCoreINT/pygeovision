# Climate and cryosphere

Climate questions are about **slow change** and **comparability**. A single scene can show you something; only the same measurement
under the same conditions, repeated, can show a trend. The pipelines below are chosen because each has a clear rule for what to
hold constant. If you cannot hold it constant, the honest answer is "not comparable".

```{admonition} What this tool is not
:class: warning
It is not a climate model, a carbon accountant or a long-term trend service. A "carbon estimation" pipeline (carbon = 50 × NDVI² × 0.47) was
[removed](../pipelines/removed.md): NDVI saturates in dense vegetation, so the numbers were not carbon. Use a biomass product or field calibration for carbon.
```

## 1. Is the glacier shrinking?

```bash
pygeovision channel glacier_monitoring \
    --bbox 7.8 46.4 8.3 46.7 \
    --date-before 2018-09 --date-after 2024-09 \
    --output ./aletsch_ice
```

**Why NDSI, and why September.** The Normalised Difference Snow Index separates snow and ice (bright in green, dark in short-wave
infrared) from most other surfaces. Compare **end-of-melt-season** dates (September in the northern hemisphere), because snow cover otherwise
moves with the weather of the week, not with the glacier.

**Expect.** `glacier_extent_after.tif` (snow/ice mask on the later date), `area_ha_before/after`, `area_change_ha`, `area_change_pct`, computed
on commonly cloud-free pixels, and `valid_fraction`.

**Cautions.** Debris-covered ice is not bright and is missed; fresh snow on a bare glacier hides it. A change between *two* dates is a
difference in two seasons' weather as much as in the ice. Use the same month in several years, or a longer record, before you call a trend.

```bash
pygeovision channel snow_cover \
    --bbox 7.8 46.4 8.3 46.7 --date 2024-03 \
    --output ./alps_snow_march
```

`snow_cover` gives snow extent (NDSI > 0.4) for one month: a clear-sky composite of the clearest scene, not an average.

## 2. How hot is the surface?

```bash
pygeovision channel land_surface_temperature \
    --bbox 16.25 48.15 16.50 48.30 \
    --date 2024-07 \
    --param threshold_c=5 \
    --output ./vienna_lst
```

Landsat Collection 2 Level-2 surface temperature (°C) with a mask of pixels more than `threshold_c` above the scene mean. This is
**surface** temperature at the satellite's overpass (late morning): cooler than a rooftop at noon, warmer than air at dawn. Compare
the same month in different years, with similar weather, and say so.

## 3. How much water does the land return?

```bash
pygeovision channel evapotranspiration \
    --bbox -2.0 6.0 -1.0 7.0 --date 2022 \
    --output ./ashanti_et_2022
```

Annual MODIS evapotranspiration, potential evapotranspiration and their ratio (500 m). One year per run: to look at change, run several
years and compare `mean_et_mm` and `mean_et_pet_ratio`, keeping the box identical. At 500 m, small fields and water bodies are mixed
inside a pixel.

## 4. Is vegetation unusually dry or green?

`vegetation_anomaly` (a month against the same month of earlier years, z-score and Vegetation Condition Index) is the drought-monitoring workflow;
see [agriculture](agriculture.md#1-is-this-month-greener-or-browner-than-usual). For a single location or region over many dates, the
[time-series tool](../advanced/timeseries.md) fits a trend with a p-value and, with `period=12`, takes the seasonal cycle out first so a
regular seasonal drop is not flagged as an anomaly.

## 5. Where is the water or shoreline moving?

| Question | Pipeline | Hold constant |
|---|---|---|
| Lakes, rivers, reservoirs over 1984-2020 | `surface_water_dynamics` | n/a (published product, ends 2020) |
| Water area through a year | `water_extent_timeseries` | Cloud-free coverage per date |
| Shoreline retreat or advance | `coastal_monitoring` | **Tidal state**: a tide difference between the two dates looks like change |

```bash
pygeovision channel coastal_monitoring \
    --bbox 0.50 5.70 1.00 5.95 \
    --date-before 2018-01 --date-after 2024-01 \
    --output ./volta_coast
```

`shoreline_change.tif` classes: 0 land both dates, 1 water both dates, 2 land to water, 3 water to land; `net_water_change_ha` is the balance.
At 10-30 m resolution this resolves hectares, not metres of shoreline.

## 6. Solar resource on terrain

```bash
pygeovision channel solar_potential \
    --bbox 7.8 46.4 8.3 46.7 \
    --param date=2024-12-21 --param hour=12 \
    --output ./alps_winter_sun
```

Relative clear-sky irradiance (0-1) from slope and aspect for one date and hour. It is **not annual yield**: it ignores cloud and atmosphere, and one date
and hour is a snapshot. It needs an OpenTopography key for the DEM (`pygeovision data auth add opentopography`).

## A pattern for any trend question

1. State the quantity and the **conditions you hold constant** (season, tide, sensor, cloud-free).
2. Use the same box, the same pipeline, the same parameters for every date (check `run_summary.json`).
3. Collect at least five dates before you fit anything; the [time-series tool](../advanced/timeseries.md) refuses fewer than five for anomalies.
4. Report the p-value *and* the slope per year with its units, and say how much of the image was valid on each date.
