# Agriculture

Three questions come up again and again: **is this season greener or browner than usual**, **which parts dropped between two
dates**, and **how much water did the land give back to the air**. They need different tools, and mixing them up is the usual
source of a wrong conclusion. This page walks through them in the order a field-monitoring project normally meets them.

```{admonition} What PyGeoVision does not do here
:class: warning
There is no yield forecast and no crop-type map unless you train one. A "yield forecast" pipeline was
[removed](../pipelines/removed.md) because it only rescaled mean NDVI; `crop_monitoring` runs *your* trained network and refuses
to start without `--checkpoint`. NDVI measures greenness. It is not biomass, and it saturates in dense canopy.
```

## 1. Is this month greener or browner than usual?

**Why.** A raw NDVI value means little: a Sahel pasture in August is naturally greener than in March. What matters is how
the value compares with the same month in earlier years.

```bash
pygeovision channel vegetation_anomaly \
    --bbox -1.20 9.20 -0.90 9.50 \
    --date 2024-08 \
    --param baseline_years=5 \
    --output ./tamale_anomaly_2024_08
```

**Expect.** Three continuous rasters (`ndvi_anomaly.tif`, `ndvi_zscore.tif`, `vci.tif`) and one class raster
(`anomaly_class.tif`: 0 much below normal … 4 much above). `run_summary.json` lists `baseline_years_used` and
`baseline_years_missing`. At least three baseline years must be usable, otherwise the run fails with a message rather than comparing
against too little history.

**Do next.**

- Open `anomaly_class.tif` and look for **patches**, not isolated pixels. A drought or a late planting forms patches that follow
  field boundaries; speckle is noise.
- Take the field polygons and ask which fields are worst. This is a zonal statistic, not a new analysis:

  ```bash
  pygeovision postprocess zonal-stats ./tamale_anomaly_2024_08/ndvi_zscore.tif fields.geojson \
      --output fields_zscore.geojson --stats mean,min,count
  ```

  Each field gets `stat_mean`, `stat_min` and `stat_count`. A field with no valid pixel gets `null`, not zero, so cloud-covered fields
  are visible in the table.
- Check `baseline_years_used`. A baseline of three wet years followed by a dry target year over-states the anomaly.

**When it looks wrong.**

| You see | Likely cause | What to do |
|---|---|---|
| "Much below normal" over a whole irrigated scheme | The baseline years were rain-fed (or the reverse) | Choose the baseline with `baseline_years=2019,2020,2021` to match the management regime |
| Large `valid_fraction` loss | Cloud in the target month | Use a drier month or widen the window; do not interpret the remaining pixels as the whole area |

## 2. How does the season develop?

**Why.** To see the green-up, peak and senescence, you need several dates in one year, not an anomaly.

```bash
pygeovision channel vegetation_indices \
    --bbox -1.20 9.20 -0.90 9.50 \
    --dates 2024-04,2024-06,2024-08,2024-10 \
    --output ./tamale_season
```

**Expect.** For each date, NDVI, EVI and NDWI, and `result.stats["per_date"]` with `mean_ndvi`, `mean_evi`, `mean_ndwi` and
`valid_fraction` for every date. Dates without usable imagery are skipped and listed (`n_dates_used` counts the others).

For a statistical view of the same question (trend, flagged odd dates), take the per-date images into the
[time-series tool](../timeseries/index.md):

```bash
pygeovision timeseries analyze scene_2024_04.tif scene_2024_06.tif scene_2024_08.tif scene_2024_10.tif \
    --index ndvi --sensor sentinel2 --dates 2024-04,2024-06,2024-08,2024-10
```

One mean per date for the whole image is the output. It is a *field-scale* summary; use the pipeline above when you need the map.

## 3. Which areas dropped between two dates?

**Why.** After a storm, a pest outbreak or a harvest, you want the places where greenness fell by more than normal variation.

```bash
pygeovision channel crop_health \
    --bbox 0.20 6.70 0.45 6.95 \
    --date-before 2024-06 --date-after 2024-08 \
    --param anomaly_threshold=0.15 \
    --output ./crop_stress
```

**Expect.** `ndvi_change.tif` (later minus earlier) and `crop_health_anomaly.tif` (1 where the drop exceeded 0.15). Statistics
include `pct_flagged`, `mean_ndvi_before`, `mean_ndvi_after` and `valid_fraction`.

```{admonition} Read the flag with the calendar in mind
:class: caution
A drop between June and August is the normal course of **harvest** in many systems. The pipeline cannot tell stress from harvest, flooding,
or a fallow field; it reports that NDVI fell. Use `vegetation_anomaly` (same month in other years) to separate "unusual" from "seasonal".
```

## 4. How much water did the land return?

```bash
pygeovision channel evapotranspiration \
    --bbox -2.0 6.0 -1.0 7.0 \
    --date 2022 \
    --output ./ashanti_et_2022
```

**Expect.** Annual evapotranspiration (`et_mm_per_year.tif`), potential evapotranspiration (`pet_mm_per_year.tif`) and their
ratio (`et_pet_ratio.tif`), from the MODIS product at 500 m. A ratio close to 1 means water was not limiting. At 500 m this is a
catchment or district indicator; do not read individual fields from it.

## 5. Which crop is where?

Only with a network you trained, on your own labelled fields:

```bash
pygeovision channel crop_monitoring \
    --bbox 0.20 6.70 0.45 6.95 --date 2023-06 \
    --checkpoint runs/crops/best.pth \
    --class-names Background,Maize,Cassava,Rice \
    --output ./crop_map
```

The route to the checkpoint is [label](../labeling/index.md) → [train](../training/index.md) → [validate](../obia/accuracy.md).
Reference land cover from the [`land_cover`](../pipelines/catalog/land_cover.md) pipeline separates *cropland* from other classes
without any training and is a reasonable first mask for a crop model. It does not distinguish crops.

For a **field-based** approach (segment parcels first, then classify each parcel by its mean spectra and texture)
see [object-based analysis](../obia/index.md), which needs fewer labels than a pixel network.

## Checklist before you report a number

- [ ] `valid_fraction` and `data_notes` read.
- [ ] Baseline years and dates are the ones you intended (in `run_summary.json`).
- [ ] Anomalies were confirmed on at least one cloud-free image you looked at yourself.
- [ ] You said "NDVI fell", not "yield fell".
