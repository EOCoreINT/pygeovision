# Forestry

Forest questions are usually "how much was lost, where, and when", followed closely by "is that real?". PyGeoVision gives you
two reference-product routes that need no training and no cloud-free luck, a burn-severity route for fire, and a
trained-network route for definitions of your own. Each has a limit that decides which to use.

```{admonition} Know what a "forest" is in each product
:class: warning
A forest/non-forest map is a **definition** applied by someone else's classifier. The JAXA product counts forest as
>90 % canopy by default; the annual land-cover product counts "Trees"; plantations are forest in both. A loss figure
from one is not comparable with a figure from another, or with a national inventory, until you have read the
`product_note` in `run_summary.json`.
```

## 1. How much forest was lost between 2015 and 2020?

```bash
pygeovision channel forest_change \
    --bbox -62.8 -9.4 -62.3 -8.9 \
    --date-before 2016 --date-after 2020 \
    --output ./rondonia_forest
```

**Why this pipeline.** It uses L-band radar (JAXA PALSAR), so cloud is irrelevant, and the result is a published product with
documented classes. It covers **2015-2020 only**.

**Expect.** `forest_change.tif` with 0 non-forest both years, 1 forest both years, 2 loss, 3 gain, 255 nodata; statistics
`forest_loss_ha`, `forest_gain_ha`, `loss_ha_per_year`, `forest_share_before/after`.

**Do next.**

1. **Look at gain next to loss.** Each year is mapped independently, so a stable forest still shows scattered "loss" and
   "gain" pixels where the classifier disagreed with itself. If gain is about as large as loss in an area you know is stable, the
   change is mostly noise; remove small patches:

   ```bash
   pygeovision postprocess sieve ./rondonia_forest/forest_change.tif --min-pixels 16 \
       --output ./rondonia_forest/forest_change_sieved.tif
   ```
2. **Run both definitions** and report the range: once as above, once with `--param include_open_forest=true` (canopy above 10 %).
3. **Per protected area or district**: `postprocess zonal-stats` for a mean of a continuous raster, or
   `postprocess class-stats` for the pixel and area table per class.

**What it cannot say.** Why the forest was lost (logging, fire, farming and dieback look alike), whether it was natural forest or a
plantation, or anything after 2020.

## 2. Loss after 2020, or with class names of your own

Two options, in this order of effort:

| Option | Covers | Needs | Limit |
|---|---|---|---|
| [`landcover_change`](../pipelines/catalog/landcover_change.md) | 2017-2023, annual 10 m land cover, with a **from → to** table | Nothing | Each year classified independently: use `--param min_change_ha=0.5` to suppress speckle |
| [`deforestation`](../pipelines/catalog/deforestation.md)\* | Any two dates, your classes | A network you trained on your own labelled change | You must measure its accuracy yourself |

```bash
pygeovision channel landcover_change \
    --bbox -3.05 5.60 -2.75 5.90 \
    --date-before 2017 --date-after 2023 \
    --param min_change_ha=0.5 \
    --output ./ashanti_lc_change
```

The statistics include `changed_area_ha`, `change_fraction`, `area_ha_before/after` per class, and
`changed_in_patches_of_3px_or_less` — **a noise indicator**: if most changed area is in 3-pixel patches, most of the "change" is
classifier disagreement.

\* `deforestation` is only for the case where the reference products are not enough: it refuses to run without `--checkpoint`.

```bash
pygeovision channel deforestation \
    --bbox -3.05 5.60 -2.75 5.90 \
    --date-before 2020 --date-after 2024 \
    --checkpoint runs/forest_change/best.pth \
    --class-names "No change,Forest loss,Regrowth" \
    --output ./forest_loss
```

## 3. Was it fire? Burn severity

```bash
pygeovision channel wildfire_severity \
    --bbox -122.6 39.7 -121.9 40.2 \
    --date-before 2018-10 --date-after 2018-12 \
    --output ./camp_fire
```

**Expect.** `dnbr.tif` (continuous) and `severity_class.tif`: 0 unburned (dNBR < 0.10), 1 low (0.10-0.27), 2 moderate
(0.27-0.66), 3 high (≥ 0.66), with `burned_area_ha` and the share in each class.

**Choose the dates well.** Pre-fire a few weeks before, post-fire shortly after and, where possible, in the same season: dNBR
compares two dates, and a change in leaf cover between them looks like burn. If a fire went through grass that regrows in a month,
a late post-fire date shows nothing.

To see *when* fire occurred in the weeks you do not have cloud-free imagery for, use `active_fire` (MODIS thermal anomalies, 1 km):

```bash
pygeovision channel active_fire \
    --bbox 29.5 -14.5 31.5 -12.5 --date 2023-09 \
    --param min_confidence=nominal \
    --output ./zambia_fire_2023_09
```

`fire_days.tif` counts days with a detection per 1 km pixel; `observed_days.tif` counts clear days. A pixel observed on few days
may simply have been missed, so compare `fire_days` with `observed_days` before saying "no fire".

## 4. Habitat variety

```bash
pygeovision channel biodiversity_hotspot \
    --bbox -2.45 5.60 -2.05 5.95 \
    --param window_px=21 --param hotspot_percentile=90 \
    --output ./diversity_ghana
```

A local Shannon index of land-cover classes (ESA WorldCover 2021) in a moving window, with the top decile flagged. It
measures **habitat heterogeneity**, a widely used proxy; it is not a species count.

## 5. Canopy height from LiDAR (if you have a point cloud)

The [point-cloud tool](../advanced/pointcloud.md) makes a canopy height model from a `.las`/`.laz` file:
`PointCloudProcessor().canopy_height_model(...)` (see its page for the call). Exclude noise classes (the default) — in the page's test, leaving
them in turned a 19 m canopy maximum into 298 m. There is no pretrained LiDAR classifier in the package.

## Putting it together

| Step | Tool | Output you keep |
|---|---|---|
| Baseline extent | `forest_change` (and `land_cover` for the 2021 map) | `forest_change.tif`, loss/gain table |
| Cross-check | `landcover_change` over the same box | from → to table; compare loss |
| Clean | `postprocess sieve` | sieved raster |
| Per area | `postprocess class-stats`, `zonal-stats` | table per district/park |
| Explain a loss | `wildfire_severity` / `active_fire` for fire; a look at the imagery for the rest | burn raster, or none |
| Publish | `pygeovision report <run-folder>` | PDF/Word/Excel and vectors ([reporting](../reporting/index.md)) |

Report loss as a **range** (the two definitions, sieved and unsieved) and say which product, which definition and which years.
