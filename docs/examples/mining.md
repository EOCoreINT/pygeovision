# Mining and land rehabilitation

A mine changes the land twice: when it is opened and when it is closed. The first change is easy to see from space. The second
is the one that ends up in a closure plan: **is the disturbed ground coming back, how much of it, and is it still improving?**
[`mine_rehabilitation`](../pipelines/catalog/mine_rehabilitation.md) answers that for a site you outline, against an undisturbed
reference, over several years. The other pipelines on this page help to explain what it shows.

```{admonition} What this is, and what it is not
:class: warning
The pipeline measures **greenness relative to a reference**. That is evidence of revegetation, not proof of a sound closure. A weed
cover or a single-species plantation is as green as the forest it replaced, and the pipeline sees neither soil, slope stability,
water chemistry nor species. Completion criteria are set by regulators and are normally checked in the field. Use the maps to
decide **where to look**, and use the field to decide whether the ground meets the criteria.
```

## 1. Outline the site and the reference

You need one or two polygon files (GeoJSON, shapefile or GeoPackage, in any coordinate system, with the coordinate system
defined in the file):

| File | What it holds | Where it comes from |
|---|---|---|
| `site` | The disturbed or rehabilitated ground to assess: a tailings dump, a backfilled pit, a spoil heap, a whole concession | Your closure plan or licence boundary, or your own digitising on recent imagery |
| `reference` | Undisturbed land of the **type the closure plan aims for**: the forest or savanna next to the site | Your own digitising; keep it away from roads, villages and farms |

**No site polygon yet?** Map the disturbed ground from imagery with object-based analysis: segment a Sentinel-2 scene, classify the objects
(forest, farmland, built-up, mine-disturbed, water) from weak labels, and export the mine-disturbed objects as a GeoJSON. The
Obuasi notebook (`obuasi_mining_landcover_obia_pygeovision.ipynb`) does this and checks that the file loads as a `site`; the method and
its limits are in [OBIA on real imagery](../obia/real-data.md). The result is a first draft: active pits and the plant are disturbed
but not rehabilitation targets, so edit it against high-resolution imagery before use.

The reference decides what "recovered" means, so choose it as carefully as the site. If you give only the `site`, the pipeline
picks persistently green land outside the site and a margin of 100 m; it tells you so in `reference_source`, and
`reference_mask.tif` shows which pixels it chose. Check that map before you trust the result.

## 2. Run it

```bash
pygeovision channel mine_rehabilitation \
    --bbox -2.20 6.00 -2.05 6.12 \
    --dates 2020-02,2021-02,2022-02,2023-02,2024-02 \
    --param site=./tailings_dump_3.geojson \
    --param reference=./intact_forest.geojson \
    --output ./dump3_rehabilitation
```

**Why these dates.** One date per year, in the **same month**, in the season when the reference is green and the sky is
clearest. Different seasons are not comparable even after the ratio to the reference; the run warns you
(`season_warning`) when the dates are more than 75 days apart within the year. Five or more years show a trend; three is the
minimum. Cloudy years are left out and listed rather than guessed.

**Expect.** `rehab_status.tif` with 0 outside the assessed area, 1 recovered, 2 recovering, 3 stalled, 4 bare or sparse,
5 declining, 6 water and 255 not enough clear observations; `recovery_index.tif` (the recent ratio to the reference),
`recovery_trend.tif` (change per year), `reference_mask.tif` and `rehabilitation_timeseries.csv` (the site's median ratio per
date), plus the map, vectors and the PDF/Word/Excel report.

## 3. Read it in this order

1. **The series first.** Open `rehabilitation_timeseries.csv` or look at `site_trend_per_year`. A median ratio that climbs from
   0.2 to 0.6 over five years is a site that is recovering; one that sits near 0.5 for three years has stalled. This number
   averages thousands of pixels, so it is far steadier than any single pixel.
2. **Then the classes.** `recovered_fraction`, `stalled_fraction`, `bare_fraction` and `declining_fraction` say how the
   classed land divides. `recovered_but_declining_ha` is land that has reached the recovered level but is falling again.
3. **Then the places.** Open `rehab_status.tif` over the imagery. Red (declining) and brown (bare) *patches* are where to send
   someone: erosion gullies, compaction, failed topsoil and acid or saline spoil look the same from orbit and only the field
   tells them apart. A lone red pixel is noise.
4. **Blue is water, not failure.** Pit lakes and tailings ponds are the water class and are left out of the percentages, so
   they are never reported as bare ground.

Per-pixel classes come from a handful of dates, so edges flicker. Clean them before counting patches or hectares per parcel:

```bash
pygeovision postprocess sieve ./dump3_rehabilitation/rehab_status.tif --min-pixels 8 \
    --output ./dump3_rehabilitation/rehab_status_sieved.tif
pygeovision postprocess class-stats ./dump3_rehabilitation/rehab_status_sieved.tif
```

## 4. Is a decline real?

A falling ratio has several explanations that are not the site. Check them before you write "declining":

| Question | Tool | What to look at |
|---|---|---|
| Was it a dry year for the whole region? | [`vegetation_anomaly`](../pipelines/catalog/vegetation_anomaly.md) for the same month, over the same box | A strongly negative z-score over the **reference** as well as the site means weather, not the site |
| Is the reference itself dormant? | `reference_ndvi_median` in the CSV | Below about 0.3 the ratio is unstable; the run drops dates below 0.1 and says so |
| Did the pond or pit lake grow into the revegetated edge? | [`water_extent_timeseries`](../pipelines/catalog/water_extent_timeseries.md) over the dump | Rising water area on the dates the edge declined |
| Is something new happening beside the site? | [`landcover_change`](../pipelines/catalog/landcover_change.md) (2017 to 2023) over a wider box | New bare ground or built area at the boundary |

## 5. Putting it together

| Step | Tool | Output you keep |
|---|---|---|
| Outline site and reference | Your closure plan, a GIS | Two polygon files |
| Annual run | `mine_rehabilitation` with one more date each year | `rehabilitation_timeseries.csv`, `rehab_status.tif` |
| Clean and count | `postprocess sieve`, `class-stats` | Area per class, per parcel |
| Explain a decline | `vegetation_anomaly`, `water_extent_timeseries`, `landcover_change` | A short note per patch |
| Verify | A field visit to the declining and bare patches | Photographs and plot data (the evidence a regulator accepts) |
| Report | `pygeovision report ./dump3_rehabilitation` | PDF, Word, Excel and vectors ([reporting](../reporting/index.md)) |

Report the **reference used**, the **dates**, the **thresholds** (`recovered_rri`, `bare_rri`, `trend_threshold` are in
`run_summary.json`) and the share of the site that was actually observed (`valid_fraction`). The pipeline does not say when a
site will meet its criteria, and a figure of "years to closure" computed from a few dates would not be a measurement.

## When it does not run

| You see | Meaning and fix |
|---|---|
| "Give `site` ... or `reference`" | Nothing tells the pipeline which ground to assess. Add `--param site=file.geojson`. |
| "Only N reference pixels were found" | The automatic reference found too little green land. Widen the box, lower `reference_min_ndvi` (savanna and dry woodland rarely reach 0.5), or draw a `reference`. |
| "do not overlap the imagery" | The polygon is outside the box, or its coordinate system is wrong. A file with no coordinate system is refused. |
| Several dates under `dates_left_out` | Cloud over the site, or too few reference pixels. Choose another scene for that year; three usable dates are enough to run. |
