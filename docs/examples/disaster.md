# Disaster response

After an event the first question is "what changed and where", the second is "how sure are you", and the third, from whoever
reads the map, is "how many people". PyGeoVision answers the first two with documented methods and says clearly what it cannot do. It does
**not** estimate damage to buildings unless you bring a trained network, and it does not give depth, duration or casualties.

```{admonition} Rapid-response ground rules
:class: important
1. **Give date ranges that contain imagery.** A single post-event day usually contains no satellite pass. Radar needs a 10-14 day window; optical needs a month or a clear range.
2. **Report a range and a coverage.** Every pipeline here states `valid_fraction` (how much of the box could be assessed). The area of the box that nothing could see is *unknown*, not unaffected.
3. **A screening map is not an assessment.** Name the method and the dates on every map you hand out.
```

## Which tool for which hazard

| Hazard | Ready pipeline (no training) | Needs a network you trained | What the ready one cannot do |
|---|---|---|---|
| Flood | `flood_mapping`, `sar_flood_mapping`, `multisensor_flood_mapping` | `disaster_assessment` for damage classes | Depth, duration, flooding between buildings or under dense vegetation |
| Wildfire | `wildfire_severity` (dNBR), `active_fire` (MODIS, 1 km) | none | Fire cause, structures lost; `active_fire` misses small or short fires |
| Landslide | `change_detection` as a first look | `landslide_detection` | The ready route says "spectrally different", not "landslide" |
| Earthquake, conflict, any structural damage | `change_detection` as a first look | `disaster_assessment` | Damage grading needs a model trained on damage labels |
| Volcano | `volcano_monitoring` (Landsat thermal candidates) | none | Confirm activity; it lists candidates |

## Workflow A: a flood, from first look to a number you can defend

1. **First look** (hours): the radar pipeline, because it ignores cloud.

   ```bash
   pygeovision channel sar_flood_mapping \
       --bbox -0.55 5.45 -0.15 5.75 \
       --date-before 2024-01-05/2024-01-31 --date-after 2024-06-10/2024-06-25 \
       --output ./flood_sar
   ```
2. **Cross-check** (when a clear optical view exists): `multisensor_flood_mapping` with the same windows. Compare
   `flooded_confirmed_ha` with `flooded_upper_ha`. A wide gap tells the reader how uncertain the figure is.
3. **Clean and cut**: remove specks, then count and outline. (Shown for the radar run, where class 1 is flooded; in the multi-sensor raster class 1 is only the radar-and-optical *confirmed* part, so pick the classes you want to outline deliberately.)

   ```bash
   pygeovision postprocess sieve ./flood_sar/flood_extent.tif --min-pixels 10 --output ./flood_sar/flood_extent_clean.tif
   pygeovision postprocess class-stats ./flood_sar/flood_extent_clean.tif
   pygeovision postprocess vectorise ./flood_sar/flood_extent_clean.tif --output flood.geojson --target-class 1 --min-area 1000
   ```
4. **Hand over**: `pygeovision report ./flood_multi --formats pdf,docx --vector-formats gpkg,kml`, which rebuilds the map, vectors and reports without
   re-downloading ([rebuilding reports](../reporting/rebuilding.md)).

For a long-form version of this with training and an independent accuracy sample, see the
[Volta flood case study](water.md#case-study-the-2023-volta-floods).

## Workflow B: a wildfire

```bash
pygeovision channel wildfire_severity \
    --bbox -122.6 39.7 -121.9 40.2 \
    --date-before 2018-10 --date-after 2018-12 \
    --output ./camp_fire

pygeovision channel active_fire \
    --bbox -122.6 39.7 -121.9 40.2 \
    --date 2018-11 --output ./camp_fire_hotspots
```

`wildfire_severity` gives `dnbr.tif` and four classes (unburned < 0.10, low, moderate 0.27-0.66, high ≥ 0.66) with `burned_area_ha`.
`active_fire` gives days with a detection per 1 km pixel and the strongest fire radiative power. They answer different questions:
*how much burned* (needs two clear scenes) and *when was it burning* (does not).

**When to doubt the result.** dNBR shows any drop in near-infrared and rise in short-wave infrared. Harvested fields,
flooded land and drying grass can resemble low severity. Treat class 1 (low) with the most scepticism and class 3 (high) the least.

## Workflow C: damage after an earthquake, landslide or conflict

There is no ready damage pipeline, on purpose: a damage classifier without trained weights produces confident random output. The route is:

1. **Locate change** with `change_detection` (change vector analysis, Otsu threshold, no training):

   ```bash
   pygeovision channel change_detection \
       --bbox 36.1 36.9 36.4 37.2 \
       --date-before 2023-01 --date-after 2023-02 \
       --param min_magnitude=0.05 --output ./change
   ```
   Read `change_magnitude.tif` and `changed_area_ha`. Use dates in the **same season**: seasonal vegetation and sun angle otherwise dominate.
2. **Interpret** with the imagery. Change says "spectrally different", not "collapsed".
3. **Grade damage** only with your own trained network:

   ```bash
   pygeovision channel disaster_assessment \
       --bbox 36.1 36.9 36.4 37.2 \
       --date-before 2023-01 --date-after 2023-02 \
       --param disaster_type=earthquake \
       --checkpoint runs/damage/best.pth \
       --class-names "No damage,Minor,Major,Destroyed" \
       --output ./damage
   ```
   Before you rely on it, measure it on your own reference points
   (`pygeovision postprocess accuracy damage.tif reference.tif`). Sentinel-2 at 10 m cannot grade damage to individual
   buildings; high-resolution imagery is what such a network needs.

## Workflow D: volcanic heat

```bash
pygeovision channel volcano_monitoring \
    --bbox 15.0 37.65 15.1 37.80 --date 2024-07 \
    --param threshold_c=15 --output ./etna_thermal
```

A pixel is a candidate when its land surface temperature is `threshold_c` above the scene mean. **Candidates are not confirmed activity**: sun-heated
bare rock, industrial sites and fires pass the same test. Use it to decide where to look, not to announce an eruption.

## What to do when…

| Situation | Action |
|---|---|
| Nothing but "no data" | The window has no pass or no clear view. Widen it; read the error, which lists the nearest acquisitions |
| The map shows a "flood" on a hillside | Use `multisensor_flood_mapping` (terrain filter) or check radar shadow |
| `valid_fraction` is 0.4 | State that 60 % of the box was not observed; do not report "no damage" there |
| Two pipelines disagree | Report both. Disagreement is the uncertainty |
| Someone asks for casualties or displaced people | This is outside the tool. Exposure needs a population layer (the Volta project uses WorldPop) and is not a casualty count |
