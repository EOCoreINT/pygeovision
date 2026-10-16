# Choosing a pipeline

Start from your **question**, not from the data. This page maps questions to pipelines, then compares the pipelines that look
alike (the flood and water families, the change pipelines, the land-cover products), because choosing between them is where most
mistakes happen. Each pipeline name links to its page, which says in more detail what it is *not* for.

## By question

### Water and floods

| Your question | Use | Notes |
|---|---|---|
| Where is open water right now? | [`water_bodies`](catalog/water_bodies.md) | NDWI on Sentinel-2. |
| How did a lake or reservoir change through a year? | [`water_extent_timeseries`](catalog/water_extent_timeseries.md) | Water area per date, and how often each pixel was water. |
| Is a mine or quarry site revegetating, and where is it stuck? | [`mine_rehabilitation`](catalog/mine_rehabilitation.md) | Greenness against an undisturbed reference over several years; see [mining](../examples/mining.md). |
| How did water change over four decades? | [`surface_water_dynamics`](catalog/surface_water_dynamics.md) | JRC Global Surface Water, 1984–2020. |
| What flooded in this event, and the sky was clear? | [`flood_mapping`](catalog/flood_mapping.md) | Optical, simple and explainable. |
| What flooded, and it was cloudy? | [`sar_flood_mapping`](catalog/sar_flood_mapping.md) | Radar sees through cloud. |
| What flooded, and I need the most reliable map? | [`multisensor_flood_mapping`](catalog/multisensor_flood_mapping.md) | Radar + optical + terrain; shows where the sensors agree. |
| Where could rivers flood in principle? | [`drainage_analysis`](catalog/drainage_analysis.md) | HAND: height above the nearest channel. |
| Where are the wetlands, wet or dry season? | [`wetland_mapping`](catalog/wetland_mapping.md) | Open water vs wet vegetation. |
| Where is the water greener or muddier? | [`water_quality`](catalog/water_quality.md) | Relative chlorophyll and turbidity patterns. |
| How much water does the land return to the air? | [`evapotranspiration`](catalog/evapotranspiration.md) | MODIS annual ET and PET. |
| Where is the shoreline moving? | [`coastal_monitoring`](catalog/coastal_monitoring.md) | Land/water change between two dates. |

### Land, forests and vegetation

| Your question | Use | Notes |
|---|---|---|
| What is on the ground, and how much of each? | [`land_cover`](catalog/land_cover.md) | ESA WorldCover 2021 reference map. |
| What changed between two years, and into what? | [`landcover_change`](catalog/landcover_change.md) | Annual 10 m maps 2017–2023, with a from→to table. |
| How much forest was lost 2015–2020? | [`forest_change`](catalog/forest_change.md) | Radar-based, works under cloud. |
| Forest loss with my own definition, or after 2020? | [`deforestation`](catalog/deforestation.md) | Needs a network you trained. |
| Is vegetation greener or browner than usual? | [`vegetation_anomaly`](catalog/vegetation_anomaly.md) | Compared with the same month in earlier years. |
| How does greenness change through the season? | [`vegetation_indices`](catalog/vegetation_indices.md) | NDVI, EVI, NDWI per date. |
| Did vegetation drop between two dates? | [`crop_health`](catalog/crop_health.md) | NDVI drop. |
| Which crop is where? | [`crop_monitoring`](catalog/crop_monitoring.md) | Needs a network you trained. |
| Where is habitat most varied? | [`biodiversity_hotspot`](catalog/biodiversity_hotspot.md) | Shannon diversity of land cover. |

### Fire and hazards

| Your question | Use | Notes |
|---|---|---|
| Where is it burning, and on how many days? | [`active_fire`](catalog/active_fire.md) | MODIS thermal anomalies, 1 km. |
| How much burned, and how severely? | [`wildfire_severity`](catalog/wildfire_severity.md) | dNBR classes. |
| Where is the ground abnormally hot (volcano)? | [`volcano_monitoring`](catalog/volcano_monitoring.md) | Candidates only. |
| Where did landslides happen? | [`landslide_detection`](catalog/landslide_detection.md) | Needs a network you trained. |
| How damaged is the area after an event? | [`disaster_assessment`](catalog/disaster_assessment.md) | Needs a network you trained. |
| What changed between two dates, whatever it was? | [`change_detection`](catalog/change_detection.md) | Change vector analysis; no training needed. |

### Cities, roads and buildings

| Your question | Use | Notes |
|---|---|---|
| Where are the buildings, how many? | [`building_footprints`](catalog/building_footprints.md) | Microsoft, Google or OpenStreetMap footprints; or your network. |
| How much road, of which kind? | [`road_network`](catalog/road_network.md) | OpenStreetMap. |
| Detect roads from imagery | [`road_extraction`](catalog/road_extraction.md) | Needs a network you trained. |
| How much built-up area was added? | [`urban_expansion`](catalog/urban_expansion.md) | Annual 10 m maps 2017–2023. |
| Urban growth with my own classes or before 2017? | [`urban_growth`](catalog/urban_growth.md) | Needs a network you trained (Landsat). |
| Which neighbourhoods are hottest? | [`urban_heat_island`](catalog/urban_heat_island.md) | Landsat surface temperature. |
| What is the surface temperature? | [`land_surface_temperature`](catalog/land_surface_temperature.md) | Same product, without the city framing. |

### Terrain, snow, ice and energy

| Your question | Use | Notes |
|---|---|---|
| Elevation, slope, aspect, ruggedness? | [`terrain_analysis`](catalog/terrain_analysis.md) | Copernicus 30 m DEM. |
| How much snow on this date? | [`snow_cover`](catalog/snow_cover.md) | NDSI. |
| Did the glacier grow or shrink? | [`glacier_monitoring`](catalog/glacier_monitoring.md) | Compare end-of-melt-season dates. |
| How does slope and aspect affect sunlight? | [`solar_potential`](catalog/solar_potential.md) | One date and hour; not annual yield. |
| Where are solar farms? | [`solar_detection`](catalog/solar_detection.md) | Needs a network you trained. |
| Which ground passes wind-farm constraints? | [`wind_farm_siting`](catalog/wind_farm_siting.md) | Screening, not wind resource. |
| Where might earthworks be visible? | [`archaeological_site`](catalog/archaeological_site.md) | Relief enhancement for an expert; no detection. |

## When several pipelines look alike

### Flood mapping: three methods, three answers

| | [`flood_mapping`](catalog/flood_mapping.md) | [`sar_flood_mapping`](catalog/sar_flood_mapping.md) | [`multisensor_flood_mapping`](catalog/multisensor_flood_mapping.md) |
|---|---|---|---|
| Sensor | Sentinel-2 optical | Sentinel-1 radar | Radar + optical + DEM |
| Cloud | **Blocks it** | Sees through | Sees through (radar); optical confirms where clear |
| Strength | Simple, explainable, good in open land | Open-water flooding, any weather | Says where the sensors agree; removes hill and shadow false alarms; gives a lower and an upper bound |
| Blind spots | Cloud; flooded vegetation and towns | Floodwater among buildings and under dense vegetation; wind-roughened water; smooth surfaces | The same radar limits; the optical part needs a clear day |
| Date input | Month or date | **Range** of one to two weeks | **Range** of one to two weeks |
| Choose it when | The flood day was clear | It was cloudy and you want a simple radar result | You need the most reliable map, especially in the tropics |

**Practical rule:** run `multisensor_flood_mapping` for the event. If you need to explain the method to a non-specialist, add `flood_mapping`
as a simple cross-check. If the results differ, the difference is informative: read the class table of the multi-sensor run (confirmed, radar
only, optical only, disputed).

No satellite map shows the *peak* of an event unless a pass happened at the peak. Read the acquisition dates in the statistics
(`flood_dates`) before comparing with news reports or gauges.

### Water: now, over a year, or over decades

| Horizon | Use |
|---|---|
| One date | `water_bodies` |
| Several dates in a year | `water_extent_timeseries` |
| 1984–2020 | `surface_water_dynamics` |
| One event | the flood pipelines |

### Change: which one?

| Need | Use | Why |
|---|---|---|
| A first look at anything that changed | `change_detection` | Change vector analysis with an Otsu threshold; no training; says only *that* something changed |
| Change in land-cover **class**, citable | `landcover_change` | Same method for each year; from→to table |
| Built-up area gain/loss | `urban_expansion` | Specialised view of the same product |
| Forest loss in the 2015–2020 window | `forest_change` | Radar-based, independent of cloud |
| Burned area | `wildfire_severity` | dNBR is built for it |
| Vegetation vs normal | `vegetation_anomaly` | Statistical comparison with history |
| Your own classes of change | the *needs weights* change pipelines | You supply a trained change network |

### Land cover products

`land_cover` (WorldCover 2021) is a single map. Use it for "what is there". Use `landcover_change` (Impact Observatory / Esri) to compare years.
**Never compare WorldCover 2020 with 2021 or with the annual product:** different algorithms make the difference meaningless.

### Reference products vs. analyses of your imagery

Some pipelines **read a published product** for your box (`land_cover`, `landcover_change`, `urban_expansion`, `forest_change`,
`surface_water_dynamics`, `evapotranspiration`, `active_fire`, `building_footprints`, `road_network`). They are quick, citable and as accurate as
the product. Others **analyse imagery** with a stated method (`flood_mapping`, `water_bodies`, `wildfire_severity`, …) so you control the date and
the threshold. When a published product exists for your question and period, start with it.

## Quick decision guide

```{figure} ../_static/img/tree-pipeline-chooser.svg
:alt: Choosing a pipeline: change or a map of what is there, then the topic
:width: 100%
:class: pgv-figure

Follow the question from the left; the pipeline names are on the right.
```

## If your question has no ready pipeline

1. **Check the indices.** [`pygeovision indices list`](../data/indices.md) has 21 indices and transforms you can compute on any raster in one command.
2. **Combine existing outputs** with the [postprocessing tools](../data/postprocessing.md): thresholds, class statistics, zonal statistics.
3. **Train a model** for the task and run it through a *needs weights* pipeline or [direct inference](../inference/index.md).
4. **Ask for it.** [Contributing](../project/contributing.md) explains how a pipeline is added.

## Pipelines that were removed

Twenty-one pipelines from earlier versions were removed because they could not give a correct result (untrained networks, invented
constants, a single static map presented as change). [Removed pipelines](removed.md) lists each with the reason and what to use
instead.
