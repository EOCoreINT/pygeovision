# Examples by domain

Each page is a **workflow**, not a feature list: the question a practitioner asks, the pipelines and tools that answer it in
order, the files to open, how to read them, and what to do when a step gives a doubtful result. All commands are
the ones documented on the [pipeline pages](../pipelines/catalog/index.md) and in the [CLI reference](../reference/cli/index.md);
the arguments shown there are checked against the code.

| Domain | Questions it covers | Main pipelines |
|---|---|---|
| [Agriculture](agriculture.md) | Is the crop greener or browner than usual? Which fields dropped? How much water did the land return? | `vegetation_anomaly`, `vegetation_indices`, `crop_health`, `evapotranspiration`, `crop_monitoring`\* |
| [Forestry](forestry.md) | How much forest was lost, where, and is it real? Where is fire damage? | `forest_change`, `landcover_change`, `wildfire_severity`, `biodiversity_hotspot`, `deforestation`\* |
| [Urban](urban.md) | Where are the buildings and roads? How fast does the city grow? Where is it hottest? | `building_footprints`, `road_network`, `urban_expansion`, `urban_heat_island`, `urban_growth`\* |
| [Water and floods](water.md) | Where is the water, and how did it change? Which ground flooded, and how sure are we? | `water_bodies`, `water_extent_timeseries`, `surface_water_dynamics`, the three flood pipelines, `drainage_analysis` |
| [Disaster response](disaster.md) | What changed after the event? How badly was it burned or flooded? | `change_detection`, `flood_mapping` family, `wildfire_severity`, `active_fire`, `disaster_assessment`\*, `landslide_detection`\* |
| [Mining](mining.md) | Is the disturbed ground of a mine or quarry coming back, where is it stuck, and is a decline real? | `mine_rehabilitation`, with `vegetation_anomaly`, `water_extent_timeseries`, `landcover_change` to explain it |
| [Climate and cryosphere](climate.md) | Is the glacier shrinking? Where is snow? What are surface temperatures and evapotranspiration doing? | `glacier_monitoring`, `snow_cover`, `land_surface_temperature`, `evapotranspiration`, `surface_water_dynamics`, `coastal_monitoring` |

\* needs a network **you** trained; the pipeline refuses to start without `--checkpoint`. See
[training](../training/index.md) and [labeling](../labeling/index.md).

```{admonition} What these pages do not contain
:class: note
Real results for a named place. The numbers on the pipeline pages come from an offline test harness with
synthetic data and say so. Where a page here says what a result *looks like*, it describes the structure
(which file, which key, which class value), never an invented measurement. The one complete end-to-end case study in the
repository, the [Ghana Volta flood atlas](water.md#case-study-the-2023-volta-floods), is described with the same
honesty: its plumbing and statistics are tested on synthetic data and its real-imagery accuracy has not been measured.
```

## Habits that apply to every workflow

1. **Run the cheapest pipeline first.** A reference-product pipeline (`land_cover`, `landcover_change`, `forest_change`)
   downloads one small raster rather than many scenes; use it to confirm your bounding box before a multi-scene run.
2. **Read `run_summary.json` before the map.** It records the dates, scenes, thresholds and `valid_fraction` actually
   used. A map built from one cloud-edge scene looks the same as one built from ten.
3. **Check `valid_fraction`** (and `data_notes`). Percentages and areas are computed over the assessable part of the box only; a low
   value means the result describes a small part of what you asked about.
4. **255 and NaN are "unknown", not "none".** Never read them as dry, unburned or unchanged.
5. **A pipeline gives a screening result, not a measurement.** Validate a figure you will publish against your own
   reference points ([`postprocess accuracy`](../reference/cli/postprocess.md), [OBIA accuracy](../obia/accuracy.md)).

```{toctree}
:hidden:
:maxdepth: 1

agriculture
forestry
urban
water
disaster
mining
climate
```
