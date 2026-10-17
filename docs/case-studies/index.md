# Case studies

Eight worked studies built from real PyGeoVision runs: ten PDF reports, nine pipelines, six places. Each page takes one report,
explains how the run was made and what every figure means, and then does the thing a quick read skips: it **checks the report
against itself**, says what it can and cannot support, and shows what to change next. Several of them end on "do not quote this
number yet", and that is deliberate. A run that goes wrong in an instructive way teaches more than one that goes right.

```{admonition} How much to trust these pages
:class: important
- **The numbers are the reports' numbers.** They were copied from the PDFs that PyGeoVision wrote for these runs, and are kept in one
  file (`docs/_data/case_study_numbers.yaml`) that the tests read.
- **The arithmetic is re-derived.** Pixel counts to areas, shares to hectares, grids to cell counts, and so on, are recomputed by
  `tests/test_case_studies.py`. Every `# check:` block on these pages is run, and its output block must match.
- **The mechanisms are reproduced.** Where a page explains why a report says something odd (a burned area that differs by 20.8 ha,
  a density raster that holds less than its headline, undefined NDCI values), the test suite reproduces the effect on the **real
  pipeline** with a small synthetic input whose answer is known.
- **Nothing was re-run on real data.** The scenes and datasets were not downloaded again, so no figure here has been recomputed from
  imagery.
- **None of it is an accuracy assessment.** No study has ground truth. Where a page offers a reason for something in a map, it says
  "a hypothesis" and gives the check that would settle it.
```

## The studies

| Study | Pipeline | Place | What the report says | What the page teaches |
|---|---|---|---|---|
| [Flood extent, Accra coast](flood-accra.md) | `multisensor_flood_mapping` | Southern Ghana, 1,470 km² | 81.4 ha flooded, 4.2 ha seen by both radar and optical | Three kinds of evidence kept apart, and how to read an agreement table. |
| [Flood extent, central Nepal](flood-nepal.md) | `multisensor_flood_mapping` | Nepal, 5,232 km² | 451.7 ha flooded, 90 % optical only | A run whose terrain step was skipped for size, and the one line that fixes it. |
| [Burn severity, northern California](wildfire.md) | `wildfire_severity` | California, 3,321 km², Oct to Dec 2018 | 49,568.1 ha burned | Two numbers that differ by 20.8 ha, and what dNBR does and does not show. |
| [Streams and HAND, Tyrol](drainage-tyrol.md) | `drainage_analysis` | Inn valley, 84 km² | 64.8 km of streams, 19.5 % of ground within 15 m of one | A density taken over the grid, not the box, and a long-tailed mean. |
| [Water quality, upper Chesapeake Bay](water-quality.md) | `water_quality` | Maryland, 1,539 km² | Mean NDCI 0.249 in 2.1 % water | Four numbers that do not fit together: 48 % of the water has no NDCI. |
| [Land cover and habitat, central London](london-landcover-habitat.md) | `land_cover`, `biodiversity_hotspot` | London, 19.3 km² | 73.8 % built-up; 10 % hotspots | A number that is true by construction (the 10 %). |
| [Buildings and roads, central London](london-buildings-roads.md) | `building_footprints`, `road_network` | London, 19.3 km² | 5,898 buildings; 1,030 km of roads | Rasters that hold less than their headline; "roads" that are mostly pavements. |
| [Vegetation change, central London](london-vegetation.md) | `crop_health` | London, 19.3 km² | 5.8 % of valid area flagged | A share of what, and a crop pipeline pointed at a city. |

## What they have in common

The same handful of questions come up in every study, and the pages are organised to practise them:

1. **A share of what?** A percentage of the valid pixels, of the grid, of the box, of the land: they are different numbers. The
   vegetation page has 5.8 % and 2.6 % for the same pixels.
2. **Do the counts agree?** Two statistics in one report over different pixel counts (the water-quality page) is a finding.
3. **Is the figure true by construction?** The share of hotspots is the percentile you chose; classes of equal size are terciles.
4. **Does the raster add up to the headline?** The density rasters on the buildings and roads pages do not.
5. **What did not run?** A skipped stage, a masked region, a valid share of 44 %. Nepal and the vegetation page start there.
6. **What would settle it?** Every doubtful finding ends in a command or a few lines of Python you can run on your own outputs.

## How the pages are built

Each page follows the same order, so you can find the same thing in the same place:

| Section | What it gives you |
|---|---|
| **The result at a glance** | The headline figures, before any interpretation. |
| **Study area and inputs** | The box, the grid, the dates, the scenes, and the settings that were used. |
| **Run it** | The `pygeovision channel` command and the equivalent Python, as the run was made. |
| **What the pipeline did** | A flow figure and a numbered list of the stages. |
| **The result** | Maps, class tables and statistics, each with a short reading. |
| **The part that needs a second look** | The specific issue of that run, with the arithmetic. |
| **Things to be careful about** | The limits that apply to the figures. |
| **Change something** | The arguments you would change next, and what each does. |
| **The files it wrote** | What is in the output folder. |

## Run one yourself

Every command on these pages has the same shape: a pipeline name, a bounding box (`WEST SOUTH EAST NORTH`), the date arguments the
pipeline takes (if any), `--param KEY=VALUE` for anything else, and an output folder.

```bash
pygeovision channel <pipeline> --bbox WEST SOUTH EAST NORTH [--date YYYY-MM] [--param KEY=VALUE ...] --output DIR
```

The reports are written into `DIR/deliverables/` unless you switch them off. The pipelines that need no imagery (`land_cover` with
WorldCover, `biodiversity_hotspot`, `building_footprints`, `road_network`, `drainage_analysis`) need no account; the ones that read
Sentinel scenes download them through the Planetary Computer. [The pipeline catalogue](../pipelines/index.md) lists every argument.

To run the checks behind these pages:

```bash
pytest tests/test_case_studies.py
```

```{toctree}
:hidden:
:maxdepth: 1

flood-accra
flood-nepal
wildfire
drainage-tyrol
water-quality
london-landcover-habitat
london-buildings-roads
london-vegetation
```
