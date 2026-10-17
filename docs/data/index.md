# Data overview

The data layer finds satellite scenes, downloads only the files you need, and turns them into **analysis-ready
rasters**: correctly scaled, cloud-masked, mosaicked over your area and cropped to your box. Every pipeline is built
on it, and every function it uses is also available directly, so you can prepare data for your own analysis.

Data work has two halves. First you **get the right imagery** for your area and dates. Then you **prepare and check** it so that a
model or an index can use it. Most of the time the pipelines do both for you; this section is for when you want to see or control
what they do, or to prepare data for your own analysis.

```{list-table}
:header-rows: 1
:widths: 28 72
* - Get imagery
  - What it answers
* - [Search and download](search-and-download.md)
  - How do I find scenes and fetch them, from Python and from the CLI? What do `SearchResult` and `DownloadResult` contain?
* - [Providers](providers.md)
  - Which of the 22 providers do I need? Which need an account? Which have radar, sub-metre imagery or STAC?
* - [Authentication](authentication.md)
  - How do I store credentials safely, and test that they work?
* - [Scene selection and mosaicking](scene-selection.md)
  - How does PyGeoVision decide *which* scenes to use, and what happens when one scene does not cover my box?
* - [Study-area boundaries](boundaries.md)
  - How do I use the true shape of an administrative area instead of a hand-typed box?
* - [Reference products and terrain](reference-data.md)
  - Where do land cover, surface water, forest maps and DEMs come from, and how are they read for just my box?
* - [Building and road footprints](footprints.md)
  - How do I get building and road vectors from Microsoft, Google and OpenStreetMap?
* - [Dataset catalogue](../datasets/index.md)
  - Which public training datasets exist for my task? (Metadata only; nothing is downloaded.)
* - [Data pipelines (YAML)](yaml-pipelines.md)
  - How do I keep a folder of imagery up to date on a schedule?
```

Once you have imagery, [prepare and check it](prepare.md): scale it to reflectance, mask cloud, compute indices, validate it and
turn model output into polygons.

## The flow in one picture

```text
  bbox + dates ─▶ search ─▶ dedupe listings ─▶ choose the fewest clear scenes that cover the box
                                                            │
        download only the bands needed (+ cloud layer) ◀────┘
                                │
   per scene: stack bands ─▶ scale DN→reflectance ─▶ mask cloud/shadow ─▶ write footprint
                                │
          mosaic scenes ─▶ fill remaining gaps from other scenes ─▶ crop to bbox
                                │
        measure how much of the box is covered and how much is clear ─▶ analysis-ready raster
```

Each arrow is described in detail on the pages above. Two design decisions shape everything:

1. **Correct before combined.** Each scene is scaled and masked *on its own* and only then mosaicked, so there is no
   calibration jump at a seam between two acquisitions.
2. **Say how much you got.** The coverage actually achieved (`aoi_coverage_fraction`) and the share of clear pixels
   are recorded and attached to the result. A run that could only see 40 % of your box tells you so.

```{toctree}
:hidden:
:maxdepth: 1

search-and-download
providers
authentication
scene-selection
boundaries
reference-data
footprints
../datasets/index
yaml-pipelines
prepare
```
