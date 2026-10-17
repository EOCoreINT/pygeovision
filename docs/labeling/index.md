# Labeling

A model learns from examples, and an example is an image **plus the right answer for every pixel**. Making those answers
by hand (tracing every building in a district) takes weeks. PyGeoVision's labeling layer gets you a **first set of
labels in minutes** from sources that already exist: OpenStreetMap, published building footprints, published land-cover
maps, and general-purpose segmentation models.

```{important}
Every label source here is an **approximation of the truth**, not the truth. OSM is mapped by volunteers, building
footprints and land-cover maps are themselves model outputs, and a segmentation model names nothing. Treat what comes
out as a **starting point**: look at it over the imagery, measure its quality, fix what you can, and keep a small
hand-checked set to measure the trained model on. Each page below says what its source gets wrong.
```

## Pick a source

| You want | Use | Needs | Output values | Read |
|---|---|---|---|---|
| Buildings, roads, water, land use in a place where people have mapped them | `OSMLabeler` | internet (Overpass) | 0 background, 1–10 per category | [OpenStreetMap](osm.md) |
| Buildings where OSM is thin (Africa, South and Southeast Asia, Latin America; most of the world) | `MicrosoftBuildingsLabeler`, `GoogleBuildingsLabeler` | internet | 0 / 1 | [Building footprints](buildings.md) |
| Coarse land cover: trees, crops, built-up, water | `ESAWorldCoverLabeler`, `DynamicWorldLabeler` | internet | product's own class numbers | [Land cover](landcover.md) |
| Object outlines in your own image, no class names | `SAMAutoLabeler`, `SamGeoLabeler` | model weights, GPU helps | instance ids 1…N | [Segment Anything](sam.md) |
| "Group similar areas" without any names | `FoundationModelLabeler` | model weights | cluster ids | [Foundation-model clusters](foundation.md) |
| Name SAM or cluster regions using an existing map | `assign_classes` | two rasters | class numbers | [Naming segments](naming-segments.md) |
| Combine several sources over one area | `AutoLabelPipeline` | the sources' needs | fused 0/1 raster | [Combining sources](pipeline.md) |
| Check a label raster, remove specks | `LabelQualityAssessor` | rasters only | score, grade, cleaned raster | [Quality and cleaning](quality.md) |
| Choose which images a person should label next | `ActiveLearner` | a trained model | ranked list | [Active learning](active.md) |
| Label by hand, with a team | `LabelStudioLabeler` | a Label Studio server | masks per tile | [Tile labelers and Label Studio](tile-labelers.md) |
| Build a whole dataset (tiles + labels + splits) | `GeoDataLoader.prepare` | imagery | a `GeoDataset` | [Tile labelers and Label Studio](tile-labelers.md) |

## Two stacks, one idea

PyGeoVision has two ways to call the same kind of source, because they answer two different questions.

| | **Area labelers** (`pygeovision.labeling`) | **Tile labelers** (`pygeovision.ai.labeling`) |
|---|---|---|
| You give | a bounding box (and optionally your image's grid) | a list of image tiles |
| You get | **one** label raster for the whole area, plus GeoJSON | **one** label GeoTIFF per tile |
| Returns | a plain `dict` with `success` and `error` | a list of `LabelingResult` |
| Reached from | `client.labeling.*`, `pygeovision label …` | `client.ai.label(...)`, `GeoDataLoader.prepare(labels=…)` |
| Best for | making labels for one scene, inspecting them, fusing sources | building a tiled dataset in one call |

If you are unsure, start with the **area labelers**: you can open the one output file in QGIS next to your imagery and
see immediately whether the labels make sense.

## Conventions every source follows

These are the same everywhere, and they match what [the trainer expects](../training/data.md).

| Convention | Meaning |
|---|---|
| **`0` is background** | "Looked, and found nothing of the classes asked for." |
| **`255` is no data / ignore** | "No information here." The trainer's loss and metrics skip these pixels |
| **bbox = `(lon_min, lat_min, lon_max, lat_max)`** | Always degrees (EPSG:4326), west–south–east–north. **Never** `(lat, lon, …)` and never projected metres |
| **Results are dicts** | `result["success"]` is `True` only if a usable raster was written; otherwise `result["error"]` says why. Labelers do not raise for network or coverage problems |
| **`reference_raster`** | Pass your imagery's path and the labels are written **on its exact grid, in its CRS**. This is what makes image/label pairs line up for training |
| **`uint8`** | Label rasters are `uint8` (SAM instance maps are `uint32`) |

```{admonition} Always pass reference_raster when the labels are for training
:class: tip
Without it you get a lon/lat grid at roughly the resolution you ask for. It will overlay the imagery in a GIS but
**will not share its pixel grid**, and [`pygeovision ai chips`](../training/data.md) refuses image/label pairs whose
grids differ. With it, the pairing works with no resampling step.
```

## Typical workflow

```text
1. Download imagery for your area                  → data/scene.tif
2. Label it from one or more sources               → labels/osm.tif   (reference_raster=data/scene.tif)
3. Look at the labels over the imagery in QGIS     → is anything missing or shifted?
4. Measure and clean                               → pygeovision label quality / LabelQualityAssessor.clean
5. Cut matching chips                              → pygeovision ai chips
6. Train                                           → GeoTrainer (see Training)
7. Predict on new areas, review errors, relabel    → ActiveLearner picks what to fix next
```

A worked example of steps 1–5 is on the [workflow page](workflow.md).

## Command line

```console
$ pygeovision label osm --reference data/nyc.tif -c buildings -c roads -o labels/nyc.tif -- -74.05 40.70 -73.95 40.80
$ pygeovision label quality labels/nyc.tif --html
```

Only these two sources have command-line commands. Everything else is Python (`client.labeling.*` or the classes
directly). See the [CLI reference](../reference/cli/label.md).

## What this layer does not do

- It does not **know whether a label is right**. `LabelQualityAssessor` checks the raster for specks, imbalance and
  missing area; it never compares to the ground. Only a hand-checked sample can tell you accuracy.
- It does not fill gaps. A pixel nobody mapped is background, not "unknown", unless you decide otherwise (see
  [OpenStreetMap](osm.md) on what to do with unmapped areas).
- It does not give SAM or cluster results class names. That takes a reference map or a person.

```{toctree}
:hidden:
:maxdepth: 1

workflow
osm
buildings
landcover
sam
foundation
naming-segments
pipeline
quality
active
tile-labelers
```
