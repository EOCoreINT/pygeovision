# Tile labelers, dataset building and Label Studio

**Package:** `pygeovision.ai.labeling` · **Entry points:** `client.ai.label(tiles, labeler, output_dir)`,
`GeoDataLoader.prepare(..., labels=...)`

## Why this second stack exists

The [area labelers](index.md#two-stacks-one-idea) take a bounding box and give you **one** raster. When you are building a
training set you do not have one raster; you have **hundreds of image tiles**, and each needs its own label file with the
same name and the same grid. The tile labelers do exactly that: give them tiles, get back one label GeoTIFF per tile and
a result per tile that says what happened.

Use them when you want the whole chain (tile imagery → label every tile → split into train/validation/test) in one
call, or when you are labeling by hand with Label Studio.

## The common interface

Every tile labeler is a `BaseLabeler`:

```python
results = labeler.label_tiles(tiles, output_dir="labels/", task="segmentation")
```

- `tiles`: a list of `TileMetadata` (from the tiling step, see [Training data](../training/data.md)).
- `output_dir`: one `<tile_id>.tif` is written per tile. The label has the tile's own grid, so image/label pairs line up.
- `task`: must be one the labeler supports (`segmentation`, `detection`, `classification`, `land_cover`); anything else
  raises `LabelingError` listing the supported tasks.
- Returns a **`LabelingResult` per tile**. A tile that fails does not stop the others.

### `LabelingResult`

| Field | Meaning |
|---|---|
| `tile_id` | The tile |
| `label_path` | The written label file, or `None` |
| `success` | `True` only if a label was written and there was no error and it was not skipped |
| `skipped` | The tile was deliberately not labelled (no features, almost no foreground) |
| `error` | Why it failed, in words |
| `class_distribution` | Share of pixels per class name |
| `confidence`, `source`, `metadata` | Labeler-specific extras |

Always count the outcomes; do not assume success:

```python
ok = [r for r in results if r.success]
skipped = [r for r in results if r.skipped]
failed = [r for r in results if r.error]
print(len(ok), len(skipped), len(failed))
for r in failed[:5]:
    print(r.tile_id, r.error)
```

## The labelers

| Name (in `client.ai.label` / `GeoDataLoader`) | Class | What it needs | Classes |
|---|---|---|---|
| `"osm"` / `"openstreetmap"` | `OSMLabeler(feature_type=…)` | internet (Overpass) | see below |
| `"microsoft_buildings"` | `MicrosoftBuildingsLabeler` | internet | 0 background, 1 building |
| `"google_buildings"` | `GoogleBuildingsLabeler` | internet, `s2sphere` | 0 background, 1 building |
| `"esa_worldcover"` | `ESAWorldCoverLabeler(year, remap)` | internet (AWS bucket), tile files | ESA numbers or your `remap` |
| `"dynamic_world"` | `DynamicWorldLabeler` | Earth Engine account | Dynamic World 0–8 |
| `"sam"` | `SAMLabeler` | SAM weights | instance ids |
| `"foundation"` | `FoundationModelLabeler` | Prithvi/RemoteCLIP weights | clusters or prompt classes |
| `"label_studio"` | `LabelStudioLabeler` | a Label Studio server | your classes |

```{note}
These are **separate implementations** from the area labelers of the same names. Differences that matter: ESA WorldCover
here comes from the public AWS bucket (the area labeler uses the Planetary Computer); the options differ; and the results
are per tile. If a source has a page elsewhere in this section, the data and caveats on that page apply here too.
```

### OSM (`OSMLabeler(feature_type="buildings")`)

```python
from pygeovision.ai.labeling import OSMLabeler
labeler = OSMLabeler(feature_type="buildings", request_delay=1.0)
```

| `feature_type` | Classes (value = index) |
|---|---|
| `"buildings"` | 0 background, 1 building |
| `"roads"` | 0 background, 1 road |
| `"water"` | 0 background, 1 water |
| `"landuse"` | 0 background, 1 residential, 2 commercial, 3 agricultural, 4 vegetation |
| `"all"` | 0 background, 1 buildings, 2 roads, 3 water, 4 vegetation, 5 agricultural, 6 commercial, 7 residential, 8 solar_panels, 9 parking, 10 sports (same numbers as the [area labeler](osm.md)) |

Geometry is handled by the same code as the area labeler: roads and rivers are buffered lines (not filled blobs), lakes
with islands are assembled, buildings sit on top of land use. Labels are burned **on the tile's own grid and CRS**.

Behaviour you must know:

- One Overpass request **per tile**, `request_delay` seconds apart (default 1 s). For thousands of tiles, use your own
  Overpass server (`overpass_url=`) or label the whole area once with the [area labeler](osm.md) and cut it with
  `pygeovision ai chips`.
- A tile with **no features, or fewer than 0.1% foreground pixels, is skipped** (no file is written). That avoids teaching
  "nothing here" in unmapped areas; it also means genuine empty tiles are not in your training set. Decide whether
  that is what you want (see below).
- A bad `feature_type` raises `OSMLabelingError` listing the valid names.

### Footprints, land cover, SAM, foundation

- **Microsoft / Google buildings**: binary building masks on the tile grid. Same coverage rules as the
  [area versions](buildings.md). Skipped when there are no footprints or under 0.1% building pixels.
- **ESA WorldCover**: `ESAWorldCoverLabeler(year=2021, remap={…})`. Downloads the 3° tiles that cover each image tile and
  resamples to the image grid. If no WorldCover tile can be fetched, the tile is an **error** (no label file), never an
  all-zero mask.
- **Dynamic World**: `DynamicWorldLabeler(start_date, end_date)`. Uses Google Earth Engine. If no scene is found the tile
  is an error. `composite_method="probability"` and `probability_bands=True` raise `NotImplementedError` (they used to be
  accepted and silently return the most common class).
- **SAM**: `SAMLabeler(model_type="vit_b", points_per_side=32, min_mask_region_area=100, rgb_bands=(0, 1, 2))`. Note that
  `rgb_bands` here is **0-based**, unlike the area-stack `SAMAutoLabeler` (1-based). Output is an instance map per tile
  (not classes); see [Segment Anything](sam.md).
- **Foundation**: see the next section.

### Foundation-model labeler (Prithvi, RemoteCLIP)

```python
from pygeovision.ai.labeling import FoundationModelLabeler

labeler = FoundationModelLabeler(model="prithvi", num_classes=8, input_bands=[0, 1, 2, 3, 4, 5])
```

| `model` | Backbone | Input | Grid |
|---|---|---|---|
| `"prithvi"` | NASA/IBM Prithvi-EO-1.0 (100M) | six HLS bands: blue, green, red, narrow NIR, SWIR-1, SWIR-2, in that order | 16 px patches, 224 px windows |
| `"prithvi_600"` | Prithvi-EO-2.0 (600M) | the same six bands | 14 px patches |
| `"remote_clip"` | RemoteCLIP ViT-L/14 (`open_clip_torch`) | RGB | one embedding per 224 px; with `text_prompts=[…]` it labels by matching prompts, so class `k` is `text_prompts[k]` |

`"clay"` and `"satmae"` raise `NotImplementedError`: they have no verified loader here. If the requested model cannot
be loaded, **the tile fails with the reason**; nothing is substituted (earlier versions silently used an unrelated model
and reported success). Only `task="land_cover"` is implemented. Cluster ids are as unnamed as in the
[DINOv2 version](foundation.md); read its warning.

## Building a dataset in one call

```python
from pygeovision.ai.data.dataloader import GeoDataLoader

loader = GeoDataLoader(data_pipeline=None)       # a fetcher is needed only when data_source is search results
dataset = loader.prepare(
    "data/scenes/",                    # a directory, a file, a list of paths, or search results
    labels=["openstreetmap"],          # one or several strategies, tried in order
    task="segmentation",
    tile_size=512, overlap=64,
    val_split=0.15, test_split=0.10,
    output_dir="datasets/lagos/",
)
print(dataset.split_counts(), dataset.class_names)
```

What `prepare` does: finds imagery → cuts tiles → labels every tile → splits by **source file** (so tiles of one scene do not
land in both train and validation) → builds the augmentation pipeline → saves the dataset and `splits.json`.

Things to know:

- **If no tile receives a label it raises `DatasetError`** with the first underlying problem (previously it carried on with
  an unlabelled dataset). If *some* tiles fail, they are logged and left out; check the log counts
  ("N labelled, M skipped, K failed").
- Strategy names: `openstreetmap`, `microsoft_buildings`, `google_buildings`, `esa_worldcover`, `dynamic_world`, `sam`,
  `foundation`. An unknown or unloadable name is logged and ignored; the raise above catches the case where that leaves
  nothing.
- `class_names` are **guessed** from the task and strategies, not read from your labels. An OpenStreetMap run with
  `task="segmentation"` is reported as the generic `["background", "foreground"]` whatever `feature_type` painted.
  Set `dataset.class_names` yourself, and check them against the values in your label files before training.
- `preprocessing_steps` is accepted but **not applied** (a warning says so). Normalise in the training step.
- **How the split works.** With three or more scenes, whole scenes go to train, validation or test, so no ground is
  shared. With fewer than three, there is not enough to give every split a scene, so each scene is cut into contiguous
  **bands of tile rows** (train on top, then validation, then test; columns if the scene is wide and short). A warning is
  logged. Tiles on either side of a band edge share up to `overlap` pixels, so the validation score of a one-scene dataset
  is slightly optimistic. Use more scenes, or set `overlap=0`, when that matters. A scene with fewer than three distinct tile
  rows and columns is all train.

Single-labeler form, through the client:

```python
results = client.ai.label(tiles, "osm", output_dir="labels/", feature_type="roads")
```

Valid names: `osm`, `microsoft_buildings`, `google_buildings`, `esa_worldcover`, `dynamic_world`, `sam`, `foundation`,
`label_studio`. Extra keyword arguments go to the labeler's constructor. An unknown name raises `ValueError` with the list.

## Label Studio: labeling by hand with a team

[Label Studio](https://labelstud.io/) is a free, open-source annotation tool with a browser interface. `LabelStudioLabeler`
sends your tiles there as tasks, then reads the finished annotations back as aligned label masks.

```python
from pygeovision.ai.labeling.label_studio import LabelStudioLabeler

ls = LabelStudioLabeler(
    url="http://localhost:8080",
    api_key="<token from Account & Settings>",
    task_type="segmentation",                      # brush painting
    class_names=["background", "building", "road"],  # list index = raster value, background first
)
project_id = ls.export_tiles(tiles, project_title="Lagos buildings")
# ... annotators work in the browser ...
results = ls.label_tiles(tiles, output_dir="labels/")
```

### The flow

1. **Start Label Studio** (`pip install label-studio`, `label-studio start`), and copy your access token.
2. **`export_tiles(tiles)`** creates a project (or uses `project_id=`) with a labeling interface built from `class_names`,
   writes each tile as a PNG, and uploads one task per tile. It returns the project id. Keep it.
3. People annotate in the browser.
4. **`label_tiles(tiles, output_dir)`** fetches the project's tasks (all pages), takes the **latest annotation** of each tile's
   task, and rasterises it into `<output_dir>/<tile>.tif` on the tile's grid.

### Task types

| `task_type` | Annotators… | Becomes |
|---|---|---|
| `"segmentation"` | paint with a brush | pixels painted in the class's raster value (brush data are decoded from Label Studio's RLE format) |
| `"polygon"` | draw polygons | filled polygons |
| `"bbox_detection"` | draw boxes | filled rectangles |
| `"classification"` | pick one label | the whole tile gets that value |

### Rules that avoid wrong labels

- **`class_names` order is the raster value**: `["background", "building", "road"]` means background 0, building 1, road 2.
  Put `"background"` first. A label name not in this list raises `LabelingError` (a half-labelled tile is never written).
- Tiles are matched to tasks by the stored tile path, then by **exact** file name (tile `t_1` does not pick up `t_10`).
- Tiles with no task, or whose task has no annotation yet, give an error result (`no_task_found`, `no_annotations`) and
  write no file. Unannotated tiles are *not* labelled "all background".
- If an annotator changed their mind, the newest annotation wins.

### What to do when…

| You see | Cause | Do this |
|---|---|---|
| `LabelingError: Label Studio API key is required` | No token | Pass `api_key=` |
| Connection refused / timeouts | Server not running or wrong `url` | Start it; check the address and port |
| `error="no_annotations"` | Task not annotated (or not submitted) | Finish and **submit** the task in the UI |
| `error="no_task_found"` | Tile was not exported to this project | Run `export_tiles` with the same tiles; check `project_id` |
| `label … not in class_names` | UI label differs from your list | Fix `class_names` or the project's labels so they match exactly |
| `brush annotation has no 'rle' data` | Exported in a format without RLE | Use the default JSON API |
| `RLE decodes to N values, expected …` | Image size mismatch | The tile image in Label Studio must be the tile you pass in |
| Brush edges look coarse | Annotators painted on a scaled-down image | Mask is resized with nearest neighbour; paint with a thinner brush or zoom in |

```{note}
The Label Studio brush-RLE decoder was written from the published algorithm and checked against an encoder written
here (round trip), and the HTTP flow against a stand-in server. It could **not** be tested against a real Label Studio
server in the documentation build environment. If decoded masks look shifted or inverted on your server, report it with
one exported task.
```

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `DatasetError: No tile received a label from […]` | Nothing could be labelled | Read the quoted first problem: network, coverage, or missing package |
| Most results are `skipped` | OSM/footprints have little in those tiles | Check the source covers the area; consider whether empty tiles should be kept |
| `LabelingError: Labeler 'x' does not support task 'y'` | Wrong `task` for that labeler | Use one of the listed tasks |
| Many `Overpass request failed` | Rate limiting | Increase `request_delay`; use your own Overpass server; or label the area once with the [area labeler](osm.md) |
| `imagery chip for tile … is not on disk` | Tiles were not saved | Tile with `save_tiles=True`, or set `tile.extra["tile_path"]` |
| `LabelingOrchestrator.label()` raises `NotImplementedError` | The old stub never worked | Use [`AutoLabelPipeline`](pipeline.md), or call a labeler's `label_tiles` and merge |
| Labels look shifted against the tiles | Labels and tiles from different grids | The tile labelers write on the tile grid; check that the tiles themselves are georeferenced correctly |

```{seealso}
[Training data](../training/data.md) · [Quality and cleaning](quality.md) · [Active learning](active.md)
```
