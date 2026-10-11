# Segment Anything labels (SAM)

**Classes:** `SAMAutoLabeler` (this package's own wrapper, using Hugging Face `transformers`), `SamGeoLabeler`
(wrapper around the `segment-geospatial` package) · **Client:** `client.labeling.sam_auto(image_path, ...)`

## Why use it

Segment Anything (SAM) is a model from Meta that outlines **objects in any picture without being trained on your
data**. Point it at an aerial image and it returns a mask for each roof, field, pond, car park or tree cluster it can
see. For labeling that is useful in two ways:

- **Speed up hand labeling.** A person chooses from outlines instead of tracing them.
- **Get object boundaries** for training sets where you already know *what* the objects are (from an existing map, or
  because the image only contains one kind), so you need only the *where*.

What SAM does **not** do is say what an object is. Its output is "segment 1, segment 2, …". You attach class names with
a reference map ([Naming segments](naming-segments.md)), with text prompts (GroundedSAM, below), or by hand.

```{important}
SAM was trained on ordinary photographs. On satellite imagery at 10 m (Sentinel-2) objects are only a few pixels wide
and SAM does poorly. It works best on **aerial or very-high-resolution imagery (about 0.1–1 m per pixel)**. If your
imagery is 10 m, expect fields and large water bodies to work and buildings not to.
```

## Two ways to use it

| | `SAMAutoLabeler.auto_label` | `SAMAutoLabeler.grounded_label` | `SamGeoLabeler.label` |
|---|---|---|---|
| Idea | Try a grid of points; keep the good masks | Type a word (`"building"`), find boxes, segment each | `segment-geospatial`'s automatic mode |
| Output | Instance ids 1…N (`uint32`) | Class `k` = `prompts[k-1]` (`uint8`) | Whatever samgeo writes |
| Extra install | `transformers` | `groundingdino-py` + a checkpoint | `segment-geospatial` |
| Handles big scenes | Yes: tiles with overlap, then merges | No: the scene is passed whole | Yes (`batch=True`) |
| Best for | "Give me all the objects" | "Give me all the buildings" | Users already on samgeo |

## `auto_label`: every object

```python
from pygeovision.labeling import SAMAutoLabeler

sam = SAMAutoLabeler(model="sam-vit-base")             # or sam-vit-large (default), sam-vit-huge, sam2-hiera-l/-b
result = sam.auto_label(
    image_path="data/aerial.tif",
    output_path="labels/sam_masks.tif",
    output_vector="labels/sam_masks.geojson",          # optional polygons
    points_per_side=32,
    min_area_m2=20,
    rgb_bands=(1, 2, 3),                               # 1-based; see the warning below
)
print(result["n_masks"], result["n_chip_errors"], result["error"])
```

Models: `sam-vit-huge`, `sam-vit-large` (default), `sam-vit-base` (fastest), `sam2-hiera-l`, `sam2-hiera-b`, or any other
Hugging Face repo id that `transformers` can load as mask generation. The names `geosam-*` were removed because they
pointed at a repository that does not exist; asking for one raises `ValueError`.

### Options

| Argument | Default | What it does | When to change it |
|---|---|---|---|
| `points_per_side` | 32 | Grid of prompts per tile (32 → 1,024 points) | Higher finds smaller objects and is slower; lower for big objects |
| `pred_iou_thresh` | 0.88 | Drop masks the model rates below this quality | Lower gets more (and rougher) masks |
| `stability_score_thresh` | 0.95 | Drop masks that change when the threshold is nudged | Lower gets more masks |
| `min_area_m2` | 10 | Smallest object kept, in real square metres | Raise to drop clutter |
| `max_area_m2` | `None` | Largest object kept | Set it to drop "the whole field is one mask" backgrounds |
| `chip_size` | 1024 | Tile size processed at once, pixels (≥ 64) | Lower if you run out of GPU memory |
| `overlap` | 128 | Tile overlap, pixels (< `chip_size`) | Raise if objects are cut at tile edges |
| `merge_overlapping` | `True` | Merge the same object seen in two tiles | Leave on |
| `rgb_bands` | `None` → bands 1, 2, 3 | Which bands (1-based) are R, G, B | **Set it** for anything but an RGB file |

```{warning}
SAM takes three display bands as red, green, blue. A Sentinel-2 file stored as B2, B3, B4 (blue, green, red) needs
`rgb_bands=(3, 2, 1)`. With the default `(1, 2, 3)` SAM sees a false-colour picture and the masks are poor with no
error message. If your file is a 4-band aerial, pass the three visible bands explicitly.
```

### What you get

```text
labels/sam_masks.tif       uint32 GeoTIFF: 0 = no mask, 1..N = one id per object (largest objects get the lowest ids)
labels/sam_masks.geojson   optional: one polygon per mask (EPSG:4326)
```

| Key | Meaning |
|---|---|
| `success` | `True` if at least one mask survived **and** not every tile failed |
| `n_masks` / `n_raw_masks` | Kept after filtering / found before filtering |
| `n_tiles` / `n_chip_errors` | Tiles processed / tiles that failed. Failed tiles are **missing** from the output |
| `vector_error` | Why the GeoJSON could not be written (the raster is still fine) |
| `duration_seconds`, `model` | |
| `quality_stats` | Mean predicted IoU of kept masks, and the thresholds used |
| `error` | In words: for example "SAM found 312 raw mask(s) but 0 survived quality/area filtering" |

Overlapping masks: where two masks cover a pixel, the **smaller** object is written on top, so a house is not erased by
the garden around it.

Each file is tagged (`source=SAM_AutoLabel`, `model`, `n_masks`). The array holds one big integer per object: **do not
train on this file directly as a class map**. `uint32` ids up to thousands would be read as thousands of classes.

## `grounded_label`: find what you name

GroundedSAM first finds boxes for a text phrase with Grounding DINO, then segments each box with SAM.

```python
sam = SAMAutoLabeler(
    model="sam-vit-base",
    gdino_config_path="GroundingDINO_SwinT_OGC.py",
    gdino_checkpoint_path="groundingdino_swint_ogc.pth",
)
r = sam.grounded_label(
    "data/aerial.tif", prompts=["building", "swimming pool"],
    output_path="labels/grounded.tif", box_threshold=0.30, text_threshold=0.25,
    rgb_bands=(1, 2, 3),
)
```

Class `1` in the output is `prompts[0]` ("building"), class `2` is `prompts[1]`, and so on; `0` is "nothing found". The
result has `n_prompts`, `n_boxes_detected`, `n_masks_applied` and `error`. Each prompt is a separate call to Grounding
DINO, so prompts should be short nouns. The whole image goes through Grounding DINO at once; for scenes larger than
a few thousand pixels on a side, cut tiles first and label each.

Needs `pip install groundingdino-py` and its config plus checkpoint files (not downloaded for you).
If `groundingdino` is missing, `success` is `False` and `error` says `pip install groundingdino-py for GroundedSAM`.

## `SamGeoLabeler`: via segment-geospatial

```python
from pygeovision.labeling import SamGeoLabeler
r = SamGeoLabeler(model_type="vit_h").label("data/aerial.tif", output_path="labels/samgeo.tif",
                                            output_vector="labels/samgeo.gpkg", min_size=50)
```

Result: `success`, `output_path`, `output_vector`, `vector_error`, `model_type`, `duration_seconds`. If samgeo finishes
without writing a file this is reported as a failure, not a success. Extra keyword arguments go straight to samgeo's
`generate()`.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `ImportError … transformers` | Optional dependency | `pip install transformers torch` |
| First call downloads gigabytes | Model weights | Normal; they are cached by Hugging Face (`cache_dir=` to choose where) |
| CUDA out of memory | Tile or model too large | `chip_size=512`, `sam-vit-base`, `device="cpu"` (slow) |
| `n_masks` is 0 and `error` says "survived … filtering" | Thresholds too strict, or objects smaller than `min_area_m2` | Lower `pred_iou_thresh` / `stability_score_thresh` / `min_area_m2` |
| `n_chip_errors > 0` | Some tiles failed (memory, bad data) | Read `error` for the first message; rerun with smaller `chip_size` |
| Masks are huge blobs covering everything | A "background" mask | Set `max_area_m2` |
| One building cut in two along a vertical/horizontal line | Tile edge | Increase `overlap`; keep `merge_overlapping=True` |
| Masks look wrong colour/texture-wise | Wrong bands as RGB | Set `rgb_bands` |
| It is very slow | Many points per tile on large scenes | `points_per_side=16`, a smaller model, a GPU |
| GroundedSAM finds nothing | Phrase too abstract or thresholds high | Lower `box_threshold`; use concrete nouns; check the imagery resolution |
| You need classes, not ids | SAM does not name things | [Naming segments](naming-segments.md) |

## Using the masks for training

1. Run `auto_label` on imagery of your area.
2. Name the segments with [`assign_classes`](naming-segments.md), using OSM or hand labels as the reference. This gives
   object-shaped outlines with classes from a map.
3. Pixels where no class wins are `255` and are ignored in training.
4. [Check the quality](quality.md) and look at the result over the imagery.

```{note}
`SAMAutoLabeler`, `grounded_label` and `SamGeoLabeler` need model weights and (practically) a GPU, which the
documentation build environment does not have. The tiling, merging, filtering, georeferencing and writing code was
tested with a stand-in mask generator. The behaviour of the actual models on your imagery is **not** covered by the tests.
```
