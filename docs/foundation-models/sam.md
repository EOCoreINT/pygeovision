# SAM and SAM2

The Segment Anything Model (SAM) outlines objects **without knowing what they are**. Give it a click, or let it try a grid of clicks
over the whole image, and it returns the shapes of the things it finds. It never says "building" or "field": the names come from
you or from another source. That is exactly what makes it useful for **making labels fast** and for **cutting out one object you
point at**.

```{figure} ../_static/img/flow-sam.svg
:alt: SAM workflow: a raster becomes a three-band picture, prompts (a click or a grid) go to SAM, masks come back, filtered and named
:width: 100%
:class: pgv-figure

SAM finds shapes; naming them is a separate step.
```

## Two ways to use it

| You want | Use | Page |
|---|---|---|
| Outline **everything** in a scene, as an instance map and optionally polygons, over a large GeoTIFF | `SAMAutoLabeler` (also `pygeovision ai segment sam`) | [Segment Anything labelling](../labeling/sam.md) |
| Outline **one object** from a click, from Python, on an image or array | `get_model("sam-vit-b").predict_points(...)` | this page |
| Give the shapes **class names** | combine with a reference map | [Naming segments](../labeling/naming-segments.md) |

## The models

| Registry name | Hugging Face id | Parameters | Use it for |
|---|---|---:|---|
| `sam-vit-b` | `facebook/sam-vit-base` | 93.7 M | Fast, the default for trying things out |
| `sam-vit-l` | `facebook/sam-vit-large` | 308 M | A good balance |
| `sam-vit-h` | `facebook/sam-vit-huge` | 636 M | The most detailed outlines; needs a GPU to be comfortable |
| `sam2-hiera-l` | `facebook/sam2-hiera-large-hf` | 224.4 M | SAM 2, an improved architecture |

All four need the `transformers` library and download their weights on first use (a network is needed; they are not bundled).
SAM 2 needs a `transformers` version that has `Sam2Model`; an older one raises `ImportError ... upgrade transformers`.

## Outline one object from a click

```python
from pygeovision.models import get_model

sam = get_model("sam-vit-b")                 # a GeoSAM wrapper; weights load on the first call, not now
masks = sam.predict_points("tile.tif", [[120, 200]], input_labels=[1])      # a click at pixel x=120, y=200
```

```text
masks            -> a list with one entry per image
masks[0].shape   -> torch.Size([1, 3, H, W]), dtype torch.bool
```

| Argument | Meaning |
|---|---|
| `image` | A file path (read as three bands with a 2nd to 98th percentile stretch, a single band repeated to grey), a PIL image, or a NumPy `(H, W, 3)` array |
| `input_points` | A list of `[x, y]` **pixel** positions: `x` is the column, `y` is the row, counted from the top-left |
| `input_labels` | One per point: `1` means "this is part of the object", `0` means "this is not". Defaults to all `1` |

What comes back, measured with the real SAM architecture and random weights, so the shapes are exact and only the pictures are
meaningless:

- **Three masks per prompt.** SAM returns three candidates, usually from small to large (for a click on a window: the window, the wall,
  the whole building). Pick one by looking, or by the score if you ask the underlying model for it.
- Every mask is a boolean array at the **original image size** `(H, W)`.
- Several points in one call are **one prompt**: two `1` points mean "both of these belong to the same object", and a `0` point
  carves a region out. To outline two different objects, call twice.

Turn a chosen mask into polygons or a raster with the tools in [post-processing](../data/postprocessing.md), or
[vectorise](../inference/index.md) it on the file's grid.

```python
mask = masks[0][0, 1].numpy()            # candidate 1 (the middle one) of the first prompt: (H, W) bool
```

```{admonition} Pixel coordinates, not map coordinates
:class: warning
The click is in **pixels of the image you passed**, not in longitude/latitude or metres. Convert a map position with the file's
transform: `row, col = rasterio.transform.rowcol(src.transform, x_map, y_map)`, then use `[[col, row]]`.
```

## Everything in a scene

```python
result = sam.generate_masks("scene.tif", output_path="masks.tif", points_per_side=32, min_area_m2=10.0)
```

| Argument | Default | Meaning |
|---|---|---|
| `image_path` | | A GeoTIFF |
| `output_path` | `./output/sam_masks.tif` | An instance map: 0 is "no mask", 1 to N are objects (`uint32`) |
| `points_per_side` | 32 | The grid of clicks: 32 means 32 × 32 = 1,024 prompts per tile. More finds smaller objects and is slower |
| `min_area_m2` | 10.0 | Throw away masks smaller than this ground area |

This hands over to `SAMAutoLabeler`, sharing the weights already loaded. The result (a dictionary) reports `n_masks`, `n_raw_masks`,
`n_tiles`, `n_chip_errors`, `output_path`, `duration_seconds` and `quality_stats`. `success` is `False` when nothing survived the
filters or when tiles failed (and `error` says which). **The wrapper does not take** the other options of the labeller
(`rgb_bands`, `chip_size`, `overlap`, `max_area_m2`, `output_vector`): for those, call `SAMAutoLabeler.auto_label` directly, as
described on [the labelling page](../labeling/sam.md). The most common one is `rgb_bands=(3, 2, 1)` for Sentinel-2 files written
B2, B3, B4, B8, because the default bands 1 to 3 are only correct for a file stored red, green, blue.

## What SAM is good and bad at

| Works well | Struggles |
|---|---|
| Objects with clear edges: buildings, fields, tanks, ships, pools | Low-contrast boundaries (wet and dry soil, shadow on dark roofs) |
| Cutting out one object for a mask or a label | Very small objects at coarse resolution (a few pixels) |
| Producing candidate labels for a person to check | Anything that needs a **class**: SAM has none |
| Photographs and high-resolution aerial imagery | Medium-resolution imagery (10 to 30 m) where objects are only a few pixels wide |

SAM was trained on photographs. On satellite imagery, tune `points_per_side` and `min_area_m2` to your **pixel size**: at 0.3 m a
building is thousands of pixels and a grid of 32 is plenty; at 10 m a building is a handful of pixels and SAM will rarely find it.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: transformers ... has no SamModel / Sam2Model; upgrade transformers` | An old `transformers` | `pip install -U transformers` |
| The first call is slow | The weights are being downloaded and loaded | Normal; later calls reuse them |
| The mask is the whole image, or nothing useful | A click on a place with no clear object, or the wrong band order made the picture odd | Check the rendered picture; try `sam-vit-l`; add a `0` point to cut away the background |
| `n_masks` is 0 and `success` is False | Every mask was below `min_area_m2`, or quality thresholds were too strict | Lower `min_area_m2`; use `SAMAutoLabeler` to change `pred_iou_thresh` |
| The picture SAM sees looks blue-tinted | Band order | Use `SAMAutoLabeler(..., rgb_bands=(3, 2, 1))` |
| Out of memory | A large model or large tiles | Use `sam-vit-b`; lower `chip_size` in `SAMAutoLabeler` |

## Related

- [Segment Anything labelling](../labeling/sam.md), [Naming segments](../labeling/naming-segments.md), [Foundation models overview](index.md).
- [Segmentation models](../segmentation/models.md) lists SAM next to the networks that do give classes.
- [`pygeovision.models.segmentation.sam` in the Python API](../reference/api/builders-segmentation.md).
