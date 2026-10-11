# Foundation-model clusters

**Class:** `pygeovision.labeling.FoundationModelLabeler` · **Client:** `client.labeling.foundation(image_path, ...)`

(A second class with the same name exists for **tiles** and for multispectral models. See
[Tile labelers](tile-labelers.md#foundation-model-labeler-prithvi-remoteclip).)

## Why use it

A *foundation model* is a large network trained on huge amounts of imagery without labels. It turns each small square
of a picture into a list of numbers (an *embedding*) in which **similar-looking ground gets similar numbers**. Group the
squares by those numbers and you get an *unsupervised segmentation*: "these areas look alike, those look alike".

Use it when:

- you have **no labels at all** and want to see what is in the scene before deciding classes;
- you want a quick **first split** (water vs. fields vs. built-up) to hand to a person to name;
- you want a *check* on another label source: places where the clusters disagree with OSM are worth a look.

It does **not** produce class names, and cluster numbers mean nothing outside the scene they came from.

## How it works

1. The scene's red, green and blue bands are stretched (2nd–98th percentile, using only valid pixels) and cut into
   windows of `tile_size` pixels.
2. A **DINOv2** vision transformer reads each window. It produces one embedding per *patch* (14 × 14 pixels for DINOv2).
   Windows are not resized, so patch `(i, j)` covers exactly its own block of ground pixels.
3. The embeddings are grouped with k-means (or HDBSCAN).
4. Each cluster id is painted back onto the pixels its patch covers.

So the result is **blocky**: a 14-pixel grid. Water-versus-land boundaries are accurate to a patch, not to a pixel.

## How to use

```python
from pygeovision.labeling import FoundationModelLabeler

lab = FoundationModelLabeler(model="dinov2-base", rgb_bands=(3, 2, 1))     # a file stored B2,B3,B4 (blue, green, red)
r = lab.pseudo_label("data/scene.tif", "labels/clusters.tif", n_classes=8)
print(r["success"], r["n_classes"], r["cell_px"], r["valid_fraction"], r.get("error"))
```

### Constructor

| Argument | Default | Meaning |
|---|---|---|
| `model` | `"dinov2-base"` | `dinov2-small`, `-base`, `-large`, `-giant`, or any Hugging Face repo id with a ViT-style DINOv2 backbone |
| `device` | CUDA if available | `"cuda"`, `"cpu"` |
| `rgb_bands` | `(1, 2, 3)` (or one band repeated for single-band files) | 1-based bands rendered as R, G, B. A file stored blue, green, red (typical for Sentinel-2 B2, B3, B4) needs `(3, 2, 1)` |

### `pseudo_label`

| Argument | Default | Meaning |
|---|---|---|
| `image_path`, `output_path` | required | Input scene, output cluster raster |
| `n_classes` | 8 | k-means clusters. Ignored by HDBSCAN. Must be ≥ 2 |
| `clustering` | `"kmeans"` | `"kmeans"` or `"hdbscan"` (`pip install hdbscan`). HDBSCAN finds its own number; noise becomes 255 |
| `tile_size` | 448 | Window per forward pass, rounded down to a multiple of the patch size |
| `batch_size` | 8 | Windows per batch |
| `patch_size` | model's own | If given, must equal the model's patch size or `ValueError` |
| `clip_annotate` | `False` | Not implemented. `True` raises `NotImplementedError` (it used to be accepted and ignored) |

## What you get

```text
labels/clusters.tif    uint8, nodata=255. Values 0..n_classes-1 are cluster ids. 255 = no data
```

Result keys: `success`, `output_path`, `n_classes` (found), `n_patches` (clustered), `cell_px` (patch size),
`valid_fraction` (share of the scene that had data), `model`. On failure: `success=False` and `error`, for example
`"could not load 'dinov2-base': … Nothing was substituted."`.

The file's tags record the model, the number of clusters, the clustering method and the bands used.

```{admonition} Cluster 3 is not a class
:class: warning
Cluster numbers are arbitrary and **change from scene to scene** (and between runs on the same scene if you change
`n_classes`). "Cluster 3 = water" in one image says nothing about another. Never train a model across scenes using raw
cluster ids as classes. Give each scene's clusters names first ([Naming segments](naming-segments.md)).
```

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `"could not load 'dinov2-base' … Nothing was substituted"` | No network for the weights, or `transformers`/`torch` missing | Install them; download the weights once; no substitute model is used on purpose |
| `"N patches exceeds the 1500000 limit"` | The scene is very large | Downsample the raster or process it in blocks |
| `"rgb_bands … not available in a K-band raster"` | Band numbers out of range | Remember they are 1-based |
| `"raster has no valid pixels"` | All nodata | Check the file |
| Clusters follow image brightness or shadows, not land cover | Normal for unsupervised features | Raise `n_classes` to let shadows separate; mask shadows; use a different `rgb_bands` |
| Every run gives different cluster numbers | k-means numbering | The seed is fixed, so the same input and settings are repeatable; changing `n_classes` or the image changes numbering |
| Boundaries look like a grid | 14 px patches | Expected. Refine with SAM masks and [`assign_classes`](naming-segments.md) |
| `NotImplementedError: clip_annotate` | Not supported | Name clusters by hand or by a reference map |

## When it works well and when it does not

Works well for scenes with a few visually distinct land covers at a resolution where each patch is a coherent patch of
ground (very-high-resolution RGB: buildings, roads, trees, grass, water are separable). Works poorly on
10 m Sentinel-2 RGB, where one 14-pixel patch is 140 m on a side and mixes everything, and for classes that differ in
infrared (crop types, burned areas), because only the three visible bands are used.

```{note}
This needs the DINOv2 weights and `transformers`, which the documentation build environment does not have.
The tiling, registration of the patch grid to pixels, nodata handling and writing were tested with a stand-in
backbone; real-model cluster quality was not measured.
```
