# DINOv3

DINOv3 is a large vision network from Meta trained **without labels** on a huge image collection. It does not give you classes;
it gives you **features**: a vector of numbers for every small patch of an image, such that patches that look alike get vectors
that are close. With those features you can cluster a scene, find "more like this" from one click, train a small classifier on
a few labelled examples, or use them as a frozen starting point for a dense model.

```{figure} ../_static/img/flow-dinov3.svg
:alt: DINOv3 workflow: a raster is read as three bands, scaled to patches, passed through a frozen backbone, and gives a patch grid of features, a global vector and attention maps
:width: 100%
:class: pgv-figure

The raster goes in whole (not cropped); a grid of feature vectors comes out, registered with the pixels.
```

```{admonition} DINOv3 repositories are gated
:class: note
Meta requires you to accept a licence on each model page and authenticate (`huggingface-cli login` or the `HF_TOKEN`
environment variable). Without access, loading raises a `ModelLoadError` that says so. Nothing is substituted: an earlier
version silently loaded DINOv2 under this name, which was removed.
```

## Which model

`list_dinov3_models()` returns the twelve names; each is a different size or pre-training set. Parameter counts and feature sizes
below are read from the package's own table (`get_dinov3_info`).

| Name | Family | Parameters | Feature size | Pretrained on | Pick it for |
|---|---|---:|---:|---|---|
| `dinov3_vits16` | ViT | 21 M | 384 | LVD-1689M (web images) | Quick tests, laptops |
| `dinov3_vits16plus` | ViT | 29 M | 384 | LVD-1689M | A little more than the smallest |
| `dinov3_vitb16` | ViT | 86 M | 768 | LVD-1689M | A balanced everyday choice for photographs |
| `dinov3_vitl16` | ViT | 300 M | 1024 | LVD-1689M | Strong features on photographs |
| `dinov3_vith16plus` | ViT | 840 M | 1280 | LVD-1689M | Large GPU |
| `dinov3_vit7b16` | ViT | 6,700 M | 4096 | LVD-1689M | Research-scale hardware |
| **`dinov3_vitl16_sat`** | ViT | 300 M | 1024 | **SAT-493M (satellite)** | **Default for satellite imagery** |
| `dinov3_vit7b16_sat` | ViT | 6,700 M | 4096 | SAT-493M | The largest satellite model |
| `dinov3_convnext_tiny` / `_small` / `_base` / `_large` | ConvNeXt | 29 / 50 / 89 / 196 M | 768 / 768 / 1024 / 1536 | LVD-1689M | Convolutional features; feature map at a coarser grid |

Models ending in `_sat` were pretrained on satellite imagery and are given the **satellite normalisation**; the others were
pretrained on web images and get ImageNet normalisation. The loader picks the matching one, so you never set it. For aerial or
satellite scenes start with `dinov3_vitl16_sat`; for close-range photographs, drone obliques or maps, a web model is the better start.

## Features from a raster

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone

backbone = DINOv3Backbone("dinov3_vitl16_sat", device="cuda", max_size=1024, rgb_bands=(3, 2, 1))
dense = backbone.extract_features("scene.tif")        # (H_p, W_p, D): one vector per patch, registered with the raster
vector = backbone.extract_embeddings("scene.tif")     # (1, D): one global descriptor for the whole image
```

| Argument | Default | Meaning |
|---|---|---|
| `model_name` | `dinov3_vitl16_sat` | A name from the table above |
| `method` | `"hf"` | `"hf"`: download from Hugging Face. `"local"`: load a snapshot you already have (needs `weights_path`), for machines without internet |
| `device` | auto | `"cuda"`, `"mps"` or `"cpu"` |
| `weights_path` | `None` | Folder of a `transformers` snapshot, with `method="local"` |
| `max_size` | 1024 | If the longer side is larger, the raster is shrunk to this before the model sees it. Lower to save memory; the grid gets coarser |
| `rgb_bands` | first three | **1-based** file band numbers shown to the model as red, green, blue. For Sentinel-2 `(3, 2, 1)` if the file is ordered B02, B03, B04, B08 |

The model sees **three bands**. Anything else in the file (near-infrared, radar) is ignored unless you choose it through `rgb_bands`
(a false-colour view such as `(4, 3, 2)` is legitimate).

### What comes out, measured

A 300 × 500 pixel, 4-band file read through a small random model of the same architecture (the shapes are properties of the code,
not of the weights):

| `max_size` | Feature grid `(H_p, W_p, D)` | `last_geometry["cell_px"]` | `["scale"]` |
|---:|---|---:|---:|
| 1024 | `(19, 32, D)` | 16.0 | 1.0 |
| 256 | `(10, 16, D)` | 31.25 | 0.512 |

- The image is **not** centre-cropped. It is shrunk if needed, padded by repeating the edge up to a multiple of the patch size (16),
  and fed whole: 300 rows become 304 and give 19 patch rows; 500 columns become 512 and give 32.
- Feature cell `(i, j)` covers source pixels `[i·cell_px, (i+1)·cell_px) × [j·cell_px, (j+1)·cell_px)`. `backbone.last_geometry`
  holds `cell_px`, `scale`, `grid` and `source_size` after every call, so you can map a cell to pixels and then to map coordinates
  with the file's transform.
- `extract_embeddings` returns the CLS token for ViTs (a pooled map for ConvNeXt), shape `(1, D)`.
- `extract_patch_features` returns the same grid flattened row by row, `(N_patches, D)`, with the register tokens removed.
- A raster is read nodata-aware and stretched per band (2nd to 98th percentile over the valid pixels). A NumPy **array** must be
  `(H, W, C)`; `uint8` is divided by 255 and floats are **clipped to 0..1** (they are not stretched), so reflectance stored as
  0..10000 must be scaled by you first or every pixel becomes 1.

## What to do with the features

### Cluster a scene without any labels

```python
import numpy as np
from sklearn.cluster import KMeans

feats = backbone.extract_features("scene.tif")
gh, gw, d = feats.shape
labels = KMeans(n_clusters=6, n_init=4, random_state=0).fit_predict(feats.reshape(-1, d)).reshape(gh, gw)

cell = int(round(backbone.last_geometry["cell_px"]))
cluster_map = np.kron(labels, np.ones((cell, cell), dtype=labels.dtype))[:300, :500]   # back to pixel size
```

The clusters have no names. Look at them over the imagery and name them yourself; two clusters may be one class (shadow and water),
and one cluster may mix classes. Use this to *explore* a scene, or to choose where to label.

### "More like this" from one click

```python
row, col = 120, 200                                    # a pixel you picked
cell = backbone.last_geometry["cell_px"]
i, j = int(row // cell), int(col // cell)
unit = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
similarity = unit @ unit[i, j]                         # (H_p, W_p), 1.0 = the clicked patch itself
```

Threshold `similarity` (start at 0.6 and look) to find look-alike patches. This is a cheap way to collect training examples.

### A classifier from a few labels

```python
model = backbone.build_classifier(num_classes=5, freeze_backbone=True)
```

`build_classifier` returns a model that is the backbone plus a small head on the global descriptor. With `freeze_backbone=True` only
the head trains, which works with very few labelled chips (tens per class). It expects normalised `(B, 3, H, W)` tensors, so
prepare chips with the transform that matches the model: `get_transform(model_name)` returns the web or satellite one.

```python
from pygeovision.models.foundation.dinov3 import get_transform, finetune_dinov3

transform = get_transform("dinov3_vitl16_sat")         # satellite statistics, resize 256 and centre-crop 224
setup = finetune_dinov3("dinov3_vitl16_sat", task="classification", num_classes=5, epochs=30)
model, optimizer, scheduler = setup["model"], setup["optimizer"], setup["scheduler"]
```

`finetune_dinov3` **prepares** a fine-tune (model, AdamW, cosine schedule, mixed precision, checkpoint manager) and returns them;
it does not train. Run them in your own loop or use [the callbacks page](../training/callbacks.md) for the checkpoint and
early-stopping classes. Only `task="classification"` exists; for segmentation, put a head of your own on `extract_features`
(or use [a segmentation model](../segmentation/models.md)). `backbone.finetune_config()` returns starting hyper-parameters
(AdamW, learning rate 1e-4, weight decay 0.05, 10 warm-up epochs); lower the learning rate to about 1e-5 for the satellite
backbones and whenever you unfreeze the backbone.

### Where the model looks

```python
attention = backbone.get_attention_maps("scene.tif")            # (H_p, W_p): the mean over heads
head_3 = backbone.get_attention_maps("scene.tif", head_idx=3)   # one head
```

The map shows how much the CLS token attends to each patch in the last layer: a rough picture of which parts of the scene the
global descriptor is made from. Attention maps need *eager* attention (the fast default cannot return the weights). The method tries to switch the model over
itself and raises a clear error, rather than returning a placeholder map, when it cannot.

## Beyond features

| Tool | What it is | Page |
|---|---|---|
| `CHMv2Model` | Canopy-height regression with Meta's released CHMv2 head on a DINOv3 backbone | [Canopy height](../lidar/canopy-height.md) |
| `DINOv3Text` | **Not available.** Zero-shot text alignment needs a checkpoint that is not public; the class raises when called | [CLIP](clip.md) does text search today |

Things that were removed because they were not sound: `estimate_biomass` (the equation had no source) and the old `load_dinov3_hub`
(it loaded DINOv2 under a DINOv3 name).

## Offline machines

```python
backbone = DINOv3Backbone("dinov3_vitl16_sat", method="local", weights_path="/models/dinov3-vitl16-sat", device="cpu")
```

Download the snapshot on a machine with access (`huggingface-cli download facebook/dinov3-vitl16-pretrain-sat493m --local-dir ...`),
copy the folder over, and point `weights_path` at it.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ModelLoadError: Could not load dinov3_... DINOv3 repositories are gated` | Licence not accepted, not logged in, or no network | Accept the licence on the model page and set `HF_TOKEN`; or use `method="local"` |
| `ValueError: rgb_bands (3, 2, 9) not available in a 4-band raster` | A band number larger than the file's band count | Bands are **1-based** and limited to the file |
| Features are all similar / clusters look random | A float array in 0..10000 was clipped to 1 | Pass the file path, or scale the array to 0..1 first |
| Out of GPU memory | Large scene, large model | Lower `max_size`, use `dinov3_vitb16`, or run on tiles |
| The grid is coarser than I need | `max_size` shrank the scene | Raise `max_size`, or cut the scene into tiles and extract each |
| `get_attention_maps` raises about attention weights | The model could not be switched to eager attention | Load it yourself with `load_dinov3_hf(name, attn_implementation="eager")` |
| Cluster boundaries are blocky | One cell is 16 pixels or more | That is the resolution of the features; refine edges with [OBIA](../obia/index.md) or a dense model |

## Related

- [Foundation models overview](index.md), [Embeddings](embeddings.md), [CLIP and RemoteCLIP](clip.md).
- [The foundation-model builders in the Python API](../reference/api/builders-foundation.md).
