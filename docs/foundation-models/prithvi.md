# Prithvi-EO

Prithvi-EO was pretrained by NASA and IBM on six-band HLS imagery (Harmonized Landsat Sentinel-2). When you have few labels for flood,
burn scars, crops or land cover, fine-tuning Prithvi usually needs fewer examples than training a network from nothing. This page covers what
goes in, what you can get out, how to train a head on top, and which parts were checked here.

```{figure} ../_static/img/flow-prithvi.svg
:alt: Using Prithvi-EO: prepare six HLS bands as reflectance, load the encoder, take features or build a segmentation head, train the head on labels, run a task with the method named
:width: 100%
:class: pgv-figure

The encoder is pretrained; the head is yours. Without a head the task falls back to rules and says so.
```

```{admonition} What was tested
:class: note
Run here, with **no real weights** (no network): the band mapping and validation, the shapes of every output, the segmentation head and its
parameter count, the saved-head file, the multi-date tools, and the spectral fallbacks (`tests/test_docs_prithvi_offline.py`). For the shapes a
random-weights stand-in with the same interface was used (`allow_random_init=True`), so **the numbers it produces mean nothing**.
**Not run here:** the real checkpoints, a fine-tuning run, or any accuracy figure. Nothing on this page claims Prithvi is accurate on your data.
```

## What it expects

Six bands in **HLS order**: Blue, Green, Red, narrow NIR, SWIR-1, SWIR-2. Sentinel-2 bands `B02, B03, B04, B8A (or B08), B11, B12` map to them
and Landsat `B2…B7` likewise. Pass **surface reflectance in [0, 1]**; the wrapper standardises it the way the weights were trained. (Dividing by
10 000 and stopping there, which earlier versions did, feeds the network inputs of the wrong scale.)

`map_bands(data, source="sentinel2", n_prithvi_bands=6, source_bands=None)` reorders an array `(C, H, W)` into that order. Verified details:

| Situation | What it does |
|---|---|
| `source="hls"` | The data is already in order; the first six bands are taken |
| The stack has both `B08` and `B8A` | `B8A` is used, because HLS's NIR band is the narrow one. `B08` is ignored |
| The stack has only `B08` | It is used as a stand-in, with the log line `using broad-NIR B08 for Prithvi's narrow-NIR band (HLS uses B8A); expect a small spectral mismatch` |
| Red-edge bands (`B05`, `B06`, `B07`) | Dropped: Prithvi takes six bands |
| A band you did not supply | Its output channel stays **zero**, silently. Pass `source_bands=[...]` to name what each input channel is |

`validate_prithvi_input(data)` checks an array that is already in HLS order and scaled. It returns `{"valid", "errors", "warnings", "stats"}`:

| Input | Result |
|---|---|
| Values up to 3000 | **Error:** `Values up to 3000 look like raw DN integers. Divide by 10000.0 before calling Prithvi.` |
| Values above 1.5 but at most 5 | Warning: are you sure it is normalised? |
| Fewer than six bands | **Error:** `Only 4 bands; Prithvi needs 6` |
| NaN values | Warning: `1 NaN pixels — will be set to 0 before inference` |
| A dtype other than float32 | Warning; it is cast |

When a GeoTIFF is read by the model wrappers, three things happen to it that you may not expect:

1. **Resampling to 30 m.** If the pixel is finer than 20 m (Sentinel-2 at 10 m), the raster is averaged down to 30 m, the resolution Prithvi was
   pretrained on. Pass 10 m data directly and the features degrade.
2. **Automatic scale detection.** A maximum above 10 means raw integers, and the data is divided by 10 000; a maximum between 1.5 and 10 is clipped
   to [0, 1] with a warning.
3. **NaN and infinity become 0**, because the encoder cannot take them.

## Load it

```python
from pygeovision.models.foundation.prithvi import Prithvi

prithvi = Prithvi("prithvi_eo_2_0").load()           # downloads the weights at a pinned revision; needs a network the first time
prithvi = Prithvi("prithvi_eo_2_0", device="cuda")    # device=None picks cuda, then mps, then cpu
prithvi.load(weights_path="Prithvi_EO_V2_300M.pt", config_path="config.json", code_path="prithvi_mae.py")   # fully local
```

| Variant | Parameters | Embedding size | Time steps | Patch | Trained on |
|---|---:|---:|---:|---:|---|
| `prithvi_eo_1_0` | 100 M | 768 | 3 | 16 | Contiguous US |
| `prithvi_eo_2_0` | 300 M | 1024 | 4 | 16 | Global |
| `prithvi_eo_2_0_600m` | 600 M | 1280 | 4 | 14 | Global |
| `prithvi_eo_1_0_finetuned_burn`, `prithvi_eo_1_0_finetuned_flood` | 100 M | | | | **Reference only**: these are TerraTorch artefacts, not backbone checkpoints, and cannot be loaded here. Use TerraTorch |

In the native registry these are `prithvi-100m`, `prithvi-300m`, `prithvi-600m` (and the underscore names above). `get_model("prithvi-300m")`
has **no architecture-only build**: use `pretrained=True`, or `allow_random_init=True` for a flagged, randomly initialised stand-in meant for tests only.
A failed download without `allow_random_init` raises `PrithviLoadError` (a `ModelLoadError`) that names the file; it never returns random weights quietly. With
`allow_random_init=True` it logs `Prithvi stand-in built (random weights, NOT the Prithvi architecture): outputs are meaningless.`

```{admonition} Prithvi loads code from the Hugging Face repository
:class: warning
The Prithvi repositories are not standard `transformers` models. PyGeoVision downloads `prithvi_mae.py`, `config.json`
and the checkpoint at a **pinned commit**, imports that file and loads the weights with `weights_only=True`. Importing
the file runs code from the repository. Review the diff before changing the pinned `revision`, and use
`load_prithvi_local(...)` or `Prithvi.load(weights_path=..., code_path=...)` for air-gapped machines.
```

## Features

```python
vector  = prithvi.extract_features("scene_hls6.tif", source="hls", pool="mean")     # (1, 1024)
patches = prithvi.extract_patch_features("scene_hls6.tif")                          # (196, 1024)
```

| Method | Returns | Notes |
|---|---|---|
| `extract_features(path, source="hls", pool="cls")` | `(1, embed_dim)` | `pool="cls"` is the class token, `pool="mean"` the mean of the patch tokens. An MAE encoder's class token is **not trained for classification**, so `"mean"` is usually the better descriptor. Anything else raises `ValueError` |
| `extract_patch_features(path, source="hls")` | `(patches, embed_dim)` in row-major order | 196 = a 14 × 14 grid of 16-pixel patches on a 224-pixel tile |

**The whole scene is resized to one 224 × 224 tile** first. So `extract_features` describes the scene, not a 30 m pixel, and a large scene loses
detail before the model sees it. For a map, cut the scene into 224-pixel chips yourself, or use the segmentation head and tiled inference.

## Fine-tune a segmentation head

```python
model = prithvi.build_segmentation_head(num_classes=2, freeze_backbone=True)   # random head on a frozen encoder
# ... train with GeoTrainer, then save only the head:
from pygeovision.models.foundation.prithvi import save_prithvi_head
save_prithvi_head(model, "heads/flood.pt", variant="prithvi_eo_2_0", task="flood", num_classes=2)
```

`build_segmentation_head` returns the encoder plus a **randomly initialised** head: a 3 × 3 convolution from the embedding size to 256 channels,
batch normalisation, ReLU, and a 1 × 1 convolution to the classes. Its output is `(batch, classes, 224, 224)` for an input `(batch, 6, 224, 224)`: the
patch grid is upsampled bilinearly to the input size. For the 300 M model (embedding 1024) the head has **2 360 835 trainable parameters
for three classes**. The `freeze_backbone=True` setting also keeps the encoder in evaluation mode while training, so its normalisation statistics do not drift.
Train it like any segmentation model ([Training](../training/index.md)): its loss, optimiser and checkpoints are the same.

`prithvi.finetune_config()` returns suggested settings (AdamW, learning rate 5e-5, weight decay 0.01, 5 warm-up epochs, cosine schedule, bf16, batch 8, and
"freeze the backbone for datasets under 10 000 samples"). They are a starting point.

`save_prithvi_head` writes only the head (about 9 MB for three classes) with its variant, task and class count; loading it into the wrong variant or
a different number of classes fails instead of loading wrongly.

## Run a task: heads or rules

`PrithviTasks` runs task inference. What it does depends on whether you supply a trained head:

```python
from pygeovision.models.foundation.prithvi import PrithviTasks

tasks = PrithviTasks("prithvi_eo_2_0", head_weights={"flood": "heads/flood.pt"})
r = tasks.flood_detection("preprocessed.tif", source="sentinel2", output_path="flood.tif")
r["method"]          # "backbone" with a head, "spectral" without
```

The key of `head_weights`, and the `task` you give `save_prithvi_head`, is one of `land_cover`, `flood`, `burn_scar`, `crop_mapping`, `biomass`. The
method-style spellings `flood_detection`, `burn_scar_detection` and `biomass_estimation` are accepted as aliases. **Any other key raises `ValueError`**:
an earlier version silently ignored an unknown key and ran the spectral rule, so a head you thought was in use was not. A head is also checked against the
variant, task and class count it was saved with, and a mismatch raises (`... was trained for task='flood', not 'burn_scar'`).

| Method | With a head | Without |
|---|---|---|
| `land_cover()` | Prithvi + your head (`method == "backbone"`), nine classes | Spectral rules from NDVI/MNDWI/NDBI (`method == "spectral"`): heuristics that do **not** use Prithvi |
| `flood_detection()`, `burn_scar_detection()` | Prithvi + your head, two classes | Spectral rules (MNDWI, NBR) |
| `crop_mapping()`, `biomass_estimation()` | Prithvi + your head (a regression head for biomass) | `NotImplementedError`: there is no honest spectral rule |

Verified without a head, on a 256 × 300 test scene: each result carries `prediction` (the same shape as the scene), `class_pct`, `n_classes`, `task`, `model`,
`method`, `source` and a note, here `spectral-index rules on reflectance; Prithvi was not used. Pass head_weights for a learned prediction.` Flood
detection also returns `flood_pct` and `class_names` (`dry / no flood`, `flooded / water`). **Check `result["method"]` on every result**: it tells you
which route produced the map. Without a head, `crop_mapping` raises `NotImplementedError: Crop-type mapping cannot be done honestly from spectral thresholds on one date...` See the [`flood_mapping`](../pipelines/catalog/flood_mapping.md) and [`wildfire_severity`](../pipelines/catalog/wildfire_severity.md) pages for the methods that need no model.

## Several dates at once

`PrithviMultiTemporal` encodes several dates together (the encoder attends over all of them). It is built for a fixed number of dates, so the model is
reloaded when you pass a different count. All of these describe how the **learned features** change; they are screening indicators.

| Method | Needs | Returns |
|---|---|---|
| `process_time_series(paths, dates=None, source="hls")` | one or more images | `frame_embeddings (T, D)`, `frame_tokens (T, N, D)`, `features (1 + T·N, D)` (class token first), `grid (14, 14)`, `n_frames`, `dates`, `model` |
| `detect_change(before, after, source="hls", output_path=None)` | two images | `change_map` (the per-patch feature distance, scaled by its maximum, on the 14 × 14 grid), `raw_distance`, `change_pct` (the share of patches above half of the maximum), and with `output_path` a one-band GeoTIFF resized to the scene |
| `monitor_trend(paths, dates=None)` | three or more | `drift_from_first` per date and `trend_direction` (`diverging`, `converging`, `stable`). Fewer than three dates returns `trend: insufficient_data (need ≥ 3 time steps)` |
| `predict_seasonal(paths, dates=None)` | four or more | The first principal component of the date embeddings: `seasonal_amplitude`, `peak_step`, `trough_step`, `peak_date`, `trough_date`. Fewer than four returns a note |

A large feature difference means "the scene looks different", not "the land changed": clouds, haze, sun angle and season also change features, and
`change_pct` is a **relative** share of patches, not an area. The change map is 14 × 14 patches, each covering one fourteenth of the scene's width (the whole scene is resized to one tile), so it is coarse;
use it to find where to look, then confirm with a pixel-level method ([change detection](../change-detection/index.md)).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `PrithviLoadError` or `ModelLoadError` | No network, wrong pinned file, or a layer the checkpoint did not fill | Read the message; it names the file. Use `load_prithvi_local` for offline use |
| Output looks random | Wrong band order or unit scale, or the stand-in is loaded | Use `map_bands(...)`, reflectance in [0, 1], then `validate_prithvi_input`; check the log for "stand-in" |
| `Values up to 3000 look like raw DN integers` | Integers were not divided by 10 000 | Divide, or let the wrapper detect it (values above 10) |
| `[Prithvi input] ...` `ValueError` from a task | The validation error above, raised | Fix the input |
| `NotImplementedError: Crop-type mapping cannot be done honestly…` | No trained head supplied | Train a head and pass `head_weights={"crop_mapping": path}` |
| `ValueError: head_weights has the unknown task 'floods'` | A misspelt task key | Use one of the five names or the aliases |
| `method` is `spectral` although you passed a head | An older version ignored a wrong key; now it raises | Update; use the exact task name |
| `ValueError: pool must be 'cls' or 'mean'` | A typo | Use one of the two |
| `trend: insufficient_data` | Fewer than three dates | Pass more dates |
| A channel is all zero in the output of `map_bands` | A band was not in the stack | Name the bands with `source_bands=[...]` |
| Memory error | 300 M and 600 M models are large | Use the 100 M model, CPU for small chips, or a larger GPU |
| `ImportError: einops` | Optional dependency | `pip install einops timm` |
| A fine-tuned checkpoint "does not load" | The two downstream entries are TerraTorch artefacts | Use TerraTorch for those |

## Related

- [Foundation models overview](index.md), [Radar with Prithvi](radar.md), [Embeddings](embeddings.md), [DOFA](dofa.md)
- [Fine-tuning](../training/fine-tuning.md), [Training callbacks](../training/callbacks.md)
- [`pygeovision.models.foundation.prithvi` in the Python API](../reference/api/builders-foundation.md)
