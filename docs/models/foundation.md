# Foundation models

A *foundation model* is a large network pretrained on huge amounts of unlabelled or loosely labelled imagery. It does
not solve your task directly. It gives you a strong **starting point**: either *features* (a vector that describes an
image or a patch, usable for search, clustering or a small classifier) or *weights to fine-tune* with far fewer labels
than training from scratch.

| Model | What it learned from | Use it for | Needs |
|---|---|---|---|
| [Prithvi-EO](#prithvi-eo) | NASA/IBM: Harmonized Landsat-Sentinel-2 (HLS), 6 bands | Fine-tuned segmentation (flood, burn, crop, land cover); change in feature space | Hub download, `einops`, `timm` |
| [DOFA](#dofa) | Many sensors, any band set | One model for Sentinel-2, NAIP, Sentinel-1… | `torchgeo` |
| [DINOv3](#dinov3) | Meta: web images (LVD) or satellite images (SAT-493M) | General visual features, dense features | **Gated** Hub access |
| [CLIP / RemoteCLIP](#clip-and-remoteclip) | Image-text pairs | Zero-shot labels, text search | `transformers` or `open_clip_torch` |
| [SAM / SAM2](#sam-and-sam2) | Promptable segmentation | Outlining objects from points | Hub download |
| [TESSERA](#embeddings) | Cambridge: a year of Sentinel-1/2 | Per-pixel embeddings, ready made | `geotessera` |
| [AlphaEarth](#embeddings) | Google DeepMind: multi-sensor annual | Per-pixel embeddings, ready made | Earth Engine account |

```{admonition} A foundation model is not a map
:class: warning
Feeding an image to Prithvi gives you *features*. To get "flooded / not flooded" you must put a head on top and train it
on labels, or use another method. PyGeoVision follows this strictly: where no trained head is supplied the task falls
back to a documented spectral rule and says so in the result, or it refuses, rather than returning random class maps.
```

## Prithvi-EO

**Why.** Prithvi-EO was pretrained by NASA and IBM on six-band HLS imagery. When you have few labels for flood, burn
scars, crops or land cover, fine-tuning Prithvi usually needs fewer examples than training a network from nothing.

**What it expects.** Six bands in **HLS order**: Blue, Green, Red, narrow NIR, SWIR-1, SWIR-2. Sentinel-2 bands
`B02, B03, B04, B08 (or B8A), B11, B12` map to them and Landsat `B2…B7` likewise. Pass **surface reflectance in [0, 1]**;
the wrapper standardises it the way the weights were trained. (Dividing by 10 000 and stopping there, which earlier
versions did, feeds the network inputs of the wrong scale.)

**How.**

```python
from pygeovision.models.foundation.prithvi import Prithvi, map_bands

prithvi = Prithvi("prithvi_eo_2_0").load()                     # downloads the weights (pinned revision)
vector = prithvi.extract_features("scene_hls6.tif")             # scene-level feature vector (np.ndarray)
patches = prithvi.extract_patch_features("scene_hls6.tif")      # (patches, embed_dim) in row-major order
```

| Variant | Parameters | Trained on |
|---|---:|---|
| `prithvi_eo_1_0` | 100 M | Contiguous US, 3 time steps |
| `prithvi_eo_2_0` | 300 M | Global, 4 time steps |
| `prithvi_eo_2_0_600m` | 600 M | Global, 4 time steps |

In the native registry these are `prithvi-100m`, `prithvi-300m`, `prithvi-600m` (and the underscore names above).
`get_model("prithvi-300m")` has **no architecture-only build**: use `pretrained=True`, or `allow_random_init=True`
for a flagged, randomly initialised stand-in meant for tests only.

```{admonition} Prithvi loads code from the Hugging Face repository
:class: warning
The Prithvi repositories are not standard `transformers` models. PyGeoVision downloads `prithvi_mae.py`, `config.json`
and the checkpoint at a **pinned commit**, imports that file and loads the weights with `weights_only=True`. Importing
the file runs code from the repository. Review the diff before changing the pinned `revision`, and use
`load_prithvi_local(...)` for air-gapped machines.
```

**Fine-tuning a segmentation head.**

```python
model = prithvi.build_segmentation_head(num_classes=2, freeze_backbone=True)   # random head on a frozen encoder
# ... train with GeoTrainer, then save only the head:
from pygeovision.models.foundation.prithvi import save_prithvi_head
save_prithvi_head(model, "heads/flood.pt", variant="prithvi_eo_2_0", task="flood_detection", num_classes=2)
```

**Using trained heads.** `PrithviTasks` runs task inference. What it does depends on whether you supply a trained head:

| Task | With `head_weights={task: path}` | Without |
|---|---|---|
| `land_cover` | Prithvi + your head (`method == "backbone"`) | Spectral rules from NDVI/MNDWI/NDBI (`method == "spectral"`): heuristics that do **not** use Prithvi |
| `flood_detection`, `burn_scar_detection` | Prithvi + your head | Spectral rules (MNDWI, NBR) |
| `crop_mapping`, `biomass_estimation` | Prithvi + your head | `NotImplementedError`: there is no honest spectral rule |

Check `result["method"]` on every result: it tells you which route produced the map. See the
[`flood_mapping`](../pipelines/catalog/flood_mapping.md) and [`wildfire_severity`](../pipelines/catalog/wildfire_severity.md)
pages for the methods that need no model.

**Multi-date analysis in feature space.** `PrithviMultiTemporal` encodes several dates and reports how far each date has
drifted from the first (`monitor_trend`), a feature-space change map between two dates (`detect_change`) and the
dominant seasonal pattern (`predict_seasonal`). These are exploratory: a large feature difference means "the scene looks
different", not "the land changed" (clouds and season also change features).

**What to do when…**

| Symptom | Cause | Fix |
|---|---|---|
| `PrithviLoadError` or `ModelLoadError` | No network, wrong pinned file, or a layer the checkpoint did not fill | Read the message; it names the file. Use `load_prithvi_local` for offline use |
| Output looks random | Wrong band order or unit scale | Use `map_bands(...)`, reflectance in [0, 1], then `validate_prithvi_input` |
| `NotImplementedError: Crop-type mapping cannot be done honestly…` | No trained head supplied | Train a head and pass `head_weights` |
| Memory error | 300 M/600 M models are large | Use the 100 M model, CPU for small chips, or a larger GPU |
| `ImportError: einops` | Optional dependency | `pip install einops timm` |

## DOFA

**Why.** DOFA (Dynamic One-For-All) accepts **any set of bands** by telling the network each band's wavelength, so one
model handles Sentinel-2 (9 bands), NAIP (3 bands) or Sentinel-1.

```python
from pygeovision.models.foundation.dofa import load_dofa_hf, extract_dofa_embedding, get_dofa_wavelengths

get_dofa_wavelengths("sentinel2")   # [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]  micrometres, 9 bands
get_dofa_wavelengths("naip")        # [0.665, 0.56, 0.49]

model = load_dofa_hf("dofa_base_patch16_224")                  # needs torchgeo; real weights, no silent substitute
embedding = extract_dofa_embedding(model, image_array, sensor="sentinel2")    # 768-number vector
```

Pass `sensor=` for a known sensor or `wavelengths=[…]` (micrometres, one per band, in your band order) for anything
else. The wavelengths are the whole point: give a wrong list and the embedding is wrong without any error.
If `torchgeo` is missing you get `ImportError: DOFA requires torchgeo`; `pip install torchgeo`.

## DINOv3

**Why.** DINOv3 produces high-quality general visual features, including **dense** features aligned with the raster,
useful for clustering, similarity and as a frozen encoder.

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone

backbone = DINOv3Backbone("dinov3_vitl16_sat", rgb_bands=(3, 2, 1))   # satellite-pretrained; 1-based bands for red, green, blue
dense = backbone.extract_features("scene.tif")        # (H_patches, W_patches, D), registered with the raster
vector = backbone.extract_embeddings("scene.tif")     # (1, D) global descriptor
```

Models ending `_sat` were pretrained on satellite imagery (SAT-493M) and use satellite normalisation; the others on web
images (LVD-1689M) and use ImageNet normalisation. The loader picks the right one.

```{admonition} DINOv3 repositories are gated
:class: note
Meta requires you to accept a licence on each model page and authenticate (`huggingface-cli login` or the `HF_TOKEN`
environment variable). Without access, loading raises a `ModelLoadError` that says so. Nothing is substituted: an earlier
version silently loaded DINOv2 under this name, which was removed.
```

## CLIP and RemoteCLIP

**Why.** CLIP embeds images and text in the same space, so you can ask "which of these descriptions fits this image?"
or "find images that look like *flooded fields*" with no training.

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
clip = client.vlm.clip("remoteclip-b32")
clip.zero_shot("tile.tif", ["forest", "water", "urban", "agriculture"])   # {category: probability}, sums to 1
clip.search("airport runway", "./tiles", top_k=5)                        # list of {path, score} best first
```

`openclip-b32`/`-l14` and `clip-vit-b32` load through `transformers`. `remoteclip-b32`/`-l14` are remote-sensing
fine-tunes in `open_clip` format: install `open_clip_torch`. RemoteCLIP is usually the better choice for aerial and
satellite imagery.

**Limits.** CLIP sees three bands (red, green, blue) and responds to the words you choose: "farmland" and "agriculture"
can rank differently. Probabilities are *relative* between your categories (they always sum to 1), so a tile that is none
of your categories still gets a confident-looking answer. Include a "something else" category and validate on labelled
tiles before relying on it.

## SAM and SAM2

**Why.** The Segment Anything Model outlines objects from a point or automatically across the whole image, without
class labels. It is the engine behind `pygeovision ai segment sam` and `buildings`, and the auto-labeller
([Labeling](../labeling/index.md)).

```python
from pygeovision.models import get_model

sam = get_model("sam-vit-b")                     # a GeoSAM wrapper; weights load on first use
result = sam.generate_masks("scene.tif", output_path="masks.tif", points_per_side=32, min_area_m2=10.0)
masks = sam.predict_points(image, [[120, 200]], input_labels=[1])    # a click at pixel (x=120, y=200)
```

`sam-vit-b` (94 M) is the lightest, `sam-vit-h` (636 M) the most detailed; `sam2-hiera-l` is SAM 2. SAM finds *shapes*,
not meanings: you decide which masks are buildings, usually by combining them with another source. It was trained on
photographs, so tune `points_per_side` and `min_area_m2` to your pixel size.

## Embeddings

Two products give you per-pixel embeddings that someone else computed, so you do not run a large encoder yourself:

| Product | Output | Source |
|---|---|---|
| **TESSERA** | 128 channels per pixel at 10 m, one year of Sentinel-1 + Sentinel-2 | `geotessera` library |
| **AlphaEarth** (Satellite Embedding) | 64 channels per pixel at 10 m, annual from 2017 | Google Earth Engine |

```python
from pygeovision.models.foundation.tessera import TesseraGeo

t = TesseraGeo()
t.embeddings_for_bbox((0.20, 6.70, 0.45, 6.95), year=2024, output_path="tessera.tif")   # multi-band GeoTIFF
t.embeddings_at_points([(0.31, 6.82), (0.33, 6.80)], year=2024)                          # (N, 128) for a classifier
t.pca_visualisation((0.20, 6.70, 0.45, 6.95), "tessera_pca.tif")                         # 3-band picture to look at
```

A common workflow: sample embeddings at a few hundred labelled points, train a simple classifier (logistic regression,
k-nearest-neighbours) on them, then predict for every pixel. The authors report that simple classifiers are competitive
on these embeddings. AlphaEarth works the same way through `AlphaEarthGeo` (`embeddings_at_points` is immediate;
`embeddings_for_bbox` and `similarity_map` start an Earth Engine export task and return its description rather than the
raster).

### Comparing embeddings

`GeoEmbeddings` wraps the extractors under one convention (a flat vector per image) and offers similarity search:

```python
from pygeovision.models.embeddings import GeoEmbeddings, cosine_similarity, nearest

e = GeoEmbeddings()
vectors = {p: e.clip(p) for p in ["a.tif", "b.tif", "c.tif"]}      # RemoteCLIP by default
nearest(e.clip("query.tif"), vectors, top_k=2)                     # [(path, similarity), ...], best first
```

`cosine_similarity(a, b)` runs offline on any two vectors. Each extractor keeps its own real input requirements: DOFA
needs an array plus a sensor, Prithvi a file path, TESSERA a bounding box. The class does not hide that.

## Vision-language

`MoondreamGeo` (Moondream2, ~1.8 B parameters) writes captions and answers questions:

```python
client.vlm.caption("scene.tif")                        # one sentence
client.vlm.vqa("scene.tif", "Is there a river?")       # a short answer
```

The same two operations are available as `pygeovision vlm caption IMAGE` and `pygeovision vlm query IMAGE "QUESTION"`.
Moondream sees an RGB rendering. Its answers are fluent and can be wrong, particularly for counts and small features,
so treat them as notes for a human, not as data. `lisat-7b` (geospatial reasoning segmentation) is listed in the
registry but has no build path in this package.

(sar-adapters)=
## Radar with optical foundation models

Prithvi and DINOv3 were trained on **optical** imagery. Sentinel-1 radar measures backscatter, a different physical
quantity, in decibels, with speckle. Feeding raw radar to an optical model fails, usually without an error.

PyGeoVision offers three documented routes:

| Route | What it does | Honest status |
|---|---|---|
| **Channel mapping** (`sar_to_hls_6ch`) | Rearranges VV and VH into six pseudo-HLS channels (VH → blue, VV → red, inverses and VV/VH ratio in the infrared positions) so an HLS model gets inputs in a sensible form | A preprocessing step. It does not make Prithvi understand radar by itself |
| **Zero-shot adapter** (`SARPrithviAdapter(mode="zero_shot")`) | **Thresholds the mapped VH channel** (below 0.35 on the [0, 1] scale = water) | Not a Prithvi inference. `result["method"] == "vh_threshold"`. Its quality is that of a simple threshold |
| **Fine-tuning** (`mode="fine_tune"`, `SARDINOv3Adapter.prepare_finetuning`) | Trains on a radar dataset such as Sen1Floods11 via TerraTorch | The route that actually adapts a model to radar. Needs TerraTorch and labelled data |

```python
import numpy as np
from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch, validate_sar_ai_input
from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter

# vv, vh: float32 arrays of shape (H, W), normalised to [0, 1] (see below), at least 64 x 64
six = sar_to_hls_6ch(vv, vh, mapping="physics_guided")        # (6, H, W) float32
check = validate_sar_ai_input(six, model="prithvi")            # {'valid': True, 'warnings': [...], 'errors': [], 'shape': (6, H, W)}

result = SARPrithviAdapter(mode="zero_shot").run(vv, vh)
result["prediction"]      # (H, W) uint8: 1 = flooded (VH below the threshold), 0 = not
result["method"]          # 'vh_threshold'
```

**Prepare the radar first.** `sar_to_hls_6ch` expects values **normalised to [0, 1]**, the output of
`normalise_sar_for_ai` after calibration, speckle filtering and dB conversion ([Radar data](../data/sar.md)). It does
not convert decibels for you: feeding it raw dB numbers makes bands such as `1 − VH` meaningless, and
`validate_sar_ai_input` warns when values are outside [0, 1]. (Only the `sentinel_6ch_db` mapping is meant for dB input.)
`mapping="physics_guided"` is the recommended default; `mean_repeat` and `replicate` are baselines for ablation studies.
Image size must be at least 64 × 64 pixels for the Prithvi adapter.

**What to do when…**

| Symptom | Cause | Fix |
|---|---|---|
| `Values [-3.9, 5.5] outside expected [0.0, 1.0]` warning | dB or z-scored values reached the mapping | Normalise to [0, 1] with `normalise_sar_for_ai` first |
| `Spatial size … below minimum 64×64` | Image too small for ViT patches | Use a larger chip |
| Flood map includes hills and runways | A threshold on backscatter cannot tell radar shadow from water | Use [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md), which uses terrain to remove them |
| You want real Prithvi radar results | Zero-shot is a threshold | Fine-tune on labelled radar (Sen1Floods11 or your own) |

## What to do when…

| Situation | What to do |
|---|---|
| You have few labels | Fine-tune with a frozen encoder first (`freeze_backbone=True`), unfreeze later (`unfreeze_backbone_epoch=N`): [Training](../training/index.md) |
| You have no labels | Use CLIP zero-shot, embeddings with nearest-neighbour search, or a [pipeline](../pipelines/index.md) that needs no model |
| The model needs bands you do not have | DOFA accepts any band set; Prithvi needs the six HLS bands |
| Offline machine | Pre-download (`pygeovision models download NAME`, `load_prithvi_local`, `HF_HOME`) before disconnecting |
| Large downloads fill the disk | Weights live in `~/.cache/huggingface`; delete what you no longer use |
