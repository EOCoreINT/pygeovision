# CLIP and RemoteCLIP

CLIP learned from hundreds of millions of image-caption pairs to place **pictures and sentences in the same space**. A tile and
the words "flooded farmland" become two lists of numbers; the closer the lists, the better the sentence describes the tile. With no
training you can therefore (1) choose between a few descriptions for a tile, (2) rank thousands of tiles by a text query, and (3)
find tiles that look like one you have.

```{figure} ../_static/img/flow-clip.svg
:alt: CLIP workflow: a tile becomes an RGB picture, an image encoder and a text encoder map picture and sentences into one space, and their similarity gives scores
:width: 100%
:class: pgv-figure

Picture and words meet in one space; the score is how close they are.
```

This page is about **choosing and loading** the models. Using them (zero-shot classification, text search, building an index) is on
[CLIP: zero-shot labels and text search](../vision-language/clip.md); the whole section is [Vision-language](../vision-language/index.md).

## Which CLIP

| Registry name | Weights | Loaded with | Embedding size | Choose it when |
|---|---|---|---:|---|
| `remoteclip-b32` | `chendelong/RemoteCLIP` (ViT-B/32) | `open_clip_torch` | 512 | **Aerial and satellite tiles.** The usual best choice |
| `remoteclip-l14` | `chendelong/RemoteCLIP` (ViT-L/14) | `open_clip_torch` | 768 | The same, larger and slower, usually more accurate |
| `georsclip` | `Zilun/GeoRSCLIP` (RS5M, ViT-B/32) | `open_clip_torch` | 512 | Trained on RS5M, a large remote-sensing image-text set; a second opinion to RemoteCLIP |
| `openclip-b32` (default) | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | `transformers` | 512 | General photographs; fast. Fine for close-range or oblique images |
| `openclip-l14` | `openai/clip-vit-large-patch14` | `transformers` | 768 | General photographs, better and slower |
| `clip-vit-b32` | `openai/clip-vit-base-patch32` | `transformers` | 512 | The original CLIP; a baseline |

The first three were **fine-tuned on remote-sensing pairs** and understand words like "airport", "terraced fields" or "burned area"
far better than a model trained on web photographs. Any other string you pass is treated as a `transformers`-compatible Hugging Face
id.

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
clip = client.vlm.clip("remoteclip-b32")                       # or: CLIPGeo("remoteclip-b32", bands=(3, 2, 1))
clip.zero_shot("tile.tif", ["forest", "water", "urban", "agriculture"])   # {category: probability}, sums to 1
clip.search("airport runway", "./tiles", top_k=5)                         # [{path, score}], best first
```

Measured with the real `ViT-B/32` architecture (random weights, so the *numbers* are not meaningful but the *shapes and format* are):
`zero_shot` returns a dictionary whose values sum to 1 (rounded to four decimals); `embed_text` and `embed_image` return unit-length
vectors of 512 numbers; `search` returns `[{"path": ..., "score": ...}]` sorted best first.

## What CLIP sees

CLIP takes an ordinary three-channel picture. Your GeoTIFF is turned into one by the same function every vision-language tool uses:
three bands (`bands=`, 1-based, default 1 to 3), nodata shown black, and a **shared** 2nd to 98th percentile stretch across the three
bands so colours keep their balance.

```{admonition} Sentinel-2 files need bands=(3, 2, 1)
:class: important
A stack stored B2, B3, B4, B8 has blue first. With the default, the picture the model sees is blue-tinted and every answer is
made on the wrong colours. Pass `bands=(3, 2, 1)`. A band number the file lacks raises `ValueError` before any download.
```

Infrared and radar bands are not seen unless you choose them as the three display bands. CLIP cannot reason about near-infrared on its
own; a false-colour picture is just a strange-looking picture to it.

## Install

| For | Install |
|---|---|
| `openclip-*`, `clip-vit-b32` | `pip install transformers torch` |
| `remoteclip-*`, `georsclip` | `pip install open_clip_torch huggingface_hub` |

The RemoteCLIP and GeoRSCLIP checkpoints are **not** in `transformers` format, although they are hosted on Hugging Face. They are
loaded into the matching `open_clip` architecture, **strictly**: a checkpoint that leaves layers at random initialisation raises
`ModelLoadError` rather than giving you a half-random model that returns confident nonsense. Unexpected extra keys are logged and
ignored.

Weights are downloaded on first use (`~/.cache/huggingface`) and need a network. For a machine without one, download on a connected
machine and copy the cache, or set `HF_HOME` to a populated folder.

## What CLIP is, and is not, good for

| Good for | Not good for |
|---|---|
| Triage: "which of these 10,000 tiles are worth looking at?" | Measuring: no area, count or coordinates come out |
| Rough scene types: airport, farmland, harbour, forest | Fine distinctions: crop types, tree species |
| Finding look-alikes of a known example | Anything small relative to the tile |
| Zero-label starting points for a labelling campaign | Final maps. Confirm with a measured method |

**Limits that bite.** CLIP responds to the words you choose: "farmland" and "agriculture" can rank differently. Probabilities are
*relative to your list* (they always sum to 1), so a tile that is none of your categories still gets a confident-looking answer.
Include a "something else" category, try two or three phrasings, and **validate on tiles you have labelled** before relying on
it. Text search scores (cosine similarities of about 0.15 to 0.35) mean nothing in absolute terms; only their order does.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: open_clip_torch required for RemoteCLIP/GeoRSCLIP` | Package missing | `pip install open_clip_torch` |
| `ModelLoadError: ... would stay randomly initialised` | The cached checkpoint does not match the architecture | Delete the cached file and download again; do not use the model meanwhile |
| Every tile gets nearly the same scores | Vague wording, or the feature is too small to see at this resolution | Use specific phrasing ("a satellite photo of ..."); try a RemoteCLIP model; use a coarser tile |
| Scores look wrong on Sentinel-2 | Blue-tinted picture | `bands=(3, 2, 1)` |
| The same tile wins every search | A bright, high-contrast tile dominates the embedding | Compare scores per query; look at the picture the model saw (`load_rgb_pil(path)`) |

## Related

- [CLIP: zero-shot labels and text search](../vision-language/clip.md): classification, search, and the reusable index.
- [Vision-language overview](../vision-language/index.md), [Foundation models overview](index.md), [DINOv3](dinov3.md) for image features without text.
- [`pygeovision.models.vlm` in the Python API](../reference/api/builders-vlm.md).
