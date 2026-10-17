# CLIP and text search

## Models

| Name | Weights | Backend | Best for |
|---|---|---|---|
| `openclip-b32` (default) | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | transformers | General web-trained CLIP; fast |
| `openclip-l14` | `openai/clip-vit-large-patch14` | transformers | Larger, better, slower |
| `clip-vit-b32` | `openai/clip-vit-base-patch32` | transformers | The original CLIP |
| `remoteclip-b32`, `remoteclip-l14` | `chendelong/RemoteCLIP` | open_clip | Fine-tuned on remote-sensing image-text pairs: usually the best for aerial/satellite tiles |
| `georsclip` | `Zilun/GeoRSCLIP` | open_clip | Fine-tuned on RS5M, a large remote-sensing set |

Any other string is treated as a transformers-compatible Hugging Face id. The RemoteCLIP/GeoRSCLIP checkpoints are loaded
**strictly**: a checkpoint that leaves layers randomly initialised raises `ModelLoadError` instead of giving a half-random
model. Weights are downloaded on first use (network needed).

## Zero-shot classification

```python
from pygeovision.models.vlm.clip import CLIPGeo

clip = CLIPGeo(model="remoteclip-b32", bands=(3, 2, 1))
clip.zero_shot("tile.tif", ["flooded farmland", "dry farmland", "forest", "urban area"])
# -> {'flooded farmland': 0.71, 'dry farmland': 0.18, 'forest': 0.07, 'urban area': 0.04}
```

*The numbers above only illustrate the format; CLIP weights could not be downloaded where the documentation was written, so
no real CLIP output is shown on this page.*

How to read it:

* The scores are a **softmax over the categories you passed**, so they always sum to 1. There is no "none of these": if you
  leave out the right answer, the least-wrong category wins. Include an "other" or "background" description.
* **Wording matters.** The text is used as given. "a satellite photo of flooded farmland" usually works better than
  "flood". Try two or three phrasings and check that the winner is stable.
* Differences between categories matter more than the winner's score. 0.45 vs 0.40 is a tie.

## Text search over a folder

```python
clip.search("coastal flooding", "./tiles", top_k=5)
# -> [{'path': './tiles/a.tif', 'score': 0.2874}, ...]
```

Scores are cosine similarities. For CLIP they typically fall around **0.15–0.35**, and only their **order** is meaningful: do
not set a threshold such as "above 0.3 is a match" without checking it on your own tiles. `search` embeds every image on every
call; unreadable files are skipped with a warning.

## `GeoImageRetrieval`: embed once, search many times

```python
from pygeovision.advanced.vlm.retrieval import GeoImageRetrieval

r = GeoImageRetrieval(model="remoteclip-b32", bands=(3, 2, 1))
r.build_index("./tiles", save_path="tiles_index.json")      # slow, once: returns the number of images indexed
r.search_by_text("flooded farmland", top_k=10)
r.search_by_image("./tiles/known_flood.tif", top_k=10)      # look-alikes; the query itself is left out
```

* `build_index` starts a fresh index. Files that cannot be read are skipped and listed with the reason in `r.skipped`
  (`{path: "OSError: ..."}`); look at it after every build.
* The index is a plain JSON file `{"model": "...", "embeddings": {path: [floats]}}`. Paths are stored as you passed the
  folder (relative stays relative), so move the index and the tiles together or use absolute paths.
* **An index belongs to one model.** `load_index` raises `ValueError` if the index was built with another model than the
  retriever's (vectors from different models cannot be compared). Older files without a model name still load; if their
  vectors have a different length the search raises `ValueError: ... built with a different model`.
* Searching an empty index raises `RuntimeError: the index is empty: call build_index() or load_index() first`.
