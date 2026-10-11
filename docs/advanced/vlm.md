# Vision-language models: describe, ask, search

**Package:** `pygeovision.models.vlm` (also importable from `pygeovision.advanced.vlm`) · **Classes:** `CLIPGeo`,
`MoondreamGeo`, `GeoImageRetrieval` · **CLI:** `pygeovision vlm caption | query | search` ·
**Needs:** `transformers`, `torch`; `open_clip_torch` for RemoteCLIP / GeoRSCLIP

## Why

Two jobs that used to need a trained model per question can be done with words:

* **Find** images: "flooded farmland", "ship at a port", "burn scar". CLIP puts pictures and sentences in the same space, so
  a text query can rank thousands of tiles in seconds, and an example image can find look-alikes.
* **Describe and ask**: a small vision-language model (Moondream2) writes a caption or answers a question about a picture.

Both are strong for **triage**: deciding what to look at. They are weak as **measurements**. The models see a small picture
rendered from three bands, answer in fluent text that can be wrong (counts, small objects, anything not in their training
data), and give no area, coordinates or error bar. Use them to prioritise, then confirm with a measured method
([a pipeline](../pipelines/index.md) or your own check).

## The four tools

| Tool | Question it answers | Output |
|---|---|---|
| `CLIPGeo.zero_shot(image, categories)` | Which of these descriptions fits this image best? | `{category: score}`, scores sum to 1 |
| `CLIPGeo.search(query, image_dir, top_k)` | Which images in this folder match this text? | `[{"path", "score"}]`, best first |
| `GeoImageRetrieval` | The same, for a collection you query many times (embeds once), and image-to-image search | `[{"path", "score"}]` |
| `MoondreamGeo.caption / vqa / batch_caption` | What is in this image? / Answer this question | `str`; batch: `[{"path", "caption"}]` |

## What the model actually sees

Both model families take an ordinary colour picture, not a GeoTIFF. PyGeoVision builds it with one shared function
(`pygeovision.models.vlm._image.load_rgb_pil`):

1. Take three bands, in the order you give them (`bands=`, 1-based file band numbers; **default: bands 1, 2, 3**). One band
   is repeated as grey, two bands become R, G, R.
2. Ignore nodata and NaN pixels when scaling; show them black.
3. Scale all three bands between **one shared** 2nd and 98th percentile (the shared pair keeps the colour balance;
   `stretch="per_band"` would scale each band alone and shift colours).

```{literalinclude} _examples/vlm_inputs_demo.py
:language: python
:caption: vlm_inputs_demo.py
:linenos:
```

{download}`Download vlm_inputs_demo.py <_examples/vlm_inputs_demo.py>`. It runs offline. Output:

```text
default: bands 1,2,3     shape (64, 64, 3) dtype uint8  mean R,G,B in valid area = [44, 143, 211]  nodata border max = 0
per-band stretch         shape (64, 64, 3) dtype uint8  mean R,G,B in valid area = [128, 126, 128]  nodata border max = 0
bands=(3, 2, 1)          shape (64, 64, 3) dtype uint8  mean R,G,B in valid area = [211, 143, 44]  nodata border max = 0
(the file is written B2, B3, B4, B8: with the default the model's 'red' is band 1 = B2, so the picture is blue-tinted;
 bands=(3, 2, 1) shows B4 as red and gives natural colour)

supported image files : ['scene.tif']
index file keys       : ['model', 'embeddings']
index model / entries : openclip-b32 2
```

```{admonition} Set bands= for Sentinel-2 files
:class: important
A Sentinel-2 stack written B2, B3, B4, B8 has blue first. With the default, the model's "red" channel is blue light: a
forest looks blue-purple and every answer is made on a wrongly coloured picture. Pass `bands=(3, 2, 1)` (Python) or
`--bands 3,2,1` (CLI). For stacks in the order R, G, B, NIR the default is right. A band number the file does not have raises
a `ValueError` naming the file and band count, **before** any model is downloaded.
```

Supported file types for folders: `.tif .tiff .png .jpg .jpeg`, any capitalisation, searched recursively.

## CLIP: zero-shot labels and text search

### Models

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

### Zero-shot classification

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

### Text search over a folder

```python
clip.search("coastal flooding", "./tiles", top_k=5)
# -> [{'path': './tiles/a.tif', 'score': 0.2874}, ...]
```

Scores are cosine similarities. For CLIP they typically fall around **0.15–0.35**, and only their **order** is meaningful: do
not set a threshold such as "above 0.3 is a match" without checking it on your own tiles. `search` embeds every image on every
call; unreadable files are skipped with a warning.

### `GeoImageRetrieval`: embed once, search many times

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

## Moondream: captions and questions

```python
from pygeovision.models.vlm.moondream import MoondreamGeo

vl = MoondreamGeo(bands=(3, 2, 1))          # ~1.8 B parameters; downloaded on first use; runs on CPU, faster on GPU
vl.caption("tile.tif")                       # 'An aerial view of ...'
vl.vqa("tile.tif", "Is there standing water?")
vl.batch_caption("./tiles", top_k=10)        # first 10 images by path; a failing file is reported, not fatal
# -> [{'path': ..., 'caption': 'An aerial view ...'}, {'path': ..., 'caption': None, 'error': 'ValueError: ...'}]
```

* The revision of the model is **pinned** (`MoondreamGeo.REVISION`), because the model ships its own code and an unpinned
  download would run whatever the repository's newest commit contains. Change it deliberately, after reading the diff.
* Answers are text. Ask **closed** questions ("Is there a river?"), and expect mistakes on counts ("How many buildings?"),
  small objects and anything that needs scale: the model has no idea how large a pixel is.
* `batch_caption` raises `ModelLoadError` if the model itself cannot load (every file would fail the same way), but records
  a per-file error for a bad image.

## The CLI

```text
pygeovision vlm caption IMAGE_PATH [--bands 3,2,1]
pygeovision vlm query   IMAGE_PATH "QUESTION" [--bands 3,2,1]
pygeovision vlm search  "QUERY" IMAGE_DIR [--top-k 5] [--model openclip-b32] [--bands 3,2,1]
```

`search` prints `Top N results for 'QUERY':` then one `score  path` line per image. Failures are one-line errors, not tracebacks.
Verified here:

```text
$ pygeovision vlm caption scene.tif
  Generating caption for: scene.tif
Error: Could not load vikhyatk/moondream2@2025-06-21: Can't load the configuration of 'vikhyatk/moondream2'. ... — Needs network access to the Hugging Face Hub (or a populated cache).

$ pygeovision vlm caption scene.tif --bands 3,9
  Generating caption for: scene.tif
Error: bands=(3, 9) does not fit scene.tif, which has 4 band(s); give 1 to 3 band numbers between 1 and the band count
```

The full option list is in the [CLI reference](../reference/cli/vlm.md).

## What to do when…

| Situation | What to do |
|---|---|
| `Error: Could not load vikhyatk/moondream2@…` / `ModelLoadError` | No network or the Hub is blocked. Download once on a connected machine (Hugging Face cache) and copy the cache, or set `HF_HOME` to a populated folder |
| Pictures look blue/purple to the model | Wrong band order: pass `bands=(3, 2, 1)` for B2,B3,B4,B8 files |
| `ValueError: bands=(…) does not fit …` | The file has fewer bands than you asked for |
| `ModelLoadError: … randomly initialised` (RemoteCLIP) | The downloaded checkpoint does not match the architecture. Delete the cached file and download again; do not use the model |
| `ImportError … open_clip` | `pip install open_clip_torch` for RemoteCLIP / GeoRSCLIP |
| All search scores nearly equal | The query is vague or not visible at this scale; try the specific phrasing "a satellite photo of …", or a RemoteCLIP model |
| The same tile wins every query | The tile is bright/high-contrast and dominates CLIP's embedding; compare scores per query, and look at the picture the model saw (`load_rgb_pil(path)`) |
| A tile is blank or black | It is all nodata, or the stretch had nothing to scale. Open it in QGIS; a file entirely nodata is shown black on purpose |
| Captions mention things that are not there | Normal for small VLMs. Treat captions as hints; verify |
| `ValueError: index … was built with model 'openclip-b32', but this retriever uses …` | Create `GeoImageRetrieval(model=<the model in the message>)`, or rebuild the index |
| Searching 100 000 tiles is slow | Build the index once on a GPU with `GeoImageRetrieval`; the index search itself is a dot product per image and is fast |
| You need a number (area, count) | Not from these tools. Use a segmentation/detection [pipeline](../pipelines/index.md) |

## Verified here, and what is not

* **Verified (tests and the script above):** the image preparation (band choice, joint vs per-band stretch, nodata/NaN
  handling, 1- and 2-band files), recursive file discovery, index writing/loading/mismatch errors, empty index, skipped-file
  reporting, the CLI error messages, the pinned Moondream revision and both of its API generations (with stand-in models).
* **Not verified:** any real output of CLIP, RemoteCLIP, GeoRSCLIP or Moondream, and their accuracy on your imagery. Their
  weights are downloaded from the Hugging Face Hub, which is not reachable from the test environment.
