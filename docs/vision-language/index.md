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

```{literalinclude} ../_examples/advanced/vlm_inputs_demo.py
:language: python
:caption: vlm_inputs_demo.py
:linenos:
```

{download}`Download vlm_inputs_demo.py <../_examples/advanced/vlm_inputs_demo.py>`. It runs offline. Output:

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
