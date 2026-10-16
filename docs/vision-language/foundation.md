# Which vision-language model

Several models live under the vision-language heading, and they answer different questions. This page puts them side by side so you
can choose, and says plainly which ones this package can run.

```{figure} ../_static/img/tree-vlm-chooser.svg
:alt: Choosing a vision-language tool: ranking many tiles by a description leads to CLIP, a sentence or an answer about one tile leads to Moondream, text-guided outlines are not available
:width: 100%
:class: pgv-figure

Start from what you need back.
```

## At a glance

| Model | Registry name | Parameters | Gives you | Runs here? |
|---|---|---:|---|---|
| CLIP (general) | `openclip-b32`, `openclip-l14`, `clip-vit-b32` | 151 to 428 M | Scores against your descriptions; embeddings for search | Yes (`transformers`) |
| RemoteCLIP, GeoRSCLIP | `remoteclip-b32`, `remoteclip-l14`, `georsclip` | 151 to 428 M | The same, better on aerial and satellite imagery | Yes (`open_clip_torch`) |
| Moondream2 | `moondream2` | about 1.8 B | A caption, or a short answer to a question | Yes (`transformers`; pinned revision) |
| LISAt-7B | `lisat-7b` | 7 B | A reasoning answer **and** a segmentation mask | **No.** Listed only: needs the authors' code, `flash-attn` and a separate checkpoint |
| ChatEarthNet | `chatearthnet` | | Remote-sensing chat | **No.** Listed only; no verified checkpoint |

Everything in the "Yes" rows takes a **three-band picture** built from your raster ([what the model sees](index.md#what-the-model-actually-sees)),
downloads its weights on first use, and needs a network the first time.

## Which one for which job

| You want to | Use | Why |
|---|---|---|
| Rank thousands of tiles by a description | `CLIPGeo.search`, or `GeoImageRetrieval` for repeated searches | A dot product per tile once embedded: seconds for thousands of tiles |
| Choose between a few named categories for a tile | `CLIPGeo.zero_shot` | Probabilities over your own list |
| Find tiles that look like one you have | `GeoImageRetrieval.search_by_image` | Image-to-image on the same embeddings |
| Get a sentence describing a tile | `MoondreamGeo.caption` | A free-text description |
| Ask a yes/no question about one tile | `MoondreamGeo.vqa` | "Is there a river?" |
| An outline of the thing you ask for in words | Not available. Use [SAM](../foundation-models/sam.md) for outlines and CLIP to pick the right one | `lisat-7b` would do both but cannot be built here |
| A **number**: area, count, extent | None of these. Use a [pipeline](../pipelines/index.md) | They give words, not measurements |

## Moondream in one picture

```python
client.vlm.caption("scene.tif")                        # one sentence
client.vlm.vqa("scene.tif", "Is there a river?")       # a short answer
```

The same two operations are `pygeovision vlm caption IMAGE` and `pygeovision vlm query IMAGE "QUESTION"`
([the command line](cli.md)). Moondream (about 1.8 billion parameters) sees an RGB rendering. Its answers are fluent and can be
wrong, particularly for counts, small features and anything that needs scale (it has no idea how large a pixel is), so treat them as
**notes for a human**, not as data. Details, the pinned revision and batch captioning are on [Moondream: captions and
questions](captions.md).

## Why LISAt-7B is listed but not built

LISAt-7B (Quenum et al., 2025) is a real published model that reads a request such as "segment the flooded roads" and returns both
an explanation and a mask. Its checkpoint is a custom LISA-style multimodal language model with a SAM decoder, not a standard
`transformers` class, so loading it needs the authors' `lisat_code` repository, `flash-attn` and a separate RemoteCLIP checkpoint. No
verified build path exists in this package, so `get_model("lisat-7b")` raises `NotImplementedError` and says so, instead of
returning something else ([the list of such models](../models/unavailable.md)). The combination it offers can be approximated
by hand: SAM outlines objects ([SAM](../foundation-models/sam.md)), CLIP ranks the outlines against your words.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError` for `lisat-7b` or `chatearthnet` | Listed-only entries | Use the alternatives above |
| A caption names things that are not in the tile | Normal for a small model | Verify; ask closed questions instead |
| You need an answer for thousands of tiles | A VLM call per tile is slow | Use CLIP to shortlist, then caption the shortlist |

## Related

- [Vision-language overview](index.md), [CLIP](clip.md), [Captions and questions](captions.md), [The command line](cli.md).
- [Foundation-model registry](../foundation-models/models.md) and [CLIP and RemoteCLIP](../foundation-models/clip.md).
