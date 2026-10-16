# Moondream: captions and questions

Moondream2 is a small vision-language model (about 1.8 billion parameters) that looks at one picture and either **writes a
sentence about it** or **answers a short question**. It is the tool for "what is this tile?" and for a quick check such as "is there
standing water?" on a handful of tiles. It is a note-taker, not a surveyor: it gives words, with no area, count or coordinates.

```{figure} ../_static/img/flow-moondream.svg
:alt: Moondream workflow: a raster is turned into an RGB picture, optionally with a question, passed to the model, and text comes back
:width: 100%
:class: pgv-figure

A raster becomes a picture, the picture (and your question) go in, text comes out.
```

## Use it

```python
from pygeovision.models.vlm.moondream import MoondreamGeo

vl = MoondreamGeo(bands=(3, 2, 1))           # downloaded on first use; runs on CPU, faster on a GPU
vl.caption("tile.tif")                       # 'An aerial view of ...'
vl.vqa("tile.tif", "Is there standing water?")
vl.batch_caption("./tiles", top_k=10)        # first 10 images by path; a failing file is reported, not fatal
# -> [{'path': ..., 'caption': 'An aerial view ...'}, {'path': ..., 'caption': None, 'error': 'ValueError: ...'}]
```

From a client: `client.vlm.caption("tile.tif")` and `client.vlm.vqa("tile.tif", "Is there a river?")`. From the shell:
`pygeovision vlm caption tile.tif` and `pygeovision vlm query tile.tif "Is there a river?"` ([the command line](cli.md)).

| `MoondreamGeo(...)` argument | Default | Meaning |
|---|---|---|
| `device` | `"cpu"` | `"cuda"` runs in 16-bit floats and is much faster; on a CPU the model uses 32-bit |
| `revision` | the pinned `MoondreamGeo.REVISION` (`2025-06-21`) | Which release of the model to load; see below |
| `bands` | first three | 1-based file band numbers shown as red, green, blue. **Use `(3, 2, 1)` for Sentinel-2 files stored B2, B3, B4, B8** |

| Method | Returns | Notes |
|---|---|---|
| `caption(image_path)` | a string | A "normal length" description |
| `vqa(image_path, question)` | a string | A short answer |
| `batch_caption(image_dir, top_k=10)` | a list of `{"path", "caption"}` | `.tif .tiff .png .jpg .jpeg`, recursive, sorted by path, the first `top_k`. A file that fails has `caption: None` and an `error` string |

The file is checked **before** the model is loaded, so a missing band or an unreadable file fails in a moment and not after a
multi-gigabyte download.

## What the model sees

An ordinary colour picture built from three bands of your file, with a shared 2nd to 98th percentile stretch and nodata shown
black ([details](index.md#what-the-model-actually-sees)). It does **not** know the pixel size, the date, the sensor or the
coordinates. That is why it cannot say how big anything is.

## Writing questions that work

| Ask | Why | Avoid |
|---|---|---|
| "Is there a river?" | A closed question with a visible answer | "What is the hydrology of this area?" |
| "Is there a road?" | Roads are large and clear | "How many roads are there?" |
| "What colour is the large field in the middle?" | Specific, visible | "What crop is it?" (needs spectral knowledge) |
| "Is the water brown or clear?" | A choice between two things | "How polluted is the water?" |
| "Are there buildings?" (yes/no) | Presence is easier than number | "How many buildings?" (counts are unreliable) |

Run the same question on **two or three tiles you already know the answer for** before trusting it on the rest. Ask it again with a
different wording: if the answer flips, treat it as "unsure".

## The pinned revision

The model ships with its own code, which Hugging Face runs when you load it (`trust_remote_code`). An unpinned download would run
whatever the repository's newest commit contains, so the revision is **pinned** (`MoondreamGeo.REVISION`). Change it deliberately,
after reading what changed in the repository. The class handles both API generations of the model (`caption()` / `query()` in the
pinned release, and the older `encode_image` / `answer_question`).

## Errors, as they appear

```text
$ pygeovision vlm caption tiles/forest.tif --bands 3,9
Error: bands=(3, 9) does not fit tiles/forest.tif, which has 3 band(s); give 1 to 3 band numbers between 1 and the band count

$ pygeovision vlm caption tiles/forest.tif --bands 3,2,1     # no network
Error: Could not load vikhyatk/moondream2@2025-06-21: We couldn't connect to 'https://huggingface.co' to load the files, and
couldn't find them in the cached files. ... — Needs network access to the Hugging Face Hub (or a populated cache).
```

`batch_caption` raises `ModelLoadError` if the model itself cannot load (every file would fail the same way) but records a
per-file error for a bad image and carries on.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ModelLoadError: Could not load vikhyatk/moondream2@...` | No network, or the Hub is blocked | Download once on a connected machine and copy the Hugging Face cache, or set `HF_HOME` to a populated folder |
| `ValueError: bands=(...) does not fit ...` | You asked for a band the file does not have | Use band numbers from 1 to the file's band count |
| Captions say "blue" for vegetation | Band order | `bands=(3, 2, 1)` for B2, B3, B4, B8 files |
| Captions mention things that are not there | Normal for a small model | Treat as a hint; confirm on the image |
| Slow on a CPU | 1.8 B parameters | Use a GPU, or caption a shortlist (rank with [CLIP](clip.md) first) |
| Different caption on each run | Sampling in the model | Do not diff captions; use the answer to a closed question |

## Related

- [Vision-language overview](index.md), [Which vision-language model](foundation.md), [CLIP](clip.md), [the command line](cli.md).
- [`MoondreamGeo` in the Python API](../reference/api/builders-vlm.md).
