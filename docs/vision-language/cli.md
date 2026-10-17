# Command line

Three commands cover the vision-language tools from the shell. They need no Python and are meant for quick looks: describe one tile,
ask it a question, or search a folder.

```text
pygeovision vlm caption IMAGE_PATH [--bands 3,2,1]
pygeovision vlm query   IMAGE_PATH "QUESTION" [--bands 3,2,1]
pygeovision vlm search  "QUERY" IMAGE_DIR [--top-k 5] [--model openclip-b32] [--bands 3,2,1]
```

| Command | Uses | Prints |
|---|---|---|
| `vlm caption` | [Moondream2](captions.md) | One sentence about the image |
| `vlm query` | Moondream2 | A short answer to your question |
| `vlm search` | A [CLIP](clip.md) model | `Top N results for 'QUERY':` and one `score  path` line per image, best first |

## Options

| Option | Commands | Default | Meaning |
|---|---|---|---|
| `--bands` | all three | the first three bands | The file's band numbers (1-based) shown to the model as red, green, blue, e.g. `3,2,1` for a Sentinel-2 file stored B2, B3, B4, B8 |
| `--top-k` | `search` | 5 | How many results to print |
| `--model` | `search` | `openclip-b32` | `openclip-b32`, `openclip-l14`, `clip-vit-b32`, `remoteclip-b32`, `remoteclip-l14` or `georsclip`. For aerial and satellite tiles try `remoteclip-b32` ([which CLIP](../foundation-models/clip.md)) |

The full option list, with generated help text, is in [the command reference](../reference/cli/vlm.md).

## Examples

```bash
pygeovision vlm caption scene.tif --bands 3,2,1
pygeovision vlm query scene.tif "Is there standing water?" --bands 3,2,1
pygeovision vlm search "airport runway" ./tiles --top-k 10 --model remoteclip-b32 --bands 3,2,1
```

Models download from the Hugging Face Hub the first time they are used (a network is needed once), and a search embeds **every image
on every call**, so for a collection you search repeatedly use [`GeoImageRetrieval`](clip.md#geoimageretrieval-embed-once-search-many-times)
in Python.

## Errors, verified

Failures are one-line errors, not tracebacks. Real output, with no network:

```text
$ pygeovision vlm caption nofile.tif
Usage: pygeovision vlm caption [OPTIONS] IMAGE_PATH
Try 'pygeovision vlm caption --help' for help.

Error: Invalid value for 'IMAGE_PATH': File 'nofile.tif' does not exist.                       (exit code 2)

$ pygeovision vlm caption tiles/forest.tif --bands 3,9
  Generating caption for: tiles/forest.tif
Error: bands=(3, 9) does not fit tiles/forest.tif, which has 3 band(s); give 1 to 3 band numbers between 1 and the band count
                                                                                               (exit code 1)

$ pygeovision vlm caption tiles/forest.tif --bands 3,2,1
  Generating caption for: tiles/forest.tif
Error: Could not load vikhyatk/moondream2@2025-06-21: We couldn't connect to 'https://huggingface.co' to load the files, and
couldn't find them in the cached files. ... — Needs network access to the Hugging Face Hub (or a populated cache).   (exit code 1)

$ pygeovision vlm search "airport" nodir
Error: Invalid value for 'IMAGE_DIR': Directory 'nodir' does not exist.                      (exit code 2)
```

Two exit codes to know: **2** is "you called the command wrongly" (a missing path, a missing argument) and **1** is "the command ran
and failed" (a bad band number, no network). Band checks happen **before** any model is downloaded.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `Error: Could not load ...` | No network, or the Hub is blocked | Run once where there is a network, or copy a populated Hugging Face cache and set `HF_HOME` |
| `Error: bands=(...) does not fit ...` | A band number beyond the file's count | Count bands with `pygeovision validate` or `gdalinfo` |
| `Error: Can't load image processor for 'laion/...'` | The CLIP weights are not cached and there is no network | As above |
| The picture seems blue-tinted | Sentinel-2 order | `--bands 3,2,1` |
| `search` is slow | It embeds every image every time | Build a `GeoImageRetrieval` index once in Python |

## Related

- [Vision-language overview](index.md), [CLIP](clip.md), [Moondream](captions.md), [Which model](foundation.md).
- [The generated command reference for `vlm`](../reference/cli/vlm.md).
