# Predicting many scenes

You have a folder of scenes (a tile grid, a year of acquisitions, a client's archive) and one trained model. This page covers
running them all: the command, the same thing in Python, what happens when a scene fails, how to restart, and how to keep memory
under control.

```{figure} ../_static/img/flow-batch.svg
:alt: Batch inference: find the files, skip those already done, predict each in tiles, collect successes and failures, report
:width: 100%
:class: pgv-figure

For each file the engine decides, predicts, and records; one bad scene never stops the others unless you ask it to.
```

## From the shell

```bash
pygeovision infer batch scenes_in/ predictions_out/ -m unet-r50 --checkpoint best.pth -c 2 --in-channels 4 --no-normalise
```

```text
  Batch inference: scenes_in → predictions_out
  ✓ 2/2 succeeded
  Time: 1.9s | 1.04 fps
```

| Option | Default | Meaning |
|---|---|---|
| `-m`, `--model` | `unet_resnet50` | A registry name ([Models](../models/index.md)); must match the checkpoint |
| `--checkpoint` | required | Your trained `.pth` ([Training](../training/index.md)) |
| `-c`, `--classes` | 2 | Number of classes the model was trained with |
| `--in-channels` | the first file's | Number of bands the model was trained with |
| `--no-normalise` | off | Use pixel values as stored; give it when training chips were not stretched ([Matching the input to training](index.md#matching-the-input-to-training)) |
| `-w`, `--workers` | 2 | Scenes predicted at the same time (threads) |
| `--pattern` | `*.tif` | Which files in the input folder to take (searched in subfolders too) |
| `--overwrite` | off | Predict again even where the output file exists |

The CLI uses chip size 512 and overlap 64. For other values, or to hold the model in memory yourself, use Python (next section).
The complete option list is in [the command reference](../reference/cli/infer.md).

- Outputs go to `predictions_out/` with the **same file names**; subfolders are mirrored (`scenes_in/north/c.tif` becomes
  `predictions_out/north/c.tif`).
- **Files whose output already exists are skipped**, so an interrupted batch can be restarted. The summary then says
  `2 skipped (output already exists; --overwrite to redo)`.
- A file that fails is listed with its error and counted as failed; the exit code is 1. When nothing failed the exit code is 0.

## From Python

```python
from pygeovision.inference import BatchInferenceEngine

engine = BatchInferenceEngine(model, device="cuda", n_workers=2, chip_size=256, overlap=32,
                              num_classes=2, normalise=False)
summary = engine.run_directory("scenes_in/", "predictions_out/", pattern="*.tif", skip_existing=True)
print(summary["n_success"], summary["n_failed"], summary["failed"])
```

### The engine's settings

| Argument | Default | Meaning |
|---|---|---|
| `model` | | A PyTorch model in the state you want (`.eval()` is applied for you) |
| `device` | auto | `"cuda"`, `"cpu"`, `"cuda:1"` |
| `n_workers` | 1 | Scenes in flight at once. Each worker holds a whole scene in RAM, so more workers means more memory |
| `chip_size`, `overlap` | 512, 64 | Tile size and overlap in pixels ([Tiles, overlap and seams](tiles.md)) |
| `num_classes` | 2 | Classes in the model's output |
| `blend_mode` | `"gaussian"` | How overlapping tiles are merged: `gaussian`, `linear` or `constant` |
| `half_precision` | `True` | Use 16-bit floats on a GPU: roughly half the model memory, slightly different numbers. Ignored on a CPU |
| `on_error` | `"skip"` | `"skip"` or `"warn"`: record the failure and carry on. `"abort"`: stop at the first failure |
| `normalise` | `True` | Stretch each band to its own 2nd to 98th percentile. **Set `False` if the model was trained on values used as stored** |

### What `run_directory` takes and returns

`run_directory(input_dir, output_dir, pattern="*.tif", band_selection=None, skip_existing=True, progress_callback=None)`

| Argument | Meaning |
|---|---|
| `pattern` | A glob, searched recursively. `"*.tif"` also finds `north/c.tif` |
| `band_selection` | A list of band numbers, to use some of a scene's bands (for a model trained on four of a file's twelve) |
| `skip_existing` | Leave scenes alone whose output file exists |
| `progress_callback` | A function `f(done, total, path)` called after each scene |

| Key in the result | Meaning |
|---|---|
| `succeeded` | List of input paths that were predicted |
| `failed` | List of `{path, error}`; the error text is the model's own message |
| `skipped` | How many files were left alone because their output exists |
| `n_success`, `n_failed` | Counts |
| `total_time_s`, `throughput_fps` | Wall-clock seconds and scenes per second |

If no file matches the pattern, `run_directory` returns `{"success": False, "error": "No files matching ..."}` and writes nothing,
so check for that key before reading `n_success`.

### A real run, with one bad scene

Four scenes: three with the four bands the model expects, and one with three. The model is the toy network from the tutorials, so
this runs in a fraction of a second on a CPU:

```python
summary = engine.run_directory("scenes_in", "pred_out", progress_callback=lambda i, n, f: print(f"{i}/{n} {f}"))
```

```text
1/4 scenes_in/a.tif
2/4 scenes_in/b.tif
3/4 scenes_in/bad_three_bands.tif
4/4 scenes_in/north/c.tif

succeeded = ['scenes_in/a.tif', 'scenes_in/b.tif', 'scenes_in/north/c.tif']
failed    = [{'path': 'scenes_in/bad_three_bands.tif',
              'error': 'Model does not accept 3 input band(s) at chip_size=64 (selected from 3 real band(s) in
                        scenes_in/bad_three_bands.tif): Given groups=1, weight of size [8, 4, 3, 3], expected input[1, 3, 64, 64]
                        to have 4 channels, but got 3 channels instead. Pass band_selection= explicitly if this file has more
                        bands than the model expects; ...'}]
skipped   = 0
n_success = 3
n_failed  = 1
```

The three good scenes are written (`pred_out/a.tif`, `pred_out/b.tif`, `pred_out/north/c.tif`) and the bad one is reported,
not hidden. The order of the progress lines is the order scenes *finish*, which with several workers is not the alphabetical order.

Run it a second time and the successes are skipped, while the **failure is tried again** (it has no output file):

```text
second run: {'n_success': 0, 'n_failed': 1, 'skipped': 3}
```

This is what makes the loop safe: fix the file or the settings, run the same command, and only the missing outputs are computed.
Skipping looks at the **file name only**, not at which model made the file. If you change the model and want new maps, give the
run a new output folder or pass `skip_existing=False` (`--overwrite` on the command line).

## Several GPUs

```python
done = engine.run_multi_gpu(["a.tif", "b.tif", "c.tif", "d.tif"], "predictions_out/", gpu_ids=[0, 1])
print(done["succeeded"], done["failed"])
```

Files are dealt out one by one to the GPUs, one worker thread per GPU, and every output is named like its input. With one GPU or
none, the same files are predicted one after another on the engine's own `device`, so a script that runs on a laptop also runs on a
multi-GPU server. The result is `{"succeeded": [...], "failed": [...]}`: lists of input paths, without the timing keys of
`run_directory`.

## When memory is the limit

Two things use memory: the **model** (it grows with the tile size and the batch) and the **scene** (the engine reads the whole scene
into RAM and keeps full-size accumulators: about `bands × 4 + classes × 8 + 8` bytes per pixel, so a 10,000 × 10,000 pixel scene with
4 bands and 3 classes is roughly 4.8 GB).

| Symptom | What to change |
|---|---|
| GPU out of memory | Lower `chip_size`, or keep `half_precision=True`; use `StreamingInference` (below) to run one tile per model call |
| RAM out of memory with a large scene | Lower `n_workers` to 1; or cut the scene into pieces first (`pygeovision preprocess clip`), predict the pieces, and mosaic |
| Too slow on a GPU | Raise `n_workers` to 2 or 3 so the GPU stays busy while another scene is read from disk |
| Too slow on a CPU | Lower `overlap`; use a smaller model |

`StreamingInference(model, chip_size=1024, overlap=128, num_classes=2)` is the lower-GPU-memory variant: it predicts one chip per model
call. Its `infer(path, output_path, normalise=True)` gives the same kind of output as `TiledInference.infer`, but note that the
**scene is still read whole**; only the model's memory use shrinks. Its `stream_chips(path)` is a generator that yields one window
at a time from disk if you want to write your own loop. It scales each chip to its own minimum and maximum, which is *not* the
per-scene stretch that `infer` uses, so do not mix the two.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `failed` lists every scene with the same message | Wrong model, checkpoint, `num_classes` or band count | Test on one scene first with [`infer predict`](index.md) |
| The same scene fails every run | Not four bands, unreadable, or a corrupt download | Read the error text; use `band_selection=` if it has more bands than the model needs |
| `n_success` is 0 and nothing happened | Every output already exists (look at `skipped`) | `--overwrite` or a new output folder |
| `{"success": False, "error": "No files matching ..."}` | Wrong folder or pattern | Check the path; note the pattern is case-sensitive on Linux (`*.TIF` is not `*.tif`) |
| Maps look washed out or all one class | `normalise` does not match training | [Matching the input to training](index.md#matching-the-input-to-training) |
| Seams between tiles | Overlap too small | [Tiles, overlap and seams](tiles.md) |
| Two runs wrote into one folder with different models | Skipping goes by file name | Use one output folder per model |

## Related

- [Inference overview](index.md), [Tiles, overlap and seams](tiles.md), [Combining several models](ensembles.md).
- [The `infer` commands](../reference/cli/infer.md)
- A year of scenes? Combine with [monitoring](../monitoring/index.md) to watch the model for drift.
