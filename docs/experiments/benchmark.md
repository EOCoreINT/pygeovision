# Benchmarking

**Package:** `pygeovision.benchmark` · **Classes:** `ModelEvaluator`, `BenchmarkResult`, `Leaderboard` ·
**CLI:** `pygeovision benchmark datasets | models | tasks` (catalogue helpers, see [below](leaderboard.md#the-benchmark-cli))

## Why

"Model A looks better than model B" is a claim that only means something if both were scored **the same way, on the same
data, with the same metric**, and if a failure is never silently counted as a score. `ModelEvaluator` is the one place in
PyGeoVision that does this: it runs any PyTorch model over any data loader, computes the metrics through the same code the
[trainer](../training/index.md) uses, times the forward pass, and hands back a `BenchmarkResult` you can print, save,
compare and keep on a `Leaderboard`.

What it deliberately **does not** do: it does not create train/validation/test splits, download datasets, or decide which
data is a fair test. The loader you pass is the test set. That choice (a scene the model never saw, ideally from another
place and date) matters more than anything on this page; see [Accuracy and validation](../obia/accuracy.md#the-training-data-trap)
for how easily a flattering number appears when it is neglected.

## How to use

```{literalinclude} ../_examples/benchmark/benchmark_demo.py
:language: python
:caption: benchmark_demo.py
:linenos:
```

{download}`Download benchmark_demo.py <../_examples/benchmark/benchmark_demo.py>`. It downloads a real Sentinel-2 scene and ESA WorldCover with
PyGeoVision's own tools ([Real imagery](../obia/real-data.md)), trains two small networks on the west of the scene, scores
them on the east, which they never saw, and adds an untrained network, a model that outputs at quarter resolution, and one
that crashes. It needs a network; the training itself takes seconds on a CPU. WorldCover is a model's output, not ground truth,
so the scores say how well a network reproduces WorldCover on new ground.

### The pieces

| Object | What it is |
|---|---|
| `ModelEvaluator(task, num_classes)` | The evaluator. `task` is one of `segmentation`, `change_detection`, `classification`, `detection` |
| `.evaluate(model, loader, dataset_name, model_name, split, device)` | Score **one** model; returns a `BenchmarkResult` (never raises for a model/data problem; see [failures](#failures-are-recorded-not-scored)) |
| `.compare({name: model}, loader, dataset_name, ...)` | Score several models on the same loader; returns results best first, failed last |
| `.print_leaderboard(results)` | Table for the task, with `FAILED` rows listed at the bottom |
| `.save_results(results, path)` | JSON list of `result.to_dict()` |
| `.confusion_matrix_plot(model, loader, class_names, save_path)` | Row-normalised confusion matrix as a matplotlib figure (segmentation-type tasks) |
| `BenchmarkResult` | One model on one dataset: metrics, per-class IoU, latency, sample count, `error` |
| `Leaderboard(path)` | A JSON file of results that survives between runs, with `print` and `export_csv` |

### What the loader must yield

| Task | Each batch | Model output |
|---|---|---|
| `segmentation`, `change_detection` | `(images, masks)` with `images` `(B, C, H, W)` and `masks` `(B, H, W)` of integer class ids, or a dict with `image`/`images` and `mask`/`label` | Logits `(B, classes, h, w)`: a tensor, an object with `.logits`, a dict with `"out"`/`"logits"`/`"pred"`, or a tuple whose first item is the logits |
| `classification` | `(images, labels)` with `labels` `(B,)` | Logits `(B, classes)` |
| `detection` | `(list_of_images, list_of_target_dicts)`, targets with `boxes` and `labels` | A list of dicts with `boxes`, `scores`, `labels` (torchvision convention) |

Things the evaluator does for you:

- **Coarse outputs are enlarged.** If the model returns a smaller map than the mask (SegFormer-style quarter-resolution
  decoders), the logits are bilinearly resized to the mask size before the argmax. In the example, `quarter-res` is scored
  rather than discarded, and typically loses some mIoU to the blur, which is the honest price of the coarse output.
- **Label 255 is ignored** (`ignore_index=255`), so padded or unlabelled pixels do not count. The evaluator does not
  expose another value; remap your labels if you use a different convention.
- **Labels are expected to start at 0.** A mask with class ids `1..N` and `num_classes=N` fails with a class-index error
  (the example remaps WorldCover codes to ids `0..4` and sets everything else to 255).
- **GPU timing is synchronised**, so the latency is the forward pass and not just the kernel launch.

## Expected output

The example prints a leaderboard in this layout (the numbers depend on your scene, so none are quoted here):

```text
══════════════════════════════════════════════════════════════════════════════════════════
  Leaderboard: east-of-scene | Task: segmentation
══════════════════════════════════════════════════════════════════════════════════════════
  #   Model                              mIoU       F1      Acc   ms/img
  ──────────────────────────────────────────────────────────────────────
  1   net-16                              ...
  2   quarter-res                         ...
  3   net-4                               ...
  4   untrained                           ...
  FAILED  broken                         RuntimeError: CUDA out of memory (simulated)
══════════════════════════════════════════════════════════════════════════════════════════
```

followed by the per-class IoU of the winner (`{'iou_class_0': ..., ...}`), a saved leaderboard and a `results.json`.

Read it like this:

- **Look at `per_class_iou`, not only the mean.** A model can win on mIoU while a small class (say `built_up`) has IoU 0.0;
  the average hides it.
- **Accuracy and mIoU disagree on imbalanced classes.** Accuracy rewards getting the big classes right; mIoU punishes every
  missed class. Pick the metric that matches the cost of your errors.
- **`untrained` is your floor.** If a trained model is not clearly above it, something is wrong with the training, not the
  test.
- **`ms/img` for a toy network on a CPU** says nothing about your real model. See [latency](#latency-is-indicative-only).
- **`broken` is listed as `FAILED`, not as 0.** The result's `error` holds the reason; its metric fields are meaningless.

### `BenchmarkResult` fields

| Field | Meaning |
|---|---|
| `model_name`, `dataset_name`, `task`, `split` | Identification (`split` is a label you set; the evaluator does not split) |
| `mean_iou`, `mean_f1` | Averages over classes that **occur** in targets or predictions (a class absent from both is ignored, not counted as 0) |
| `accuracy` | Correct pixels / counted pixels (for classification: correct samples / samples) |
| `mAP50`, `mAP50_95` | Detection only |
| `per_class_iou` | `{"iou_class_0": ..., ...}`, one entry per class id (0.0 for an absent class: do not average these yourself) |
| `inference_ms_per_image` | Mean forward-pass time per image, excluding data loading |
| `n_samples` | Images actually scored |
| `duration_seconds` | Wall time of the whole evaluation |
| `error` / `failed` | Non-empty reason / `True` when the evaluation did not produce trustworthy numbers |
| `primary_metric` | The number to rank by: `mean_iou` (segmentation, change detection), `mAP50` (detection), `accuracy` (classification) |

`to_dict()` rounds to 4 decimals and is what `save_results` and the leaderboard store.

### Absent classes

If a class occurs in neither the targets nor the predictions, it is left out of `mean_iou` and `mean_f1`. A class that is
absent from the labels but still **predicted** does count, with IoU 0: every pixel predicted as that class is an error.
Averaging a class that is empty everywhere as "0" would only lower every score for no reason, so that does not happen.

### Failures are recorded, not scored

A real evaluation can go wrong in many ways, and each is reported in `error` with all metrics zero:

| `error` starts with | Cause | What it means |
|---|---|---|
| `RuntimeError: CUDA out of memory` (or any exception text) | The model or loader raised | Nothing was scored |
| `the loader produced no batches` | Empty loader | No metrics exist |
| `N batch(es) could not be scored (prediction/target shape mismatch or class index >= num_classes)` | `num_classes` too small, labels starting at 1, or mismatched sizes | Metrics would be computed from partial data, so the result is marked failed |
| `missing dependency: ...` | `torch` not importable | Install it |
| `torchmetrics ... pycocotools` note | Detection metrics need extra packages | `pip install torchmetrics pycocotools` |

`Leaderboard.add` refuses a failed result (`ValueError`), so a crash can never appear on a saved leaderboard as a score of 0.

## Detection

Detection needs `torchmetrics` and `pycocotools`. Models must follow the torchvision convention: they take a *list* of image
tensors and return a *list* of `{"boxes", "scores", "labels"}` dicts, and the loader yields `(list_of_images, list_of_targets)`
(`collate_fn=lambda b: tuple(zip(*b))`). With a toy model that returns the ground truth perfectly and two copies shifted
by 4 and 12 pixels (real run):

| Model | mAP@50 | mAP@50-95 |
|---|---:|---:|
| exact | 1.000 | 1.000 |
| shifted 4 px | 0.000 | 0.000 |
| shifted 12 px | 0.000 | 0.000 |

The boxes are 20 × 20 to 20 × 15 pixels, so a 4-pixel shift already pushes IoU below 0.5 and the box counts as a miss. Detection scores on small
objects are very sensitive to small localisation errors; this is a property of the metric, not a fault in the evaluator.

## Latency is indicative only

`inference_ms_per_image` is the mean wall-clock time of the forward pass per image, with GPU work synchronised, on whatever
device and batch size you used. It excludes data loading, pre- and post-processing, and warm-up is **not** excluded, so the
first batch (CUDA initialisation, cuDNN autotuning) inflates it, noticeably so with few batches. Compare latencies only
between models run in the same process, on the same device, with the same batch size, on enough batches. For a
deployment figure use `pygeovision edge benchmark-onnx` ([Edge deployment](../deployment/edge.md)) on the target hardware.

## What the numbers can and cannot tell you

- **They are only as good as the test set.** Scoring on training chips, on chips that overlap training chips, or on a scene
  from the same flight, reports memorisation.
- **No confidence interval.** With 64 chips from one scene, differences of a few hundredths in mIoU are within noise. Repeat
  with other scenes or bootstrap the chips before declaring a winner.
- **Averages are over chips pooled together.** The confusion matrix is accumulated over the whole loader, so large images
  count more than small ones. This is the standard pooled mIoU, not a per-image mean.
- **Task types are limited.** `regression`, `time series` and other task names in the dataset catalogue are not evaluated by
  `ModelEvaluator`: a task it does not know is scored as segmentation, which is wrong for regression.
- **Change detection** is scored as ordinary segmentation of the change mask. `mean_iou` therefore averages "no change" and
  "change"; for the usual change metric read the IoU of the change class, `per_class_iou["iou_class_1"]`.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `FAILED  <model>  RuntimeError: CUDA out of memory` | Batch too large for the GPU | Use a smaller `batch_size`, or `device="cpu"` |
| `N batch(es) could not be scored` | Labels start at 1, `num_classes` too small, or sizes differ and cannot be resized | Subtract 1 from 1-based labels, set `num_classes` to the real count, check mask shape `(B, H, W)` |
| Every metric is 0.0 and `error` says `the loader produced no batches` | Empty loader or dataset | Check the dataset length before evaluating |
| `error` is `KeyError: "batch dict needs 'image'/'images' and 'mask'/'label' keys; got [...]"` | Dict batches with other key names | Rename the keys or yield `(images, masks)` tuples |
| mIoU far below accuracy | Rare classes missed | Read `per_class_iou`; consider class weights ([Losses](../training/losses.md)) |
| Same model gives different scores between runs | Dropout left on, or non-deterministic data order with random augmentation in the test loader | The evaluator calls `model.eval()`; remove random transforms from the test loader |
| Detection result has `error: ... torchmetrics` | Optional packages missing | `pip install torchmetrics pycocotools` |
| Leaderboard raises `ValueError: not adding failed evaluation` | You added a failed result | Check `result.failed` first, fix the cause, re-evaluate |
| Leaderboard row vanished | Same model + dataset + split was added again | Give versions different `model_name`s |
| `ms/img` differs wildly from run to run | Few batches, warm-up, shared GPU | Run more batches; ignore the first; fix the device and batch size |

```{seealso}
[Training](../training/index.md) · [Datasets catalogue](../datasets/index.md) · [Model registry](../models/registry.md) ·
[Monitoring performance in production](../monitoring/performance.md) · [Explainability](../explainability/index.md)
```
