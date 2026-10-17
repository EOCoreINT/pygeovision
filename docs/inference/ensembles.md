# Combining several models

An **ensemble** asks several models about the same scene and combines their answers. It is the oldest trick for a steadier map:
mistakes that one model makes alone tend to be outvoted. It is not magic, though, and the example below shows a case where an
ensemble is *worse* than its best member. Read the "when it helps" section before you spend the extra compute.

```{figure} ../_static/img/flow-ensemble.svg
:alt: Ensemble inference: each model predicts class probabilities for the same scene, the probabilities are weighted and fused, and the most likely class is written
:width: 100%
:class: pgv-figure

Each model predicts the whole scene; the fused probabilities, not the three finished maps, decide each pixel.
```

## The class

```python
from pygeovision.inference.stream import EnsembleInference     # also importable from pygeovision.inference

ens = EnsembleInference([model_a, model_b, model_c], weights=[1, 1, 2], fusion="mean", num_classes=2,
                        normalise=False)
result = ens.infer("scene.tif", "ensemble.tif")
print(result)      # {'success': True, 'output_path': 'ensemble.tif', 'n_models': 3}
```

| Argument | Default | Meaning |
|---|---|---|
| `models` | required | A list of PyTorch models, **all** trained for the same classes, bands and input scaling. At least one |
| `weights` | equal | One relative importance per model (`[1, 1, 2]` makes the third count double). A different length raises `ValueError` |
| `fusion` | `"mean"` | `"mean"`: average the weighted probabilities. `"max"`: take, for each class, the highest weighted probability. Anything else raises `ValueError` |
| `num_classes` | 2 | Classes in each model's output |
| `device` | auto | `"cuda"` when available, otherwise `"cpu"` |
| `normalise` | `True` | The per-band 2nd to 98th percentile stretch of [`TiledInference`](index.md). Use `False` when the models were trained on values used as stored |

How a scene is processed:

1. Each model predicts the scene with the normal tiled engine (default chip size and overlap) and writes **class probabilities** to
   a temporary file (8-bit: probability in steps of 1/255). The temporary files are deleted, also when something fails.
2. The probabilities are multiplied by the model's weight and fused per pixel by `fusion`.
3. The class with the highest fused value is written as a one-band `uint8` map. Pixels that no model was asked to predict (outside
   the data, or nodata) become **255**, and the file declares `nodata=255`.

It costs **one full prediction per model**, one after the other, so three models take about three times as long as one. A model that
cannot read the scene (wrong band count) stops the run with `{"success": False, "error": ..., "n_models": 3}`.

```{admonition} The chip size and overlap are the engine's defaults
:class: note
`EnsembleInference` does not take `chip_size` or `overlap`; each model is run with the defaults of `TiledInference` (512 and 64).
If you need other values, run `TiledInference(..., return_probabilities=True)` for each model yourself and fuse the rasters with
NumPy; the fusion itself is a few lines (see "Fusing by hand" below).
```

## A real comparison

Three small networks trained on the same kind of noisy scene (a bright square on a noisy background): `a` and `b` are trained
properly with different random seeds; `c` is deliberately over-eager about the "bright" class, as a poorly calibrated model would
be. Mean IoU against the true square, on a scene none of them trained on:

| Prediction | Mean IoU |
|---|---:|
| Model `a` alone | 0.951 |
| Model `b` alone | 0.947 |
| Model `c` alone | 0.793 |
| Ensemble, `mean`, equal weights | 0.926 |
| Ensemble, `mean`, weights `[1, 1, 0.25]` | 0.944 |
| Ensemble, `max`, equal weights | 0.902 |

Read this carefully:

- **The ensemble is not better than the best member.** One weak model drags the average down, and the lowest weight for it only
  recovers most of the loss.
- **`max` is the most sensitive to a weak or over-confident model**, because it takes the loudest vote per class, not the typical one.
  Prefer `mean` unless you want "any model says so" behaviour (for example a flood alarm where a missed flood costs far more than a
  false one).
- **Weights are for models you trust differently**, for example a model trained on your region and one trained elsewhere.

The numbers come from toy networks and one scene; the *pattern* is the point. Measure on your own validation data before choosing.

## When an ensemble helps

| Situation | Ensemble worth it? |
|---|---|
| Several models of similar quality trained on **different data or seeds** | Yes. Errors are partly independent, so the average is steadier |
| The same model at several training epochs | Rarely. The errors are almost the same |
| One strong model and several weak ones | No. Use the strong one, or give the weak ones small weights |
| You need a measure of how sure the map is | Better: [uncertainty maps](../explainability/index.md) show where a model is unsure |
| Limited time or GPU | No. The cost is linear in the number of models |

## Checklist before you trust one

1. Every model gets the **same bands in the same order** and was trained with the **same scaling**. Mixing a model trained on
   stretched values with one trained on raw values gives nonsense even if both load.
2. Every model has the same class meanings (class 1 is "building" for all).
3. Score each member **and** the ensemble on data none of them saw ([Evaluating](../experiments/benchmark.md)).
4. Look at the difference map: where do the models disagree? Disagreement is information. (Subtract two of the output maps in a GIS or with NumPy.)

## Fusing by hand

When you need your own chip size, or a fusion rule the class does not have (for example median, or "at least two models agree"):

```python
import numpy as np
import rasterio
from pygeovision.inference import TiledInference

probs = []
for model in models:
    TiledInference(model, chip_size=256, overlap=32, num_classes=2, device="cpu").infer(
        "scene.tif", "tmp.tif", return_probabilities=True, normalise=False)
    with rasterio.open("tmp.tif") as src:
        probs.append(src.read().astype("float32") / 255.0)           # (classes, H, W), 0..1
        profile = src.profile

agree = (np.stack([p.argmax(0) for p in probs]) == 1).sum(0) >= 2   # "at least two models say class 1"
profile.update(count=1, dtype="uint8", nodata=None)
with rasterio.open("two_of_three.tif", "w", **profile) as dst:
    dst.write(agree.astype("uint8")[None])
```

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: fusion must be 'mean' or 'max'` | Another word was given | Use one of the two, or [fuse by hand](#fusing-by-hand) |
| `ValueError: 2 weights for 3 models` | `weights` and `models` differ in length | One weight per model |
| `{"success": False, "error": ...}` | One model could not read the scene | The error is that model's message; check bands and `normalise` |
| The map has 255 where there is imagery | The scene's nodata value covers those pixels | Check the file's nodata ([Inference: what the output is](index.md#what-the-output-is)) |
| The ensemble looks the same as one model | The other models agree with it, or have tiny weights | Compare each member's own map |
| Out of memory | Each model's prediction is held in memory until fusion | Run on a smaller piece, or fuse by hand one scene piece at a time |

## Related

- [Inference overview](index.md), [Predicting many scenes](batch.md), [Tiles, overlap and seams](tiles.md).
- [Explainability and uncertainty](../explainability/index.md): where a model is unsure.
