# Uncertainty maps (Monte Carlo Dropout)

**Class:** `UncertaintyEstimator` (`pygeovision.explainability`) · **Python:** `client.xai.uncertainty(model, n_passes=20)`

## Why

A segmentation map looks equally confident everywhere. It is not. Pixels on a field boundary, in shadow, or on a class the
model hardly saw in training are guesses, and you want to **see which ones**, so you can send a person to check them, add
training samples there, or mask them from area statistics.

Monte Carlo (MC) Dropout is a cheap way to get such a map from a network you already have. Dropout normally switches off
random units during *training* only. Here it stays on during inference, and the chip is predicted many times; every pass is
a slightly different network. Where those predictions disagree, the model is unsure.

## How to use

```python
from pygeovision.explainability import UncertaintyEstimator

est = UncertaintyEstimator(model, n_passes=50)
result = est.estimate("scene.tif", "scene_uncertainty.tif", chip_size=512, overlap=64)
print(result)
```

```text
{'success': True, 'image_size': (160, 160), 'n_mc_passes': 50, 'n_dropout_layers': 1,
 'mean_entropy': 0.240, 'mean_epistemic': 0.00186, 'mean_aleatoric': 0.199,
 'high_uncertainty_fraction': 0.110, 'output_path': 'scene_uncertainty.tif'}
```

(real output of the [example model](index.md#a-worked-example-used-on-these-pages) on demo scene B, 20 passes.)

### `UncertaintyEstimator(model, n_passes=20, device=None)`

| Argument | Meaning |
|---|---|
| `model` | A segmentation network that returns class scores `(B, classes, H, W)` and **contains at least one dropout layer** (`nn.Dropout`, `Dropout2d`, `Dropout3d`, `AlphaDropout`, …) |
| `n_passes` | Stochastic passes per chip. More is steadier and slower (see below) |
| `device` | Default `cuda` if available, else `cpu` |

### `estimate(image_path, output_path=None, chip_size=512, overlap=64, save_epistemic=True, save_aleatoric=True)`

| Argument | Meaning |
|---|---|
| `image_path` | GeoTIFF with the bands the model was trained on |
| `output_path` | Where to write the result raster; `None` computes the statistics only |
| `chip_size`, `overlap` | Tiling. `chip_size` must be larger than `overlap` or you get `{"success": False, "error": ...}`. Where chips overlap **the later chip wins** (no blending) |
| `save_epistemic`, `save_aleatoric` | Whether to include those bands |

What happens: the model is put in eval mode (BatchNorm uses its stored statistics and is **not** updated), then only the
dropout layers are switched back to random mode. Every chip is min–max normalised per band, run `n_passes` times, and
summarised. Afterwards the model is back in eval mode.

### The output raster

A float32 GeoTIFF with band descriptions:

| Band | Name | Meaning | Range |
|---|---|---|---|
| 1 | `label` | Class with the highest **mean** probability over the passes (stored as float) | 0…classes−1 |
| 2 | `entropy` | Entropy of the mean probabilities: **total uncertainty** | 0 (certain) … ln(classes) |
| 3 | `epistemic` | Variance of each class probability across passes, averaged over classes | 0 … 0.25 |
| 4 | `aleatoric` | Mean entropy of the individual passes | 0 … ln(classes) |

For 6 classes the maximum entropy is ln 6 = 1.79 nats. Tags: `bands` (the names, `|`-separated) and `n_mc_passes`.

How to read them:

- **Entropy** is the number to map first. High: the mean prediction is spread over several classes.
- **Epistemic high**: the passes disagree with each other, so the *model* is unsure. More training data of that kind helps.
- **Aleatoric high** while epistemic is low: every pass is itself undecided, so the *data* is ambiguous (mixed pixels,
  boundaries). More data will not fix that; a different input or class definition might.
- Epistemic here is a variance, not the mutual information used in some papers. Use it to rank, not as a calibrated
  probability.

The returned dict:

| Key | Meaning |
|---|---|
| `success` | `False` with an `error` text on bad input |
| `image_size`, `n_mc_passes` | Scene size; passes actually run (1 if the model has no dropout) |
| `n_dropout_layers` | Dropout layers found |
| `mean_entropy`, `mean_epistemic`, `mean_aleatoric` | Scene averages |
| `high_uncertainty_fraction` | Share of pixels with entropy above **0.5 nats** (a fixed threshold, not tuned to the number of classes) |
| `warning` | Present when no dropout layer exists |
| `output_path` | The written file |

## What the maps say, tested

The example model on demo scene B (160 × 160 pixels, true classes known), 20 passes:

| | Mean entropy |
|---|---:|
| Pixels in a 3-pixel band around true class boundaries (23 % of the scene) | **0.389** |
| All other pixels | 0.195 |
| Pixels the model got **wrong** (3.6 % of the scene) | **0.433** |
| Pixels it got right | 0.233 |

So the uncertainty **is** concentrated on boundaries and on errors, roughly twice as high. That is the useful finding.

Now the sobering one. Take the 10 % of pixels with the highest entropy: they contain only **28 %** of the wrong pixels.
Uncertainty is **not an error detector**. Most mistakes are made confidently (this model is confidently wrong about whole
objects), and a large share of the high-entropy pixels are correct ones on boundaries. Use the map to prioritise review,
not to promise that low-uncertainty pixels are right.

### How many passes?

Summary numbers move with the random draw. Mean entropy of the same model and scene over 6 different seeds:

| Passes | Mean over seeds | Range |
|---:|---:|---:|
| 20 | 0.209 | 0.178 – 0.286 |
| 100 | 0.214 | 0.202 – 0.243 |

With 20 passes on a layer of 16 channels (the example uses `Dropout2d`, which drops whole channels) the scene average
varied by about ±25 %. Use **50–100 passes** for numbers you want to compare, and set `torch.manual_seed(...)` for
reproducible maps. Time grows linearly: on the 160 × 160 scene on a CPU, 5 / 20 / 50 passes took 0.2 / 0.7 / 1.5 s.

## Memory and speed

Four `(H, W)` float32 arrays plus the image are held in memory, independent of `n_passes` and the number of classes. A
10 000 × 10 000 scene needs about 1.6 GB for the outputs plus the image itself. Run time is
`n_passes × number of chips × one forward pass`; on a GPU choose the largest `chip_size` that fits.

## Requirements and limits

- **The model needs dropout.** Architectures differ: with randomly initialised registry models, `unet-r50` has **0**
  dropout layers, `deeplab-r50` 1, `unet-efficientb4` 1 and `segformer-b0` 17. Count yours:
  `sum(isinstance(m, torch.nn.modules.dropout._DropoutNd) for m in model.modules())`. Without it every pass is identical, the result is `n_passes = 1`, epistemic is 0, and the
  dict carries a `warning`. Do not add dropout at inference to a model trained without it: the model never learned to cope
  with it and the "uncertainty" is just damage.
- **Segmentation output only.** A classification model (`(B, classes)`) returns
  `{"success": False, "error": "model output must be (B, classes, H, W) class scores; ..."}`.
- **Tiling is per chip**: normalisation and the passes are per chip, so chip borders can show faint seams.
- **MC Dropout is an approximation.** Uncertainty depends on where dropout sits and its rate. Compare maps from the *same*
  model; do not compare absolute values between models.
- **Scores must be logits.** The estimator applies `softmax`. A model whose last layer already outputs probabilities gives
  flattened results.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `warning: model has no dropout layers…` | The network has none | Retrain with dropout, or use another uncertainty method (ensembles of models) |
| `{"success": False, "error": "chip_size (64) must be greater than overlap…"}` | Bad tiling | Reduce `overlap` or increase `chip_size` |
| `error: model output must be (B, classes, H, W)` | Classification model | This tool is for segmentation models |
| Scene average changes every run | Few passes | 50–100 passes; seed torch |
| Uncertainty is high everywhere | Model untrained, wrong input scaling, or a domain far from training | Check preprocessing; compare with a scene from the training area |
| Uncertainty is near 0 but results are wrong | Confidently wrong | Expected for systematic errors; validate with reference data |
| Straight seams along chip edges | Per-chip normalisation, no blending | Raise `overlap` and `chip_size` |
| `MemoryError` | Very large raster | Process in parts |
| Model weights changed after the call | Used to happen with models in train mode (BatchNorm updated). Fixed: the model is evaluated in eval mode and returned in eval mode | Update to the current version |
| `high_uncertainty_fraction` always ~1 | 0.5 nat threshold with many classes or weak model | Use the entropy band and your own threshold |

```{seealso}
[GradCAM](gradcam.md) · [Training](../training/index.md) · [Inference](../inference/index.md)
```
