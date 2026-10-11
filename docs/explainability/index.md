# Explainability

**Package:** `pygeovision.explainability` · **Python:** `client.xai` · **CLI:** `pygeovision explain gradcam`

## Why

A model that scores well can still be right for the wrong reason: it may recognise the *sensor stripe* next to the
buildings, the *season* rather than the crop, or the *scene edge* rather than the water. Aggregate metrics such as IoU
cannot show that. Explainability tools look at one prediction at a time and ask:

| Question | Tool | Page |
|---|---|---|
| **Where** in the image did the model look to decide this class? | GradCAM, GradCAM++ | [GradCAM](gradcam.md) |
| **How sure** is the model, pixel by pixel? | Monte Carlo Dropout | [Uncertainty](uncertainty.md) |
| **Which patches** does a transformer attend to? | Attention maps | [Attention](attention.md) |
| **Which spectral bands** matter most? | SHAP | [Band importance (SHAP)](shap.md) |

Use them to **find problems**: a saliency map on the wrong object, an uncertainty map that lights up along scene
borders, a model that ignores the near-infrared band you expected it to need. They are diagnostics, not proofs. None of
them tells you the model is *correct*; for that you need [held-out accuracy](../training/index.md).

## The four tools at a glance

| | GradCAM / GradCAM++ | Uncertainty | Attention | SHAP |
|---|---|---|---|---|
| Output | `(H, W)` map in [0, 1] | GeoTIFF: label, entropy, epistemic, aleatoric | Patch-grid map `(n, n)` | One number per band |
| Models | Any with a Conv2d layer | Segmentation models with dropout | Transformers that expose attention weights | Any PyTorch model `shap` can handle |
| Scope | One chip, or a whole GeoTIFF (`batch_explain`) | Whole GeoTIFF | One image | One chip |
| Cost | One forward + backward | `n_passes` forward passes (default 20) | One forward | Slow: minutes on large inputs |
| Needs | `torch` | `torch`, `rasterio` | `torch` | `torch`, `shap` |
| Main caveat | Layer choice changes the answer | Entropy catches only part of the errors | Weights are not importance | Zero background; noisy per chip |

## From Python

```python
from pygeovision import PyGeoVision
client = PyGeoVision()
cam   = client.xai.gradcam(model, target_layer="enc2.0")      # or gradcam_pp
unc   = client.xai.uncertainty(model, n_passes=20)
attn  = client.xai.attention(model)
shap_ = client.xai.shap(model)
```

`client.xai` only builds the objects; the classes can be imported directly from `pygeovision.explainability`
(`GradCAM`, `GradCAMPlusPlus`, `UncertaintyEstimator`, `AttentionMapExtractor`, `GeospatialSHAP`).

## From the command line

```console
$ pygeovision explain gradcam scene.tif -m unet-r50 --checkpoint best.pth --classes 6 -c 0 -o scene_gradcam.tif
  ✓ GradCAM (1 chips) → scene_gradcam.tif
```

Only GradCAM has a command; uncertainty, attention and SHAP are Python APIs. See the [GradCAM page](gradcam.md#command-line).

## A worked example used on these pages

The pages share one small model so that the numbers are comparable and reproducible. A three-layer convolutional
segmentation network (two `Conv-BatchNorm-ReLU` blocks, a `Dropout2d(0.2)`, a 1×1 classifier) was trained for 300 steps
on random 64 × 64 crops of the synthetic [OBIA demo scene](../obia/demo.md) (seed 0), then explained on a second scene
(seed 1). It reaches **96.4 % pixel accuracy** on that second scene. It is a toy, chosen because the true classes are
known exactly, so we can check whether the explanation points at the right thing. Real models are bigger and their
maps are messier.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| The map highlights the wrong object, or the whole image evenly | Wrong target layer, untrained weights, or a real model weakness | [GradCAM → choosing the layer](gradcam.md#choosing-the-target-layer); check you loaded your trained weights |
| `ImportError: pip install shap` (returned as `{"error": ...}`) | Optional dependency | `pip install shap` |
| Everything is zeros | Hooks did not fire / no attention found | See the "What to do" table on each page |
| Maps look different on every run (uncertainty) | Dropout is random by design | Raise `n_passes`; fix `torch.manual_seed` for exact repeats |
| You want to explain classical (non-neural) OBIA classifiers | These tools are for neural networks | OBIA rules and trees are readable already: [Classification → rules](../obia/classification.md) |

```{toctree}
:hidden:
:maxdepth: 1

gradcam
uncertainty
attention
shap
```
