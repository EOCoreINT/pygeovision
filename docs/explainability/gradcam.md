# GradCAM and GradCAM++

**Classes:** `GradCAM`, `GradCAMPlusPlus` (`pygeovision.explainability`) · **CLI:** `pygeovision explain gradcam`

## Why

When a segmentation or classification network says "water" for a chip, GradCAM shows **which parts of the input pushed the
score for that class up**. It does so by taking the activations of one convolution layer and weighting each channel by
how strongly the class score depends on it. The result is a heat map the size of the input.

Use it to check that the model looks at what a human would look at. A building detector that lights up the shadows,
or a flood model that ignores the water body and lights up the scene border, has learned a shortcut.

## How to use

```python
from pygeovision.explainability import GradCAM

cam = GradCAM(model, target_layer="enc2.0")             # a convolution layer, by its name in model.named_modules()
saliency = cam.explain(chip, class_idx=0)               # chip: tensor/array (C, H, W) or (1, C, H, W)
print(saliency.shape, saliency.min(), saliency.max())   # (H, W) of the chip, scaled to 0.0 ... 1.0
cam.visualise("scene.tif", saliency, "gradcam_water.png")
```

### `GradCAM(model, target_layer=None, device=None)`

| Argument | Meaning |
|---|---|
| `model` | Any `torch.nn.Module` with at least one `Conv2d`/`ConvTranspose2d`. Classification (`(B, classes)`) and segmentation (`(B, classes, H, W)`) outputs are supported |
| `target_layer` | Name of a convolution layer as printed by `[n for n, m in model.named_modules()]`. `None` = the **last** convolution |
| `device` | Default `cuda` if available, else `cpu`. **The model is moved to this device and put in eval mode** |

An unknown `target_layer` raises `ValueError` listing the convolution layers, instead of explaining something else.

### `explain(image, class_idx=None, normalize=True)`

| Argument | Meaning |
|---|---|
| `image` | One chip as a tensor or NumPy array, `(C, H, W)` or `(1, C, H, W)`. Preprocess it **exactly as in training** |
| `class_idx` | The class to explain. `None` = the predicted class (highest logit; for segmentation the highest mean logit over the chip) |
| `normalize` | Scale to `[0, 1]` (default). With `False` you get the raw weighted activations, comparable between chips |

Returns a float `(H, W)` array at the **input size** (the map is computed at the layer's resolution and enlarged by
bilinear interpolation, so it is smoother than the pixels). If the hooks never fired (a layer not on the forward path),
you get zeros and a log warning.

**What is explained.** For a classification model: the class's score. For a segmentation model: the class's score
**summed over every pixel of the chip**. The map therefore answers "where does the evidence for this class, anywhere in
the chip, come from?", not "why is this one pixel water?".

### Whole images: `batch_explain`

```python
res = cam.batch_explain("scene.tif", "scene_gradcam.tif", class_idx=0, chip_size=512, overlap=64)
print(res)      # {'success': True, 'output_path': 'scene_gradcam.tif', 'image_size': (rows, cols), 'n_chips': ...}
```

Tiles the GeoTIFF into chips, explains each, and averages overlaps into a one-band float32 GeoTIFF with the same
georeferencing (tags `method` and `class_idx`). Be aware that:

- **Every chip is normalised on its own**: each band is min–max scaled per chip, and each chip's map is scaled to 0–1.
  Colours are comparable *within* a chip, not strictly across chips. If the model was trained on differently scaled data
  the explanation is of a different input than the one it saw in training.
- The whole image is loaded into memory.
- Problems with the tiling or a missing dependency come back as `{"success": False, "error": ...}`, not as exceptions.
  `chip_size` must exceed `overlap`.

### Picture: `visualise`

```python
cam.visualise("scene.tif", saliency, "gradcam.png", alpha=0.6, colormap="jet")
```

Writes a three-panel PNG: the input (first three bands, stretched; one band is shown grey), the map, and an overlay. The
image is read only for display; it should show the area that was explained.

Open the result next to the image and ask the one question that matters: is the heat where a person would look? A good check
is a class with an unmistakable object in the scene (a lake for "water"): measure the share of the saliency that falls on
those pixels and compare it with the share of the scene they cover. If they are about the same, the map is uninformative.

## Choosing the target layer

This choice changes the answer more than any other setting.

| Layer | You get | Use when |
|---|---|---|
| The final classifier convolution (what `target_layer=None` picks in most segmentation nets) | Essentially the **class logit map**, i.e. the prediction itself | Almost never; it explains nothing the output does not already show |
| The last *feature* convolution before the head | A coarse "evidence" map | **Default choice** for segmentation |
| An early layer | Fine detail, but it reflects edges and textures rather than classes | To check low-level cues |
| A deep encoder layer (low resolution) | Blobby map, correct region | Classification networks (e.g. `layer4` of a ResNet) |

To compare layers (and GradCAM against GradCAM++) objectively, use a class whose pixels you know, for example water from a
reference mask `ref_water`:

```python
for layer in ["enc1.0", "enc2.0", "head"]:
    sal = GradCAM(model, target_layer=layer).explain(chip, class_idx=water_idx)
    share_on_water = float(sal[ref_water].sum() / sal.sum())      # compare with ref_water.mean()
    print(layer, round(share_on_water, 2), "uninformative would be", round(float(ref_water.mean()), 2))
```

Read the result carefully:

- With the final classifier layer the share is high almost by construction, because the map **is** the class logit map. A
  high score there is not evidence of anything.
- **GradCAM++ is not a better GradCAM.** It was designed for classification scores; with a score summed over a whole chip its
  weighting behaves differently, and on a segmentation model it can put much less mass on the object. Where the two disagree,
  test each against something you know before trusting either.
- Only an intermediate feature layer is informative. On your model, try several layers and look at the maps.

For torchvision models the last block of the backbone is the usual choice (`"layer4.2.conv3"` for a ResNet-50), for
U-Net-like models the last decoder convolution before the head.

## GradCAM++

`GradCAMPlusPlus(model, target_layer=None, device=None)` has the same interface. It replaces the plain mean gradient
with per-pixel coefficients built from higher powers of the gradients (Chattopadhay et al., 2018). The claim in the
literature is better localisation when **several objects of the class** are in one image. Try both on your model and
compare with objects you know.

## Command line

```console
$ pygeovision explain gradcam scene.tif -m unet-r50 --checkpoint best.pth --classes 6 -c 0 -o scene_gradcam.tif
  ✓ GradCAM (1 chips) → scene_gradcam.tif
```

| Option | Meaning |
|---|---|
| `IMAGE_PATH` | The GeoTIFF |
| `-m, --model` | A model name from the [registry](../models/registry.md) (for example `unet-r50`) |
| `--checkpoint` | Your trained weights (strict load). **Without it the registry model may download pretrained weights, and the saliency of a network that never saw your task is noise** |
| `--classes` | Number of output classes (default 2) |
| `-c, --class-idx` | Class to explain (default 1) |
| `--in-channels` | Default: the band count of the image |
| `-o, --output` | Default `<image>_gradcam.tif` |

The command uses the last convolution (`target_layer=None`) and a chip size of 512; for another layer use Python. A
failure (unknown model, wrong tiling, missing file) prints `✗ <reason>` and exits with status 1.

Open the result in QGIS over the scene, set a pseudo-colour ramp 0–1 and 50 % opacity.

## Limits

- **A saliency map shows sensitivity, not reasoning.** It says where changing the input changes the score, not what the
  model concluded.
- **Sharp maps are not necessarily right.** A map can look convincing and be unrelated to the decision. Always check
  against objects you know (the share-on-water check above is one).
- **Resolution is limited by the layer.** A layer at 1/16 of the input resolution yields a blob roughly 16 pixels wide.
- **Models with transformers** have no convolution to explain; use [attention maps](attention.md) (and note that attention
  is not importance either).
- The explained model is **moved to `device` and set to eval mode**; if you need it in train mode afterwards, switch it back.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| Map is the same as the prediction | Target layer is the final classifier | Choose an earlier layer ([table](#choosing-the-target-layer)) |
| Map is nearly uniform or noise | Weights untrained / wrong checkpoint; wrong preprocessing | Load your checkpoint; preprocess as in training |
| `ValueError: target_layer 'x' is not a convolution layer…` | Typo or non-conv layer | Use a name from the printed list |
| `ValueError: No Conv2d layers found` | Pure transformer/MLP | Use [attention](attention.md) or [SHAP](shap.md) |
| All zeros and "hooks did not fire" | Layer is unused in the forward pass | Pick a layer on the forward path |
| `IndexError` on the class | `class_idx` ≥ number of outputs | Use a valid class index |
| Out-of-memory in `batch_explain` | Large image loaded whole | Cut the scene into parts first, or lower `chip_size` |
| `batch_explain` returns `success: False` | `chip_size <= overlap` or missing dependency | Read `res["error"]` |
| Visible chip seams in the output | Per-chip normalisation | Raise `overlap`; use `normalize=False` in `explain` for chip-by-chip work |
| `RuntimeError` about in-place operations | Model modifies the layer's output in place | Handled for the usual `ReLU(inplace=True)` case; for others pick another layer |
| Colours differ between two scenes | Each map is scaled to its own 0–1 | Use `normalize=False` and a shared scale |
| GradCAM++ disagrees with GradCAM | Different weighting | Check both against known objects |

```{seealso}
[Uncertainty](uncertainty.md) · [Attention maps](attention.md) · [Band importance](shap.md) · [Models](../models/index.md)
```
