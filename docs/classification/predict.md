# Predict with a classifier

## Zero-shot with CLIP (no training)

```bash
pygeovision ai classify scene --input tile.tif --classes "forest,water,urban,agriculture"
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
scores = client.classification.scene("tile.tif", categories=["forest", "water", "urban", "agriculture"])
```

The result is a dictionary from each category you gave to a probability; the probabilities sum to 1 **over the categories
you supplied**, so adding a category changes the others. Wording matters: "flooded farmland" and "water" are different
questions. For remote-sensing imagery try RemoteCLIP (`remoteclip-b32`, [CLIP and RemoteCLIP](../foundation-models/clip.md)). To
rank many tiles against one description, use [CLIP search](../vision-language/clip.md).

## Land cover from a reference map

```bash
pygeovision ai classify land-cover --input s2.tif --output lc.tif
```

This downloads the ESA WorldCover map for the extent of the image and writes it on the image's grid. It is a published
product (10 m, 2021 by default), not a classification of your image: use it for labels and context.

## With a classifier you trained

Load the checkpoint into the same model you trained, then classify chips:

```python
import torch
from pygeovision.models import get_model

classes = ["a", "b", "c"]                                 # in the order used for training
model = get_model("convnext-t", num_classes=3, in_channels=4, pretrained=False)
model.load_state_dict(torch.load("best.pth", weights_only=True)["model_state_dict"])
model.eval()

with torch.no_grad():
    probabilities = model(chip_batch).softmax(dim=1)      # chip_batch: (n, 4, H, W), scaled as in training
    best = probabilities.argmax(dim=1)
```

## A grid of chips over a scene

A classifier labels a chip, not a pixel. To map a scene, classify a grid of chips and keep each chip's georeferenced corner.
Verified on a synthetic 128 × 192 four-band scene with 64-pixel chips:

```python
import rasterio
import torch
from rasterio.windows import Window

CHIP = 64
rows = []
with rasterio.open("scene.tif") as src, torch.no_grad():
    for row in range(0, src.height - CHIP + 1, CHIP):
        for col in range(0, src.width - CHIP + 1, CHIP):
            x = torch.from_numpy(src.read(window=Window(col, row, CHIP, CHIP)).astype("float32"))[None]
            p = model(x).softmax(1)[0]
            west, north = src.transform * (col, row)
            rows.append({"col": col, "row": row, "class": classes[int(p.argmax())],
                         "confidence": round(float(p.max()), 3), "west": west, "north": north})
```

```text
{'col': 0, 'row': 0, 'class': 'a', 'confidence': 0.392, 'west': 500000.0, 'north': 5000000.0}
{'col': 64, 'row': 0, 'class': 'a', 'confidence': 0.407, 'west': 500640.0, 'north': 5000000.0}
{'col': 128, 'row': 0, 'class': 'a', 'confidence': 0.386, 'west': 501280.0, 'north': 5000000.0}
…                                                    (6 chips: 2 rows × 3 columns)
```

(The model here is the toy classifier from the [training page](train.md); its confidence near 0.4 for three classes is
close to guessing, which is what an under-trained model looks like.) Write the rows to a table or turn them into
polygons with `rasterio.windows.bounds`.

## Few-shot, with 2 to 10 labelled chips per class

See [Few-shot classification](few-shot.md): a frozen DINOv2 plus class prototypes; nothing is trained.

## Reading the output

| Value | Meaning |
|---|---|
| `class` | The most likely class |
| `confidence` | Its softmax probability. High is not the same as correct; check it against a held-out set |
| Many chips with confidence near `1 / classes` | The model is unsure; do not trust those labels |

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `size mismatch` loading the checkpoint | Model name, classes or bands differ from training | Build the model exactly as in training |
| Everything is one class on real data | Pixel scaling differs from training | Apply the same scaling ([why](../inference/index.md#matching-the-input-to-training)) |
| Edge chips are skipped | The grid loop only takes full chips | Pad the scene or add an edge window |
