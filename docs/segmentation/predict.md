# Predict

There are three ways to get a segmentation map, in increasing order of effort: a method that needs **no training**, a
**trained network** on a file you already have, and a **pipeline** that finds the imagery for you.

## Without training

| Command | Method | Needs |
|---|---|---|
| `pygeovision ai segment water -i stack.tif -o water.tif` | NDWI = (green − NIR) / (green + NIR) > 0 | A blue-green-red-NIR stack (green = band 2, NIR = band 4). Other layouts: `--green-band N --nir-band M` |
| `pygeovision ai segment sam -i image.tif -o masks.tif` | Segment Anything, automatic masks | The SAM weights (downloaded on first use) |
| `pygeovision ai segment buildings -i image.tif -o b.tif` | SAM automatic masks used as building candidates | SAM weights. Candidate shapes, not a validated building map |

**Expected output of `ai segment water`.** A single-band `uint8` GeoTIFF on the input grid: `1` = water, `0` = not
water. On a 4-band synthetic scene:

```text
Running native water segmentation on scene.tif...
✓ Segmentation complete → water.tif
```

An image with fewer than four bands needs the band numbers, and the command stops with a message instead of guessing
(an RGB image would otherwise yield "red minus green", which is not water):

```text
Error: scene.tif has 3 band(s). Water mapping by NDWI needs a green and a near-infrared band: pass green_band= and nir_band= ...
```

For real water mapping use the [`water_bodies`](../pipelines/catalog/water_bodies.md) pipeline: it corrects the
radiometry, masks cloud and reports areas. The command above is a quick tool for imagery you already prepared.

## With a network you trained

Training produces `best.pth`. Running it over a scene needs the same three things you gave the trainer: the **model
name**, the **number of classes and bands**, and the **same scaling of pixel values**.

```bash
pygeovision infer predict scene.tif -m unet-r50 --checkpoint runs/seg/checkpoints/best.pth \
        -c 2 --in-channels 4 --chip-size 128 --overlap 16 --no-normalise -o prediction.tif
```

```python
import torch
from pygeovision.models import get_model
from pygeovision.inference import TiledInference

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)
model.load_state_dict(torch.load("runs/seg/checkpoints/best.pth", weights_only=True)["model_state_dict"])

engine = TiledInference(model, chip_size=128, overlap=16, num_classes=2, blend_mode="gaussian")
result = engine.infer("scene.tif", "prediction.tif", normalise=False)
```

The scene is cut into overlapping chips, each is predicted, and the results are blended into one seamless raster on the
input grid. The output is a single-band `uint8` GeoTIFF of class numbers; 255 marks pixels that were no data in the
input. Everything about chip size, overlap, blending and memory is on the [inference pages](../inference/index.md):

- [Predicting a scene](../inference/index.md#predicting-a-scene) and [matching the input to training](../inference/index.md#matching-the-input-to-training)
- [Tiles, overlap and seams](../inference/tiles.md)
- [Predicting many scenes](../inference/batch.md) and [combining several models](../inference/ensembles.md)

## With the model object in Python

```python
import torch
from pygeovision.models import get_model

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)
state = torch.load("runs/seg/checkpoints/best.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state["model_state_dict"])
model.eval()

with torch.no_grad():
    chip = torch.rand(1, 4, 128, 128)                    # replace with a real, correctly scaled chip
    probabilities = model(chip).softmax(dim=1)           # (1, 2, 128, 128)
    labels = probabilities.argmax(dim=1)                 # (1, 128, 128)
```

Use this for experiments and for custom post-processing. For a whole scene use the tiled engine above; running a large
scene as one tensor runs out of memory and shows seams.

## After the map

| You want | Do |
|---|---|
| Polygons instead of pixels | [Post-processing](../data/postprocessing.md) (`vectorise`, `sieve_filter`, `smooth`) |
| Areas per class | `pygeovision postprocess class-stats` (or `PostProcessor.class_statistics`) |
| Accuracy numbers | Compare with independent reference samples: [validation](../data/validation.md) |
| Why a pixel got its class | [Explainability](../explainability/index.md) |
| A report | [Reporting](../reporting/index.md) |
