# Super-resolution

Super-resolution produces a **larger, sharper-looking image from a smaller one**. It makes a 10 m image into a 2.5 m
image by *predicting* the detail that is not there. The result is plausible, not measured: **do not use it to measure
small objects, count things, or compute areas.**

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Model
:link: models
:link-type: doc
`swinir-m` (Swin2SR, 4×): measured size and shapes.
:::

:::{grid-item-card} Run it on a GeoTIFF
:link: run
:link-type: doc
Tiling, georeferencing the output, bands, and checks.
:::
::::

## What it is for

| Good use | Not a good use |
|---|---|
| Making imagery easier to read for a person (a labeller, a stakeholder) | Measuring a building, a field or a road width |
| Giving a model a sharper look at texture it can already see | Counting vehicles, trees or houses |
| Making a figure for a report, clearly labelled as enhanced | Feeding the result into a change-detection or area calculation as if it were data |

A super-resolved image has more pixels but **no more information** than the original. A road that is one pixel wide in the
input does not become a measured road in the output; it becomes a plausible-looking line.

## What goes in and what comes out

```{figure} ../_static/img/flow-super-resolution.svg
:alt: Super-resolution: tile the image, upscale each tile four times, stitch, write with a finer pixel size
:width: 100%
:class: pgv-figure

A low-resolution raster in, a raster with four times as many pixels in each direction out, on the same bounds.
```

`swinir-m` (Swin2SR, 4×) takes `(1, 3, 32, 32)` and returns `{'reconstruction': (1, 3, 128, 128)}`: every pixel becomes
4 × 4. The pretrained weights are for **three-band RGB** only; for other band counts build with `pretrained=False` and train,
or process three bands at a time.

```python
import torch
from pygeovision.models import get_model

model = get_model("swinir-m", pretrained=False, in_channels=3).eval()
with torch.no_grad():
    y = model(torch.rand(1, 3, 32, 32))["reconstruction"]        # (1, 3, 128, 128)
```

## Limits to know

- **RGB only for the pretrained model.** With more bands you must train your own weights, with pairs of low- and
  high-resolution images of the same sensor.
- **The pretrained model was trained on photographs** degraded in a particular way. Satellite imagery degraded differently
  (a coarser sensor, not a blurred photo) may come out with artefacts. Look at the result before using it.
- **Edges of tiles.** Large rasters are processed in tiles; without overlap you get visible seams ([run it](run.md) shows how to avoid them).
- **Window size.** The network needs input whose sides are a multiple of 8; the recipe pads tiles to a fixed size.
- **Cost.** Four times the pixels in each direction is 16 times the pixels. A 10,000 × 10,000 raster becomes 40,000 × 40,000.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError` about RGB-only weights | `in_channels != 3` with `pretrained=True` | Use three bands, or `pretrained=False` and train |
| Visible grid of seams | Tiles without context | Add padding and crop it, as in [run it](run.md) |
| Hallucinated texture (stripes, grain) | The model invents detail | Use a smaller scale, compare with the original, never measure on it |
| Shapes error inside the model | Tile sides not a multiple of 8 | Pad every tile to the same size, as in [run it](run.md) |
