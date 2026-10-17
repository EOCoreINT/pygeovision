# Run super-resolution

The model upscales a tensor; to upscale a **georeferenced raster** you also tile it, stitch the result and write it with a
finer pixel size so it lands on the same ground. This recipe does that. It was run on a synthetic 4-band 130 × 100 pixel
raster at 10 m with `pretrained=False` (random weights, so the *values* mean nothing; the *geometry* is what is checked).

```python
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from pygeovision.models import get_model

SCALE, TILE, PAD = 4, 48, 8                  # the model makes every pixel 4 x 4; each tile is read with 8 pixels of context
SIZE = TILE + 2 * PAD                        # 64, a multiple of the model's window size (8)

model = get_model("swinir-m", in_channels=4, pretrained=False).eval()      # RGB + pretrained=True for real use

with rasterio.open("low.tif") as src:
    profile = src.profile
    h, w = src.height, src.width
    out = np.zeros((src.count, h * SCALE, w * SCALE), dtype="float32")
    with torch.no_grad():
        for row in range(0, h, TILE):
            for col in range(0, w, TILE):
                r0, c0 = row - PAD, col - PAD
                x = src.read(window=((r0, r0 + SIZE), (c0, c0 + SIZE)), boundless=True, fill_value=0).astype("float32")
                y = model(torch.from_numpy(x)[None])["reconstruction"][0].numpy()     # (bands, 256, 256)
                hh, ww = min(TILE, h - row), min(TILE, w - col)
                out[:, row * SCALE:(row + hh) * SCALE, col * SCALE:(col + ww) * SCALE] = \
                    y[:, PAD * SCALE:(PAD + hh) * SCALE, PAD * SCALE:(PAD + ww) * SCALE]
    transform = src.transform * Affine.scale(1 / SCALE)                              # pixels are four times smaller

profile.update(height=h * SCALE, width=w * SCALE, transform=transform)
for key in ("blockxsize", "blockysize", "tiled"):
    profile.pop(key, None)
with rasterio.open("high.tif", "w", **profile) as dst:
    dst.write(out)
```

```text
520 400 (2.5, 2.5) BoundingBox(left=500000.0, bottom=4999000.0, right=501300.0, top=5000000.0)
```

The output is 520 × 400 pixels at 2.5 m with exactly the bounds of the input. That is the check to make with any
super-resolution result: **same bounds, one quarter of the pixel size**.

## How the recipe avoids seams

Each tile is read with `PAD` pixels of context on every side (outside the raster the read is filled with zeros), run
through the model, and the context is cropped away before the tile is stored. The network therefore never sees an artificial
edge in the part that is kept. Every tile has the same size, so the network's window requirement is met.

## Real use

| Step | Detail |
|---|---|
| Bands | `get_model("swinir-m", pretrained=True)` expects three bands (`in_channels=3`). Pick red, green, blue (for Sentinel-2: B04, B03, B02) |
| Scaling | The weights were trained on images scaled 0–1. Scale reflectance to 0–1 before, and keep the same scale after |
| Several bands | Run three bands at a time and stack the results; or train weights for your band count |
| Memory | `TILE = 48` with `PAD = 8` is small and safe on a CPU. On a GPU use bigger tiles, a multiple of 8 |
| Check | Compare a crop of the result with the original at the same place; do not publish areas measured on it |

Write the result with a note that it is enhanced imagery, for example in a GeoTIFF tag:
`dst.update_tags(PYGEOVISION_NOTE="4x super-resolved; not for measurement")`.
