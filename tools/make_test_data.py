#!/usr/bin/env python
"""Synthetic test data for ``CLI_TESTING_GUIDE.md`` -- no network needed.

    python tools/make_test_data.py [OUTPUT_DIR]        # default: ./pgv_test

Scene: 256 x 256 px, 10 m, UTM 33N, 6 bands in the order the CLI defaults expect
(1 blue, 2 green, 3 red, 4 nir, 5 swir1, 6 swir2), reflectance 0-1. Five land-cover classes (water, forest,
crop, urban, bare) with known positions, so every result can be checked against the truth:

    scene_t1.tif / scene_t2.tif   two dates; t2 has a 60x60 px forest->bare change at rows 40-100, cols 120-180
    scene_rgb.tif / .png          3-band 8-bit view of t1 (detect / segment / VLM / Grad-CAM)
    labels.tif                    ground-truth class map (uint8, nodata 255)
    reference.tif                 truth with ~8 % of pixels flipped (so accuracy comes out < 100 %)
    classified.tif                a "prediction" equal to the truth
    scene_bands/B02..B12.tif      the six bands of t1 as separate Sentinel-2-style files, plus SCL.tif (a cloud patch)
    series/s2_<date>.tif          five dates with a seasonal NDVI cycle
    ref_chips/ cur_chips/         128 px chips; the current ones are brightened so drift detection fires
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import PIL.Image
import rasterio
from rasterio.transform import from_origin

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "pgv_test")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "series").mkdir(exist_ok=True)
rng = np.random.default_rng(42)
H = W = 256
TF = from_origin(500000, 5000000, 10, 10)
SPECTRA = {  # blue, green, red, nir, swir1, swir2
    0: (.05, .07, .05, .03, .02, .01),    # water
    1: (.03, .06, .04, .35, .18, .08),    # forest
    2: (.05, .09, .07, .30, .25, .15),    # crop
    3: (.15, .17, .19, .22, .28, .25),    # urban
    4: (.12, .16, .20, .25, .32, .30),    # bare
}

def classes() -> np.ndarray:
    c = np.full((H, W), 2, np.uint8)       # crop background
    c[:, :60] = 0                          # river / lake on the left
    c[20:120, 100:200] = 1                 # forest block
    c[150:230, 120:220] = 3                # town
    c[:40, 200:] = 4                       # bare ground
    return c

def render(c: np.ndarray, noise=0.01, season=1.0) -> np.ndarray:
    out = np.zeros((6, H, W), np.float32)
    for k, spec in SPECTRA.items():
        m = c == k
        for b in range(6):
            v = spec[b] * (season if (k in (1, 2) and b == 3) else 1.0)   # NIR of vegetation follows the season
            out[b][m] = v
    return np.clip(out + rng.normal(0, noise, out.shape), 0, 1).astype(np.float32)

def write(path, arr, dtype="float32", nodata=None, crs="EPSG:32633"):
    arr = np.asarray(arr); arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=dtype, crs=crs, transform=TF, nodata=nodata, compress="lzw") as d:
        d.write(arr.astype(dtype))

c1 = classes()
c2 = c1.copy(); c2[40:100, 120:180] = 4       # deforestation-like change between t1 and t2
write(OUT / "scene_t1.tif", render(c1))
write(OUT / "scene_t2.tif", render(c2))
write(OUT / "labels.tif", c1, "uint8", nodata=255)
ref = c1.copy(); flip = rng.random(c1.shape) < 0.08; ref[flip] = rng.integers(0, 5, flip.sum()).astype(np.uint8)
write(OUT / "reference.tif", ref, "uint8", nodata=255)                  # ~7% label noise -> accuracy < 100%
write(OUT / "classified.tif", c1, "uint8", nodata=255)

# per-band files + a scene-classification layer, as a Sentinel-2 download delivers them
(OUT / "scene_bands").mkdir(exist_ok=True)
_t1 = render(c1)
for i, name in enumerate(("B02", "B03", "B04", "B08", "B11", "B12")):
    write(OUT / "scene_bands" / f"{name}.tif", _t1[i])
scl = np.full((H, W), 4, np.uint8); scl[200:240, 20:60] = 9          # 4 = vegetation, 9 = cloud (high probability)
write(OUT / "scene_bands" / "SCL.tif", scl, "uint8", nodata=0)

# a 3-band 8-bit RGB view (for detect / segment / vlm / gradcam)
rgb = (np.stack([render(c1)[2], render(c1)[1], render(c1)[0]]) * 255 * 4).clip(0, 255).astype(np.uint8)
write(OUT / "scene_rgb.tif", rgb, "uint8")
PIL.Image.fromarray(rgb.transpose(1, 2, 0)).save(OUT / "scene_rgb.png")

# five dated scenes with a seasonal NDVI cycle (+ one cloudy/dry outlier) for the time-series commands
for i, (d, s) in enumerate([("2024-02-15", .55), ("2024-04-15", .9), ("2024-06-15", 1.3), ("2024-08-15", 1.15), ("2024-10-15", .35)]):
    write(OUT / "series" / f"s2_{d}.tif", render(c1, season=s))

# chips for drift detection / VLM search
(OUT / "ref_chips").mkdir(exist_ok=True); (OUT / "cur_chips").mkdir(exist_ok=True)
for n in range(4):
    write(OUT / "ref_chips" / f"c{n}.tif", render(c1, noise=.01)[:, :128, :128])
    write(OUT / "cur_chips" / f"c{n}.tif", np.clip(render(c1, noise=.01)[:, :128, :128] * 1.6 + .05, 0, 1))   # brightened -> drift
print("wrote", OUT.resolve(), sorted(p.name for p in OUT.iterdir()))
