# Texture features (GLCM)

Two fields can have the same average brightness and still be obviously different on the ground: one smooth (bare
soil, calm water, a maize canopy), one rough (forest, a built-up area, a rough sea). **Texture** measures that
roughness. It is a classic way to separate classes that spectral values alone confuse — urban from bare soil, forest from
plantation, ships from waves — and it is the main input for radar classification.

PyGeoVision computes the **grey-level co-occurrence matrix (GLCM)** of Haralick, Shanmugam and Dinstein (1973) — the
real matrix of how often one grey level sits next to another — and derives the standard Haralick measures from it. It
is not a faster "local variance" shortcut under a GLCM label.

## Features

| Feature | Reads as | High where |
|---|---|---|
| `contrast` | Local intensity differences, weighted by how far apart the levels are | Edges, mixed surfaces, buildings |
| `dissimilarity` | Like contrast but weighted linearly (less dominated by extremes) | Same as contrast; more robust |
| `homogeneity` | Closeness of the matrix to its diagonal | Smooth, uniform surfaces |
| `energy` | Square root of the sum of squared matrix entries | Very regular, repetitive patterns |
| `ASM` | Angular second moment (energy squared) | Same as energy |
| `correlation` | Linear dependence between neighbouring grey levels | Directional structure |

`homogeneity` and `contrast` move in opposite directions and `energy` and `ASM` carry the same information, so two or
three features (`contrast`, `homogeneity`, `energy` or `correlation`) are usually enough.

## How to use it

```python
from pygeovision.data.texture import GLCMTexture

result = GLCMTexture().compute(
    "nir_band.tif",                 # use ONE band: GLCM is defined per band
    output_path="nir_texture.tif",
    band=1,
    window=7,                       # 7×7 pixel windows
    mode="block",                   # "block" (fast) or "sliding" (per pixel)
    levels=32,                      # grey levels to quantise to
    features=["contrast", "homogeneity", "energy"],
)
print(result["output_path"], result["features"], result["shape"])
```

From an array, without files:

```python
from pygeovision.data.texture import compute_glcm_texture

feats = compute_glcm_texture(band_array, window=9, mode="sliding", features=["contrast"])
feats["contrast"].shape        # same as band_array in "sliding" mode
```

Texture for **objects** (segments) rather than windows is part of the OBIA feature table:
`pygeovision obia features … --texture nir` ([OBIA](../obia/index.md)).

### Parameters

| Parameter | Default | Meaning and how to choose |
|---|---|---|
| `band` | `1` | 1-based band in the file. Radar: use dB values of VV or VH; optical: NIR or the brightest informative band |
| `window` | `7` | Window size in pixels. It should be a few times the size of the texture element you want to describe — larger windows describe larger patterns but blur boundaries |
| `mode` | `"block"` | `block`: one matrix per non-overlapping window, output is `window` times coarser. `sliding`: one per pixel, native resolution, **very** slow (one matrix per pixel) |
| `levels` | `32` | Grey levels after quantisation. 16–64 is standard; 256 gives sparse, noisy matrices |
| `distances` | `[1]` | Pixel-pair distance(s) |
| `angles` | 0°, 45°, 90°, 135° | Directions. Results are averaged over all (distances × angles), giving a direction-independent measure |
| `features` | all six | Subset of the table above |

## Expected output

A multi-band float32 GeoTIFF, **one band per feature** with the feature name as band description, and tags
`method=GLCM (Haralick 1973)`, `window`, `mode`, `levels`. In `block` mode the raster is correctly georeferenced at the
coarser resolution: an input at 10 m with `window=8` gives an 80 m raster.

On a synthetic 64×64 scene with a smooth left half and a noisy right half (`window=8`), the mean values were:

| Half | contrast | homogeneity | energy |
|---|---|---|---|
| smooth | 0.48 | 0.77 | 0.55 |
| noisy | 48.7 | 0.16 | 0.12 |

— contrast about 100× higher, homogeneity and energy several times lower on the rough half, as they should be. The call
returns `{"success": True, "output_path": …, "features": […], "shape": (bands, rows, cols)}`.

## Things to know before you trust it

- **Per-image quantisation.** Grey levels are stretched from the image's own minimum to maximum, so a feature value
  is relative to *that* image. Two scenes, or two tiles with different value ranges, are **not directly comparable**.
  For comparable values across images, scale all of them with the same fixed limits first.
- **Outliers set the scale.** One bright pixel (a cloud edge, a ship) compresses everything else into a few levels. Clip
  to a percentile range first ([Preprocessing](preprocessing.md), `percentile`).
- **Nodata is not ignored.** `NaN`/invalid pixels are assigned the lowest grey level, so windows touching cloud or
  edge look artificially rough. Mask the output with your valid-data mask, and avoid windows that are mostly nodata.
- **Speckle.** Raw radar is dominated by speckle, which looks like very high texture everywhere. Despeckle first and
  use dB values ([Radar preprocessing](sar.md)).
- **Cost.** `block` mode is fast. `sliding` mode on a 1 000 × 1 000 pixel image computes a million matrices — use it
  on small areas only.
- **Needs scikit-image.** `pip install scikit-image` if you get an `ImportError`.

## What to do when…

| Symptom | Fix |
|---|---|
| `ValueError: window=… is larger than the input` | Reduce `window` or use a larger raster |
| Output has one value over the whole image | The band is constant (or all invalid): check the input band |
| Texture "lights up" along cloud edges | Mask clouds before computing, and mask the output |
| Output is too coarse to use | Lower `window` (minimum useful ≈ 5), or use `mode="sliding"` on a small area |
| Radar texture is uniformly high | Despeckle and convert to dB first |
| You want the result at the original grid | Resample the block output back with `rasterio`/`gdal_warp` (nearest), or use `sliding` |
