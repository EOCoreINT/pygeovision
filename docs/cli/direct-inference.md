# Direct Native Inference (`ai segment` / `ai detect` / `ai classify` / `ai change`)

A genuinely different interface from `pygeovision channel <pipeline>`:
these commands run inference directly on a file you already have on
disk — no search, no download, no AOI coverage checking. If you
already have a GeoTIFF and just want a real model run on it, this is
the faster path.

## `ai segment`

```bash
pygeovision ai segment buildings --input sentinel2.tif --output buildings.tif
pygeovision ai segment water --input s2.tif --output water.tif
pygeovision ai segment sam --input scene.tif --output masks.tif
pygeovision ai segment custom --input scene.tif --output pred.tif \
    --model unet_resnet50 --checkpoint best.pth --num-classes 5
```

- **`buildings`**, **`sam`** — real delegation to `SAMAutoLabeler`
  (this codebase's own from-scratch SAM wrapper).
- **`water`** — a real, direct NDWI computation (not a model at all):
  `(Green - NIR) / (Green + NIR)`, thresholded at 0 by default, with a
  real, sensible band-index fallback for scenes with fewer than 4
  bands.
- **`custom`** — runs *your* trained model: a registry name from either
  [registry](../models/registry.md) plus `--checkpoint` (required) and the
  `--num-classes` it was trained for. The checkpoint is loaded strictly, so one trained for a different
  architecture, class count or band count fails with a clear message instead of loading nothing; a model
  that cannot be built is never replaced by another one.

```{warning}
A real bug was found and fixed in `custom` while writing this page:
the model name string was previously passed straight to the tiled
inference engine, which requires an actual, built model object — this
failed with a confusing "band count"/"str object is not callable"
error rather than working, for both `ai segment custom` and
`ai detect custom`. Confirmed by direct testing before the fix and
after. Now resolves a real string name via `ModelHub.load()`
automatically.
```

## `ai detect`

```bash
pygeovision ai detect generic --input aerial.tif --output detections.tif
pygeovision ai detect ships --input port.tif --output ships.tif
pygeovision ai detect cars --input parking_lot.tif --output cars.tif
pygeovision ai detect custom --input scene.tif --output out.tif \
    --model unet_resnet50 --checkpoint best.pth --num-classes 2
```

`ships`/`cars` use a real, dedicated `GeoYOLO` wrapper with a real
COCO-class filter restricting detections to the relevant category —
not a full 80-class generic detector run and filtered after the fact.

## `ai classify`

```bash
pygeovision ai classify scene --input tile.tif --classes "forest,water,urban,agriculture"
pygeovision ai classify land-cover --input s2.tif --output lc.tif
```

`scene` mode is real CLIP zero-shot classification — you supply the
real candidate class names at call time, no fixed taxonomy or
retraining needed. `land-cover` mode delegates to the same real ESA
WorldCover pull documented under
[`land_cover`](../pipelines/catalog/land_cover.md).

## `ai change`

```bash
pygeovision ai change --before 2020.tif --after 2024.tif --output change.tif
pygeovision ai change --before 2020.tif --after 2024.tif --output change.tif \
    --method changeformer --checkpoint cf.pth          # also: bit, dsamnet
```

The default, `spectral_diff`, is change vector analysis: the length of the per-pixel band-difference
vector, thresholded with Otsu's method (`--min-magnitude` sets an absolute floor so a scene that did not
change reports ~0 % rather than an arbitrary split). Pixels that are nodata in either raster are written as
255 and excluded from the reported `change_fraction`. It needs no weights.
`changeformer`, `bit` and `dsamnet` run a siamese network and **require `--checkpoint`**; there is no
fallback to the spectral method (an earlier version silently fell back and, in that mode, always flagged
exactly 10 % of the pixels). Same idea as [`change_detection`](../pipelines/catalog/change_detection.md), but operating on two
files you already have rather than two dates `pygeovision` searches and downloads for you.

## One tiled-inference engine for every command

`infer predict`, `infer batch`, `ai infer` and `ai segment|detect custom` all run
`pygeovision.inference.tiled.TiledInference` (the other class,
`pygeovision.ai.inference.tiled_inference.TiledInference`, is still importable from Python but no command
uses it). All of them require `--checkpoint`, check the band count before running and exit non-zero if
the run fails. See [Tiled Inference](../inference/index.md).
