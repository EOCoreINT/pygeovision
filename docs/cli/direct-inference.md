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
pygeovision ai segment custom --input scene.tif --output pred.tif --model unet_resnet50
```

- **`buildings`**, **`sam`** — real delegation to `SAMAutoLabeler`
  (this codebase's own from-scratch SAM wrapper).
- **`water`** — a real, direct NDWI computation (not a model at all):
  `(Green - NIR) / (Green + NIR)`, thresholded at 0 by default, with a
  real, sensible band-index fallback for scenes with fewer than 4
  bands.
- **`custom`** — runs any real model from either
  [registry](../core-features/model-registry.md) by name.

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
pygeovision ai detect custom --input scene.tif --output out.tif --model rf-detr-b
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
[`land_cover`](../pipelines/urban.md).

## `ai change`

```bash
pygeovision ai change --before 2020.tif --after 2024.tif --output change.tif --method changeformer
```

Real, direct access to the same ChangeFormer/spectral-diff choice
documented under [Change Detection](../pipelines/change-detection.md)
— operating on two files you already have, not two dates `pygeovision`
searches and downloads for you.

## Three real, genuinely different tiled-inference code paths

If you're comparing this section to
[Tiled Inference](../inference/tiled-inference.md), it's worth being
explicit that there are three, not two:

| Command | Real engine |
|---|---|
| `pygeovision infer predict`/`infer batch` | `pygeovision.inference.tiled.TiledInference` |
| `pygeovision ai infer` | `pygeovision.ai.inference.tiled_inference.TiledInference` (a different class, different method name — `.run()` not `.infer()`) |
| `ai segment custom`/`ai detect custom` (`client.segmentation.custom`/`client.detection.custom`) | Also `pygeovision.inference.tiled.TiledInference` |

All three now correctly validate band counts before running (the
real, confirmed fix documented in
[Tiled Inference](../inference/tiled-inference.md) applies to both
underlying engine classes), but they are genuinely separate code
paths, not three names for the same function.
