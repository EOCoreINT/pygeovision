# Full CLI Reference

## Global options

```bash
pygeovision --version
pygeovision --help
```

## All command groups

| Group | Real commands |
|---|---|
| `run` | Run any of the 49 real pipelines by name |
| `ai train` | Build a real `TrainingConfig` and print the real Python to actually train (does not train itself — see [Training](../training/index.md)) |
| `models` | `list`, `zoo list`, `zoo search` — browse both real registries and the real, metadata-only catalog |
| `infer` | `predict`, `batch` — real tiled inference via `pygeovision.inference.tiled` |
| `label` | `osm`, `quality` — real, live entry points into the labeling module |
| `preprocess` | `stack`, `clip`, `normalise`, `resample`, `pipeline` — real, composable raster preparation primitives |
| `explain` | `gradcam` — real Grad-CAM explainability |
| `monitor` | drift detection via a real, published PSI (Population Stability Index) formula |
| `channel` | Manage local, cached model weights |

```{note}
`preprocess stack` intentionally does **not** apply radiometric
scaling — it's a real, honest, low-level "combine files" primitive.
`preprocess normalise --method scale_factor` is the real, separate,
documented step for Sentinel-2 DN-to-reflectance conversion.
```

## Example: a real pipeline run

```bash
pygeovision run water_bodies \
    --bbox -0.15,51.47,-0.10,51.52 \
    --date 2024-06 \
    --output ./output/
```

## Example: real tiled inference

```bash
pygeovision infer predict scene.tif \
    --model unet_resnet50 \
    --classes 2 \
    --chip-size 512
```

```{seealso}
[Direct Native Inference](../cli/direct-inference.md) for
`ai segment`/`ai detect`/`ai classify`/`ai change` — a genuinely
different, faster interface for files you already have on disk.
[Data & Authentication](../cli/data-and-auth.md) for the real
`pygeofetch`-delegated commands.
```
