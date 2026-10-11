# Full CLI Reference

## Global options

```bash
pygeovision --version
pygeovision --help
```

## All command groups

| Group | Real commands |
|---|---|
| `channel` | Run any of the 38 pipelines by name (`channel list` shows each one and whether it needs your trained weights) |
| `ai train` | Build a real `TrainingConfig` and print the real Python to actually train (does not train itself — see [Training](../training/index.md)) |
| `models` | `list`, `zoo list`, `zoo search` — browse both real registries and the real, metadata-only catalog |
| `infer` | `predict`, `batch` — real tiled inference via `pygeovision.inference.tiled` |
| `label` | `osm`, `quality` — real, live entry points into the labeling module |
| `preprocess` | `stack`, `clip`, `normalise`, `resample`, `pipeline` — real, composable raster preparation primitives |
| `explain` | `gradcam` — real Grad-CAM explainability |
| `monitor` | drift detection via a real, published PSI (Population Stability Index) formula |
| `ai-models` | `list`, `info`, `cache` — browse the legacy AI registry and manage cached model weights |

```{note}
`preprocess stack` intentionally does **not** apply radiometric
scaling — it's a real, honest, low-level "combine files" primitive.
`preprocess normalise --method scale_factor` is the real, separate,
documented step for Sentinel-2 DN-to-reflectance conversion.
```

## Example: a real pipeline run

```bash
pygeovision channel water_bodies \
    --bbox -0.15 51.47 -0.10 51.52 \
    --date 2024-06 \
    --output ./output/
```

Every pipeline is reachable from `channel`. Dates are `--date`, or
`--date-before` / `--date-after` for two-date pipelines; `--checkpoint` supplies
your trained weights for the seven pipelines that need them (and for the network mode of `building_footprints`); any other argument is
passed as `--param KEY=VALUE`. A removed or unknown name exits with code 2 and
prints the reason. See [AI Task Pipelines](../pipelines/index.md).

```bash
pygeovision channel list
pygeovision channel building_footprints --bbox -0.15 51.47 -0.10 51.52 \
    --date 2024-06 --checkpoint runs/buildings/checkpoints/best.pth
```

## Example: real tiled inference

```bash
pygeovision infer predict scene.tif \
    --model unet_resnet50 --checkpoint best.pth \
    --classes 2 \
    --chip-size 512
```

`--checkpoint` is required: the registry holds architectures, not trained weights. It is loaded strictly
and no other model is ever substituted. Every command exits non-zero when it fails, so the commands can be
used in scripts and CI.

```{seealso}
[Direct Native Inference](../cli/direct-inference.md) for
`ai segment`/`ai detect`/`ai classify`/`ai change` — a genuinely
different, faster interface for files you already have on disk.
[Data & Authentication](../cli/data-and-auth.md) for the real
`pygeofetch`-delegated commands.
```

```{seealso}
[CLI testing guide](../../CLI_TESTING_GUIDE.md): a runnable example for every command, what to expect, and
which ones need network access, credentials or trained weights.
```
