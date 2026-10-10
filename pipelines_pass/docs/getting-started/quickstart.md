# Quick Start (5 Minutes)

## 1. Install

```bash
pip install pygeovision[torch,models]
```

## 2. Run a real pipeline

Every one of `pygeovision`'s 25 pipelines follows the same shape:
give it a bounding box and a date, get back a result with real
statistics. 19 run out of the box; 6 need a model you trained
(`--checkpoint`).

```bash
pygeovision channel water_bodies \
    --bbox -0.15 51.47 -0.10 51.52 \
    --date 2024-06 \
    --output ./output/
```

Run `pygeovision channel list` to see every pipeline. All of them are
also available from Python: `client.pipeline("wildfire_severity", bbox=...,
date_before="2018-10", date_after="2018-12")`. See
[AI Task Pipelines](../pipelines/index.md).

## 3. Verify

The CLI prints real stats directly to the terminal:

```
✓ channel complete!
  Output: ./output/water_bodies_result.tif
  Stats:
    water_area_ha: 42.1800
    water_pct: 0.0731
```

```{note}
The CLI does not write a separate stats.json file -- `result.stats` is
printed directly. If you want the real stats as structured data (to
save, parse, or feed into another step), use the Python API below,
which returns the real `PipelineResult` object directly.
```

A real, successful run reports `success=True` and real statistics
specific to that pipeline. A pipeline that couldn't find real, usable
imagery reports `success=False` with a clear `error` — it does not
silently return an empty or fabricated result. See
[AI Task Pipelines](../pipelines/index.md) for the full list and what
each one actually computes.

## The same thing in Python

```python
import pygeovision as pgv
from pygeovision.ai.pipelines import WaterBodiesPipeline

client = pgv.PyGeoVision()

result = WaterBodiesPipeline(client).run(
    bbox=(-0.15, 51.47, -0.10, 51.52),
    output_dir="./output/",
    date="2024-06",
)

print(f"Success: {result.success}")
print(f"Output:  {result.output_path}")
print(f"Stats:   {result.stats}")
```

```{note}
`client.pipeline("building_footprints", bbox=..., date=...)` is a real,
correct shortcut for running a named pipeline through the client
directly (it delegates to the same real `get_pipeline()` +
`PipelineResult` machinery as the explicit class-based form above) --
an equally real alternative to importing the pipeline class directly.
```

## 4. Train or finetune a real model

```python
from pygeovision.ai.training.trainer import GeoTrainer, TrainingConfig
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="resnet50", in_channels=4, classes=2)

cfg = TrainingConfig(
    output_dir="./checkpoints",
    max_epochs=50,
    freeze_backbone=True,               # real: actually freezes encoder params now
    checkpoint_path="prior_run.pth",    # real: finetune from an existing checkpoint
)

trainer = GeoTrainer(model, cfg)
results = trainer.fit(train_dataset, val_dataset)
```

See [Training & Finetuning](../training/index.md) for the real, current status
here — including a second, genuinely separate `GeoTrainer` (the one
`pygeovision ai train` actually uses) that had the identical scheduler
and `freeze_backbone` bugs, found and fixed independently in a later
audit pass.

## What's next

- [AI Task Pipelines](../pipelines/index.md) — all 25 pipelines, which ones need your trained weights, and the 24 that were removed
- [Model Registry](../core-features/model-registry.md) — how model loading, verification, and fallback actually work
- [Full CLI Reference](../reference/cli.md) — every real command
