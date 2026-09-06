# Quick Start (5 Minutes)

## 1. Install

```bash
pip install pygeovision[torch,models]
```

## 2. Run a real pipeline

Every one of `pygeovision`'s 49 real pipelines follows the same shape:
give it a bounding box and a date, get back a real result with real
statistics.

```bash
pygeovision run water_bodies \
    --bbox -0.15,51.47,-0.10,51.52 \
    --date 2024-06 \
    --output ./output/
```

## 3. Verify

```python
import json
with open("./output/stats.json") as f:
    print(json.load(f))
```

A real, successful run reports `"success": true` and real statistics
specific to that pipeline (e.g. water-body area in hectares). A
pipeline that couldn't find real, usable imagery reports
`"success": false` with a clear reason — it does not silently return
an empty or fabricated result. See
[AI Task Pipelines](../pipelines/index.md) for the full list
and what each one actually computes.

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
from pygeovision.training.trainer import GeoTrainer, TrainingConfig
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

See [Training & Finetuning](../training/index.md) for what's real here
and the two significant bugs (a silently-ignored `freeze_backbone`,
and a default learning-rate schedule that oscillated 50x instead of
decaying once) that were found and fixed in an earlier audit of this
exact code path.

## What's next

- [AI Task Pipelines](../pipelines/index.md) — the real, current status of all 49 pipelines
- [Model Registry](../core-features/model-registry.md) — how model loading, verification, and fallback actually work
- [Full CLI Reference](../reference/cli.md) — every real command
