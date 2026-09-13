# Change Detection

One real, generic pipeline — the building block several domain-specific
pipelines above (`infrastructure_monitoring`, `coastal_monitoring`)
call with `method="change_detection"`.

## `change_detection`

**What it does:** Generic, real bi-temporal segmentation-model change
detection between any two dates — not restricted to any specific
change type.

**How it actually works:** Runs a real Siamese encoder-decoder model
(`siamese_unet` by default, or `changeformer` — a real, published
transformer-based change-detection architecture, `changeformer-mit-b0`
/`b4`) over the aligned image pair. Both images pass through the
same, shared encoder weights (the real "Siamese" property — identical
feature extraction applied independently to each date), and the
decoder learns to highlight real pixel-level differences between the
two resulting feature maps.

```python
from pygeovision.ai.pipelines import ChangeDetectionPipeline

result = ChangeDetectionPipeline(client).run(
    bbox=(-74.1, 40.6, -73.7, 40.9),
    date_before="2020-01", date_after="2024-01",
    output_dir="./output",
    model="changeformer",
)
print(result.metadata)
# {"date_before": "2020-01", "date_after": "2024-01", "model": "changeformer",
#  "bands": ("red","green","blue"), "num_classes": 2}
```

```{note}
Corrected from an earlier version of this page: the real
`ChangeDetectionPipeline.run()` doesn't populate `result.stats` at
all — only `result.metadata`, shown above. There's no
`changed_area_ha`/`changed_pct` computed by this pipeline itself; if
you need real area statistics from the output change mask, compute
them yourself from the real, returned raster.
```

**Model choice matters here more than in most pipelines** — `siamese_unet`
is smaller and faster; `changeformer` (transformer-based, real global
attention across the image) generally gives more accurate boundaries
on larger, more complex scenes at real additional compute cost.

**Honest limitation:** a real image-alignment bug was found and fixed
in this exact code path during this project's audit — a filename
collision in the bi-temporal search step meant every "before/after"
pipeline was, at one point, silently comparing an image to itself.
This is now fixed and verified, but the alignment step remains the
single most important thing to get right for any bi-temporal pipeline:
always confirm `date_before`/`date_after` genuinely resolve to two
different, real scenes before trusting a change-detection result.
