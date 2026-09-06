# Python API Reference

## Full workflow example

```python
import pygeovision as pgv
from pygeovision.ai.pipelines.domains import get_pipeline

client = pgv.PyGeoVision()
pipeline = get_pipeline("wildfire_severity", client)

result = pipeline.run(
    bbox=(-120.5, 38.5, -120.0, 39.0),
    output_dir="./output",
    date_before="2023-06",
    date_after="2023-09",
)

if result.success:
    print(result.stats)
else:
    print(f"Failed: {result.error}")
```

## `PipelineResult`

Every real pipeline returns the same real result shape:

```python
@dataclass
class PipelineResult:
    name: str
    success: bool
    output_path: Path | None = None
    stats: dict = field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0
```

`success=False` means exactly that — a real check failed (no usable
imagery found, zero real features detected, invalid input). It is
never `True` with a fabricated or placeholder result underneath.

## `ModelHub`

```python
from pygeovision.ai.models.hub import ModelHub

hub = ModelHub()
model = hub.load("unet_resnet50", num_classes=2, in_channels=4)
```

See [Model Registry](../core-features/model-registry.md) for the real
fallback behavior on build failure.

## `GeoTrainer` / `TrainingConfig`

See [Training](../training/index.md) for the real, current fields and
what each one actually does.

```{note}
`client.pipeline(pipeline_name, bbox=..., **kwargs)` is a real,
correct shortcut for running a named pipeline directly through the
client -- confirmed by reading the real source, it delegates to the
same `get_pipeline()` + `.run()` machinery shown above. Not to be
confused with `client.create_pipeline(...)`, which builds a real,
separate `pygeofetch` YAML *data* pipeline (search → filter → download
→ postprocess), a genuinely different concept.
```
