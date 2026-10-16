# Write your own pipeline

The built-in pipelines are Python classes. You can write one of your own, run it exactly like the built-in ones, and get
the same result object, `run_summary.json`, maps, vectors and reports. This page shows the smallest complete pipeline, how
to register it so that `client.pipeline("name", ...)` finds it, and the rules a pipeline should follow.

```{admonition} What this page was verified with
:class: note
The example below was run: it writes a raster, returns a successful `PipelineResult`, appears in `list_pipelines()`, fails
cleanly on a bad argument, and leaves the usual deliverables behind. It needs no network. A real pipeline would also search
and download imagery (see [the common helpers](#helpers-you-can-use)).
```

## The smallest pipeline

```python
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from pygeovision import PyGeoVision
from pygeovision.ai import pipelines
from pygeovision.ai.pipelines import BasePipeline


class EmptyGrid(BasePipeline):
    """A pipeline that needs no download: writes a constant raster over the bounding box."""

    name = "empty_grid"                      # the name you pass to client.pipeline(...)
    description = "Write a constant raster over the bounding box (demonstration of a custom pipeline)"
    domain = "other"                         # a topic area, used to group the documentation
    kind = "ready"                           # "ready", or "needs_weights" if it requires checkpoint=
    tags = ["demo"]

    def run(self, bbox, output_dir, value=1, size=64, **kwargs):
        import time
        t0 = time.time()
        if size < 1:
            return self._fail("size must be at least 1", t0)            # a clear message, not an exception
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "grid.tif"
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=from_bounds(*bbox, size, size)) as dst:
            dst.write(np.full((1, size, size), value, dtype="uint8"))
        return self._ok(path, {"value": value, "pixels": size * size}, {"bbox": list(bbox)}, t0)


pipelines._CORE_PIPELINES["empty_grid"] = EmptyGrid          # register it for this Python process

client = PyGeoVision()
result = client.pipeline("empty_grid", bbox=(-0.15, 51.47, -0.10, 51.52), output_dir="./out", value=7)
print(result.success, result.output_path, result.stats)
```

```text
True out/grid.tif {'value': 7, 'pixels': 4096}
```

```python
print("empty_grid" in pipelines.list_pipelines())
bad = client.pipeline("empty_grid", bbox=(-0.15, 51.47, -0.10, 51.52), output_dir="./out", size=0)
print(bad.success, bad.error)
```

```text
True
False size must be at least 1
```

The run left these files in `out/`, the same set a built-in pipeline writes:

```text
empty_grid.gpkg  empty_grid_aoi.geojson  empty_grid_chart.png  empty_grid_map.png  empty_grid_output.geojson
empty_grid_report.docx  empty_grid_report.pdf  empty_grid_report.xlsx  grid.tif  run_summary.json
```

## The rules of a pipeline

| Rule | Why |
|---|---|
| Subclass `BasePipeline` and set `name`, `description`, `domain`, `kind` and `tags` | They drive listing, grouping and the checks before a run |
| `run(self, bbox, output_dir, **named_arguments)` returns a `PipelineResult` | The client, the CLI and the reports all read it |
| Return `self._ok(output_path, stats, metadata, t0)` on success | It records the smallest input coverage and the timing |
| Return `self._fail(message, t0)` on a problem the user can fix; do not raise | A message is more useful than a traceback; the caller checks `result.success` |
| Declare every argument in the signature, with a default | The documentation, the argument check and `run_summary.json` all read the signature |
| Treat cloud and missing data as **no data (255 or NaN)**, never as a class | [Core concepts](../getting-started/concepts.md) |
| If you need weights, set `kind = "needs_weights"` and check the checkpoint **before** downloading anything | A pipeline that downloads gigabytes and then cannot run wastes the user's time |
| Write only what you can explain | Every number in the report should trace to a step in the code |

## Helpers you can use

`BasePipeline` has the helpers the built-in pipelines use. Read their docstrings in
`pygeovision/ai/pipelines/__init__.py`:

| Helper | What it does |
|---|---|
| `self._search_and_download(bbox, date, output_dir, ...)` | Finds and downloads one cloud-masked image for a month |
| `self._search_and_download_pair(...)` | The same for two dates, on the same grid |
| `self._template_raster(bbox, path)` | A grid covering the box, for reference products |
| `self._ok(...)`, `self._fail(...)` | Build the result |

## Subclass an existing pipeline

To change only a default or add a step after a built-in pipeline, subclass it instead:

```python
Base = pipelines.pipeline_catalogue()["flood_mapping"]


class StrictFloodMapping(Base):
    name = "strict_flood_mapping"

    def run(self, bbox, output_dir, **kwargs):
        kwargs.setdefault("mndwi_threshold", 0.15)
        return super().run(bbox, output_dir, **kwargs)
```

Checked on the offline test scene: the original reported a threshold of 0.0 and the subclass 0.15, and the subclass found
a different flooded area. Every pipeline page has a *Modify it* section with a version of this written for that pipeline.

## Using it from the command line

`pygeovision channel NAME` knows only the built-in pipelines, because the CLI is a separate process and your registration
lives in your Python process. To run your pipeline from a shell, put the registration in a small script and run that:

```bash
python run_my_pipeline.py
```

Or wire a sequence of steps, including your own functions, in a [YAML workflow](yaml-workflows.md).

## Contributing a pipeline to PyGeoVision

A pipeline that belongs in the package also needs: a report specification (`pygeovision/reporting/specs.py`), a documentation
entry (`docs/_data/pipeline_notes.yaml` and `docs/_data/pipeline_modify.yaml`), an offline test with a known answer, and an
entry in the smoke harness. See [Contributing](../project/contributing.md).
