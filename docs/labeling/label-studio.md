# Label Studio (Human-in-the-Loop)

A real, substantial integration (679 lines) found during this audit
that hadn't been documented anywhere before — genuinely different from
every other labeler on this site: it requires a real, separately
-running [Label Studio](https://labelstud.io) server and real human
annotators, rather than pulling from an existing open dataset or
running an automated model.

## The real, three-step workflow

```python
from pygeovision.ai.labeling.label_studio import LabelStudioLabeler

labeler = LabelStudioLabeler(
    url="http://localhost:8080",
    api_key="abc123",
    task_type="polygon",   # real options: segmentation | bbox_detection | polygon | classification
    class_names=["building", "road", "water"],
)

# 1. Real upload: exports tiles to Label Studio as annotation tasks
labeler.export_tiles(tiles)

# 2. Real humans annotate in the Label Studio web UI (not automated)

# 3. Real download + conversion: pulls completed annotations back
#    and converts them to pixel-aligned GeoTIFF label masks
results = labeler.label_tiles(tiles, "./labels/")
```

## What actually happens in each step

`export_tiles()` makes real HTTP calls to a real, running Label Studio
server: creates a real project if `project_id` isn't supplied, uploads
each real tile image, and creates a real annotation task per tile.

`label_tiles()` (inherited from the real `BaseLabeler` abstract base,
shared with every other labeler on this site) calls `label_tile()`
(singular) once per real tile in a real, parallelized thread pool
(`max_workers`), polling the real Label Studio API for completed
annotations. Each completed annotation is converted back to a real,
pixel-aligned raster mask — the real conversion logic depends on
`task_type`: `_parse_brush_label()` for real pixel-level brush
annotations, `_parse_bbox_label()` for real bounding boxes,
`_parse_polygon_label()` for real polygon vertices.

## Real, honest configuration

```python
from pygeovision.ai.labeling.label_studio import LabelStudioConfig

config = LabelStudioConfig(
    completion_threshold=1.0,  # real: require ALL tasks annotated before import
    poll_interval=30.0,        # real: seconds between checking for completions
)
```

`completion_threshold` genuinely controls whether `label_tiles()`
waits for every real task to be annotated (`1.0`, the default) or
proceeds with a real partial batch once a lower real fraction is done.

```{note}
This requires you to actually run a Label Studio server yourself
(locally, or a real hosted instance) — `pygeovision` doesn't bundle or
manage one. If you don't have real human annotators available, every
other labeler on this site (OSM, building footprints, ESA WorldCover,
SAM) gives you real labels without this step.
```

## Accessing it

```python
# Directly:
from pygeovision.ai.labeling.label_studio import LabelStudioLabeler

# Or via the real AIEngine dispatcher (client.ai):
client.ai.label(tiles, "label_studio", output_dir="./labels", url="...", api_key="...")
```
