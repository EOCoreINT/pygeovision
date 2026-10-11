# YAML workflows

The [named pipelines](index.md) answer one fixed question each. A **workflow** is different: you describe your own chain
**search → download → infer → export** in a YAML file, and PyGeoVision runs it with dependency order, retries and
overrides from the command line. Use it to apply **a network you trained** to many scenes and get vector files out, repeatably.

```{note}
Two other things have similar names. `pygeovision channel NAME` runs one of the 40 [named pipelines](catalog/index.md). `pygeovision data pipeline` runs
the lighter [data pipelines](../data/yaml-pipelines.md) (search, download, preprocess, index, post-process, no network). This page is about
`pygeovision pipeline`, the workflow runner that includes model inference.
```

## When to use it

- You trained a segmentation network and want to run it over every scene that matches a search, then export polygons.
- You want the same chain on different areas and dates, changing only the box and the dates (`--set`).
- You want a file you can keep in version control, review, and schedule.

**Do not use it** to get a result without a trained model (use a ready [named pipeline](choosing.md)), or for analysis that is not
search → download → infer → export. For anything else, write the steps in Python (below).

## The file

```yaml
name: water_pipeline
description: Water scaffold (bring your own area, dates and trained checkpoint)

steps:
  - name: search
    action: search
    params:
      bbox: [15.0, 45.1, 15.1, 45.2]
      date_range: ["2024-06-01", "2024-06-30"]
      providers: [planetary_computer, aws_earth]
      cloud_cover_max: 15
      sort_by: cloud_cover

  - name: download
    action: download
    depends_on: [search]
    retry_on_fail: 3
    params:
      output_dir: ./output/water/data/
      parallel: 4
      post_process: [reproject:EPSG:4326, cog]

  - name: infer
    action: infer
    depends_on: [download]
    params:
      model: unet_resnet50
      checkpoint: runs/water/best.pth
      num_classes: 2
      chip_size: 512
      overlap: 64
      blend_mode: gaussian
      output_dir: ./output/water/predictions/

  - name: export
    action: export
    depends_on: [infer]
    params:
      format: geojson
      target_class: 1
      min_area_m2: 1000
      output_dir: ./output/water/vectors/
```

### Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Name of the workflow. |
| `steps` | yes | At least one step. |
| `description` | no | Free text. |

### Step keys

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique within the file. Other steps refer to it in `depends_on`; `--set` uses it. |
| `action` | yes | `search`, `download`, `infer` or `export`. Anything else is rejected when the file is read, rather than running as a silent no-op. |
| `params` | no | Arguments of the action (below). |
| `depends_on` | no | Names of steps that must succeed first. Steps run in dependency order. A circular dependency is an error. |
| `retry_on_fail` | no | Attempts for a step that fails with an unexpected error (default **1**, i.e. no retry). Waits 1, 2, 4… seconds between attempts. A step that fails deterministically (for example a missing checkpoint) is never retried, because trying again cannot help. |

### The four actions

````{tab-set}
```{tab-item} search
Calls `client.search(**params)`. Parameters are those of [`client.search`](../data/search-and-download.md): `bbox` and `date_range` (both
required), `providers`, `collections`, `satellite`, `cloud_cover_max`, `max_results`, `sort_by`, `sort_order`, …

**Output:** the list of scenes. **Fails** when nothing matches: every later step needs scenes, so an empty search is an error here and not a
quiet empty run downstream.
```
```{tab-item} download
Calls `client.download(scenes, **params)` with the scenes of the search step. Parameters: `output_dir`, `parallel`, `post_process`
(for example `reproject:EPSG:4326`, `cog`), `bands`, `resume`, `overwrite`, …

**Output:** the list of downloaded scenes. **Fails** unless at least one scene arrived, and quotes the provider's errors.
```
```{tab-item} infer
Runs **your** network over each downloaded scene with tiled inference, writing `<scene>_pred.tif`.

| Parameter | Meaning |
|---|---|
| `model` | Architecture name from the [registry](../models/registry.md). Required with `checkpoint`. |
| `checkpoint` | **Your** trained `.pth`. Required. PyGeoVision ships architectures, not trained weights, so the step stops with a clear message without it. |
| `num_classes` | Classes of your network (default 2). |
| `in_channels` | Bands your network expects (default: the scene's band count). |
| `chip_size`, `overlap`, `blend_mode`, `batch_tiles`, `activation`, `tta`, `half_precision` | Tiled inference settings ([Inference](../inference/index.md)). |
| `output_dir` | Where predictions go (default: next to the scene). |

An unknown parameter is an error, not ignored. The checkpoint is loaded strictly.
```
```{tab-item} export
Vectorises each prediction raster and writes it in the format you choose.

| Parameter | Meaning |
|---|---|
| `format` | `geojson` (default), `gpkg`, `shp` or `kml`. |
| `output_dir` | Where vectors go (default `./output/`). |
| `target_class` | Polygonise only this class (default: every class). |
| `min_area_m2` | Drop polygons smaller than this many **true square metres**. |
| `simplify_tolerance` | Simplify outlines (in the raster's units: degrees for lon/lat rasters). |

**Output:** the paths of the written files. See [Postprocessing](../data/postprocessing.md) for what vectorising does and the pitfalls.
```
````

## Run it

```bash
pygeovision pipeline list-templates                      # six scaffolds: agriculture, climate, disaster, forestry, urban, water
pygeovision pipeline run water.yaml --dry-run            # check the file and step order; runs nothing

pygeovision pipeline run water.yaml \
    --set search.bbox='[15.0,45.1,15.1,45.2]' \
    --set search.date_range='["2024-06-01","2024-06-30"]' \
    --set infer.checkpoint=runs/water/best.pth
```

**Expected output**

```text
  Loading pipeline: water.yaml
  Pipeline: water_pipeline (4 steps)
  ✓ completed: ['search', 'download', 'infer', 'export'] in 412.7s
```

On failure the command prints the failed steps and the reason, and exits with status 1:

```text
  ✗ completed: ['search', 'download'] in 95.1s
  failed: ['infer', 'export']
  infer: checkpoint not found: runs/water/best.pth; export: skipped: a dependency failed
```

### `--set STEP.PARAM=VALUE`

Overrides one parameter of one step, so the file can stay generic. The value is read as JSON when it is valid JSON (`[15.0,45.1]`,
`0.5`, `true`), otherwise as text (`best.pth`). Repeat for several. A step name that does not exist is an error that lists the real step names.

### `--dry-run`

Reads the file, applies the overrides, checks the structure (required keys, actions, unknown or circular dependencies) and lists the steps in
the order they would run. **It does not check your parameters, credentials or checkpoint, and it downloads nothing.** A passing dry run means
the file is well formed, not that the run will succeed.

### The templates

`list-templates` shows six files, one per domain. They differ only in the name and output folder. **They are scaffolds, not ready models**: they have no
area, no dates and no checkpoint, which you must supply with `--set`. Copy one from `pygeovision/pipelines/templates/` and edit it.

## What happens when a step fails

- Steps run in dependency order; a step whose dependency failed is **skipped** and reported as such. The run's `success` is `False` and every
  failure is listed.
- A step that raises an unexpected error is retried up to `retry_on_fail` times; a `StepError` (the step knows it cannot succeed) is not.
- Nothing is silently skipped or substituted: an empty search, a download with no files, a missing checkpoint and a run with no
  predictions are all failures with a reason.

## In Python

```python
from pygeovision import PyGeoVision
from pygeovision.pipelines import Pipeline

client = PyGeoVision()
workflow = Pipeline.from_yaml("water.yaml")                 # validates the file
result = workflow.run({"client": client})                  # dry_run=True to only order the steps

print(result.success, result.steps_completed, result.steps_failed)
print(result.error)                                        # "step: reason; step: reason" or None
files = result.outputs["export"]                           # each step's output is stored under its name
```

`result` has `pipeline_name`, `success`, `steps_completed`, `steps_failed`, `duration_s`, `outputs` (the context, with every step's output under
its step name and the shortcuts `search_results`, `downloads`, `predictions`) and `error`.

Overrides in Python:

```python
from pygeovision.pipelines.yaml_parser import PipelineYAMLParser

parser = PipelineYAMLParser()
config = parser.apply_overrides(parser.load("water.yaml"),
                                ["search.bbox=[15.0,45.1,15.1,45.2]", "infer.checkpoint=best.pth"])
Pipeline.from_config(config).run({"client": client})
```

### Your own steps

YAML supports the four actions. For anything else, give a `Step` a function. It receives the context and the step's `params`, and its return value is stored
under the step name:

```python
from pygeovision.pipelines import Pipeline
from pygeovision.pipelines.steps import SearchStep, Step

def keep_clearest(context, top=3):
    scenes = sorted(context["search_results"], key=lambda s: s.cloud_cover or 100)
    return scenes[:top]

workflow = Pipeline("clearest_scenes")
workflow.add(SearchStep(name="search", params={"bbox": (-0.15, 51.47, -0.10, 51.52),
                                               "date_range": ("2024-06-01", "2024-06-30")}))
workflow.add(Step(name="clearest", action="custom", params={"top": 3}, depends_on=["search"], _fn=keep_clearest))
result = workflow.run({"client": client})
print([s.id for s in result.outputs["clearest"]])
```

### Hooks

```python
workflow.on("before_step", lambda step, ctx: print("starting", step.name))
workflow.on("after_step", lambda step, ctx: print("finished", step.name))
workflow.on("on_error", lambda step, ctx, exc: print("failed", step.name, exc))
```

### Several workflows

```python
from pygeovision.pipelines.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(client=client)
orchestrator.register_yaml("water.yaml").register_yaml("urban.yaml")
orchestrator.list()                                         # ['water_pipeline', 'urban_pipeline']
orchestrator.run("water_pipeline", dry_run=True)
```

## Scheduling

Two options, depending on whether the schedule must survive a restart.

**The operating system's cron** (recommended for anything that matters):

```bash
# crontab -e   — every Monday at 06:00, with the environment activated
0 6 * * 1  cd /data/monitoring && /home/me/.venv/bin/pygeovision pipeline run water.yaml \
             --set search.date_range='["2024-06-01","2024-06-30"]' >> run.log 2>&1
```

**Inside a Python process** with `PipelineScheduler`:

```python
from pygeovision.pipelines.scheduler import PipelineScheduler

scheduler = PipelineScheduler()
scheduler.add_job("water", fn=lambda: workflow.run({"client": client}), cron="0 6 * * 1")   # Mondays 06:00 UTC
scheduler.start()          # background thread; stops when your program exits
scheduler.status()         # [{"name": "water", "cron": "0 6 * * 1", "n_runs": 0, "errors": 0, "last_run": None}]
scheduler.stop()
```

Times are **UTC**; an invalid expression is rejected when you add the job; a job runs at most once per matching minute; and a job that raises is
logged and counted in `errors` without stopping the others. The expression has five fields (minute hour day-of-month month day-of-week) and
supports `*`, lists (`1,15`), ranges (`9-17`) and steps (`*/15`).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `Error: override 'x.y=1': no step named 'x'` | Wrong step name in `--set` | The message lists the steps; use one of them |
| `Step '…' has action '…'; YAML pipelines support […]` | An action YAML cannot express | Use `search`, `download`, `infer`, `export`, or add a Python `Step` |
| `Steps depend on each other in a circle: a -> b -> a` | A `depends_on` loop | Break the loop |
| `Step '…' depends on unknown step '…'` | A misspelt name | Fix `depends_on` |
| `search parameters are incomplete` | No `bbox` or `date_range` | Put them in the file or pass `--set search.bbox=…` / `--set search.date_range=…` |
| `search returned no scenes` | Nothing matches | Widen the dates, raise `cloud_cover_max`, check the box order (longitude first) |
| `infer needs your trained model` | No `model`/`checkpoint` | `--set infer.checkpoint=best.pth` and a `model` name from `pygeovision models list` |
| `checkpoint not found` | Wrong path | Give the path relative to where you run the command |
| `infer got unknown parameter(s) […]` | A typo in `infer` params | Use the parameters in the table above |
| `infer has no scenes: it must depend on a successful download step` | Missing `depends_on` | Add `depends_on: [download]` |
| A dry run passes but the real run fails | Dry runs do not test parameters | Run on a small box first |
| Strict-load error (`size mismatch`) | Checkpoint trained with another architecture or class count | Set `model` and `num_classes` to match training |

## Related

- [Training](../training/index.md) to produce the checkpoint, and [Inference](../inference/index.md) for the tiling options.
- [Postprocessing](../data/postprocessing.md) for what `export` does to the raster.
- [Data pipelines](../data/yaml-pipelines.md) for YAML without a model.
- [Reference: `pygeovision pipeline`](../reference/cli/pipeline.md).
