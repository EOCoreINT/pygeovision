# Data pipelines (YAML)

A **data pipeline** is a small recipe that says *search these providers → keep the clear scenes → download them →
convert them → copy them somewhere*, written once in a YAML file or built in Python, and then run on demand or on a
schedule. It is for **keeping a folder or bucket of imagery up to date** — for example "every Monday, fetch last week's
clear Sentinel-2 scenes over Accra, reproject them and put them in our S3 bucket".

It is not how the 40 analysis pipelines work (those are [pipelines](../pipelines/index.md) in the other sense: a
named analysis with a report). A data pipeline moves and prepares files; it does not compute flood extent or NDVI. The
two combine well: a data pipeline keeps the archive current, and analysis pipelines or your own code read from it.

```{note}
PyGeoVision has a second YAML mechanism: `pygeovision pipeline run file.yaml` runs a **workflow** whose steps are
`search`, `download`, `infer` and `export` and which *can* run a trained network
([YAML workflows](../pipelines/yaml-workflows.md)). The two are different tools: use this page's
`pygeovision data pipeline …` to maintain an archive of imagery files, and workflows to go from imagery to model output
in one file.
```

The engine behind it is **pygeofetch's pipeline runner**. PyGeoVision builds, validates, saves and launches the
YAML, and records the run history.

## The file

```yaml
name: weekly-sentinel2
description: Last week's clear Sentinel-2 over Accra
schedule: "0 6 * * 1"             # optional; Monday 06:00 UTC
steps:
  - search:
      providers: [planetary_computer]
      bbox: "-0.30,5.55,-0.10,5.70"      # min_lon,min_lat,max_lon,max_lat as one string
      date_range: last_7_days
      cloud_cover: 0-10
      max_results: 50
  - filter:
      expression: "data.cloud_cover < 5"
  - download:
      output: ./raw/
      parallel: 4
      retry: 5
      verify_checksum: true
  - process:
      actions: [reproject:EPSG:4326, cog]
  - export:
      destination: s3://my-bucket/accra/
```

A pipeline has a `name`, an optional `description` and `schedule`, and an ordered list of `steps`. Each step is a
mapping with **exactly one key** — the step type. The steps run top to bottom and share state: `filter` reads the
search results, `download` reads the filtered (or searched) items, `process` and `export` read the downloaded files.
**The pipeline stops at the first step that fails.**

### Step reference

| Step | Reads | Options the runner uses |
|---|---|---|
| `search` | — | `providers` (list), `bbox` (`"minx,miny,maxx,maxy"`), `date_range` (`last_7_days`, `last_30_days`, `last_year`) **or** `start_date` + `end_date`, `cloud_cover` (`"MIN-MAX"`), `max_results` |
| `filter` | search results | `expression` — a Python expression on `data` (one scene), e.g. `data.cloud_cover < 5` |
| `download` | filtered or searched scenes | `output` (directory), `parallel`, `retry`, `verify_checksum` |
| `process` | downloaded files | `actions` — a list or a comma-separated string of actions (below) |
| `export` | processed or downloaded files | `destination` (directory, `s3://…`, `gs://…`), `format` (label only), `notify` |

**Process actions** are written `name` or `name:value`: `unzip`, `reproject:EPSG:4326`, `cloud_mask`, `compress:lzw`,
`cog`, `clip`, `resample`, `ndvi`, `merge`. `cloud_mask` with the default method needs the `SCL` band downloaded
with the other bands.

```{important}
**What the runner does not do** (so you do not rely on it):

- `search` ignores `satellites`, `collections`, `processing_level`, `sort_by` and `cql2`. Narrow results with `filter`.
  `date_range` understands only the three keywords above; other words (for example `this_month`) become "no start
  date". Use explicit dates instead.
- `download` does not read `resume`, `bandwidth_limit` or `on_failure`.
- `export` **copies files as they are**. `format` is recorded in the summary but nothing is converted, and `compress` is
  not read. To produce a Cloud-Optimized GeoTIFF, add `cog` to a `process` step *before* the export.
- A `filter` expression that raises an error (for example a field a scene does not have) **keeps** the scene.
- There is no step that runs a neural network. See [Running a model afterwards](#running-a-model-afterwards).
```

## Build it in Python

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
pipe = (
    client.create_pipeline("weekly-sentinel2", description="Last week over Accra")
    .search(providers=["planetary_computer"], bbox=(-0.30, 5.55, -0.10, 5.70),
            date_range="last_7_days", cloud_cover="0-10")
    .filter("data.cloud_cover < 5")
    .download(output="./raw/", parallel=4, post_process=["reproject:EPSG:4326", "cog"])
    .export(destination="s3://my-bucket/accra/")
    .set_schedule("0 6 * * 1")
)

print(pipe.to_yaml())            # inspect
pipe.save("weekly-sentinel2.yaml")
assert pipe.validate()
summary = pipe.run()             # run it now, once
print(summary["success"], [s["status"] for s in summary["steps"]])
```

Each builder method returns the pipeline so calls chain. Passing `post_process=[…]` to `download()` appends a separate
`process` step, because that is where the runner performs the actions. `PipelineStep` objects are in `pipe.steps`;
`DataPipeline.from_yaml("file.yaml")` loads a saved file back.

`run()` accepts `output_dir=` to redirect every download step and `step="search"` to run only the steps of one type
(handy for checking a search before downloading gigabytes).

## Command line

```bash
pygeovision data pipeline validate weekly-sentinel2.yaml     # structure only
pygeovision data pipeline run weekly-sentinel2.yaml          # run now
pygeovision data pipeline run weekly-sentinel2.yaml --step search    # only the search step
pygeovision data pipeline schedule weekly-sentinel2.yaml --cron "0 6 * * 1"
pygeovision data pipeline list                               # scheduled pipelines
pygeovision data pipeline history --limit 20                 # recent runs, newest first
```

**Expected output** of a run:

```text
Running pipeline: weekly-sentinel2.yaml
✓ Pipeline completed successfully (142.7s).
  search: ok
  filter: ok
  download: ok
  process: ok
  export: ok
```

On failure the failing step and its error are printed and the exit status is 1. Every run made this way is appended to
`~/.pygeovision/pipeline_history.jsonl`, which `pipeline history` reads.

### What validation checks

`validate` confirms that the file parses, has a `name`, and that every step is a one-key mapping; it also requires at least one
step. It does **not** check option values, provider names or credentials — those fail when the step runs. A typo in a
step *type* (`dowload:`) is caught when the pipeline reaches that step, with "Unknown step type".

## Scheduling

`pipeline schedule` records *what* should run and *when* in pygeofetch's schedule registry (`scheduled_pipelines.json`
in its config directory). **It does not start anything running.** Something must keep the scheduler alive, such as the
pygeofetch scheduler process or an operating-system job:

```bash
# crontab -e   — Monday 06:00, using the saved file
0 6 * * 1  /path/to/venv/bin/pygeovision data pipeline run /path/to/weekly-sentinel2.yaml >> /var/log/pgv.log 2>&1
```

The cron format is the five-field POSIX form (`minute hour day-of-month month day-of-week`), in UTC. The pygeofetch
scheduler's own cron handling is simplified; for anything beyond "every N hours" or "weekly at a time", use the
operating-system scheduler or a CI schedule as above. Because `date_range: last_7_days` is resolved at *run* time, the
same file always means "the most recent seven days".

## Running a model afterwards

Data pipelines end with files on disk. To act on them, run the next stage yourself:

```bash
pygeovision data pipeline run weekly-sentinel2.yaml
pygeovision channel water_bodies --bbox -0.30 5.55 -0.10 5.70 --date 2024-06     # a named analysis
pygeovision infer --help                                                            # or your own network
```

`DataPipeline.ai_process(...)` exists only to *describe* such a step. Because the runner cannot execute it,
`validate()` returns `False` and `run()` raises `ValueError` for a pipeline that contains one, instead of failing half
way through a download.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown step type: 'ai'` / `'dowload'` | A step the runner does not have | Use only `search`, `filter`, `download`, `process`, `export` |
| Search returns 0 scenes | Wrong bbox order, narrow dates, or `cloud_cover` too strict | `--step search` first; check `min_lon < max_lon` |
| Everything is downloaded, nothing filtered | Filter expression errored and kept all scenes | Test the expression on one scene; field names are those of the search result |
| Export "completed" but files unchanged | Export copies; it does not convert | Add `cog`/`reproject`/`compress` to a `process` step |
| `boto3 not installed` / `google-cloud-storage not installed` | Cloud export needs the client library | `pip install boto3` (S3) or `pip install google-cloud-storage` (GCS), and set up credentials |
| `cloud_mask requires an SCL band` | SCL not downloaded | Add `SCL` to the bands of the download |
| Scheduled pipeline never runs | Registering does not start a scheduler | Run it from cron or a CI schedule |
| Provider asks for credentials | Not signed in | [Authentication](authentication.md) |
