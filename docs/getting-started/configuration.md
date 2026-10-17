# Configuration

PyGeoVision works with no configuration. This page is for when you want to change a default — which providers are
searched, how long searches are cached, where files are kept — and for knowing **exactly which settings have an
effect**, because a setting that is silently ignored is worse than none.

## Where PyGeoVision keeps things

| Location | What is stored | Safe to delete? |
|---|---|---|
| `~/.pygeovision/config.yaml` | Your user configuration (you create it) | Yes |
| `./.pygeovision.yaml` | Project configuration, read from the directory you run in | Yes |
| `~/.pygeovision/search_cache/` | Search results, reused for one hour | Yes (`pygeovision data cache clear`) |
| `~/.pygeovision/cache/` | Downloaded reference data: building footprints, WorldCover, Dynamic World, SAM and foundation weights | Yes; re-downloaded on demand |
| `~/.pygeovision/models/` | Model weights fetched by the registry | Yes |
| `~/.pygeovision/experiments/` | Experiment records | Only if you do not need the history |
| `~/.pygeovision/pipeline_history.jsonl` | Runs of `pygeovision data pipeline run` | Yes |
| pygeofetch's own config directory | Stored provider credentials | Use `pygeovision data auth remove` |

On Windows `~` is your user profile folder. To move the search cache, pass `cache_dir=` to the client
(`PyGeoVision(cache_dir=Path("/data/cache"))`).

## Precedence

Later items override earlier ones:

1. Built-in defaults
2. `~/.pygeovision/config.yaml`
3. `./.pygeovision.yaml`
4. A file given as `PyGeoVision(config_path="my.yaml")`
5. `PYGEOVISION_*` environment variables

A file that cannot be parsed is **skipped with a warning** (naming the file and the YAML error), so a typo does not stop
a run — but its settings are not applied. Look for the warning if a setting seems ignored.

## The configuration file

```yaml
# ~/.pygeovision/config.yaml
log_level: INFO

pygeofetch:
  default_providers: [planetary_computer, aws_earth, element84]
  cache_ttl_seconds: 3600
  download_parallel: 4
  download_retry_attempts: 5
  verify_checksum: false
  timeout_seconds: 120

gpu:
  device: auto            # auto | cuda | mps | cpu
  mixed_precision: auto   # true | false | auto
  memory_fraction: 0.9

training:
  batch_size: 16
  learning_rate: 1.0e-4
  max_epochs: 100
  warmup_epochs: 5
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

model_hub:
  cache_dir: ~/.pygeovision/models
  checksum_verify: true
```

### Which settings take effect

| Setting | Effect |
|---|---|
| `pygeofetch.default_providers` | **Applied.** The providers searched when you name none (and no satellite shortcut applies). Providers that need credentials you have not stored are skipped from the default list, so a long list is safe |
| `pygeofetch.cache_ttl_seconds` | **Applied.** How long a cached search result is reused; `0` disables reuse |
| `pygeofetch.download_parallel`, `download_retry_attempts`, `verify_checksum`, `timeout_seconds` | Stored on `client.config` and available through `client.config.as_pygeofetch_config()`. **Downloads use the arguments you pass** (`parallel=`, `retry_attempts=`, …), whose defaults are 4 and 5 |
| `gpu.*`, `training.*` | Stored on `client.config` for your own scripts. **Training and inference choose their device and hyper-parameters from their own arguments** (`TrainingConfig(devices=…, batch_size=…)`) and are not read from this file |
| `model_hub.*`, `log_level` | Stored on `client.config`. Model downloads use `~/.pygeovision/models`; the log level comes from `PyGeoVision(log_level=…)` or `pygeovision -v` |

So for the things that matter most in daily use — *which providers* and *how long to cache* — the file works as you
would expect. For *which GPU*, use the standard PyTorch mechanism, which every part of PyGeoVision respects:

```bash
CUDA_VISIBLE_DEVICES=1 pygeovision infer predict …      # only GPU 1 is visible
CUDA_VISIBLE_DEVICES= pygeovision infer predict …       # force CPU
```

## Environment variables

| Variable | Meaning |
|---|---|
| `PYGEOVISION_GPU_DEVICE` | Sets `gpu.device` in `client.config` |
| `PYGEOVISION_GPU_MIXED_PRECISION` | Sets `gpu.mixed_precision` |
| `PYGEOVISION_MODEL_HUB_CACHE_DIR` | Sets `model_hub.cache_dir` |
| `PYGEOVISION_LOG_LEVEL` | Sets `log_level` |
| `PYGEOVISION_TRAINING_BATCH_SIZE`, `PYGEOVISION_TRAINING_LR` | Set `training.batch_size`, `training.learning_rate` |
| `PYGEOVISION_REPORT` | `0`, `false`, `no` or `off` turns off the map, vectors and reports of every pipeline run (the same as `--no-report` / `report=False`). Default on |
| `PC_SDK_SUBSCRIPTION_KEY` | Optional Planetary Computer subscription key. Lifts the anonymous rate limit when reading reference data |
| `CUDA_VISIBLE_DEVICES` | Standard PyTorch GPU selection |
| `RANK`, `WORLD_SIZE`, `LOCAL_RANK` | Read by distributed training; set by `torchrun` |

An invalid value (for example a non-numeric batch size) is an error when the client is created, naming the setting.
Provider credentials are **not** configured here — see [Authentication](../data/authentication.md).

## In Python

```python
from pathlib import Path
from pygeovision import PyGeoVision

client = PyGeoVision(
    config_path="project.yaml",       # optional extra config file
    cache_dir=Path("/data/pgv-cache"),# where search results are cached
    log_level="DEBUG",
)

client.config.pygeofetch.default_providers      # what is in force
client.data.default_providers
client.data.set_cache_ttl(600)                  # change the cache lifetime for this client
```

## Logging

PyGeoVision logs through Python's `logging`. By default you see progress at `INFO`. For more detail:

```bash
pygeovision -v channel flood_mapping …            # verbose, from the command line
```

```python
client = PyGeoVision(log_level="DEBUG")
```

`DEBUG` shows which providers were queried, which scenes were chosen and why others were dropped, and every file read
or written. Include this output when you report a problem.

## Disabling the report

Deliverables are on by default. To skip them for one run, pass `--no-report` (CLI) or `report=False` (Python). To skip
them everywhere, export `PYGEOVISION_REPORT=0`. A run with reports off writes no `run_summary.json` either; for a quicker run that keeps it,
use `--report-formats none --vector-formats none`. An analysis is never turned into a failure by a reporting problem: if
the map or report cannot be built, the rasters are still written and `result.metadata["deliverables_error"]` says why
([Deliverables](../reporting/index.md)).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| Warning "Failed to load config file …" | YAML syntax error | Fix the file; its settings were not applied |
| My `default_providers` change does nothing | A provider named in the call overrides it, or a satellite shortcut applied | Call without `providers=`; shortcuts are listed in the [provider reference](../data/providers.md) |
| Old search results keep appearing | Cache lifetime | `pygeovision data cache clear`, or set `cache_ttl_seconds: 0` |
| `gpu.device: cpu` still uses the GPU | The file's `gpu` section is not read by training/inference | Set `CUDA_VISIBLE_DEVICES=` (empty) or pass `devices="cpu"` |
| Disk filling up | Reference-data and weight caches | Delete `~/.pygeovision/cache` and `~/.pygeovision/models`; they are re-created on demand |
