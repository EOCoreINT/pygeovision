# Configuration reference

Every setting of `PyGeoVisionConfig`, its type, default and, most important, **whether anything reads it**. The guide with
precedence rules, file locations and "what to do when" is [Configuration](../getting-started/configuration.md); this page is the lookup table.

| Setting | Type | Default | Valid values | Read by |
|---|---|---|---|---|
| `log_level` | str | `INFO` | Python log level names | Stored only; use `PyGeoVision(log_level=…)` or `pygeovision -v` |
| `pygeofetch.default_providers` | list[str] | `[planetary_computer, aws_earth, element84]` | provider names ([providers](../data/providers.md)) | **Search** (when you name no provider) |
| `pygeofetch.cache_ttl_seconds` | int | `3600` | ≥ 0 (`0` = no reuse) | **Search cache** |
| `pygeofetch.download_parallel` | int | `4` | > 0 | Available via `as_pygeofetch_config()`; downloads use their own arguments |
| `pygeofetch.download_retry_attempts` | int | `5` | > 0 | same |
| `pygeofetch.verify_checksum` | bool | `false` | | same |
| `pygeofetch.timeout_seconds` | int | `120` | > 0 | same |
| `gpu.device` | str | `auto` | `auto`, `cuda`, `mps`, `cpu` | Stored only; training and inference take `devices=` |
| `gpu.mixed_precision` | bool or `auto` | `auto` | `true`, `false`, `auto` | Stored only |
| `gpu.memory_fraction` | float | `0.9` | 0 < x ≤ 1 | Stored only |
| `training.batch_size` | int | `16` | > 0 | Stored only; `TrainingConfig(batch_size=…)` is what trains |
| `training.learning_rate` | float | `1e-4` | > 0 | Stored only |
| `training.max_epochs` | int | `100` | > 0 | Stored only |
| `training.warmup_epochs` | int | `5` | ≥ 0 | Stored only |
| `training.gradient_accumulation_steps` | int | `1` | > 0 | Stored only |
| `training.max_grad_norm` | float | `1.0` | ≥ 0 | Stored only |
| `model_hub.cache_dir` | path | `~/.pygeovision/models` | any writable folder | Stored only; model downloads use `~/.pygeovision/models` |
| `model_hub.checksum_verify` | bool | `true` | | Stored only |

"Stored only" means the value is available as `client.config.<section>.<name>` for your own scripts and is **not** applied
to training, inference or downloads, which take their settings from their own arguments. This is deliberate documentation
rather than an apology: a setting that looks like it works but does not is the failure to avoid, so the
[guide](../getting-started/configuration.md#which-settings-take-effect) repeats it next to the equivalent working mechanism
(`CUDA_VISIBLE_DEVICES` for the GPU, the arguments of `TrainingConfig`).

## Environment variables

| Variable | Sets | Notes |
|---|---|---|
| `PYGEOVISION_GPU_DEVICE` | `gpu.device` | |
| `PYGEOVISION_GPU_MIXED_PRECISION` | `gpu.mixed_precision` | `true` / `false` / `auto` |
| `PYGEOVISION_MODEL_HUB_CACHE_DIR` | `model_hub.cache_dir` | |
| `PYGEOVISION_LOG_LEVEL` | `log_level` | |
| `PYGEOVISION_TRAINING_BATCH_SIZE` | `training.batch_size` | Non-numeric → an error naming the setting at client creation |
| `PYGEOVISION_TRAINING_LR` | `training.learning_rate` | |
| `PYGEOVISION_REPORT` | whether pipelines write maps, vectors and reports | `0`, `false`, `no`, `off` disable; default on |
| `PC_SDK_SUBSCRIPTION_KEY` | Planetary Computer rate limit | Optional |
| `CUDA_VISIBLE_DEVICES` | Which GPUs PyTorch can see | Standard PyTorch |
| `RANK`, `WORLD_SIZE`, `LOCAL_RANK` | Distributed training | Set by `torchrun` |

## In code

```python
from pygeovision.core.config import PyGeoVisionConfig

cfg = PyGeoVisionConfig.load()                 # defaults + files + environment
cfg.pygeofetch.default_providers
cfg.as_pygeofetch_config()                     # the subset pygeofetch understands
cfg.save()                                     # writes ~/.pygeovision/config.yaml
```

```{eval-rst}
.. autoclass:: pygeovision.core.config.PyGeoVisionConfig
   :members: load, as_pygeofetch_config, save
   :no-index:
```

## Files read, in order

1. Built-in defaults
2. `~/.pygeovision/config.yaml`
3. `./.pygeovision.yaml`
4. The file passed as `config_path`
5. `PYGEOVISION_*` environment variables

Sections are merged key by key, so a project file that sets only `pygeofetch.cache_ttl_seconds` leaves the other keys of that
section as the user file or defaults set them. A file with a YAML syntax error is skipped with a warning that names it.
