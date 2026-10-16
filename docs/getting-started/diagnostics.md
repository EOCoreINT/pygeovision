# Diagnostics

Three commands tell you what is installed, what a run did, and where a failure came from. Use them before you open an issue,
and include their output when you do.

## `pygeovision doctor`

`doctor` checks every component PyGeoVision relies on and exits non-zero when one is missing or broken.

```bash
pygeovision doctor
```

```text
══════════════════════════════════════════
         PyGeoVision Diagnostics
══════════════════════════════════════════
  ✓  pygeovision
     python            : 3.13.16
     platform          : Linux
  ✓  pygeofetch          v2.6.3.0.4
  ✓  torch               v2.14.1+cu130
  ✓  rasterio            v1.5.2
  ✓  geopandas           v1.2.0
  ✓  credentials         0 provider(s) with stored credentials
  ✓  cache               1 entries

  → 7/7 components healthy
```

| Line | What it checks | If it shows ✗ |
|---|---|---|
| `pygeovision` | The package imports and the client can be created | Reinstall; if the message names a module, an install was partly copied (see `tools/verify_install.py` in the repository) |
| `pygeofetch` | The data layer is installed and answers | `pip install --upgrade pygeofetch` |
| `torch` | PyTorch is installed; shows CUDA and the GPU when there is one | Only needed for networks: `pip install "pygeovision[ai]"`. Absent PyTorch does not stop the 34 ready pipelines |
| `rasterio`, `geopandas` | The geo stack | `pip install "pygeovision[geo]"` |
| `credentials` | How many providers have stored credentials | A count of `0` is normal: three providers need no account ([authentication](../data/authentication.md)) |
| `cache` | The search cache is readable | Delete the cache folder named in [configuration](configuration.md) |

The same data is available from Python:

```python
import pygeovision as pgv

report = pgv.PyGeoVision().doctor()
print(report["summary"])            # "7/7 components healthy"
```

## `-v`: detailed logging

Put `-v` before the command to see every step: which providers were searched, how many scenes each returned, which scene was
chosen and why, what was downloaded and which cloud limit was applied.

```bash
pygeovision -v channel water_bodies --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --output ./first_run
```

A search that returns nothing explains itself: when the cloud limit hid every scene, the log names the limit and says how far to
raise it. See [Troubleshooting](troubleshooting.md#finding-imagery).

## `run_summary.json`

Every pipeline writes `run_summary.json` into its output folder: the dates, the scenes, the thresholds, the layers and the
quality figures it actually used. Read it before you read the map.

```bash
python -m json.tool first_run/run_summary.json | head -40
```

Check `valid_fraction` first. A map built from one cloud-edged scene looks the same as one built from ten, and only this figure tells
you how much of your box the percentages describe. How to read the rest is on [Reading results](../pipelines/results.md).

## Asking for help

Include, in this order:

1. The output of `pygeovision doctor`.
2. The exact command or Python call.
3. The `-v` log up to the failure.
4. `run_summary.json`, if the run got that far.

Then follow [Contributing](../project/contributing.md) for where to send it.
