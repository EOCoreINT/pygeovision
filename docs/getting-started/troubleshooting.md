# Troubleshooting

Find your symptom, learn what it means, and do the thing that fixes it. The page follows the order of a run: the
install, finding imagery, downloading, producing a result, reading it, and producing reports.

Before anything else, three commands answer most questions:

```bash
pygeovision doctor          # is every component installed and reachable?
pygeovision -v channel …    # the same command with detailed logging
cat <output>/run_summary.json   # what the run actually used and measured
```

When you ask for help, include the output of `pygeovision doctor`, the exact command, and the `-v` log.

## How a failure is reported

A pipeline **never returns an empty file or a zero-filled raster** to stand in for a failure. It returns
`success=False` and `error` (Python) or prints `✗` and exits non-zero (CLI). The message says what was missing. Any
other outcome — a file that exists — is a real result, which may still be a poor one; the sections below on reading
results help you judge.

```python
result = client.pipeline("flood_mapping", bbox=bbox, date_before="2024-05", date_after="2024-06")
if not result.success:
    print(result.error)
```

## Installation and environment

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: … rasterio` / `geopandas` / `shapely` | The geo stack is not installed | `pip install "pygeovision[geo]"` |
| `ImportError: … matplotlib` or no PDF/Word/Excel | The report extra is missing | `pip install "pygeovision[report]"`. Analysis still works; only reports are skipped (`deliverables_error` says why) |
| `AI engine requires additional dependencies` | PyTorch and friends are missing | `pip install "pygeovision[ai]"` (only needed for networks) |
| `ImportError: scikit-image required` | Texture or OBIA | `pip install scikit-image` or the `obia` extra |
| `pip` refuses with "externally managed environment" | System Python | Use a virtual environment (`python -m venv .venv`) |
| `pygeovision: command not found` | The environment is not active | Activate the environment (`source .venv/bin/activate`) or call `.venv/bin/pygeovision` |
| `doctor` shows pygeofetch ✗ | Core dependency missing or broken | `pip install --upgrade pygeofetch` |
| Works in the terminal, fails in a notebook | The notebook uses another interpreter | Install into the kernel's environment (`%pip install …`) |

## Finding imagery

| Symptom | Cause | Fix |
|---|---|---|
| "No imagery found." / "No usable imagery." | Nothing matched, or everything was rejected | See the checklist below |
| Zero search results | Box in the wrong order, dates in the future, nothing in that window | `min_lon < max_lon`, `min_lat < max_lat`; **longitude first**. Widen the dates |
| The box is in the wrong place on a map | Latitude and longitude swapped | Google Maps gives *lat, lon*; PyGeoVision wants *lon, lat* ([Core concepts](concepts.md)) |
| Results only from one provider | The others need credentials or lack the collection | `pygeovision data providers`; [Authentication](../data/authentication.md) |
| Search is slow the first time, instant the second | One-hour cache | Intended. `pygeovision data cache clear` to force a fresh search |
| `HTTP 429` / `406` from Overpass (roads, buildings from OSM) | Public Overpass server is busy or refusing | Retry later; the client already rotates servers and backs off. For heavy use run a private Overpass ([Footprints](../data/footprints.md)) |
| A provider returns authentication errors | Credentials missing or expired | `pygeovision data auth test <provider>`, then `auth add` |

**"No usable imagery" checklist**

1. Is the **date window** wide enough? A single day rarely has a scene; use a month or a 2–3 week range
   ([Dates](concepts.md#dates)).
2. Is the area cloudy at that time of year (tropical wet season)? Widen the window to the dry season or use a
   radar pipeline ([`sar_flood_mapping`](../pipelines/catalog/sar_flood_mapping.md),
   [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md)).
3. Does the satellite exist for that date? Landsat 8/9 imagery starts in 2013/2021, Sentinel-2 in mid-2015 (L2A
   widely from late 2018), Sentinel-1 in 2014.
4. Is the box tiny or in open ocean? Reference products report "no data for this bounding box".
5. Run with `-v` and read which scenes were found, which were dropped and why.

## Downloading

| Symptom | Cause | Fix |
|---|---|---|
| Download "succeeds" but the file is tiny or unreadable | The provider returned an error page, or an interrupted earlier download was taken for complete | Pipelines retry once without resume automatically. By hand: `client.download(..., resume=False, overwrite=True)` |
| `HTTP 403` fetching a Landsat file from AWS | Requester-pays bucket | Use `planetary_computer` or `usgs` for Landsat |
| Slow or stalling downloads | Bandwidth, or too many parallel workers | Smaller box; lower `parallel`; pick a closer provider |
| Disk fills up | Scenes are cached | Choose `--output` on a larger disk; delete old run folders |
| Shell tools (`curl`, `wget`) are blocked but Python works | Locked-down network | PyGeoVision's own downloads use the Python clients; ask your administrator to allow the provider hosts |

## Getting a result that looks wrong

First read three numbers in `result.stats` / `run_summary.json`:

| Number | Healthy | If not |
|---|---|---|
| `valid_fraction` / `clear_fraction` | High (> 0.7) | Cloud or shadow covers much of the box. Widen the dates; the statistics describe only the visible part |
| `aoi_coverage_fraction` | ≈ 1.0 | Imagery covers only part of your box. Do not quote totals for the whole box ([Scene selection](../data/scene-selection.md)) |
| The pipeline's threshold, scene dates | As expected | A pipeline uses the documented threshold; check `stats` for the value actually used |

| Symptom | Likely cause | Fix |
|---|---|---|
| Everything is `255` / empty | All pixels masked as cloud/shadow, or no overlap between the two dates | Another date; check `valid_fraction` |
| Output all zeros | Index computed on unscaled data, or wrong band order | Use the pipelines or `stack_and_prepare_bands`; see [Radiometry](../data/radiometry.md) |
| Values around ±0.0 for every index | Reflectance scaled twice | Check the Sentinel-2 offset rule in [Radiometry](../data/radiometry.md) |
| Straight seam or colour jump across the map | Two acquisitions mosaicked | Normal for a box wider than one scene; per-scene correction keeps radiometry consistent, but the surface may truly differ between dates ([Scene selection](../data/scene-selection.md)) |
| Map offset from the basemap | The raster is fine; a vector was read in the wrong CRS | Check units and CRS ([Postprocessing](../data/postprocessing.md#units-read-this-before-choosing-a-threshold)) |
| Hundreds of tiny polygons | Pixel noise | `sieve` then `vectorise --min-area` |
| Percentages do not add up to 100 | They are shares of **valid** pixels; nodata is excluded | Intended ([Core concepts](concepts.md#nodata-masks-and-what-no-data-means)) |
| `index` result NaN in file but `0` in array | Array return values replace NaN with 0 for convenience | Use the file output when you need nodata ([Indices](../data/indices.md)) |
| SAR flood map flags hills or airports | Radar shadow and smooth surfaces look like water | The multi-sensor pipeline uses terrain (HAND) to remove them ([Radar](../data/sar.md)) |
| Estimate disagrees with an official figure | Different sensor, date, threshold or definition | Compare definitions; the pipeline page's "Interpreting results" lists the usual reasons |

## Models and weights

| Symptom | Cause | Fix |
|---|---|---|
| `ModelWeightsRequired` / "needs your trained weights" | A *needs weights* pipeline was run without `--checkpoint` | Pass your trained `.pth`, or use the *ready* alternative named in the message. Nothing is downloaded before this check |
| `size mismatch` / `Missing key(s)` loading a checkpoint | Checkpoint was trained with another architecture, class count or band count | Build the model exactly as it was trained. Checkpoints are loaded strictly on purpose |
| Predictions are uniform or random-looking | Wrong input scaling or band order for the model | Match the preprocessing used in training ([AI-ready data](../data/ai-ready.md)) |
| `CUDA out of memory` | Tile or batch too large | Smaller `tile_size`/`batch_size`; or CPU: `CUDA_VISIBLE_DEVICES=` |
| Pretrained weights cannot be downloaded | No network to the model hub | Pre-download, or train from scratch; see [Model registry](../models/registry.md) |
| `Model '…' not found in the registry` | Wrong name | `pygeovision models list` |

## Reports and maps

| Symptom | Cause | Fix |
|---|---|---|
| No `deliverables/` folder | `--no-report`, `PYGEOVISION_REPORT=0`, or the report failed | After `--no-report`, run again without it. After a failure, check `result.metadata["deliverables_error"]` and, if `run_summary.json` exists, rebuild with `pygeovision report <output>` |
| Report missing PDF/Word/Excel | The report extra is missing | `pip install "pygeovision[report]"` |
| Class names show as numbers | The pipeline has no names for a custom network | `pygeovision report <output> --class-names …` ([Rebuilding](../reporting/rebuilding.md)) |
| Shapefile fields are truncated | The 10-character Shapefile limit | Use the GeoPackage |

## Performance and memory

- **Large boxes** download several scenes and mosaic them. Use a smaller box or `resolution_m` (where a pipeline offers
  it) for a first look.
- **Raster size**: a 100 km × 100 km box at 10 m is 100 million pixels per band. Process country-scale work in tiles
  or at 30–100 m.
- **Speed** is dominated by download. Re-running the same box reuses cached searches for an hour; keep the output
  folder to avoid re-downloading scenes.
- **GPU** matters only for networks. Index, terrain, hydrology and reference-map pipelines run on the CPU in seconds
  to minutes once the data is local.

## Still stuck?

1. Re-run with `pygeovision -v` and save the log.
2. Open `run_summary.json` and look at `parameters`, `metadata.input_coverage` and `stats`.
3. Check the pipeline's own page: each has a "Troubleshooting" section for its specific failures.
4. Search the [FAQ](faq.md).
5. Open an issue with `pygeovision doctor`, the command, the log, and the bounding box and dates.

```{toctree}
:hidden:
:maxdepth: 1

diagnostics
faq
```
