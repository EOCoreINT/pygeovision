# AI-ready data

A neural network is stricter than a GIS. It wants a `float32` array of shape `(bands, height, width)`, scaled to the
range it was trained on, with no `NaN`, no `Inf`, the right number of bands and a believable pixel size. A raster that
looks fine on a map can violate any of these and give you confident nonsense.

PyGeoVision therefore puts a **gate** between downloaded data and any model. Three entry points make up the gate:

| Entry point | Use it for |
|---|---|
| [`client.prepare_for_ai()`](#prepare_for_ai-the-general-gate) | Any raster or scene folder → a clipped, masked, scaled, validated file and array |
| [`prepare_optical_for_ai()`](#optical-rasters-prepare_optical_for_ai) | A raster that is already preprocessed; you only want it checked and returned as an array |
| [`prepare_sar_for_ai()`](#radar-prepare_sar_for_ai) | Sentinel-1 VV/VH → the 6-channel tensor that optical foundation models expect |

The pipelines and `pygeovision infer` call the gate for you. Use it directly when you run your own model, build a
training set, or want to see exactly what a network will be fed.

## `prepare_for_ai`: the general gate

**Why.** To make the steps in the right order, every time. Clipping before masking before scaling before resampling is
not interchangeable: scaling before masking changes what "nodata" means, and resampling before clipping wastes memory.

**How.**

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
ready = client.prepare_for_ai(
    "./downloads/S2_scene_stack.tif",
    bbox=(-0.30, 5.55, -0.10, 5.70),      # WGS84; reprojected to the raster's CRS for you
    scl_path="./downloads/SCL.tif",       # optional: Sentinel-2 scene classification
    normalise="scale_factor",             # | "minmax" | "zscore" | "percentile" | None
    scale_factor=10000.0,
    resample_m=None,                      # e.g. 20 to resample to 20 m
    model_type="segmentation",
    output_path="./ready/scene_ready.tif",
)

arr = ready.array          # float32 (C, H, W), validated
print(ready.shape, ready.resolution_m)
print(ready.steps)         # ['pgf.clip((-0.3, 5.55, -0.1, 5.7))', 'normalise(scale_factor)', 'validate(segmentation)']
print(ready.report)        # ValidationReport
```

### Order of operations

1. **Stack** the band files (`stack_bands`, `stack_dir`) when the input is a scene folder.
2. **Clip** to `bbox` (or `clip_geojson`).
3. **Mask** with `cloud_mask_path` (binary, 1 = cloud) and/or `scl_path` with `scl_keep_classes`.
4. **Normalise** the values.
5. **Resample** to `resample_m`.
6. **Validate** with the [validator](validation.md): fix dtype, `NaN`, `Inf` and outliers and refuse hopeless input.
7. **Return** the array and metadata.

Each step is skipped when its argument is not given.

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `input_path` | required | A stacked GeoTIFF, or a scene directory (then give `stack_bands`) |
| `stack_bands` / `stack_dir` | `None` | Band names to find and stack, and an alternative folder to look in |
| `bbox`, `bbox_crs` | `None`, `EPSG:4326` | Crop box. Reprojected to the raster's CRS automatically, so always give lon/lat |
| `clip_geojson` | `None` | Polygon clip. Wins over `bbox` when both are given |
| `cloud_mask_path` | `None` | Binary mask raster, 1 = cloud |
| `scl_path` | `None` | Sentinel-2 SCL raster |
| `scl_keep_classes` | `(4, 5, 6)` | SCL codes to keep. The default also removes dark areas, unclassified pixels and snow ([SCL classes](preprocessing.md)) |
| `normalise` | `"scale_factor"` | Scaling method; `None` leaves values unchanged |
| `scale_factor` | `10000` | Divisor for `scale_factor` |
| `resample_m` | `None` | Target pixel size in metres |
| `model_type` | `"segmentation"` | Hint for the validator |
| `output_path` | next to the input, `<name>_ai_ready.tif` | Where to write the file |
| `use_pgf_preprocess` | `True` | Allow the pygeofetch engine (below). `False` forces PyGeoVision's own preprocessor |

```{warning}
`normalise="scale_factor"` only divides. Sentinel-2 products processed after January 2022 also carry a **+1000
offset** that must be subtracted first, which this gate does not do. Rasters written by the pipelines and by
`stack_and_prepare_bands` are already correct reflectance — pass `normalise=None` for them
([Radiometry](radiometry.md)). Applying `scale_factor` to already-scaled data squashes everything to ≈0.
```

### What comes back: `AIReadyRaster`

The return value is a **string** (the output path) that also carries attributes. Use it anywhere a path is accepted,
or read the extras:

| Attribute | Meaning |
|---|---|
| `str(ready)` / `ready` as path | Path of the written GeoTIFF |
| `ready.array` | `float32` array `(C, H, W)` after validation; `None` if validation did not produce one |
| `ready.shape` | Shape of the output |
| `ready.resolution_m` | Pixel size (map units of the file, metres for projected data) |
| `ready.report` | The validation report: warnings, fixes applied, errors |
| `ready.steps` | The list of steps that were actually applied |

`ready["array"]` and `ready.get("steps")` also work, for code written against the earlier dict return.

### Two engines, one result

When the **pygeofetch processing engine** (pygeofetch ≥ 2.0) is installed, `prepare_for_ai` uses it for clipping, SCL
masking and resampling (steps are labelled `pgf.…` in `ready.steps`); otherwise it uses PyGeoVision's own
`Preprocessor`. The engine cannot honour every argument, so a call that uses **`clip_geojson`, `cloud_mask_path`,
`stack_bands`/`stack_dir`, a non-default `scl_keep_classes`, or a scene directory** is automatically run by the
PyGeoVision preprocessor instead (an INFO log line says so). You never get a result with an argument silently ignored.
Force the PyGeoVision route with `use_pgf_preprocess=False`, for instance if the engine fails on an unusual CRS.

### What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: … input_path=None` | The download failed upstream | Check `DownloadResult.success` and `.error` before calling |
| `FileNotFoundError` | Wrong path | Pass the stacked file, or the folder plus `stack_bands` |
| All output ≈ 0 | `scale_factor` applied to already-scaled data | `normalise=None` |
| Output all NaN / empty after clipping | `bbox` does not overlap the raster, or the lon/lat order is swapped | Check `min_lon < max_lon`, and that the bbox is in lon/lat |
| `Pixel size … is not physically plausible` (SAR) | Lost georeference (identity transform) | [Georeference recovery](sar.md) |
| Warning "Expected 6 bands, got 4" | Band selection at download | Re-download with the right bands, or change `expected_bands` |
| Network output looks washed out | Wrong scaling for the model | Use the normalisation in the model's page ([Models](../models/index.md)) |

## Optical rasters: `prepare_optical_for_ai`

For a raster you have already preprocessed. It does not clip, mask or scale — it checks it and returns the array.

```python
from pygeovision.data.ai_prep import prepare_optical_for_ai

res = prepare_optical_for_ai("scene_ready.tif", expected_bands=6, expected_range=(0.0, 1.0))
if res["ready"]:
    x = res["array"]                  # float32 (6, H, W)
else:
    print(res["validation"]["errors"], res["validation"]["warnings"])
```

**Checks performed**, in order:

1. The file opens and has a CRS (no CRS → error: run georeference recovery).
2. Pixel size is between 0.5 m and 10 km (a lon/lat raster is converted from degrees; outside that range the transform is almost certainly lost → error).
3. At least 16×16 pixels (otherwise the clip box was wrong → error).
4. No band is entirely `NaN` (error); a band more than half `NaN` is a warning ("check cloud masking").
5. `Inf` → 0 and `NaN` → 0 (recorded under `fixes_applied`).
6. Values are inside `expected_range` ± 0.1 (otherwise a warning "check normalisation").
7. The band count equals `expected_bands` (otherwise a warning).

The result contains `array`, `working_path`, `validation` (with `valid`, `warnings`, `errors`, `fixes_applied`, `shape`,
`value_range`, `pixel_size_m`, `crs`) and `ready`. `ready` is `True` only when the file passed the hard checks and an
array was produced; **warnings do not make `ready` false**, so read them.

```{note}
`NaN` → 0 makes cloud-masked pixels look like very dark ground to the network. That is the correct input for most
networks, but remember that the *prediction* over those pixels is not meaningful. Keep your cloud mask and apply it
to the output.
```

## Radar: `prepare_sar_for_ai`

**Why.** Foundation models such as Prithvi were pre-trained on six optical bands. To use them on Sentinel-1 you must
present the two radar channels (VV and VH) as six. This function recovers a lost georeference, builds the six
channels and validates them.

```python
from pygeovision.data.ai_prep import prepare_sar_for_ai

res = prepare_sar_for_ai("s1_normalised.tif", channel_mapping="physics_guided")
if res["ready"]:
    x = res["six_channel"]            # float32 (6, H, W)
else:
    print(res.get("error") or res["validation"]["errors"])
```

**Input.** A two-band GeoTIFF, VV = band 1, VH = band 2, **normalised to 0–1** (the output of the SAR chain's
normalisation step, [Radar preprocessing](sar.md)). A lost georeference (identity transform) is repaired first
when `fix_georeference_first=True` (the default); if it cannot be repaired the call returns `ready=False` with an
`error`.

**Channel mappings** (`channel_mapping`):

| Name | Six channels (in order) | Use for |
|---|---|---|
| `physics_guided` (default) | VH, (VH+VV)/2, VV, 1−VH, 1−VV, VV/VH (clipped to 0–10, scaled to 0–1) | Water and flood work with a fine-tuned model |
| `mean_repeat` | (VH+VV)/2 six times | Ablation only |
| `vv_vh_ratio` | VV, VH, ratio, VV, VH, ratio | Visual inspection |
| `replicate` | VV, VH, VV, VH, VV, VH | Baseline for ablations |
| `sentinel_6ch_db` | Same layout as `physics_guided` but for dB input, each derived channel rescaled to 0–1 | Passing dB values straight in |

```{important}
These are **heuristic** mappings that place radar information where an optical network expects "dark water, bright
vegetation". They are not a calibrated transformation, and a model that has never seen radar will give weak results
with any of them. For good radar results fine-tune the network on labelled SAR
([SAR adapters](../foundation-models/radar.md#radar-with-optical-foundation-models)) or use the SAR-native pipelines
([`sar_flood_mapping`](../pipelines/catalog/sar_flood_mapping.md)), which need no network.
```

If only VV is available, VH is approximated as 0.6 × VV and a warning is logged. Treat that as a last resort: the
two polarisations carry different information.

Returned keys: `six_channel`, `vv`, `vh`, `working_path`, `channel_mapping`, `metadata` (CRS, transform, shape, pixel
size), `validation`, `ready`. `validate_sar_ai_input(six_ch, model="prithvi")` re-checks any six-channel tensor for shape,
range, empty channels, minimum size (64×64 pixels) and aspect ratio.

## The fluent `EOPipeline`

For scripts that go from search to a model-ready SAR or optical array in one chain:

```python
from pygeovision.data.ai_prep import EOPipeline

ai_input = (
    EOPipeline(provider="planetary_computer")
    .search(bbox, ("2024-06-01", "2024-06-30"), satellite="Sentinel-1")
    .select(n=1, strategy="best")           # best | newest | oldest | score | as_is
    .download(output_dir="./data/", bands=["vv", "vh"])
    .prepare_sar(mapping="physics_guided")   # terminal: returns the dict described above
)
```

`search`, `select`, `download`, `preprocess` and `authenticate` return the pipeline so they chain;
`prepare_sar()` and `prepare_optical()` end the chain. `select(strategy="best")` orders by cloud cover, with scenes that
have no cloud value (radar) last — for radar prefer `newest` or `oldest`. For optical work over a whole study area, use
a [pipeline](../pipelines/index.md): `EOPipeline` takes the first downloaded scene and does not mosaic or choose scenes
to cover a box ([Scene selection](scene-selection.md)).

## Choosing between the gates

```{figure} ../_static/img/tree-ai-ready.svg
:alt: Which AI-ready gate to use for a scene folder, clipped optical data or clipped radar
:width: 100%
:class: pgv-figure

Pick the gate by what you already have.
```

Then hand the array to [inference](../inference/index.md), or write it into a [training set](../training/index.md).
