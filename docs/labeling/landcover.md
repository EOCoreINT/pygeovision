# Land-cover labels

**Classes:** `ESAWorldCoverLabeler`, `DynamicWorldLabeler` in `pygeovision.labeling` ·
**Client:** `client.labeling.esa_worldcover(...)`, `client.labeling.dynamic_world(...)`

## Why use them

For "what covers the ground here: trees, crops, built-up, water, bare soil", there are free **10 m maps covering the
whole planet**. Using one as labels gives you millions of labelled pixels for any place in minutes. They are a good
fit for **coarse land-cover models** and for pre-training; they are a poor fit for anything small (single houses, narrow
roads) or anything that changed after the map was made.

| | ESA WorldCover | "Dynamic World" labeler |
|---|---|---|
| Product | ESA WorldCover 2020 (v100) or 2021 (v200) | **Default backend: Impact Observatory/Esri 10 m Annual Land Use Land Cover (9-class)**, *not* Google's Dynamic World. Earth Engine backend: real Dynamic World |
| Classes | 11 | 9 |
| Time | One map per year | Annual map per year, 2017–2023 (Planetary Computer) |
| Account needed | No | No (Planetary Computer); Earth Engine account for the other backend |
| Writes a file | Yes | Planetary Computer: yes. Earth Engine: **no**, it starts a Google Drive export |

```{warning}
`DynamicWorldLabeler` with the default backend does **not** return Google's Dynamic World. It returns the Impact
Observatory annual map, with **different class numbers**. The result includes `"product": "io-lulc-9-class (…NOT
Google Dynamic World)"` and the matching `class_map` so you cannot mix them up silently. If you need the real Dynamic
World, use `backend="earth_engine"`.
```

## Class numbers

**ESA WorldCover** (`0` = no data):

| Value | Class | Value | Class |
|---:|---|---:|---|
| 10 | Tree cover | 70 | Snow and ice |
| 20 | Shrubland | 80 | Permanent water bodies |
| 30 | Grassland | 90 | Herbaceous wetland |
| 40 | Cropland | 95 | Mangroves |
| 50 | Built-up | 100 | Moss and lichen |
| 60 | Bare / sparse vegetation | | |

**Planetary Computer backend** (Impact Observatory 9-class, `0` = no data):

| Value | Class | Value | Class |
|---:|---|---:|---|
| 1 | Water | 7 | Built area |
| 2 | Trees | 8 | Bare ground |
| 4 | Flooded vegetation | 9 | Snow/ice |
| 5 | Crops | 10 | Clouds |
| 11 | Rangeland | | |

**Earth Engine backend** (real Dynamic World label band):

| Value | Class | Value | Class |
|---:|---|---:|---|
| 0 | Water | 5 | Shrub and scrub |
| 1 | Trees | 6 | Built |
| 2 | Grass | 7 | Bare |
| 3 | Flooded vegetation | 8 | Snow and ice |
| 4 | Crops | | |

The numbers are **not interchangeable** between the three. `IO_TO_DYNAMIC_WORLD` (importable from
`pygeovision.labeling.landcover`) converts Impact Observatory numbers to Dynamic World numbers for the seven classes
they share; rangeland and clouds have no equivalent and keep their numbers unless you remap them yourself.

## How to use

```python
from pygeovision.labeling import ESAWorldCoverLabeler, DynamicWorldLabeler

bbox = (-87.7, 41.8, -87.5, 41.9)

esa = ESAWorldCoverLabeler(year=2021).label(
    bbox, output_path="labels/chicago_esa.tif",
    reference_raster="data/chicago_s2.tif",
)
print(esa["success"], esa["year"], esa["classes_found"])

io = DynamicWorldLabeler().label(                       # backend="planetary_computer"
    bbox, date_range=("2023-01-01", "2023-12-31"),
    output_path="labels/chicago_lulc.tif", reference_raster="data/chicago_s2.tif",
)
print(io["product"], io["year"], io["class_map"])
```

### `ESAWorldCoverLabeler(year=2021).label(...)`

| Argument | Default | Meaning |
|---|---|---|
| `bbox` | required | Degrees, `(lon_min, lat_min, lon_max, lat_max)` |
| `output_path` | `./labels/esa_worldcover.tif` | Output |
| `remap_classes` | `None` | `{esa_value: new_value}`, applied to all pixels **at once** (so `{10: 20, 20: 30}` does not chain 10 → 30). Values you do not list keep their ESA number |
| `reference_raster` | `None` | Resample (nearest neighbour) onto your imagery's grid and CRS |
| `clip_to_bbox` | `True` | Kept for compatibility; the output is always the box window, snapped outward to whole pixels |

`year` must be 2020 or 2021; anything else raises `ValueError`.

### `DynamicWorldLabeler(backend=…).label(...)`

| Argument | Default | Meaning |
|---|---|---|
| `bbox` | required | Degrees |
| `date_range` | `("2023-01-01", "2023-12-31")` | Planetary Computer: the newest annual map overlapping this range. Earth Engine: per-pixel most common label over the range |
| `output_path` | `./labels/dynamic_world.tif` | Output |
| `label_mode` | `"most_likely"` | The only supported value. `"all_probabilities"` returns an error instead of being ignored |
| `remap_classes` | `None` | `{value_in_file: new_value}` |
| `reference_raster` | `None` | Planetary Computer only |

### Turning a land-cover map into a training set for one class

Often you want "built-up or not", not eleven classes:

```python
built = {50: 1, **{v: 0 for v in (10, 20, 30, 40, 60, 70, 80, 90, 95, 100)}}
ESAWorldCoverLabeler().label(bbox, output_path="labels/built.tif", remap_classes=built,
                             reference_raster="data/scene.tif")
```

Value `0` (no data) keeps its number, and the file declares `0` as its nodata value. The trainer reads raw numbers, so
pixels the map does not cover are seen as class `0`, "background". If the map has holes inside your area (open water
masked out, for example), convert them yourself afterwards (`label[label == 0] = 255`, written to a new file) so the loss
skips them.

## What you get

```text
labels/chicago_esa.tif      uint8, nodata=0, values from the table above (or yours after remap)
```

Result keys:

| Key | ESA | Planetary Computer | Earth Engine |
|---|:-:|:-:|:-:|
| `success`, `error`, `output_path` | ✓ | ✓ | `output_path` is `None` |
| `year` | ✓ | ✓ (the year actually used) | |
| `classes_found` | ✓ value → pixel count | ✓ | |
| `class_map` | ✓ | ✓ (`None` after remap) | ✓ |
| `n_tiles` | ✓ | ✓ | |
| `product` | | ✓ | |
| `submitted`, `ee_task`, `note` | | | ✓ (export started; nothing written yet) |

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `"No ESA WorldCover 2021 tiles found for bbox"` | Box outside the product, or lon/lat swapped | Check the box order; try `year=2020` |
| `"No annual land-cover map … overlaps … (the collection covers 2017-2023)"` | You asked for 2024 or later | Use `date_range` within 2017–2023 |
| `STAC search failed` / `HTTP 4xx/5xx` | No network or the catalogue is down | Retry; check a proxy; nothing is cached |
| HTTP 403 when reading tiles | The signing package is missing | `pip install planetary-computer` (a warning says so before the failure) |
| Output is all `0` | Ocean, or the area has no coverage | Look at `classes_found`; check the box |
| Pixels look blocky against the imagery | 10 m map resampled onto a finer grid | Expected. Nearest-neighbour is used on purpose; interpolating would invent classes. Train at 10 m or use these labels only as pre-training |
| Small roads/houses absent | 10 m resolution | Use [OSM](osm.md) or [buildings](buildings.md) for those |
| Labels disagree with your 2025 imagery | Map is from 2020–2023 | Use only for areas that have not changed, or check the date you actually need |
| `"remap_classes / reference_raster are not supported by the earth_engine backend"` | Earth Engine only starts an export | Use the Planetary Computer backend, or remap after downloading from Drive |
| `submitted: True` but no file | Earth Engine export | The file appears in your **Google Drive** when the task finishes. Check the task in the Earth Engine console |

## Honest accuracy

Published overall accuracies for these products are roughly 75–85% globally, and lower for classes like shrubland,
grassland and wetland, which are easily confused with each other. Treat them as **noisy labels**: a model trained on them
will reproduce their mistakes. If you only need one class (trees, built-up, water) and evaluate it against a handful of
your own hand-labelled points, the result is more reliable than the 11-class headline number.

The ESA data are CC BY 4.0 (credit ESA WorldCover). Check the licence of the annual map you use before redistribution.

```{note}
Everything on this page that reaches the network (Planetary Computer, Earth Engine) was written against the
published interfaces but **could not be exercised in the documentation build environment**. The offline tests cover the
mosaicking, georeferencing, remapping and grid alignment on small local GeoTIFFs. If a live call fails in a way this
page does not describe, please report it with the full `error` text.
```

```{seealso}
[Reference data and terrain](../data/reference-data.md) uses the same products inside the land-cover pipelines ·
[Combining sources](pipeline.md) · [Quality and cleaning](quality.md)
```
