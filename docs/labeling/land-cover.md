# Land Cover & Dynamic World Labelers

Two real, independent global land-cover data sources — genuinely
different update cadences.

## ESA WorldCover

```python
from pygeovision.labeling.landcover import ESAWorldCoverLabeler

result = ESAWorldCoverLabeler().label(
    bbox=(-0.15, 51.47, -0.10, 51.52),
    output_path="./labels/worldcover.tif",
)
```

Real, static 10m global land cover — the real STAC search
automatically resolves to whichever WorldCover release (2020 or 2021)
covers the requested area; there's no `year=` parameter to choose one
explicitly. Searches real STAC catalogs (Microsoft Planetary Computer)
for the covering real tiles, with a real HTTPS fallback if the STAC
path fails.

```{warning}
A real bug was found and fixed in this exact download fallback: the
real HTTP response status was never checked before writing content to
disk and appending it to the tile list — a 403/404 error page could be
silently written as if it were a real tile, only surfacing later as a
confusing `rasterio`-open error rather than a clear download-failure
message. Fixed to check the status first.
```

## Dynamic World

```python
from pygeovision.labeling.landcover import DynamicWorldLabeler

result = DynamicWorldLabeler().label(
    bbox=(...), date_range=("2024-06-01", "2024-06-30"),
    output_path="./labels/dynamic_world.tif",
)
```

Real, near-real-time (sub-weekly) global land cover from Google
Dynamic World — genuinely more current than WorldCover's annual
release, at the cost of being a real, per-scene classification rather
than a curated annual product. Runs via a real Google Earth Engine
asynchronous export task.

```{warning}
A real, severe bug was found and fixed here: the HTTP response status
was never checked before writing content to disk and declaring
success — the same class of bug as the WorldCover fallback above,
found independently in this different code path. A 403/404 response
could be silently written as a "real" label file. Fixed the same way.
```

```{note}
A `success: True` from the Earth Engine export path means the real
task was genuinely *submitted* to Earth Engine's queue — an honestly
weaker, different claim than "the data is ready." The real return
value reflects this: `output_path` is `"See Google Drive"` plus a
trackable `ee_task` ID, not a ready-to-use local file.
```
