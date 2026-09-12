# Building Footprint Labelers

Two real, independent open building-footprint datasets.

## Microsoft Buildings

```python
from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler

result = MicrosoftBuildingsLabeler().label(
    bbox=(-0.15, 51.47, -0.10, 51.52),
    output_path="./labels/ms_buildings.tif",
)
# {"success": True, "n_buildings": 1204, "output_path": "...", "source": "Microsoft"}
```

Pulls real, open building footprint polygons from Microsoft's global
dataset and rasterizes them to a real, aligned binary mask.

## Google Buildings

```python
from pygeovision.labeling.buildings import GoogleBuildingsLabeler

result = GoogleBuildingsLabeler().label(bbox=(...), output_path="./labels/google_buildings.tif")
```

Same real workflow, Google's real Open Buildings dataset instead —
real, independent coverage that can differ meaningfully from
Microsoft's in a given region.

```{warning}
Both previously returned `success: True` unconditionally, even when
`n_buildings` was 0 — a real, plausible outcome for a rural area with
no real footprints in either dataset. Fixed the same way as
[OSMLabeler](osm.md): `success` now genuinely reflects whether real
buildings were found.
```

```{seealso}
For per-pixel building *segmentation* from a real model (rather than a
lookup against these two pre-existing datasets), see the
[`building_footprints` pipeline](../pipelines/infrastructure.md).
```
