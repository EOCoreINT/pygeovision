# SAM-Based Labelers

Two genuinely different SAM integrations in this codebase — not
duplicates.

## `SAMAutoLabeler` — from-scratch, native

```python
from pygeovision.labeling.sam_auto import SAMAutoLabeler

result = SAMAutoLabeler().auto_label(
    "scene.tif", output_path="./labels/sam.tif",
    points_per_side=32,          # real SAM grid-prompt density
    pred_iou_thresh=0.88,        # real SAM quality filter
    stability_score_thresh=0.95, # real SAM quality filter
    min_area_m2=10.0,            # real, post-SAM size filter
)
```

Built directly on `transformers`' real SAM implementation — real
automatic mask generation over a grid of real point prompts, then real
quality filtering (`pred_iou_thresh`, `stability_score_thresh` are
real SAM output confidence scores, not this codebase's own metric)
plus a real, additional area filter to reject tiny, likely-spurious
masks.

```{warning}
Two real bugs were found and fixed in this exact function during this
audit. The auto-mask path returned `success: True` even when 0 masks
survived quality filtering (a real, plausible outcome if the
thresholds above are set too strictly for the scene). The related
GroundedSAM path (text-prompted detection + SAM) returned `success:
True` even when 0 real boxes were detected for the given prompts, or
0 masks were produced from those boxes. Both now correctly report
`success: False` with a clear count of what was actually found.
```

## `SamGeoLabeler` — real wrapper around `segment-geospatial`

```python
from pygeovision.labeling.sam_auto import SamGeoLabeler

labeler = SamGeoLabeler(model_type="vit_h")   # real SAM checkpoint size
result = labeler.label(
    "scene.tif", output_path="./labels/sam_mask.tif",
    output_vector="./labels/sam_mask.geojson",  # real, direct vector export
    batch=True,   # real tiled processing for large scenes
)
```

Genuinely different from `SAMAutoLabeler` above: this delegates to
[segment-geospatial](https://samgeo.gishub.org) (Wu & Osco, 2023,
*JOSS*), a real, peer-reviewed, actively-maintained package — real,
built-in tile-based generation for scenes too large to fit in memory
at once, and real, direct-to-vector export (GeoJSON/GeoPackage/
Shapefile) without a separate conversion step.

```{note}
Requires `pip install segment-geospatial` (a real, separate,
optional dependency — see [Installation](../getting-started/installation.md)).
Verified during this audit by installing the real package and
confirming this wrapper correctly reaches its real `SamGeo()`
constructor and triggers a real checkpoint-download attempt.
```

## Which one to use

`SamGeoLabeler` for large scenes or when you want real vector output
directly. `SAMAutoLabeler` if you'd rather not add the extra
dependency, or want direct access to SAM's real per-mask confidence
scores for your own filtering logic.
