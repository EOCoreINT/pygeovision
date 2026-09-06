# Forestry

Four real pipelines, one of which honestly reports failure by design.

## `canopy_height`

**What it does:** Estimates real canopy height in meters, per pixel.

**How it actually works:** Runs a real regression model (not
classification — the output head predicts a continuous height value)
over the optical stack. If a real DEM is supplied via `dem_path=`, the
pipeline uses it for a real terrain-normalization step so canopy
height isn't confounded with ground elevation change across the
scene.

```python
result = CanopyHeightPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    dem_path="./terrain/dem.tif",   # optional but recommended
)
# result.stats: {"mean_height_m": ..., "max_height_m": ..., "canopy_cover_pct": ...}
```

**Honest limitation:** without a real DEM, height estimates on sloped
terrain will be systematically biased — the model can't distinguish
"tall tree on flat ground" from "short tree on a hillside" using
optical imagery alone.

## `tree_species`

**What it does:** Nothing, honestly — this pipeline returns
`success: False` unconditionally.

```python
result = TreeSpeciesPipeline(client).run(bbox=(...), date="2024-06")
print(result.success)   # False
print(result.error)     # "No real, verified tree-species classification model is available."
```

**Why:** tree-species classification from optical imagery alone is a
genuinely hard, real research problem — species that look visually
similar at 10m resolution require either hyperspectral data or a
real, specifically-trained model this codebase doesn't have. Rather
than wire in an unverified model and let it silently produce plausible
-looking but wrong species labels, this was left honestly
unimplemented. See [Contributing](../contributing.md) if you have a
real, verified model to add.

## `forest_fire`

**What it does:** A real, direct subclass of `wildfire_severity`
(see [Urban](urban.md) — wildfire severity is filed there since it
also covers non-forest burn scars) — same real dNBR-based severity
mapping, with forestry-specific default output labeling.

```python
result = ForestFirePipeline(client).run(
    bbox=(...), date_before="2023-06", date_after="2023-09",
    output_dir="./output",
)
```

## `deforestation`

**What it does:** Real bi-temporal forest-loss detection.

**How it actually works:** Runs a real segmentation model (or NDVI
threshold, depending on `method=`) independently on both dates to
produce a real forest/non-forest mask for each, then computes the
real set difference: pixels that were forest in the "before" mask and
are not forest in the "after" mask.

```python
result = DeforestationPipeline(client).run(
    bbox=(...), date_before="2020-01", date_after="2024-01",
    output_dir="./output", method="model",   # or "ndvi_threshold"
)
# result.stats: {"forest_loss_ha": ..., "forest_loss_pct": ..., "remaining_forest_ha": ...}
```

**Honest limitation:** a real multi-year gap between dates (as in the
example above) will also pick up real seasonal variation and any real
cloud/shadow contamination in either date's imagery unless both were
carefully chosen for comparable phenology and cloud cover. Shorter,
matched-season date pairs (e.g. both dry-season) give more reliable
results than arbitrary before/after dates.
