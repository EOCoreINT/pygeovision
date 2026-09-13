# The 10 CLI-Reachable Pipelines

These 10 pipelines are genuinely different from the other 39 covered
elsewhere on this site: they're the only ones reachable through the
CLI's `channel` command, and — this matters — they run **completely
separate, real implementations** (`pygeovision.ai.pipelines`, not
`pygeovision.ai.pipelines.domains`). `get_pipeline()` checks a local
registry first; for exactly these 10 names, that registry maps to
`None`, an explicit marker meaning "fall through to this file." See
[Architecture](../architecture.md) for why two implementations exist
at all.

This split matters practically, not just architecturally: several
real, documented bugs and their fixes live only in this file's code
comments, and the honest limitations here are specific to *this*
implementation, not the domain-page one covering the same name for
other tasks in the broader catalog.

```{note}
If you've read the `crop_monitoring`, `land_cover`, `water_bodies`,
etc. entries elsewhere and something here looks different — that's
not an inconsistency to reconcile. These are the real, actually-used
implementations for these 10 specific names.
```

## Shared machinery every pipeline here uses

### `_search_and_download` / `_search_and_download_pair`

Every pipeline here goes through one of these two real
`BasePipeline` methods rather than calling `pgv.search()`/
`pgv.download()` directly. The pair variant is real, non-trivial
machinery: two different scenes (different dates, different real
acquisition footprints) each get independently reprojected and
cropped, then the second is explicitly re-aligned to the first's exact
pixel grid — confirmed necessary from a real production failure where
two correctly-processed images still didn't align pixel-for-pixel,
silently corrupting every downstream bi-temporal comparison.

### `_run_model`

Real dispatch to `pygeovision.ai.inference.tiled_inference.TiledInference`
— note this is the *other* tiled-inference engine from the one covered
in [Tiled Inference](../inference/tiled-inference.md), reachable
through the CLI's `ai infer` command; see
[Architecture](../architecture.md) for the full three-engine picture.
Accepts an `image_path_b` parameter for real two-image (siamese/
change) models, dispatching to `TiledInference.run_pair()` instead of
`run()` — a real fix for a real production crash where two-image
models (`forward(t1, t2)`) were being called with only one tensor,
with no mechanism to supply the second at all.

### `real_pixel_area_ha`

Used by `carbon_estimation` (and anywhere else real area statistics in
hectares matter) — a real, geodesically-accurate per-pixel area
calculation, not `abs(resolution² ) / 10000`. That naive formula is
only correct when the raster's resolution is genuinely in meters; this
pipeline's real, standard output CRS is EPSG:4326 (degrees), so
treating pixel width in degrees as if it were meters would be wrong by
roughly `111,320²` — confirmed directly: an early test scene produced
"4.7 billion vehicles per km²" before this fix. For a geographic CRS,
this uses `pyproj.Geod` for real geodesic polygon area (accounting for
Earth's curvature and latitude-dependent longitude compression); for
an already-projected, meters-based CRS, the simple formula is
genuinely correct and used directly.

---

## `change_detection`

Real bi-temporal change detection — a real siamese U-Net or
ChangeFormer over two independently-acquired, aligned Sentinel-2
scenes.

```python
from pygeovision.ai.pipelines import ChangeDetectionPipeline

result = ChangeDetectionPipeline(client).run(
    bbox=(2.2, 48.8, 2.4, 48.9),
    output_dir="./output",
    date_before="2022-06", date_after="2024-06",
    model="siamese_unet",          # or "changeformer"
    cloud_cover_max=25.0,
    bands=("red", "green", "blue"),  # in_channels derived automatically
    num_classes=2,
    tile_size=512, overlap=64,
)
```

```bash
pygeovision channel change_detection --bbox 2.2 48.8 2.4 48.9 \
    --date-before 2022-06 --date-after 2024-06 --output ./output/
```

**Real parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `model` | `"siamese_unet"` | Or `"changeformer"` — any real model whose `forward()` accepts two images |
| `bands` | `("red","green","blue")` | Change if your model expects different real input (e.g. add NIR); `in_channels` is derived automatically |
| `tile_size`/`overlap` | 512/64 | Different real models have different receptive fields/memory footprints |

**Real output:** `result.output_path` → `change_mask.tif`.
`result.metadata` carries both real dates and the model/bands/classes
actually used.

---

## `land_cover`

The only pipeline here with a real, three-way dispatch: two real
labeling sources that need no model at all, or any real segmentation
model.

```python
from pygeovision.ai.pipelines import LandCoverPipeline

# Real ESA WorldCover lookup -- no inference, a real 2020/2021 product
result = LandCoverPipeline(client).run(bbox=(...), date="2023-06", source="worldcover")

# Real Google Dynamic World -- near-real-time, via Earth Engine
result = LandCoverPipeline(client).run(bbox=(...), date="2023-06", source="dynamic_world")

# Real segmentation model instead of either lookup
result = LandCoverPipeline(client).run(bbox=(...), date="2023-06",
                                        source="segformer_b2", num_classes=11)
```

```bash
pygeovision channel land_cover --bbox ... --date 2023-06
```

`source="worldcover"`/`"dynamic_world"` dispatch to the real
`ESAWorldCoverLabeler`/`DynamicWorldLabeler` covered in
[Labeling](../labeling/land-cover.md) — real, pre-existing global
products, not a model run by this pipeline. Any other string is
treated as a real model name and dispatches through `_run_model`
instead — `num_classes`/`bands` only apply in this branch.

```{note}
A real, confirmed bug fixed here: `bands` was previously accepted in
this method's real signature but silently discarded in the
model-dispatch branch specifically — never actually passed to the
download step, so changing it had zero effect on what the model
received regardless of what was requested. Fixed to pass through
correctly.
```

---

## `building_footprints`

Real building segmentation — background/building binary mask by
default.

```python
from pygeovision.ai.pipelines import BuildingFootprintsPipeline

result = BuildingFootprintsPipeline(client).run(
    bbox=(...), date="2024-06", model="unet_resnet50",
    cloud_cover_max=15.0, num_classes=2,
)
print(result.stats["building_coverage"])  # real fraction of pixels classified as building
```

```bash
pygeovision channel building_footprints --bbox ... --date 2024-06
```

```{note}
The identical `bands`-silently-discarded bug as `land_cover` above was
found and fixed here too, independently — confirmed by direct testing
before the fix (changing `bands` had no real effect on the downloaded
imagery).
```

---

## `crop_monitoring`

Real multi-class crop segmentation from a single date.

```python
from pygeovision.ai.pipelines import CropMonitoringPipeline

result = CropMonitoringPipeline(client).run(
    bbox=(...), date="2023-06",
    crop_classes=["maize", "wheat", "soy"],  # num_classes = len(crop_classes) + 1 (background)
    model="segformer_b2",
)
```

```bash
pygeovision channel crop_monitoring --bbox ... --date 2023-06
```

Without `crop_classes`, defaults to a real, generic 10-class head. As
with every other real model-backed pipeline here, class *labels* are
whatever the chosen model's real training distribution actually
supports — this pipeline doesn't ship a universal crop taxonomy.

---

## `disaster_assessment`

Real bi-temporal damage-severity segmentation.

```python
from pygeovision.ai.pipelines import DisasterAssessmentPipeline

result = DisasterAssessmentPipeline(client).run(
    bbox=(...), pre_date="2024-01", post_date="2024-02",
    disaster_type="generic",  # metadata only -- doesn't change the real algorithm
    model="siamese_unet", num_classes=4,  # e.g. none/minor/major/destroyed
)
```

```bash
pygeovision channel disaster_assessment --bbox ... --date-before 2024-01 --date-after 2024-02
```

`disaster_type` is recorded in `result.metadata` but doesn't change
the real processing — the same siamese-network approach handles any
disaster type; there's no real, disaster-specific model variant
selection happening here.

---

## `deforestation`

Real bi-temporal forest-loss detection, defaulting to ChangeFormer.

```python
from pygeovision.ai.pipelines import DeforestationPipeline

result = DeforestationPipeline(client).run(
    bbox=(...), baseline_year="2020", analysis_year="2024",
    model="changeformer", num_classes=3,
)
```

```bash
pygeovision channel deforestation --bbox ... --date-before 2020-07 --date-after 2024-07
```

Both real dates are anchored to July (`f"{year}-07"`) for both years —
a real, deliberate choice to compare the same real seasonal window,
avoiding a false "change" signal from comparing, say, a leaf-on summer
scene against a leaf-off winter one.

---

## `urban_growth`

Real bi-temporal impervious-surface change, using **Landsat**, not
Sentinel-2.

```python
from pygeovision.ai.pipelines import UrbanGrowthPipeline

result = UrbanGrowthPipeline(client).run(
    bbox=(...), start_year="2018", end_year="2024",
    model="siamese_unet", num_classes=2,
)
```

```bash
pygeovision channel urban_growth --bbox ... --date-before 2018-06 --date-after 2024-06
```

```{note}
A deliberate, real choice worth knowing: this pipeline explicitly
requests `collections=["landsat-c2-l2"]`, not Sentinel-2 — Landsat's
longer historical archive and more consistent multi-year radiometric
calibration make it the better real choice for long-horizon urban
growth comparisons (2018→2024 here), even though Sentinel-2 has finer
spatial resolution.
```

---

## `water_bodies`

Real surface-water mapping — a real, direct NDWI formula by default,
or any real segmentation model.

```python
from pygeovision.ai.pipelines import WaterBodiesPipeline

# Real McFeeters NDWI (default) -- no model, no training needed
result = WaterBodiesPipeline(client).run(bbox=(...), date="2024-06", method="ndwi")
print(result.stats["water_coverage"])

# Or a real trained segmentation model instead
result = WaterBodiesPipeline(client).run(bbox=(...), date="2024-06",
                                          method="unet_resnet50")
```

```bash
pygeovision channel water_bodies --bbox ... --date 2024-06
```

`method="ndwi"` computes the real McFeeters (1996) formula directly —
`NDWI = (Green - NIR) / (Green + NIR)`, thresholded at `0.3` — from
correctly-requested green+NIR bands. Any other string is treated as a
real model name.

```{warning}
A real, severe, confirmed bug was fixed here: NDWI was previously
requested via a `post_process=[...,"ndwi"]` step that never actually
ran for any real multi-asset scene — a different, earlier code path
(`_search_and_download`'s internal band-stacking) always intercepted
first, so `ndwi = src.read(1)` was silently reading raw **red-band
reflectance** and thresholding *that* as if it were NDWI. Real water
has low red reflectance, so this happened to produce plausible-looking
results in many cases — but bright bare soil or urban surfaces can
exceed a raw red-reflectance threshold of 0.3 and get wrongly flagged
as water. Fixed to request the real green+NIR bands directly and
compute the genuine formula.
```

---

## `solar_detection`

Real solar-panel segmentation, with real, deliberately different
tiling.

```python
from pygeovision.ai.pipelines import SolarDetectionPipeline

result = SolarDetectionPipeline(client).run(
    bbox=(...), date="2024-06", cloud_cover_max=5.0,
    model="unet_efficientnet_b4",
)
print(result.stats["panel_coverage"])
```

```bash
pygeovision channel solar_detection --bbox ... --date 2024-06
```

```{note}
Real, deliberate default: `tile_size=256, overlap=32` — smaller than
every other pipeline here's default 512/64. Solar panels are small,
fine-grained features; a smaller real tile keeps more of each panel's
relative size within a single inference window rather than diluting
it across a much larger 512px tile.
```

---

## `carbon_estimation`

An honestly-labeled proxy, not a calibrated carbon model — and worth
reading the caveat before using the numbers for anything real.

```python
from pygeovision.ai.pipelines import CarbonEstimationPipeline

result = CarbonEstimationPipeline(client).run(
    bbox=(...), date="2024-06",
    agb_scale=50.0,        # generic, uncalibrated allometric constant
    agb_max=500.0,         # Mg/ha clip ceiling
    carbon_fraction=0.47,  # IPCC default biomass-to-carbon ratio
)
print(result.stats)
# {"mean_carbon_mg_ha": ..., "total_area_ha": ..., "total_carbon_mg": ...}
```

```bash
pygeovision channel carbon_estimation --bbox ... --date 2024-06
```

```{warning}
This computes `agb = agb_scale * NDVI²`, `carbon = agb * carbon_fraction`
— a real, simple, generic NDVI-based allometric proxy, **not** a
trained above-ground-biomass model. Checked directly: no real AGB/
carbon model exists anywhere in this codebase's registry to swap in —
this isn't a placeholder awaiting a trivial upgrade. The default
`agb_scale=50.0` is generic and not calibrated for any specific biome.
Override it with a real, locally-fitted value if you have real ground
-truth biomass data for your region; otherwise, treat the output as a
rough, uncalibrated order-of-magnitude estimate, not a number to report
as-is.
```

```{note}
The identical real NDVI-misread bug as `water_bodies` above was found
and fixed here too: NDVI previously came from a `post_process` step
that never actually ran, silently substituting raw red-band
reflectance for NDVI in every real carbon estimate this pipeline had
ever produced. Fixed to request real red+NIR bands directly.
```
