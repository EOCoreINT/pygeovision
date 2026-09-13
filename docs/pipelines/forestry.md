# Forestry

Two real pipelines with their own domain implementation, plus
`forest_fire`/`wildfire_severity` (a real subclass relationship worth
understanding), and `deforestation` — one of the 10 CLI-reachable
pipelines with a separate implementation, documented in
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md), not here.

```{note}
This page was substantially rewritten after checking every claim
directly against the real source. The previous version claimed a
`dem_path=` terrain-normalization parameter on `canopy_height` that
does not exist, and included a full `deforestation` section describing
a different implementation than the one actually used for that name.
```

## `canopy_height`

**What it does:** Real, per-pixel canopy height regression in meters.

**How it actually works:** Delegates directly to the real
`CHMv2Model.predict_canopy_height()` — the same real DINOv2-based
model documented in
[Foundation Models](../models/foundation-models.md) — over the
downloaded, radiometrically-corrected optical scene.

```python
from pygeovision.ai.pipelines import CanopyHeightPipeline

result = CanopyHeightPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
```

```{warning}
Corrected from an earlier version of this page: there is no real
`dem_path=` parameter or DEM-based terrain-normalization step in this
pipeline — checked directly against the source. `CanopyHeightPipeline
.run()` takes only `bbox`, `output_dir`, and `date`; it calls
`CHMv2Model().predict_canopy_height()` with no DEM integration at all.
```

**Real output:** `result.output_path` → `canopy_height.tif`. See
[Foundation Models](../models/foundation-models.md) for
`predict_canopy_height()`'s own real output structure
(`height_map`/`statistics`, including `mean_m`/`max_m`/`p95_m`/
`coverage_pct`).

---

## `tree_species`

**What it does:** Downloads and correctly preprocesses real imagery,
then honestly reports that it cannot classify it.

```python
from pygeovision.ai.pipelines import TreeSpeciesPipeline

result = TreeSpeciesPipeline(client).run(bbox=(...), date="2024-06")
print(result.success)      # False
print(result.output_path)  # the real, preprocessed scene -- not a species map
print(result.error)
# "No trained tree-species classifier exists in this codebase --
#  imagery was downloaded and correctly preprocessed but not
#  classified. output_path points to the preprocessed scene, not
#  a species classification."
```

```{note}
Corrected from an earlier version of this page, which paraphrased a
different error message than the real one shown above, and didn't
mention that `result.output_path` is genuinely set (to the
preprocessed scene) rather than `None`.
```

**Why this is honest rather than broken:** tree-species classification
from optical imagery alone is a genuinely hard, real research problem
— species that look visually similar at 10m resolution require either
hyperspectral data or a real, specifically-trained model this codebase
doesn't have. Rather than wire in an unverified model and let it
silently produce plausible-looking but wrong species labels, real
preprocessing runs (so the pipeline is ready for a real classifier the
moment one exists) but classification itself honestly fails. See
[Contributing](../contributing.md) if you have a real, verified model
to add.

---

## `forest_fire` and `wildfire_severity`

**What they do:** Real, published dNBR (differenced Normalized Burn
Ratio) burn-severity mapping — literally the same real implementation
under two names, not two different techniques.

```{important}
`ForestFirePipeline` is a real Python subclass of
`WildfireSeverityPipeline` with no overridden `run()` method at all —
just a different `name`/`description`. Calling either produces
identical real output from identical real code. This is documented
here once rather than described twice as if they were different.
```

**How it actually works:** A real, well-cited technique (Key & Benson,
2006): `NBR = (NIR − SWIR2) / (NIR + SWIR2)`, computed independently
for a real pre-fire and post-fire date, then `dNBR = NBR_pre −
NBR_post`. Classified into the real, published USFS/MTBS 4-class
severity scale (unburned / low / moderate / high), using the real,
standard thresholds (`0.10`, `0.27`, `0.66`).

```python
from pygeovision.ai.pipelines.domains import ForestFirePipeline  # or WildfireSeverityPipeline

result = ForestFirePipeline(client).run(
    bbox=(...), date_before="2023-06", date_after="2023-09",
    output_dir="./output",
)
print(result.stats)
# {"mean_dnbr": ..., "pct_unburned": ..., "pct_low_severity": ...,
#  "pct_moderate_severity": ..., "pct_high_severity": ...}
```

`date_before`/`date_after` are real, required parameters — omitting
either raises a clear error, since dNBR is inherently a before/after
comparison.

```{warning}
**Honest scope limitation, stated directly in the real source code**:
this covers burn-scar mapping (a real, bi-temporal comparison) — it
does **not** cover active-fire, real-time hotspot detection, despite
what an earlier, more general catalog description implied. Real-time
active fire detection needs a genuinely different technique (e.g.
MODIS/VIIRS thermal-anomaly products), not implemented here.
```
