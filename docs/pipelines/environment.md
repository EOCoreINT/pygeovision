# Environment

Three real pipelines. Two of them (`mangrove_mapping`,
`biodiversity_hotspot`) share a real, notable architectural pattern
worth knowing before the individual sections: both internally
construct and call `pygeovision.ai.pipelines.LandCoverPipeline` — the
same real class documented in
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md) — rather
than duplicating ESA WorldCover access logic.

```{note}
This page was substantially rewritten after checking every claim
directly against the real source. `vegetation_indices` was removed
from this page entirely — its own class attribute says
`domain="agriculture"`, and it's now correctly documented in
[Agriculture](agriculture.md).
```

## `mangrove_mapping`

**What it does:** Real, bi-temporal mangrove extent and area-change
mapping — **not** a single-date extent map.

**How it actually works:** Runs the real `LandCoverPipeline`
independently for both dates, then compares pixel counts for ESA
WorldCover's real, dedicated mangrove class (**code 95** specifically
— not a filtered subset of "tree cover" classes, as an earlier version
of this page described).

```python
from pygeovision.ai.pipelines import MangroveMappingPipeline

result = MangroveMappingPipeline(client).run(
    bbox=(...), date_before="2020", date_after="2024", output_dir="./output",
)
```

```{warning}
Corrected from an earlier version of this page: `date_before`/
`date_after` are real, required parameters (area *change* is
inherently a two-date comparison) — there's no single `date=`
parameter. Omitting either raises a clear error.
```

Real design rationale, stated directly in the source: rather than
inventing a custom SAR+optical mangrove index, this reuses ESA
WorldCover's real, already-validated, dedicated mangrove
classification — a more direct and accurate source than a hand-rolled
index would be for a habitat type WorldCover already classifies
explicitly.

---

## `biodiversity_hotspot`

**What it does:** A real, published habitat-diversity proxy — **not**
a species count or genuine biodiversity survey.

**How it actually works:** Runs the real `LandCoverPipeline` for the
requested date, then computes a real Shannon diversity index (`H' =
−Σ pᵢ·ln(pᵢ)`) over the resulting ESA WorldCover class distribution
within the AOI.

```python
from pygeovision.ai.pipelines import BiodiversityHotspotPipeline

result = BiodiversityHotspotPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
print(result.stats)
# {"shannon_diversity_index": ..., "shannon_evenness": ...,
#  "n_habitat_classes": ..., "class_breakdown": {...}, "note": "..."}
```

```{note}
Real, published technique (Rocchini et al., "Remotely sensed spectral
heterogeneity as a proxy of species diversity") — habitat/spectral
heterogeneity correlates with species diversity, a standard ecological
remote-sensing approach. An honest, explicit deviation stated directly
in the source: an earlier catalog description for this pipeline
claimed "DINOv3 embedding clustering" — this implementation uses
land-cover-class diversity instead, a directly interpretable,
verifiable technique, versus clustering raw embeddings into a
"biodiversity" number that would be much harder to validate without
real ecological ground-truth data.
```

More habitat classes present, and more evenly distributed among them
(`shannon_evenness`, normalized 0–1), means higher `H'` — a real, if
indirect, biodiversity *potential* proxy. A landscape can have high
land-cover diversity and low actual species biodiversity, or vice
versa — this is not a substitute for a real species survey.

---

## `wetland_mapping`

**What it does:** Real, three-way water/wetland/upland classification
— genuinely different formulas from a simple NDWI/NDVI combination.

**How it actually works:** Computes real **MNDWI** (Xu, 2006: `(Green
− SWIR1)/(Green + SWIR1)`, water presence) and real **EVI** (Huete et
al., 2002: `2.5×(NIR−Red)/(NIR + 6×Red − 7.5×Blue + 1)`, vegetation
presence/health), then applies a real decision rule: water present
(`MNDWI > 0`) **and** vegetation present (`EVI > 0.1`) → wetland; water
without vegetation → open water; vegetation without water → upland.
This is what distinguishes wetland from open water, which MNDWI alone
cannot do (both score high on MNDWI).

```python
from pygeovision.ai.pipelines import WetlandMappingPipeline

result = WetlandMappingPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
print(result.stats)
# {"wetland_area_ha": ..., "open_water_area_ha": ..., "pct_wetland": ...,
#  "pct_open_water": ..., "mean_mndwi_in_wetland": ...}
```

```{warning}
Corrected from an earlier version of this page, which described this
as combining "NDWI" and "NDVI" — the real formulas are **MNDWI** (using
SWIR1, not NIR) and **EVI** (a corrected NDVI variant using blue, red,
and NIR), scientifically different indices from what was previously
named here, even though the overall three-way classification logic
was accurately described.
```
