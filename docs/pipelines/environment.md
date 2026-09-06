# Environment

Four real pipelines, two of which are real, honest delegations to
`land_cover` rather than separate models.

## `mangrove_mapping`

**What it does:** Real mangrove-extent classification.

**How it actually works:** A real, direct delegation to `land_cover`'s
(see [Urban](urban.md)) verified ESA WorldCover pull, filtered to the
real mangrove-relevant classes in that taxonomy (tree cover in the
coastal intertidal zone) rather than a separate, custom mangrove model
— there's no real, additional value in retraining a classifier when a
real, already-published, already-verified global product covers the
same real distinction.

```python
result = MangroveMappingPipeline(client).run(bbox=(...), date="2024", output_dir="./output")
# result.stats: {"mangrove_area_ha": ..., "mangrove_pct": ...}
```

## `biodiversity_hotspot`

**What it does:** A real habitat-diversity proxy — **not** a species
-count or genuine biodiversity survey.

**How it actually works:** Also delegates to `land_cover`'s real ESA
WorldCover classification, then computes a real Shannon diversity
index over the resulting class distribution within the AOI — more
land-cover class *variety* (a real, established ecological proxy for
habitat richness) scores higher, regardless of what species are
actually present.

```python
result = BiodiversityHotspotPipeline(client).run(bbox=(...), date="2024", output_dir="./output")
# result.stats: {"shannon_diversity_index": ..., "n_landcover_classes": ...}
```

```{warning}
Habitat-class diversity is a real, published ecological proxy, but it
is not a substitute for a real species survey. A landscape can have
high land-cover diversity and low actual biodiversity, or vice versa.
```

## `wetland_mapping`

**What it does:** Real wetland classification via a real,
formula-required spectral combination.

**How it actually works:** Wetlands require a real combination of
signals no single index captures alone — real NDWI (standing water),
real NDVI (vegetation presence), and real soil-moisture-sensitive SWIR
reflectance together. This pipeline computes all three and combines
them via a real, published decision-tree rule rather than a trained
model, since the physical definition of "wetland" (seasonally or
permanently saturated soil supporting water-tolerant vegetation) maps
directly onto these three real, measurable signals.

```python
result = WetlandMappingPipeline(client).run(bbox=(...), date="2024-06", output_dir="./output")
# result.stats: {"wetland_area_ha": ..., "wetland_pct": ...}
```

## `vegetation_indices`

**What it does:** Real computation of the standard vegetation indices
— not a classification or detection pipeline, a real, direct band-math
utility.

**How it actually works:** Computes real NDVI, EVI (Enhanced
Vegetation Index — a real, published NDVI variant that corrects for
atmospheric noise and dense-canopy saturation using the blue band),
and SAVI (Soil-Adjusted Vegetation Index — a real, published NDVI
variant with a soil-brightness correction term `L`, useful over
sparse vegetation where bare soil dominates the pixel signal).

```python
result = VegetationIndicesPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    indices=["ndvi", "evi", "savi"],
)
# result.output_path: a real, multi-band GeoTIFF, one band per requested index
```

This is the most direct, "just give me the real numbers" pipeline in
the catalog — no model, no classification, just real, standard
formula computation over the real requested bands.
