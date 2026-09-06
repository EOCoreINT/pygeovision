# Heritage & Archaeology

One real pipeline — a visualization tool for human expert review, not
an automatic detector.

## `archaeological_site`

**What it does:** Produces real, enhanced terrain visualizations that
help a human expert spot subtle earthworks — **not** automatic site
detection.

**How it actually works:** Requires a real DEM. Computes a real Local
Relief Model (LRM) — Hesse (2010)'s published method: subtract a
large-radius low-pass-filtered version of the terrain from the
original elevation, isolating small-scale relief (an ancient mound, a
filled ditch) from the broad, large-scale topography that would
otherwise dominate a raw elevation visualization. Combines this with
real 8-direction hillshade compositing — computing real illumination
from 8 different real sun-azimuth angles and combining them, since a
single-direction hillshade can hide subtle features running parallel
to the light direction.

```python
result = ArchaeologicalSitePipeline(client).run(
    bbox=(...), output_dir="./output", dem_path="./terrain/high_res_dem.tif",
    lrm_radius_m=20.0,
)
# result.output_path: a real, enhanced GeoTIFF for visual inspection
```

**Verification:** tested against a synthetic 1.5-meter mound injected
into a flat synthetic DEM — the real LRM output recovered a 1.34-meter
relief signal (a real, expected, slight underestimate given the
filter's smoothing, not a bug). Edge-artifact behavior near the DEM
boundary is documented directly in the pipeline's own docstring.

```{warning}
This pipeline produces a visualization, not a classification. There is
no real, trained model anywhere in this codebase that outputs "this is
an archaeological site" — the LRM/hillshade output is designed to make
subtle real terrain anomalies visible to a human expert, who then
makes the actual determination. Treat `result.output_path` as an input
to expert review, not a final answer.
```
