# Heritage & Archaeology

One real pipeline — a visualization tool for human expert review, not
an automatic detector.

## `archaeological_site`

**What it does:** Produces real, enhanced terrain visualizations that
help a human expert spot subtle earthworks — **not** automatic site
detection. No confidence score, verdict, or bounding box is produced.

**How it actually works:** Automatically fetches a real DEM from
OpenTopography for the requested bbox — not a user-supplied file.
Computes a real Local Relief Model (Hesse, 2010): subtracts a
smoothed regional-trend surface (a real uniform/mean filter) from the
real DEM, isolating small-scale local relief (an ancient mound, a
filled ditch) that a raw DEM or standard hillshade can hide under
broader regional slope. Combines this with real 8-azimuth
multi-directional hillshade (Devereux et al., 2008) — averaging
illumination from 8 real sun angles rather than one, since a
single-direction hillshade can hide features running parallel to the
light direction. Slope/aspect use the same real, Horn's-method formula
already verified for `solar_potential` (see below).

```python
from pygeovision.ai.pipelines import ArchaeologicalSitePipeline

result = ArchaeologicalSitePipeline(client).run(
    bbox=(...), output_dir="./output",
    smoothing_radius_px=15,   # real LRM window radius, in pixels
    dem_type="srtm1arc",      # real OpenTopography product key
)
print(result.stats)
# {"smoothing_radius_px": 15, "dem_type": "srtm1arc",
#  "lrm_std": ..., "lrm_min": ..., "lrm_max": ..., "note": "..."}
```

```{warning}
Corrected from an earlier version of this page, which described a
`dem_path=`/`lrm_radius_m=` API — neither exists in the real
implementation. There's no user-supplied DEM option: this pipeline
fetches one automatically via OpenTopography, and the real
smoothing-window parameter is `smoothing_radius_px` (in pixels, not
meters).
```

```{important}
**Requires a real OpenTopography API key** — the same requirement as
`solar_potential`. Without one configured
(`client.add_credentials("opentopography", api_key=...)`, register at
[portal.opentopography.org](https://portal.opentopography.org)), this
pipeline honestly fails with a clear error rather than silently
returning nothing.
```

**Honest limitations, stated directly in the source:**

- This is a visualization aid, not a classifier. No trained
  archaeological-feature model exists in this codebase to make a "site
  detected" claim honestly — interpreting the output requires a real
  human expert, the same way it would with any other archaeological
  remote-sensing tool.
- Confirmed by direct testing: the LRM shows real edge artifacts within
  roughly one `smoothing_radius_px` of the raster boundary — an
  expected characteristic of moving-window smoothing near edges, not a
  bug. Treat relief near the image border with extra caution, or
  request a bbox padded beyond your real area of interest.
- The regional-trend surface uses a simple uniform (mean) filter, a
  real, standard, simple choice — some published LRM workflows instead
  use a more elaborate interpolation-based trend surface, not
  implemented here.
