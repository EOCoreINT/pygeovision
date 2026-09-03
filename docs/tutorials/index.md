# Tutorials

**Verification status — a major issue found and fixed this cycle, worth
understanding even now that it's resolved**: checking the 22 Jupyter
notebooks in `docs/tutorials/*.ipynb` this cycle found three widespread,
confirmed-broken patterns, not merely "unverified" ones. All three are
now fixed across all 22 notebooks.

1. **`client.geoai.*`** — a namespace that does not exist at all
   (confirmed directly: `hasattr(client, 'geoai')` is `False`). Found in
   13 notebooks, 20+ individual occurrences. Every one replaced with the
   real, verified equivalent — `client.segmentation.*`,
   `client.detection.*`, `client.change.*`, `client.classification.*`
   (real client proxies), direct model imports
   (`pygeovision.models.foundation.dinov3.DINOv3Backbone`,
   `pygeovision.models.foundation.prithvi.load_prithvi_hf`,
   `pygeovision.advanced.vlm.moondream_geo.MoondreamGeo`), or a real
   pipeline class, depending on what each specific call needed.
2. **`client.pipeline(name, bbox=..., date=...)`** used as if it were an
   AI pipeline runner — it is not; it's PyGeoFetch's chainable
   *data-processing* builder. Found in 10 notebooks, 30+ occurrences
   (`05_data_pipeline_orchestration.ipynb` had the most by count, but
   turned out to be the *least* broken by proportion — see below). Every
   one replaced with the real pipeline class
   (`from pygeovision.ai.pipelines import XPipeline` or
   `pygeovision.ai.pipelines.domains.XPipeline`) called directly.
3. Several notebooks' introductory or labeling cells falsely claimed
   PyGeoVision is built on a separate `GeoAI` package and an
   `EarthNets` dataset source — both false. PyGeoVision is independent
   of `geoai-py` (see [FAQ](../faq.md)), and this documentation found
   no evidence of an "EarthNets" dependency anywhere in the real
   codebase — "EarthNets" mislabels on pygeovision's own, real dataset
   registry were corrected across 6 notebooks.

**A real lesson from fixing `05_data_pipeline_orchestration.ipynb`**:
it had the highest raw count of the broken `client.pipeline()` pattern,
which looked like it would need a near-total rewrite. Reading it fully
before touching anything showed the opposite: 15 of its 17 cells were
already using a different, genuinely real system correctly
(`client.create_pipeline()` → `DataPipeline.search().filter().download()...`,
`client.run_pipeline_yaml()`, `client.data.schedule_pipeline()`, and
more — every method and parameter individually checked against the
real source before trusting it). Only the notebook's final two cells
had the actual broken pattern. High occurrence counts don't reliably
predict how broken something is — reading the real structure first,
rather than assuming from a grep count, is what found this.

**Real gaps found and honestly documented, not papered over**: fixing
these revealed a few real capability gaps in the codebase itself, not
just naming problems --- `client.segmentation.water()` is a real NDWI
threshold, not the trained deep-learning model some notebooks implied;
`TreeSpeciesPipeline` honestly returns `success=False` (no trained
classifier exists); a real sliding-window geo-referenced captioning
method doesn't exist for Moondream; and `TilingEngine` tiles a single
raster but doesn't itself pair image/label tiles, augment, or split
train/val/test the way one notebook implied. Each of these is called
out explicitly in its notebook rather than quietly worked around.

For SAR/InSAR-flavored notebook content: pygeovision's own SAR/InSAR
layer was removed this cycle — see [Architecture](../architecture.md)
for calling [pygeofetch](https://pypi.org/project/pygeofetch/) directly
instead.

**Not done this cycle**: actually executing these notebooks against
real data and real network access. Every fix here was verified by
reading the real source code and confirming each replacement class,
method, and parameter name exists and matches — a real, substantive
verification standard, but not the same as running the notebook
end-to-end against live satellite data. Treat that as the next layer
of verification, not yet done.
