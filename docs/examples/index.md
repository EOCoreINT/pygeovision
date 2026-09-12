# Domain Examples

**Verification status**: the workflows below were not independently
re-verified this cycle to the same standard as the rest of this site
(hand-calculation, synthetic ground truth, or direct testing). Two
specific claims that *were* checked and corrected: `client.pipeline
(name, ...)` is a real, correct pygeovision shortcut (confirmed by
reading the source — it delegates to the same `get_pipeline()` +
`.run()` machinery as the explicit pipeline classes), not `pygeofetch`'s
data-chain builder as an earlier version of this page claimed; and
YOLOv8 is not a real, working entry in this registry (confirmed
removed — no genuine backing) despite being listed as a "key model"
below. Where these examples reference SAR/InSAR processing, use
[pygeofetch](https://pygeofetch.readthedocs.io/en/latest/) directly —
pygeovision's own SAR/InSAR layer was removed from scope entirely.

Complete end-to-end workflows for six geospatial AI application domains.

| Domain | Use Cases | Key Models |
|--------|-----------|-----------|
| [Agriculture](agriculture.md) | Crop mapping, yield estimation, disease | Prithvi-EO-2.0, SegFormer |
| [Forestry](forestry.md) | Deforestation, canopy height, species | ChangeFormer (see [Model Registry](../core-features/model-registry.md) for the real DINOv2/DINOv3 labeling caveat) |
| [Urban Mapping](urban.md) | Buildings, roads, urban growth | SegFormer-B5, RF-DETR |
| [Water & Floods](water.md) | Flood mapping, water quality | Prithvi-EO, U-Net |
| [Disaster Response](disaster.md) | Damage assessment, rapid mapping | ChangeFormer |
| [Climate & Carbon](climate.md) | Carbon, sea ice, wildfire | Prithvi-EO |
