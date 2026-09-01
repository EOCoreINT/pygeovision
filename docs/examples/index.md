# Domain Examples

**Verification status**: these examples were not independently re-verified
this cycle. Where they use `client.pipeline(name)`, note that this is
PyGeoFetch's data-processing chain builder, not an AI pipeline runner —
see [PyGeoVision Client](../api/pygeovision.md). Where they reference SAR
processing, use [pygeofetch](https://pypi.org/project/pygeofetch/) directly
instead — pygeovision's own SAR/InSAR layer was removed this cycle.

Complete end-to-end workflows for the six primary geospatial AI application domains.

| Domain | Use Cases | Key Models |
|--------|-----------|-----------|
| [Agriculture](agriculture.md) | Crop mapping, yield estimation, disease | Prithvi-EO-2.0, SegFormer |
| [Forestry](forestry.md) | Deforestation, canopy height, species | DINOv3 CHMv2, ChangeFormer |
| [Urban Mapping](urban.md) | Buildings, roads, urban growth | SegFormer-B5, YOLOv8 |
| [Water & Floods](water.md) | Flood mapping, water quality | Prithvi-EO, U-Net |
| [Disaster Response](disaster.md) | Damage assessment, rapid mapping | ChangeFormer, xBD |
| [Climate & Carbon](climate.md) | Carbon, sea ice, wildfire | Prithvi-EO, DINOv3 |
