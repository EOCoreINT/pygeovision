# Changelog

All notable changes to PyGeoVision are documented here.

## [2.1.5] — 2026-07-03

### Added
- **GeoAgent** — autonomous geospatial AI agent with 10 tools, heuristic + LLM planners,
  session memory, and streaming support.  Decides sensor/task/approach from natural language.
- **InSAR chain** — `pygeovision.insar`: interferogram, coherence, displacement,
  deformation rate, interpretation reports, visualization dashboard.
- **Visualization layer** — `pygeovision.viz`: `Map`, `RasterViewer`, `VectorViewer`,
  `ChangeViewer`, `TimeSeriesViewer`, split-view comparison.
- **Enterprise module** — RBAC, API key management, SSO stub, audit logging (JSONL),
  GDPR/SOC2 compliance checker.
- **NB08** — Complete InSAR project: Turkey earthquake co-seismic deformation +
  12-month post-seismic time-series + 4-class severity + fusion risk map.
- **NB09** — Accra Flood Intelligence Platform: Odaw Basin Ghana, historical flood
  frequency 2019-2024, FloodWatch Ghana alert payload.
- **MkDocs** — Material theme documentation site with full API reference.
- **Deployment** — Kubernetes manifests, Helm chart, Terraform for AWS/Azure/GCP,
  production Docker image, GitHub Actions CI/CD (3 workflows).

### Fixed
- **BUG 1** `validate_sar_georeference()` — detects identity-transform CRS corruption
  after PyGeoFetch reproject (pixel_width=1.0, origin at 0,0).
- **BUG 2** `verify_sar_downloads()` — catches partial downloads via tile-read test
  before any despeckle step is attempted.
- **BUG 3** `clip_sar_to_bbox()` — auto-reprojects WGS84 bbox to raster CRS before
  calling rasterio.mask (prevents "shapes do not overlap" error).
- `ChangeDetection` alias added accepting `model_variant='changeformer'` kwarg.
- `GeoSegDataset` module created (`pygeovision.training.data`).
- `PrithviTasks.LAND_COVER_CLASSES` and `CROP_CLASSES` exposed as class attributes.
- 9 test files: hard-coded container paths replaced with portable `__file__`-relative paths.
- 7 torch-dependent test files: `pytest.importorskip("torch")` added for clean skips.
- SAR notebooks: removed `client._pgf_bridge.sar_despeckle()` private API calls.
- Cell 2 dangling expression fixed in 6 of 7 uploaded SAR notebooks.

### Improved
- Heuristic planner decision matrix: 12/12 correct routes (sensor × task × approach).
- README: GeoAgent section, 34-notebook two-tier listing, v2.1.5 badges.
- Docs website: GeoAgent section, SAR-08/09 flagship cards, all 34 notebooks.
- 451 tests pass / 24 skipped (torch absent) / 0 failed.

## [2.1.5] — 2026-06-27

### Added
- Full PyGeoVision v2 platform unifying PyGeoFetch + GeoAI.
- 119 model architectures, 503 registered datasets.
- 25 standard project notebooks + 9 SAR domain notebooks.
- SAR preprocessing pipeline (S0-S9) with three production bug-fixes.
- Prithvi-EO-2.0 domain adaptation for SAR via 6-channel HLS mapping.
- DINOv3 SAR adapter with Gamma multiplicative speckle augmentation.
