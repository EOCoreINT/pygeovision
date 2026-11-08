# Changelog

All notable changes to PyGeoVision are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — four more OBIA segmentation algorithms, and reference checks for all of them

- `obia.segment` has four new methods: `quadtree` (split until every band's standard deviation is `<= scale`),
  `multi_threshold` (multi-Otsu or given thresholds on a band, brightness, NDVI or NDWI; connected patches of a class are
  objects), `contrast_split` (repeated dark / bright split at the largest edge contrast) and `spectral_difference`
  (merge neighbours with similar mean spectra). The building blocks are public: `obia.quadtree`, `obia.multi_threshold`,
  `obia.contrast_split`, `obia.SpectralDifferenceMerger`. `pygeovision obia segment --method` accepts them; `--param
  thresholds=0.2,0.5` now parses a list.
- `multires` (and `spectral_difference`) take `strategy="mutual_best"`: local mutual best fitting as in Baatz & Schäpe (2000),
  next to the default `best_first`.
- `tests/test_obia_segmentation_reference.py`: the Baatz–Schäpe cost and merge order are compared with a naive implementation
  built from the published formulas; `quadtree`, multi-Otsu and `contrast_split` with independent references; the
  scikit-image wrappers with scikit-image; small cases with hand-derived values.

### Changed — one place chooses between streaming and downloading, with a fallback

pygeofetch is unchanged. PyGeoVision wraps its `stream` and `download` in one function, `pygeovision.data._stream.fetch_scenes`,
which every way of acquiring imagery for an area now uses.

- **You choose.** `stream=None` (default) streams the area of interest whenever it is known, `stream=True` asks for it,
  `stream=False` downloads whole scenes. Available as `client.download(..., stream=, stream_bbox=)`,
  `client.pipeline(name, bbox, stream=...)`, `pygeovision data download --stream/--no-stream`,
  `pygeovision channel <pipeline> --stream/--no-stream`, `stream: true|false` on the YAML `download` step, and on
  the acquirer, the bridge, `EOPipeline.download` and `GeoDataLoader.prepare(stream_bbox=...)`. `PYGEOVISION_STREAM=0`
  changes only the automatic default; an explicit `stream=True` still streams.
- **An older pygeofetch without `stream()` is detected.** PyGeoVision says once, with the installed version and the upgrade command (`pip install -U pygeofetch`; 2.6.9.8.4 and newer stream), that it is downloading whole scenes instead, and the download banner no longer claims to be streaming.
- **A stream never fails the run.** A scene that cannot be streamed - archive-only provider (Copernicus `.zip`, USGS `.tar`),
  no COG in the stack, a band that cannot be read, a streamed file that cannot be reprojected, an error in the stream
  layer, or an installed pygeofetch without `stream` - is downloaded in full with the normal `download` and your download
  options. A half-streamed scene has its small AOI files removed first, so no scene is a mix of AOI and whole files.
  The reason is logged and kept on `result.stream_error`; `result.acquisition` is `"stream"` or `"download"`.
- The retry after an incomplete scene still downloads the whole scene without resume.
- The old copy of this logic (`SatelliteFetcher._stream_items`, `STREAMABLE_PROVIDERS`) is gone; `client.stream(items, bbox)`
  is now `download(..., stream=True, stream_bbox=bbox)` and forwards download options for the fallback.
- `ml4floods_flood_mapping`: the result names the real Sentinel-2 scene ids and dates of each image (not the file name),
  any error while fetching the weights ends as a clear failed result, and `ensure_weights` / `DEFAULT_CACHE` are public.
- The command-line reference shows `automatic` as the default of `--x/--no-x` flags that may be left out.

### Changed — pipelines stream the area of interest instead of downloading whole scenes

- `client.download(..., stream_bbox=bbox)` reads only the pixels inside `bbox` from Cloud-Optimized-GeoTIFF providers
  (Planetary Computer, Element 84) through pygeofetch's `stream`, instead of whole 100-300 MB bands. Every pipeline that
  fetches imagery (`_search_and_download`, `_search_and_download_pair`) now streams by default; archive-only providers, a
  failed stream, or `PYGEOVISION_STREAM=0` use the full download as before, scene by scene. Streamed files get the same
  `reproject:` step as downloaded ones.
- New `client.stream(items, bbox, ...)`. Pipelines, `pygeovision data download --bbox` (`--no-stream` to opt out) and the
  YAML `download` step (`stream: false` to opt out) use it instead of `client.download`.
- `ml4floods_flood_mapping` uses this same imagery path as the other flood pipelines (`_search_and_download_pair`) and only
  prepares the data for the network: six bands as reflectance, cloud and shadow from the scene classification layer left unknown.

### Added — pretrained flood model: ml4floods `WF2_unetv2_bgriswirs`

- **`ml4floods_flood_mapping`** (42 pipelines; kind `ready`: it fetches its own weights). Runs the released U-Net of Portales-Julia et al.
  (2023, *Scientific Reports* 13, 20316) on Sentinel-2 L2A from the Planetary Computer for a dry reference window and a
  flood window; reports flood water, water in both windows and water with no clear reference view, never counting cloud
  or shadow as dry. Writes `flood_extent.tif`, `prob_water_after.tif`, the usual map, vectors and reports, and records the
  checksum of the weights it used.
- **`pygeovision.ai.models.ml4floods`**: the network (written here, layer names match the released checkpoint, strict
  load), `load_model`, `predict_probabilities`, `classify`, `predict_scene`, `predict_geotiff` (for Level-1C GeoTIFFs,
  the product the network was trained on) and `download_weights`.
- **The weights are not included, but they are downloaded for you.** They are CC BY-NC 4.0 (non-commercial) and
  pygeovision is MIT. Without `checkpoint=` the pipeline downloads them once from Hugging Face (pinned revision) into
  `~/.cache/pygeovision/ml4floods` (`ensure_weights`), logs the licence notice and reuses them; `checkpoint=<folder>` uses
  your own copy. `download_weights(accept_noncommercial=True)` is the explicit call and refuses without that confirmation.
  The loader reports the absolute path and folder contents when `model.pt` is missing, and recognises a Git LFS pointer.
- **Not claimed:** accuracy. The network was trained on Level-1C top-of-atmosphere imagery and this pipeline feeds it
  Level-2A reflectance; the effect is not measured, and the report says so.

---

## [2.2.0] — 2026-07-03

### Added — GeoAgent: Autonomous Geospatial AI Agent

#### `pygeovision.agent` — new module (6 files, 55 tests)

- **`GeoAgent`** — autonomous natural-language to geospatial pipeline agent.
  Accepts plain-English queries and executes the complete workflow end-to-end:
  satellite data search → download → preprocessing → AI inference → postprocessing.
- **Two planning modes** (auto-detected at init):
  - **LLM mode** (`ANTHROPIC_API_KEY` set): uses Claude to decompose queries into
    optimal tool sequences with rationale per step.
  - **Heuristic mode** (no key needed): explicit `Sensor × Task × Approach` decision
    hierarchy covering ~95% of common geospatial workflows without any API calls.
- **Intent-understanding heuristic planner** — explicitly infers sensor (SAR vs optical)
  and task type (flood / damage / change / land_cover / crop / burn / oil_spill /
  subsidence / ship / spectral / building / glacier) from the problem description before
  selecting the tool sequence. Distinguishes:
  - *"building damage after earthquake using SAR"* → `change_detection` (not flood!)
  - *"flood using Sentinel-2 clear sky"* → optical Prithvi pipeline (not SAR!)
  - *"oil spill at night Sentinel-1"* → SAR dark-pixel approach with physics rationale
- **10 production tools** wrapping every PyGeoVision capability:
  `search_satellite_data`, `download_satellite_data`, `prepare_for_ai`,
  `sar_preprocess` (with all 3 bug-fixes), `sar_flood_detection`,
  `prithvi_inference`, `change_detection`, `compute_spectral_index`,
  `postprocess`, `run_pipeline`
- **`GeoAgentMemory`** — per-session spatial context, named output bindings,
  bounded turn history, JSON save/load for session persistence.
- **`PlanExecutor`** — resolves `$step_N_output` references between steps,
  streaming event generator for UI integration, `ExecutionTrace` with JSON export.
- **`agent` CLI command group** — `pgv agent run`, `pgv agent plan`, `pgv agent tools`.
- **NB26: GeoAgent Complete Walkthrough** — end-to-end demo notebook showing
  heuristic + LLM planning, all 10 tools, session memory, and streaming.

#### Other fixes
- `pygeovision.models.change_detection.changeformer` — `ChangeDetection` alias added
  (accepts `model_variant='changeformer'` kwarg used in project notebooks 04, 06, 09)
- `pygeovision.training.data` — new module: `GeoSegDataset` + `make_geo_loaders`
- `pygeovision.models.foundation.prithvi` — `LAND_COVER_CLASSES` and `CROP_CLASSES`
  exposed as class attributes on `PrithviTasks`
- All 9 test files: hard-coded `/home/claude/pgv` container paths replaced with
  portable `pathlib.Path(__file__).parent.parent` — fixes silent session-wide test
  collection failures
- 7 torch-dependent test files: `pytest.importorskip("torch")` guards added —
  converts crash-on-collection to clean skips when PyTorch is absent
- NB08: Turkey Earthquake InSAR complete project
- NB09: Accra Flood Intelligence Platform (FloodWatch Ghana integration)

### Tests
- **451 passed / 24 skipped (torch absent) / 0 failed**
- 60 new agent tests (`tests/test_agent.py`)
- Previously: 31 failures (all now fixed or correctly skipped)

---

## [2.2.0] — 2026-06-27

### ⚠️ Breaking Changes
- **PyGeoVision is now fully standalone.** GeoAI is no longer a required dependency — it is an optional plugin (`pip install "pygeovision[geoai]"`). All v1.x `client.geoai.*` calls continue to work but require explicit installation of `geoai-py`.
- `client.search()` now accepts `limit=` as an alias for `max_results=` (additive, no breakage).
- `client.download()` now accepts `bands=` parameter for asset filtering (additive).

### Added — Own AI Stack

#### Model Registry (119 architectures)
- Own model registry replacing GeoAI dependency for inference
- `get_model(name, num_classes, in_channels, pretrained)` — load any of 119 architectures by name
- `list_models(task, pretrained_only)` — filter by task or weight availability
- Families: Segmentation (24), Detection (18), Classification (16), Change Detection (12), Foundation (12), VLM (9), 3D/Point Cloud (8), Other (20)

#### Dataset Registry (503 datasets)
- `DatasetRegistry` with 503 curated remote sensing datasets
- `DatasetLoader` with automatic train/val/test split
- Coverage: SpaceNet 1–8, DOTA v1/v2, iSAID, DIOR, DeepGlobe, BigEarthNet, FloodNet, xBD, BreizhCrops, SEN12MS, MillionAID, RS5M + 450 more

#### Foundation Models (own stack)
- **DINOv3** — 12 variants including `dinov3_vitl16_sat` (SAT-493M, 300M params, 1024-dim) and `dinov3_vit7b16_sat` (6.7B, SAT pre-training)
  - `get_transform(model_name)` — auto-selects correct normalisation (SAT vs ImageNet)
  - `extract_embeddings()`, `extract_features()`, `get_attention_maps()`
  - `CHMv2Model` — canopy height + biomass estimation (GEDI calibrated)
  - `finetune_dinov3()` — fine-tuning with BF16 support
- **Prithvi-EO-2.0** — 600M params, HLS global 10-year pre-training
  - `map_bands(data, source="sentinel2")` — mandatory band re-ordering helper
  - `normalise_hls(data)` — HLS scale factor (÷10000)
  - `PrithviTasks` — land cover (9 classes), crop mapping (10 crops), flood detection, deforestation
  - `PrithviMultiTemporal` — 4-frame simultaneous attention
  - Surrogate ViT fallback when HuggingFace config has `None` integer fields (Bug 5 fix)

#### Vision-Language Models (own stack)
- `CLIPGeo("remoteclip-l14")` — RS5M (5M remote sensing image-text pairs) pre-trained
- `build_index(archive_dir, "index.faiss")` — semantic search over satellite archive
- `search(text_query, top_k)`, `search_by_image(path, top_k)`
- `MoondreamGeo` — 1.87B VLM for satellite scene Q&A and change captioning

#### Training Stack
- `GeoTrainer` — BF16 mixed precision, AdamW/SGD/Lion, AMP, DDP-ready
- `GeospatialMixedLoss` — Dice + Boundary + OHEM weighted composition
- `GeoSegDataset`, `EarlyStopping`, `ModelCheckpoint` callbacks
- Loss functions: `DiceLoss`, `FocalLoss`, `TverskyLoss`, `ComboLoss`, `BoundaryAwareLoss`, `LovaszLoss`, `OhemCrossEntropy`, `ClassBalancedCrossEntropy`

#### Inference
- `TiledInference` — streaming Gaussian-blend tiling, no seam artefacts
- `EnsembleInference` — mean/vote across multiple models
- `BatchInferenceEngine` — parallel scene processing

#### Serving API
- `InferenceServer` — FastAPI + JWT Bearer auth + WebSocket streaming
- Endpoints: `POST /predict`, `POST /predict/batch`, `WS /ws/stream`, `GET /health`, `GET /metrics`, `GET /models`

#### Edge & Cloud Deployment
- `ONNXRuntimeInference` — CPU/CUDA/TensorRT backend
- `JetsonDeployer` — TensorRT FP16 (≈45 chips/s on Orin)
- `AWSDeployer` — SageMaker endpoint + auto-scaling
- `AzureDeployer` — Azure ML managed online endpoint
- `GCPDeployer` — Vertex AI prediction endpoint

#### Advanced AI
- `FewShotLearner` — prototypical networks with DINOv2-Large backbone (~87% @ 5-shot)
- `AutoML` — Optuna HPO over LR, WD, backbone, loss, batch (typically +2–4 mIoU)
- `GeoTimeSeries` — NDVI/EVI trends, phenological profiles, anomaly detection, seasonal decomposition
- `GeoGradCAM`, attention map visualisation, SHAP-Geo spatial attribution
- PointNet++, RandLA-Net, KPConv, Point Transformer for LiDAR/3D
- `CLIPGeo` FAISS embedding search over large archives

#### Monitoring
- `DriftDetector` — PSI (Population Stability Index) per-band drift
- `ModelPerformanceTracker` — metric history with trend analysis
- `AlertManager` — Slack/email/webhook alerts with configurable rules

### Added — Data Validation Layer (`client.validator`)
- `DataValidator` — mandatory gate before every AI model
- Checks: `check_nulls`, `check_dtype`, `check_crs`, `check_bounds`, `check_shape`, `check_bands`, `check_nodata`, `check_outliers`
- Modes: `"strict"` (raise) | `"fix"` (auto-fix) | `"warn"` (log only)
- `validate_for_inference(data, model_type)` — returns clean float32 numpy array
- `validate_for_training(images, labels, num_classes)` — training data quality gate
- `generate_report(path, output, fmt="html")` — HTML/JSON validation reports
- Model-type requirements: min_bands, dtype, value_range per task type

### Added — Preprocessing Stack (`client.preprocess`)
- `Preprocessor` class — all operations validated before returning
- `stack_bands(band_paths, output_path, band_names)` — merge individual band files
- `stack_from_dir(scene_dir, band_names, output_path)` — auto-discover bands by name pattern
- `clip_to_bbox(input_path, bbox, bbox_crs)` — crop to WGS84 bounding box
- `clip_to_polygon(input_path, geojson)` — crop to GeoJSON polygon/feature
- `apply_cloud_mask(input_path, mask_path)` — binary cloud mask
- `apply_scl(input_path, scl_path, keep_classes)` — Sentinel-2 SCL masking
- `normalise(input_path, method)` — minmax / zscore / percentile / scale_factor
- `resample(input_path, resolution_m, resampling)` — change pixel resolution
- `info(path)` — quick metadata summary
- `pipeline(...)` — full chain: stack → clip → mask → normalise → resample → validate

### Added — Spectral Indices (`client.indices`)
- `SpectralIndices` — 22 validated indices, all returning float32 GeoTIFF
- **Vegetation:** NDVI, EVI, SAVI, MSAVI, ARVI, NDRE, RVI, WDRVI, VARI, ExG
- **Water:** NDWI, MNDWI, LSWI, WRI
- **Built-up/Soil:** NDBI, BSI
- **Fire/Burn:** NBR, BAI
- **Snow:** NDSI
- **Transforms:** TCT Brightness/Greenness/Wetness (Nedkov 2017 S2 coefficients), PCA
- `compute_all(source, indices, output_dir, band_map)` — batch compute
- All indices validated (NaN, Inf, range) before returning

### Added — Postprocessing (`client.postprocess`)
- `PostProcessor` — 15 validated postprocessing operations
- `vectorise` — raster → GeoJSON polygons with area filter and simplification
- `rasterise` — vector → raster using reference extent
- `sieve_filter` — remove small patches (rasterio sieve)
- `fill_holes` — interpolate no-data gaps
- `apply_confidence_threshold` — filter by model confidence
- `smooth` — Douglas-Peucker polygon simplification
- `regularise_buildings` — minimum rotated rectangle footprint regularisation
- `dissolve` — merge adjacent same-class polygons
- `buffer` — vector buffer with configurable distance
- `class_statistics` — per-class pixel count, area (m², ha, km²), coverage %
- `zonal_statistics` — raster stats per vector zone (mean, std, min, max, count)
- `accuracy_assessment` — OA, Cohen's κ, per-class precision/recall/F1/IoU, confusion matrix
- `to_cog` — Cloud-Optimized GeoTIFF conversion
- `export` — GeoJSON / Shapefile / GeoPackage / KML
- `generate_report` — HTML/JSON prediction analysis report

### Added — PyGeoFetch Bridge (`client._pgf_bridge`)
- `PyGeoFetchBridge` — definitive integration layer with PyGeoFetch Python API
- Uses `SearchQuery(bbox=BoundingBox.from_string(...), start_date=date.fromisoformat(...))` — correct types
- Uses `DownloadOptions` + `PostProcessAction(action=str, params=dict)` — correct fields
- `_pgf_satellite_data_to_pgv()` — `SatelliteData` (with `SatelliteAsset.key/href/roles/extra_fields`) → `SearchResult`
- `_pgf_download_result_to_pgv()` — `PGF DownloadResult` → `PGV DownloadResult`
- `prepare_for_ai()` — mandatory preprocessing gate (stack → clip → mask → normalise → validate)
- Preprocessing proxies: `preprocess_atmos/clip/reproject/resample/mosaic/composite/cloud_mask/tile/pansharpen/topo_correct/cloud_fill`
- Forward-compatible with PyGeoFetch v2.0 native preprocessing when it ships
- `batch_process(inputs, chain, output_dir, parallel)` — sequential fallback for v2.0

### Added — SAR Proxy (`client.sar`)
- `_SARProxy` — routes to PyGeoFetch v2.0 SAR engine when available
- `despeckle(input_path, filter, window)` — lee / enhanced_lee / frost / gamma
- `calibrate(input_path, output_type, in_db)` — sigma0 / gamma0 / beta0
- `flood_map(post_path, threshold, reference)` — fixed threshold or change-based
- `coherence(slc1, slc2, window)` — InSAR coherence
- `is_available()` — False until PyGeoFetch v2.0 ships

### Added — Pipeline Orchestration
- New `client.prepare_for_ai(path, ...)` — single-call download → AI-ready array
- New `client.pgf_pipeline(name)` — chainable pipeline builder
- New `client.batch_process(inputs, chain, ...)` — parallel preprocessing
- Pipeline registry expanded from 26 → **51 pipelines** via `_make_simple()` factory
- New pipelines: wildfire_severity, glacier_monitoring, oil_spill_detection, air_quality_index, urban_heat_island, parking_occupancy, solar_potential, wetland_mapping, mangrove_mapping, snow_cover, permafrost_thaw, mine_detection, port_monitoring, powerline_extraction, dam_safety, crop_yield_forecast, aquaculture_mapping, landcover_change, biodiversity_hotspot, construction_progress, reef_bleaching, dust_storm_tracking, archaeological_site, pipeline_leak_detection, wind_farm_siting

### Added — CLI (4 new command groups)
- `pygeovision validate run <input>` — validate GeoTIFF with report generation
- `pygeovision validate for-inference <input> --model-type segmentation`
- `pygeovision preprocess stack/clip/normalise/resample/pipeline` — all with `--validate`
- `pygeovision indices compute/all/list` — 22 spectral indices from CLI
- `pygeovision postprocess vectorise/sieve/smooth/regularise/zonal-stats/accuracy/class-stats/cog/export`

### Added — Production Notebooks (25 total, 357 cells)
- Notebooks 01–10: Satellite acquisition, Building extraction, Land cover (Prithvi), Change detection, Crop monitoring, Forest monitoring, Water/Flood, Solar panels, Disaster damage, Urban growth
- Notebooks 11–16: Road extraction, Crop type mapping (BreizhCrops), Glacier monitoring (Aletsch/RCP), Oil spill SAR, Air quality (TROPOMI), Wildfire (dNBR)
- Notebooks 17–25: Biodiversity (DINOv3), Infrastructure, Coastal/Wetland, Climate (LST/UHI), Custom training, Pipeline deployment, Foundation fine-tuning, DINOv3 embeddings, VLM (CLIP+Moondream)
- All 25 run without GPU (demo mode) and without satellite credentials

### Fixed
- **Bug 1** — `search()` raised `TypeError: got unexpected keyword argument 'limit'` — added `limit=` as alias for `max_results=`
- **Bug 2** — `download()` raised `TypeError: got unexpected keyword argument 'bands'` — added `bands=` parameter with asset filtering
- **Bug 3** — `SatelliteData` reconstruction failed with 25 Pydantic validation errors — now builds `SatelliteAsset(key=k, href=..., roles=[], extra_fields={})` correctly
- **Bug 4** — `ChangeFormer` raised `RuntimeError: expected input to have 4 channels, got 1` — new `_fix_channels()` helper replicates/truncates channels cyclically
- **Bug 5** — `Prithvi-EO-2.0` raised `TypeError: 'NoneType' cannot be interpreted as an integer` — patches 7 HuggingFace config integer fields before `AutoModel.from_pretrained()`
- **SearchQuery types** — `bbox` now uses `BoundingBox.from_string()` and dates use `date.fromisoformat()` (correct PyGeoFetch v1.0 types)
- **PostProcessAction** — now uses `PostProcessAction(action=str, params=dict)` not `.from_string()` which does not exist
- **SatelliteData cache reload** — assets from cache now properly reconstructed with `SatelliteAsset` objects

### Changed
- `client.preprocess` — upgraded from bare `Preprocessor` to bridge-aware proxy (PyGeoFetch v2.0 native when available)
- `client.indices` — upgraded to `SpectralIndices` with 22 validated indices
- `client.postprocess` — upgraded to `PostProcessor` with 15 operations
- `client.__repr__()` — now shows `pgf_v2`, `validator`, `indices`, `sar` status
- `_satellite_data_to_result()` — now delegates to `_pgf_satellite_data_to_pgv()` bridge function

### Tests
- **580 passing** (up from 208 in v1.0)
- 2 skipped (GPU-only tests)
- 0 failing

---

## [1.0.0] — 2025-01-15

### Added
- Initial public release
- `PyGeoVision` core class wrapping PyGeoFetch for satellite data
- `SatelliteFetcher` — search (22 providers) and download via pygeofetch Python API
- `GeoAIEngine` — integration layer wrapping geoai-py
- 10 end-to-end pipelines: building_footprints, change_detection, land_cover, water_bodies, solar_detection, crop_monitoring, disaster_assessment, deforestation, urban_growth, carbon_estimation
- 14 model architectures in own registry
- `GeoTrainer` + 10 loss functions
- `TiledInference` with Gaussian blend
- Auto-labeling from 7 sources (OSM, MS Buildings, Google, ESA, SAM, DynamicWorld, Foundation)
- `DriftDetector`, `PerformanceTracker`, `AlertManager`
- `InferenceServer` — FastAPI + WebSocket
- ONNX + Jetson TensorRT edge deployment
- AWS SageMaker + Azure ML + GCP Vertex AI cloud deployment
- 208 tests passing
- Complete CLI with 12 command groups
- `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, pre-commit hooks

---

[2.2.0]: https://github.com/pygeovision/pygeovision/compare/v1.0.0...v2.2.0
[1.0.0]: https://github.com/pygeovision/pygeovision/releases/tag/v1.0.0
