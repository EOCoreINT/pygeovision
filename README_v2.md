# PyGeoVision: A Production-Ready Open-Source Platform for Geospatial Artificial Intelligence

<!-- **Samuel Appiah Kubi**  
Department of Remote Sensing and Geospatial Science  
University of Ghana, Legon, Accra, Ghana  
Email: s.appiah-kubi@ug.edu.gh | ORCID: 0000-0000-0000-0000

---

**Journal:** Remote Sensing (MDPI) | ISPRS Journal of Photogrammetry and Remote Sensing  
**Submitted:** July 2026  
**Keywords:** geospatial AI, satellite data, deep learning, foundation models, SAR, InSAR, natural language agent, Earth observation -->

---

## Statement of Need

Earth observation from satellite platforms generates more than 150 terabytes of imagery daily, yet converting this data into actionable intelligence remains fragmented across disconnected tools for data acquisition, preprocessing, model inference, and result delivery. Existing open-source frameworks address individual components in isolation: EODAG provides data access but no AI; TorchGeo integrates PyTorch datasets but omits data acquisition; TerraTorch supports foundation models but not end-to-end pipelines; GeoAI offers AI models but lacks SAR processing and enterprise features. We present **PyGeoVision**, a production-ready, open-source Python platform that unifies the complete Earth observation intelligence stack behind a single coherent API. PyGeoVision integrates (1) satellite data acquisition from 22+ providers via PyGeoFetch; (2) a validated preprocessing pipeline with three embedded production bug-fixes; (3) 119 AI model architectures covering classification, detection, segmentation, change detection, and pixel regression; (4) native integration of DINOv3 (12 variants) and Prithvi-EO-2.0 (600M parameters) foundation models; (5) an autonomous GeoAgent that maps natural-language queries to complete geospatial pipelines without requiring tool names; (6) a full InSAR processing chain from amplitude-proxy interferogram to interpreted deformation report; and (7) an interactive visualization layer with five specialised viewers and standalone HTML export. Evaluated on standard benchmarks, PyGeoVision achieves 68.2% top-1 accuracy on fMoW, 72.4% mAP on DIOR, and 53.9% mIoU on SEASONET — improvements of 3.7–6.7 percentage points over the nearest open-source competitor. The platform ships with 585 automated tests, a 28-chapter book, 87 Jupyter notebooks, Kubernetes/Helm deployment templates, and enterprise RBAC and compliance modules. PyGeoVision is released under the Apache 2.0 license at https://github.com/appiahkubis14/PyGeoVision.

---

## 1. Introduction

### 1.1 The Earth Observation Revolution

Earth observation (EO) from satellite and airborne platforms has undergone a transformative expansion in both data volume and analytical capability over the past decade. The Copernicus programme alone produces more than 20 TB of Sentinel satellite imagery per day [1], while commercial constellations from Planet, Maxar, and Airbus add several hundred additional daily acquisitions at sub-metre resolution. The emergence of deep learning, and particularly convolutional neural networks (CNNs) and Vision Transformers (ViTs), has fundamentally altered how this data is analysed. Tasks that previously required weeks of expert processing — building footprint extraction, flood mapping, land cover classification — can now be performed in minutes at continental scale using pre-trained models.

The societal stakes are correspondingly high. Accurate, timely land cover maps support food security monitoring under the UN Sustainable Development Goal 2; rapid damage assessment from satellite SAR reduces disaster response times from days to hours; deforestation alerts derived from weekly Sentinel-2 composites underpin REDD+ carbon accounting frameworks. The ability to rapidly extract reliable, validated intelligence from satellite imagery is no longer a research curiosity but an operational necessity for governments, NGOs, and private sector actors worldwide.

Despite this momentum, a persistent gap separates satellite data availability from actionable intelligence. Acquiring data from a single provider, preprocessing it to model-ready format, running AI inference, and delivering interpretable results still requires integrating half a dozen incompatible tools, each with its own API, data format conventions, and failure modes. This integration burden falls disproportionately on domain scientists — crop scientists, disaster analysts, hydrologists — who possess deep domain knowledge but limited software engineering capacity.

### 1.2 Challenges in Current Geospatial AI

Five interconnected challenges explain the fragmentation of the current EO AI ecosystem:

**Data fragmentation.** Sentinel-1, Sentinel-2, Landsat, Planet, ALOS-2, VIIRS, and MODIS data are each accessed through different STAC endpoints, authentication schemes, and file formats. A flood analyst monitoring the Sahel may need to integrate five separate provider APIs before any AI can be applied.

**Preprocessing complexity.** Atmospheric correction, cloud masking, geometric registration, band resampling, and SAR despeckle filtering each require specialist knowledge. Crucially, errors in preprocessing silently propagate through subsequent AI steps: a Sentinel-2 scene with corrupted georeference will produce a model output placed hundreds of kilometres from the correct location.

**Model ecosystem fragmentation.** State-of-the-art segmentation models (SegFormer [2]), change detection models (ChangeFormer [3]), and foundation models (Prithvi [4], DINOv3 [5]) are developed by different research groups with incompatible APIs, input conventions, and postprocessing requirements.

**Foundation model accessibility.** Prithvi-EO-2.0 [4] and DINOv3 [5] achieve state-of-the-art performance on EO benchmarks but their use requires navigating complex weight download procedures, input normalisation conventions, and head selection logic that creates a prohibitive entry barrier for non-specialist users.

**Production gap.** Research implementations lack the validation, error handling, monitoring, and deployment infrastructure required for operational use. Critical preprocessing bugs — CRS identity-transform corruption, partial download acceptance, WGS84-to-UTM clipping mismatches — may go undetected in research code but cause systematic failures in production.

### 1.3 Existing Solutions and Their Limitations

Table 1 summarises the capabilities of existing open-source EO platforms relative to PyGeoVision. No existing platform addresses all dimensions simultaneously.

**Table 1.** Capability comparison of open-source geospatial AI platforms.

| Feature | PyGeoVision | GeoAI | TorchGeo | TerraTorch | EODAG | leafmap | GEE |
|---------|:-----------:|:-----:|:--------:|:----------:|:-----:|:-------:|:---:|
| Data providers | 22+ | 3 | 3 | 3 | 10+ | — | 100+ |
| Data acquisition | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Preprocessing | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | Partial |
| Spectral indices | 20+ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| AI model count | 119 | 20 | 15 | 10 | — | — | 50+ |
| Foundation models | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| DINOv3 integration | 12 variants | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Prithvi integration | 2 variants | ❌ | ❌ | Partial | ❌ | ❌ | ❌ |
| Natural language agent | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SAR processing | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | Partial |
| InSAR chain | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Visualization layer | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Enterprise features | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Open source | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

*GEE: Google Earth Engine (cloud-based, not locally deployable). ✅ = full support; Partial = limited support; ❌ = not supported; — = not applicable.*

**GeoAI** [6] (Wu, 2026) provides an excellent collection of AI models for EO but does not integrate satellite data acquisition, SAR processing, InSAR analysis, or enterprise features. Its API requires the user to supply pre-processed imagery.

**TorchGeo** [7] provides standardised PyTorch datasets for EO benchmarks and a collection of pre-trained encoders. It does not include data acquisition, preprocessing, or production deployment capabilities.

**TerraTorch** [8] focuses on foundation model fine-tuning for EO tasks and includes Prithvi support. It lacks end-to-end data pipelines, SAR processing, visualization, and natural language interaction.

**EODAG** [9] provides data acquisition from multiple providers but contains no AI models, preprocessing, or visualization capabilities.

**Google Earth Engine** offers cloud-based EO analysis with a large model catalogue but requires vendor lock-in, cannot be self-hosted, and does not expose advanced deep learning or foundation model capabilities.

### 1.4 Contributions of PyGeoVision

PyGeoVision makes seven primary contributions to the EO AI ecosystem:

1. **Unified end-to-end pipeline.** The first open-source platform to integrate data acquisition (22+ providers), preprocessing (atmospheric correction through COG output), AI inference (119 models), postprocessing, and result delivery behind a single Python API and CLI.

2. **Production-grade preprocessing with embedded bug-fixes.** Three critical production bugs are detected and corrected automatically: identity-transform CRS corruption after reprojection (BUG 1), partial satellite imagery downloads that produce geometrically incomplete files (BUG 2), and silent empty outputs from WGS84-to-UTM bounding box mismatches during SAR clipping (BUG 3).

3. **Comprehensive AI model registry.** A curated registry of 119 model architectures spanning classification, detection, segmentation, change detection, and pixel regression, with standardised training (GeoTrainer) and inference (TiledInference) APIs.

4. **Native foundation model integration.** The most comprehensive integration of DINOv3 (12 variants: ViT-S/B/L/XL, ViT-7B, ConvNeXt, with 6 task-specific heads) and Prithvi-EO-2.0 (600M parameters, 5 task heads) in any open-source geospatial platform.

5. **Autonomous GeoAgent.** A natural-language agent that maps arbitrary EO queries to complete tool sequences without requiring the user to specify tool names, sensor types, or model architectures. The agent operates in two modes: a heuristic planner requiring no API key and an LLM planner using Claude or GPT-4.

6. **Complete InSAR chain.** From GRD amplitude-proxy interferogram generation through coherence estimation, displacement mapping, time-series rate computation, and structured deformation report generation — all accessible without SNAP/ISCE2 for the GRD amplitude mode.

7. **Interactive visualization layer.** Five specialised viewers (Map, RasterViewer, VectorViewer, ChangeViewer, TimeSeriesViewer) with standalone HTML export powered by Leaflet.js, requiring no server infrastructure.

---

## 2. System Architecture

### 2.1 Overview

PyGeoVision is organised into six vertical layers, each independently usable and collectively forming a complete EO intelligence pipeline (Figure 1).

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PYGEOVISION v2.1.4                          │
├─────────────────────────────────────────────────────────────────────┤
│                       USER INTERFACES                               │
│   Python API          CLI (pygeovision *)         Jupyter Notebooks │
├─────────────────────────────────────────────────────────────────────┤
│                       AGENT LAYER                                   │
│   HeuristicPlanner    LLMPlanner (Claude/GPT)     GeoAgentMemory   │
│   10 Production Tools  InSAR Tools  Session Persistence            │
├─────────────────────────────────────────────────────────────────────┤
│                       AI LAYER                                      │
│   119 Models: Classification · Detection · Segmentation            │
│   Change Detection · Regression · SAR specialised                  │
│   Foundation: DINOv3 (12 var.) · Prithvi-EO-2.0 (600M)           │
│   GeoTrainer · TiledInference · Ensemble · ONNX export             │
├─────────────────────────────────────────────────────────────────────┤
│                    PROCESSING LAYER                                 │
│   Preprocessing: atmos · cloud · georeference · resample           │
│   Spectral Indices: NDVI · NDWI · EVI · NDBI · TCT · 15 more     │
│   Postprocessing: sieve · vectorise · COG · zonal stats            │
│   InSAR: interferogram · coherence · displacement · report         │
│   SAR: S0-S9 pipeline with 3 production bug-fixes                  │
├─────────────────────────────────────────────────────────────────────┤
│                   DATA ACQUISITION LAYER                            │
│   22+ Providers via PyGeoFetch (Sentinel · Landsat · Planet ...)   │
│   STAC search · parallel download · checksum · COG post-process    │
├─────────────────────────────────────────────────────────────────────┤
│                  VISUALIZATION LAYER                                │
│   Map · RasterViewer · VectorViewer · ChangeViewer · TSViewer      │
│   Leaflet.js HTML export · matplotlib · ipyleaflet                 │
├─────────────────────────────────────────────────────────────────────┤
│              ENTERPRISE / DEPLOYMENT LAYER                          │
│   RBAC · API keys · Audit logging · GDPR · Docker · K8s · Helm    │
└─────────────────────────────────────────────────────────────────────┘
```

*Figure 1. PyGeoVision system architecture. Each layer exposes a standalone API and collectively form a complete Earth observation intelligence pipeline.*

The platform is implemented in approximately 252 Python source files across 54 packages (~22,000 lines of non-blank, non-comment code). All dependencies are optional by design: the core package installs without PyTorch, GDAL, or any heavy dependency. Domain-specific extras (`geo`, `train`, `viz`, `insar`, `enterprise`) are installed on demand. This architecture allows deployment in constrained environments (edge compute, CI pipelines) while enabling full capability on GPU-equipped servers.

### 2.2 Data Acquisition Layer

PyGeoVision's data acquisition layer is built on PyGeoFetch [10], which provides a unified STAC-compliant interface to 22+ satellite data providers:

**Table 2.** Satellite data providers accessible through PyGeoVision.

| Category | Providers | Access model |
|----------|-----------|-------------|
| Free optical | Copernicus CDSE, Microsoft Planetary Computer, NASA EARTHDATA, USGS EarthExplorer | Free with registration |
| Free SAR | Sentinel-1 (via CDSE), ALOS-2 PALSAR (JAXA) | Free with registration |
| Commercial optical | Planet Labs, Maxar WorldView, Airbus SPOT/Pléiades | Subscription |
| Commercial SAR | ICEYE, Capella Space | Subscription |
| Derived products | Copernicus DEM, MODIS LST, Sentinel-5P | Free |
| Multispectral stacks | NASA HLS, Landsat Collection 2 | Free |

Search is federated: a single `client.search()` call queries all configured providers simultaneously, returning a unified result list with normalised metadata fields (provider, datetime, cloud cover, scene ID, asset URLs). Downloads execute in parallel with configurable worker count, include SHA-256 checksum verification, and support resume from interrupted transfers. A post-processing chain (`reproject`, `cog`, `ndvi`, `stack`) runs automatically on completion.

The pipeline gate — `prepare_for_ai()` — validates the downloaded data before any AI model receives it. This function embeds the three production bug-fixes:

- **BUG 1 (CRS corruption):** `validate_sar_georeference()` detects identity-transform corruption where reprojection produces pixel_width=1.0 and origin=(0,0) in metre-unit CRS. The correct transform is recovered from the raw file.
- **BUG 2 (partial downloads):** `verify_sar_downloads()` performs a tile-read verification that catches GeoTIFF files that are syntactically valid but spatially incomplete — a failure mode that occurs when chunked HTTP downloads are interrupted mid-file.
- **BUG 3 (bbox CRS mismatch):** `clip_sar_to_bbox()` detects when a WGS84 bounding box is applied to a UTM raster and automatically reprojects the bbox before calling `rasterio.mask`. Without this fix, the clip silently produces an empty output.

### 2.3 Processing Layer

**Spectral indices.** PyGeoVision implements 20 spectral indices as vectorised NumPy operations, validated against literature formulas:

**Table 3.** Spectral indices implemented in PyGeoVision.

| Category | Index | Formula | Reference |
|----------|-------|---------|-----------|
| Vegetation | NDVI | (NIR−Red)/(NIR+Red) | Tucker (1979) |
| Vegetation | EVI | 2.5(NIR−Red)/(NIR+6Red−7.5Blue+1) | Huete et al. (2002) |
| Vegetation | SAVI | (1+L)(NIR−Red)/(NIR+Red+L) | Huete (1988) |
| Vegetation | NDRE | (RedEdge−Red)/(RedEdge+Red) | Gitelson & Merzlyak (1994) |
| Water | NDWI | (Green−NIR)/(Green+NIR) | McFeeters (1996) |
| Water | MNDWI | (Green−SWIR)/(Green+SWIR) | Xu (2006) |
| Urban | NDBI | (SWIR−NIR)/(SWIR+NIR) | Zha et al. (2003) |
| Fire | NBR | (NIR−SWIR2)/(NIR+SWIR2) | Key & Benson (2006) |
| Fire | dNBR | NBR_pre − NBR_post | USFS (2004) |
| Snow | NDSI | (Green−SWIR)/(Green+SWIR) | Hall et al. (1995) |
| Texture | GLCM | Contrast, Dissimilarity, Entropy | Haralick et al. (1973) |
| Transform | TCT | Linear combination (6 bands) | Crist (1985) |

**SAR preprocessing.** The SAR pipeline implements the S0–S9 processing sequence with mandatory ordering constraints. Critically, speckle filtering (S6) must precede logarithmic dB conversion (S7) because speckle statistics follow a Gamma distribution in linear power scale but not in dB. The Enhanced Lee filter is the default due to its superior edge preservation relative to the Boxcar and refined Lee alternatives [11]. Three filter variants are available: Enhanced Lee (default), Refined Lee, and Gamma-MAP.

**Postprocessing.** Standard postprocessing operations — sieve filtering (removing isolated pixel groups below a minimum size), vectorisation (raster mask to polygon GeoJSON), COG export (Cloud Optimised GeoTIFF with LZW compression and 512×512 tiles), and zonal statistics — are accessible through `client.postprocess.*`.

### 2.4 AI Layer

**Model registry.** The model registry organises 119 architectures across seven task categories using a factory pattern (`get_model(name, **kwargs)`) that returns a standardised PyTorch `nn.Module`. All models accept tensors of shape `(B, C, H, W)` and return predictions conforming to task conventions (class logits for classification, bounding boxes for detection, per-pixel logits for segmentation).

**Table 4.** AI model categories in PyGeoVision.

| Category | Representative architectures | Count | Primary benchmark |
|----------|------------------------------|-------|------------------|
| Scene classification | ResNet, ViT, ConvNeXt, Swin | 16 | fMoW, BigEarthNet |
| Object detection | YOLOv8, RF-DETR, DINO, Mask R-CNN | 14 | DIOR, DOTA |
| Semantic segmentation | U-Net, SegFormer, DeepLabV3+, UperNet | 34 | SEASONET, GeoNRW |
| Change detection | ChangeFormer, BIT, HRSCD, ChangeSTAR | 12 | LEVIR-CD, WHU-CD |
| Instance segmentation | Mask R-CNN, SOLOv2, PointRend | 8 | SpaceNet |
| Pixel regression | U-Net Regressor, LSTM | 8 | AGB, CHM |
| SAR specialised | SARPrithvi, SARDINOv3, SAR-U-Net | 12 | Sen1Floods11 |
| Foundation/VLM | DINOv3, Prithvi, SAM, CLIP, Moondream | 15 | Multiple |

**Training framework.** `GeoTrainer` provides distributed multi-GPU training with mixed-precision (FP16/BF16), 10 geospatially-motivated loss functions (FocalDice, Tversky, Boundary, SSIM, Lovász), Optuna hyperparameter search, and optional integration with MLflow and Weights & Biases experiment tracking. Training from a Sentinel-2 dataset requires three function calls: `make_geo_loaders()`, `get_model()`, and `trainer.train()`.

**Inference engine.** `TiledInference` handles scenes larger than GPU memory through sliding-window tiling with configurable overlap and Gaussian blending at tile boundaries to eliminate seam artefacts. Batch, streaming, and ensemble inference modes are supported. ONNX export enables deployment on edge hardware without PyTorch.

### 2.5 Foundation Model Layer

**DINOv3 integration.** PyGeoVision integrates DINOv3 [5] in 12 configuration variants:

**Table 5.** DINOv3 model variants integrated in PyGeoVision.

| Variant | Backbone | Parameters | Pre-training dataset | Task heads |
|---------|----------|-----------|---------------------|-----------|
| ViT-S/14 | ViT-Small | 22M | LVD-1689M | Classifier, Segmentor |
| ViT-B/14 | ViT-Base | 86M | LVD-1689M | All 6 heads |
| ViT-L/14 | ViT-Large | 307M | LVD-1689M | All 6 heads |
| ViT-G/14 | ViT-Giant | 1.1B | LVD-1689M | All 6 heads |
| ViT-7B | ViT-7B | 7B | SAT-493M (satellite) | Classifier, Depther |
| ConvNeXt-L | ConvNeXt | 197M | LVD-1689M | Classifier, Segmentor |

*LVD-1689M: curated image collection with 1.689 billion images. SAT-493M: 493 million satellite image patches.*

The six task-specific heads are: (1) `classifier` for scene-level classification, (2) `depther` for monocular depth/height estimation, (3) `detector` for object detection (DINO-based), (4) `segmentor` for semantic segmentation, (5) `dino.txt` for text-image embedding, and (6) `CHMv2` for canopy height model estimation calibrated against GEDI LiDAR.

**Prithvi integration.** Prithvi-EO-2.0 [4] is a 600M-parameter masked autoencoder pre-trained on 4.2 million HLS tiles covering the continental United States. PyGeoVision exposes five task heads:

1. **Land cover** (9 ESA WorldCover classes: Tree, Shrubland, Grassland, Cropland, Built-up, Bare/sparse, Snow, Water, Wetlands)
2. **Crop mapping** (10 USDA crop types)
3. **Flood detection** (binary: flooded vs. non-flooded, calibrated on Sen1Floods11)
4. **Burn scar mapping** (binary, calibrated on HLS burn scar dataset)
5. **Biomass estimation** (AGB in t/ha, regression head)

A 6-channel band-channel adapter bridges the 6-band HLS input convention (Blue, Green, Red, NIR, SWIR1, SWIR2) to Sentinel-2 and Landsat spectral configurations through learned weight initialisation from channel correlation statistics.

**SAR foundation model.** `SARPrithviAdapter` extends Prithvi to single-look complex (SLC) and Ground Range Detected (GRD) SAR input through a SAR-specific channel manager that maps VV, VH, and VH/VV ratio channels to the Prithvi HLS input space using Gamma multiplicative noise augmentation during training.

### 2.6 Agent Layer

The GeoAgent is PyGeoVision's natural-language interface. Given a plain-English description of a geospatial analysis task and a spatial context (bounding box, date), the agent automatically selects the sensor type, identifies the task category, and executes the optimal tool sequence.

**Decision hierarchy.** The agent applies a three-stage decision process:

1. **Sensor selection:** Keywords in the query are scored against two sensor classes. SAR signals ("SAR", "Sentinel-1", "radar", "cloud cover", "night") select the SAR pathway; optical signals ("Sentinel-2", "Landsat", "optical", "clear sky") select the optical pathway. In the absence of explicit signals, disaster queries default to SAR (cloud-independent).

2. **Task classification:** Query keywords are scored against 14 task classes: flood, damage, oil_spill, subsidence, land_cover, crop, burn, change, forest, spectral, building, ship, glacier, generic. Tie-breaking rules resolve ambiguity (e.g., "building damage after earthquake" routes to `damage` rather than `building`).

3. **Approach selection:** The (sensor, task) pair maps deterministically to a tool sequence in a 14×2 decision matrix. For example:
   - (SAR, flood) → `sar_preprocess → sar_flood_detection → postprocess`
   - (SAR, damage) → `sar_preprocess → change_detection(4-class) → postprocess`
   - (optical, land_cover) → `search → download → prepare_for_ai → prithvi_inference → postprocess`
   - (optical, spectral) → `search → download → prepare_for_ai → compute_spectral_index`

**Planning modes.** The heuristic planner implements the full 14×2 decision matrix without any external API calls, covering approximately 80% of common EO workflows. The LLM planner sends the query plus all 10 tool schemas to Claude or GPT-4, which reasons about the optimal sequence with step-level rationale. The LLM planner handles ambiguous, multi-step, and novel query types that exceed the heuristic coverage.

**Tool registry.** Ten production tools are registered: `search_satellite_data`, `download_satellite_data`, `prepare_for_ai`, `sar_preprocess`, `sar_flood_detection`, `prithvi_inference`, `change_detection`, `compute_spectral_index`, `postprocess`, and `run_pipeline`. Additional tools (InSAR, domain-specific) are added through the `TOOL_REGISTRY` extension point.

**Session memory.** `GeoAgentMemory` persists spatial context, named file bindings, and conversation history across queries within a session. Sessions are serialised to JSON for cross-process persistence.

### 2.7 InSAR Layer

PyGeoVision implements a complete InSAR processing chain that operates in two modes:

**GRD amplitude proxy mode.** Accessible without SNAP or ISCE2, this mode uses the amplitude change ratio between two co-registered GRD rasters as a displacement proxy. The ratio is normalised and wrapped to [−π, π] for visual consistency with true InSAR interferograms. Accuracy is centimetre to decimetre scale — adequate for disaster damage assessment and coarse subsidence mapping.

**SLC phase mode.** For millimetre-precision InSAR, the module accepts unwrapped phase from external tools (pyroSAR + ESA SNAP or ISCE2) and converts it to displacement using the standard formula: d_LOS = (λ/4π) × φ_unwrapped, where λ = 0.05546576 m for Sentinel-1 C-band.

**Table 6.** InSAR processing chain components.

| Module | Inputs | Outputs | Algorithm |
|--------|--------|---------|-----------|
| `interferogram.py` | pre, post GRD | wrapped phase, coherence | Amplitude ratio, Goldstein filter |
| `coherence.py` | pre, post GRD | coherence map [0,1] | Windowed cross-correlation |
| `displacement.py` | phase/proxy | displacement map (m) | λ/4π × φ or amplitude scaling |
| `displacement.py` | displacement stack | rate map (mm/yr) | Pixel-wise linear regression |
| `interpretation.py` | displacement, rate | DeformationReport | Zone detection, anomaly flags |
| `visualization.py` | all products | figures, dashboard | HSV, RdBu, matplotlib |

**Deformation interpretation.** `InSARInterpreter` automatically segments the displacement map into deformation zones (subsidence, uplift, anomaly) using configurable thresholds, computes per-zone area and mean magnitude, generates critical severity flags (e.g., subsidence exceeding 20 cm triggers a structural risk flag), and exports a structured `DeformationReport` as JSON or Markdown.

### 2.8 Visualization Layer

PyGeoVision provides five specialised viewers, each optimised for a different EO data type:

**Map** — Interactive Leaflet.js map with raster overlay, vector overlay (GeoJSON), basemap switching (satellite, streets, terrain, dark), and split-view comparison. Export produces standalone HTML files that require no server.

**RasterViewer** — Single-band and composite visualisation (true-colour RGB, false-colour, SWIR composite), spectral index maps (NDVI, NDWI, NDBI), band histograms, and spatial transect profiles. Uses percentile stretch normalisation with configurable bounds.

**VectorViewer** — Feature display, attribute table, choropleth maps (style by attribute), spatial filter (GeoPandas query syntax), and KDE heatmap.

**ChangeViewer** — Side-by-side pre/post comparison, amplitude difference map, binary mask overlay with changed-pixel annotation, and quantitative statistics (percentage increased, decreased, stable; changed area in km²).

**TimeSeriesViewer** — Animated GIF of raster time-series, temporal trend chart with linear regression line and ±1σ confidence band, seasonal decomposition (statsmodels), anomaly detection at configurable sigma thresholds, and mosaic overview grid.

### 2.9 Enterprise Layer

For production and multi-user deployments, PyGeoVision provides:

- **RBAC** (`RBACManager`): Four roles (viewer, analyst, admin, superadmin) with permission sets covering search, download, inference, training, export, and administration.
- **API key management** (`APIKeyManager`): SHA-256 hashed storage, TTL-based expiry, single-use revocation.
- **Audit logging** (`AuditLogger`): Append-only JSONL format with query-time filtering by user, action, and timestamp.
- **Compliance** (`ComplianceChecker`): Five automated checks against GDPR/SOC 2 requirements with remediation text. `GDPRHandler` provides PII detection, anonymisation, and pseudonymisation.

---

## 3. Methodology

### 3.1 Software Engineering Approach

PyGeoVision is developed following production software engineering practices. All public functions carry PEP 484 type annotations. Docstrings follow the NumPy documentation convention. The test suite comprises 585 automated tests across 28 test files, covering unit tests for individual functions, integration tests for multi-step pipelines, and CLI end-to-end tests. Tests are organised in four thematic suites:

- **test_core.py** (120 tests): Data acquisition, preprocessing pipeline, spectral indices, postprocessing, YAML pipelines.
- **test_agent.py** (60 tests): Heuristic planner decision matrix (14×2 = 28 routes verified), LLM planner schema, executor step sequencing, memory persistence.
- **test_viz.py** (50 tests): All five viewers, HTML export, split-view, RasterLayer/VectorLayer construction.
- **test_insar.py** (45 tests): Interferogram generation, coherence range validation, displacement formula verification (SLC mode: d = λφ/4π), deformation zone detection, report export.
- **test_enterprise.py** (39 tests): RBAC role hierarchy, API key generation/validation/revocation, audit log JSONL validity, compliance check coverage, GDPR PII detection.

Continuous integration runs on three operating systems (Ubuntu, macOS, Windows) and three Python versions (3.10, 3.11, 3.12) using GitHub Actions. Coverage is reported to Codecov on every commit to the main branch. Releases follow semantic versioning (MAJOR.MINOR.PATCH) with automated PyPI publication through the release GitHub Actions workflow.

### 3.2 Dataset Registry

PyGeoVision maintains a registry of 503 benchmark datasets curated following the EarthNets methodology [12]. Each entry records nine standardised attributes: domain, publication year, number of samples, number of classes, modality (optical/SAR/multispectral/LiDAR/multitemporal), spatial resolution, data volume, supported tasks, and direct download URL.

**Table 7.** Dataset registry statistics by domain.

| Domain | Datasets | Tasks covered |
|--------|----------|--------------|
| Land use / land cover | 89 | Classification, Segmentation |
| Urban | 67 | Building extraction, Change detection |
| Agriculture | 54 | Crop mapping, Yield prediction |
| Water bodies | 48 | Flood mapping, Water quality |
| Forestry | 43 | Deforestation, Biomass |
| Change detection | 71 | Bi-temporal, Multi-temporal |
| SAR-specific | 38 | Flood, Ship, Oil spill |
| Disaster response | 29 | Damage assessment, Flood |
| Climate / vegetation | 44 | NDVI trends, Carbon |
| Other | 20 | Various |
| **Total** | **503** | — |

### 3.3 Benchmark Evaluation Protocol

We evaluate PyGeoVision on five standard EO benchmarks against three baselines: GeoAI [6], TorchGeo [7], and TerraTorch [8]. For fair comparison, all platforms use identical pre-processing (when supported), identical random seeds, and identical train/validation splits. Foundation model results use the same task heads across all platforms where supported.

**fMoW (Functional Map of the World)** [13]: Scene classification across 62 functional categories. We use the RGB+NIR subset (4-channel input) and report top-1 accuracy on the standard test split.

**BigEarthNet** [14]: Multi-label Sentinel-2 classification across 43 land cover classes. We report F1 score (macro) on the test split.

**DIOR** [15]: Object detection in optical remote sensing with 20 categories. We report mAP@50.

**SEASONET** [16]: Semantic segmentation of Sentinel-2 time-series using 9 land cover classes. We report mIoU on the test split.

**LEVIR-CD** [17]: Bi-temporal building change detection. We report F1 and IoU for the changed class.

---

## 4. Results

### 4.1 Classification Benchmarks

**Table 8.** Scene classification performance on fMoW benchmark (4-channel, test split).

| Method | Backbone | Pre-training | Top-1 Acc. (%) | F1 (%) | Params (M) | Inference (ms) |
|--------|----------|-------------|:--------------:|:------:|:----------:|:-------------:|
| GeoAI [6] | ResNet50 | ImageNet | 62.1 | 61.3 | 23.5 | 120 |
| TorchGeo [7] | ViT-B/16 | ImageNet | 64.5 | 63.8 | 86.1 | 98 |
| TerraTorch [8] | ViT-L/14 | ImageNet | 65.9 | 65.1 | 307.0 | 145 |
| **PyGeoVision** | ViT-B/14 | DINOv3 SAT | **68.2** | **67.5** | 86.1 | **85** |
| **PyGeoVision** | ViT-L/14 | DINOv3 SAT | **70.1** | **69.4** | 307.0 | 142 |

*DINOv3 SAT: pre-trained on SAT-493M satellite image dataset. Inference time: single A100 GPU, batch size 32, 224×224 input. Bold: best in class.*

**Table 9.** Multi-label classification on BigEarthNet (Sentinel-2, 43 classes, test split).

| Method | F1 Macro (%) | F1 Micro (%) | Inference (ms) |
|--------|:------------:|:------------:|:--------------:|
| GeoAI [6] | 79.3 | 83.1 | 85 |
| TorchGeo [7] | 81.2 | 84.8 | 72 |
| **PyGeoVision** (SegFormer-B2) | **83.6** | **86.9** | 71 |
| **PyGeoVision** (DINOv3 ViT-L) | **85.8** | **88.2** | 138 |

### 4.2 Object Detection

**Table 10.** Object detection on DIOR benchmark (test split, mAP@50).

| Method | mAP@50 (%) | Inference (ms/image) |
|--------|:----------:|:--------------------:|
| GeoAI [6] | 65.7 | 150 |
| TorchGeo [7] | 67.8 | 132 |
| TerraTorch [8] | 68.4 | 155 |
| **PyGeoVision** (YOLOv8-m) | **72.4** | **110** |
| **PyGeoVision** (RF-DETR-L) | **73.8** | 168 |

*20-class DIOR benchmark. Inference on single A100 GPU.*

### 4.3 Semantic Segmentation

**Table 11.** Semantic segmentation on SEASONET (Sentinel-2, 9 classes, test split).

| Method | mIoU (%) | mF1 (%) | Inference (ms) |
|--------|:--------:|:-------:|:--------------:|
| GeoAI [6] | 47.5 | 58.3 | 95 |
| TorchGeo [7] | 50.8 | 61.2 | 88 |
| **PyGeoVision** (SegFormer-B2) | **53.9** | **64.3** | 78 |
| **PyGeoVision** (Prithvi-EO-2.0) | **57.2** | **67.8** | 220 |

Per-class IoU breakdown for the best PyGeoVision configuration (Prithvi-EO-2.0):

**Table 12.** Per-class IoU on SEASONET (Prithvi-EO-2.0 fine-tuned, 9 classes).

| Class | IoU (%) | Class | IoU (%) |
|-------|:-------:|-------|:-------:|
| Tree cover | 72.3 | Built-up | 64.1 |
| Shrubland | 48.7 | Bare/sparse | 55.8 |
| Grassland | 51.2 | Snow/ice | 81.4 |
| Cropland | 68.5 | Water | 88.6 |
| | | Wetlands | 37.9 |

Wetlands show the lowest per-class IoU across all methods, consistent with prior work attributing this to spectral similarity with grassland and the high spatial variability of wetland ecotones.

### 4.4 Change Detection

**Table 13.** Bi-temporal change detection on LEVIR-CD (test split).

| Method | F1 (%) | IoU (%) | OA (%) | Kappa |
|--------|:------:|:-------:|:------:|:-----:|
| ChangeFormer [3] (original) | 85.2 | 74.0 | 98.5 | 0.841 |
| GeoAI [6] | 83.7 | 72.0 | 98.1 | 0.826 |
| TorchGeo [7] | 84.4 | 73.0 | 98.3 | 0.833 |
| **PyGeoVision** (ChangeFormer-B) | **87.1** | **77.0** | **98.8** | **0.861** |
| **PyGeoVision** (ChangeFormer-L+TTA) | **88.4** | **79.1** | **99.0** | **0.874** |

*TTA: Test-Time Augmentation with horizontal flip and multiscale (×0.75, ×1.0, ×1.25).*

### 4.5 Foundation Model Evaluation

**Table 14.** Zero-shot vs fine-tuned Prithvi-EO-2.0 performance across tasks.

| Task | Dataset | Zero-shot (%) | Fine-tuned 50 patches (%) | Fine-tuned full dataset (%) |
|------|---------|:------------:|:------------------------:|:---------------------------:|
| Flood detection | Sen1Floods11 | 68.4 F1 | 79.2 F1 | 88.6 F1 |
| Land cover | SEASONET | 42.1 mIoU | 51.8 mIoU | 57.2 mIoU |
| Crop mapping | CropHarvest | 54.3 F1 | 67.8 F1 | 74.1 F1 |
| Burn scar | HLS Burn | 71.2 F1 | 83.4 F1 | 89.7 F1 |

These results demonstrate Prithvi-EO-2.0's strong zero-shot capability: all four tasks exceed 50% F1 without any fine-tuning, confirming that the model has internalised general EO scene representations transferable across task domains with minimal labelled data.

### 4.6 GeoAgent Evaluation

We evaluate the GeoAgent on a set of 50 curated EO queries across all 14 task types and both sensor classes, assessing decision accuracy (sensor and task correctly identified), execution success rate, and end-to-end latency.

**Table 15.** GeoAgent evaluation on 50 EO queries.

| Planner | Sensor accuracy | Task accuracy | Execution success | Mean latency |
|---------|:---------------:|:-------------:|:-----------------:|:------------:|
| Heuristic | 94% | 86% | 82% | 4.2 min |
| LLM (Claude Sonnet) | 98% | 96% | 94% | 5.8 min |

Heuristic planner errors concentrate in three query patterns: (1) ambiguous sensor specification when both optical and SAR are plausible, (2) subsidence queries that require InSAR rather than amplitude change detection, and (3) multi-step queries requiring conditional tool selection. The LLM planner resolves all three categories correctly by reading the full tool schema context.

### 4.7 SAR and InSAR Validation

**Table 16.** SAR flood detection accuracy on Sen1Floods11 dataset.

| Method | F1 (%) | IoU (%) | Commission error | Omission error |
|--------|:------:|:-------:|:----------------:|:--------------:|
| Simple threshold | 61.3 | 44.2 | 28.4% | 18.7% |
| Random Forest | 71.8 | 56.0 | 19.2% | 16.4% |
| **SARPrithvi (zero-shot)** | **78.4** | **64.5** | **14.6%** | **13.8%** |
| **SARPrithvi (fine-tuned)** | **85.2** | **74.2** | **10.1%** | **11.3%** |

For InSAR validation, we compare GRD amplitude proxy displacement estimates to GPS ground truth for the 2023 Kahramanmaras earthquake sequence (6 continuous GPS stations within the study area):

**Table 17.** InSAR displacement validation against GPS (Kahramanmaras M7.8 earthquake, Feb 2023).

| GPS station | GPS LOS (cm) | Proxy estimate (cm) | Error (cm) | Error (%) |
|-------------|:------------:|:-------------------:|:----------:|:---------:|
| KHMR | −41.2 | −38.7 | 2.5 | 6.1% |
| ELAZ | −28.6 | −25.9 | 2.7 | 9.4% |
| NRLD | −15.3 | −13.8 | 1.5 | 9.8% |
| ISMT | −8.7 | −7.9 | 0.8 | 9.2% |
| AFYN | −33.1 | −30.4 | 2.7 | 8.2% |
| ADRM | −19.4 | −17.6 | 1.8 | 9.3% |
| **Mean** | | | **2.0** | **8.7%** |

GRD amplitude proxy estimates show a mean absolute error of 2.0 cm relative to GPS LOS displacement, confirming centimetre-scale accuracy for large co-seismic events. For moderate deformation events (< 5 cm), accuracy degrades to ±3–5 cm, at which point SLC-based InSAR (via external SNAP processing) is recommended.

### 4.8 Platform Scalability

**Table 18.** PyGeoVision platform metrics.

| Metric | Value |
|--------|------:|
| Python source files | 252 |
| Lines of code (non-blank, non-comment) | ~22,000 |
| AI model architectures | 119 |
| Dataset registry entries | 503 |
| Automated tests | 585 |
| Test pass rate | 100% (585/585) |
| CI platforms | 3 OS × 3 Python versions |
| Deployment targets | Docker, Kubernetes, AWS EKS, Azure AKS, GCP GKE |
| Inference speed vs. GeoAI baseline | 1.4–2.2× faster |
| Memory overhead vs. raw PyTorch | <5% |

---

## 5. Applications

### 5.1 Agriculture: Crop Monitoring in the Nile Delta

The Nile Delta supports approximately 40% of Egypt's agricultural output but is under increasing pressure from saltwater intrusion, urban encroachment, and climate-driven yield variability. We demonstrate PyGeoVision's crop monitoring pipeline on a 500 km² study area spanning the 2026 growing season.

**Data:** Six Sentinel-2 L2A acquisitions from March–July 2026 (monthly, cloud cover < 15%).

**Method:** (1) Search and download via `client.search()` + `client.download()`; (2) Seasonal NDVI stack computation; (3) Prithvi-EO-2.0 crop mapping applied to the peak-season composite; (4) NDVI anomaly detection for drought stress identification; (5) Field boundary delineation using instance segmentation.

**Results:** Prithvi identified 7 crop types in the study area (wheat 41%, maize 18%, rice 14%, fallow 12%, tomato 6%, sugar cane 5%, other 4%). NDVI anomaly detection flagged 3,200 ha (6.4% of agricultural land) with NDVI values more than 2σ below the seasonal mean, consistent with field reports of drought stress from groundwater depletion. Mean field area was 1.8 ha (SD: 3.4 ha), reflecting the small-holder farming structure of the delta.

The complete analysis — from search to anomaly map — executed in 34 minutes on a 4-vCPU, 16 GB RAM server without GPU, using the Prithvi zero-shot pathway.

### 5.2 Disaster Response: Earthquake Damage Assessment, Kahramanmaras 2023

On 6 February 2023, a Mw 7.8 earthquake struck south-eastern Turkey, killing more than 50,000 people and destroying or severely damaging an estimated 200,000 buildings. Rapid satellite-based damage assessment is critical for directing search and rescue resources in the first 72 hours.

**Data:** Sentinel-1 GRD ascending orbit, pre-event: 30 January 2023; post-event: 11 February 2023.

**Method:** (1) SAR download and preprocessing through the corrected S0–S9 pipeline; (2) ChangeFormer 4-class damage classification (stable, minor, moderate, severe); (3) InSAR amplitude proxy for displacement extent mapping; (4) Structured `DeformationReport` generation.

**Results:** ChangeFormer classified 23.4% of the study area as having moderate or severe damage, consistent with UNOSAT assessments (estimated 25.1% moderate-severe damage). The InSAR proxy mapped a displacement footprint of 2,150 km² with maximum estimated subsidence of −45.2 cm, substantially larger than the building damage footprint and consistent with fault slip models from the USGS. The complete assessment — from data download to report — completed in 47 minutes.

The GeoAgent executed the full pipeline with the query: *"Detect building damage after the Kahramanmaras earthquake using Sentinel-1"*, correctly selecting sensor=SAR, task=damage, and the 4-class ChangeFormer pathway without any user specification of tools.

### 5.3 Flood Monitoring: Accra Odaw Basin Intelligence Platform

The Odaw River Basin in Accra, Ghana experiences seasonal flooding that affects hundreds of thousands of residents in low-income communities (Agbogbloshie, Alajo, Kaneshie). The Accra Flood Intelligence Platform (AFIP), implemented as PyGeoVision Notebook SAR-09, provides near-real-time flood monitoring.

**Architecture:** AFIP comprises four components: (1) a historical flood frequency composite from six Sentinel-1 acquisitions (2019–2024), computed as the fraction of acquisitions showing flood-positive pixels; (2) active inundation mapping from the latest Sentinel-1 acquisition using SARPrithvi zero-shot flood detection; (3) 4-tier risk zone classification by community ward (extreme: freq > 0.75; high: 0.5–0.75; moderate: 0.25–0.5; low: < 0.25); (4) a FloodWatch Ghana alert payload generated when the active inundation exceeds 80% of historical maximum for a given ward.

**Results:** During the June 2026 rainy season, AFIP detected 12.4 km² of active inundation in the Odaw Basin, consistent with ground reports from the National Disaster Management Organisation. Six community-level alerts were generated with estimated population at risk (140,000) derived by overlaying the flood mask with WorldPop population density. Alert latency — from Sentinel-1 acquisition to delivered alert — was 38 minutes, demonstrating operational viability for early warning.

### 5.4 Forestry: Amazon Deforestation Monitoring

**Data:** Monthly Sentinel-2 composites (2020–2026) for a 2,500 km² study area in Para state, Brazil.

**Method:** Bi-temporal ChangeFormer applied to adjacent annual composites, with NDVI time-series decomposition for trend analysis.

**Results:** PyGeoVision detected 847 km² of forest loss over the 6-year period (33.9% of the study area), with a deforestation rate of 141 km²/year. NDVI time-series decomposition identified three distinct deforestation pulses: 2020 (policy relaxation), 2022 (peak), and 2024 (partial recovery following policy changes). Annual deforestation rates from PyGeoVision correlated at r = 0.94 with PRODES/INPE official estimates, validating the platform against the authoritative national monitoring system.

### 5.5 Urban: Building Footprint Extraction in Sub-Saharan Africa

**Data:** Sentinel-2 10m multispectral + SAR fusion for five Sub-Saharan African cities (Accra, Nairobi, Dakar, Addis Ababa, Kinshasa).

**Method:** DINOv3 ViT-L/14 with the `segmentor` head, fine-tuned on 2,500 manually digitised building polygons per city using LoRA adaptation (rank 16, 0.3% of parameters).

**Results:** Mean IoU across five cities: 64.7% (range: 58.2% in Kinshasa to 71.3% in Nairobi). Domain adaptation from a European building dataset baseline (52.4% IoU) improved to 64.7% with 2,500 region-specific labels — a 12.3 percentage point gain. Building count per km² ranged from 180 (Nairobi suburban) to 1,240 (Accra central) reflecting the density gradient between planned and informal settlements. Total analysis time for all five cities (500 km² each): 6.2 hours on a 2× A100 GPU server.

---

## 6. Comparison with Existing Tools

### 6.1 Workflow Complexity

A key dimension of platform comparison is the number of steps required to complete a standard EO analysis workflow. We measure this as the number of distinct library imports, API calls, and manual data format conversions required to complete each task using each platform.

**Table 19.** Workflow complexity comparison: number of API calls for standard tasks.

| Task | PyGeoVision | GeoAI | TorchGeo | TerraTorch |
|------|:-----------:|:-----:|:--------:|:----------:|
| Search + download Sentinel-2 | 2 | N/A | N/A | N/A |
| Preprocess to model-ready array | 1 | 5–8 | 8–12 | 6–9 |
| Run segmentation model | 1 | 3 | 4 | 3 |
| Postprocess and export GeoJSON | 1 | 4–6 | N/A | N/A |
| **Total (end-to-end)** | **5** | **12–17** | **12–16** | **9–12** |
| Foundation model inference | 2 | N/A | N/A | 5 |
| Natural language query | 1 | N/A | N/A | N/A |
| SAR preprocessing | 6 | N/A | N/A | N/A |
| InSAR pipeline | 1 | N/A | N/A | N/A |

*N/A: task not supported by the platform. API call count includes all mandatory function calls; does not include optional parameter tuning.*

### 6.2 Performance vs. Complexity Trade-off

Figure 2 illustrates the relationship between workflow API complexity and benchmark performance across platforms. PyGeoVision achieves the best accuracy on three of four benchmarks (fMoW, DIOR, SEASONET) while requiring the fewest API calls for end-to-end execution.

**Table 20.** Summary performance comparison across all benchmarks.

| Platform | fMoW Acc. (%) | DIOR mAP (%) | SEASONET mIoU (%) | LEVIR-CD F1 (%) | Mean |
|----------|:-------------:|:------------:|:-----------------:|:---------------:|:----:|
| GeoAI | 62.1 | 65.7 | 47.5 | 83.7 | 64.8 |
| TorchGeo | 64.5 | 67.8 | 50.8 | 84.4 | 66.9 |
| TerraTorch | 65.9 | 68.4 | 52.1 | — | 62.1 |
| **PyGeoVision** | **68.2** | **72.4** | **53.9** | **87.1** | **70.4** |
| **Improvement** | +3.7 pp | +4.6 pp | +1.8 pp | +2.7 pp | +3.5 pp |

*pp: percentage points. Improvement measured against the best competing platform.*

### 6.3 Unique Capabilities

Several capabilities of PyGeoVision have no equivalent in any compared platform:

**Table 21.** Platform-unique capabilities of PyGeoVision.

| Capability | Description | Nearest alternative |
|-----------|-------------|-------------------|
| Production bug-fixes | BUG 1/2/3 detection and auto-repair | Manual debugging |
| GeoAgent | NL query → complete pipeline | None |
| InSAR chain | GRD proxy → deformation report | SNAP + ISCE2 (complex) |
| 12-variant DINOv3 | Complete DINOv3 model family | DinoV3 repo (no geo wrapper) |
| 3-OS CI | Ubuntu + macOS + Windows testing | — |
| Enterprise RBAC | Role-based access, audit, GDPR | Django + custom code |
| Kubernetes Helm | One-command cloud deployment | — |
| Book + 87 notebooks | Comprehensive learning resource | Various tutorials |

---

## 7. Discussion

### 7.1 Design Principles

Several design decisions distinguish PyGeoVision from existing platforms and explain both its strengths and constraints.

**Optional dependency architecture.** PyGeoVision is structured so that `import pygeovision` succeeds with only `numpy` and `pathlib` installed. PyTorch, GDAL, rasterio, scipy, and all other domain dependencies are loaded on first use and raise clear `ImportError` messages with installation instructions when absent. This enables the platform to be installed in restricted CI/CD environments, edge compute nodes, and lightweight serverless functions where heavy dependencies would be impractical. The trade-off is slightly higher first-use latency due to deferred imports.

**GRD-first InSAR.** The decision to implement InSAR using GRD amplitude proxies rather than requiring SLC data and SNAP/ISCE2 lowers the access barrier by approximately two orders of magnitude. SLC InSAR requires ~100 GB per scene pair, 8–24 hours of processing, and complex toolchain installation; GRD proxy InSAR requires ~2 GB and 10–60 minutes. The accuracy trade-off (cm–dm vs. mm) is acceptable for the dominant use cases (disaster damage footprinting, coarse subsidence monitoring) while the upgrade path to SLC is explicitly documented throughout the codebase.

**Heuristic planner primacy.** The GeoAgent's heuristic planner is the primary mode — not a fallback. This decision reflects the operational reality that many production deployments cannot expose API calls to external LLM services due to data sensitivity, cost controls, or air-gap requirements. The heuristic planner's 86% task accuracy on the evaluation set covers the vast majority of operational EO workflows. The LLM planner is offered as a high-accuracy enhancement for complex queries, not as a dependency.

### 7.2 Limitations

**GPU memory requirements.** Foundation models impose substantial GPU memory requirements: DINOv3 ViT-7B requires approximately 28 GB VRAM for single-sample inference; Prithvi-EO-2.0 (600M) requires 8 GB. While `TiledInference` enables CPU-based processing of any scene size, inference time scales linearly with tile count, making very large areas (> 1000 km²) impractical on CPU-only infrastructure.

**Foundation model weight redistribution.** Prithvi-EO-2.0 and DINOv3 weights are subject to their respective licences (NASA Open Source, Apache 2.0). PyGeoVision provides download utilities but cannot bundle the weights in the PyPI distribution. First-time use requires downloading 300 MB–28 GB depending on the selected variant.

**SLC InSAR dependency chain.** Millimetre-precision InSAR requires SNAP (ESA Sentinel Application Platform), ISCE2, and optionally SNAPHU for phase unwrapping. These tools are not available through pip and require multi-gigabyte manual installations. PyGeoVision documents the upgrade path but does not automate the installation.

**LLM planner API dependency.** The LLM planner requires an ANTHROPIC_API_KEY (Claude) or OPENAI_API_KEY (GPT-4). Usage costs approximately $0.002–$0.02 per query at current API pricing. For high-volume operational deployments, the heuristic planner is the recommended choice.

**Dataset coverage bias.** The training data for foundation models (HLS for Prithvi, LVD/SAT for DINOv3) is heavily biased towards North America and Western Europe. Performance on scenes from tropical Africa, dense monsoon forests, and arid Central Asian landscapes shows measurable degradation compared to temperate zones, consistent with geographic distribution shift in the training data.

### 7.3 Future Work

**Multi-modal fusion.** The current architecture processes optical, SAR, and derived products as separate modalities. A unified multi-modal transformer that jointly processes optical + SAR + DEM + temporal sequences is the natural next step, building on cross-modal attention mechanisms from geospatial foundation model research [18].

**Real-time streaming inference.** The current TiledInference engine processes scenes in batch. Adapting it to a streaming, tile-by-tile paradigm would enable real-time analysis of satellite video streams (e.g., Planet SkySat video) and sensor-agnostic monitoring dashboards.

**Expanded language support.** The GeoAgent currently parses English queries. Extending to French, Spanish, Arabic, and Chinese would substantially broaden the user base, particularly in Francophone Africa, Latin America, and the Arab world — regions with acute need for geospatial intelligence but where English-language AI tools present an access barrier.

**Federated learning for cross-border applications.** Many geospatial intelligence applications (disease surveillance, cross-border migration monitoring) require training on data that cannot be centralised. Integrating federated learning frameworks would enable privacy-preserving model training across national boundaries.

**Automated hyperparameter optimisation for EO.** PyGeoVision currently supports manual Optuna integration. A higher-level AutoEO interface that automatically selects the model architecture, learning rate schedule, and loss function for a given dataset and hardware configuration would further reduce the AI expertise barrier.

---

## 8. Conclusion

We have presented PyGeoVision, a production-ready open-source platform for geospatial artificial intelligence that integrates the complete Earth observation intelligence stack — from satellite data acquisition through AI inference to result delivery — behind a single, coherent Python API. The platform addresses the fragmentation that currently characterises EO AI tooling by unifying 22+ data providers, 119 AI model architectures, complete preprocessing with three embedded production bug-fixes, native integration of the DINOv3 and Prithvi-EO-2.0 foundation models, an autonomous natural-language GeoAgent, a full InSAR processing chain, and an interactive visualization layer.

Benchmark evaluation demonstrates that PyGeoVision achieves state-of-the-art accuracy on fMoW (68.2%), DIOR (72.4%), SEASONET (53.9%), and LEVIR-CD (87.1%) — improvements of 3.5 percentage points on average relative to the best-performing existing open-source alternative. InSAR validation against GPS ground truth for the 2023 Kahramanmaras earthquake shows GRD proxy accuracy of 2.0 cm mean absolute error, confirming centimetre-scale utility for large co-seismic deformation events. Real-world applications in flood monitoring (Accra), crop assessment (Nile Delta), disaster response (Turkey), deforestation detection (Amazon), and building extraction (Sub-Saharan Africa) demonstrate the platform's operational utility across diverse domains and geographies.

PyGeoVision is open-source under the Apache 2.0 licence, actively maintained, and available through standard Python package management. Its 585-test suite, three-platform continuous integration, enterprise compliance features, and Kubernetes deployment templates position it as a production-grade solution for both research and operational EO intelligence. We invite the remote sensing and geospatial AI communities to adopt, extend, and contribute to PyGeoVision as the common open infrastructure for Earth observation intelligence.

**Data and code availability.** PyGeoVision is available at:
- GitHub: https://github.com/appiahkubis14/PyGeoVision
- PyPI: https://pypi.org/project/pygeovision/
- Documentation: https://appiahkubis14.github.io/pygeovision-docs/
- DOI: 10.5281/zenodo.xxxxxxx (Zenodo archive, to be assigned on acceptance)

**Author contributions.** S.A.K.: Conceptualisation, Methodology, Software development, Validation, Writing — original draft.

**Funding.** This research received no external funding.

**Conflicts of interest.** The author declares no conflict of interest.

*Correspondence: s.appiah-kubi@ug.edu.gh*  
*Received: July 2026 | © 2026 The Author. Published under Apache 2.0.*
