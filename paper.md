---
title: 'PyGeoVision: A Unified Open-Source Python Platform for Satellite Earth Observation and Geospatial AI'
tags:
  - Python
  - Earth observation
  - remote sensing
  - synthetic aperture radar
  - deep learning
  - geospatial AI
  - Africa
  - flood mapping
  - InSAR
authors:
  - name: Samuel Appiah Kubi
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: EOCoreINT, Accra, Ghana
    index: 1
date: 10 August 2026
version: 2.1.6
bibliography: paper.bib
---

# Summary

`pygeovision` is an open-source Python library that provides a unified,
production-ready interface for satellite Earth observation (EO) analysis and
geospatial AI inference. Built on top of `pygeofetch` [@appiahkubi2026pygeofetch] — a
companion library that abstracts data access across 22+ satellite data providers and
implements a complete native InSAR processing chain — `pygeovision` covers the full EO
workflow from raw satellite acquisition to AI-derived geospatial intelligence:
preprocessing pipelines for optical and SAR imagery, spectral index computation, cloud
masking, change detection, classification, segmentation, InSAR deformation analysis,
and inference via foundation models including Prithvi-EO-2.0 [@jakubik2023prithvi] and
DINOv2 [@oquab2023dinov2].

A natural-language query interface (GeoAgent) allows users to describe an analysis in
plain English and have the system automatically select the appropriate sensor, task,
and tool sequence. Together, `pygeovision` and `pygeofetch` are designed to make
operational EO analysis accessible to researchers and government analysts in Africa
and the Global South, where in-situ monitoring infrastructure is sparse but satellite
coverage is complete.

The platform is production-ready, with 944+ automated tests and continuous
integration, ensuring reliability for operational monitoring systems. `pygeovision`
is released under the Apache 2.0 open-source license.

# Statement of Need

Satellite Earth observation is uniquely well-suited to address monitoring challenges
in data-sparse environments. Every point on Earth is covered by free Sentinel-1 SAR
imagery every 6–12 days and by free Sentinel-2 optical imagery every 5 days
[@drusch2012sentinel2; @torres2012sentinel1]. Yet the gap between data availability
and operational use remains wide: acquiring, preprocessing, and interpreting
multi-sensor satellite data requires expertise across geodesy, signal processing,
machine learning, and domain science that is difficult to assemble in
resource-constrained institutions.

Existing tools address parts of this gap. Google Earth Engine [@gorelick2017gee]
provides cloud-based EO processing but requires a proprietary account, processes data
on infrastructure outside users' control, and produces outputs whose computational
lineage cannot be fully audited — a disqualifying constraint for government monitoring
systems subject to legal challenge. Standalone deep learning frameworks (PyTorch,
TensorFlow) require users to assemble their own geospatial data pipeline. No existing
open-source package integrates all layers — data access, preprocessing, AI inference,
and production orchestration — in a single auditable, locally deployable system.

`pygeovision` fills this gap. A complete flood-mapping analysis — acquisition, SAR
preprocessing, foundation-model inference, and vector export — is expressible in a few
lines:

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
scene = client.fetch_and_preprocess(
    provider="copernicus",
    bbox=[-0.30, 5.50, -0.10, 5.70],   # Odaw River Basin
    date="2024-06-15",
    sensor="S1_GRD",
)
flood = client.infer(scene, model="prithvi_eo_2_0", task="flood_detection")
client.export(flood, "odaw_flood_2024-06-15.geojson")
```

The same analysis can be requested in natural language via the GeoAgent: *"Map flood
extent over the Odaw basin for 15 June 2024."*

Its target users are:

- **Government analysts** (national disaster management, environmental protection
  agencies, forestry commissions) who need reproducible, audit-ready flood,
  deforestation, and land cover products without depending on proprietary cloud
  platforms.
- **Academic researchers** in Africa and the Global South who need a full EO analysis
  stack with minimal infrastructure investment.
- **NGO programme officers** running food security, disaster risk reduction, or
  climate adaptation programmes who require satellite-derived evidence integrated into
  their existing reporting workflows.

# Installation and Availability

`pygeovision` requires Python 3.10+ and is distributed via the Python Package Index:

```bash
pip install pygeovision
```

Optional extras are available for training (`[train]`), foundation models
(`[foundation]`), InSAR processing (`[insar]`), and the full stack (`[all]`). The
source code, issue tracker, and comprehensive documentation — including 25 runnable
end-to-end Jupyter notebooks — are hosted at
https://github.com/EOCoreINT/pygeovision under the Apache 2.0 license.

# State of the Field

Several open-source packages address subsets of the `pygeovision` scope. `rasterio`
[@gillies2019rasterio] and `geopandas` [@kelsey2020geopandas] provide foundational
raster and vector I/O but no domain-specific EO workflows. `sentinelsat` and `pystac`
[@stac2021] provide access to specific satellite catalogues but do not normalise
across providers or handle preprocessing. `eo-learn` [@sentinelhub2021eolearn]
provides a workflow framework for Sentinel Hub (a commercial service) with limited
support for free Copernicus direct access. `torchgeo` [@stewart2022torchgeo] supplies
EO-specific PyTorch datasets and transforms but has no data acquisition layer and
limited preprocessing support.

At the foundation model level, `terratorch` [@terratorch2024] provides a fine-tuning
framework for geospatial foundation models but requires users to source and preprocess
their own data. `pygeovision` wraps foundation model inference — including
Prithvi-EO-2.0 [@jakubik2023prithvi] and DINOv2 [@oquab2023dinov2] — within the same
package that acquires and preprocesses the input data, removing the integration burden
that currently prevents operational adoption.

The closest integrated alternative is `odc-stac` combined with `datacube`, but these
require significant server infrastructure (ODC database, Kubernetes) that is
inaccessible to most African research institutions. `pygeovision` runs on a standard
laptop or a $12/month virtual machine.

GeoAI [@wu2026geoai] and leafmap [@wu2021leafmap] provide high-level interfaces for
geospatial AI and interactive mapping, but lack data acquisition and preprocessing
layers. PyGeoVision builds on the vision of these tools while extending to the
complete EO workflow.

# Software Design

## Architecture

`pygeovision` is structured in four layers, each independently usable:

**Data layer (`pygeovision.data`)** wraps `pygeofetch` to provide a provider-agnostic
search and download API. `pygeofetch` normalises across 22+ providers — including
Copernicus Data Space, Microsoft Planetary Computer, USGS EarthExplorer, and ASF
Vertex — behind a single interface. The data layer handles authentication, caching,
SHA-256 checksum verification, partial-download detection, and coordinate reference
system validation after reprojection. Four silent bugs discovered during production
flood response work over Accra are guarded against explicitly: CRS identity-transform
corruption, partial download truncation, COG vs SAFE download format mismatch
(HTTP 422), and WGS84/UTM clipping mismatch.

**Processing layer (`pygeovision.data.processors`)** implements sensor-specific
preprocessing pipelines. The SAR pipeline applies nine ordered steps (S0–S9) to
Sentinel-1 GRD products: download verification, georeference validation, band
selection, resampling, co-registration, study area clipping, Lee speckle filtering on
linear power data, dB conversion, and normalisation. The optical pipeline applies the
Sentinel-2 Scene Classification Layer (SCL) for cloud masking, atmospheric correction
validation, and computation of 12 spectral indices. Processing order is strictly
enforced: speckle filtering must precede dB conversion because the Lee filter's
multiplicative noise model applies only to linear power data, not logarithmic dB
values — a common source of silent errors in published SAR workflows.

**AI layer (`pygeovision.models`)** provides inference wrappers for: foundation models
(Prithvi-EO-2.0 with five task heads: land cover, crop mapping, flood, burn scar, and
biomass regression; DINOv2 [@oquab2023dinov2] with six task heads including CHMv2
canopy height estimation calibrated against GEDI LiDAR); change detection models
(ChangeFormer [@bandara2022changeformer]); and custom model training via `GeoTrainer`
(FocalDice loss, mixed-precision BF16 training, ONNX export for CPU deployment). A
band adapter normalises Sentinel-2 inputs to the HLS canonical six-band order required
by Prithvi-EO-2.0.

**InSAR layer (`pygeovision.insar`)** implements true SLC InSAR processing natively
through PyGeoFetch's complete chain, which includes TOPS burst handling with ESD
coregistration [@yaguemartinez2016; @scheiber2000], SNAPHU phase unwrapping
[@chen2001], SBAS time-series inversion [@berardino2002], and ERA5-based atmospheric
correction [@jolivet2014]. The chain runs natively on Windows with zero configuration.
An optional SNAP [@snap2024] backend is provided for cross-validation against the ESA
reference implementation, but the native implementation is the default and recommended
path. Unwrapped phase is converted to line-of-sight displacement using
$d_\mathrm{LOS} = \frac{\lambda}{4\pi} \varphi$, where $\lambda = 0.05547$ m for
Sentinel-1 C-band. A documented bug in the literature uses 170° as the descending
satellite heading; at this value $\sin(170°) \approx \sin(10°)$, making the 2×2 LOS
decomposition matrix nearly singular. The correct descending heading is 350° (headings
measured clockwise from geographic north; this preserves the negative sine term
required for a well-conditioned decomposition), and `pygeovision` enforces this to
prevent mathematical instability.

## GeoAgent

The GeoAgent maps natural-language queries to complete analysis pipelines through a
three-stage decision hierarchy: sensor classification (SAR vs optical, based on 40+
signal word sets), task classification (14 task types including flood, damage, change,
InSAR, and subsidence), and tool sequence planning (a 14×2 decision matrix mapping
task×sensor to ordered tool calls). The heuristic planner achieves 86% task routing
accuracy on a held-out set of 150 annotated queries, without any external API key.
With an LLM backend (Claude or GPT-4), accuracy rises to 96% and multi-intent queries
are supported.

## Stateful Reproducibility

To prevent silent processing failures, every analysis step produces stateful result
objects (e.g., `PreprocessingResult`, `InferenceResult`) that encapsulate validated
data paths, provenance metadata, and processing parameters. Every analysis run writes
a provenance JSON record documenting input scene IDs, software versions, and all
parameters, enabling independent verification of results submitted to government
agencies or used as evidence in legal proceedings.

## Key Dependencies

`pygeovision` depends on: `pygeofetch` (data access and native InSAR), `rasterio`
[@gillies2019rasterio], `geopandas` [@kelsey2020geopandas], `NumPy`
[@harris2020numpy], `PyTorch` [@paszke2019pytorch], `scikit-learn`
[@pedregosa2011sklearn], and `FastAPI` for optional REST API deployment. All
dependencies are freely available and OSI-licensed. SNAP is not required for any core
functionality; it is an optional validation tool only.

## Testing and Continuous Integration

`pygeovision` ships 944+ automated tests (unit, integration, and data-contract tests)
executed via GitHub Actions across Python 3.10–3.12 on Linux, macOS, and Windows, with
an enforced coverage gate. This cross-platform CI is deliberate: operational users in
African institutions frequently run Windows workstations, and the test matrix
guarantees the zero-configuration promise holds on all three platforms.

# Research Impact Statement

`pygeovision` has been applied to operational flood monitoring over Ghana's Odaw River
Basin, producing SAR-derived flood extent maps within 18 hours of a Sentinel-1
overpass during the June 2024 Accra flood event, with 847 hectares of inundation
detected and community-level impact assessed against census population data.
Bi-temporal Sentinel-2 change detection using ChangeFormer [@bandara2022changeformer]
detected 234.7 hectares of newly cleared forest in Ghana's Atewa Range Forest Reserve
between 2020 and 2024. SLC InSAR processing of Sentinel-1 SLC pairs over the 2023
Kahramanmaraş earthquake achieved 2.0 cm mean absolute error on line-of-sight
displacement against GPS CORS ground truth, consistent with published results from
SNAP-based processing chains. Crucially, these results were obtained using PyGeoFetch's
native InSAR implementation, demonstrating that the SNAP-free pipeline produces
equivalent or better accuracy without the installation complexity.

![PyGeoVision Unified Workflow and Operational Outputs](figures/figure1.png)
*Figure 1: (Left) PyGeoVision end-to-end architecture integrating data acquisition,
preprocessing, and AI/InSAR inference. The InSAR component uses PyGeoFetch's native
implementation by default. (Right) Operational outputs: (A) SAR-derived flood extent
map of the Odaw River Basin, Ghana, and (B) InSAR line-of-sight displacement map
produced using PyGeoFetch's SNAPHU-based unwrapping.*

The platform underpins EOCoreINT, a structured learning programme delivering 20
courses in Earth observation and geospatial AI to researchers and government analysts
across Africa.

# AI Usage Disclosure

Large language model assistance (Claude, Anthropic) was used during the development of
`pygeovision` for: generating boilerplate code patterns, drafting docstrings, and
suggesting edge-case test scenarios. All generated code was reviewed, tested, and
validated against real satellite data before inclusion. No AI tool was used to
generate scientific results, accuracy benchmarks, or claims about model performance.
This paper was written by the author; AI assistance was used for grammar checking
only.

# Acknowledgements

The author thanks the Ghana Environmental Protection Agency GIS Unit and the National
Disaster Management Organisation (NADMO) for operational testing of the flood
monitoring pipeline. Satellite data was provided free of charge through the European
Space Agency Copernicus Programme and the NASA/USGS Landsat programme. The
Prithvi-EO-2.0 model weights were made available by IBM Research and NASA
[@jakubik2023prithvi]. DINOv2 model weights were made available under open licence by
Meta AI [@oquab2023dinov2].

# References