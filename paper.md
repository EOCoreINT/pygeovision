---
title: 'pygeovision: A Unified Open-Source Python Platform for Satellite Earth Observation AI'
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
date: 10 July 2026
version: 2.0.9
bibliography: paper.bib
---

# Summary

`pygeovision` is an open-source Python library that provides a unified,
production-ready interface for satellite Earth observation (EO) analysis and
geospatial AI inference. Built on top of `pygeofetch` — a companion library
that abstracts data access across 22+ satellite data providers — `pygeovision`
covers the complete EO workflow from raw satellite acquisition to AI-derived
geospatial intelligence: preprocessing pipelines for optical and SAR imagery,
spectral index computation, cloud masking, change detection, flood mapping,
InSAR deformation analysis, and inference via foundation models including
Prithvi-EO-2.0 [@jakubik2023prithvi] and DINOv3 [@oquab2023dinov2].

A natural-language query interface (GeoAgent) allows users to describe an
analysis in plain English and have the system automatically select the
appropriate sensor, task, and tool sequence. Together, `pygeovision` and
`pygeofetch` are designed to make operational EO analysis accessible to
researchers and government analysts in Africa and the Global South, where
in-situ monitoring infrastructure is sparse but satellite coverage is complete.

# Statement of Need

Satellite Earth observation is uniquely well-suited to address monitoring
challenges in data-sparse environments. Every point on Earth is covered by
free Sentinel-1 SAR imagery every 6–12 days and by free Sentinel-2 optical
imagery every 5 days [@drusch2012sentinel2; @torres2012sentinel1]. Yet the gap
between data availability and operational use remains wide: acquiring,
preprocessing, and interpreting multi-sensor satellite data requires expertise
across geodesy, signal processing, machine learning, and domain science that
is difficult to assemble in resource-constrained institutions.

Existing tools address parts of this gap. Google Earth Engine [@gorelick2017gee]
provides cloud-based EO processing but requires a proprietary account,
processes data on infrastructure outside users' control, and produces outputs
whose computational lineage cannot be fully audited — a disqualifying
constraint for government monitoring systems subject to legal challenge.
The ESA Sentinel Application Platform (SNAP) handles SAR preprocessing but
has no Python-native API and no AI inference capability. Standalone deep
learning frameworks (PyTorch, TensorFlow) require users to assemble their own
geospatial data pipeline. No existing open-source package integrates all
layers — data access, preprocessing, AI inference, and production
orchestration — in a single auditable, locally deployable system.

`pygeovision` fills this gap. Its target users are:

- **Government analysts** (national disaster management, environmental
  protection agencies, forestry commissions) who need reproducible,
  audit-ready flood, deforestation, and land cover products without
  depending on proprietary cloud platforms.
- **Academic researchers** in Africa and the Global South who need a
  full EO analysis stack with minimal infrastructure investment.
- **NGO programme officers** running food security, disaster risk reduction,
  or climate adaptation programmes who require satellite-derived evidence
  integrated into their existing reporting workflows.

# State of the Field

Several open-source packages address subsets of the `pygeovision` scope.
`rasterio` [@gillies2019rasterio] and `geopandas` [@kelsey2020geopandas]
provide foundational raster and vector I/O but no domain-specific EO workflows.
`sentinelsat` and `pystac` [@stac2021] provide access to specific satellite
catalogues but do not normalise across providers or handle preprocessing.
`eo-learn` [@sentinelhub2021eolearn] provides a workflow framework for
Sentinel Hub (a commercial service) with limited support for free Copernicus
direct access. `torchgeo` [@stewart2022torchgeo] supplies EO-specific PyTorch
datasets and transforms but has no data acquisition layer and limited
preprocessing support.

At the foundation model level, `terratorch` [@terratorch2024] provides a
fine-tuning framework for geospatial foundation models but requires users to
source and preprocess their own data. `pygeovision` wraps foundation model
inference — including Prithvi-EO-2.0 [@jakubik2023prithvi] and DINOv3
[@oquab2023dinov2] — within the same package that acquires and preprocesses
the input data, removing the integration burden that currently prevents
operational adoption.

The closest integrated alternative is `odc-stac` combined with `datacube`,
but these require significant server infrastructure (ODC database, Kubernetes)
that is inaccessible to most African research institutions. `pygeovision`
runs on a standard laptop or a $12/month virtual machine.

# Software Design

## Architecture

`pygeovision` is structured in four layers, each independently usable:

**Data layer (`pygeovision.data`)** wraps `pygeofetch` to provide a
provider-agnostic search and download API. `pygeofetch` normalises across
22+ providers — including Copernicus Data Space, Microsoft Planetary Computer,
USGS EarthExplorer, and ASF Vertex — behind a single interface. The data
layer handles authentication, caching, SHA-256 checksum verification,
partial-download detection, and coordinate reference system validation after
reprojection. Four silent bugs discovered during production flood response
work over Accra are guarded against explicitly: CRS identity-transform
corruption, partial download truncation, COG vs SAFE download format
mismatch (HTTP 422), and WGS84/UTM clipping mismatch.

**Processing layer (`pygeovision.data.processors`)** implements sensor-specific
preprocessing pipelines. The SAR pipeline applies nine ordered steps (S0–S9)
to Sentinel-1 GRD products: download verification, georeference validation,
band selection, resampling, co-registration, study area clipping, Lee speckle
filtering on linear power data, dB conversion, and normalisation. The optical
pipeline applies the Sentinel-2 Scene Classification Layer (SCL) for cloud
masking, atmospheric correction validation, and computation of 12 spectral
indices. Processing order is enforced: speckle filtering must precede dB
conversion because the Lee filter's Gamma noise model applies only to linear
power data, not logarithmic dB values — a common source of silent errors in
published SAR workflows.

**AI layer (`pygeovision.models`)** provides inference wrappers for:
foundation models (Prithvi-EO-2.0 with five task heads: land cover, crop
mapping, flood, burn scar, and biomass regression; DINOv3 with six task
heads including CHMv2 canopy height estimation calibrated against GEDI
LiDAR); change detection models (ChangeFormer [@bandara2022changeformer]);
and custom model training via `GeoTrainer` (FocalDice loss, mixed-precision
BF16 training, ONNX export for CPU deployment). A band adapter normalises
Sentinel-2 inputs to the HLS canonical six-band order required by
Prithvi-EO-2.0.

**InSAR layer (`pygeovision.insar`)** implements true SLC InSAR processing
via a Python API to ESA SNAP [@snap2024] and SNAPHU
[@chen2002snaphu]. The pipeline runs the eleven-step TOPSAR
interferometric chain — TOPSAR-Split through terrain correction — and
converts unwrapped phase to line-of-sight displacement using
$d_\mathrm{LOS} = \frac{\lambda}{4\pi} \varphi$, where $\lambda = 0.05547$ m
for Sentinel-1 C-band. A documented bug in the literature uses 170° as the
descending satellite heading; at this value $\sin(170°) \approx \sin(10°)$,
making the 2×2 LOS decomposition matrix nearly singular. The correct
descending heading is 350°, and `pygeovision` enforces this.

## GeoAgent

The GeoAgent maps natural-language queries to complete analysis pipelines
through a three-stage decision hierarchy: sensor classification (SAR vs
optical, based on 40+ signal word sets), task classification (14 task types
including flood, damage, change, InSAR, and subsidence), and tool sequence
planning (a 14×2 decision matrix mapping task×sensor to ordered tool calls).
The heuristic planner achieves 86% task routing accuracy without any external
API key. With an LLM backend (Claude or GPT-4), accuracy rises to 96% and
multi-intent queries are supported.

## Key Dependencies

`pygeovision` depends on: `pygeofetch` (data access), `rasterio`
[@gillies2019rasterio], `geopandas` [@kelsey2020geopandas], `NumPy`
[@harris2020numpy], `PyTorch` [@paszke2019pytorch], `scikit-learn`
[@pedregosa2011sklearn], and `FastAPI` for optional REST API deployment.
All dependencies are freely available and OSI-licensed.

# Research Impact Statement

`pygeovision` has been applied to operational flood monitoring over Ghana's
Odaw River Basin, producing SAR-derived flood extent maps within 18 hours
of a Sentinel-1 overpass during the June 2024 Accra flood event, with
847 hectares of inundation detected and community-level impact assessed
against census population data. Bi-temporal Sentinel-2 change detection
using ChangeFormer [@bandara2022changeformer] detected 234.7 hectares of
newly cleared forest in Ghana's Atewa Range Forest Reserve between 2020
and 2024. SLC InSAR processing of Sentinel-1 SLC pairs over the 2023
Kahramanmaraş earthquake achieved 2.0 cm mean absolute error on
line-of-sight displacement against GPS CORS ground truth, consistent
with published results from SNAP-based processing chains.

The platform underpins EOCoreINT, a structured learning programme delivering
20 courses in Earth observation and geospatial AI to researchers and
government analysts across Africa. Reproducibility is a design constraint:
every analysis run produces a provenance JSON record documenting input scene
IDs, software versions, and all processing parameters, enabling independent
verification of results submitted to government agencies or used as evidence
in legal proceedings.

# AI Usage Disclosure

Large language model assistance (Claude, Anthropic) was used during the
development of `pygeovision` for: generating boilerplate code patterns,
drafting docstrings, and suggesting edge-case test scenarios. All generated
code was reviewed, tested, and validated against real satellite data before
inclusion. No AI tool was used to generate scientific results, accuracy
benchmarks, or claims about model performance. This paper was written by the
author; AI assistance was used for grammar checking only.

# Acknowledgements

The author thanks the Ghana Environmental Protection Agency GIS Unit and the
National Disaster Management Organisation (NADMO) for operational testing of
the flood monitoring pipeline. Satellite data was provided free of charge
through the European Space Agency Copernicus Programme and the NASA/USGS
Landsat programme. The Prithvi-EO-2.0 model weights were made available by
IBM Research and NASA [@jakubik2023prithvi]. DINOv3 model weights were made
available under open licence by Meta AI [@oquab2023dinov2].

# References
