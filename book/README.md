# The PyGeoVision Book

**The definitive, comprehensive, production-ready guide to PyGeoVision — 
the world's most complete open-source geospatial AI platform.**

---

## About this Book

PyGeoVision brings together satellite data acquisition (22+ providers),
AI inference (119 model architectures), SAR processing, InSAR analysis,
and an autonomous AI agent — all behind a single, coherent Python API.

This book teaches you to use every part of that platform, from a five-minute
quick start to deploying production inference pipelines on AWS EKS.

## Structure

| Part | Chapters | Topics |
|------|----------|--------|
| I: Foundations | 1–4 | Installation, data acquisition, preprocessing, spectral indices |
| II: AI & Machine Learning | 5–11 | Classification, detection, segmentation, change detection, regression |
| III: Advanced Topics | 12–16 | Foundation models, SAR, InSAR, time-series |
| IV: Visualization | 17–19 | Interactive maps, InSAR viz, reporting |
| V: Automation & Deployment | 20–22 | YAML pipelines, GeoAgent, cloud deployment |
| VI: Applications | 23–28 | Agriculture, forestry, urban, water, disaster, climate |
| VII: Reference | A–F | API, CLI, config, troubleshooting, contributing, glossary |

## How to Use this Book

**Beginners:** Start with Chapters 1–4. Every chapter has a companion
Jupyter notebook in `notebooks/` — run them alongside the text.

**AI practitioners:** Jump to Chapters 5–13 for the complete AI stack,
or Chapter 21 for the GeoAgent natural-language interface.

**SAR specialists:** Chapters 14–15 cover the corrected SAR preprocessing
pipeline (3 production bug-fixes) and the complete InSAR chain.

**Domain experts:** Part VI (Chapters 23–28) applies PyGeoVision to
specific domains with real-world case studies.

## Requirements

```bash
pip install "pygeovision[all]"
jupyter lab
```

## Version

This book covers **PyGeoVision v2.1.2** (July 2026).

---

*PyGeoVision Contributors · Apache 2.0 License · https://pygeovision.org*
