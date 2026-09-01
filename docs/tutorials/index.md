# Tutorials

**Verification status**: the markdown tutorials below and the 22 Jupyter
notebooks in `docs/tutorials/*.ipynb` were not independently re-run or
verified this cycle — that would mean actually executing each notebook
against real data, a separate, larger undertaking from the markdown-page
audit this cycle covered. Treat their code the way you'd treat any
unverified example: check it against the real, current API before relying
on it, particularly anywhere it touches SAR/InSAR (removed this cycle —
see [Architecture](../architecture.md)) or the `client.pipeline()` /
`client.geoai` namespaces (the latter does not exist).

Step-by-step guides for common geospatial AI workflows. Each tutorial is self-contained and includes working code.

| Tutorial | Time | Difficulty |
|----------|------|------------|
| [Getting Started](getting_started.md) | 5 min | Beginner |
| [Authentication](authentication.md) | 10 min | Beginner |
| [Data Search](data_search.md) | 15 min | Beginner |
| [Data Download](data_download.md) | 15 min | Beginner |
| [Building Extraction](building_extraction.md) | 30 min | Intermediate |
| [Land Cover Mapping](land_cover.md) | 30 min | Intermediate |
| [Change Detection](change_detection.md) | 30 min | Intermediate |
| [Custom Training](custom_training.md) | 45 min | Intermediate |
| [Foundation Models](foundation_models.md) | 45 min | Advanced |
| [Deployment](deployment.md) | 30 min | Advanced |
