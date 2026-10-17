# PyGeoVision

:::{container} pgv-lead
**Geospatial AI from search to finished map.** One CLI and one Python API for finding satellite imagery, preparing it correctly,
running an analysis and delivering rasters, vectors, tables and a report.
:::

```bash
pip install "pygeovision[geo,report]"
pygeovision doctor
```

:::{container} pgv-hero
![PyGeoVision data flow, from 22 data providers through search and preparation to 41 pipelines or models and OBIA, then checks and deliverables](_static/img/workflow.svg)
:::

::::{grid} 2 2 4 4
:gutter: 2
:class-container: pgv-stats

:::{grid-item-card} 41
:class-card: pgv-stat
:class-title: pgv-stat-n

pipelines: 34 run out of the box, 7 run your trained network
:::

:::{grid-item-card} 22
:class-card: pgv-stat
:class-title: pgv-stat-n

data providers, through pygeofetch
:::

:::{grid-item-card} 92
:class-card: pgv-stat
:class-title: pgv-stat-n

CLI commands in 25 groups, every one documented
:::

:::{grid-item-card} 79
:class-card: pgv-stat
:class-title: pgv-stat-n

buildable model architectures
:::
::::

## What it does

PyGeoVision answers a question about a place and a time. You give it an area, a date and a pipeline name; it finds the clearest
satellite scenes (through [pygeofetch](https://pygeofetch.readthedocs.io/en/latest/)), downloads only the bands the method needs,
masks cloud and shadow, computes the result and records exactly what it used. The answer comes back as a GeoTIFF you can open in
a GIS, a GeoPackage of polygons, a map, a table of figures and a PDF, Word and Excel report.

Beyond the ready-made pipelines, the same package lets you build labels, train a network on your own classes, run it over a scene or
an archive, explain a prediction, watch a deployed model for drift and export it to an edge device or the cloud. This documentation is
also available in French, Spanish, Portuguese, German, Chinese and Arabic: use the globe icon in the top bar.

### Your first analysis

```bash
pygeovision doctor                                   # check the installation
pygeovision channel water_bodies --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --output ./first_run
cat first_run/run_summary.json                       # read what the run measured before you look at the map
```

The [Quickstart](getting-started/quickstart.md) explains each step and each file in ten minutes, with no account.

## Where to go

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/quickstart
:link-type: doc

Install, run a first pipeline and read its result.
:::

:::{grid-item-card} Guides
:link: guides/index
:link-type: doc

Choose your area and dates, script it, run many areas, use the results in a GIS.
:::

:::{grid-item-card} Pipelines
:link: pipelines/choosing
:link-type: doc

Go from "I want to map X" to the one of 41 pipelines that does it.
:::

:::{grid-item-card} Data
:link: data/index
:link-type: doc

Find, download, mask and prepare imagery from 22 providers.
:::

:::{grid-item-card} Models
:link: models/index
:link-type: doc

Segmentation, classification, detection, change, foundation and language models.
:::

:::{grid-item-card} Train and deploy
:link: training/index
:link-type: doc

Label, train, predict, explain, monitor and ship your own model.
:::

:::{grid-item-card} Case studies
:link: case-studies/index
:link-type: doc

Eight real runs read closely: floods, a wildfire, streams, water quality and London.
:::

:::{grid-item-card} Reference
:link: reference/cli/index
:link-type: doc

All 92 commands, every setting, the error messages and the known issues.
:::

:::{grid-item-card} Something went wrong
:link: getting-started/troubleshooting
:link-type: doc

Symptom-first fixes, plus the [FAQ](getting-started/faq.md).
:::
::::

## How the documentation is organised

The sidebar has nine groups. Open one to see its pages.

| Group | Use it to |
|---|---|
| **Get started** | Install PyGeoVision, run a first pipeline, learn the core ideas, and fix problems. |
| **Guides** | Follow a task from start to finish, and see complete workflows by domain. |
| **Data** | Find and download imagery; prepare, mask, index and validate it. |
| **Pipelines** | Choose among the 41 pipelines, read each on its own page, and chain them in YAML. |
| **Models** | Use segmentation, classification, detection, change detection, foundation, vision-language, LiDAR, super-resolution and time-series models, or object-based analysis. |
| **Train and deploy** | Label data, train, predict, track experiments, explain, monitor and deploy a model. |
| **Case studies** | Read eight real reports closely: what each number means and what the run can and cannot support. |
| **Reference** | Look up commands, settings, exceptions, the glossary and known issues. |
| **Project** | Read the changelog and roadmap, and learn how to contribute. |

## What PyGeoVision will and will not do

- It **does** run analyses whose method is documented and whose output was checked against a known answer, and tells
  you when it cannot (no data, too much cloud, no checkpoint) instead of producing a plausible-looking map.
- It **does not** ship trained task models (crop type, solar panels, species …). It ships architectures, and fetches
  public pretrained weights where a genuine checkpoint exists. A pipeline that needs your weights says so before
  downloading any imagery.
- It **does not** give accuracy figures it has not measured. Accuracy sections of the pages tell you how to measure it
  on your own reference data ([validation](data/validation.md), [OBIA accuracy](obia/accuracy.md)).

:::{note}
The [Known issues](reference/known-issues.md) page lists behaviours that differ from what you might expect, and the less common
tools carry a "Verified here?" line saying what they were run against.
:::

```{toctree}
:maxdepth: 1
:caption: Get started
:hidden:

getting-started/installation
getting-started/quickstart
getting-started/concepts
getting-started/configuration
getting-started/troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Guides
:hidden:

guides/index
examples/index
```

```{toctree}
:maxdepth: 1
:caption: Data
:hidden:

data/index
```

```{toctree}
:maxdepth: 1
:caption: Pipelines
:hidden:

pipelines/index
pipelines/choosing
pipelines/catalog/index
pipelines/results
reporting/index
pipelines/yaml-workflows
pipelines/custom
```

```{toctree}
:maxdepth: 1
:caption: Models
:hidden:

models/index
segmentation/index
classification/index
detection/index
change-detection/index
foundation-models/index
vision-language/index
lidar/index
super-resolution/index
timeseries/index
obia/index
```

```{toctree}
:maxdepth: 1
:caption: Train and deploy
:hidden:

labeling/index
training/index
inference/index
experiments/index
explainability/index
monitoring/index
deployment/index
```

```{toctree}
:maxdepth: 1
:caption: Case studies
:hidden:

case-studies/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

reference/cli/index
reference/config
reference/exceptions
reference/glossary
reference/known-issues
reference/security
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

project/changelog
project/roadmap
project/contributing
```
