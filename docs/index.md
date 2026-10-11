# PyGeoVision

**Geospatial AI from search to finished map.** PyGeoVision finds satellite imagery (through
[pygeofetch](https://pygeofetch.readthedocs.io/en/latest/)), prepares it correctly, runs an analysis, and writes the
deliverables: rasters, vectors, tables and a report. It also contains the machinery to label, train, run, explain and
monitor neural networks on imagery.

| | |
|---|---|
| **40 pipelines** | 33 run out of the box (no deep-learning library needed); 7 run *your* trained network and refuse to start without it |
| **22 data providers** | Sentinel-1/2, Landsat, MODIS, DEMs, reference land-cover and building products, OpenStreetMap … |
| **3 model lists** | 79 buildable architectures · 14 names for your checkpoints · 96 catalogue entries ([why three](models/registry.md)) |
| **25 CLI groups, 92 commands** | Every one documented with a captured `--help`, when to use it, what you get and what to watch for |
| **Python API** | `import pygeovision as pgv`: the same tools, scriptable |

```{admonition} How to read this documentation
:class: tip
Every page answers four questions: **why** you would use the tool, **how** to use it, **what output to expect**, and
**what to do** when it does not behave. Example outputs are captured from real runs; where an example uses synthetic
data, or a feature could not be run in the documentation environment (no network, no GPU, an optional package), the page
says so. A short "Verified here?" line at the top of the less common tools tells you which.
```

## Start here

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Install
:link: getting-started/installation
:link-type: doc
Python, optional extras, credentials, `pygeovision doctor`.
:::

:::{grid-item-card} Quick start
:link: getting-started/quickstart
:link-type: doc
From an empty folder to a first flood or vegetation map.
:::

:::{grid-item-card} Core concepts
:link: getting-started/concepts
:link-type: doc
Runs, pipelines, checkpoints, class codes, nodata, the 0/255 convention.
:::

:::{grid-item-card} Pick a pipeline
:link: pipelines/choosing
:link-type: doc
"I want to map X" to the pipeline that does it, with its limits.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc
Domain workflows: agriculture, forestry, urban, water, disaster, climate.
:::

:::{grid-item-card} Something went wrong
:link: getting-started/troubleshooting
:link-type: doc
Symptom-first fixes, plus the [FAQ](getting-started/faq.md).
:::
::::

## What PyGeoVision will and will not do

- It **does** run analyses whose method is documented and whose output was checked against a known answer, and tells
  you when it cannot (no data, too much cloud, no checkpoint) instead of producing a plausible-looking map.
- It **does not** ship trained task models (crop type, solar panels, species …). It ships architectures, and fetches
  public pretrained weights where a genuine checkpoint exists. A pipeline that needs your weights says so before
  downloading any imagery.
- It **does not** give accuracy figures it has not measured. Accuracy sections of the pages tell you how to measure it
  on your own reference data ([validation](data/validation.md), [OBIA accuracy](obia/accuracy.md)).

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting-started/installation
getting-started/quickstart
getting-started/concepts
getting-started/configuration
getting-started/troubleshooting
getting-started/faq
architecture
```

```{toctree}
:maxdepth: 2
:caption: Data
:hidden:

data/index
datasets/index
```

```{toctree}
:maxdepth: 2
:caption: Pipelines
:hidden:

pipelines/index
```

```{toctree}
:maxdepth: 2
:caption: Models and training
:hidden:

models/index
labeling/index
training/index
inference/index
cli/direct-inference
```

```{toctree}
:maxdepth: 2
:caption: Object-based analysis
:hidden:

obia/index
```

```{toctree}
:maxdepth: 2
:caption: Trust, monitor, deploy
:hidden:

explainability/index
monitoring/index
benchmark/index
edge-cloud/index
reporting/index
```

```{toctree}
:maxdepth: 2
:caption: Advanced tools
:hidden:

advanced/index
```

```{toctree}
:maxdepth: 1
:caption: Examples
:hidden:

examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/cli/index
reference/python-api
reference/exceptions
reference/config
reference/glossary
reference/changelog
reference/roadmap
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

contributing
```
