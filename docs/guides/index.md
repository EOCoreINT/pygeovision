# Task guides

Task-focused walkthroughs. Each takes one job from start to finish and links to the reference pages for the detail. If you are new,
read the [Quickstart](../getting-started/quickstart.md) first; it takes ten minutes and needs no account.

## I want to…

| Task | Start here |
|---|---|
| Map something for my area and read the answer | [Quickstart](../getting-started/quickstart.md), then [Choose a pipeline](../pipelines/choosing.md) |
| Decide the box and the dates for a run | [Choose your area and dates](your-area.md) |
| Script an analysis, or use the results in code | [Use PyGeoVision from Python](python.md) |
| Run many areas or dates, or run every week | [Run many areas and dates](automate.md) |
| Open the results in QGIS, share a report, compare two runs | [Use the results](use-results.md) |
| Decide whether a number can be trusted | [Reading results](../pipelines/results.md) and the [case studies](../case-studies/index.md) |
| Map my own classes with a network I train | [Train a model and use it](own-model.md) |
| See a complete workflow for my field | [Examples by domain](../examples/index.md) |
| Look up a command or a setting | [Command-line reference](../reference/cli/index.md), [Configuration](../reference/config.md) |
| Fix an error | [Troubleshooting](../getting-started/troubleshooting.md) |

## How the pieces fit

```{figure} ../_static/img/workflow.svg
:alt: PyGeoVision data flow, from data providers through search and preparation to pipelines or models, then checks and deliverables
:width: 100%
:class: pgv-figure

Data in, a question answered, a checked result out. Everything in these guides is one of these steps.
```

There are three ways to work, and most people start at the first:

1. **Run a ready pipeline.** You give a place, a date and a pipeline name. PyGeoVision finds the imagery, prepares it, computes the
   answer by a documented method, and writes rasters, polygons and a report. No training, no model. This covers 34 questions from
   floods to land cover.
2. **Run a pipeline on your own network.** For a question with no published method (your crop types, your roof materials), you
   [train a model](own-model.md) and run it inside the same pipeline machinery, so you still get the report and the run record.
3. **Use the parts.** Search, download, preprocess, label, train, predict, explain and report are separate tools. Use any of them
   alone, from Python or the command line.

## Related sections

- [Pipelines](../pipelines/index.md): what each of the 41 pipelines does, and how to choose.
- [Data](../data/index.md): finding, preparing and checking imagery.
- [Examples by domain](../examples/index.md): agriculture, forestry, cities, water, disasters, mining and climate.
- [Case studies](../case-studies/index.md): eight real reports read closely.

```{toctree}
:hidden:
:maxdepth: 1

your-area
python
automate
use-results
own-model
```
