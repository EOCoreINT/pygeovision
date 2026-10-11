# Advanced tools

Six tools that sit beside the main pipelines. None of them is needed for the everyday workflow
([get data → label → train → infer → report](../getting-started/concepts.md)); each solves one specific problem, and each has
limits that are easy to miss. Read the "What it is not" line of a tool before you plan a project around it.

| You want to… | Use | What it is **not** | Page | Verified here? |
|---|---|---|---|---|
| Find good hyper-parameters for *any* training function | `GeoAutoML` (Optuna) | A model designer. It tunes the numbers your function exposes | [AutoML](automl.md) | **Yes**: search, pruning, failure handling, importance (toy objective). Ray Tune path: no |
| Classify image chips with 2–10 labelled examples per class | `FewShotLearner` (frozen DINOv2 + prototypes) | A trained classifier. Nothing is learned, and "none of these" is not an answer | [Few-shot](few-shot.md) | Logic, GeoTIFF output and the embedding path with a tiny random backbone. **Not** the accuracy of real DINOv2 |
| Train one network for several tasks (segmentation + classification …) | `MultiTaskLearner` | A multi-object detector or a siamese change model | [Multi-task](multitask.md) | **Yes**: a real CPU training run on synthetic chips |
| Describe, question or search images in words | `MoondreamGeo`, `CLIPGeo`, `GeoImageRetrieval` | A measuring tool. The answers are plausible text, not counts or areas | [Vision-language](vlm.md) | Image loading, index format, CLI errors. **Not** model outputs (weights are downloaded from Hugging Face; no network here) |
| Follow an index (NDVI …) through a stack of dated images; flag odd dates; fit a trend | `GeoTimeSeries` | A per-pixel analysis. One number per date for the whole image | [Time series](timeseries.md) | **Yes**, on scenes with a known answer |
| Make a canopy height model, label LiDAR points, outline buildings from a point cloud | `PointCloudProcessor` | A pretrained LiDAR classifier. You must supply the trained network | [Point clouds](pointcloud.md) | **Yes** (real LAS files). Classification is verified as plumbing only |

```{admonition} Three things that apply to every page here
:class: important
1. **The examples run.** Each page includes a script from `docs/advanced/_examples/`; the output shown was produced by
   running it. Where an example uses synthetic data or a stand-in model, the page says so, and the numbers are *not* a
   claim about accuracy on real imagery.
2. **Model downloads need a network.** `FewShotLearner` (DINOv2), `MoondreamGeo`, `CLIPGeo` and pretrained backbones fetch
   weights from the Hugging Face Hub or torchvision on first use. On an air-gapped machine copy the model folder over and
   pass its path (few-shot) or pre-populate the cache.
3. **Optional packages.** `optuna` (AutoML), `transformers` (few-shot, VLM), `open_clip_torch` (RemoteCLIP, GeoRSCLIP),
   `laspy` (+ `lazrs` for `.laz`), `scipy`, `plotly` (AutoML plots). A missing one raises an `ImportError` that names it.
```

## Choosing between them

```{list-table}
:header-rows: 1
:widths: 34 66

* - Situation
  - Better choice
* - "I have 300 labelled chips and want a classifier"
  - Train a normal classifier ([training](../training/index.md)). Few-shot is for when you cannot label more
* - "I need pixel-accurate maps"
  - A segmentation model ([models](../models/tasks.md)). Few-shot maps are as coarse as the chip spacing
* - "How much did vegetation change at every pixel?"
  - The [`vegetation_anomaly`](../pipelines/catalog/vegetation_anomaly.md) and
    [`water_extent_timeseries`](../pipelines/catalog/water_extent_timeseries.md) pipelines. `GeoTimeSeries` gives one
    number per date
* - "Which of these 5 000 tiles show flooding?"
  - Zero-shot CLIP ranking to *prioritise* what you look at, then check. Do not count the ranking as a result
* - "Tune the learning rate and loss for my U-Net"
  - [`OptunaHPO`](../training/advanced.md#hyper-parameter-search) if the trial is one `GeoTrainer` run;
    [`GeoAutoML`](automl.md) for any other function, or when trials may crash or should be pruned
```

```{toctree}
:maxdepth: 1
:hidden:

automl
few-shot
multitask
vlm
timeseries
pointcloud
```

Related: [Foundation models](../models/foundation.md) · [Training](../training/index.md) ·
[Inference](../inference/index.md) · [Explainability](../explainability/index.md)
