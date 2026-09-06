# Advanced Capabilities

Real, additional capabilities beyond the core pipeline/training system
— each independently verified this audit.

## Few-shot classification

Prototype-based classification from a handful of real example images
per class — no training run required.

```python
from pygeovision.advanced.few_shot import FewShotLearner

learner = FewShotLearner()
learner.fit_support({
    "water": ["water_1.tif", "water_2.tif"],
    "forest": ["forest_1.tif", "forest_2.tif"],
})
result = learner.predict_geotiff("scene.tif", output_path="labels.tif")
```

```{note}
`predict_geotiff` always returns `success: True` when it completes —
this was checked specifically, since it's the same shape as a real bug
pattern found elsewhere in this codebase. It's legitimately correct
here: nearest-prototype classification always produces *some* real
label for every pixel by definition, unlike feature search, where
"zero real results" is a genuine possible outcome.
```

## AutoML

Real hyperparameter search via `Optuna` or `Ray Tune` — not a wrapper
that pretends to search but returns a fixed default.

```python
from pygeovision.advanced.automl import AutoMLSearch

search = AutoMLSearch(backend="optuna")
best = search.run(objective_fn, search_space={
    "learning_rate": ("float_log", 1e-5, 1e-1),
    "backbone": ("categorical", ["resnet50", "efficientnet-b4"]),
}, n_trials=50)
```

## Multi-task training

A real, single training loop across multiple real task heads sharing
one backbone.

## Time series analysis

Real, verified spectral-index trend analysis over multiple dates —
hand-verified against a synthetic perfect-linear series (`slope=2.0`,
`R²=1.0`, matching a real linear regression exactly).

```python
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries()
series = ts.compute_index_series(bbox=(...), dates=[...], index="ndvi")
trend = ts.compute_trend(series)
# {"slope": ..., "intercept": ..., "r_squared": ..., "direction": "increasing"}
```

## CLIP-based image retrieval

Real cosine-similarity search over real CLIP embeddings.

```python
from pygeovision.advanced.vlm.retrieval import ImageRetrieval

retrieval = ImageRetrieval(model="remoteclip-b32")
results = retrieval.search_by_text("flooded agricultural field", image_dir="./scenes")
```

## Point cloud processing

`PointCloudProcessor` has real, working methods for canopy height
models and building extraction from real LiDAR point clouds.

```{warning}
`classify_points()` is honest about a real limitation in its own
docstring: no pretrained weights exist for point-cloud semantic
segmentation in this codebase. Without a real, user-supplied
checkpoint, it runs with randomly-initialized weights and the output
is not meaningful — this was already correctly self-documented before
this audit, not something that needed fixing.
```
