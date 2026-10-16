# Experiments

An **experiment** is a question you answer with a run: *does a larger model help? which learning rate works? is the new
loss better on the rare class?* Doing this well means recording what you ran, comparing like with like, and not fooling
yourself with a lucky number. This section collects the four tools for it.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Tracking
:link: tracking
:link-type: doc
Record settings and per-epoch metrics with TensorBoard, MLflow or Weights & Biases.
:::

:::{grid-item-card} Hyper-parameter search
:link: search
:link-type: doc
`OptunaHPO`: try many learning rates and batch sizes around one `GeoTrainer` run.
:::

:::{grid-item-card} AutoML
:link: automl
:link-type: doc
`GeoAutoML`: search any training function, with pruning and failed-trial handling.
:::

:::{grid-item-card} Benchmark
:link: benchmark
:link-type: doc
Run several models on the same data and rank them on one leaderboard.
:::
::::

## How the pieces fit

```{figure} ../_static/img/flow-experiments.svg
:alt: Experiment loop: propose, train, track, compare, choose
:width: 100%
:class: pgv-figure

The experiment loop. Tracking records every run; search proposes the next one; the benchmark compares finished models.
```

| You want to… | Use | Page |
|---|---|---|
| Remember which settings gave which curve | A tracker (`tracker="tensorboard"`, `"mlflow"` or `"wandb"`) | [Experiment tracking](tracking.md) |
| Tune the learning rate, weight decay or batch size of a `GeoTrainer` run | `OptunaHPO` | [Hyper-parameter search](search.md) |
| Tune anything, including non-network code, with pruning and crash-tolerant trials | `GeoAutoML` | [AutoML](automl.md) |
| Compare U-Net, SegFormer and DeepLab on your data | `GeoBenchmark` | [Benchmarking](benchmark.md) |
| Read the ranking, export it, and use the command line | The leaderboard and `pygeovision benchmark` | [Leaderboard](leaderboard.md) |

## Good experimental practice

1. **Change one thing at a time.** If you change the model and the learning rate together you cannot say which helped.
2. **Fix the seed** (`TrainingConfig(seed=42)`) and repeat important runs with two or three seeds. A difference smaller than
   the spread between seeds is not a difference.
3. **Keep one validation set that never changes**, and one **test set** that you look at once, at the end. Choosing the best of
   many runs on the validation set makes the validation score optimistic.
4. **Compare on the right metric.** For a rare class, read the per-class score, not the mean ([why](../segmentation/train.md#read-the-per-class-values)).
5. **Record the data too.** Which scenes, which dates, which split. Settings alone do not reproduce a result.
6. **Write down failures.** A search that tried 20 settings and found nothing is information.

## What this section does not do

- It does not choose your metric or tell you whether a difference is meaningful. That is your judgement and, for important
  claims, statistics on repeated runs.
- It does not track data versions. Keep the list of training scenes with the run.
- A leaderboard on one dataset is not a ranking of models in general ([what the numbers can and cannot tell you](benchmark.md#what-the-numbers-can-and-cannot-tell-you)).

Related: [Training configuration](../training/config.md) · [Callbacks and custom metrics](../training/callbacks.md) ·
[Monitoring a deployed model](../monitoring/index.md)
