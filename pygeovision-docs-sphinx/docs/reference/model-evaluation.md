# Model Evaluation & Leaderboards

`ModelEvaluator` runs a real model against a real `DataLoader` and
returns real metrics — built on the same `SegmentationMetrics` and
`torchmetrics`-backed `DetectionMetrics` verified in
[Training](../training/index.md), not a separate, unverified
implementation.

```python
from pygeovision.benchmark.evaluator import ModelEvaluator

evaluator = ModelEvaluator(task="segmentation", num_classes=5)
result = evaluator.evaluate(model, test_loader, dataset_name="LoveDA")

results = evaluator.compare(
    models={"UNet": unet_model, "SegFormer": segformer_model},
    loader=test_loader, dataset_name="LoveDA",
)
evaluator.print_leaderboard(results)
evaluator.save_results(results, "results.json")
```

Real per-model latency and throughput are measured alongside accuracy
metrics, so `compare()` produces a genuine accuracy/speed leaderboard,
not just an accuracy ranking.
