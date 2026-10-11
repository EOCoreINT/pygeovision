# Changelog

What changed in behaviour, grouped by area. This release (2.1.8) is the result of documenting every feature by running it: each
item below was found because an example did not do what its page said it would, and each has a test. Entries that **change what you
get** (a different default, an error where there used to be silence) are marked **behaviour change**. For the
older, longer narrative of audits see the [roadmap](roadmap.md).

## 2.1.8

### Pipelines, data and labeling

- **Labeling.** The tile OpenStreetMap labeler was rewritten; its constants `OSM_BUILDING_CLASSES`, `OSM_LANDUSE_CLASSES`, `OSM_ROAD_CLASSES` were
  removed in favour of `feature_type` tables. The Microsoft Buildings tile labeler uses a consistent grid. Area labeling is fixed.
- **Behaviour change.** `ActiveLearner` has a new default `n_select`. `AutoLabelPipeline` refuses to combine sources with incompatible class alphabets.
  `LabelQualityAssessor` accepts `num_classes=None`. OSM roads use a corrected pattern and categories. The Planetary Computer Dynamic World default year is 2023.
  `LabelingOrchestrator` raises instead of returning partial success. Dynamic World probability options on the `ai` labeler raise `NotImplementedError`.
  Google and Microsoft building caps are reported in the result.
- A scene split with very few scenes no longer produces an empty set. `obia classify` reports clean errors.
- **Pipelines.** The thermal example data used by the documentation harness now gives realistic temperatures.

### Training, inference and augmentation

- Fixes in the training loop, inference engines, augmentation, and the ONNX list; GeoTIFF inference takes `num_classes`.
- `LabelSmoothingCrossEntropy` formula corrected.
- **Behaviour change.** ONNX export is fixed-size unless you pass `dynamic_spatial`. `TiledInference` checks its inputs before starting.
- Edge and cloud: Jetson, SageMaker (packaging, IAM, argmax output), Azure and GCP fixes.

### Explainability, monitoring, benchmark

- **Explainability.** Grad-CAM, uncertainty, attention and SHAP corrected; `GeospatialSHAP.band_importance` added; `UncertaintyEstimator` memory use reduced.
- **Monitoring.** Drift statistics expose `bin_edges`. `DriftDetector.check` reports incomplete data and errors and takes `recent_n`. The `monitor drift` CLI exits 1 on failure.
- **Benchmark.** `SegmentationMetrics.mean_iou` and `mean_f1` added. `BenchmarkResult.error` records a failed evaluation and the leaderboard refuses failed results. The evaluator accepts dict, tuple and logit outputs.
- `BenchmarkConfig` validates its inputs.

### Datasets and models

- **Behaviour change.** The dataset catalogue was de-duplicated (503 to 460 entries); unknown licences read `unknown`; `top_for_task` takes `modality=`.
  `DatasetLoader.download` states that it downloads nothing and where to get the data.
- Model registry honesty: failed pretrained loads raise `ModelLoadError`; no model is substituted for another.

### Advanced tools

- **AutoML.** Failed trials are recorded and the study continues; the search is quiet; plotting rewritten. If no trial completes, `success` is `False` and the first error is reported.
- **Multi-task.** `MultiTaskLearner` rewritten around per-task losses and a shared encoder.
- **Few-shot.** `FewShotLearner` rewritten: prototype classifier on frozen features, a euclidean temperature set automatically, a `value_range` option, and a class map stored as JSON in the output tag.
- **Vision-language.**
  - One shared image loader for CLIP, Moondream and retrieval; `bands=` and `--bands 3,2,1` choose bands.
  - Retrieval index format `{"model", "embeddings"}` with mismatch errors, and a `skipped` list.
  - A query image is excluded by absolute path.
  - The image is validated **before** a model is loaded.
  - `batch_caption` is recursive and takes all image types.
- **Time series.**
  - Rewritten: aligned anomaly dates, `mask_paths` replacing the no-op `cloud_mask`, correct EVI (with blue), nodata handling, an `errors` dict.
  - **Behaviour change.** `ValueError` instead of error dicts for bad input.
  - `period` for seasonal handling, p-values, `extract_point`, `plot`.
- **Point clouds.**
  - Rewritten: real exceptions, exact grid, below-sea-level heights, nodata cells, noise classes 7 and 18 excluded.
  - **Behaviour change.** A scene with no ground raises when `filter_ground=True`.
  - Block-wise classification with `seed`, checkpoint checks and `class_codes`.
  - Building extraction closes gaps with padding, refuses geographic CRS, writes `crs` only with an EPSG code, and simplifies outlines.

### Preprocessing and indices

- `preprocess clip` explains an empty or inverted box instead of failing obscurely.
- The temporary band profile in spectral indices is fixed.
- Counts in docstrings (indices, public methods) corrected; start-up logs moved to debug level.

### Documentation

- A complete Read-the-Docs site: 40 pipeline pages with captured outputs, 25 generated CLI group pages with notes, model and dataset catalogues,
  tutorials for labeling, training, inference, OBIA, explainability, monitoring, benchmark, edge and cloud, reporting, and the advanced tools.
- Generated pages (`docs/reference/cli/`, `docs/models/catalog.md`, `docs/pipelines/catalog/`) are produced by the scripts in `tools/` and checked
  by tests, so they cannot drift from the code.

## Earlier releases

See the [roadmap history](roadmap.md#history).
