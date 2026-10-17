# Changelog

What changed in behaviour, grouped by area. This release (2.1.8) is the result of documenting every feature by running it: each
item below was found because an example did not do what its page said it would, and each has a test. Entries that **change what you
get** (a different default, an error where there used to be silence) are marked **behaviour change**. For the
older, longer narrative of audits see the [audit history](audit-history.md).

## 2.1.8

### Documentation

- **The documentation was rebuilt on the same system as the pygeofetch documentation.** A sticky top navigation bar with search (`Ctrl K`),
  a collapsible sidebar, breadcrumbs, a version badge, a right-hand table of contents, light and dark themes, the Ubuntu typeface, a
  reading-progress bar, a back-to-top button, fade-in motion that respects `prefers-reduced-motion`, click-to-enlarge figures and a print
  stylesheet. Two animated diagrams (the pipeline run and the OBIA workflow) were added as figures.
- **Seven languages.** English, French, Spanish, Portuguese, German, Simplified Chinese and Arabic (right-to-left), with a language switcher.
  The navigation bar and the landing page, and the Quickstart, are translated; every other page falls back to English and says "partly
  translated". See `docs/locale/README.md` to add pages or languages, and `docs/tools/check_translations.py` to check them.
- **New pages:** [Diagnostics](../getting-started/diagnostics.md), [Known issues](../reference/known-issues.md),
  Testing and [Security](../reference/security.md).
- **Moved:** the changelog, the roadmap and the contributing guide now live under `project/`. The old addresses redirect.
- **Build:** `make -C docs html` builds every language and fails on a warning; `make -C docs html-en` builds English only.
  `docs/requirements.txt` gained `furo`, `sphinx-reredirects` and `polib`.

### OBIA on real imagery

- **Behaviour change: cloud and no-data.** `obia.segment`, `segment_levels` and the scale tools used to fail or return meaningless objects on an image with NaN pixels. They now give **label 0** to every pixel that is NaN in any band (or equal to `nodata` in all bands), build no object there, and keep ids `1…n`, each one connected. New `obia.valid_mask`; `SegmentationResult.valid`. Rescaling, GLCM texture, `class_context` and `segmentation_quality` ignore those pixels. Region merging on images with many objects is much faster (bounding-box relabelling).
- **Behaviour change.** `samples_from_geometries` never returns object 0, raises `ValueError` when the geometries lie wholly outside the image (a CRS mix-up), and takes `crs=` to reproject them; `assess_classification` passes it through.
- **New:** `obia.samples_from_raster` (weak labels from a land-cover map, with a purity threshold), `obia.fetch_scene`, `obia.fetch_worldcover` and `obia.fetch_osm_polygons` (Sentinel-2, ESA WorldCover and OpenStreetMap polygons onto the OBIA layout, through the same search, download and Overpass code as the rest of PyGeoVision), the client proxy `client.obia`, the command `pygeovision obia fetch`, and `obia classify --weak-labels --class-map`. The download paths are tested with stand-in clients; they were not run against the live services. See [Real imagery](../obia/real-data.md).
- **Fix: silent, apparently frozen downloads.** `obia.fetch_scene`, `fetch_worldcover`, `fetch_osm_polygons` and `obia fetch` now report progress (stage, the library's search/download messages, and a heartbeat with elapsed time and megabytes received that says when nothing new arrives); `verbose=False` / `--quiet` silences it. A failure names the reasons the download code logged. The pygeofetch command-line search now has a time limit, the Overpass default waits shrank from 120 s x 3 rounds to 60 s x 2, WorldCover remote reads time out after 60 s, and `_search_and_download` logs when a search finds nothing or a scene download fails. When the cloud limit is what emptied the search (providers return their scenes and the limit then drops the cloudy ones, so a rainy-season month shows "54 scenes" and "0 results"), it searches once more without the limit and reports the clearest cloud cover that exists and the value to raise `cloud_cover_max` above.
- **Removed: all synthetic scene generators.** `obia.make_demo_scene`, `obia.make_mine_scene`, the `DemoScene`/`MineScene` types, the `pygeovision obia demo` command and the demo-scene documentation page are gone. OBIA now works on real imagery only, and a failed download raises `RuntimeError` instead of substituting data. Documentation numbers that came from those scenes (scores, timings, confusion matrices, figures) were removed rather than replaced; the OBIA, benchmark, explainability and drift pages describe how to measure on your own data. The unit tests use tiny hand-built arrays (`tests/obia_testdata.py`), which are test fixtures and not part of the package.
- **Fix:** `python -m pygeovision.cli.main` stopped before the later command groups (`obia`, ...) were registered, so `python -m pygeovision.cli.main obia ...` said "No such command". The `pygeovision` script was not affected.

### Pipelines, data and labeling

- **New pipeline: `mine_rehabilitation`** (41 pipelines, 34 ready). Compares a mine, quarry or tailings site with an undisturbed reference on each of several same-season dates (relative recovery index), and classes every pixel as recovered, recovering, stalled, bare, declining or water. Outputs a status raster, index and trend rasters, the reference mask and a per-date CSV. Cloud is nodata; dates seen on too little of the site are listed and left out; it refuses to run without a `site` or `reference`, with fewer than three dates, or with too small a reference. It does not predict when a site will meet its criteria. See [Mining](../examples/mining.md).
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

See the [audit history](audit-history.md).
