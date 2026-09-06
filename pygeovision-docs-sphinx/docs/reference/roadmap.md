# Roadmap

This page follows the same honesty standard as
[pygeofetch's own roadmap](https://pygeofetch.readthedocs.io/en/latest/reference/roadmap.html):
a real, specific accounting of what was found and fixed during this
package's audit, and what remains genuinely open. Nothing here is
softened for presentation.

## Fixed during this documentation audit

**Pipelines & data preparation**
- Band selection silently downloaded every band regardless of what was
  requested (`client.download()`'s real `bands=` parameter was never
  used).
- An AOI straddling two scene footprints silently returned a
  smaller-than-requested result with no warning; real mosaicking with
  coverage checking now handles this.
- Three layers of defense added against a band-count mismatch reaching
  a model — found via a real production failure ("expected 3 bands,
  got 13").
- The same band-count bug, found separately and fixed separately, in
  a genuinely different, parallel `TiledInference` engine
  (`pygeovision.inference.tiled`) that shares no code with the first.
- A silent filename collision in bi-temporal alignment meant every
  "before/after" pipeline was comparing an image to itself.
- 10 pipelines were bypassing all real preprocessing (band stacking,
  radiometric scaling, cloud masking, bbox cropping) via a raw,
  unprocessed download path.

**Model registry**
- 59 of 121 entries in the general-purpose registry had no genuine
  working backing and were removed, including every `dinov3_*` entry
  (labeled DINOv3, actually loaded DINOv2) and three U-Net variants
  that all silently returned the identical generic model regardless of
  the named backbone.
- `dofa-base`'s `hf_id` pointed to a repository that doesn't exist;
  fixed to the real, confirmed repository.
- `changeformer-mit-b0/b4` had zero registry wiring despite a real,
  203-line implementation already existing; wired in.
- A real, task-aware fallback added for genuine model-build failures
  (previously only "name not found" triggered any fallback at all).
- Three real, severe bugs found in model-loading code: `load_prithvi_hf`
  silently returned a randomly-initialized model on failure; a SAR
  feature extractor silently returned all-zero features; a SAR flood
  detector silently returned an all-zero (misleadingly "no flood
  detected"-looking) prediction on invalid input. All three now raise
  clearly by default.
- A compound integration bug found while fixing the SAR feature
  extractor above: the real model constructor, the real method name,
  and the real input format were all wrong simultaneously, meaning
  that specific integration had never worked at all, with or without
  network access.

**Training**
- The default learning-rate scheduler oscillated 50 times instead of
  decaying once across a training run, affecting every user who didn't
  explicitly override it.
- `freeze_backbone` was a real config field with zero effect on
  training.
- No way existed to finetune from an existing checkpoint at all.
- `GeoTrainer` silently ignored the CLI's `task` selection, always
  training as if for segmentation even when detection was requested.
- `pretrained=False` on two detection-model builders did not actually
  prevent a network download.

**Labeling**
- Six labeling functions reported success unconditionally, regardless
  of whether any real content was found.
- A download function never checked the real HTTP status before
  writing content to disk and declaring success.

**Natural-language agent**
- The heuristic planner's solar and road routes referenced
  `"solar_panels"` and `"road_network"` -- neither exists in the real
  pipeline registry (the real names are `solar_detection` and
  `road_extraction`). A generated plan would look correct and fail
  only when actually executed. The same two names, plus two more
  (`"crop_mapping"`, `"forest_monitoring"`), were also stale in
  `EndToEndPipelineTool`'s own parameter metadata -- the schema an
  LLM-based planner reads to decide what `pipeline_name` value to
  pass. Left uncorrected, an LLM planner would have been told these
  fake names were valid options.
- `"subsidence"` was still a real, detected task keyword (a leftover
  from before InSAR removal), but had no real routing branch --
  silently fell through to a generic land-cover plan with no
  indication a real, understood request couldn't be fulfilled. Now
  raises clearly.
- A documentation error on this site's own earlier pages, corrected
  after direct verification: `client.pipeline(...)` was described as
  `pygeofetch`'s data-chain builder based on a stale assumption.
  Reading the real, current source shows it correctly delegates to
  the same `get_pipeline()` + `.run()` machinery as the explicit
  class-based form -- a real, correct shortcut, not a mistake to avoid.
- Session `load()` never restored conversation history at all, despite
  its own log message claiming otherwise -- fixed to reconstruct real
  turns from what was actually saved.
- The `flood_mask` follow-up-query convenience checked the wrong field
  (`tool_name` instead of the real task argument) and had never once
  worked for the real flood pipeline -- verified directly with a
  synthetic trace matching its actual step structure, then fixed.

**Dataset catalog**
- `DatasetLoader.download()` (a real, CLI-reachable command claiming to
  "download and extract a dataset by name") created an empty output
  directory and returned it as if a real download had happened, for
  every one of the small number of entries with a real `download_url`
  set. Checking those URLs directly found why no real fetch was ever
  implemented: they're all human-facing project pages, not direct file
  links — a real HTTP request against them would have downloaded HTML
  mislabeled as the dataset, which would have been a worse bug than
  the one it replaced. Now raises clearly for every real dataset name,
  pointing to the real URL to visit.

**Removed entirely (InSAR)**
- Two pipelines requiring InSAR displacement processing were removed
  from scope, along with two phantom function references in docstrings
  claiming a `prepare_insar_for_ai()` function existed when it did not,
  anywhere in the codebase.

**Dead code removed**
- 4,619 lines across five top-level directories confirmed to have zero
  imports from anywhere in the codebase.
- A superseded data-access design (`PyGeoVisionEngine`) confirmed
  entirely dead, wrapping `pygeofetch`'s CLI while the real, current
  data layer uses `pygeofetch`'s Python API directly.
- 12 files that were pure 4-line facades re-exporting the real registry
  with no unique content of their own.

## Found during this documentation pass, not yet fixed

- `lisat-7b` is a real, published model with real weights, but has no
  verified build path — its own model card's usage example doesn't
  match its real, documented architecture. Left honestly unimplemented
  rather than guessing.
- `air_quality_index` remains unimplemented — the same technique used
  for `dust_storm_tracking` would risk being read as calibrated health
  guidance under this specific name.
- `tree_species` honestly returns `success: False` — no real, verified
  tree-species model is wired in.
- ~700 lines across five modules in the training package
  (`callbacks.py`, `checkpoint.py`, `mixed_precision.py`,
  `distributed.py`, `experiment.py`) are real, substantial code
  confirmed unused by `GeoTrainer`, which reimplements simpler inline
  versions instead.
- The real, sophisticated discriminative fine-tuning optimizer
  (backbone learning-rate multiplier, layer-wise decay) in
  `ai/training/optimizers.py` is unused even within its own package —
  neither real `GeoTrainer` implementation calls it.
- None of the 503 real, published datasets in the dataset catalog can
  currently be fetched automatically through `pygeovision` — the
  `download()` fix above makes this honest rather than misleading, but
  the underlying capability (a real, working download mechanism for at
  least the handful of datasets with directly-fetchable archives) is
  still genuinely missing.
- `SkySense` (a real model) was not added to the registry: its weights
  are GitHub-hosted, non-HuggingFace-standard, and were judged too
  complex to integrate responsibly without real end-to-end testing.
- `RS-YOLO` was not added: it isn't a single model but a category of
  independent research papers, several with confirmed-unavailable
  weights — a generic entry would repeat the exact problem this audit
  spent most of its time fixing.

## Added since this documentation audit

- Real object-detection training support (`task="detection"` in
  `TrainingConfig`), verified end-to-end with a real `torchvision`
  `RetinaNet`.
- A real wrapper (`SamGeoLabeler`) around
  [segment-geospatial](https://samgeo.gishub.org), a real, peer-reviewed
  SAM integration for remote sensing.
