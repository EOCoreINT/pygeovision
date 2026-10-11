# Roadmap

This page follows the same honesty standard as
[pygeofetch's own roadmap](https://pygeofetch.readthedocs.io/en/latest/reference/roadmap.html):
a specific accounting of what was found and fixed, and what remains open. Nothing here is softened for presentation.

## Where things stand (this release, 2.1.8)

| | |
|---|---|
| Pipelines | 40: 33 run out of the box, 7 need a checkpoint you trained. 21 stay [removed](../pipelines/removed.md) |
| Providers | 22 (through pygeofetch) |
| CLI | 25 groups, 92 commands, each documented with a captured `--help` |
| Models | 79 buildable architectures, 14 names for your checkpoints, 96 catalogue entries ([why three](../models/registry.md)) |
| Tests | Synthetic data throughout. No test ran against live imagery, a GPU, or a downloaded pretrained checkpoint |

### Open items (checked against the code for this release)

**Things that do not exist yet**

- **No trained task models.** By design; see [what PyGeoVision will and will not do](../index.md). Seven pipelines need your checkpoint.
- **No automatic dataset download.** The 460-entry [catalogue](../datasets/index.md) is metadata. `DatasetLoader.download` explains where to get a dataset; it never downloads one.
- **`lisat-7b`** is listed but cannot be built: its model card's usage example does not match its documented architecture, so it stays unimplemented rather than guessed.
- **SkySense** (weights hosted outside the Hugging Face standard) and **RS-YOLO** (a category of papers, several with unavailable weights) are not in the registry.
- **No air-quality or dust pipeline**: both would need aerosol data (Sentinel-5P / MODIS AOD), and an uncalibrated proxy could be read as health guidance.

**Things that exist but are not verified here**

- Outputs of real DINOv2, CLIP/RemoteCLIP, Moondream, DOFA, Prithvi and SAM weights. The documentation environment had no network and no GPU. The surrounding logic is tested with tiny random local models.
- Ray Tune, MLflow / W&B / TensorBoard logging, mixed precision, DDP, TensorRT and the live AWS / Azure / GCP paths.
- LiDAR **classification accuracy**. Only the plumbing is tested; there is no pretrained point-cloud network.
- Accuracy of any pipeline on real imagery. Methods are documented and checked on synthetic scenes with known answers.

**Things that look like they do something and do not**

- The `gpu` and `training` sections of the [configuration file](config.md) are stored but not read by training or inference.
- Of the 36 classes in `pygeovision.core.exceptions`, about nine are raised; the others are reserved names ([exceptions](exceptions.md)).
- `pygeovision/training/callbacks.py` is imported by nothing in the package or its tests (the trainer uses `pygeovision/ai/training/callbacks.py`).

**Housekeeping**

- The `PyGeoVision` client in `pygeovision/__init__.py` is large and should move to its own module.
- Seven pairs of classes [exist twice](../architecture.md#systems-that-exist-twice); the trainer pair was merged by re-export, the others remain.

## History

The sections below are the dated record of what each audit found. Counts in them (for example "49 pipelines became 25")
describe the project *at that time*; the table above is the present state.

## Pre-launch pipeline audit: 49 pipelines became 25

Every pipeline was run on synthetic data with a known answer. Of the 49 names:

- **24 were removed** ([reasons](../pipelines/removed.md)): untrained networks that
  returned random maps with `success=True` (two identical runs disagreed on ~38% of
  pixels), placeholder indices (a water mask called irrigation), invented
  constants (carbon = 50 x NDVI^2 x 0.47), a land-cover "change" pipeline that
  compared the same static WorldCover map with itself, and COCO detectors run on
  10 m pixels.
- **19 now run out of the box** with documented formulas (NDVI/EVI/NDCI, MNDWI,
  dNBR, NDSI, Landsat surface temperature, Horn slope/aspect, Local Relief Model,
  change vector analysis). Cloud- and shadow-masked pixels used to be counted as
  "not burned", "not water", "no change" and divided into the percentages; they are
  now nodata and excluded.
- **6 need your trained weights** and refuse to run, before downloading imagery,
  without `checkpoint=`.
- `client.pipeline()` could not reach any of the 39 domain pipelines ("Unknown
  pipeline"); one registry now serves Python, the CLI and `list_all_pipelines()`.
- The WorldCover 2020 URL pointed at the 2021 edition's path. 2020 is v100 and 2021
  is v200, and ESA says the two are not comparable for change detection.
- `dem_type` was accepted but ignored; the DEM provider now selects the product
  that was asked for.

**Not done:** `client.change.detect` still has a fallback path and an untrained
`changeformer`; `client.detection.ships/cars` use a COCO detector; `ai train` prints
Python instead of training. All pipeline tests use synthetic rasters; none ran
against live imagery.

## A real, significant gap found and fixed: detection training had no real accuracy metric

Checked directly: `DetectionMetrics` (real, `torchmetrics`-backed mAP,
independently hand-verified) had zero references anywhere in
`GeoTrainer.fit()` — real detection validation only ever computed
`val_loss`, with no real accuracy signal available for early stopping
or checkpoint selection. Fixed by adding a second, `eval()`-mode
forward pass during detection validation (a real `torchvision`
detection model returns its internal loss dict *or* real predictions
from one forward call, never both) to feed `DetectionMetrics` real
predictions, matching its existing, real `predictions`/`targets` dict
format exactly. `early_stopping_metric`/`checkpoint_metric` now
default to the newly-available real `val_map50`/`max` for detection
(previously auto-switched to `val_loss`/`min`, since no real accuracy
metric existed yet); an explicit override is still always respected.
Verified end-to-end with a real `torchvision` RetinaNet: confirmed
`best_metric_name == "val_map50"`, a real `best_val_map50` key
present, and a real, populated `history["val_map50"]` curve. Full,
836-test real suite re-run clean afterward. See
[Object Detection Training](../training/index.md) for the
complete detail.

## A real, significant training bug found and fixed: augmentation was completely dead

`TrainingConfig.augment`/`flip_prob`/`rotate_limit`/`scale_limit`/
`brightness_limit`/`contrast_limit` (`pygeovision.ai.training.trainer`)
had zero references anywhere in the real training loop — `augment=True`
(the real default) had no effect on training at all. A real,
substantial, geospatially-aware augmentation module
(`pygeovision.ai.data.augmentations.GeoAugmentationPipeline` — 90°-only
rotation to preserve georeferencing, real atmospheric haze/sensor
noise/band-dropout simulation) already existed elsewhere in this
codebase; it was simply never wired into `GeoTrainer.fit()`. Given how
much this matters specifically for fine-tuning on small, real-world
labeled datasets — the primary real use of this training module — this
was fixed rather than just documented: a new `_AugmentedSegmentationDataset`
wrapper applies the real pipeline to training data (never validation)
when `augment=True` and `task="segmentation"`; `task="detection"` logs
a clear, honest warning and proceeds unaugmented, since detection
samples have a genuinely different real format this pipeline isn't
built for. Verified end-to-end: a real segmentation training run
completed successfully through the wrapped pipeline with correct
output shapes/dtypes at every step (including catching and fixing a
real dtype bug in the fix itself — the augmented mask needed an
explicit cast back to `torch.long` for cross-entropy loss), and a real
detection training run correctly triggered the warning and completed
normally. This changes real, existing behavior for any caller
currently relying on the previous (broken) no-op default.

Separately confirmed real and correct via direct code inspection (not
previously documented on this site): the real distributed-training
path (`strategy`/`devices`/`sync_batchnorm`/`find_unused_parameters` —
genuine `torch.distributed`/`DistributedDataParallel`/
`FullyShardedDataParallel` wiring) and real experiment tracking
(`tracker`/`mlflow_uri`/`wandb_project` — genuine `mlflow`/`wandb`
integration). Also noticed and worth recording, though already fixed
in an earlier pass (found via its own real, existing code comment, not
newly fixed here): the training result dict's `best_val_iou` key was
previously always present and named that way even for
`task="detection"`, where the real tracked metric is `val_loss`, not
IoU — confirmed misleading by a real test run reporting
`"best_val_iou": 1.77` (IoU is bounded 0–1). Already corrected to only
include `best_val_iou` when the tracked metric genuinely is
`val_iou`.

## The natural-language agent has been removed

`pygeovision.agent` (`GeoAgent`, the LLM/heuristic planner, executor,
and 8-tool registry documented in earlier revisions of this site) was
a real, working, tested module — not dead code, not a stub. It was
removed as a deliberate scope decision, not an audit finding. Every
capability the agent orchestrated is still directly available through
the pieces it called: [AI Task Pipelines](../pipelines/index.md),
[the Python API](python-api.md), and [the CLI](cli/index.md) — the agent was
a convenience layer routing natural-language requests to these, not a
capability that existed only inside it. `pip install anthropic`/
`pip install groq` are no longer needed at all, since those were
exclusively the agent's own lazy, optional dependencies.

If you were calling `GeoAgent` directly, the equivalent real, direct
call for any of its 8 real tools is documented on the corresponding
page above — e.g. what the agent's `run_pipeline` tool did is exactly
`client.pipeline(name, ...)`.

## Fixed while writing deeper pipeline documentation

- **`road_extraction` had a real, severe bug that meant it could never
  detect an actual road.** It used generic, COCO-pretrained object
  detection with `class_names=["road"]` — but roads aren't a real COCO
  category, and the underlying detector only relabels a detection when
  no real class filter is given and the class index happens to match
  position 0 in the name list (COCO class 0 is "person"). Confirmed by
  direct calculation: every real "person" detected in the image was
  mislabeled "road," while every other real detection (car, truck,
  etc.) kept its correct, unrelated COCO label. Fixed to use real
  segmentation instead — the standard framing for road-network
  extraction in the remote-sensing literature, matching the pattern
  already used by `landslide_detection`/`crop_type_mapping`.
- **`parking_occupancy`'s documented `total_spaces=`
  parameter/`occupancy_pct` output don't exist.** The real
  implementation provides vehicle count and density only, stated
  explicitly in its own docstring — "occupancy" would need real
  parking-capacity data this pipeline doesn't have.
- **`pipeline_leak_detection`'s documented `corridor_path=` parameter
  doesn't exist.** The real pipeline analyzes the entire requested
  bbox, not a route-constrained corridor.
- **`port_monitoring`'s and `infrastructure_monitoring`'s real stats/
  metadata keys were wrong** in an earlier version of this page
  (`n_vessels`/`vessel_density_per_km2` vs. the real
  `n_ships_detected`; a claimed "configurable" spectral-diff threshold
  that's actually a fixed 90th-percentile constant).



- **`flood_mapping` was completely fabricated as using a real Prithvi
  foundation model.** Checked directly: it's a single, direct NDWI
  call (`client.segmentation.water()`), nothing more — no foundation
  model, no HLS channel mapping, no "flood_detection task mode" as an
  earlier version of this page claimed.
- **`water_quality`'s real NDCI (chlorophyll proxy) computation was
  omitted entirely** from an earlier version of this page, which
  described only the turbidity proxy as if it were the whole pipeline.
- **`coastal_monitoring`'s documented NDWI shoreline-extraction step
  doesn't exist.** The real pipeline calls `client.change.detect()`
  directly on the raw image pair — a real, third distinct
  change-detection code path (separate from both the `domains.py`
  bi-temporal pipelines and the standalone `ChangeFormer` class) that
  tries `ChangeDetection(model_variant="changeformer")` first and
  genuinely, silently falls back to a fixed-percentile spectral-diff
  method if that fails. *(Fixed: `client.change.detect` now defaults to Otsu-thresholded change vector
  analysis and the network methods require a checkpoint and never fall back.)*
- **`oil_spill_detection` overclaimed an "adaptive local threshold."**
  The real implementation uses a fixed, global `-18.0` dB constant.
- **`reef_bleaching` overstated its own real bi-temporal design**: an
  earlier version of this page implied both depth and turbidity are
  cancelled out by comparing two dates. Checked directly against the
  source: only the *static* seafloor depth is cancelled — turbidity
  and tidal state can still genuinely differ between dates and produce
  a false positive, a real limitation the source states explicitly
  but this page previously didn't reflect.



- **`urban_heat_island`'s documented NDVI-based emissivity correction
  doesn't exist.** Checked directly: the real implementation is a
  direct DN→Kelvin→Celsius conversion (real USGS Collection 2
  calibration constants), nothing more. Also discovered and documented
  for the first time: `land_surface_temperature`, an identical-
  implementation sibling (a real Python subclass with no overridden
  logic) that had never been documented anywhere before.
- **`solar_potential` overclaimed a "direct and diffuse components"
  irradiance model.** The real docstring explicitly states there is no
  atmospheric attenuation modeling beyond the geometric
  cosine-incidence factor. Also had the same fabricated `dem_path=`
  parameter as `archaeological_site`/`wind_farm_siting` (both fetch a
  real DEM automatically via OpenTopography; neither accepts a
  user-supplied file), and both were missing the real OpenTopography
  API key requirement.
- **`landcover_change`'s real stats key is `top_transitions`** (the
  top 10 class transitions by frequency), not a full
  `transition_matrix` as previously documented.
- **Three more real, notable cross-file `LandCoverPipeline`
  compositions found and documented**: `landcover_change`,
  `mine_detection`, and `construction_progress` (bringing the total to
  five real pipelines in `domains.py` that reuse this one class from
  the separate `pygeovision.ai.pipelines` module, rather than
  duplicating ESA WorldCover access logic).



- **`wetland_mapping` was described with the wrong formulas** — an
  earlier version claimed NDWI+NDVI; the real implementation uses
  MNDWI (Xu, 2006, using SWIR1) and EVI (Huete et al., 2002),
  scientifically different indices, though the overall three-way
  classification logic was accurately described.
- **`mangrove_mapping`'s real bi-temporal requirement was undocumented**
  — described as a single-date `date=` extent map; the real pipeline
  requires `date_before`/`date_after` and computes area *change*
  against ESA WorldCover's specific mangrove class code (95), not a
  filtered subset of "tree cover" classes as previously described.
- **`archaeological_site`'s documented `dem_path=`/`lrm_radius_m=`
  parameters don't exist.** The real pipeline auto-fetches a DEM from
  OpenTopography (requiring a real API key) rather than accepting a
  user-supplied file, and the real smoothing parameter is
  `smoothing_radius_px` (pixels, not meters).
- **Two real, notable cross-file compositions found and documented for
  the first time**: `mangrove_mapping` and `biodiversity_hotspot`
  (both in `domains.py`) internally construct and call the real
  `LandCoverPipeline` from the separate `pygeovision.ai.pipelines`
  module — genuine code reuse across the two-implementation split
  described in [Architecture](../architecture.md), not duplicated
  logic.



- **All 10 CLI-reachable pipeline names had full, incorrect duplicate
  descriptions scattered across 6 other domain pages**
  (`water-and-disasters.md`, `urban.md`, `infrastructure.md`,
  `cryosphere-and-climate.md`, `forestry.md`, `change-detection.md`)
  — each describing a plausible-sounding but different implementation
  than the real one actually dispatched for that name (different
  parameters, different formulas, different `result.stats`/
  `result.metadata` shapes). Every one checked directly against the
  real source; all removed and replaced with a cross-reference to
  the pipeline pages (since replaced; see
  [AI Task Pipelines](../pipelines/index.md)).
- **`canopy_height`'s documented `dem_path=` terrain-normalization
  parameter doesn't exist.** The real pipeline takes only `bbox`,
  `output_dir`, and `date`, with no DEM integration at all.
- **`vegetation_indices` was filed under the wrong domain** (its own
  class says `domain="agriculture"`, not `environment`) and described
  inaccurately in three ways: claimed a selectable `indices=`
  parameter (doesn't exist — NDVI/EVI/NDWI are always all three
  computed), claimed SAVI was one of them (it isn't), and claimed
  multi-band GeoTIFF output (the real output is per-date statistics).



- **`crop_type_mapping`'s default model name was fake.** Confirmed
  directly: `"crop_type_model"` is not a real, registered entry in
  either registry — calling this pipeline with its own defaults always
  raised `RuntimeError`. It had never worked out of the box. Fixed the
  default to `"segformer_b2"`, a real, working segmentation model.
- **`landslide_detection`'s hardcoded model name was also fake**, and
  worse than the bug above: there was no `kwargs.get()` override at
  all, so this pipeline could never be run successfully under any
  configuration. Fixed to a real, working default, made properly
  overridable.
- **`irrigation_detection` never requested NIR data.** It downloaded
  only the default red/green/blue bands, but its own real NDWI
  computation (`segmentation.water()`) needs green+NIR to be
  meaningful. With only 3 bands, the fallback band-indexing used red
  as "green" and green as "nir" — silently computing
  `(red−green)/(red+green)` instead of real NDWI, a scientifically
  meaningless result for every real run. Confirmed by direct
  calculation before fixing.



- **`pygeovision channel`'s `--date-before`/`--date-after` flags had
  zero effect for 3 of the 10 real, CLI-reachable pipelines**
  (`disaster_assessment`, `deforestation`, `urban_growth`) — found
  while writing full documentation for these pipelines and checking
  the actual CLI-to-pipeline argument flow rather than assuming it
  worked because the flags existed. The CLI passed these through with
  literal `date_before`/`date_after` kwarg names regardless of which
  pipeline was selected, but these three pipelines' real `run()`
  signatures use different parameter names (`pre_date`/`post_date`,
  `baseline_year`/`analysis_year`, `start_year`/`end_year`) — confirmed
  directly by inspecting each signature. The flags were silently
  absorbed into `**kwargs` and never used, with each pipeline falling
  back to its own hardcoded default dates regardless of what a real
  user requested via the CLI. Fixed to translate to the correct real
  parameter name per pipeline, with year-extraction for the two
  pipelines that expect a bare year rather than a full date.

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

**A second, CLI-connected trainer had the identical unfixed bugs**
- `freeze_backbone` and the default cosine LR scheduler were fixed
  earlier in `pygeovision.ai.training.trainer` (Python-API-only), but
  writing deeper documentation for this system prompted checking the
  *other*, genuinely separate `pygeovision.training.trainer` -- the
  one `pygeovision ai train` (the actual CLI command) uses -- rather
  than assuming the earlier fix covered both. It didn't: both bugs
  were present here too, completely unfixed. `freeze_backbone`
  appeared exactly once in the file (its own definition) with zero
  effect; the default scheduler had the identical `T_max=max_epochs`
  vs. per-batch-stepping unit mismatch, verified by simulation to
  produce dozens of oscillations instead of one smooth decay, and the
  `step` scheduler had the same unit mismatch too. Every real user of
  the CLI's `ai train` command up to this point in the audit was
  silently affected by both. Fixed the same way as the other trainer
  and independently re-verified (60/92 real parameters freeze; zero
  oscillations in a simulated full training run).
- A related documentation conflation, also corrected: earlier pages on
  this site used `checkpoint_path=...`/`task="detection"` in code
  examples that imported `pygeovision.training.trainer` -- neither
  field exists on that class at all; both are real, but only on
  `pygeovision.ai.training.trainer`. Fixed the imports and added an
  explicit comparison table clarifying which of the two real trainers
  has which real capability.

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

- `client.segmentation.custom()` and `client.detection.custom()` (and
  the CLI commands built on them, `ai segment custom`/`ai detect
  custom`) passed a real model-name string straight to the tiled
  inference engine, which requires an actual, built model object --
  confirmed by direct testing, this failed with a confusing "band
  count"/"str object is not callable" error rather than working. Now
  resolves a real string name via `ModelHub.load()` first.

**Documentation corrections found while writing deeper detail**
- Several earlier pages on this site used `pygeovision run <pipeline>
  --bbox lon,lat,lon,lat` as the CLI example for running a named AI
  pipeline. Checked directly against the real CLI source: no `run`
  command exists at all. The real command is `channel`, its `--bbox`
  takes 4 separate space-separated floats (not a comma-separated
  string), and critically, `pipeline_name` is a real, closed
  `click.Choice` of only 10 of the 49 real pipelines -- the other 39
  have no CLI entry point at all, Python-API-only. Corrected across
  every page that repeated the error.
- A related fabricated claim on the same pages: that the CLI writes a
  `stats.json` file. Checked the real command's output handling
  directly -- it only prints `result.stats` to the terminal via
  `click.echo()`. Corrected.

**Newly-discovered, previously-undocumented functionality**
- `pygeovision.ai.labeling.label_studio.LabelStudioLabeler` (679 real
  lines, a genuine human-in-the-loop Label Studio integration) had
  never been documented or audited before this pass. Checked it for
  the same HTTP-status-not-checked pattern found in the WorldCover/
  Dynamic World fallback downloads above, since it makes just as many
  real HTTP calls -- it doesn't have that bug. Its shared
  `_api_get()`/`_api_post()` helpers already correctly call
  `raise_for_status()` on every request.

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

## Found during the first documentation pass

Items listed here at the time were reviewed for this release; those still open are in the table at the top. Resolved since:
`tree_species` and `air_quality_index` became [removed pipelines](../pipelines/removed.md) with stated reasons; the dataset
catalogue was de-duplicated (503 to 460 entries) and its `download` now says plainly that it downloads nothing; the unused
optimiser module is gone and `training/optimizer.py` (`build_optimizer`, `build_scheduler`) is what the trainer calls.

## Added since this documentation audit

- Real object-detection training support (`task="detection"` in
  `TrainingConfig`), verified end-to-end with a real `torchvision`
  `RetinaNet`.
- A real wrapper (`SamGeoLabeler`) around
  [segment-geospatial](https://samgeo.gishub.org), a real, peer-reviewed
  SAM integration for remote sensing.
- **A real, dedicated DOFA integration** (`pygeovision.models.foundation.dofa`),
  replacing the previously-unverified generic `AutoModel` dispatch.
  Built on [torchgeo](https://github.com/microsoft/torchgeo)'s official
  implementation and real, direct checkpoint URLs. Verified end-to-end:
  the same model instance correctly embeds both 9-band Sentinel-2 and
  3-band NAIP input. See [Embeddings](../models/foundation.md).
- **`GeoEmbeddings`** (`pygeovision.models.embeddings`) — a new, unified
  embedding-extraction interface across DOFA, Prithvi, DINOv2,
  RemoteCLIP, and Tessera, plus a real, verified `cosine_similarity`/
  `nearest` pair. Found and fixed a real gap while building it:
  `TesseraGeo.embeddings_for_bbox()` never actually returned the real
  embedding array, only its shape/metadata.

**A full test-suite audit round** (the project's real `tests/`
directory, not written during this documentation work) surfaced
several more real findings, resolved down to a clean 936 passed / 0
failed:

- `pygeovision ai train`'s CLI command crashed with `TypeError` on
  every single invocation (both segmentation and detection) — it
  imported the wrong one of the two parallel `TrainingConfig` classes.
- `MultiTaskLearner.pretrained` was a real, documented constructor
  parameter with zero effect — `pretrained=True` was hardcoded in the
  actual model-building code regardless of what was requested. Same
  pattern as the `freeze_backbone` bug above, found independently.
- The heuristic agent planner's subsidence routing was reconsidered:
  an earlier fix made it raise a clear error; real, independent
  evidence from the test suite showed the intended design was a real,
  working plan using `change_detection` with an honest caveat instead.
  Also found and fixed a second dangling pipeline reference
  (`"forest_monitoring"` → real name `"deforestation"`) while
  investigating.
- Nine registry entries initially removed as fake were reconsidered and
  restored as honest, discoverable stubs instead of silent absence —
  see [Model Registry](../models/registry.md).
- Five orphaned test files (`test_slc_insar.py`, `test_viz.py`,
  `test_enterprise.py`, `test_serving.py`, `test_utils_phase8.py`)
  tested modules confirmed dead and already removed from the codebase;
  deleted rather than fixed, since the functionality they tested was
  deliberately out of scope, not buggy.


## Deep-learning layer audit

Found by reading the code and running it on synthetic data; every fix has a test (several were mutation-checked).

- **Fabricated or mislabeled models removed.** DINOv3 names loaded DINOv2 weights; `CHMv2Model` put a randomly
  initialised decoder on the backbone and scaled it to "metres"; `DINOv3Text` compared DINOv3 features with CLIP
  text embeddings (different spaces); Prithvi returned placeholder embeddings when its loader failed. DINOv3 and CHMv2
  now load the real `facebook/dinov3-*` repositories (gated), and the rest raise or are gone.
- **Registration.** The foundation labelers resized + centre-cropped the scene and stretched the label grid over the
  whole image, and the SAM auto-labeler tiled without a grid. Labels now land on the pixels they describe; nodata
  stays nodata.
- **Labeler contract.** Four labelers built `LabelingResult` with fields it does not have, five called a GeoTIFF
  writer with the wrong arguments, and `label_tiles` could not drive any path-oriented labeler. One contract now.
- **Data-source bugs.** Google Open Buildings geometry is WKT but was parsed as JSON (every row silently skipped).
- **Wrong hub ids and revisions** in the model zoo, CLIP/RemoteCLIP, DOFA and SAM2 corrected; Moondream pinned.
- **Packaging.** The `pygeovision[ai]` extra that error messages recommend did not exist; `foundation` lacked
  `einops`; four dead re-export stubs and two stale copies of source modules in `tests/` were deleted.

Still open: no pretrained checkpoint, GPU or real scene was used in any test; the Prithvi remote `prithvi_mae.py`
and checkpoint filenames were checked against the model cards, not executed; the CHMv2 output units and intended
imagery are undocumented upstream; the `PyGeoVision` client in `pygeovision/__init__.py` (about 59 KB) should be
moved to its own module.
