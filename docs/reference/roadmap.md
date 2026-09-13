# Roadmap

This page follows the same honesty standard as
[pygeofetch's own roadmap](https://pygeofetch.readthedocs.io/en/latest/reference/roadmap.html):
a real, specific accounting of what was found and fixed during this
package's audit, and what remains genuinely open. Nothing here is
softened for presentation.

## Fixed while writing deeper pipeline documentation

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
  method if that fails.
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
  [The 10 CLI-Reachable Pipelines](../pipelines/cli-reachable-pipelines.md),
  the one page with verified, accurate detail for these 10 names.
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
- **A real, dedicated DOFA integration** (`pygeovision.models.foundation.dofa`),
  replacing the previously-unverified generic `AutoModel` dispatch.
  Built on [torchgeo](https://github.com/microsoft/torchgeo)'s official
  implementation and real, direct checkpoint URLs. Verified end-to-end:
  the same model instance correctly embeds both 9-band Sentinel-2 and
  3-band NAIP input. See [Embeddings](../models/embeddings.md).
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
  see [Model Registry](../core-features/model-registry.md).
- Five orphaned test files (`test_slc_insar.py`, `test_viz.py`,
  `test_enterprise.py`, `test_serving.py`, `test_utils_phase8.py`)
  tested modules confirmed dead and already removed from the codebase;
  deleted rather than fixed, since the functionality they tested was
  deliberately out of scope, not buggy.
