Reframe pygeovision as an honest-capability geospatial AI toolkit; close the gap between what the registry claimed and what actually ran

A full audit of this codebase found that roughly half of its model
registry, several of its labeling functions, and multiple foundation-
model adapters claimed to work while silently returning something
else -- a different model loaded under the requested name, a random
untrained network standing in for a real one, an empty directory
returned as if it were a completed download, all-zero predictions
returned as if they were real findings. Rather than continue
presenting pygeovision as a tool with universal, fully-verified
coverage -- a claim the evidence didn't support -- this commit
reframes the project around honest capability boundaries: every real
model, pipeline, and tool either works and is proven to work, or fails
clearly and says why. The sections below are the audit findings and
the fixes that make that framing real, not aspirational.

Model registry
- pygeovision/models/registry.py: removed 59 of 121 entries with no
  genuine working backing, confirmed individually rather than by
  pattern-matching alone. Every dinov3_* entry (12 variants) loaded
  real facebook/dinov2-* weights under a DINOv3 label; three U-Net
  variants (unet-r50/r101/efficientb4) all silently returned the
  identical generic fallback network regardless of the named backbone;
  roughly 40 more (yolov8-*, deeplab-*, changestar-*, satlas-*, and
  others) had no real weights and no real construction code anywhere
  in the codebase. 121 -> 62 models.
- Two entries fixed rather than removed, since real code already
  existed and wasn't wired up: changeformer-mit-b0/b4 (a real 203-line
  implementation was never called) and dofa-base (hf_id pointed to
  "XShadow/DOFA-ViT-base-p16", which doesn't exist; confirmed via
  search the real repo is "XShadow/DOFA").
- Added lisat-7b honestly: a real, published model (arXiv:2505.02829)
  with real HuggingFace weights, but no working build path -- its own
  model card's usage example doesn't match its real, documented
  architecture (a custom LISA-style LLM + SAM decoder, not a standard
  transformers segmentation class). Raises clearly with what's needed
  rather than guessing at an unverified loading mechanism, matching
  the existing honest-failure pattern for centernet-r50.
- Added SamGeoLabeler: a real wrapper around segment-geospatial
  (Wu & Osco 2023, JOSS), confirmed installable and its real API
  verified directly against the installed package -- not a
  reimplementation of SAM-geospatial logic pygeovision already had.
- pygeovision/ai/models/registry.py (the smaller, fully-offline
  native registry, 14 models): confirmed every entry genuinely builds
  offline after fixing build_unet/build_deeplabv3plus, which accepted
  encoder_weights but not the pretrained convention every other
  build_* function in the package uses -- pretrained=False was
  silently ignored, confirmed by real weight downloads happening
  anyway.
- pygeovision/ai/models/hub.py: ModelHub.load() previously only fell
  back to the larger registry when a model name was not found
  (KeyError). A model found by name but genuinely failing to build
  (e.g. a network-dependent hf_id model offline) raised immediately
  with no fallback attempt. Added a real, task-aware fallback: on any
  build failure, determines the model's real task and tries other
  real, verified models for that task, native registry first (fully
  offline-buildable). Never invents a model name; raises with the
  full attempt history if every real candidate also fails.
- pygeovision/ai/models/zoo.py (a separate, 98-entry metadata-only
  catalog with no build capability): 77 of 98 entries claimed
  pretrained_available=True with no real hf_model_id backing the
  claim. Corrected programmatically to match reality; removed the one
  remaining dinov3-as-dinov2 mislabeled entry.

Foundation models and SAR adapters
- pygeovision/models/foundation/prithvi.py: load_prithvi_hf (its own
  docstring: "recommended for most users") silently returned a
  randomly-initialized, untrained model with Prithvi's shape but zero
  real pretrained knowledge on any load failure -- the caller received
  what looked like a successful model object with no reliable
  indication anything was wrong. Now raises clearly by default;
  allow_random_init=True is a real, explicit opt-in, and the returned
  model is marked so it can't be silently mistaken for the real thing.
- pygeovision/models/adapters/sar_dinov3.py: the same pattern, zeros
  instead of random weights -- extract_features() silently returned
  all-zero "features" in the same shape as real output on failure.
  Same fix: raises by default, explicit allow_mock_on_failure opt-in.
- pygeovision/models/adapters/sar_channel_manager.py: two more real
  bugs. SARPrithviAdapter.run() silently returned an all-zero flood
  prediction on invalid input -- dangerous specifically because "all
  zero" looks like a real "no flood detected" finding, not an error
  state; now returns success=False with None instead. Separately,
  SARDINOv3Adapter.extract_features() had a compound integration bug
  exposed by fixing its own silent-fallback: the real model
  constructor's parameter names, the real method name, and the real
  input array format were all wrong simultaneously, meaning this
  integration had never worked once, with or without network access.
  Fixed all three; verified the corrected path reaches a real
  HuggingFace load attempt rather than failing on broken plumbing.

Pipeline system
- 59 pipelines audited; 10 new implementations added using real,
  documented techniques (solar_potential: Horn's method + solar
  position, verified against independent zenith-angle calculation;
  archaeological_site: real Local Relief Model, explicitly not
  automatic detection; wind_farm_siting: explicitly not a wind
  resource assessment; and others), each verified against synthetic
  ground truth designed to distinguish a correct result from an
  incorrect one.
- Removed InSAR entirely (permafrost_thaw, dam_safety, and two phantom
  prepare_insar_for_ai() docstring references that pointed to a
  function that existed nowhere in the codebase). 51 -> 49 pipelines.
  air_quality_index remains deliberately unimplemented -- no real
  calibrated data source exists, and the available proxy technique
  risks being read as health guidance under that specific name.
- Fixed real band-selection and AOI-coverage bugs in
  _search_and_download: requested bands were ignored and every band
  downloaded regardless; a bbox spanning multiple scene footprints
  silently returned a smaller-than-requested result. Added real
  greedy-selected multi-scene mosaicking with per-scene radiometric
  correction before merge.
- Found and fixed the identical band-count validation gap in a second,
  genuinely separate TiledInference engine
  (pygeovision.inference.tiled, reached via the CLI) that shares no
  code with the one already fixed in pygeovision.ai.inference --
  reproduced the original "expected 3 bands, got 13" failure directly
  against this engine to confirm the fix.
- Fixed a real CLI regression this cleanup caused: infer predict/batch
  defaulted --model to "unet-r50", a fake entry removed above. Routed
  both through ModelHub.load() for the real fallback mechanism instead
  of a new hardcoded name that could break the same way again.

Training
- The default scheduler (CosineAnnealingLR) was built with
  T_max=max_epochs while the loop steps it once per batch -- verified
  by simulation to produce 50 oscillation cycles instead of one smooth
  decay across a 10-epoch run. Affected every user not explicitly
  overriding the scheduler. Fixed to use total_steps, consistent with
  how linear/onecycle already handled this correctly in the same
  function.
- freeze_backbone was a real config field with zero effect on
  training; wired in and verified against a real
  segmentation_models_pytorch UNet (60/92 encoder params correctly
  freeze).
- Added real finetune-from-checkpoint support (didn't exist at all);
  verified against both an exact-match case and a 2-class-to-5-class
  finetuning case.
- GeoTrainer silently ignored the CLI's task selection, always
  training as if for segmentation. Added real detection-training
  support: correct collate_fn for variable-length boxes, correct
  torchvision forward pass, correct train()-mode validation-loss
  pattern. Verified end-to-end with a real RetinaNet.
- build_retinanet/build_fcos: pretrained=False did not prevent a
  network download (weights_backbone defaults independently).

Labeling
- Six functions across osm.py, buildings.py (Microsoft and Google),
  and sam_auto.py (auto-mask and GroundedSAM) returned success=True
  unconditionally regardless of whether any real content was found --
  a remote/rural query would report success with an empty, meaningless
  output and no indication anything was wrong. All six now correctly
  report failure when zero real results are found.
- DynamicWorldLabeler's download never checked the real HTTP status
  before writing to disk and declaring success -- a 403/404 error page
  could be silently written as a "real" label file. Fixed the same way
  in this path and a matching ESA WorldCover HTTPS fallback found
  nearby.

Dataset catalog
- DatasetLoader.download() (a real, CLI-reachable command) created an
  empty output directory and returned it as if a download had
  happened, for the 5 of 503 entries with a real download_url set.
  Checking those URLs directly found why no fetch logic existed: all
  5 point to human-facing project pages, not direct file links -- a
  real HTTP GET would have fetched HTML mislabeled as the dataset,
  worse than the bug it replaced. Now raises clearly for every real
  name; documented as a genuine, currently-open capability gap rather
  than silently left implicit.

Agent
- Systematically cross-checked every tool_name and pipeline_name
  string the planner and executor construct against the real, current
  registries, rather than trusting routing logic written before this
  project's pipelines were renamed and InSAR was removed.
- Two dangling pipeline names (solar_panels, road_network; real names
  solar_detection, road_extraction) in the heuristic planner's
  routing, and the identical two plus two more (crop_mapping,
  forest_monitoring) in EndToEndPipelineTool's own parameter schema --
  the schema an LLM-based planner reads to decide what's valid. Fixed
  in both places; verified end-to-end through a real GeoAgent run that
  the corrected name flows through planner, executor, and tool call.
- "subsidence"/"deformation" were still real, detected task keywords
  (a leftover from before InSAR removal) with no routing branch at
  all -- silently fell through to the same default used for queries
  the system doesn't understand, a generic land-cover plan. A user
  asking about ground subsidence would get a plan for something
  unrelated with no indication the real request failed. Now raises
  clearly, pointing to pygeofetch.insar directly.
- GeoAgentMemory.load() never restored conversation history despite
  its own log message claiming otherwise; verified via a real
  save/reload round trip that turns were silently lost, then fixed.
- The flood_mask follow-up-query convenience checked tool_name for
  "flood" (the real tool is named prithvi_inference and never matches)
  instead of the real task argument, exactly like the working
  land_cover_map check beside it. Verified with a synthetic trace
  matching the real flood pipeline's actual step structure that the
  bind never fired, then fixed.
- This completes the audit of every file in pygeovision.agent.

Dead code removed
- pygeovision/api/, enterprise/, serving/, utils/, viz/: 4,619 lines,
  confirmed zero imports anywhere (static and dynamic) before removal.
- pygeovision/core/engine.py: a superseded design wrapping
  pygeofetch's CLI, while the real, current data layer uses
  pygeofetch's Python API directly; confirmed dead including its
  lazy-import helper.
- 12 files that were pure 4-line facades re-exporting the registry
  with no unique content, several pointing at model names already
  confirmed fake above.

Documentation
- Rebuilt as a real Sphinx + MyST + sphinx_rtd_theme site (the
  previous docs predated this cleanup and still described removed
  modules as live), matching pygeofetch's own toolchain and section
  structure rather than inventing a different one. 24 pages, built
  successfully with only expected, harmless network-fetch warnings.
- Corrected a real documentation error found during the agent audit:
  client.pipeline() was described as pygeofetch's data-chain builder
  based on a stale assumption never re-checked against current source;
  it's a real, correct pygeovision shortcut. Fixed on both pages that
  repeated the error.
- Roadmap page follows the same honest Fixed/Found/Added format used
  for every finding above, including the capability gaps left
  genuinely open (tree_species has no real model behind it; ~700
  lines of confirmed-dead training modules not yet removed; SkySense
  and RS-YOLO deliberately not added to the registry, the former too
  complex to integrate responsibly, the latter not a single real,
  reliably-available model).

121 -> 62 general-purpose registry models, 51 -> 49 pipelines, 98 -> 97
zoo catalog entries, 5 directories and 12 files of confirmed-dead code
removed. Every fix in this commit was verified directly -- hand-
calculated where the claim was numerical (dNBR, PSI, GIoU/CIoU, focal
loss, linear regression trend), end-to-end through the real call path
where the claim was behavioral (model loading, pipeline execution,
agent planning) -- rather than assumed correct from reading the code
or its docstring alone.
