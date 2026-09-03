# Commit Messages

Six atomic, conventional-format commits covering this session's fixes across
pipeline preprocessing, AOI coverage, new pipeline implementations, and the
training/finetuning system. Use individually or squash as needed.

---

## 1. Pipeline band selection and AOI coverage

```
fix(pipelines): correct band selection and AOI coverage before model inference

_search_and_download previously downloaded every asset in a scene
regardless of the requested `bands` tuple, since client.download()'s
real bands= parameter was never used. Fixed by expanding each generic
band name (e.g. "red") into every real, known alias across Sentinel-2
and both Landsat mission families before calling download(), confirmed
against pygeofetch's official documentation.

Also adds real AOI coverage checking: previously only the single
best-scored scene was ever downloaded, with no check that its real
footprint fully covered the requested bbox. Partial overlap silently
produced a smaller-than-requested result. Now checks coverage via
SearchResult.bbox before downloading, and when a single scene isn't
enough, downloads + individually radiometrically-corrects each scene
in a greedy-selected covering set before mosaicking (rasterio.merge)
and cropping to the exact requested bbox.

Adds three layers of defense against band-count mismatches reaching
_run_model: real-time validation in the single-asset download path,
a final check before _search_and_download returns, and an explicit
check immediately before model construction in _run_model.

Verified: real multi-tile mosaic test (output contains pixel values
from both source tiles), band-mismatch scenario correctly refused
instead of silently passed through, full regression suite across
single-image/bi-temporal/DEM pipelines.
```

---

## 2. ESAWorldCoverLabeler silent failure

```
fix(labeling): ESAWorldCoverLabeler silently reported success on failed downloads

When real WorldCover reference tiles could not be downloaded (network
failure, no credentials), label_tile() wrote an all-zero background
mask and returned a LabelingResult with no error/skipped set. Since
LabelingResult.success is a computed property checking only those
fields, this silently evaluated to success=True -- LandCoverPipeline
and everything delegating to it (MangroveMapping, LandcoverChange,
BiodiversityHotspot) could report a meaningful-looking classification
result that was actually empty.

Found while testing WindFarmSitingPipeline's land-use exclusion step,
which hit this exact failure mode. Now explicitly sets `error` when no
real tiles are found.
```

---

## 3. New pipeline implementations

```
feat(pipelines): implement 10 previously-stub pipelines with real techniques

Replaces NotImplementedError stubs with real, empirically-verified
implementations: solar_potential, archaeological_site (real DEM/GIS
techniques -- Horn's method slope/aspect, solar position, Local Relief
Model), aquaculture_mapping (shape-regularity classification),
wind_farm_siting (terrain/land-use suitability screen, explicitly not
a wind resource assessment), mine_detection, construction_progress
(size/class-filtered land-cover transitions), powerline_extraction
(linear-corridor detection via shape elongation), reef_bleaching,
dust_storm_tracking (bi-temporal anomaly proxies).

air_quality_index deliberately remains unimplemented -- the same
proxy technique used for dust_storm_tracking would risk being read as
health guidance it cannot honestly provide.

Fixes a severe 4-connectivity bug found via testing: scipy.ndimage.label's
default connectivity fragmented a thin diagonal linear corridor into
20 isolated single-pixel regions instead of 1. Fixed with 8-connectivity
across all affected pipelines.

Each implementation independently verified against synthetic ground
truth (e.g. solar_potential: south-facing terrain scores higher than
flat at winter solstice, matching physical expectation; aquaculture
mapping: rectangular pond flagged, winding river correctly not flagged).
```

---

## 4. Training: freeze_backbone, checkpoint loading, scheduler bug

```
fix(training): freeze_backbone, checkpoint loading, and default scheduler bug

TrainingConfig.freeze_backbone was defined but referenced nowhere else
in trainer.py -- setting it to True had zero effect on training.
Now actually freezes backbone/encoder parameters by name pattern before
the optimizer is built. Verified against a real segmentation_models_pytorch
UNet: 60/92 encoder parameters correctly freeze, decoder stays trainable.

Adds real checkpoint_path support for finetuning from an existing
checkpoint -- previously no such capability existed at all. Uses
strict=False with a clear mismatch report, verified against both an
exact-match case and a realistic finetuning case (2-class checkpoint
loaded into a 5-class model: encoder weights transfer correctly, only
the differently-shaped head is left for fresh training).

Fixes a severe bug in the DEFAULT scheduler: the training loop steps
the scheduler once per batch, but CosineAnnealingLR was built with
T_max=cfg.max_epochs (epoch count, not step count). Verified via
simulation: with max_epochs=10 and 50 steps/epoch, this produced 50
full oscillation cycles instead of one smooth decay across the run.
Since cosine is the default, every user not explicitly overriding the
scheduler was affected. Fixed to use total_steps, consistent with how
"linear"/"onecycle" already handled this correctly in the same function.
Same unit-mismatch bug fixed in "step" and the fallback branch.
```

---

## 5. Training: real object-detection support

```
feat(training): real object-detection training support

GeoTrainer.fit() previously hardcoded CrossEntropyLoss + SegmentationMetrics
unconditionally, with zero task branching, despite the CLI accepting
a task=[segmentation,detection] choice that was silently discarded
(never passed into TrainingConfig at all).

Adds a real `task` field and full detection-training support: a
collate_fn for variable-length bounding-box targets, the correct
torchvision detection forward pass (model(images, targets) -> loss
dict, summed -- no external loss_fn, since the model computes its own
loss internally), and the documented train()-mode + no_grad() pattern
for validation loss (these models only compute losses in train mode
with real targets supplied; eval mode returns predictions instead).

Also fixes build_retinanet/build_fcos: pretrained=False did not
actually prevent a network download, since torchvision's
weights_backbone parameter defaults to a pretrained ResNet50
independently of the main weights arg.

Fixes a mislabeled results dict: "best_val_iou" was always present
even for detection, where the tracked value is actually val_loss
(confirmed by a test run showing "best_val_iou": 1.77, impossible for
a bounded-0-1 IoU). Now reports the real metric name and value, and
only includes best_val_iou when it's genuinely an IoU.

Verified end-to-end with a real torchvision RetinaNet and synthetic
bounding-box data; segmentation and detection tested side by side to
confirm neither path broke the other.
```

---

## 6. Training: ONNX export error handling

```
fix(training): actionable error when ONNX export needs onnxscript

Modern PyTorch's default ONNX exporter requires onnxscript separately
from the top-level onnx package this code already checked for. Without
it, torch.onnx.export() raised a raw, unwrapped ModuleNotFoundError
with no indication of what to install. Found by hitting this directly
while verifying the export path end-to-end (confirmed exported models
are numerically identical to the source PyTorch model via onnxruntime).
```
