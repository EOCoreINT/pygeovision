# Model Registry

`pygeovision` has two real model registries, used for different real
purposes, plus a real, browsable catalog with no build capability.

## The two registries

| Registry | Real models | What it's for |
|---|---|---|
| `pygeovision.models.registry` | 77 | General-purpose: classification, detection, segmentation, change-detection, and foundation models, most backed by a real `hf_id` or `timm_id` |
| `pygeovision.ai.models.registry` | 14 | Smaller, every entry confirmed to build **fully offline** — real `segmentation_models_pytorch`/`torchvision` backends, no network dependency for the architecture itself |

`ModelHub.load(name)` (in `pygeovision.ai.models.hub`) checks both.

## What "real" means here

A full audit of the general-purpose registry found **59 of 121
original entries** (roughly half) had no genuine, working backing —
either no real `hf_id`/`timm_id` at all, or a "real" `hf_id` that
turned out to be a different, unrelated model silently substituted
under a misleading name. All 59 were removed initially, rather than
left as plausible-looking dead ends. Examples of what was found:

- Every `dinov3_*` **backbone** entry: labeled DINOv3, but the real
  `hf_id` pointed to `facebook/dinov2-*` — genuine DINOv2 weights, not
  the real, separately-released DINOv3. The "SAT-493M
  satellite-pretrained" variants had the identical issue: the same
  generic, non-satellite DINOv2 weights regardless of which "`_sat`"
  name was requested. These stay removed — there's no honest, useful
  stub for a mislabeled weight source, only a corrected one (real
  DINOv2 entries remain, correctly labeled).
- `unet-r50`/`unet-r101`/`unet-efficientb4`: all three silently
  returned the *identical* generic fallback U-Net regardless of the
  backbone named in the model string — confirmed by direct testing,
  not assumed from the name.
- Roughly 40 more entries (`yolov8-*`, `deeplab-*`, `changestar-*`,
  `satlas-*`, `centernet-*`, and others) had no real weights, and no
  real, dedicated construction code, anywhere in the codebase.

Two entries were fixed rather than removed, because real code already
existed and just wasn't wired up: `changeformer-mit-b0`/`b4` (a real,
203-line implementation existed but the registry never called it) and
`dofa-base` (a real model, but the registry's `hf_id` pointed to a
repo name — `XShadow/DOFA-ViT-base-p16` — that doesn't exist; the real
repo is `XShadow/DOFA`).

**A later reconsideration, found by reading this project's own test
suite**: removing a fake entry entirely means a user searching for
that real, named architectural concept finds nothing at all —
indistinguishable from "never heard of it." For entries that are real,
named concepts in the field with no working implementation here
specifically (as opposed to entries that never should have existed
under that name at all, like the DINOv3-mislabeled backbones), keeping
them registered but making `get_model()` honestly raise is more useful
than silent absence. Nine entries were restored this way: `unet-r50`/
`unet-r101`/`unet-efficientb4` (see "Honest entries" below — the
*concept* wasn't fake, the *implementation* was), plus
`deeplab-r50`/`deeplab-r101`, `pspnet-r50`, `fcn-r50`, `upernet-swin-b`,
and `dinov3_segmentor` (a segmentation head that genuinely can't work
regardless of backbone mislabeling, since no real DINOv3 backbone
exists here to build it on).

```{seealso}
The full accounting, including which specific entries were removed
and why, is in [Roadmap](../reference/roadmap.md).
```

## Honest entries: real, but no verified build path

Several entries are deliberately kept **without** a working
`get_model()` path, rather than guessing at one or silently deleting
the concept:

- **`lisat-7b`** — a real, published (arXiv:2505.02829) vision-language
  model for geospatial reasoning segmentation, with real HuggingFace
  weights at `jquenum/LISAt-7b`. Its own model card's usage example
  (`AutoModelForImageSegmentation.from_pretrained(...)`) doesn't match
  its real, documented architecture — a custom LISA-style multimodal
  LLM + SAM decoder needing `flash-attn` and a separate RemoteCLIP
  checkpoint. Calling `get_model("lisat-7b")` raises a clear error
  explaining exactly what's needed, rather than attempting an
  unverified load.
- **`centernet-r50`** — `torchvision` has no real CenterNet
  implementation. Calling it raises clearly rather than silently
  substituting the architecturally-different FCOS model under
  CenterNet's name.
- **`changestar-r18`**, **`satlas-pretrain`**, **`chatearthnet`** — real,
  named architectures/models in the field with no real construction
  code or weights in this codebase.
- **`unet-r50`**, **`unet-r101`**, **`unet-efficientb4`** — the
  *concept* (a U-Net with a specific real encoder) isn't fake; the
  registry's implementation was, silently building the identical
  generic fallback for all three regardless of the named backbone. For
  a real, working, backbone-specific U-Net, use `unet_resnet50`/
  `unet_efficientnet_b4` in the native registry
  (`pygeovision.ai.models.registry`) instead.
- **`deeplab-r50`/`r101`**, **`pspnet-r50`**, **`fcn-r50`**,
  **`upernet-swin-b`** — real, published segmentation architectures
  with no real construction code or weights here.
- **`dinov3_segmentor`** — a segmentation head for a DINOv3 backbone.
  Genuinely can't work regardless of the head implementation itself,
  since no real, non-mislabeled DINOv3 backbone exists in this
  codebase to build it on.

Every one of these is discoverable via `list_models()` — the honest
failure happens at `get_model()`, with a message explaining what's
actually missing.

## Real, verified integrations

```{seealso}
For the complete, detailed catalog of every real model in every
category — segmentation, detection, change detection, foundation
models, VLMs, classification, 3D — see
[Models — Full Catalog](../models/index.md).
```

- **SAM / SAM2**: `sam-vit-h/l/b`, `sam2-hiera-l` — real `hf_id`
  entries, plus a real wrapper (`SamGeoLabeler`) around
  [segment-geospatial](https://samgeo.gishub.org) (Wu & Osco, 2023,
  *JOSS*), a real, actively-maintained, peer-reviewed package —
  genuinely different from `pygeovision`'s own from-scratch
  `SAMAutoLabeler`, offering real tile-based generation for large
  scenes and real direct-to-vector export.
- **RF-DETR**: `rf-detr-b/l` — real DINOv2+DETR hybrid detection.
- **Prithvi**: 5 real variants (`prithvi-100m/300m`,
  `prithvi_eo_1_0/2_0`, `prithvi_burn_scar`).
- **SwinIR/Swin2SR**: `swinir-m` — the real hybrid CNN-transformer
  super-resolution architecture.

## Real fallback on genuine build failure

`ModelHub.load()` doesn't just fail if a named model can't build (e.g.
a real `hf_id` model failing offline). It determines the model's real
task, and tries other real, verified models for that same task —
the fully-offline native registry first, then the larger registry.
The returned model is marked so you can tell a fallback happened:

```python
from pygeovision.ai.models.hub import ModelHub

model = ModelHub().load("segformer-b0", num_classes=2)
if getattr(model, "_pygeovision_fallback_used", False):
    print(f"Requested {model._pygeovision_requested_name}, "
          f"got {model._pygeovision_actual_name} instead")
```

This never invents a model — only real, registry-verified names are
ever tried, and if every real candidate for the task also fails, it
raises with the full attempt history rather than returning `None` or
a wrong-task model silently.

## Real, silent-fallback bugs found in model *code* (not the registry)

Separately from registry entries, three real, severe bugs were found
in model-loading *code* during this audit — cases where a function
silently returned a fake, meaningless result on failure instead of
raising:

- `load_prithvi_hf()` — silently returned a randomly-initialized,
  untrained model with Prithvi's shape but zero real pretrained
  knowledge whenever the real HuggingFace load failed. Now raises
  clearly by default; an explicit `allow_random_init=True` is required
  to opt into the old behavior, and the returned model is then
  programmatically marked so it's never mistaken for the real thing.
- A SAR-DINOv3 feature extractor — silently returned all-zero
  "features" in the same shape as real output on failure. Same fix:
  raises by default, explicit opt-in required.
- A SAR flood-detection adapter — silently returned an all-zero
  prediction on invalid input, which is particularly dangerous for a
  flood-detection task, since "all zero" looks like a real "no flood
  detected" finding rather than an error state. Now returns
  `success: False` with `None` values instead.

See [Roadmap](../reference/roadmap.md) for the complete list.
