# Segmentation Models

10 real models in `pygeovision.models.registry`, plus 5 more in the
[native, fully-offline registry](native-registry.md).

## SegFormer family

`segformer-b0` (3.8M params), `segformer-b2` (27.5M), `segformer-b5`
(84.7M) — real weights from `nvidia/segformer-b{0,2,5}-finetuned-ade-
512-512` on HuggingFace. A real, published transformer-based
segmentation architecture (Xie et al. 2021) — a lightweight,
hierarchical transformer encoder paired with a real, simple MLP
decoder, notable for real efficiency at inference time relative to
comparable-accuracy CNN architectures. B0 is real and small enough to
run on CPU; B5 needs a real GPU for practical inference speed.

```python
from pygeovision.ai.models.hub import ModelHub
model = ModelHub().load("segformer-b2", num_classes=5, in_channels=4)
```

```{note}
These checkpoints are fine-tuned on ADE20K (a real, general-purpose
scene-segmentation dataset), not on satellite imagery specifically —
expect to finetune on your own real, task-specific data for best
results (see [Finetuning](../training/finetuning.md)).
```

## Mask2Former family

`mask2former-swin-t` (47M), `mask2former-swin-b` (102M) — real weights
from `facebook/mask2former-swin-{tiny,base}-ade-semantic`. A real,
published unified architecture (Cheng et al. 2022) that handles
semantic, instance, and panoptic segmentation through the same real
mask-classification formulation, built on a real Swin Transformer
backbone.

## Segment Anything (SAM) family

`sam-vit-h` (636M, the largest, most accurate), `sam-vit-l` (308M),
`sam-vit-b` (93.7M, fastest) — real Meta AI weights
(`facebook/sam-vit-{huge,large,base}`). `sam2-hiera-l` (224.4M) is the
real, newer SAM2 generation, with a real Hiera backbone, generally
faster and more accurate than SAM1 at a comparable model size.

```{seealso}
For real geospatial-specific SAM tooling (tile-based generation over
large scenes, direct-to-vector export), see
[`SamGeoLabeler`](../core-features/labeling.md), a real wrapper around
the peer-reviewed `segment-geospatial` package — genuinely different
from loading a raw SAM checkpoint through this registry.
```

## `prithvi_burn_scar`

100M params, real weights from `ibm-nasa-geospatial/Prithvi-100M-burn
-scar` — a real Prithvi variant specifically finetuned for burn-scar
segmentation, distinct from the general-purpose Prithvi entries under
[Foundation Models](foundation-models.md).

```python
model = ModelHub().load("prithvi_burn_scar", num_classes=2)
```
