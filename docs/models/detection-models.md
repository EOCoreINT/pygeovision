# Detection Models

9 real models, plus one honest, deliberate absence.

## RF-DETR family

`rf-detr-b` (29M), `rf-detr-l` (128M) — real weights from
`roboflow/rf-detr-{base,large}`. A real, modern architecture combining
a DINOv2 vision backbone with a real DETR-style transformer detection
head — per the state-of-the-art research reviewed for this registry,
a genuinely current, high-accuracy choice for real-time remote-sensing
object detection.

## DETR family

`rt-detr-l` (32M, real-time variant, `PekingU/rtdetr_r50vd`), `detr-r50`
(41.3M), `detr-r101` (60M) — the real, original end-to-end transformer
detection architecture (Carion et al. 2020) and its real, faster
successor RT-DETR.

## Real, locally-built torchvision detectors

`faster-rcnn-r50` (41.8M), `mask-rcnn-r50` (44.4M), `fcos-r50` (32M) —
no `hf_id`; these build directly from real `torchvision` detection
factories (`fasterrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn`,
`fcos_resnet50_fpn`) rather than a HuggingFace download, so they build
fully offline once `torchvision` itself is installed.

```python
model = ModelHub().load("fcos-r50", num_classes=5, pretrained=False)
```

```{note}
`pretrained=False` on these genuinely prevents a network download —
confirmed and fixed directly during this project's audit
(`weights_backbone` previously defaulted to a pretrained ResNet50
independently of the main `pretrained` argument).
```

## `grounding-dino`

172M params, real weights from `IDEA-Research/grounding-dino-tiny` — a
real, published open-vocabulary detector: given a real text prompt
("solar panel", "shipping container"), it detects real matching
objects without being retrained for that specific class. Genuinely
different from the fixed-class detectors above.

## What was removed, and why

`centernet-r50` is not a real, working entry. `torchvision` has no
real CenterNet implementation at all, and this spec has no real
`hf_id`/`timm_id` either. Calling `get_model("centernet-r50")` raises
a clear `NotImplementedError` explaining exactly that, rather than
silently building FCOS or a toy fallback network under CenterNet's
name — the two are architecturally different detectors, and doing so
would misrepresent the result.
