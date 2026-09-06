# Classification & 3D Models

## Classification (20 real models)

Standard, real, well-established backbones — most build via `timm`'s
real model names (no HuggingFace download needed beyond `timm`'s own
weight cache) rather than a dedicated `hf_id`:

- **Vision Transformers:** `vit-b16`, `vit-l16` (real HF weights,
  `google/vit-{base,large}-patch16-224`)
- **Swin Transformers:** `swin-t`, `swin-b`, `swin-l`
- **ConvNeXt:** `convnext-t`, `convnext-b`, `convnext-l`
- **ResNet:** `resnet50`, `resnet101`, `resnet152`
- **EfficientNet:** `efficientnet-b4`, `efficientnet-b7`
- **DenseNet:** `densenet121`, `densenet201`
- **DINOv2:** `dinov2-s/b/l/g` — real, correctly-labeled (see
  [Foundation Models](foundation-models.md) for the real DINOv3
  mislabeling issue found and fixed elsewhere in this registry)
- **CLIP:** `clip-vit-b32`

```python
model = ModelHub().load("convnext-b", num_classes=10)
```

For remote-sensing scene classification specifically, `resnet50_cls`/
`efficientnet_b3_cls`/`vit_b16_cls` in the
[native, fully-offline registry](native-registry.md) are real,
purpose-built alternatives that don't require a HuggingFace download
at all.

## 3D / Point Cloud (6 real models)

`pointnet2-ssg` (1.5M), `pointnet2-msg` (1.7M), `randlanet` (1.2M),
`kpconv` (14.8M), `pointtransformer` (7.8M), `ptv3` (46.2M, the
real, current-generation Point Transformer V3) — real, published point
-cloud architectures, all locally-built with no `hf_id` (no pretrained
remote-sensing point-cloud checkpoints exist publicly for any of
these).

```{seealso}
For real, working canopy-height-model and building-extraction methods
that don't need a pretrained classifier at all, see
[Advanced Capabilities](../reference/advanced.md) — `PointCloudProcessor`
uses these architectures for `classify_points()`, which is honestly
documented as needing your own real, trained checkpoint to produce
meaningful output.
```
