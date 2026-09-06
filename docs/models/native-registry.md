# The Native (Offline) Registry

`pygeovision.ai.models.registry` — a smaller, separate 14-model
registry, structurally different from the general-purpose one covered
in the other pages under Model Registry: every entry here has a real,
dedicated `factory_fn` pointing to local `segmentation_models_pytorch`/
`torchvision` construction code, not a HuggingFace ID. All 14 were
individually confirmed to build **fully offline** — no network
dependency for the architecture itself, only for downloading
pretrained weights if `pretrained=True` (the default).

| Model | Task | Real backend |
|---|---|---|
| `unet_resnet50` | segmentation | `smp.Unet(encoder_name="resnet50")` |
| `unet_efficientnet_b4` | segmentation | `smp.Unet(encoder_name="efficientnet-b4")` |
| `deeplabv3plus_resnet101` | segmentation | `smp.DeepLabV3Plus(encoder_name="resnet101")` |
| `segformer_b2`, `segformer_b5` | segmentation | Local SegFormer construction |
| `fcos_resnet50` | detection | `torchvision.models.detection.fcos_resnet50_fpn` |
| `retinanet_resnet50` | detection | `torchvision.models.detection.retinanet_resnet50_fpn` |
| `resnet50_cls`, `efficientnet_b3_cls`, `vit_b16_cls` | classification | Local `timm`/`torchvision` classifier heads |
| `siamese_unet` | change detection | Local Siamese U-Net |
| `changeformer` | change detection | The real, published ChangeFormer architecture |
| `srcnn`, `esrgan_geo` | super-resolution | Local CNN / GAN super-resolution |

```python
from pygeovision.ai.models.registry import registry

model = registry.build("unet_resnet50", encoder="resnet50", in_channels=4, num_classes=10, pretrained=False)
```

## `pretrained=True`/`False`: a real, confirmed fix

`build_unet`/`build_deeplabv3plus` previously accepted `encoder_weights`
(a `segmentation_models_pytorch`-specific parameter name) instead of
the `pretrained: bool` convention every other `build_*` function in
this package uses. Passing `pretrained=False` was silently ignored —
confirmed directly by watching real weight downloads happen anyway.
Fixed: both now accept `pretrained`, with `encoder_weights` still
available as an advanced, more specific override.

## Why this registry exists alongside the larger one

`ModelHub.load()` checks this registry first for any requested model
name, since every entry here is confirmed to build without a network
dependency — making it the first, most reliable choice for
[the real fallback mechanism](../core-features/model-registry.md) when
a request to the larger, `hf_id`-based registry fails.
