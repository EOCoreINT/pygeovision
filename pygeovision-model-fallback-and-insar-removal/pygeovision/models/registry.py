"""
Central model registry — 50+ geospatial architectures.
Each entry stores metadata and a factory function.
"""
from __future__ import annotations

import builtins
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    task: str
    family: str
    params_m: float
    hf_id: str | None = None
    timm_id: str | None = None
    description: str = ""
    supports_multispectral: bool = True
    pretrained_on: str = "imagenet"
    paper: str = ""
    factory: Callable | None = field(default=None, repr=False)


# ── Registry data ──────────────────────────────────────────────────────────────
_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec | None:
    return _REGISTRY.get(name)


def list_models(task: str | None = None, family: str | None = None) -> list[str]:
    return [n for n, s in _REGISTRY.items()
            if (task is None or s.task == task) and (family is None or s.family == family)]


# ── 50+ model specifications ───────────────────────────────────────────────────
_SPECS = [
    # ── Classification (20) ──────────────────────────────────────────────────
    ModelSpec("vit-b16",         "classification", "vit",       86.6,  hf_id="google/vit-base-patch16-224",      description="Vision Transformer Base"),
    ModelSpec("vit-l16",         "classification", "vit",      307.0,  hf_id="google/vit-large-patch16-224",     description="Vision Transformer Large"),
    ModelSpec("swin-t",          "classification", "swin",      28.3,  timm_id="swin_tiny_patch4_window7_224",    description="Swin Transformer Tiny"),
    ModelSpec("swin-b",          "classification", "swin",      87.8,  timm_id="swin_base_patch4_window7_224",    description="Swin Transformer Base"),
    ModelSpec("swin-l",          "classification", "swin",     196.5,  timm_id="swin_large_patch4_window7_224",   description="Swin Transformer Large"),
    ModelSpec("convnext-t",      "classification", "convnext",  28.6,  timm_id="convnext_tiny",                   description="ConvNeXt Tiny"),
    ModelSpec("convnext-b",      "classification", "convnext",  88.6,  timm_id="convnext_base",                   description="ConvNeXt Base"),
    ModelSpec("convnext-l",      "classification", "convnext", 197.8,  timm_id="convnext_large",                  description="ConvNeXt Large"),
    ModelSpec("resnet50",        "classification", "resnet",    25.6,  timm_id="resnet50",                        description="ResNet-50"),
    ModelSpec("resnet101",       "classification", "resnet",    44.5,  timm_id="resnet101",                       description="ResNet-101"),
    ModelSpec("resnet152",       "classification", "resnet",    60.2,  timm_id="resnet152",                       description="ResNet-152"),
    ModelSpec("efficientnet-b4", "classification", "efficientnet", 19.3, timm_id="efficientnet_b4",               description="EfficientNet-B4"),
    ModelSpec("efficientnet-b7", "classification", "efficientnet", 66.3, timm_id="efficientnet_b7",               description="EfficientNet-B7"),
    ModelSpec("densenet121",     "classification", "densenet",   8.0,  timm_id="densenet121",                     description="DenseNet-121"),
    ModelSpec("densenet201",     "classification", "densenet",  20.0,  timm_id="densenet201",                     description="DenseNet-201"),
    ModelSpec("dinov2-s",        "classification", "dinov2",    21.0,  hf_id="facebook/dinov2-small",             description="DINOv2 Small", pretrained_on="LVD-142M"),
    ModelSpec("dinov2-b",        "classification", "dinov2",    86.0,  hf_id="facebook/dinov2-base",              description="DINOv2 Base",  pretrained_on="LVD-142M"),
    ModelSpec("dinov2-l",        "classification", "dinov2",   307.0,  hf_id="facebook/dinov2-large",             description="DINOv2 Large", pretrained_on="LVD-142M"),
    ModelSpec("dinov2-g",        "classification", "dinov2",  1100.0,  hf_id="facebook/dinov2-giant",             description="DINOv2 Giant", pretrained_on="LVD-142M"),
    ModelSpec("clip-vit-b32",    "classification", "clip",      151.0, hf_id="openai/clip-vit-base-patch32",      description="CLIP ViT-B/32"),

    # ── Detection (17) ───────────────────────────────────────────────────────
    ModelSpec("rf-detr-b",       "detection", "detr",   29.0,   hf_id="roboflow/rf-detr-base",      description="RF-DETR Base"),
    ModelSpec("rf-detr-l",       "detection", "detr",   128.0,  hf_id="roboflow/rf-detr-large",     description="RF-DETR Large"),
    ModelSpec("rt-detr-l",       "detection", "detr",   32.0,   hf_id="PekingU/rtdetr_r50vd",       description="RT-DETR Large"),
    ModelSpec("detr-r50",        "detection", "detr",   41.3,   hf_id="facebook/detr-resnet-50",    description="DETR ResNet-50"),
    ModelSpec("detr-r101",       "detection", "detr",   60.0,   hf_id="facebook/detr-resnet-101",   description="DETR ResNet-101"),
    ModelSpec("faster-rcnn-r50", "detection", "rcnn",   41.8,   description="Faster R-CNN ResNet-50"),
    ModelSpec("mask-rcnn-r50",   "detection", "rcnn",   44.4,   description="Mask R-CNN ResNet-50"),
    ModelSpec("fcos-r50",        "detection", "anchor_free", 32.0, description="FCOS ResNet-50"),
    ModelSpec("grounding-dino",  "detection", "vlm",   172.0,   hf_id="IDEA-Research/grounding-dino-tiny", description="Grounding DINO"),

    # ── Segmentation (17) ────────────────────────────────────────────────────
    ModelSpec("segformer-b0",    "segmentation", "segformer",  3.8,  hf_id="nvidia/segformer-b0-finetuned-ade-512-512", description="SegFormer B0"),
    ModelSpec("segformer-b2",    "segmentation", "segformer", 27.5,  hf_id="nvidia/segformer-b2-finetuned-ade-512-512", description="SegFormer B2"),
    ModelSpec("segformer-b5",    "segmentation", "segformer", 84.7,  hf_id="nvidia/segformer-b5-finetuned-ade-512-512", description="SegFormer B5"),
    ModelSpec("mask2former-swin-t","segmentation","mask2former",47.0, hf_id="facebook/mask2former-swin-tiny-ade-semantic", description="Mask2Former Swin-T"),
    ModelSpec("mask2former-swin-b","segmentation","mask2former",102.0,hf_id="facebook/mask2former-swin-base-ade-semantic", description="Mask2Former Swin-B"),
    ModelSpec("sam-vit-h",       "segmentation", "sam",      636.0,  hf_id="facebook/sam-vit-huge",   description="SAM ViT-H"),
    ModelSpec("sam-vit-l",       "segmentation", "sam",      308.0,  hf_id="facebook/sam-vit-large",  description="SAM ViT-L"),
    ModelSpec("sam-vit-b",       "segmentation", "sam",       93.7,  hf_id="facebook/sam-vit-base",   description="SAM ViT-B"),
    ModelSpec("sam2-hiera-l",    "segmentation", "sam2",     224.4,  hf_id="facebook/sam2-hiera-large",description="SAM2 Hiera-L"),

    # ── Change Detection (10) ────────────────────────────────────────────────
    ModelSpec("changeformer-mit-b0","change_detection","changeformer", 13.9, description="ChangeFormer MiT-B0"),
    ModelSpec("changeformer-mit-b4","change_detection","changeformer", 67.4, description="ChangeFormer MiT-B4"),
    ModelSpec("bit-r50",         "change_detection","bit",         26.1, description="BIT ResNet-50"),
    ModelSpec("dsamnet",         "change_detection","dsamnet",     16.0, description="DSAMNet"),

    # ── Foundation Models (11) ───────────────────────────────────────────────
    ModelSpec("prithvi-100m",    "foundation","prithvi",  100.0, hf_id="ibm-nasa-geospatial/Prithvi-100M",   description="NASA/IBM Prithvi 100M (multitemporal)", pretrained_on="HLS"),
    ModelSpec("prithvi-300m",    "foundation","prithvi",  300.0, hf_id="ibm-nasa-geospatial/Prithvi-300M",   description="NASA/IBM Prithvi 300M", pretrained_on="HLS"),
    ModelSpec("dofa-base",       "foundation","dofa",      86.0, hf_id="XShadow/DOFA-ViT-base-p16",          description="Dynamic One-For-All (multi-sensor)", pretrained_on="Sentinel-1/2,Landsat"),
    ModelSpec("tessera",         "foundation","tessera",   0.0, description="TESSERA precomputed embeddings (Sentinel-1+2, 128ch/10m) — real, precomputed via the geotessera library, not a locally-run encoder", pretrained_on="Sentinel-1+2"),
    ModelSpec("alphaearth",      "foundation","alphaearth", 0.0, description="AlphaEarth Foundations / Satellite Embedding (64ch/10m, annual 2017+) — real, precomputed via Google Earth Engine, not a locally-run encoder", pretrained_on="Multi-sensor (Sentinel-1/2, Landsat, etc.)"),
    ModelSpec("remoteclip-b32",  "foundation","clip",     151.0, hf_id="BAAI/RemoteCLIP-ViT-B-32", description="RemoteCLIP ViT-B/32", pretrained_on="RS5M"),
    ModelSpec("remoteclip-l14",  "foundation","clip",     428.0, hf_id="BAAI/RemoteCLIP-ViT-L-14", description="RemoteCLIP ViT-L/14", pretrained_on="RS5M"),

    # ── Vision-Language (7) ──────────────────────────────────────────────────
    ModelSpec("moondream2",      "vlm","moondream",   1800.0, hf_id="vikhyatk/moondream2", description="Moondream2 (satellite VQA)"),
    ModelSpec("openclip-b32",    "vlm","clip",          151.0, hf_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", description="OpenCLIP ViT-B/32"),
    ModelSpec("openclip-l14",    "vlm","clip",          428.0, hf_id="openai/clip-vit-large-patch14",          description="OpenCLIP ViT-L/14"),

    # ── 3D / LiDAR (6) ──────────────────────────────────────────────────────
    ModelSpec("pointnet2-ssg",   "3d","pointnet",    1.5,  description="PointNet++ SSG"),
    ModelSpec("pointnet2-msg",   "3d","pointnet",    1.7,  description="PointNet++ MSG"),
    ModelSpec("randlanet",       "3d","randlanet",   1.2,  description="RandLA-Net"),
    ModelSpec("kpconv",          "3d","kpconv",      14.8, description="KPConv"),
    ModelSpec("pointtransformer","3d","transformer",  7.8, description="Point Transformer"),
    ModelSpec("ptv3",            "3d","transformer", 46.2, description="Point Transformer V3"),

    # ── Time Series (6) ─────────────────────────────────────────────────────

    # ── Super-Resolution (4) ─────────────────────────────────────────────────
    ModelSpec("swinir-m",        "super_resolution","swin", 11.8, hf_id="caidas/swin2SR-realworld-sr-x4-64",  description="SwinIR Medium"),

    # ── DINOv3 — all 12 variants ─────────────────────────────────────────────────

    # ── DINOv3 task heads ────────────────────────────────────────────────────────

    # ── Prithvi ─────────────────────────────────────────────────────────────────
    ModelSpec("prithvi_eo_1_0",    "foundation","prithvi",  100.0,  hf_id="ibm-nasa-geospatial/Prithvi-100M",          description="Prithvi-EO-1.0 100M — HLS US 30m",    pretrained_on="HLS-US",    supports_multispectral=True),
    ModelSpec("prithvi_eo_2_0",    "foundation","prithvi",  600.0,  hf_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",   description="Prithvi-EO-2.0 600M — HLS Global 30m", pretrained_on="HLS-Global",supports_multispectral=True),
    ModelSpec("prithvi_burn_scar", "segmentation","prithvi", 100.0, hf_id="ibm-nasa-geospatial/Prithvi-100M-burn-scar", description="Prithvi fine-tuned burn scar",         pretrained_on="HLS-US"),

]

for spec in _SPECS:
    register_model(spec)

logger.info("Model registry: %d architectures loaded", len(_REGISTRY))


# ── Public API ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Queryable registry of all PyGeoVision model architectures."""

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __contains__(self, name: str) -> bool:
        return name in _REGISTRY

    def __getitem__(self, name: str) -> ModelSpec:
        if name not in _REGISTRY:
            raise KeyError(f"Model '{name}' not found. Use list_models() to see all options.")
        return _REGISTRY[name]

    def list(self, task: str | None = None, family: str | None = None,
              max_params_m: float | None = None) -> builtins.list[str]:
        return [n for n, s in _REGISTRY.items()
                if (task is None or s.task == task)
                and (family is None or s.family == family)
                and (max_params_m is None or s.params_m <= max_params_m)]

    def search(self, query: str) -> builtins.list[ModelSpec]:
        q = query.lower()
        return [s for s in _REGISTRY.values()
                if q in s.name.lower() or q in s.description.lower()
                or q in s.task.lower() or q in s.family.lower()]

    def by_task(self) -> dict[str, builtins.list[str]]:
        tasks: dict[str, list[str]] = {}
        for n, s in _REGISTRY.items():
            tasks.setdefault(s.task, []).append(n)
        return tasks

    def top_by_task(self, task: str, n: int = 5) -> builtins.list[ModelSpec]:
        """Return top-n models for a task, sorted by param count."""
        models = [s for s in _REGISTRY.values() if s.task == task]
        return sorted(models, key=lambda s: s.params_m)[:n]

    def summary(self) -> dict:
        by_task = self.by_task()
        return {
            "total": len(_REGISTRY),
            "by_task": {t: len(ms) for t, ms in by_task.items()},
            "with_hf_weights": sum(1 for s in _REGISTRY.values() if s.hf_id),
            "with_timm_weights": sum(1 for s in _REGISTRY.values() if s.timm_id),
        }


model_registry = ModelRegistry()


def get_model(name: str, num_classes: int = 2, in_channels: int = 4,
               pretrained: bool = True, device: str | None = None,
               **kwargs) -> Any:
    """Load a model by name with geospatial configuration.

    Args:
        name: Model name from the registry (e.g. "segformer-b2", "unet-r50")
        num_classes: Number of output classes
        in_channels: Number of input channels (4 for Sentinel-2 BGRN)
        pretrained: Load pretrained weights
        device: Target device ("cuda", "cpu", "mps")

    Returns:
        Loaded PyTorch model ready for inference/fine-tuning

    Example::

        model = get_model("segformer-b2", num_classes=7, in_channels=4)
    """
    spec = model_registry[name]
    model = _build_model(spec, num_classes=num_classes, in_channels=in_channels,
                          pretrained=pretrained, **kwargs)
    if device:
        model = model.to(device)
    return model


def _build_model(spec: ModelSpec, num_classes: int, in_channels: int,
                  pretrained: bool, **kwargs) -> Any:
    """Factory: build a model from its spec."""
    # CLIP-family and Moondream are VLMs with a genuinely different usage
    # pattern (zero-shot classification, image-text embedding, VQA/caption
    # — not a single-tensor-in/logits-out forward pass) and, for
    # RemoteCLIP/GeoRSCLIP specifically, a different loading mechanism
    # entirely (open_clip checkpoints, not transformers-native). Dedicated,
    # correct wrappers already exist for these — route to them BEFORE the
    # generic timm/HF dispatch below, which would otherwise try (and fail,
    # or silently produce something unusable) a generic AutoModel load.
    if spec.family == "clip":
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        return CLIPGeo(model=spec.name)
    if spec.family == "moondream":
        from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
        return MoondreamGeo()
    if spec.name == "tessera":
        # TESSERA's own design philosophy is precomputed embeddings, not
        # running an encoder — same "not a plain nn.Module" reasoning as
        # CLIP/Moondream above, routed to the real wrapper before the
        # generic dispatch below (which has no timm_id/hf_id for this
        # spec and would otherwise raise NotImplementedError).
        from pygeovision.models.foundation.tessera import TesseraGeo
        return TesseraGeo()
    if spec.name == "alphaearth":
        from pygeovision.models.foundation.alphaearth_geo import AlphaEarthGeo
        return AlphaEarthGeo()

    # Try timm first
    if spec.timm_id:
        try:
            import timm
            model = timm.create_model(
                spec.timm_id,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                **kwargs,
            )
            return model
        except ImportError:
            logger.warning("timm not installed — pip install timm")
        except AssertionError as exc:
            if "img_size" in spec.family or spec.family == "swin" or "doesn't match model" in str(exc):
                raise ValueError(
                    f"'{spec.name}' uses a fixed-resolution architecture (window-based "
                    f"attention) and doesn't accept arbitrary chip sizes by default. "
                    f"Pass img_size=<your chip size> explicitly, e.g. "
                    f"get_model('{spec.name}', img_size=512, ...) — it must match "
                    f"whatever chip_size you use for tiled inference. Original error: {exc}"
                ) from exc
            logger.warning("timm build failed for %s: %s", spec.name, exc)
        except Exception as exc:
            logger.warning("timm build failed for %s: %s", spec.name, exc)

    # Try transformers
    if spec.hf_id:
        try:
            return _build_hf_model(spec, num_classes, in_channels, pretrained, **kwargs)
        except ImportError:
            logger.warning("transformers not installed — pip install transformers")
        except Exception as exc:
            logger.warning("HF build failed for %s: %s", spec.name, exc)

    # Generic PyTorch fallback for common architectures
    return _build_pytorch_fallback(spec, num_classes, in_channels, pretrained=pretrained, **kwargs)


def _build_hf_model(spec: ModelSpec, num_classes: int, in_channels: int,
                     pretrained: bool, **kwargs) -> Any:
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(spec.hf_id)
    if hasattr(config, "num_labels"):
        config.num_labels = num_classes
    if pretrained:
        return AutoModel.from_pretrained(spec.hf_id, config=config, ignore_mismatched_sizes=True)
    return AutoModel.from_config(config)


def _build_pytorch_fallback(spec: ModelSpec, num_classes: int, in_channels: int, **kwargs) -> Any:
    """Build common models from torchvision when timm/hf not available."""
    try:
        import torch.nn as nn
        import torchvision.models as tvm
    except ImportError:
        raise ImportError("torch + torchvision required: pip install torch torchvision")

    family = spec.family
    if family == "resnet":
        variant = spec.name.replace("resnet", "").split("-")[0] if "-" in spec.name else spec.name.replace("resnet", "")
        model_fn = {
            "50": tvm.resnet50, "101": tvm.resnet101, "152": tvm.resnet152,
        }.get(variant, tvm.resnet50)
        model = model_fn(pretrained=False)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif family == "rcnn":
        return _build_rcnn(spec, num_classes, in_channels, **kwargs)

    elif family == "anchor_free":
        return _build_anchor_free_detector(spec, num_classes, in_channels, **kwargs)

    elif family == "changeformer":
        from pygeovision.models.change_detection.changeformer import ChangeFormer
        backbone = "mit-b4" if "b4" in spec.name else "mit-b0"
        return ChangeFormer(num_classes=num_classes, in_channels=in_channels, backbone=backbone).build()

    elif family == "bit":
        from pygeovision.models.change_detection.bit import build_bit
        return build_bit(
            num_classes=num_classes, in_channels=in_channels,
            backbone="resnet50" if "r50" in spec.name else "resnet18",
            pretrained=kwargs.pop("pretrained", False), **kwargs,
        )

    elif family == "dsamnet":
        from pygeovision.models.change_detection.dsamnet import build_dsamnet
        return build_dsamnet(
            num_classes=num_classes, in_channels=in_channels,
            backbone="resnet50" if "r50" in spec.name else "resnet18",
            pretrained=kwargs.pop("pretrained", False), **kwargs,
        )

    elif family == "pointnet":
        from pygeovision.models._3d.pointnet import build_pointnet2
        kwargs.pop("pretrained", None)  # no pretrained weights — trained from scratch on your LiDAR data
        return build_pointnet2(
            num_classes=num_classes, in_channels=in_channels,
            msg="msg" in spec.name, **kwargs,
        )

    elif family == "randlanet":
        from pygeovision.models._3d.randlanet import build_randlanet
        kwargs.pop("pretrained", None)  # no pretrained weights — trained from scratch on your LiDAR data
        return build_randlanet(num_classes=num_classes, in_channels=in_channels, **kwargs)

    elif spec.name in ("pointtransformer", "ptv3"):
        from pygeovision.models._3d.pointtransformer import build_ptv3
        kwargs.pop("pretrained", None)  # no pretrained weights — trained from scratch on your LiDAR data
        return build_ptv3(num_classes=num_classes, in_channels=in_channels, **kwargs)

    elif family == "kpconv":
        from pygeovision.models._3d.kpconv import build_kpconv
        kwargs.pop("pretrained", None)  # no pretrained weights — trained from scratch on your LiDAR data
        return build_kpconv(num_classes=num_classes, in_channels=in_channels, **kwargs)

    else:
        raise NotImplementedError(
            f"No real implementation registered for '{spec.name}' (family={family!r}). "
            f"This architecture has no timm_id/hf_id and no dedicated PyTorch factory — "
            f"returning a generic conv classifier would silently produce meaningless "
            f"results, so this raises instead. Either provide your own model and skip "
            f"get_model() for this architecture, or contribute a real factory in "
            f"pygeovision/models/registry.py::_build_pytorch_fallback()."
        )


def _adapt_first_conv_channels(conv: Any, in_channels: int, pretrained_loaded: bool):
    """Replace a Conv2d's input channel count, preserving pretrained RGB
    weights in the first 3 channels (averaged into any extra channels) when
    the original weights were loaded — a silent RGB->multispectral swap
    otherwise discards the pretrained weights entirely without saying so."""
    import torch
    import torch.nn as nn

    if in_channels == conv.in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=(conv.bias is not None),
    )
    if pretrained_loaded:
        with torch.no_grad():
            if in_channels >= 3:
                new_conv.weight[:, :3] = conv.weight[:, :3]
                if in_channels > 3:
                    mean_w = conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
            else:
                new_conv.weight[:] = conv.weight[:, :in_channels]
        logger.warning(
            "Adapted first-conv from %d to %d input channels — pretrained RGB "
            "weights kept in channels 0-2, extra channels initialised from the "
            "channel-mean of the pretrained weights (not from real pretraining "
            "on those bands).",
            conv.in_channels, in_channels,
        )
    return new_conv


def _build_rcnn(spec: ModelSpec, num_classes: int, in_channels: int, pretrained: bool = False, **kwargs) -> Any:
    """Real torchvision Mask R-CNN / Faster R-CNN, with detection heads
    correctly resized to num_classes (torchvision's box/mask predictors are
    tied to a fixed COCO class count and must be swapped, not just relabeled).
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if in_channels != 3 and pretrained:
        logger.warning(
            "%s: pretrained=True with in_channels=%d — ImageNet/COCO backbone "
            "weights cannot map onto non-RGB channels without adaptation. "
            "The first conv layer will be widened (see _adapt_first_conv_channels); "
            "extra channels are NOT pretrained on real multispectral data.",
            spec.name, in_channels,
        )

    is_mask = "mask" in spec.name
    if is_mask:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None,
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            weights_backbone="DEFAULT" if pretrained else None,
        )

    if in_channels != 3:
        backbone_body = model.backbone.body
        old_conv = backbone_body.conv1
        backbone_body.conv1 = _adapt_first_conv_channels(old_conv, in_channels, pretrained)

    # +1 for background class, matching torchvision's convention
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    if is_mask:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes + 1)

    return model


def _build_anchor_free_detector(spec: ModelSpec, num_classes: int, in_channels: int, pretrained: bool = False, **kwargs) -> Any:
    """Real torchvision FCOS. CenterNet has no torchvision equivalent and no
    HF/timm id — raises rather than silently substituting FCOS or a toy CNN,
    since they are architecturally different detectors."""
    if "fcos" not in spec.name:
        raise NotImplementedError(
            f"'{spec.name}' has no real factory (family=anchor_free but not FCOS). "
            f"torchvision has no CenterNet implementation, and this spec has no "
            f"timm_id/hf_id — provide your own model rather than silently getting "
            f"a different architecture."
        )

    import torchvision
    from torchvision.models.detection.fcos import FCOSClassificationHead

    if in_channels != 3 and pretrained:
        logger.warning(
            "%s: pretrained=True with in_channels=%d — see _build_rcnn's warning; "
            "the same caveat applies here.", spec.name, in_channels,
        )

    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
    )

    if in_channels != 3:
        backbone_body = model.backbone.body
        old_conv = backbone_body.conv1
        backbone_body.conv1 = _adapt_first_conv_channels(old_conv, in_channels, pretrained)

    num_anchors = model.head.classification_head.num_anchors
    out_channels = model.backbone.out_channels
    model.head.classification_head = FCOSClassificationHead(
        out_channels, num_anchors, num_classes + 1,
    )
    return model