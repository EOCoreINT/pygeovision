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
    ModelSpec("rf-detr-b",       "detection", "detr",   29.0,   description="RF-DETR Base -- registered for discovery; distributed by Roboflow's `rfdetr` package, not transformers, so get_model() raises NotImplementedError"),
    ModelSpec("rf-detr-l",       "detection", "detr",  128.0,   description="RF-DETR Large -- see rf-detr-b"),
    ModelSpec("rt-detr-l",       "detection", "detr",   42.7,   hf_id="PekingU/rtdetr_r50vd",       description="RT-DETR (ResNet-50-vd backbone)"),
    ModelSpec("detr-r50",        "detection", "detr",   41.3,   hf_id="facebook/detr-resnet-50",    description="DETR ResNet-50"),
    ModelSpec("detr-r101",       "detection", "detr",   60.0,   hf_id="facebook/detr-resnet-101",   description="DETR ResNet-101"),
    ModelSpec("faster-rcnn-r50", "detection", "rcnn",   41.8,   description="Faster R-CNN ResNet-50"),
    ModelSpec("mask-rcnn-r50",   "detection", "rcnn",   44.4,   description="Mask R-CNN ResNet-50"),
    ModelSpec("fcos-r50",        "detection", "anchor_free", 32.0, description="FCOS ResNet-50"),
    ModelSpec("centernet-r50",   "detection", "anchor_free", 32.0, description="CenterNet, ResNet-50 backbone (hand-written: torchvision has none)"),
    ModelSpec("grounding-dino",  "detection", "vlm",   172.0,   hf_id="IDEA-Research/grounding-dino-tiny", description="Grounding DINO"),

    # ── Segmentation (17) ────────────────────────────────────────────────────
    ModelSpec("segformer-b0",    "segmentation", "segformer",  3.8,  hf_id="nvidia/segformer-b0-finetuned-ade-512-512", description="SegFormer B0"),
    ModelSpec("segformer-b2",    "segmentation", "segformer", 27.5,  hf_id="nvidia/segformer-b2-finetuned-ade-512-512", description="SegFormer B2"),
    ModelSpec("segformer-b5",    "segmentation", "segformer", 84.7,  hf_id="nvidia/segformer-b5-finetuned-ade-512-512", description="SegFormer B5"),
    ModelSpec("mask2former-swin-t","segmentation","mask2former",47.0, hf_id="facebook/mask2former-swin-tiny-ade-semantic", description="Mask2Former Swin-T"),
    ModelSpec("mask2former-swin-b","segmentation","mask2former",102.0,hf_id="facebook/mask2former-swin-base-ade-semantic", description="Mask2Former Swin-B"),
    ModelSpec("unet-r50",         "segmentation","unet_variant", 32.5, description="U-Net, ResNet-50 encoder (segmentation-models-pytorch)"),
    ModelSpec("unet-r101",        "segmentation","unet_variant", 51.5, description="U-Net, ResNet-101 encoder (segmentation-models-pytorch)"),
    ModelSpec("unet-efficientb4", "segmentation","unet_variant", 20.2, description="U-Net, EfficientNet-B4 encoder (segmentation-models-pytorch)"),
    ModelSpec("deeplab-r50",      "segmentation","deeplab",     39.6, description="DeepLabV3, ResNet-50 encoder (segmentation-models-pytorch)"),
    ModelSpec("deeplab-r101",     "segmentation","deeplab",     58.6, description="DeepLabV3, ResNet-101 encoder (segmentation-models-pytorch)"),
    ModelSpec("pspnet-r50",       "segmentation","pspnet",      24.3, description="PSPNet, ResNet-50 encoder (segmentation-models-pytorch)"),
    ModelSpec("fcn-r50",          "segmentation","fcn",         32.9, description="FCN, ResNet-50 (torchvision)"),
    ModelSpec("upernet-swin-b",   "segmentation","upernet",    120.0, hf_id="openmmlab/upernet-swin-base", description="UperNet with a Swin-B backbone (transformers)"),
    ModelSpec("dinov3_segmentor", "segmentation","dinov3_head", 0.0, description="Segmentation head on a DINOv3 backbone -- registered for discovery only; get_model() raises NotImplementedError (use pygeovision.models.foundation.dinov3 for features)"),
    ModelSpec("sam-vit-h",       "segmentation", "sam",      636.0,  hf_id="facebook/sam-vit-huge",   description="SAM ViT-H"),
    ModelSpec("sam-vit-l",       "segmentation", "sam",      308.0,  hf_id="facebook/sam-vit-large",  description="SAM ViT-L"),
    ModelSpec("sam-vit-b",       "segmentation", "sam",       93.7,  hf_id="facebook/sam-vit-base",   description="SAM ViT-B"),
    ModelSpec("sam2-hiera-l",    "segmentation", "sam2",     224.4,  hf_id="facebook/sam2-hiera-large-hf",description="SAM2 Hiera-L"),

    # ── Change Detection (10) ────────────────────────────────────────────────
    ModelSpec("changeformer-mit-b0","change_detection","changeformer", 13.9, description="ChangeFormer MiT-B0"),
    ModelSpec("changestar-r18",   "change_detection","changestar",   26.0, description="ChangeStar ResNet-18 -- no implementation available; get_model() raises NotImplementedError (use changeformer, bit-r50 or dsamnet)"),
    ModelSpec("changeformer-mit-b4","change_detection","changeformer", 67.4, description="ChangeFormer MiT-B4"),
    ModelSpec("bit-r50",         "change_detection","bit",         26.1, description="BIT ResNet-50"),
    ModelSpec("dsamnet",         "change_detection","dsamnet",     16.0, description="DSAMNet"),

    # ── Foundation Models (11) ───────────────────────────────────────────────
    ModelSpec("prithvi-100m",    "foundation","prithvi",  100.0, hf_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M",   description="NASA/IBM Prithvi-EO-1.0 100M (multitemporal)", pretrained_on="HLS"),
    ModelSpec("prithvi-300m",    "foundation","prithvi",  300.0, hf_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",   description="NASA/IBM Prithvi-EO-2.0 300M", pretrained_on="HLS"),
    ModelSpec("prithvi-600m",    "foundation","prithvi",  600.0, hf_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M",   description="NASA/IBM Prithvi-EO-2.0 600M", pretrained_on="HLS"),
    ModelSpec("dofa-base",       "foundation","dofa",      86.0, hf_id="torchgeo/dofa",          description="Dynamic One-For-All (multi-sensor) -- real, dedicated integration via torchgeo (github.com/microsoft/torchgeo), verified end-to-end this session (same model instance correctly embeds both 9-band Sentinel-2 and 3-band NAIP input); real pretrained weights confirmed at hf.co/torchgeo/dofa", pretrained_on="Sentinel-1/2,NAIP,EnMAP,Gaofen", paper="https://arxiv.org/abs/2403.15356"),
    ModelSpec("dofa-large",      "foundation","dofa",      304.0, hf_id="torchgeo/dofa",          description="DOFA large (ViT-L/16) -- same real torchgeo integration as dofa-base, real pretrained weights confirmed via torchgeo's DOFALarge16_Weights enum", pretrained_on="Sentinel-1/2,NAIP,EnMAP,Gaofen", paper="https://arxiv.org/abs/2403.15356"),
    ModelSpec("satlas-pretrain",  "foundation","satlas",    100.0, description="SatlasPretrain -- no verified checkpoint or construction code here; get_model() raises NotImplementedError"),
    ModelSpec("tessera",         "foundation","tessera",   0.0, description="TESSERA precomputed embeddings (Sentinel-1+2, 128ch/10m) — real, precomputed via the geotessera library, not a locally-run encoder", pretrained_on="Sentinel-1+2"),
    ModelSpec("alphaearth",      "foundation","alphaearth", 0.0, description="AlphaEarth Foundations / Satellite Embedding (64ch/10m, annual 2017+) — real, precomputed via Google Earth Engine, not a locally-run encoder", pretrained_on="Multi-sensor (Sentinel-1/2, Landsat, etc.)"),
    ModelSpec("remoteclip-b32",  "foundation","clip",     151.0, hf_id="chendelong/RemoteCLIP", description="RemoteCLIP ViT-B/32 (open_clip format; needs open_clip_torch)", pretrained_on="RemoteCLIP corpus (RS5M is GeoRSCLIP)"),
    ModelSpec("remoteclip-l14",  "foundation","clip",     428.0, hf_id="chendelong/RemoteCLIP", description="RemoteCLIP ViT-L/14 (open_clip format; needs open_clip_torch)", pretrained_on="RemoteCLIP corpus (RS5M is GeoRSCLIP)"),
    ModelSpec("chatearthnet",     "vlm","chatearthnet",    0.0, description="ChatEarthNet -- no verified checkpoint or construction code here; get_model() raises NotImplementedError"),

    # ── Vision-Language (7) ──────────────────────────────────────────────────
    ModelSpec("moondream2",      "vlm","moondream",   1800.0, hf_id="vikhyatk/moondream2", description="Moondream2 (satellite VQA)"),
    ModelSpec("openclip-b32",    "vlm","clip",          151.0, hf_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", description="OpenCLIP ViT-B/32"),
    ModelSpec("openclip-l14",    "vlm","clip",          428.0, hf_id="openai/clip-vit-large-patch14",          description="OpenCLIP ViT-L/14"),
    ModelSpec("lisat-7b",        "vlm","lisat",         7000.0, hf_id="jquenum/LISAt-7b", description="LISAt-7B -- geospatial reasoning segmentation VLM (Quenum et al. 2025, arXiv:2505.02829); real, published SOTA model, but no verified build path here -- see family=='lisat' dispatch", paper="https://arxiv.org/abs/2505.02829"),

    # ── 3D / LiDAR (6) ──────────────────────────────────────────────────────
    ModelSpec("pointnet2-ssg",   "3d","pointnet",    1.5,  description="PointNet++ SSG"),
    ModelSpec("pointnet2-msg",   "3d","pointnet",    1.7,  description="PointNet++ MSG"),
    ModelSpec("randlanet",       "3d","randlanet",   1.2,  description="RandLA-Net"),
    ModelSpec("kpconv",          "3d","kpconv",      14.8, description="KPConv"),
    ModelSpec("pointtransformer","3d","transformer",  7.8, description="Point Transformer"),
    ModelSpec("ptv3",            "3d","transformer", 46.2, description="Point Transformer V3"),

  
    # ── Super-Resolution (4) ─────────────────────────────────────────────────
    ModelSpec("swinir-m",        "super_resolution","swin", 12.1, hf_id="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",  description="Swin2SR real-world x4 super-resolution (transformers)"),

    
    # ── Prithvi ─────────────────────────────────────────────────────────────────
    ModelSpec("prithvi_eo_1_0",    "foundation","prithvi",  100.0,  hf_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M",          description="Prithvi-EO-1.0 100M — HLS US 30m",    pretrained_on="HLS-US",    supports_multispectral=True),
    ModelSpec("prithvi_eo_2_0",    "foundation","prithvi",  300.0,  hf_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",   description="Prithvi-EO-2.0 300M — HLS Global 30m", pretrained_on="HLS-Global",supports_multispectral=True),
    ModelSpec("prithvi_eo_2_0_600m", "foundation","prithvi", 600.0, hf_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M", description="Prithvi-EO-2.0 600M — HLS Global 30m", pretrained_on="HLS-Global", supports_multispectral=True),
    ModelSpec("prithvi_burn_scar", "segmentation","prithvi", 100.0, hf_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar", description="Prithvi fine-tuned burn scar -- registered for discovery; get_model() raises NotImplementedError (TerraTorch/MMSegmentation artefact)", pretrained_on="HLS-US"),

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

# ── Construction ───────────────────────────────────────────────────────────────

def get_model(name: str, num_classes: int = 2, in_channels: int = 4,
               pretrained: bool = True, device: str | None = None,
               **kwargs) -> Any:
    """Build a registered model.

    Args:
        name: Registry name, e.g. ``"segformer-b2"``, ``"unet-r50"``, ``"faster-rcnn-r50"``.
        num_classes: Number of output classes (object classes for detectors; background is
            added where the architecture needs it).
        in_channels: Input bands.  Backbones are widened for ``in_channels != 3`` and the
            extra bands start from the mean RGB filter -- they are *not* pretrained.
        pretrained: Load pretrained weights.  If they cannot be loaded this raises
            :class:`~pygeovision.core.exceptions.ModelLoadError`; it never quietly returns
            random weights.
        device: Move the model to this device.
        **kwargs: ``allow_random_init=True`` returns an architecture-only model when
            pretrained weights cannot be loaded (flagged ``_pygeovision_is_random_init``);
            ``img_size`` for window/position-embedding based models; builder-specific
            options such as ``min_size`` / ``image_mean`` for the torchvision detectors.

    Returns:
        A ``torch.nn.Module`` (or the dedicated wrapper for CLIP / SAM / Moondream /
        embedding-only models).  Segmentation models return ``(B, C, H, W)`` logits at the
        input resolution; classifiers return ``(B, C)`` logits; torchvision detectors follow
        torchvision's list-in / list-of-dicts-out API.

    Raises:
        KeyError: unknown name.
        NotImplementedError: the architecture is registered but this package has no
            verified build path for it (see its registry description).
        ModelLoadError: pretrained weights, or a Hub-only config, could not be fetched.

    Example::

        model = get_model("segformer-b2", num_classes=7, in_channels=4, pretrained=False)
    """
    spec = model_registry[name]
    model = _build_model(spec, num_classes=num_classes, in_channels=in_channels,
                          pretrained=pretrained, **kwargs)
    if device:
        model = model.to(device)
    return model


# Registry names whose backbone is spelled in the name.
_ENCODER_BY_NAME = {
    "unet-r50": "resnet50", "unet-r101": "resnet101", "unet-efficientb4": "efficientnet-b4",
    "deeplab-r50": "resnet50", "deeplab-r101": "resnet101",
    "pspnet-r50": "resnet50", "fcn-r50": "resnet50",
}

_PRITHVI_KEY = {
    "prithvi-100m": "prithvi_eo_1_0", "prithvi_eo_1_0": "prithvi_eo_1_0",
    "prithvi-300m": "prithvi_eo_2_0", "prithvi_eo_2_0": "prithvi_eo_2_0",
    "prithvi-600m": "prithvi_eo_2_0_600m", "prithvi_eo_2_0_600m": "prithvi_eo_2_0_600m",
    "prithvi_burn_scar": "prithvi_eo_1_0_finetuned_burn",
}

_REFUSALS = {
    "satlas": "SatlasPretrain has no verified public checkpoint or construction code in this codebase.",
    "chatearthnet": "ChatEarthNet has no verified public checkpoint or construction code in this codebase.",
    "changestar": "ChangeStar has no implementation in torchvision/timm/transformers and none in this "
                  "codebase (use ChangeFormer, BIT or DSAMNet).",
    "dinov3_head": "A DINOv3 segmentation head needs a DINOv3 backbone; use "
                   "pygeovision.models.foundation.dinov3 for features.",
}


def _build_model(spec: ModelSpec, num_classes: int, in_channels: int,
                  pretrained: bool, **kwargs) -> Any:
    """Dispatch a registry entry to its builder."""
    fam = spec.family
    allow_random_init = kwargs.pop("allow_random_init", False)

    # -- not "tensor in, logits out": dedicated wrappers ------------------------------
    if fam == "clip":
        from pygeovision.models.vlm.clip import CLIPGeo
        return CLIPGeo(model=spec.name)
    if fam == "moondream":
        from pygeovision.models.vlm.moondream import MoondreamGeo
        return MoondreamGeo()
    if fam in ("sam", "sam2"):
        from pygeovision.models.segmentation.sam import GeoSAM
        return GeoSAM(hf_id=spec.hf_id, device=kwargs.get("device", "cpu"))
    if fam == "dofa":
        from pygeovision.models.foundation.dofa import load_dofa_hf
        real_name = {
            "dofa-base": "dofa_base_patch16_224",
            "dofa-large": "dofa_large_patch16_224",
        }.get(spec.name, "dofa_base_patch16_224")
        return load_dofa_hf(real_name, device=kwargs.get("device", "cpu"),
                             allow_random_init=allow_random_init)
    if spec.name == "tessera":
        from pygeovision.models.foundation.tessera import TesseraGeo
        return TesseraGeo()
    if spec.name == "alphaearth":
        from pygeovision.models.foundation.alphaearth_geo import AlphaEarthGeo
        return AlphaEarthGeo()
    if fam == "prithvi":
        from pygeovision.models.foundation.prithvi import load_prithvi_hf
        if not pretrained and not allow_random_init:
            raise ValueError(
                f"'{spec.name}': there is no architecture-only Prithvi build. Use pretrained=True, "
                f"or allow_random_init=True for a flagged, randomly initialised stand-in."
            )
        return load_prithvi_hf(_PRITHVI_KEY.get(spec.name, "prithvi_eo_2_0"),
                               device=kwargs.get("device", "cpu"),
                               allow_random_init=allow_random_init)

    # -- registered, but with no verified build path ---------------------------------
    if fam in _REFUSALS:
        raise NotImplementedError(f"'{spec.name}': {_REFUSALS[fam]} Provide your own model "
                                  f"rather than silently getting a different architecture.")
    if fam == "lisat":
        raise NotImplementedError(
            f"'{spec.name}' is a real, published model (Quenum et al. 2025, arXiv:2505.02829) "
            f"with weights at {spec.hf_id!r}, but its checkpoint is a custom LISA-style multimodal "
            f"LLM + SAM decoder, not a standard transformers class. Loading it needs the authors' "
            f"lisat_code repository, flash-attn and a separate RemoteCLIP checkpoint; no verified "
            f"build path exists here. See https://huggingface.co/jquenum/LISAt-7b."
        )

    # -- torchvision / smp / timm / hand-written families ----------------------------
    if fam == "rcnn":
        from pygeovision.models.detection.fasterrcnn import build_faster_rcnn
        from pygeovision.models.detection.maskrcnn import build_mask_rcnn
        build = build_mask_rcnn if "mask" in spec.name else build_faster_rcnn
        return build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, **kwargs)
    if fam == "anchor_free":
        if "fcos" in spec.name:
            from pygeovision.models.detection.fcos import build_fcos
            return build_fcos(num_classes=num_classes, in_channels=in_channels,
                              pretrained=pretrained, **kwargs)
        from pygeovision.models.detection.centernet import build_centernet
        return build_centernet(num_classes=num_classes, in_channels=in_channels,
                               pretrained=pretrained, **kwargs)
    if fam == "unet_variant":
        from pygeovision.models.segmentation.unet import build_unet
        return build_unet(_ENCODER_BY_NAME[spec.name], num_classes, in_channels, pretrained, **kwargs)
    if fam == "deeplab":
        from pygeovision.models.segmentation.deeplab import build_deeplab
        return build_deeplab(_ENCODER_BY_NAME[spec.name], num_classes, in_channels, pretrained,
                             plus=False, **kwargs)
    if fam == "pspnet":
        from pygeovision.models.segmentation.pspnet import build_pspnet
        return build_pspnet(_ENCODER_BY_NAME[spec.name], num_classes, in_channels, pretrained, **kwargs)
    if fam == "fcn":
        from pygeovision.models.segmentation.fcn import build_fcn
        return build_fcn(_ENCODER_BY_NAME[spec.name], num_classes, in_channels, pretrained, **kwargs)
    if fam == "changeformer":
        from pygeovision.models.change_detection.changeformer import ChangeFormer
        backbone = "mit-b4" if "b4" in spec.name else "mit-b0"
        return ChangeFormer(num_classes=num_classes, in_channels=in_channels, backbone=backbone).build()
    if fam == "bit":
        from pygeovision.models.change_detection.bit import build_bit
        return build_bit(num_classes=num_classes, in_channels=in_channels,
                         backbone="resnet50" if "r50" in spec.name else "resnet18",
                         pretrained=pretrained, **kwargs)
    if fam == "dsamnet":
        from pygeovision.models.change_detection.dsamnet import build_dsamnet
        return build_dsamnet(num_classes=num_classes, in_channels=in_channels,
                             backbone="resnet50" if "r50" in spec.name else "resnet18",
                             pretrained=pretrained, **kwargs)
    if fam in ("pointnet", "randlanet", "kpconv") or spec.name in ("pointtransformer", "ptv3"):
        return _build_3d(spec, num_classes, in_channels, **kwargs)

    # -- generic: timm, then Hugging Face ---------------------------------------------
    if spec.timm_id and spec.task != "super_resolution":
        from pygeovision.models.classification._timm import build_timm_classifier
        try:
            return build_timm_classifier(spec.timm_id, num_classes, in_channels, pretrained,
                                         allow_random_init=allow_random_init, **kwargs)
        except ImportError:
            if fam != "resnet":
                raise
            return _build_torchvision_resnet(spec, num_classes, in_channels, pretrained)
        except AssertionError as exc:  # window-attention models are tied to the chip size
            raise ValueError(
                f"'{spec.name}' uses fixed-resolution window attention and does not accept "
                f"arbitrary chip sizes. Pass img_size=<your chip size> explicitly, e.g. "
                f"get_model('{spec.name}', img_size=512, ...). Original error: {exc}"
            ) from exc
    if spec.hf_id or fam == "upernet" or spec.task == "super_resolution":
        from pygeovision.models._hf import build_hf_model
        return build_hf_model(spec, num_classes, in_channels, pretrained,
                              allow_random_init=allow_random_init, **kwargs)

    raise NotImplementedError(
        f"No build path is registered for '{spec.name}' (family={fam!r}). It has no timm_id or "
        f"hf_id and no dedicated builder; returning a generic network would give meaningless "
        f"results, so this raises. Provide your own model."
    )


def _build_3d(spec: ModelSpec, num_classes: int, in_channels: int, **kwargs) -> Any:
    kwargs.pop("device", None)
    if spec.family == "pointnet":
        from pygeovision.models._3d.pointnet import build_pointnet2
        return build_pointnet2(num_classes=num_classes, in_channels=in_channels,
                               msg="msg" in spec.name, **kwargs)
    if spec.family == "randlanet":
        from pygeovision.models._3d.randlanet import build_randlanet
        return build_randlanet(num_classes=num_classes, in_channels=in_channels, **kwargs)
    if spec.family == "kpconv":
        from pygeovision.models._3d.kpconv import build_kpconv
        return build_kpconv(num_classes=num_classes, in_channels=in_channels, **kwargs)
    from pygeovision.models._3d.pointtransformer import build_ptv3
    return build_ptv3(num_classes=num_classes, in_channels=in_channels, **kwargs)


def _build_torchvision_resnet(spec: ModelSpec, num_classes: int, in_channels: int,
                               pretrained: bool) -> Any:
    """ResNet via torchvision, used only when timm is not installed."""
    import torch.nn as nn
    import torchvision.models as tvm

    from pygeovision.models import _adapt

    fn = {"resnet50": tvm.resnet50, "resnet101": tvm.resnet101,
          "resnet152": tvm.resnet152}.get(spec.name, tvm.resnet50)
    model = fn(weights="DEFAULT" if pretrained else None)
    if in_channels != 3:
        model.conv1 = _adapt.adapt_first_conv(model.conv1, in_channels, pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
