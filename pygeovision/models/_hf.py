"""Hugging Face model construction for the general model registry.

What this fixes compared with a bare ``AutoModel.from_pretrained``:

* ``AutoModel`` returns the headless encoder -- no classification or segmentation
  head -- so ``num_classes`` was silently ignored.  The task-specific class is used.
* Building with ``pretrained=False`` still downloaded the *config* from the Hub,
  so it failed offline.  For the families in :data:`LOCAL_CONFIGS` the config is
  constructed in code (parameter counts checked against the published models by
  the test-suite), so an architecture can be built, trained from scratch and
  unit-tested with no network.
* Input bands other than three are handled (:mod:`pygeovision.models._adapt`).
* Segmentation/classification models are wrapped so ``model(x)`` returns a plain
  logits tensor at the input resolution (:mod:`pygeovision.models._wrappers`).
* A failed pretrained download raises :class:`~pygeovision.core.exceptions.ModelLoadError`
  instead of quietly returning random weights.
"""
from __future__ import annotations

import logging
from typing import Any

from pygeovision.core.exceptions import ModelLoadError
from pygeovision.models import _adapt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Which transformers class implements which registry family / name
# ---------------------------------------------------------------------------

_SEGFORMER = {  # depths, hidden sizes, decoder width -- the published MiT-b0..b5
    "b0": ([2, 2, 2, 2], [32, 64, 160, 256], 256),
    "b1": ([2, 2, 2, 2], [64, 128, 320, 512], 256),
    "b2": ([3, 4, 6, 3], [64, 128, 320, 512], 768),
    "b3": ([3, 4, 18, 3], [64, 128, 320, 512], 768),
    "b4": ([3, 8, 27, 3], [64, 128, 320, 512], 768),
    "b5": ([3, 6, 40, 3], [64, 128, 320, 512], 768),
}


def _model_class_name(spec: Any) -> tuple[str, str | None]:
    """Return ``(transformers class name, adapter kind)`` for a registry spec."""
    fam, name = spec.family, spec.name
    if fam == "vit":
        return "ViTForImageClassification", "classification"
    if fam == "dinov2":
        return "Dinov2ForImageClassification", "classification"
    if fam == "segformer":
        return "SegformerForSemanticSegmentation", "segmentation"
    if fam == "upernet":
        return "UperNetForSemanticSegmentation", "segmentation"
    if fam == "mask2former":
        return "Mask2FormerForUniversalSegmentation", None
    if fam == "detr" and name.startswith("rt-detr"):
        return "RTDetrForObjectDetection", None
    if fam == "detr" and name.startswith("detr-"):
        return "DetrForObjectDetection", None
    if spec.task == "super_resolution":
        return "Swin2SRForImageSuperResolution", None
    if fam == "vlm" and "grounding" in name:
        return "GroundingDinoForObjectDetection", None
    raise ModelLoadError(
        f"'{name}' (family={fam!r}, task={spec.task!r}) has no Hugging Face "
        f"model class mapped in pygeovision.models._hf."
    )


def local_config(spec: Any, num_classes: int, img_size: int | None = None,
                 in_channels: int = 3) -> Any | None:
    """Build the published architecture's config in code (no network), or ``None``.

    ``in_channels`` only changes the config for super-resolution, where the *output* has the
    same band count as the input and so cannot be fixed up afterwards by patching one conv.
    """
    import transformers as T

    name, fam = spec.name, spec.family
    if fam == "vit":
        if name == "vit-l16":
            cfg = T.ViTConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                              intermediate_size=4096)
        else:
            cfg = T.ViTConfig()
        if img_size:
            cfg.image_size = img_size
    elif fam == "dinov2":
        sizes = {
            "dinov2-s": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6),
            "dinov2-b": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
            "dinov2-l": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16),
            "dinov2-g": dict(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24,
                             use_swiglu_ffn=True),
        }
        if name not in sizes:
            return None
        cfg = T.Dinov2Config(image_size=img_size or 518, **sizes[name])
    elif fam == "segformer":
        variant = name.rsplit("-", 1)[-1]
        if variant not in _SEGFORMER:
            return None
        depths, hidden, dec = _SEGFORMER[variant]
        cfg = T.SegformerConfig(depths=depths, hidden_sizes=hidden, decoder_hidden_size=dec)
    elif fam == "upernet" and name == "upernet-swin-b":
        backbone = T.SwinConfig(
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
        cfg = T.UperNetConfig(backbone_config=backbone, use_auxiliary_head=False)
    elif fam == "detr" and name in ("detr-r50", "detr-r101"):
        cfg = T.DetrConfig(
            use_pretrained_backbone=False,
            backbone="resnet101" if name == "detr-r101" else "resnet50",
        )
    elif fam == "detr" and name == "rt-detr-l":
        cfg = T.RTDetrConfig()
    elif spec.task == "super_resolution" and name == "swinir-m":
        cfg = T.Swin2SRConfig(upscale=4, embed_dim=180, depths=[6] * 6, num_heads=[6] * 6,
                              window_size=8, upsampler="nearest+conv", num_channels=in_channels)
    else:
        return None
    _set_labels(cfg, num_classes, spec)
    return cfg


def _set_labels(cfg: Any, num_classes: int, spec: Any) -> None:
    if spec.task in ("classification", "segmentation", "detection"):
        # HF DETR adds its own "no object" slot, so num_labels is the number of real classes.
        cfg.num_labels = num_classes
        cfg.id2label = {i: f"class_{i}" for i in range(num_classes)}
        cfg.label2id = {v: k for k, v in cfg.id2label.items()}


def build_hf_model(
    spec: Any,
    num_classes: int,
    in_channels: int,
    pretrained: bool,
    allow_random_init: bool = False,
    img_size: int | None = None,
) -> Any:
    """Build a registry entry from Hugging Face ``transformers``.

    Args:
        pretrained: Load the Hub checkpoint.  A failure raises
            :class:`ModelLoadError` unless ``allow_random_init`` is true, in which
            case an architecture-only model is returned, flagged with
            ``_pygeovision_is_random_init = True``.
        allow_random_init: Opt in to the random-weights fallback described above.
        img_size: Override the config image size (ViT / DINOv2).  Changing it
            discards a pretrained position embedding, which is warned about.
    """
    try:
        import transformers
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover - transformers is a declared dependency
        raise ImportError("This model needs transformers: pip install transformers") from exc

    cls_name, adapter = _model_class_name(spec)
    if spec.task == "super_resolution" and in_channels != 3 and pretrained:
        raise ValueError(
            f"'{spec.name}': the pretrained super-resolution weights are RGB-only (input and output "
            f"are both 3-band). Use pretrained=False to train a {in_channels}-band model from "
            f"scratch, or super-resolve three bands at a time."
        )
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        raise ModelLoadError(
            f"transformers {transformers.__version__} has no class {cls_name}; "
            f"upgrade transformers to build '{spec.name}'."
        )

    def _hub_config() -> Any:
        cfg = AutoConfig.from_pretrained(spec.hf_id)
        _set_labels(cfg, num_classes, spec)
        if spec.family == "detr" and hasattr(cfg, "use_pretrained_backbone"):
            cfg.use_pretrained_backbone = False  # the full checkpoint already holds the backbone
        if img_size and hasattr(cfg, "image_size"):
            cfg.image_size = img_size
        return cfg

    model = None
    loaded_pretrained = False
    failure: Exception | None = None

    if pretrained:
        if not spec.hf_id:
            failure = ModelLoadError(f"'{spec.name}' has no Hugging Face checkpoint to load")
        else:
            try:
                cfg = _hub_config()
                if img_size and spec.family in ("vit", "dinov2"):
                    logger.warning(
                        "img_size=%d differs from the pretrained image size: the position "
                        "embedding is re-initialised and must be fine-tuned.", img_size)
                model = cls.from_pretrained(spec.hf_id, config=cfg, ignore_mismatched_sizes=True)
                loaded_pretrained = True
            except Exception as exc:  # network, auth, missing repo, version mismatch ...
                failure = exc
        if model is None:
            if not allow_random_init:
                raise ModelLoadError(
                    f"Could not load pretrained weights for '{spec.name}' "
                    f"({spec.hf_id}): {failure}",
                    details="Nothing was substituted. Pass pretrained=False for a randomly "
                            "initialised model, or allow_random_init=True to fall back to one "
                            "automatically (it is flagged _pygeovision_is_random_init).",
                ) from failure

    if model is None:
        cfg = local_config(spec, num_classes, img_size, in_channels)
        if cfg is None:
            try:
                cfg = _hub_config()
            except Exception as exc:
                raise ModelLoadError(
                    f"'{spec.name}' cannot be built offline: its architecture config is only "
                    f"available from the Hugging Face Hub ({spec.hf_id}) and the Hub could "
                    f"not be reached: {exc}",
                    details="Connect to the Hub (or set HF_HOME to a populated cache) and retry.",
                ) from exc
        model = cls(cfg)
        if pretrained:  # allow_random_init fallback
            _adapt.mark_random_init(model, f"pretrained load failed: {failure}")

    if in_channels != 3 and spec.task != "super_resolution":
        _adapt.adapt_hf_input_channels(model, in_channels, pretrained_loaded=loaded_pretrained)

    from pygeovision.models import _wrappers
    if adapter == "segmentation":
        wrapped = _wrappers.HFSegmentationAdapter(model)
    elif adapter == "classification":
        wrapped = _wrappers.HFClassificationAdapter(model)
    else:
        return model
    if getattr(model, "_pygeovision_is_random_init", False):
        _adapt.mark_random_init(wrapped, model._pygeovision_random_init_reason)
    return wrapped
