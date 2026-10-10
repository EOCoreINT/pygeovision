"""
DINOv3 foundation backbones (Meta AI) for PyGeoVision, loaded through Hugging Face ``transformers``.

All twelve released backbones are wired to their real repositories (``facebook/dinov3-*``):

======================  =========================================  ==========
registry name           repository                                 pre-training
======================  =========================================  ==========
dinov3_vits16           dinov3-vits16-pretrain-lvd1689m            LVD-1689M
dinov3_vits16plus       dinov3-vits16plus-pretrain-lvd1689m        LVD-1689M
dinov3_vitb16           dinov3-vitb16-pretrain-lvd1689m            LVD-1689M
dinov3_vitl16           dinov3-vitl16-pretrain-lvd1689m            LVD-1689M
dinov3_vith16plus       dinov3-vith16plus-pretrain-lvd1689m        LVD-1689M
dinov3_vit7b16          dinov3-vit7b16-pretrain-lvd1689m           LVD-1689M
dinov3_vitl16_sat       dinov3-vitl16-pretrain-sat493m             SAT-493M
dinov3_vit7b16_sat      dinov3-vit7b16-pretrain-sat493m            SAT-493M
dinov3_convnext_*       dinov3-convnext-{tiny,small,base,large}    LVD-1689M
======================  =========================================  ==========

(The repository ids were checked against the Hugging Face model index.)

Access
------
Every DINOv3 repository is **gated** (manual approval by Meta). Accept the licence on the model page and
authenticate (``huggingface-cli login`` or ``HF_TOKEN``) before loading; otherwise loading raises
:class:`~pygeovision.core.exceptions.ModelLoadError` that says so. Nothing is ever substituted: an earlier
version of this module loaded DINOv2 weights under DINOv3 names, which is why that code is gone.

Normalisation
-------------
Input statistics are read from each repository's own preprocessor (SAT-493M models use satellite
statistics, web models use ImageNet's). :data:`WEB_MEAN` / :data:`SAT_MEAN` are the documented constants,
used only if a preprocessor cannot be read.

What is *not* here
------------------
* ``dino.txt`` zero-shot text alignment: no checkpoint exists in the public model index, so ``DINOv3Text``
  raises ``NotImplementedError`` instead of comparing DINOv3 features with CLIP text embeddings (different
  spaces; the similarity would be meaningless). Use :class:`~pygeovision.models.vlm.clip.CLIPGeo`.
* torch.hub loading and the old "local .pth into a hand-built skeleton" path (``strict=False`` hid the fact
  that no weight matched).
* Allometric biomass from canopy height (the coefficients had no source).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


# ── Registry ──────────────────────────────────────────────────────────────────

def _vit(arch: str, params_m: int, embed: int, repo: str, *, sat: bool = False) -> dict:
    return {"arch": arch, "family": "vit", "params_m": params_m, "patch": 16, "embed": embed,
            "dataset": "SAT-493M" if sat else "LVD-1689M", "hf_id": f"facebook/{repo}",
            "gated": True, **({"sat": True} if sat else {})}


def _cnx(params_m: int, embed: int, size: str) -> dict:
    return {"arch": "convnext", "family": "convnext", "params_m": params_m, "patch": 32, "embed": embed,
            "dataset": "LVD-1689M", "hf_id": f"facebook/dinov3-convnext-{size}-pretrain-lvd1689m",
            "gated": True}


DINOV3_MODELS: dict[str, dict] = {
    "dinov3_vits16":      _vit("vit_small", 21, 384, "dinov3-vits16-pretrain-lvd1689m"),
    "dinov3_vits16plus":  _vit("vit_small", 29, 384, "dinov3-vits16plus-pretrain-lvd1689m"),
    "dinov3_vitb16":      _vit("vit_base", 86, 768, "dinov3-vitb16-pretrain-lvd1689m"),
    "dinov3_vitl16":      _vit("vit_large", 300, 1024, "dinov3-vitl16-pretrain-lvd1689m"),
    "dinov3_vith16plus":  _vit("vit_huge", 840, 1280, "dinov3-vith16plus-pretrain-lvd1689m"),
    "dinov3_vit7b16":     _vit("vit_7b", 6700, 4096, "dinov3-vit7b16-pretrain-lvd1689m"),
    "dinov3_vitl16_sat":  _vit("vit_large", 300, 1024, "dinov3-vitl16-pretrain-sat493m", sat=True),
    "dinov3_vit7b16_sat": _vit("vit_7b", 6700, 4096, "dinov3-vit7b16-pretrain-sat493m", sat=True),
    "dinov3_convnext_tiny":  _cnx(29, 768, "tiny"),
    "dinov3_convnext_small": _cnx(50, 768, "small"),
    "dinov3_convnext_base":  _cnx(89, 1024, "base"),
    "dinov3_convnext_large": _cnx(196, 1536, "large"),
}

# Task heads that exist as checkpoints in this package's loaders. (Meta's classifier / detector / segmentor
# heads ship only with the original torch.hub code and are not wired here.)
CHMV2_HF_ID = "facebook/dinov3-vitl16-chmv2-dpt-head"
DINOV3_HEADS: dict[str, dict] = {
    "chmv2": {"dataset": "global_chm", "task": "canopy_height", "hf_id": CHMV2_HF_ID,
              "backbone": "dinov3_vitl16_sat"},
}


# ── Transforms ────────────────────────────────────────────────────────────────

WEB_MEAN = (0.485, 0.456, 0.406)    # ImageNet statistics (LVD-1689M models)
WEB_STD = (0.229, 0.224, 0.225)

SAT_MEAN = (0.430, 0.411, 0.296)    # SAT-493M statistics. Different from web!
SAT_STD = (0.213, 0.156, 0.143)


def _transform(mean: tuple, std: tuple, resize_size: int, crop_size: int) -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install torchvision") from exc
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def dinov3_web_transform(resize_size: int = 256, crop_size: int = 224) -> Any:
    """ImageNet-statistics transform for LVD-1689M models. (Centre-crops: use for classification-style
    inputs, not for dense maps that must stay registered with the raster.)"""
    return _transform(WEB_MEAN, WEB_STD, resize_size, crop_size)


def dinov3_sat_transform(resize_size: int = 256, crop_size: int = 224) -> Any:
    """Satellite-statistics transform for SAT-493M models (not ImageNet's)."""
    return _transform(SAT_MEAN, SAT_STD, resize_size, crop_size)


def get_transform(model_name: str, **kwargs) -> Any:
    """Web or satellite transform, chosen from the model's pre-training data."""
    spec = DINOV3_MODELS.get(model_name, {})
    if spec.get("sat"):
        return dinov3_sat_transform(**kwargs)
    return dinov3_web_transform(**kwargs)


# ── Loading ───────────────────────────────────────────────────────────────────

def _spec(model_name: str) -> dict:
    spec = DINOV3_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown DINOv3 model: '{model_name}'. Available: {list(DINOV3_MODELS)}")
    return spec


def _attach(model: Any, model_name: str, spec: dict, processor: Any | None) -> Any:
    mean, std = (SAT_MEAN, SAT_STD) if spec.get("sat") else (WEB_MEAN, WEB_STD)
    if processor is not None and getattr(processor, "image_mean", None) is not None:
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    model._pgv_processor = processor
    model._pgv_spec = spec
    model._pgv_name = model_name
    model._pgv_is_sat = bool(spec.get("sat"))
    model._pgv_mean, model._pgv_std = mean, std
    return model


def load_dinov3_hf(model_name: str = "dinov3_vitl16", device: str = "cpu",
                   attn_implementation: str | None = None) -> Any:
    """Load a real DINOv3 backbone from Hugging Face.

    Raises
    ------
    ModelLoadError
        If the repository cannot be downloaded (typically: licence not accepted / not logged in, or no
        network). Nothing is substituted.
    """
    spec = _spec(model_name)
    hf_id = spec["hf_id"]
    try:
        from transformers import AutoImageProcessor, AutoModel
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        model = AutoModel.from_pretrained(hf_id, **kw).to(device).eval()
        try:
            processor = AutoImageProcessor.from_pretrained(hf_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("No preprocessor for %s (%s); using the documented constants", hf_id, exc)
            processor = None
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Could not load {model_name} ({hf_id}): {type(exc).__name__}: {exc}. DINOv3 repositories are "
            "gated - accept the licence on the model page and authenticate (huggingface-cli login / "
            "HF_TOKEN). Nothing was substituted."
        ) from exc
    logger.info("DINOv3 loaded: %s (%s, ~%dM params) on %s", model_name, hf_id, spec["params_m"], device)
    return _attach(model, model_name, spec, processor)


def load_dinov3_hub(*_: Any, **__: Any) -> Any:
    """Removed. The old implementation called ``torch.hub.load('facebookresearch/dinov2', ...)`` and so
    returned DINOv2. Use :func:`load_dinov3_hf` or :func:`load_dinov3_local`."""
    raise NotImplementedError("torch.hub loading was removed (it loaded DINOv2). "
                              "Use load_dinov3_hf() or load_dinov3_local().")


def load_dinov3_local(model_name: str, weights_path: str, device: str = "cpu") -> Any:
    """Load DINOv3 from a local ``transformers`` snapshot directory (air-gapped installs).

    ``weights_path`` must be a directory holding ``config.json`` and the weights, e.g. the output of
    ``huggingface-cli download facebook/dinov3-vitl16-pretrain-sat493m --local-dir DIR`` or of
    ``model.save_pretrained(DIR)``. The config's architecture must match ``model_name``.
    """
    import json

    spec = _spec(model_name)
    path = Path(weights_path)
    if not path.is_dir():
        raise ValueError(f"weights_path must be a transformers snapshot directory, got {weights_path!r}")
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise ModelLoadError(f"{path} has no config.json - not a transformers snapshot")
    model_type = json.loads(cfg_file.read_text()).get("model_type", "")
    want = "dinov3_convnext" if spec["family"] == "convnext" else "dinov3_vit"
    if model_type != want:
        raise ModelLoadError(f"{path} holds a '{model_type}' model but {model_name} needs '{want}'")
    try:
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(str(path)).to(device).eval()
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not load {model_name} from {path}: {type(exc).__name__}: {exc}") from exc
    try:
        processor = AutoImageProcessor.from_pretrained(str(path))
    except Exception:  # noqa: BLE001
        processor = None
    return _attach(model, model_name, spec, processor)


# ── Image preparation ─────────────────────────────────────────────────────────

def _to_rgb01(image: Any, rgb_bands: tuple[int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rgb, valid)``: float32 ``(3, H, W)`` in [0, 1] and a bool ``(H, W)`` validity mask.

    * path: rasters are read nodata-aware and stretched 2-98 % over the valid pixels (``rgb_bands`` are
      1-based; default ``(1, 2, 3)``, or the single band repeated);
    * ``(H, W, C)`` array: uint8 is divided by 255, floats are taken to already be in [0, 1];
    * PIL image: converted to RGB.
    """
    if isinstance(image, (str, Path)):
        import rasterio
        with rasterio.open(str(image)) as src:
            bands = list(rgb_bands or ((1, 2, 3) if src.count >= 3 else (1, 1, 1)))
            if min(bands) < 1 or max(bands) > src.count:
                raise ValueError(f"rgb_bands {tuple(bands)} not available in a {src.count}-band raster")
            data = src.read(bands).astype(np.float32)
            valid = np.ones(data.shape[1:], bool)
            for b in set(bands):
                valid &= src.read_masks(b) > 0
        valid &= np.isfinite(data).all(0)
        if not valid.any():
            raise ValueError(f"{image} has no valid pixels")
        out = np.zeros_like(data)
        for k in range(3):
            lo, hi = np.percentile(data[k][valid], (2, 98))
            out[k] = np.clip((data[k] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        out[:, ~valid] = 0.0
        return out, valid
    if hasattr(image, "convert"):                       # PIL
        image = np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"expected an (H, W, >=3) array, got shape {arr.shape}")
    arr = arr[..., :3]
    arr = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else np.clip(arr.astype(np.float32), 0, 1)
    return arr.transpose(2, 0, 1), np.ones(arr.shape[:2], bool)


# ── DINOv3Backbone ────────────────────────────────────────────────────────────

class DINOv3Backbone:
    """Real DINOv3 feature extractor with registered dense features.

    Images are *not* centre-cropped: the raster is optionally down-scaled so its long side is at most
    ``max_size``, padded (edge replication) to a multiple of the patch size and fed whole. Feature cell
    ``(i, j)`` therefore covers source pixels ``[i*cell_px, (i+1)*cell_px) x [j*cell_px, ...)`` where
    ``cell_px = patch / scale`` (see :attr:`last_geometry`).

    Example::

        backbone = DINOv3Backbone("dinov3_vitl16_sat", device="cuda")
        feats = backbone.extract_features("scene.tif")      # (H_p, W_p, D)
        emb = backbone.extract_embeddings("scene.tif")      # (1, D)
    """

    def __init__(self, model_name: str = "dinov3_vitl16_sat", method: str = "hf",
                 device: str | None = None, weights_path: str | None = None,
                 max_size: int = 1024, rgb_bands: tuple[int, int, int] | None = None) -> None:
        self.model_name = model_name
        self.method = method
        self.device = device or self._auto_device()
        self.weights_path = weights_path
        self.max_size = max_size
        self.rgb_bands = rgb_bands
        self._model: Any = None
        self._spec = DINOV3_MODELS.get(model_name, {})
        self._is_sat = bool(self._spec.get("sat"))
        self.last_geometry: dict[str, Any] = {}

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.method == "hf":
            self._model = load_dinov3_hf(self.model_name, device=self.device)
        elif self.method == "local":
            if not self.weights_path:
                raise ValueError("method='local' needs weights_path")
            self._model = load_dinov3_local(self.model_name, self.weights_path, self.device)
        elif self.method == "hub":
            load_dinov3_hub()
        else:
            raise ValueError(f"unknown method {self.method!r}; use 'hf' or 'local'")

    # -- model introspection (works on injected models too)
    def _cfg(self) -> Any:
        return getattr(self._model, "config", None)

    def _patch(self) -> int:
        p = getattr(self._cfg(), "patch_size", None)
        return int(p) if isinstance(p, int) else int(self._spec.get("patch", 16))

    def _embed_dim(self) -> int:
        cfg = self._cfg()
        if getattr(cfg, "hidden_sizes", None):
            return int(cfg.hidden_sizes[-1])
        h = getattr(cfg, "hidden_size", None)
        return int(h) if isinstance(h, int) else int(self._spec.get("embed", 768))

    def _stats(self) -> tuple[tuple, tuple]:
        mean = getattr(self._model, "_pgv_mean", None)
        if mean is not None:
            return mean, self._model._pgv_std
        return (SAT_MEAN, SAT_STD) if self._is_sat else (WEB_MEAN, WEB_STD)

    def _is_convnext(self) -> bool:
        return self._spec.get("family") == "convnext" or "convnext" in str(getattr(self._cfg(), "model_type", ""))

    # -- forward
    def _prepare(self, image: Any) -> tuple[Any, int, int, float, tuple[int, int]]:
        """Image -> normalised, patch-aligned tensor ``(1, 3, H', W')`` plus ``(gh, gw, scale, (h0, w0))``."""
        import torch
        import torch.nn.functional as F

        rgb, _ = _to_rgb01(image, self.rgb_bands)
        h0, w0 = rgb.shape[1:]
        x = torch.from_numpy(rgb)[None]
        scale = 1.0
        if max(h0, w0) > self.max_size:
            scale = self.max_size / max(h0, w0)
            x = F.interpolate(x, size=(max(1, round(h0 * scale)), max(1, round(w0 * scale))),
                              mode="bilinear", antialias=True, align_corners=False)
        patch = self._patch()
        h1, w1 = x.shape[-2:]
        ph, pw = (-h1) % patch, (-w1) % patch
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        mean, std = self._stats()
        x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        return x.to(self.device), x.shape[-2] // patch, x.shape[-1] // patch, scale, (h0, w0)

    def _run(self, image: Any) -> tuple[Any, Any, dict]:
        """Return ``(cls (D,), grid (gh, gw, D), geometry)`` as torch tensors / dict."""
        import torch

        self._load()
        x, gh, gw, scale, (h0, w0) = self._prepare(image)

        with torch.no_grad():
            out = self._model(pixel_values=x)
        hs = out.last_hidden_state
        if hs.ndim == 4:                                       # (1, C, h, w) feature map
            grid = hs[0].permute(1, 2, 0)
            gh, gw = grid.shape[:2]
            pooled = getattr(out, "pooler_output", None)
            cls = pooled[0] if pooled is not None else grid.mean((0, 1))
            cell = x.shape[-2] / gh / scale
        else:
            # (1, n_prefix + gh*gw, D). ViT: CLS, then register tokens, then patches; ConvNeXt (transformers
            # 5.x): the pooled "CLS" token, then the h*w map positions row-major. Patches are always last.
            tok = hs[0]
            if tok.shape[0] < 1 + gh * gw:
                raise RuntimeError(f"backbone returned {tok.shape[0]} tokens for a {gh}x{gw} grid")
            cls = tok[0]
            grid = tok[-gh * gw:].reshape(gh, gw, -1)
            cell = x.shape[-2] / gh / scale
        geometry = {"cell_px": cell, "scale": scale, "grid": (int(gh), int(gw)), "source_size": (h0, w0)}
        self.last_geometry = geometry
        return cls, grid, geometry

    def extract_features(self, image: Any) -> np.ndarray:
        """Dense patch features, ``(H_p, W_p, D)`` (registered with the raster; see class docstring)."""
        return self._run(image)[1].float().cpu().numpy()

    def extract_embeddings(self, image: Any) -> np.ndarray:
        """Global descriptor, ``(1, D)``: the CLS token (ViT) or pooled map (ConvNeXt)."""
        return self._run(image)[0].float().cpu().numpy()[None]

    def extract_patch_features(self, image: Any) -> np.ndarray:
        """Per-patch features ``(N_patches, D)`` in row-major order (register tokens removed)."""
        grid = self._run(image)[1]
        return grid.reshape(-1, grid.shape[-1]).float().cpu().numpy()

    def get_attention_maps(self, image: Any, head_idx: int | None = None) -> np.ndarray:
        """CLS -> patch attention of the last layer: ``(H_p, W_p)`` for ``head_idx`` or, when it is None, the
        mean over heads (also ``(H_p, W_p)``).

        Needs a model that exposes attention weights (eager attention; the method tries to switch the model
        over). It raises instead of returning a placeholder map when it cannot.
        """
        import torch

        self._load()
        if self._is_convnext():
            raise NotImplementedError("attention maps exist only for the ViT backbones")
        x, gh, gw, _, _ = self._prepare(image)

        def _forward() -> Any:
            with torch.no_grad():
                return self._model(pixel_values=x, output_attentions=True)

        out = _forward()
        if (not getattr(out, "attentions", None) or out.attentions[-1] is None) \
                and hasattr(self._model, "set_attn_implementation"):
            self._model.set_attn_implementation("eager")
            out = _forward()
        if not getattr(out, "attentions", None) or out.attentions[-1] is None:
            raise RuntimeError("this model does not return attention weights; load it with "
                               "attn_implementation='eager'")
        attn = out.attentions[-1][0]                           # (heads, T, T)
        n_reg = int(getattr(self._cfg(), "num_register_tokens", 0) or 0)
        cls_to_patch = attn[:, 0, 1 + n_reg:]
        if cls_to_patch.shape[-1] != gh * gw:
            raise RuntimeError(f"attention covers {cls_to_patch.shape[-1]} patches, expected {gh * gw}")
        maps = cls_to_patch.reshape(-1, gh, gw).float().cpu().numpy()
        return maps[head_idx] if head_idx is not None else maps.mean(0)

    def build_classifier(self, num_classes: int, freeze_backbone: bool = True) -> Any:
        """Backbone + MLP head on the global descriptor. Input: normalised ``(B, 3, H, W)`` tensors with
        H, W multiples of the patch size (use :meth:`preprocess_batch` or your own pipeline)."""
        import torch.nn as nn

        self._load()
        dim = self._embed_dim()
        head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(512, num_classes))
        if freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad = False
        backbone = self._model

        class GeoClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                out = backbone(pixel_values=x)
                hs = out.last_hidden_state
                g = hs.mean((2, 3)) if hs.ndim == 4 else hs[:, 0]       # CLS / pooled token
                return self.head(g)

        return GeoClassifier()

    def finetune_config(self) -> dict:
        """Starting hyper-parameters for fine-tuning (a sensible default, not a published recipe)."""
        return {
            "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.05, "warmup_epochs": 10,
            "scheduler": "cosine_annealing", "mixed_precision": "bf16", "batch_size": 16,
            "note": "Use a lower LR (about 1e-5) for the SAT-493M backbones and when unfreezing.",
        }

    def __repr__(self) -> str:
        s = self._spec
        return (f"DINOv3Backbone(model={self.model_name!r}, params={s.get('params_m', 0)}M, "
                f"dataset={s.get('dataset', '?')!r}, embed={s.get('embed', 768)}, sat={self._is_sat}, "
                f"device={self.device})")


# ── CHMv2 - canopy height ─────────────────────────────────────────────────────

class CHMv2Model:
    """Canopy-height regression with Meta's released CHMv2 head (``facebook/dinov3-vitl16-chmv2-dpt-head``:
    a DINOv3 ViT-L/16 backbone with a DPT depth head, ``CHMv2ForDepthEstimation``).

    The earlier version put a **randomly initialised** decoder on the backbone and multiplied its output by
    ``max_height_m``, so every "height" was noise.

    Caveats (read before trusting numbers)
    --------------------------------------
    * The model takes 3-channel imagery. The model card does not state a ground sampling distance, and the
      SAT-493M backbone reportedly pre-trained on high-resolution RGB; do not assume Sentinel-2 10 m input
      is in-distribution without validating against reference heights.
    * Output units are not documented; the config range (0.001 - 96) suggests metres. Validate before use.
    * 8-bit imagery is used as-is; other dtypes are stretched 2-98 % to 0-255 per band (a visual stretch,
      not a radiometric calibration).
    """

    def __init__(self, device: str | None = None, hf_id: str = CHMV2_HF_ID,
                 tile_size: int = 384, overlap: int = 32) -> None:
        self.device = device or DINOv3Backbone._auto_device()
        self.hf_id = hf_id
        if tile_size % 16 or not 0 <= overlap < tile_size:
            raise ValueError("tile_size must be a multiple of 16 and overlap < tile_size")
        self.tile_size, self.overlap = tile_size, overlap
        self._model: Any = None
        self._mean: tuple = SAT_MEAN
        self._std: tuple = SAT_STD

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._model = AutoModelForDepthEstimation.from_pretrained(self.hf_id).to(self.device).eval()
            proc = AutoImageProcessor.from_pretrained(self.hf_id)
            self._mean, self._std = tuple(proc.image_mean), tuple(proc.image_std)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Could not load {self.hf_id}: {type(exc).__name__}: {exc}. The repository is gated - accept "
                "the licence and authenticate (huggingface-cli login / HF_TOKEN). Nothing was substituted."
            ) from exc

    def _predict_tiles(self, rgb255: np.ndarray) -> np.ndarray:
        """Overlapping-tile inference on a ``(3, H, W)`` float array in 0-255; overlaps are averaged."""
        import torch

        _, h, w = rgb255.shape
        t, step = self.tile_size, self.tile_size - self.overlap
        acc, cnt = np.zeros((h, w), np.float64), np.zeros((h, w), np.float64)
        mean = torch.tensor(self._mean).view(1, 3, 1, 1)
        std = torch.tensor(self._std).view(1, 3, 1, 1)
        ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - t, 0) + 1, step)) or [0]
        if ys[-1] + t < h:
            ys.append(h - t)
        if xs[-1] + t < w:
            xs.append(w - t)
        for y0 in ys:
            for x0 in xs:
                y0c, x0c = max(0, min(y0, h - t)), max(0, min(x0, w - t))
                tile = rgb255[:, y0c:y0c + t, x0c:x0c + t]
                th, tw = tile.shape[1:]
                ph, pw = (-th) % 16, (-tw) % 16
                if ph or pw:
                    tile = np.pad(tile, ((0, 0), (0, ph), (0, pw)), mode="edge")
                x = (torch.from_numpy(np.ascontiguousarray(tile))[None] / 255.0 - mean) / std
                with torch.no_grad():
                    pred = self._model(pixel_values=x.float().to(self.device)).predicted_depth
                pred = pred[0, :th, :tw].float().cpu().numpy()
                acc[y0c:y0c + th, x0c:x0c + tw] += pred
                cnt[y0c:y0c + th, x0c:x0c + tw] += 1
        return (acc / np.maximum(cnt, 1)).astype(np.float32)

    def predict_canopy_height(self, image_path: str, output_path: str | None = None,
                              rgb_bands: tuple[int, int, int] | None = None) -> dict[str, Any]:
        """Predict a canopy-height map for a raster.

        Returns ``{"height_map", "output_path", "statistics", "model", "image_path"}`` or
        ``{"error": ...}`` (missing file, no valid pixels, load failure).
        """
        if not Path(str(image_path)).exists():
            return {"error": f"File not found: {image_path}"}
        try:
            import rasterio
        except ImportError as exc:
            return {"error": f"Missing dependency: {exc}"}
        try:
            self._load()
            with rasterio.open(str(image_path)) as src:
                profile = src.profile.copy()
                native_u8 = src.dtypes[0] == "uint8"
            if native_u8:
                with rasterio.open(str(image_path)) as src:
                    bands = list(rgb_bands or ((1, 2, 3) if src.count >= 3 else (1, 1, 1)))
                    rgb255 = src.read(bands).astype(np.float32)
                    valid = np.ones(rgb255.shape[1:], bool)
                    for b in set(bands):
                        valid &= src.read_masks(b) > 0
            else:
                rgb01, valid = _to_rgb01(str(image_path), rgb_bands)
                rgb255 = rgb01 * 255.0
            heights = self._predict_tiles(rgb255)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

        heights = np.where(valid, np.maximum(heights, 0.0), np.nan).astype(np.float32)
        v = heights[valid]
        stats = {
            "mean_m": float(v.mean()), "max_m": float(v.max()), "p95_m": float(np.percentile(v, 95)),
            "coverage_pct": float((v > 2.0).mean() * 100),     # share of valid pixels taller than 2
        }
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            profile.update(count=1, dtype="float32", nodata=float("nan"), compress="lzw")
            for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                profile.pop(_k, None)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(heights[None])
        return {"height_map": heights, "output_path": output_path, "statistics": stats,
                "model": f"CHMv2 ({self.hf_id})", "image_path": str(image_path)}

    def estimate_biomass(self, *_: Any, **__: Any) -> dict[str, Any]:
        """Removed: the allometric coefficients (``0.112 * H**2.40``) had no source and the equation is not
        a height-only biomass model. Bring a calibrated height-to-AGB relation for your biome."""
        raise NotImplementedError("estimate_biomass was removed: no validated height->biomass relation ships "
                                  "with PyGeoVision")

    def detect_deforestation(self, before_path: str, after_path: str, min_height_loss_m: float = 2.0,
                             output_path: str | None = None) -> dict[str, Any]:
        """Flag pixels whose predicted height dropped by more than ``min_height_loss_m`` between two
        co-registered rasters of the same grid. Differences of two noisy predictions are noisy: treat the
        result as a screening layer, not a measurement."""
        try:
            import rasterio
            with rasterio.open(before_path) as a, rasterio.open(after_path) as b:
                if (a.height, a.width, a.crs, a.transform) != (b.height, b.width, b.crs, b.transform):
                    return {"error": "before/after rasters must share the same grid (size, CRS, transform); "
                                     "reproject one onto the other first"}
                profile, transform, crs = a.profile.copy(), a.transform, a.crs
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}
        before = self.predict_canopy_height(before_path)
        after = self.predict_canopy_height(after_path)
        if "error" in before or "error" in after:
            return {"error": before.get("error") or after.get("error")}

        hb, ha = before["height_map"], after["height_map"]
        ok = np.isfinite(hb) & np.isfinite(ha)
        mask = (ok & ((hb - ha) > min_height_loss_m)).astype(np.uint8)
        pixel_area_m2 = abs(transform.a * transform.e) if crs is not None and crs.is_projected else None
        result = {
            "deforestation_mask": mask,
            "deforested_pct": float(mask[ok].mean() * 100) if ok.any() else 0.0,
            "area_ha": float(mask.sum() * pixel_area_m2 / 10000) if pixel_area_m2 else None,
            "output_path": output_path,
        }
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            profile.update(count=1, dtype="uint8", nodata=None, compress="lzw")
            for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                profile.pop(_k, None)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(mask[None])
        return result


# ── dino.txt ──────────────────────────────────────────────────────────────────

class DINOv3Text:
    """Not available. Zero-shot text alignment needs the ``dino.txt`` checkpoint, which is not in the public
    model index; the earlier class compared DINOv3 patch features with CLIP *text* embeddings, which live in
    unrelated spaces, so its masks were arbitrary. Use ``pygeovision.models.vlm.clip.CLIPGeo`` or
    ``FoundationModelLabeler(model='remote_clip', text_prompts=[...])`` for prompt-based labelling."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "DINOv3Text is unavailable (no dino.txt checkpoint). For text prompts use CLIPGeo / RemoteCLIP."
        )


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune_dinov3(
    model_name: str = "dinov3_vitl16_sat",
    dataset: Any = None,
    task: str = "classification",
    num_classes: int = 2,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    mixed_precision: bool = True,
    distributed: bool = False,
    output_dir: str = "./checkpoints/dinov3/",
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare (not run) a DINOv3 classification fine-tune: model, optimiser, scheduler, mixed-precision and
    checkpoint managers. Pass the returned objects to your training loop / ``GeoTrainer``.

    Only ``task="classification"`` is implemented. (``segmentation`` used to return an unrelated randomly
    initialised SegFormer; use ``models.get_model('segformer_*')`` or a DINOv3 head of your own.)
    """
    if task != "classification":
        raise NotImplementedError(
            f"finetune_dinov3 supports task='classification' only (got {task!r}); build a dense head on "
            "DINOv3Backbone.extract_features for segmentation."
        )
    import torch

    from pygeovision.training.checkpoint import CheckpointManager
    from pygeovision.training.mixed_precision import MixedPrecisionManager

    backbone = DINOv3Backbone(model_name)
    model = backbone.build_classifier(num_classes, freeze_backbone=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.05, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    mp_manager = MixedPrecisionManager(precision="bf16" if mixed_precision else "fp32")
    ckpt_mgr = CheckpointManager(output_dir, monitor="val_loss", mode="min")
    if distributed:
        from pygeovision.training.distributed import wrap_ddp
        model = wrap_ddp(model)
    model = model.to(backbone.device)
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler, "mp_manager": mp_manager,
            "ckpt_manager": ckpt_mgr, "config": backbone.finetune_config(), "status": "ready",
            "note": "Objects are prepared, not trained: run your loop or GeoTrainer.fit()."}


# ── Convenience ───────────────────────────────────────────────────────────────

def list_dinov3_models() -> list[str]:
    """All DINOv3 model names."""
    return list(DINOV3_MODELS)


def list_satellite_models() -> list[str]:
    """DINOv3 models pre-trained on SAT-493M."""
    return [n for n, s in DINOV3_MODELS.items() if s.get("sat")]


def get_dinov3_info(model_name: str) -> dict:
    """Spec for one model, plus the transform it needs."""
    spec = _spec(model_name)
    return {**spec, "name": model_name,
            "transform": "satellite stats (SAT-493M)" if spec.get("sat") else "ImageNet web stats (LVD-1689M)",
            "gated": True}


__all__ = [
    "CHMV2_HF_ID", "CHMv2Model", "DINOV3_HEADS", "DINOV3_MODELS", "DINOv3Backbone", "DINOv3Text", "SAT_MEAN",
    "SAT_STD", "WEB_MEAN", "WEB_STD", "dinov3_sat_transform", "dinov3_web_transform", "finetune_dinov3",
    "get_dinov3_info", "get_transform", "list_dinov3_models", "list_satellite_models", "load_dinov3_hf",
    "load_dinov3_hub", "load_dinov3_local",
]
