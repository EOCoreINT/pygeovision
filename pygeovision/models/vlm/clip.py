"""CLIP, OpenCLIP, RemoteCLIP, and GeoRSCLIP for geospatial zero-shot
classification and retrieval (G2).

Two genuinely different loading mechanisms are needed here, and using the
wrong one for a given model silently fails or produces garbage:

  - Standard OpenAI/LAION CLIP releases (openclip-b32, openclip-l14,
    clip-vit-b32) ARE packaged in the `transformers`-compatible format —
    `transformers.CLIPModel.from_pretrained(hf_id)` works directly.

  - RemoteCLIP (official repo: chendelong/RemoteCLIP) and GeoRSCLIP
    (official repo: Zilun/GeoRSCLIP) are NOT transformers-compatible,
    despite being hosted on HuggingFace — both are raw `open_clip`-format
    `.pt` state dicts. Loading them requires building the base
    architecture with `open_clip.create_model_and_transforms()` and then
    `model.load_state_dict()`-ing the downloaded checkpoint. A prior
    version of this file used `transformers.CLIPModel.from_pretrained()`
    for these too, pointed at a nonexistent/incompatible repo path — that
    would have failed (or silently loaded the wrong thing) for every
    RemoteCLIP/GeoRSCLIP call.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Standard transformers-compatible CLIP repos.
_TRANSFORMERS_HF_MODELS = {
    "clip-vit-b32":   "openai/clip-vit-base-patch32",
    "openclip-b32":   "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "openclip-l14":   "openai/clip-vit-large-patch14",
}

# RemoteCLIP / GeoRSCLIP: (open_clip base architecture name, HF repo,
# checkpoint filename on that repo). These require open_clip, NOT transformers.
_OPENCLIP_CHECKPOINTS = {
    "remoteclip-b32": ("ViT-B-32", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
    "remoteclip-l14": ("ViT-L-14", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"),
    "georsclip":      ("ViT-B-32", "Zilun/GeoRSCLIP", "RS5M_ViT-B-32.pt"),
}


class CLIPGeo:
    """CLIP / RemoteCLIP / GeoRSCLIP for zero-shot geospatial classification
    and image-text search (G2).

    Remote-sensing-finetuned variants (RemoteCLIP, GeoRSCLIP) understand
    domain terms like "flooded farmland", "urban expansion", "deforestation"
    much better than the base OpenAI/LAION CLIP, which was trained on
    general web images.

    Example::

        clip = CLIPGeo(model="remoteclip-b32")
        scores = clip.zero_shot(
            image_path="./sentinel2.tif",
            categories=["deforestation", "healthy forest", "agriculture", "urban"],
        )
        similar = clip.search(query="coastal flooding", image_dir="./images/")
    """

    # Kept for backwards compatibility with any code reading HF_MODELS
    # directly — only the genuinely transformers-compatible entries.
    HF_MODELS = dict(_TRANSFORMERS_HF_MODELS)

    def __init__(self, model: str = "openclip-b32", device: str | None = None,
                 cache_dir: str | None = None) -> None:
        self.model_name = model
        self.device = device or "cpu"
        self.cache_dir = cache_dir
        self._backend: str | None = None  # "transformers" | "open_clip"
        self._model = None
        self._processor = None       # transformers backend
        self._preprocess = None       # open_clip backend (image transform)
        self._tokenizer = None        # open_clip backend

    def _load(self):
        if self._model is not None:
            return

        if self.model_name in _OPENCLIP_CHECKPOINTS:
            self._load_open_clip()
        elif self.model_name in _TRANSFORMERS_HF_MODELS:
            self._load_transformers()
        else:
            # Unknown name — assume it's a raw transformers-compatible HF id
            # the caller passed directly (matches the old HF_MODELS.get(model, model) fallback).
            self._load_transformers(hf_id_override=self.model_name)

    def _load_transformers(self, hf_id_override: str | None = None) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("transformers required: pip install transformers")
        hf_id = hf_id_override or _TRANSFORMERS_HF_MODELS[self.model_name]
        self._processor = CLIPProcessor.from_pretrained(hf_id)
        self._model = CLIPModel.from_pretrained(hf_id).to(self.device).eval()
        self._backend = "transformers"

    def _load_open_clip(self) -> None:
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch required for RemoteCLIP/GeoRSCLIP: "
                "pip install open_clip_torch"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")
        import torch

        arch, repo_id, filename = _OPENCLIP_CHECKPOINTS[self.model_name]
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=False)
        checkpoint_path = hf_hub_download(repo_id, filename, cache_dir=self.cache_dir)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Some checkpoints wrap the actual state dict in a bit more structure
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            logger.warning(
                "%s: checkpoint load had %d missing / %d unexpected keys — "
                "verify the checkpoint matches architecture %r.",
                self.model_name, len(msg.missing_keys), len(msg.unexpected_keys), arch,
            )
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(arch)
        self._backend = "open_clip"

    def _load_image(self, image_path: str):
        import numpy as np
        import rasterio
        from PIL import Image
        with rasterio.open(image_path) as src:
            data = src.read(list(range(1, min(src.count, 4) + 1))).astype(float)
        for b in range(data.shape[0]):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        return Image.fromarray(data[:3].transpose(1, 2, 0).astype(np.uint8))

    # ── Unified public API — works identically regardless of backend ──────────

    def zero_shot(self, image_path: str, categories: list[str]) -> dict[str, float]:
        """Zero-shot classify an image against text categories."""
        import torch
        self._load()
        img = self._load_image(image_path)

        if self._backend == "transformers":
            inputs = self._processor(text=categories, images=img,
                                      return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()
        else:  # open_clip
            img_t = self._preprocess(img).unsqueeze(0).to(self.device)
            text_t = self._tokenizer(categories).to(self.device)
            with torch.no_grad():
                img_feat = self._model.encode_image(img_t)
                txt_feat = self._model.encode_text(text_t)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * img_feat @ txt_feat.T)
            probs = logits.softmax(dim=-1)[0].cpu().tolist()

        return {cat: round(p, 4) for cat, p in zip(categories, probs)}

    def embed_image(self, image_path: str) -> Any:
        """Get a normalised CLIP image embedding for semantic search."""
        import torch
        import torch.nn.functional as F
        self._load()
        img = self._load_image(image_path)

        if self._backend == "transformers":
            inputs = self._processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = self._model.get_image_features(**inputs)
        else:
            img_t = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self._model.encode_image(img_t)

        return F.normalize(feat, dim=-1).cpu().numpy().squeeze()

    def embed_text(self, text: str) -> Any:
        """Get a normalised CLIP text embedding."""
        import torch
        import torch.nn.functional as F
        self._load()

        if self._backend == "transformers":
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feat = self._model.get_text_features(**inputs)
        else:
            text_t = self._tokenizer([text]).to(self.device)
            with torch.no_grad():
                feat = self._model.encode_text(text_t)

        return F.normalize(feat, dim=-1).cpu().numpy().squeeze()

    def search(self, query: str, image_dir: str, top_k: int = 5) -> list[dict]:
        """Search a directory of images by text query."""
        import pathlib

        import numpy as np
        image_paths = list(pathlib.Path(image_dir).rglob("*.tif")) + \
                      list(pathlib.Path(image_dir).rglob("*.png"))
        query_emb = self.embed_text(query)
        results = []
        for p in image_paths:
            try:
                img_emb = self.embed_image(str(p))
                score = float(np.dot(query_emb, img_emb))
                results.append({"path": str(p), "score": round(score, 4)})
            except Exception:
                continue
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]