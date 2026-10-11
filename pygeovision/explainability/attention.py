"""Attention map extraction for transformer-based geospatial models (G6)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AttentionMapExtractor:
    """Extract self-attention maps from Vision Transformer (ViT) based models.

    Works with models that expose their attention probabilities: either an
    attention module that returns ``(output, attention_weights)`` (for example
    ``torch.nn.MultiheadAttention``), or a model whose output carries an
    ``.attentions`` tuple (Hugging Face ``output_attentions=True``).
    Modules whose name contains ``attn`` or ``attention`` are inspected.

    Example::

        extractor = AttentionMapExtractor(model)
        attn = extractor.extract(image_tensor, layer_idx=-1)
        extractor.visualise(image_path, attn, "./results/attention.png")

    ``extract`` returns the attention over the patch grid (for example
    ``14 x 14`` for a 224-pixel input with 16-pixel patches), not at pixel
    resolution; ``visualise`` resizes it.
    """

    def __init__(self, model: Any, device: str | None = None) -> None:
        self.model = model
        self.device = device or self._model_device(model)
        self._attention_maps: list = []

    @staticmethod
    def _model_device(model: Any) -> str:
        try:
            return str(next(model.parameters()).device)
        except Exception:                      # no parameters, or not a torch module
            return "cpu"

    @staticmethod
    def _as_probs(t: Any):
        """Return ``t`` as ``(B, heads, N, N)`` if it looks like attention weights, else None."""
        try:
            if t.ndim == 3 and t.shape[-1] == t.shape[-2]:       # head-averaged (B, N, N)
                return t.unsqueeze(1)
            if t.ndim == 4 and t.shape[-1] == t.shape[-2]:
                return t
        except AttributeError:
            pass
        return None

    def _register_attention_hooks(self) -> list:
        """Hook attention modules; returns the hook handles so they can be removed."""
        self._attention_maps = []
        seen: set = set()                      # nested attention modules can return the same weights

        def _keep(t):
            probs = self._as_probs(t)
            if probs is not None and probs.data_ptr() not in seen:
                seen.add(probs.data_ptr())
                self._attention_maps.append(probs.detach().cpu())

        def _hook(_, __, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                _keep(output[1])
            elif hasattr(output, "attentions") and output.attentions:
                for a in output.attentions:
                    _keep(a)

        handles = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handles.append(module.register_forward_hook(_hook))
        return handles

    def extract(self, image: Any, layer_idx: int = -1,
                head_idx: int | None = None, *, forward_kwargs: dict | None = None,
                strict: bool = False) -> Any:
        """Extract an attention map for an image.

        Args:
            image: Tensor or array ``(C, H, W)`` or ``(1, C, H, W)``.
            layer_idx: Which captured attention layer (default the last).
            head_idx: One head, or ``None`` to average over heads.
            forward_kwargs: Extra arguments for the model call, e.g.
                ``{"output_attentions": True}`` for Hugging Face models.
            strict: Raise instead of returning zeros when nothing can be extracted.

        Returns:
            ``(n, n)`` array over the patch grid. With a class token (the
            token count minus one is a square) the map is the attention the
            class token pays to each patch; with ``n*n`` tokens it is the
            attention each patch receives on average. If the tokens do not
            form a grid a 1-D vector is returned. On failure (and
            ``strict=False``) an array of zeros with the input's height and
            width is returned and the reason is logged as a warning.
        """
        import numpy as np
        handles: list = []
        shape_hw = (64, 64)
        try:
            import torch
            if isinstance(image, np.ndarray):
                image = torch.tensor(image, dtype=torch.float32)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            shape_hw = (image.shape[-2], image.shape[-1])
            image = image.to(self.device)
            self.model.eval()
            handles = self._register_attention_hooks()
            with torch.no_grad():
                out = self.model(image, **(forward_kwargs or {}))

            maps = list(self._attention_maps)
            if not maps and getattr(out, "attentions", None):
                maps = [p.detach().cpu() for p in (self._as_probs(a) for a in out.attentions) if p is not None]
            if not maps:
                raise RuntimeError(
                    "no attention weights were produced. Attention modules must return "
                    "(output, weights); for Hugging Face models pass "
                    "forward_kwargs={'output_attentions': True} and load the model with "
                    "attn_implementation='eager'."
                )

            attn = maps[layer_idx]                       # (B, heads, N, N)
            if head_idx is not None:
                if not -attn.shape[1] <= head_idx < attn.shape[1]:
                    raise IndexError(
                        f"head_idx {head_idx} out of range: this layer's attention has "
                        f"{attn.shape[1]} head(s) (modules that return head-averaged weights "
                        "report a single head)")
                attn = attn[:, head_idx]
            else:
                attn = attn.mean(dim=1)
            attn = attn[0]                               # (N, N)
            n_tok = attn.shape[0]
            side = int(round(n_tok ** 0.5))
            if side * side == n_tok:
                return attn.mean(0).reshape(side, side).numpy()
            side = int(round((n_tok - 1) ** 0.5))
            if side * side == n_tok - 1:
                return attn[0, 1:].reshape(side, side).numpy()     # class token -> patches
            logger.warning("%d attention tokens do not form a square grid; returning a vector", n_tok)
            return attn.mean(0).numpy()
        except Exception as exc:
            if strict:
                raise
            logger.warning("Attention extraction failed: %s", exc)
            return np.zeros(shape_hw)
        finally:
            for h in handles:
                h.remove()

    def visualise(self, image_path: str, attention: Any, output_path: str,
                  alpha: float = 0.6, colormap: str = "jet") -> None:
        """Save a PNG with the input, the attention map and an overlay."""
        import numpy as np
        attn = np.asarray(attention, dtype=np.float32)
        if attn.ndim != 2:
            raise ValueError("attention must be a 2-D map; got shape %s" % (attn.shape,))
        rng = attn.max() - attn.min()
        attn = (attn - attn.min()) / rng if rng > 0 else attn * 0
        try:
            from pygeovision.explainability.gradcam import _save_overlay
            _save_overlay(image_path, attn, output_path, alpha=alpha,
                          colormap=colormap, title="Attention")
        except ImportError as exc:
            logger.warning("matplotlib required for visualisation: %s", exc)
