"""Moondream2 for satellite image captioning and visual question answering."""
from __future__ import annotations

import logging
from typing import Any

from pygeovision.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

__all__ = ["MoondreamGeo"]


class MoondreamGeo:
    """Moondream2 -- lightweight VLM for satellite image understanding.

    Example::

        vl = MoondreamGeo()
        caption = vl.caption("./sentinel2.tif")
        answer  = vl.vqa("./sentinel2.tif", "How many buildings are visible?")
        results = vl.batch_caption("./data/", top_k=10)

    The model ships its own code (``trust_remote_code``), so the revision is **pinned**:
    an unpinned load would execute whatever the repository's latest commit contains.  Change
    :attr:`REVISION` deliberately, after reading the diff.  ``REVISION`` is the release whose
    ``caption()`` / ``query()`` API this class uses; older releases (``encode_image`` /
    ``answer_question``) are still handled.
    """

    MODEL_ID = "vikhyatk/moondream2"
    REVISION = "2025-06-21"

    def __init__(self, device: str | None = None, revision: str | None = None) -> None:
        self.device = device or "cpu"
        self.revision = revision or self.REVISION
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers + torch required: pip install transformers torch") from exc
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, revision=self.revision, trust_remote_code=True, torch_dtype=dtype,
            ).to(self.device).eval()
            if not hasattr(self._model, "caption"):  # pre-2025 releases use the tokenizer API
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_ID, revision=self.revision, trust_remote_code=True)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load {self.MODEL_ID}@{self.revision}: {exc}",
                details="Needs network access to the Hugging Face Hub (or a populated cache).",
            ) from exc

    @staticmethod
    def _load_pil(image_path: str):
        import numpy as np
        import rasterio
        from PIL import Image
        with rasterio.open(image_path) as src:
            data = src.read(list(range(1, min(src.count, 3) + 1)), masked=True).astype("float32")
        out = np.zeros(data.shape, dtype="float32")
        for b in range(data.shape[0]):
            valid = data[b].compressed()
            if valid.size:
                p2, p98 = np.percentile(valid, (2, 98))
                out[b] = np.clip((data[b].filled(p2) - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
        if out.shape[0] == 1:
            out = np.repeat(out, 3, axis=0)
        elif out.shape[0] == 2:
            out = np.stack([out[0], out[1], out[0]], axis=0)
        return Image.fromarray(out[:3].transpose(1, 2, 0).astype(np.uint8))

    def _ask(self, image_path: str, question: str | None) -> str:
        self._load()
        img = self._load_pil(image_path)
        model = self._model
        if hasattr(model, "caption") and hasattr(model, "query"):      # current API
            if question is None:
                return str(model.caption(img, length="normal")["caption"])
            return str(model.query(img, question)["answer"])
        enc = model.encode_image(img)                                     # legacy API
        return str(model.answer_question(
            enc, question or "Describe this satellite image in detail.", self._tokenizer))

    def caption(self, image_path: str) -> str:
        """Natural-language caption for a satellite image."""
        return self._ask(image_path, None)

    def vqa(self, image_path: str, question: str) -> str:
        """Answer a question about a satellite image."""
        return self._ask(image_path, question)

    def batch_caption(self, image_dir: str, top_k: int = 10) -> list[dict]:
        """Caption up to ``top_k`` GeoTIFFs under ``image_dir`` (sorted by path).

        A file that fails is reported in its result (``"error"``) rather than aborting the batch.
        """
        import pathlib
        paths = sorted(pathlib.Path(image_dir).rglob("*.tif"))[:top_k]
        results = []
        for p in paths:
            try:
                results.append({"path": str(p), "caption": self.caption(str(p))})
            except ModelLoadError:
                raise  # the model itself is unusable: every file would fail identically
            except Exception as exc:
                logger.warning("Moondream: %s failed (%s: %s)", p, type(exc).__name__, exc)
                results.append({"path": str(p), "caption": None, "error": f"{type(exc).__name__}: {exc}"})
        return results
