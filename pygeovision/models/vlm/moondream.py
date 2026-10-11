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

    def __init__(self, device: str | None = None, revision: str | None = None,
                 bands: tuple[int, ...] | None = None) -> None:
        """``bands``: file band numbers shown to the model as R, G, B (default: the first three)."""
        self.bands = tuple(bands) if bands else None
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

    def _load_pil(self, image_path: str):
        """The file as an 8-bit RGB picture (``bands``, shared percentile stretch, nodata black)."""
        from pygeovision.models.vlm._image import load_rgb_pil
        return load_rgb_pil(image_path, bands=self.bands)

    def _ask(self, image_path: str, question: str | None) -> str:
        img = self._load_pil(image_path)      # bad file or bands fail here, before any model download
        self._load()
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
        """Caption the first ``top_k`` images under ``image_dir`` (``.tif/.tiff/.png/.jpg/.jpeg``, recursive, sorted by path).

        A file that fails is reported in its result (``"error"``) rather than aborting the batch.
        """
        from pygeovision.models.vlm._image import list_images
        paths = list_images(image_dir)[:top_k]
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
