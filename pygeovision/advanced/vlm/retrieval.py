"""Geo image retrieval using CLIP embeddings."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class GeoImageRetrieval:
    """Image-to-image and text-to-image search over a folder of georeferenced images.

    ``build_index`` embeds every image once; searches then only embed the query. The index is a plain
    JSON file (``{"model": ..., "embeddings": {path: vector}}``) that can be saved and reloaded; an index
    is only valid for the CLIP model that built it.

    Example::

        r = GeoImageRetrieval(model="remoteclip-b32")
        r.build_index("./tiles", save_path="tiles_index.json")
        r.search_by_text("flooded farmland", top_k=5)
        r.search_by_image("./tiles/a.tif")
    """

    def __init__(self, model: str = "openclip-b32", device: str | None = None,
                 bands: tuple[int, ...] | None = None) -> None:
        """``bands``: file band numbers shown to the model as R, G, B (default: the first three)."""
        from pygeovision.models.vlm.clip import CLIPGeo
        self.model = model
        self._clip = CLIPGeo(model=model, device=device, bands=bands)
        self._index: dict[str, Any] = {}
        self.skipped: dict[str, str] = {}

    def build_index(self, image_dir: str, save_path: str | None = None) -> int:
        """Embed every image under ``image_dir`` (``.tif/.tiff/.png/.jpg/.jpeg``, recursive) and keep the vectors.

        Starts a new index (earlier entries are dropped). Files that cannot be read are skipped and listed,
        with the reason, in ``self.skipped``. Returns the number of images indexed.
        """
        from pygeovision.models.vlm._image import list_images

        self._index, self.skipped = {}, {}
        for p in list_images(image_dir):
            try:
                self._index[str(p)] = self._clip.embed_image(str(p)).tolist()
            except Exception as exc:
                self.skipped[str(p)] = f"{type(exc).__name__}: {exc}"
                logger.warning("Skipping %s: %s: %s", p, type(exc).__name__, exc)
        if save_path:
            with open(save_path, "w") as f:
                json.dump({"model": self.model, "embeddings": self._index}, f)
        logger.info("Index built: %d images (%d skipped)", len(self._index), len(self.skipped))
        return len(self._index)

    def load_index(self, path: str) -> None:
        """Load an index written by :meth:`build_index` (older files without a model name are accepted).

        Raises ``ValueError`` if the index was built with a different model than this retriever uses,
        because vectors from different models cannot be compared.
        """
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "embeddings" in data:
            if data.get("model") not in (None, self.model):
                raise ValueError(f"index {path} was built with model {data['model']!r}, but this retriever "
                                 f"uses {self.model!r}; create GeoImageRetrieval(model={data['model']!r})")
            self._index = data["embeddings"]
        else:
            self._index = data

    def _require_index(self) -> None:
        if not self._index:
            raise RuntimeError("the index is empty: call build_index() or load_index() first")

    def _rank(self, query_emb: Any, exclude: str | None, top_k: int) -> list[dict]:
        import numpy as np
        self._require_index()
        out = []
        skip = os.path.abspath(exclude) if exclude else None
        for p, emb in self._index.items():
            if skip is not None and os.path.abspath(p) == skip:      # the query itself, however its path is spelled
                continue
            vec = np.asarray(emb)
            if vec.shape != query_emb.shape:
                raise ValueError(f"index vectors have {vec.shape[0]} values but the query has "
                                 f"{query_emb.shape[0]}: the index was built with a different model")
            out.append((p, float(np.dot(query_emb, vec))))
        out.sort(key=lambda x: x[1], reverse=True)
        return [{"path": p, "score": round(s, 4)} for p, s in out[:top_k]]

    def search_by_text(self, query: str, top_k: int = 10) -> list[dict]:
        """The ``top_k`` indexed images closest to a text description: ``[{"path", "score"}]``, best first."""
        self._require_index()
        return self._rank(self._clip.embed_text(query), None, top_k)

    def search_by_image(self, image_path: str, top_k: int = 10) -> list[dict]:
        """The ``top_k`` indexed images most similar to ``image_path`` (the image itself is left out)."""
        self._require_index()
        return self._rank(self._clip.embed_image(image_path), str(image_path), top_k)
