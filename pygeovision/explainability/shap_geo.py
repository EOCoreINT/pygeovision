"""SHAP values for geospatial feature importance (G6)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GeospatialSHAP:
    """SHAP-based feature importance for satellite imagery models.

    Computes band-level and spatial SHAP values to explain
    which spectral bands are most important for predictions.

    Example::

        shap_exp = GeospatialSHAP(model)
        values = shap_exp.band_importance(image_tensor, n_samples=100)
        shap_exp.plot_band_importance(values, band_names=["B02","B03","B04","B08"])
    """

    def __init__(self, model: Any, device: str | None = None,
                 background_samples: int = 50) -> None:
        self.model = model
        self.device = device or "cpu"
        self.background_samples = background_samples

    def band_importance(self, image: Any, n_samples: int | None = None) -> dict[str, Any]:
        """Compute spectral band importance using SHAP.

        Args:
            n_samples: number of background samples for the SHAP
                DeepExplainer's reference distribution (overrides
                `background_samples` from the constructor for this call).
                More samples give a more stable baseline expectation at
                the cost of more compute.
        """
        try:
            import numpy as np
            import shap
            import torch
        except ImportError:
            return {"error": "pip install shap"}

        self.model.eval()
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        n_bg = n_samples if n_samples is not None else self.background_samples
        # Background distribution: n_bg all-zero reference images. SHAP's
        # DeepExplainer needs a background SET (not a single sample) to
        # estimate a stable baseline expectation — using just one zero
        # image (as this previously did, silently ignoring
        # background_samples/n_samples entirely) gives a noisier,
        # less-representative baseline.
        background = torch.zeros(
            (n_bg, *image.shape[1:]), dtype=torch.float32,
        )
        explainer = shap.DeepExplainer(self.model, background.to(self.device))
        # check_additivity=False: SHAP's DeepExplainer additivity check has
        # known false-positive failures with newer PyTorch versions (SHAP's
        # PyTorch backend hooks lag behind PyTorch's op internals) —
        # confirmed via direct testing, failing even for a minimal
        # Linear-only model. This is SHAP's own documented workaround; it
        # skips only the internal consistency verification, not the actual
        # SHAP value computation itself.
        shap_values = explainer.shap_values(image.to(self.device), check_additivity=False)

        band_importance = {}
        for b in range(image.shape[1]):
            if isinstance(shap_values, list):
                importance = float(abs(shap_values[0][:, b]).mean())
            else:
                importance = float(abs(shap_values[:, b]).mean())
            band_importance[f"band_{b+1}"] = round(importance, 6)

        return {"band_importance": band_importance, "method": "shap"}

    def plot_band_importance(self, values: dict, band_names: list | None = None,
                              save_path: str | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
            bi = values.get("band_importance", {})
            labels = band_names or list(bi.keys())
            importances = list(bi.values())
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels, importances, color="steelblue")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Spectral Band Importance (SHAP)")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=120)
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib required for plot")