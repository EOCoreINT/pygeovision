"""Uncertainty estimation via Monte Carlo Dropout (G6)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Monte Carlo Dropout uncertainty estimation for segmentation models.

    Runs ``n_passes`` stochastic forward passes (dropout on, everything else
    in eval mode) over every chip and reports, per pixel:

        - predictive entropy of the mean prediction (total uncertainty)
        - epistemic uncertainty: variance of the class probabilities across
          passes (the passes disagree, so the *model* is unsure)
        - aleatoric uncertainty: mean entropy of the individual passes
          (each pass is itself unsure, so the *data* is ambiguous)

    The model must output class scores ``(B, classes, H, W)`` and contain at
    least one dropout layer; without dropout every pass is identical and the
    result carries a ``"warning"``.

    Example::

        estimator = UncertaintyEstimator(model, n_passes=20)
        result = estimator.estimate(
            "./data/sentinel2.tif",
            "./results/uncertainty_map.tif",
        )
    """

    def __init__(self, model: Any, n_passes: int = 20,
                 device: str | None = None) -> None:
        self.model = model
        self.n_passes = n_passes
        self.device = device or self._auto_device()

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _enable_dropout(self) -> int:
        """Put the model in eval mode, then switch every dropout layer back on.

        Eval mode keeps BatchNorm on its running statistics (and stops it
        updating them); only dropout stays stochastic. Returns the number of
        dropout layers found.
        """
        import torch.nn as nn
        self.model.eval()
        n = 0
        for m in self.model.modules():
            if isinstance(m, nn.modules.dropout._DropoutNd):
                m.train()
                n += 1
        return n

    def estimate(
        self,
        image_path: str | Any,
        output_path: str | None = None,
        chip_size: int = 512,
        overlap: int = 64,
        save_epistemic: bool = True,
        save_aleatoric: bool = True,
    ) -> dict[str, Any]:
        """Estimate per-pixel uncertainty across the GeoTIFF.

        Every chip is min-max normalised per band, run ``n_passes`` times and
        written into the output; where chips overlap the later chip wins.
        Memory use is one ``(H, W)`` float32 array per output band plus the
        image itself. The model is returned to eval mode afterwards.

        Returns a dict with ``success`` and summary statistics
        (``mean_entropy``, ``mean_epistemic``, ``mean_aleatoric``,
        ``high_uncertainty_fraction`` = share of pixels with entropy above
        0.5 nats, ``n_mc_passes`` actually used, ``n_dropout_layers``) and,
        when ``output_path`` is given, ``output_path``. Problems with the
        inputs are returned as ``{"success": False, "error": ...}``.
        """
        try:
            import numpy as np
            import rasterio
            import torch
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        stride = chip_size - overlap
        if stride <= 0:
            return {
                "success": False,
                "error": (
                    f"chip_size ({chip_size}) must be greater than overlap "
                    f"({overlap}) — got stride={stride}. Reduce overlap or "
                    f"increase chip_size."
                ),
            }

        self.model = self.model.to(self.device)
        n_dropout = self._enable_dropout()
        n_passes = self.n_passes
        warning = None
        if n_dropout == 0:
            warning = ("model has no dropout layers, so every pass is identical: "
                       "epistemic uncertainty is 0. Use a model trained with dropout.")
            logger.warning(warning)
            n_passes = 1

        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            H, W = src.height, src.width
            image = src.read().astype(np.float32)

        entropy = np.zeros((H, W), np.float32)
        epistemic = np.zeros((H, W), np.float32)
        aleatoric = np.zeros((H, W), np.float32)
        label = np.zeros((H, W), np.uint8)

        try:
            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    r2, c2 = min(row + chip_size, H), min(col + chip_size, W)
                    chip = image[:, row:r2, col:c2].copy()
                    for b in range(chip.shape[0]):
                        mn, mx = chip[b].min(), chip[b].max()
                        chip[b] = (chip[b] - mn) / (mx - mn + 1e-8)
                    padded = np.zeros((image.shape[0], chip_size, chip_size), np.float32)
                    padded[:, :chip.shape[1], :chip.shape[2]] = chip
                    chip_t = torch.tensor(padded).unsqueeze(0).to(self.device)

                    ah, aw = r2 - row, c2 - col
                    s1 = s2 = s_ent = None
                    with torch.no_grad():
                        for _ in range(n_passes):
                            out = self.model(chip_t)
                            if out.ndim != 4:
                                return {"success": False, "error": (
                                    "model output must be (B, classes, H, W) class scores; "
                                    f"got shape {tuple(out.shape)}")}
                            p = torch.softmax(out, dim=1)[0, :, :ah, :aw].double().cpu().numpy()
                            ent = np.maximum(-np.sum(p * np.log(p + 1e-10), axis=0), 0.0)
                            if s1 is None:
                                s1, s2, s_ent = p.copy(), p ** 2, ent
                            else:
                                s1 += p; s2 += p ** 2; s_ent += ent
                    mean_p = s1 / n_passes
                    entropy[row:r2, col:c2] = np.maximum(-np.sum(mean_p * np.log(mean_p + 1e-10), axis=0), 0.0)
                    # variance across passes, averaged over classes
                    epistemic[row:r2, col:c2] = np.maximum(s2 / n_passes - mean_p ** 2, 0).mean(axis=0)
                    aleatoric[row:r2, col:c2] = s_ent / n_passes
                    label[row:r2, col:c2] = np.argmax(mean_p, axis=0)
        finally:
            self.model.eval()

        result = {
            "success": True,
            "image_size": (H, W),
            "n_mc_passes": n_passes,
            "n_dropout_layers": n_dropout,
            "mean_entropy": float(entropy.mean()),
            "mean_epistemic": float(epistemic.mean()),
            "mean_aleatoric": float(aleatoric.mean()),
            "high_uncertainty_fraction": float((entropy > 0.5).mean()),
        }
        if warning:
            result["warning"] = warning

        if output_path:
            from pathlib import Path
            op = Path(output_path)
            op.parent.mkdir(parents=True, exist_ok=True)

            # Save uncertainty raster: label + entropy always, epistemic/
            # aleatoric bands included per the save_epistemic/save_aleatoric flags.
            bands = [label.astype(np.float32), entropy]
            band_names = ["label", "entropy"]
            if save_epistemic:
                bands.append(epistemic)
                band_names.append("epistemic")
            if save_aleatoric:
                bands.append(aleatoric)
                band_names.append("aleatoric")

            out_profile = profile.copy()
            out_profile.update(count=len(bands), dtype="float32", compress="lzw")
            for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                out_profile.pop(_k, None)
            with rasterio.open(str(op), "w", **out_profile) as dst:
                dst.write(np.stack(bands))
                dst.update_tags(bands="|".join(band_names), n_mc_passes=str(n_passes))
                for i, n in enumerate(band_names, start=1):
                    dst.set_band_description(i, n)
            result["output_path"] = str(op)

        return result
