"""
GradCAM and GradCAM++ for geospatial model explainability (G6).
Produces class activation maps showing which regions influenced the prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _save_overlay(image_path: str, saliency: Any, output_path: str, *, alpha: float = 0.6,
                  colormap: str = "jet", title: str = "Saliency") -> None:
    """Three-panel figure (input, map, overlay) shared by GradCAM and attention maps.

    Reads the first three bands of ``image_path`` (one band is shown as grey)
    and stretches them for display; the map is resized to the image size.
    """
    import matplotlib
    import numpy as np
    import rasterio
    from PIL import Image as PILImage

    import matplotlib.pyplot as plt

    with rasterio.open(image_path) as src:
        n = min(src.count, 3)
        bands = src.read(list(range(1, n + 1))).astype(float)
    if n == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif n == 2:
        bands = np.concatenate([bands, bands[:1]], axis=0)
    rgb = bands[:3].transpose(1, 2, 0)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)

    sal = np.asarray(saliency, dtype=np.float32)
    if sal.ndim != 2:
        raise ValueError(f"saliency must be 2-D, got shape {sal.shape}")
    sal_up = np.array(PILImage.fromarray(sal).resize((rgb.shape[1], rgb.shape[0]), PILImage.BILINEAR))

    cmap = matplotlib.colormaps[colormap]
    blended = (1 - alpha) * rgb + alpha * cmap(sal_up)[:, :, :3]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(sal_up, cmap=colormap); axes[1].set_title(title); axes[1].axis("off")
    axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for geospatial models.

    Produces saliency maps showing which image regions most influenced
    the model's prediction — crucial for validating that models are
    detecting real features rather than spurious correlations.

    Works with classification models (output ``(B, classes)``) and
    segmentation models (output ``(B, classes, H, W)``). For segmentation
    the explained score is the class logit **summed over the whole chip**,
    so the map answers "which regions drive this class anywhere in the chip".

    Example::

        cam = GradCAM(model, target_layer="encoder.layer4")
        saliency = cam.explain(image_tensor, class_idx=1)
        cam.visualise(image_path, saliency, "./results/gradcam_buildings.png")
    """

    def __init__(self, model: Any, target_layer: str | None = None,
                 device: str | None = None) -> None:
        self.model = model
        self.target_layer_name = target_layer
        self.device = device or self._auto_device()
        self._gradients: Any | None = None
        self._activations: Any | None = None
        self._hooks: list = []

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _find_target_layer(self) -> Any:
        """The layer named ``target_layer``, else the last convolution."""
        import torch.nn as nn
        convs = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
        if not convs:
            raise ValueError("No Conv2d layers found. Specify target_layer manually.")
        if self.target_layer_name is None:
            return convs[-1][1]
        for name, layer in convs:
            if name == self.target_layer_name:
                return layer
        raise ValueError(
            f"target_layer {self.target_layer_name!r} is not a convolution layer of this "
            f"model. Convolution layers: {[n for n, _ in convs]}"
        )

    def _register_hooks(self, layer: Any) -> None:
        """Capture the layer's output and the gradient flowing back into it.

        The gradient is caught with a tensor hook on the layer's output rather
        than ``register_full_backward_hook``, which fails on models whose
        activation follows the layer in-place (e.g. torchvision ResNets).
        """
        self._gradients = None
        self._activations = None

        def _forward_hook(_, __, output):
            self._activations = output.detach()
            if output.requires_grad:
                output.register_hook(lambda g: setattr(self, "_gradients", g.detach()))

        self._hooks = [layer.register_forward_hook(_forward_hook)]

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _prepare(self, image: Any):
        import numpy as np
        import torch
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.to(self.device)

    def _backprop(self, image: Any, class_idx: int | None):
        """Forward + backward for one class; returns the target layer's (gradients, activations)."""
        self.model = self.model.to(self.device).eval()
        target_layer = self._find_target_layer()
        self._register_hooks(target_layer)
        try:
            self.model.zero_grad()
            logits = self.model(image)
            if logits.ndim == 4:                       # segmentation: (B, classes, H, W)
                if class_idx is None:
                    class_idx = int(logits[0].mean(dim=(-2, -1)).argmax().item())
                score = logits[:, class_idx].sum()
            else:                                      # classification: (B, classes)
                if class_idx is None:
                    class_idx = int(logits[0].argmax().item())
                score = logits[0, class_idx]
            score.backward()
        finally:
            self._remove_hooks()
        return self._gradients, self._activations

    def _weights_and_cam(self, grads: Any, acts: Any) -> Any:
        """Channel weights from the gradients, then the weighted sum of activations."""
        weights = grads.mean(dim=(-2, -1), keepdim=True)
        return (weights * acts).sum(dim=1)[0]

    def explain(
        self,
        image: Any,
        class_idx: int | None = None,
        normalize: bool = True,
    ) -> Any:
        """Compute the saliency map for an input image.

        Args:
            image: Tensor or array ``(C, H, W)`` or ``(1, C, H, W)``
            class_idx: Target class to explain (default: the class with the
                highest logit, or highest mean logit for segmentation models)
            normalize: Scale the output to ``[0, 1]``

        Returns:
            Saliency map as a numpy array ``(H, W)`` at the input size
        """
        try:
            import numpy as np
            import torch
        except ImportError:
            raise ImportError("torch required")

        image = self._prepare(image)
        grads, acts = self._backprop(image, class_idx)

        if grads is None or acts is None:
            logger.warning("GradCAM hooks did not fire. Check target_layer.")
            return np.zeros((image.shape[-2], image.shape[-1]))

        cam = torch.relu(self._weights_and_cam(grads, acts)).cpu().numpy()

        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray(cam.astype(np.float32))
        cam_up = cam_pil.resize((image.shape[-1], image.shape[-2]), PILImage.BILINEAR)
        cam_np = np.array(cam_up)

        if normalize and cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        return cam_np

    def batch_explain(
        self,
        image_path: str,
        output_path: str,
        class_idx: int = 1,
        chip_size: int = 512,
        overlap: int = 64,
    ) -> dict[str, Any]:
        """Apply GradCAM across a full GeoTIFF via tiled inference.

        Every chip is min-max normalised per band and its map is scaled to
        ``[0, 1]`` on its own, so values are comparable inside a chip but not
        strictly across chips. Overlapping chips are averaged. The whole image
        is read into memory. Returns ``{"success": False, "error": ...}``
        instead of raising for a missing dependency or an unusable tiling.
        """
        try:
            import numpy as np
            import rasterio
            import torch  # noqa: F401
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        stride = chip_size - overlap
        if stride <= 0:
            return {"success": False,
                    "error": f"chip_size ({chip_size}) must be greater than overlap ({overlap})"}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            H, W = src.height, src.width
            image = src.read().astype(np.float32)

        saliency_map = np.zeros((H, W), dtype=np.float32)
        count_map    = np.zeros((H, W), dtype=np.float32)

        n_chips = 0
        for row in range(0, H, stride):
            for col in range(0, W, stride):
                r2, c2 = min(row+chip_size, H), min(col+chip_size, W)
                chip = image[:, row:r2, col:c2]
                # Normalise
                chip_norm = chip.copy()
                for b in range(chip_norm.shape[0]):
                    mn, mx = chip_norm[b].min(), chip_norm[b].max()
                    chip_norm[b] = (chip_norm[b] - mn) / (mx - mn + 1e-8)
                cam = self.explain(chip_norm, class_idx=class_idx)
                actual_h, actual_w = r2-row, c2-col
                saliency_map[row:r2, col:c2] += cam[:actual_h, :actual_w]
                count_map[row:r2, col:c2]    += 1.0
                n_chips += 1

        saliency_map /= np.maximum(count_map, 1e-10)

        profile.update(count=1, dtype="float32", compress="lzw")
        for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
            profile.pop(_k, None)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(saliency_map[np.newaxis])
            dst.update_tags(method=type(self).__name__, class_idx=str(class_idx))

        return {"success": True, "output_path": output_path, "image_size": (H, W),
                "n_chips": n_chips}

    def visualise(
        self,
        image_path: str,
        saliency: Any,
        output_path: str,
        alpha: float = 0.6,
        colormap: str = "jet",
    ) -> None:
        """Save a PNG with the input, the saliency map and an overlay.

        ``image_path`` is read for display only (first three bands; a single
        band is shown as grey) and does not have to be the exact chip that
        was explained: the map is resized to the image. Logs a warning
        instead of failing when matplotlib is missing.
        """
        try:
            _save_overlay(image_path, saliency, output_path, alpha=alpha,
                          colormap=colormap, title=f"{type(self).__name__} saliency")
            logger.info("%s visualisation saved → %s", type(self).__name__, output_path)
        except ImportError as exc:
            logger.warning("matplotlib required for visualisation: %s", exc)


class GradCAMPlusPlus(GradCAM):
    """GradCAM++ — Chattopadhay et al. (2018), "Grad-CAM++: Generalized
    Gradient-Based Visual Explanations for Deep Convolutional Networks".

    Unlike vanilla GradCAM (which weights each activation channel by the
    plain average gradient), GradCAM++ computes per-pixel weighting
    coefficients (alpha) derived from higher-order gradient terms — this
    gives meaningfully better localisation when a class appears multiple
    times in one image (e.g. several buildings in a chip), which is
    exactly the geospatial use case this matters for.
    """

    def _weights_and_cam(self, grads: Any, acts: Any) -> Any:
        import torch
        import torch.nn.functional as F

        # GradCAM++ alpha coefficients (standard first-order-gradient
        # practical formulation — see class docstring):
        #   alpha = grad^2 / (2*grad^2 + sum(activations) * grad^3)
        grads_2 = grads ** 2
        grads_3 = grads ** 3
        sum_activations = acts.sum(dim=(-2, -1), keepdim=True)
        eps = 1e-8
        alpha_denom = grads_2 * 2.0 + sum_activations * grads_3
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.full_like(alpha_denom, eps),
        )
        alphas = grads_2 / alpha_denom

        weights = (alphas * F.relu(grads)).sum(dim=(-2, -1), keepdim=True)
        return (weights * acts).sum(dim=1)[0]
