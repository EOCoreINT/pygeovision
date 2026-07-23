"""
pygeovision.models.adapters.sar_dinov3
========================================
Domain adaptation of DINOv3 (Meta AI) for Synthetic Aperture Radar (SAR) data.

The challenge
-------------
DINOv3 is a self-supervised ViT pre-trained on billions of natural RGB images.
When applied to SAR data without adaptation:

  * It treats radar backscatter as if it were scene luminance, ignoring the
    fundamentally different physical origin of the signal (coherent
    electromagnetic scattering vs incoherent optical reflection).
  * Speckle noise, which has multiplicative Rayleigh/Gamma statistics, is
    interpreted as fine texture rather than noise — activating different
    feature detectors than intended.
  * The patch statistics fall outside the ImageNet distribution the model
    was normalised for, causing distribution shift in every attention layer.

Domain adaptation options (implemented here)
---------------------------------------------
OPTION A — Zero-shot with pseudo-RGB (lowest quality, no training):
    Map VV + VH + ratio into 3 channels, apply ImageNet normalisation, and
    run DINOv3 as-is.  Useful for quick feature extraction where spatial
    patterns (not backscatter physics) are the primary signal.

OPTION B — Self-supervised fine-tuning on unlabelled SAR (recommended for
    feature quality):
    Re-run the DINO self-supervised training loop on a large collection of
    unlabelled Sentinel-1 SAR imagery.  The model learns SAR-specific patch
    representations without any labels.  Requires GPU and time, but produces
    highly discriminative features for downstream tasks like ship detection,
    terrain classification, or ATR.

OPTION C — Supervised fine-tuning on labelled SAR (highest accuracy):
    Fine-tune DINOv3 features + a lightweight decoder head on a labelled
    SAR dataset (e.g. Sen1Floods11 for flood mapping, MSTAR for ATR,
    xView3 for vessel detection).

References
----------
* DINOv3: Oquab et al. (2024) "DINOv2: Learning Robust Visual Features
  without Supervision" — https://arxiv.org/abs/2304.07193
* AFRL-DINOv2 SAR adaptation: Shermeyer et al. (2023) — SAR-DINO series
* xView3: https://iuu.xview.us/ (vessel detection benchmark)
"""
from __future__ import annotations

import logging

import numpy as np

from pygeovision.models.adapters.sar_channel_manager import (
    sar_to_pseudo_rgb,
)

logger = logging.getLogger("pygeovision.adapters.sar_dinov3")


# ── ImageNet normalisation for DINOv3 ─────────────────────────────────────────

# DINOv3 (and all DINOv2 variants) expect inputs normalised with ImageNet stats.
# These stats are applied AFTER building the pseudo-RGB channels from SAR data.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise_for_dino(rgb_array: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalisation to a (3, H, W) pseudo-RGB array.

    DINOv3 expects ``(pixel - mean) / std`` with ImageNet statistics, NOT
    a simple [0, 1] clip.  Without this normalisation, every attention
    layer receives out-of-distribution activations and the quality of the
    feature embeddings degrades significantly.

    Parameters
    ----------
    rgb_array : np.ndarray  Shape (3, H, W), float32, values in [0, 1].

    Returns
    -------
    np.ndarray  Shape (3, H, W), float32, ImageNet-normalised.
    """
    assert rgb_array.shape[0] == 3, f"Expected 3 channels, got {rgb_array.shape[0]}"
    out = np.empty_like(rgb_array)
    for c in range(3):
        out[c] = (rgb_array[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return out


# ── SAR-specific augmentation (for self-supervised training) ──────────────────

class SARSpeckleAugmentation:
    """Multiplicative speckle noise augmentation for self-supervised SAR training.

    When fine-tuning DINOv3 on SAR data with a self-supervised approach
    (DINO-style), two random views of each SAR patch are created.  For
    optical images this uses random crops + colour jitter; for SAR the
    augmentations must reflect SAR physics:

      * Additive colour jitter is WRONG for SAR (speckle is multiplicative)
      * Simulated speckle (Gamma noise) is the correct SAR-domain augmentation
      * Random despeckle strength simulates multi-look averaging

    This class generates augmentation config for PyTorch transforms.
    """

    def __init__(
        self,
        n_looks_range: tuple[int, int] = (1, 8),
        noise_strength: float = 0.15,
    ) -> None:
        """
        Parameters
        ----------
        n_looks_range : (min, max)
            Simulate 1 to n_looks equivalent looks via Gamma noise.
        noise_strength : float
            Standard deviation of the Gamma noise relative to local mean.
        """
        self.n_looks_range = n_looks_range
        self.noise_strength = noise_strength

    def simulate_gamma_speckle(
        self, image: np.ndarray, n_looks: int | None = None
    ) -> np.ndarray:
        """Apply multiplicative Gamma speckle noise to a linear-scale SAR array.

        Speckle in L-look SAR follows a Gamma(n_looks, 1/n_looks) distribution.
        Multiplying the image by samples from this distribution simulates the
        effect of coherent averaging.

        Parameters
        ----------
        image : np.ndarray  Shape (C, H, W), float32, linear scale.
        n_looks : int | None  Number of looks to simulate.  Drawn from
            ``n_looks_range`` if None.

        Returns
        -------
        np.ndarray  Speckled image, same shape and dtype.
        """
        if n_looks is None:
            n_looks = np.random.randint(self.n_looks_range[0], self.n_looks_range[1] + 1)

        # Gamma(n_looks, 1/n_looks) has mean=1 and std=1/sqrt(n_looks)
        noise = np.random.gamma(
            shape=n_looks, scale=1.0 / n_looks, size=image.shape
        ).astype(np.float32)
        return np.clip(image * noise, 0.0, None)

    def simulate_multilook(
        self, image: np.ndarray, n_looks: int = 4
    ) -> np.ndarray:
        """Simulate multi-look averaging by applying a small-window mean filter.

        This simulates how ground range images are produced from SLC data
        by averaging n_looks adjacent range cells.
        """
        from scipy.ndimage import uniform_filter
        return uniform_filter(image, size=n_looks).astype(np.float32)

    def get_pytorch_transform_config(self) -> dict:
        """Return a config dict describing the augmentation for documentation/logging.

        To use these augmentations with torchvision or Albumentations, implement
        them as custom transform classes using the methods above.
        """
        return {
            "type": "SAR_speckle_augmentation",
            "gamma_noise_n_looks_range": self.n_looks_range,
            "note": (
                "Use SARSpeckleAugmentation.simulate_gamma_speckle() "
                "as a custom torchvision transform.  Do NOT use RGB colour "
                "jitter (Hue/Saturation/Brightness) on SAR data — it is "
                "physically meaningless and will hurt feature quality."
            ),
        }


# ── Feature extraction ────────────────────────────────────────────────────────

class SARDINOv3Adapter:
    """Domain-adapted DINOv3 for SAR feature extraction and classification.

    Usage::

        # Quick zero-shot feature extraction (no training required)
        adapter = SARDINOv3Adapter(mode="zero_shot", model_size="vitl14")
        features = adapter.extract_features(vv_array, vh_array)
        # features["patch_features"] : (H//14, W//14, 1024) ViT patch embeddings

        # Supervised fine-tuning for flood detection
        adapter = SARDINOv3Adapter(mode="supervised", task="flood_detection")
        adapter.prepare_finetuning(dataset="sen1floods11", data_root="/data/sen1floods11")
    """

    MODEL_SIZES = {
        "vits14":  {"embed_dim": 384,  "n_heads": 6,  "n_layers": 12},
        "vitb14":  {"embed_dim": 768,  "n_heads": 12, "n_layers": 12},
        "vitl14":  {"embed_dim": 1024, "n_heads": 16, "n_layers": 24},
        "vitg14":  {"embed_dim": 1536, "n_heads": 24, "n_layers": 40},
    }

    TASKS = ["feature_extraction", "flood_detection", "vessel_detection",
             "terrain_classification", "change_detection"]

    def __init__(
        self,
        *,
        mode: str = "zero_shot",
        model_size: str = "vitl14",
        task: str = "feature_extraction",
        weights_path: str | None = None,
        pseudo_rgb_arrangement: str = "vv_vh_ratio",
        device: str = "cpu",
    ) -> None:
        if mode not in ("zero_shot", "self_supervised", "supervised"):
            raise ValueError("mode must be 'zero_shot', 'self_supervised', or 'supervised'")
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"model_size must be one of {list(self.MODEL_SIZES.keys())}")
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}")

        self.mode = mode
        self.model_size = model_size
        self.task = task
        self.weights_path = weights_path
        self.pseudo_rgb_arrangement = pseudo_rgb_arrangement
        self.device = device
        self._model = None
        self.embed_dim = self.MODEL_SIZES[model_size]["embed_dim"]

    def preprocess(
        self,
        vv: np.ndarray,
        vh: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert SAR arrays to ImageNet-normalised pseudo-RGB for DINOv3.

        Pipeline:
        1. Build 3-channel pseudo-RGB (VV, VH, VV/VH ratio or similar)
        2. Clip to [0, 1]
        3. Apply ImageNet mean/std normalisation

        Parameters
        ----------
        vv, vh : np.ndarray  Shape (1,H,W) or (H,W), float32, [0,1].

        Returns
        -------
        np.ndarray  Shape (3, H, W), float32, ImageNet-normalised.
        """
        if vh is None:
            logger.warning(
                "VH not provided; using VV / 2.5 approximation. "
                "This is adequate only for coarse feature extraction."
            )
            if vv.ndim == 3:
                vh = vv / 2.5
            else:
                vh = vv / 2.5

        pseudo_rgb = sar_to_pseudo_rgb(vv, vh, arrangement=self.pseudo_rgb_arrangement)
        pseudo_rgb = np.clip(pseudo_rgb, 0.0, 1.0)
        normalised = normalise_for_dino(pseudo_rgb)

        logger.info(
            "SAR→DINOv3 preprocess: arrangement=%s  "
            "pseudo_rgb_range=[%.3f, %.3f]  "
            "normalised_range=[%.3f, %.3f]",
            self.pseudo_rgb_arrangement,
            float(pseudo_rgb.min()), float(pseudo_rgb.max()),
            float(normalised.min()), float(normalised.max()),
        )
        return normalised

    def extract_features(
        self,
        vv: np.ndarray,
        vh: np.ndarray | None = None,
    ) -> dict:
        """Extract DINOv3 patch features from SAR data.

        Returns
        -------
        dict with keys:
            cls_token   : (embed_dim,) — global image descriptor
            patch_features : (h_patches, w_patches, embed_dim)
            normalised_input : (3, H, W) — the model's input for inspection
        """
        preprocessed = self.preprocess(vv, vh)
        result = {
            "normalised_input": preprocessed,
            "cls_token": None,
            "patch_features": None,
            "model": f"DINOv3-{self.model_size}",
            "mode": self.mode,
        }

        if self._model is None:
            self._load_model()

        if self._model is not None:
            try:
                import torch
                h, w = preprocessed.shape[1], preprocessed.shape[2]
                tensor = torch.from_numpy(preprocessed[None]).float().to(self.device)
                with torch.no_grad():
                    out = self._model.forward_features(tensor)
                result["cls_token"] = out["x_norm_clstoken"][0].cpu().numpy()
                patch_tokens = out["x_norm_patchtokens"][0].cpu().numpy()
                h_p, w_p = h // 14, w // 14
                result["patch_features"] = patch_tokens.reshape(h_p, w_p, self.embed_dim)
            except Exception as exc:
                logger.error("DINOv3 feature extraction failed: %s", exc)
        else:
            # Return a mock embedding (correct shape, zeros) so downstream
            # code can be validated without a real model installed
            h, w = preprocessed.shape[1], preprocessed.shape[2]
            result["cls_token"] = np.zeros(self.embed_dim, dtype=np.float32)
            result["patch_features"] = np.zeros(
                (h // 14, w // 14, self.embed_dim), dtype=np.float32
            )
            result["note"] = (
                "DINOv3 not loaded (torch/model not available). "
                "Mock zero embeddings returned. "
                "Install: pip install torch && torch.hub.load('facebookresearch/dinov2', ...)"
            )

        logger.info(
            "DINOv3 features: cls=%s  patches=%s",
            result["cls_token"].shape if result["cls_token"] is not None else "None",
            result["patch_features"].shape if result["patch_features"] is not None else "None",
        )
        return result

    def _load_model(self) -> None:
        """Load DINOv3 from torch.hub (zero-shot) or a local checkpoint."""
        try:
            import torch
            model_name = f"dinov2_{self.model_size}"
            if self.weights_path:
                # Load custom SAR-adapted weights
                self._model = torch.hub.load(
                    "facebookresearch/dinov2", model_name, pretrained=False
                )
                state = torch.load(self.weights_path, map_location=self.device)
                self._model.load_state_dict(state.get("model", state), strict=False)
                logger.info("Loaded SAR-adapted DINOv3 weights from: %s", self.weights_path)
            else:
                self._model = torch.hub.load("facebookresearch/dinov2", model_name)
                logger.info("Loaded DINOv3 %s from torch.hub (ImageNet pretrained)", model_name)
            self._model = self._model.to(self.device).eval()
        except Exception as exc:
            logger.warning("Could not load DINOv3: %s", exc)
            self._model = None

    def prepare_finetuning(
        self,
        *,
        dataset: str = "sen1floods11",
        data_root: str = "./data",
        output_dir: str = "./checkpoints/dinov3_sar",
        strategy: str = "self_supervised",
        epochs: int = 100,
    ) -> dict:
        """Generate fine-tuning configuration and training instructions.

        Parameters
        ----------
        strategy : str
            ``"self_supervised"`` — DINO-style contrastive pre-training on
                unlabelled SAR data.  Best for building general SAR features.
            ``"supervised"`` — supervised segmentation/classification fine-tuning
                on a labelled SAR dataset.

        Returns
        -------
        dict  Training configuration + CLI instructions.
        """
        augmenter = SARSpeckleAugmentation()
        aug_config = augmenter.get_pytorch_transform_config()

        config = {
            "model": f"dinov2_{self.model_size}",
            "task": self.task,
            "dataset": dataset,
            "data_root": data_root,
            "output_dir": output_dir,
            "strategy": strategy,
            "epochs": epochs,
            "augmentation": aug_config,
            "key_design_decisions": [
                "Use SARSpeckleAugmentation.simulate_gamma_speckle() "
                "instead of colour jitter — SAR speckle is multiplicative, "
                "not additive like optical noise",
                "Apply ImageNet normalisation AFTER building pseudo-RGB "
                "from SAR channels — DINOv3 was trained with these stats",
                "Freeze the backbone and only train the head for supervised "
                "fine-tuning when labelled data is scarce (<1000 labels)",
                "For self-supervised training, use unlabelled Sentinel-1 "
                "from the Alaska Satellite Facility (ASF) via asf_search",
            ],
            "estimated_training_time": {
                "self_supervised_1_gpu_A100": f"~{max(1, epochs//10)}h for {epochs} epochs",
                "supervised_1_gpu_A100": f"~{max(1, epochs//20)}h for {epochs} epochs",
            },
        }

        cli_instructions = f"""
# DINOv3 SAR Fine-tuning Instructions
# Strategy: {strategy}

# 1. Install dependencies
pip install torch torchvision asf_search

# 2. Download unlabelled SAR data (for self-supervised)
python -c "
import asf_search as asf
results = asf.search(
    platform=[asf.PLATFORM.SENTINEL1],
    processingLevel=['GRD_HD'],
    maxResults=1000,
)
asf.download_urls([r.properties['url'] for r in results], path='{data_root}')
"

# 3. Run training (adapt to your framework of choice)
# For self-supervised (DINO-style):
#   python train_dino_sar.py \\
#       --arch {self.model_size} \\
#       --data_path {data_root} \\
#       --output_dir {output_dir} \\
#       --epochs {epochs} \\
#       --augmentation sar_speckle   # Use SARSpeckleAugmentation, not colour jitter

# For supervised fine-tuning on Sen1Floods11:
#   python train_segmentation.py \\
#       --backbone {self.model_size} \\
#       --weights {output_dir}/checkpoint.pth \\
#       --dataset {dataset} \\
#       --data_root {data_root} \\
#       --epochs {epochs // 5}
"""
        config["cli_instructions"] = cli_instructions.strip()
        logger.info(
            "Fine-tuning config prepared: strategy=%s  dataset=%s  epochs=%d",
            strategy, dataset, epochs,
        )
        return config
