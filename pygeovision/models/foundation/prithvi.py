"""
Prithvi-EO -- NASA/IBM geospatial foundation models, integrated for PyGeoVision.

Models (all take **six** HLS bands: Blue, Green, Red, narrow-NIR, SWIR-1, SWIR-2):
  prithvi_eo_1_0       100M  Prithvi-EO-1.0-100M (3 frames, contiguous US)
  prithvi_eo_2_0       300M  Prithvi-EO-2.0-300M (4 frames, global)
  prithvi_eo_2_0_600m  600M  Prithvi-EO-2.0-600M (4 frames, global)

How the real model is loaded
----------------------------
The Prithvi repositories are **not** ``transformers`` models: ``config.json`` has no
``model_type`` and there is no ``auto_map``, so ``AutoModel`` cannot load them.  Each repo
ships ``prithvi_mae.py`` (the architecture), ``config.json`` and a ``.pt`` checkpoint.  This
module downloads those three files at a **pinned commit** (``spec["revision"]``), imports
``prithvi_mae.py`` and loads the checkpoint with ``torch.load(weights_only=True)``.  The
checkpoint is checked key-by-key: a layer that the checkpoint does not fill raises
:class:`~pygeovision.core.exceptions.ModelLoadError` instead of staying randomly initialised.

.. warning::
   Importing ``prithvi_mae.py`` executes code from the Hugging Face repository.  The default
   commits were chosen when this module was written; read the diff before changing a
   ``revision``.  Needs ``einops`` and ``timm``.

Input preparation
-----------------
The pretrained weights expect **per-band z-scored HLS digital numbers** (reflectance x 10000),
using the mean/std stored in each repo's ``config.json``.  Dividing by 10000 and stopping
there -- what earlier versions did -- feeds the network inputs of the wrong scale.  The
:class:`PrithviBackbone` wrapper applies the standardisation itself; every public method here
takes surface **reflectance in [0, 1]**.

Task heads
----------
A backbone produces features, not land-cover maps.  Segmentation / regression heads must be
trained (see :func:`finetune_prithvi`) and handed to :class:`PrithviTasks` as ``head_weights``.
Without a trained head, :class:`PrithviTasks` falls back to documented spectral-index rules and
says so in ``result["method"]``; tasks that have no such rule (crop type, biomass) raise
``NotImplementedError`` rather than invent a result.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pygeovision.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class PrithviLoadError(ModelLoadError, RuntimeError):
    """Prithvi weights/architecture could not be loaded.

    Also a ``RuntimeError`` because earlier versions of :func:`load_prithvi_hf` raised that.
    """


# ── Model Registry ────────────────────────────────────────────────────────────
# ``revision`` pins the commit whose ``prithvi_mae.py`` is executed (see module docstring).

PRITHVI_MODELS: dict[str, dict] = {
    "prithvi_eo_1_0": {
        "params_m": 100,
        "hf_id":    "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        "revision": "f3a9ea7a1723621b0aeceb4de993093704d712c5",
        "checkpoint": "Prithvi_EO_V1_100M.pt",
        "coverage": "US",
        "resolution_m": 30,
        "n_bands":  6,
        "n_frames": 3,           # frames the model was pre-trained with
        "embed_dim": 768,
        "patch_size": 16,
        "temporal": True,
        "loadable": True,
        "description": "Prithvi-EO-1.0 MAE, trained on contiguous-US HLS imagery",
    },
    "prithvi_eo_2_0": {
        "params_m": 300,
        "hf_id":    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "revision": "9eb1b1102806593963daa333bcc491b1c6f8562f",
        "checkpoint": "Prithvi_EO_V2_300M.pt",
        "coverage": "Global",
        "resolution_m": 30,
        "n_bands":  6,
        "n_frames": 4,
        "embed_dim": 1024,
        "patch_size": 16,
        "temporal": True,
        "loadable": True,
        "description": "Prithvi-EO-2.0 300M, global HLS (this key used to claim 600M while loading the 300M repo)",
    },
    "prithvi_eo_2_0_600m": {
        "params_m": 600,
        "hf_id":    "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "revision": "2a5cf6cedea1c0b31734021e2a5ac90082825414",
        "checkpoint": "Prithvi_EO_V2_600M.pt",
        "coverage": "Global",
        "resolution_m": 30,
        "n_bands":  6,
        "n_frames": 4,
        "embed_dim": 1280,
        "patch_size": 14,
        "temporal": True,
        "loadable": True,
        "description": "Prithvi-EO-2.0 600M, global HLS",
    },
    "prithvi_eo_1_0_finetuned_burn": {
        "params_m": 100,
        "hf_id":    "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar",
        "task":     "burn_scar_segmentation",
        "loadable": False,
        "load_note": "fine-tuned downstream checkpoints are MMSegmentation/TerraTorch artefacts, not "
                     "backbone checkpoints; load them with TerraTorch",
        "description": "Prithvi-EO-1.0-100M fine-tuned for burn-scar mapping (reference only)",
    },
    "prithvi_eo_1_0_finetuned_flood": {
        "params_m": 100,
        "hf_id":    "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11",
        "task":     "flood_segmentation",
        "loadable": False,
        "load_note": "fine-tuned downstream checkpoints are MMSegmentation/TerraTorch artefacts, not "
                     "backbone checkpoints; load them with TerraTorch",
        "description": "Prithvi-EO-1.0-100M fine-tuned on Sen1Floods11 (reference only; this entry "
                       "used to point at the crop-classification repo)",
    },
}

# Per-band HLS statistics in digital numbers (reflectance x 10000), from the Prithvi-EO-2.0-300M
# config.  Used only by the random-init stand-in; real models read their own config.json.
HLS_MEAN_DN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
HLS_STD_DN = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)

# ── Band Mappings ────────────────────────────────────────────────────────────

# Sentinel-2 band name → Prithvi HLS position (Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5)
SENTINEL2_TO_PRITHVI = {
    "B02": 0,   # Blue  (10m)
    "B03": 1,   # Green (10m)
    "B04": 2,   # Red   (10m)
    "B08": 3,   # broad NIR (10m) -- stand-in for HLS narrow NIR; B8A is preferred when both exist
    "B11": 4,   # SWIR1 (20m)
    "B12": 5,   # SWIR2 (20m)
    # Red-edge bands: Prithvi takes six bands only, so these are ignored by map_bands(n_prithvi_bands=6)
    "B05": 6,   # Red Edge 1 (20m)
    "B06": 7,   # Red Edge 2 (20m)
    "B07": 8,   # Red Edge 3 (20m)
    "B8A": 3,   # narrow NIR (20m) -- what HLS (and so Prithvi) actually uses as its NIR band
}

# Landsat band name → Prithvi HLS position
LANDSAT_TO_PRITHVI = {
    "B2": 0,    # Blue  (30m)
    "B3": 1,    # Green (30m)
    "B4": 2,    # Red   (30m)
    "B5": 3,    # NIR   (30m)
    "B6": 4,    # SWIR1 (30m)
    "B7": 5,    # SWIR2 (30m)
}

# Canonical input order for each source when band_names not provided explicitly
SENTINEL2_CANONICAL = ["B02", "B03", "B04", "B08", "B11", "B12"]
LANDSAT_CANONICAL   = ["B2",  "B3",  "B4",  "B5",  "B6",  "B7" ]

# HLS surface reflectance scale factor
HLS_SCALE_FACTOR = 10000.0

# ESA WorldCover-aligned land cover classes (Prithvi-EO-2.0, 9 classes)
LAND_COVER_CLASSES = [
    "Tree cover",       # 0 — closed/open forest
    "Shrubland",        # 1 — shrubs <5m
    "Grassland",        # 2 — natural herbaceous
    "Cropland",         # 3 — annual cultivated land
    "Built-up",         # 4 — buildings, roads, hardscape
    "Bare / sparse",    # 5 — exposed soil, rock, desert
    "Snow / ice",       # 6 — permanent or seasonal snow
    "Water",            # 7 — rivers, lakes, ocean
    "Wetland",          # 8 — herbaceous wetland / mangrove
]

# ESA WorldCover colour palette (RGB) matching the 9-class scheme
LAND_COVER_PALETTE = [
    (0,   100,  0),    # 0 Tree cover — dark green
    (255, 187, 34),    # 1 Shrubland — yellow-orange
    (255, 255, 76),    # 2 Grassland — yellow
    (240, 150, 255),   # 3 Cropland — violet-pink
    (250,   0, 86),    # 4 Built-up — bright red
    (180, 180, 180),   # 5 Bare / sparse — grey
    (240, 240, 240),   # 6 Snow / ice — near white
    (0,   100, 200),   # 7 Water — blue
    (0,   150, 160),   # 8 Wetland — teal
]

# Crop type classes
CROP_CLASSES = [
    "corn", "soybeans", "cotton", "winter_wheat", "spring_wheat",
    "rice", "sorghum", "other_grains", "vegetables", "other",
]


def normalise_hls(data: np.ndarray) -> np.ndarray:
    """Normalise HLS surface reflectance integers to [0, 1].

    HLS/Sentinel-2 L2A values are stored as SR × 10000.
    Values outside [0, 10000] are clipped.
    """
    return np.clip(data.astype(np.float32) / HLS_SCALE_FACTOR, 0.0, 1.0)


def map_bands(
    data: np.ndarray,
    source: str = "sentinel2",
    n_prithvi_bands: int = 6,
    source_bands: list[str] | None = None,
) -> np.ndarray:
    """Reorder bands from satellite native order to Prithvi HLS order.

    Prithvi HLS order: [Blue, Green, Red, NIR, SWIR1, SWIR2]
    Indices:           [  0,     1,   2,   3,     4,     5  ]

    Args:
        data: ``(C, H, W)`` array in source satellite band order.
        source: ``"sentinel2"`` | ``"landsat"`` | ``"hls"`` (already ordered).
        n_prithvi_bands: Output bands (6 standard, 10 extended HLS-S2).
        source_bands: Explicit list of band names for each channel in ``data``
            (e.g. ``["B02","B03","B04","B08","B11","B12"]``).
            When ``None``, the canonical order for *source* is assumed.

    Returns:
        ``(n_prithvi_bands, H, W)`` reordered float32 array.
    """
    if source == "hls":
        # Already in Prithvi order — just take the first n bands
        return data[:n_prithvi_bands].astype(np.float32)

    mapping = SENTINEL2_TO_PRITHVI if source == "sentinel2" else LANDSAT_TO_PRITHVI
    H, W   = data.shape[1], data.shape[2]
    out    = np.zeros((n_prithvi_bands, H, W), dtype=np.float32)

    # Determine which name is at each source position
    if source_bands is not None:
        names = list(source_bands)
    else:
        # Assume canonical sensor order
        names = (SENTINEL2_CANONICAL if source == "sentinel2" else LANDSAT_CANONICAL)

    # HLS band 4 is *narrow* NIR (Sentinel-2 B8A).  If a stack has both B08 and B8A use B8A;
    # B08 alone is accepted as a (spectrally slightly different) stand-in.
    if source == "sentinel2" and "B08" in names and "B8A" in names:
        names = [("B08_unused" if n == "B08" else n) for n in names]
    elif source == "sentinel2" and "B08" in names and "B8A" not in names:
        logger.warning("map_bands: using broad-NIR B08 for Prithvi's narrow-NIR band (HLS uses B8A); "
                       "expect a small spectral mismatch.")

    for src_idx, band_name in enumerate(names):
        if src_idx >= data.shape[0]:
            break
        if band_name in mapping:
            prithvi_idx = mapping[band_name]
            if prithvi_idx < n_prithvi_bands:
                out[prithvi_idx] = data[src_idx].astype(np.float32)

    return out


def validate_prithvi_input(
    data: np.ndarray,
    source: str = "sentinel2",
    n_prithvi_bands: int = 6,
) -> dict[str, Any]:
    """Validate and report on Prithvi input data quality.

    Args:
        data: ``(C, H, W)`` array **already in HLS order and normalised to [0,1]**.
        source: Source sensor (for error messages).
        n_prithvi_bands: Expected number of bands.

    Returns:
        Dict with ``valid``, ``warnings``, ``errors``, ``stats``.
    """
    errors   = []
    warnings = []

    if data.ndim != 3:
        errors.append(f"Expected (C,H,W), got shape {data.shape}")
    else:
        C, H, W = data.shape
        if n_prithvi_bands > C:
            errors.append(f"Only {C} bands; Prithvi needs {n_prithvi_bands}")
        if data.dtype != np.float32:
            warnings.append(f"dtype={data.dtype}; will cast to float32")
        _vmin, vmax = float(data.min()), float(data.max())
        if vmax > 5.0:
            errors.append(
                f"Values up to {vmax:.0f} look like raw DN integers. "
                f"Divide by {HLS_SCALE_FACTOR} before calling Prithvi.")
        elif vmax > 1.5:
            warnings.append(
                f"Values reach {vmax:.3f} > 1.5 — are you sure data is normalised?")
        nan_count = int(np.isnan(data).sum())
        if nan_count:
            warnings.append(f"{nan_count:,} NaN pixels — will be set to 0 before inference")

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
        "stats":    {
            "min":  float(data.min()) if data.size else 0,
            "max":  float(data.max()) if data.size else 0,
            "mean": float(data.mean()) if data.size else 0,
            "nan":  int(np.isnan(data).sum()) if data.size else 0,
        },
    }


# ── Spectral rule-based predictions (used when backbone weights unavailable) ─

def _spectral_land_cover(
    data_hls: np.ndarray,
) -> np.ndarray:
    """Physics-informed land cover classification from HLS reflectance.

    Produces ESA WorldCover-compatible 9-class map from spectral indices.
    Priority order ensures physically valid assignments.

    Args:
        data_hls: ``(6, H, W)`` float32 in HLS order [Blue,Green,Red,NIR,SWIR1,SWIR2],
            values normalised to ``[0, 1]``.

    Returns:
        ``(H, W)`` uint8 class map (0–8, matching ``LAND_COVER_CLASSES``).
    """
    if data_hls.shape[0] < 6:
        raise ValueError(f"Need 6 HLS bands, got {data_hls.shape[0]}")

    B, G, R, NIR, SWIR1, SWIR2 = [data_hls[i].astype(np.float32) for i in range(6)]
    eps = 1e-8

    # Spectral indices
    NDVI  = (NIR  - R)     / (NIR  + R     + eps)  # vegetation
    (G    - NIR)   / (G    + NIR   + eps)  # water
    MNDWI = (G    - SWIR1) / (G    + SWIR1 + eps)  # water (modified, less vegetation interference)
    NDBI  = (SWIR1 - NIR)  / (SWIR1 + NIR  + eps)  # built-up / bare
    NDSI  = (G    - SWIR1) / (G    + SWIR1 + eps)  # snow (same formula as MNDWI — use NIR constraint)
    BSI   = ((SWIR1 + R) - (NIR + B)) / ((SWIR1 + R) + (NIR + B) + eps)  # bare soil

    H, W = B.shape
    cls  = np.full((H, W), 5, dtype=np.uint8)   # default: Bare / sparse

    # Assignment in priority order — later assignments override earlier ones
    # 5. Bare / sparse  (default above)
    # 2. Grassland
    cls[(NDVI > 0.08) & (NDVI <= 0.40) & (BSI < 0.10) & (SWIR1 < 0.20)]  = 2
    # 1. Shrubland
    cls[(NDVI > 0.30) & (NDVI <= 0.45) & (NIR < 0.45) & (SWIR1 >= 0.08)]  = 1
    # 3. Cropland — cultivated but moderate NDVI and flat terrain proxy
    cls[(NDVI > 0.18) & (NDVI <= 0.55) & (BSI < 0.0) & (SWIR1 > 0.05)] = 3
    # 0. Tree cover — dense vegetation
    cls[NDVI > 0.45]                                     = 0
    # 4. Built-up — high SWIR1, reliably positive NDBI
    #    NDBI > 0.10 avoids false-positives on bare soil (NDBI 0.00-0.08)
    cls[(NDBI > 0.10) & (NDVI < 0.20)]                  = 4
    # 8. Wetland — marginal MNDWI with active vegetation
    cls[(MNDWI > 0.05) & (MNDWI <= 0.30) & (NDVI > 0.08)] = 8
    # 7. Water — strong MNDWI AND low NIR (water absorbs NIR aggressively)
    cls[(MNDWI > 0.25) & (NIR < 0.15)]                  = 7
    # 7. Deep / turbid water — very strong MNDWI, no NIR guard
    cls[(MNDWI > 0.45) & (NIR < 0.25)]                  = 7
    # 6. Snow / ice — LAST, overrides water when NIR is non-trivial.
    #    Snow and water share the MNDWI/NDSI formula; the discriminator
    #    is NIR reflectance: snow 0.15-0.45, liquid water < 0.10.
    cls[(NDSI > 0.40) & (NIR >= 0.15)]                  = 6

    return cls


def _spectral_crop_mapping(data_hls: np.ndarray) -> np.ndarray:
    """Spectral crop type proxy classification from HLS bands.

    Uses NIR/SWIR ratios and vegetation density as proxies. Produces
    plausible spatial patterns but should be replaced with a fine-tuned
    model for production use.
    """
    B, G, R, NIR, SWIR1, SWIR2 = [data_hls[i].astype(np.float32) for i in range(6)]
    eps = 1e-8

    NDVI   = (NIR - R)     / (NIR + R     + eps)
    LSWI   = (NIR - SWIR1) / (NIR + SWIR1 + eps)   # Leaf water
    (NIR - R)     / (NIR + R     + eps)    # proxy for red-edge ratio
    SWIR_R = SWIR2 / (SWIR1 + eps)                  # SWIR ratio

    H, W  = B.shape
    cls   = np.full((H, W), 9, dtype=np.uint8)   # default: other

    # 0. Corn — high NDVI, high LSWI mid-season
    cls[(NDVI > 0.60) & (LSWI > 0.25)]                  = 0
    # 1. Soybeans — high NDVI, moderate SWIR
    cls[(NDVI > 0.50) & (NDVI <= 0.75) & (LSWI < 0.25)] = 1
    # 2. Cotton — moderate NDVI, high SWIR2
    cls[(NDVI > 0.30) & (SWIR_R > 0.85)]                = 2
    # 3. Winter wheat — moderate NDVI, low LSWI
    cls[(NDVI > 0.25) & (NDVI <= 0.55) & (LSWI < 0.10)] = 3
    # 5. Rice — very high LSWI (flooded paddy)
    cls[(LSWI > 0.35) & (NDVI > 0.20)]                  = 5
    # 8. Vegetables — high NIR, low SWIR
    cls[(NIR > 0.45) & (SWIR1 < 0.12) & (NDVI > 0.35)]  = 8

    return cls


def _spectral_flood(data_hls: np.ndarray) -> np.ndarray:
    """Binary flood mask from MNDWI on HLS bands.

    Uses Modified NDWI (MNDWI > 0.15 = water / flooded) with a
    SAR-independent optical flood proxy.
    """
    G, SWIR1 = data_hls[1].astype(np.float32), data_hls[4].astype(np.float32)
    eps    = 1e-8
    MNDWI  = (G - SWIR1) / (G + SWIR1 + eps)
    # Flood = water present AND some reflectance (not cloud shadow)
    flood  = (MNDWI > 0.15) & (G > 0.02)
    return flood.astype(np.uint8)


def _spectral_burn_scar(data_hls: np.ndarray) -> np.ndarray:
    """Binary burn scar mask from NBR on HLS bands."""
    NIR, SWIR2 = data_hls[3].astype(np.float32), data_hls[5].astype(np.float32)
    eps  = 1e-8
    NBR  = (NIR - SWIR2) / (NIR + SWIR2 + eps)
    # Low NBR (<0 or strongly negative) = burned area
    burned = (NBR < -0.10)
    return burned.astype(np.uint8)






# ── Input standardisation ─────────────────────────────────────────────────────

def standardise_hls(
    reflectance: np.ndarray,
    mean: tuple[float, ...] | list[float] = HLS_MEAN_DN,
    std: tuple[float, ...] | list[float] = HLS_STD_DN,
) -> np.ndarray:
    """Reflectance ``(C, H, W)`` in [0, 1] -> the z-scored DN input Prithvi was trained on.

    ``z = (reflectance * 10000 - mean) / std`` per band.  :class:`PrithviBackbone` does this
    internally; the function is public for users who call a raw ``PrithviMAE`` themselves.
    """
    c = reflectance.shape[0]
    m = np.asarray(mean, dtype=np.float32)[:c].reshape(-1, 1, 1)
    s = np.asarray(std, dtype=np.float32)[:c].reshape(-1, 1, 1)
    return ((reflectance.astype(np.float32) * HLS_SCALE_FACTOR - m) / s).astype(np.float32)


# ── Backbone wrapper ──────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _backbone_classes() -> tuple[type, type]:
    """Build the torch classes lazily so ``import prithvi`` works without torch."""
    import torch
    import torch.nn as nn

    class StandInEncoder(nn.Module):
        """Small random-init ViT with the PrithviViT ``forward_features`` interface.

        **Not the Prithvi architecture** and not sized like it (2 blocks).  It exists so that
        code built on :class:`PrithviBackbone` can be developed and tested without downloading
        a checkpoint.  Anything it outputs is meaningless.
        """

        def __init__(self, in_chans: int, embed_dim: int, patch_size: tuple[int, int, int],
                     depth: int = 2) -> None:
            super().__init__()
            heads = next(h for h in (16, 12, 8, 4, 2, 1) if embed_dim % h == 0)
            self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4, dropout=0.0, batch_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(embed_dim)

        def forward_features(self, x, temporal_coords=None, location_coords=None):
            p = self.patch_embed(x).flatten(2).transpose(1, 2)        # (B, T*h*w, D), order (t, h, w)
            tokens = torch.cat([self.cls_token.expand(p.shape[0], -1, -1), p], dim=1)
            return [self.norm(self.blocks(tokens))]

    class PrithviBackbone(nn.Module):
        """Uniform front-end over a Prithvi encoder.

        ``forward(pixel_values)`` takes surface reflectance in [0, 1] as ``(B, C, H, W)`` (one
        frame) or ``(B, T, C, H, W)``, standardises it with the model's own band statistics, and
        returns an object with

        * ``last_hidden_state`` ``(B, 1 + T*N, D)`` -- CLS token, then patch tokens ordered (t, h, w)
        * ``cls``               ``(B, D)``
        * ``frame_tokens``      ``(B, T, N, D)``
        * ``grid``              ``(h, w)`` patches per frame
        """

        def __init__(self, encoder: nn.Module, *, embed_dim: int, patch_size: tuple[int, int, int],
                     img_size: int | None, num_frames: int | None, n_bands: int,
                     mean: list[float], std: list[float]) -> None:
            super().__init__()
            if patch_size[0] != 1:
                raise ValueError(f"temporal patch size must be 1, got {patch_size}")
            self.encoder = encoder
            self.embed_dim = embed_dim
            self.patch_size = tuple(patch_size)
            self.img_size = img_size
            self.num_frames = num_frames
            self.n_bands = n_bands
            self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1, 1))
            self.register_buffer("_std", torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1, 1))

        def forward(self, pixel_values=None, x=None, temporal_coords=None, location_coords=None, **_):
            x = pixel_values if pixel_values is not None else x
            if x is None:
                raise ValueError("Pass pixel_values=... (or x=...)")
            if x.ndim == 4:
                x = x.unsqueeze(1)
            elif x.ndim != 5:
                raise ValueError(f"expected (B, C, H, W) or (B, T, C, H, W), got {tuple(x.shape)}")
            b, t, c, h, w = x.shape
            if c != self.n_bands:
                raise ValueError(f"Prithvi takes {self.n_bands} bands, got {c}")
            if self.num_frames is not None and t != self.num_frames:
                raise ValueError(
                    f"this model was built for {self.num_frames} frame(s) but got {t}; reload it with "
                    f"num_frames={t} (the positional embedding depends on it)")
            if self.img_size is not None and (h, w) != (self.img_size, self.img_size):
                raise ValueError(f"this model was built for {self.img_size}x{self.img_size} tiles, "
                                 f"got {h}x{w}")
            ph, pw = self.patch_size[1:]
            if h % ph or w % pw:
                raise ValueError(f"tile {h}x{w} is not a multiple of the patch size {ph}x{pw}")

            z = (x * HLS_SCALE_FACTOR - self._mean) / self._std            # reflectance -> z-scored DN
            z = z.permute(0, 2, 1, 3, 4).contiguous()                      # (B, C, T, H, W)
            tokens = self.encoder.forward_features(z, temporal_coords, location_coords)[-1]
            gh, gw = h // ph, w // pw
            if tokens.shape[1] != 1 + t * gh * gw:
                raise RuntimeError(f"encoder returned {tokens.shape[1]} tokens, expected {1 + t * gh * gw}")
            return SimpleNamespace(
                last_hidden_state=tokens,
                cls=tokens[:, 0],
                frame_tokens=tokens[:, 1:].reshape(b, t, gh * gw, -1),
                grid=(gh, gw),
            )

    return PrithviBackbone, StandInEncoder


def _build_prithvi_surrogate(spec: dict, device: str = "cpu") -> Any:
    """Random-init stand-in for development and tests (flagged ``_pygeovision_is_random_init``).

    It is **not** Prithvi: a 2-block ViT behind the same :class:`PrithviBackbone` interface.
    """
    import torch  # noqa: F401  (fail early with a clear error if torch is missing)

    backbone_cls, encoder_cls = _backbone_classes()
    embed_dim = spec.get("embed_dim", 768)
    n_bands = spec.get("n_bands", 6)
    ps = spec.get("patch_size", 16)
    ps = tuple(ps) if isinstance(ps, (list, tuple)) else (1, ps, ps)
    model = backbone_cls(
        encoder_cls(n_bands, embed_dim, ps), embed_dim=embed_dim, patch_size=ps, img_size=None,
        num_frames=None, n_bands=n_bands, mean=list(HLS_MEAN_DN[:n_bands]), std=list(HLS_STD_DN[:n_bands]),
    ).to(device).eval()
    model._pygeovision_is_random_init = True
    model._pygeovision_random_init_reason = "stand-in encoder, not the Prithvi architecture"
    logger.warning("Prithvi stand-in built (random weights, NOT the Prithvi architecture): outputs are meaningless.")
    return model


# ── Loading ───────────────────────────────────────────────────────────────────

def _import_prithvi_mae(code_path: str | Path) -> Any:
    """Import the repository's ``prithvi_mae.py`` (executes that file -- see module docstring)."""
    code_path = Path(code_path)
    name = f"_pygeovision_prithvi_mae_{abs(hash(str(code_path.resolve())))}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, code_path)
    if spec is None or spec.loader is None:
        raise PrithviLoadError(f"cannot import {code_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        del sys.modules[name]
        raise ImportError(f"prithvi_mae.py needs extra packages: pip install einops timm ({exc})") from exc
    except Exception:
        del sys.modules[name]
        raise
    if not hasattr(module, "PrithviMAE"):
        del sys.modules[name]
        raise PrithviLoadError(f"{code_path} does not define PrithviMAE")
    return module


def _assemble_backbone(
    model_name: str, spec: dict, config_path: str | Path, code_path: str | Path,
    weights_path: str | Path, num_frames: int, device: str,
) -> Any:
    """Build ``PrithviMAE`` from ``config.json``, load the checkpoint strictly, wrap the encoder."""
    import torch

    try:
        with open(config_path) as f:
            cfg = dict(json.load(f)["pretrained_cfg"])
    except (KeyError, ValueError, TypeError) as exc:
        raise PrithviLoadError(f"{config_path} has no usable 'pretrained_cfg' section: {exc}") from exc

    n_bands = int(cfg.get("in_chans", spec.get("n_bands", 6)))
    patch = tuple(cfg.get("patch_size", (1, spec.get("patch_size", 16), spec.get("patch_size", 16))))
    cfg["num_frames"] = int(num_frames)
    mean = cfg.get("mean") or list(HLS_MEAN_DN[:n_bands])
    std = cfg.get("std") or list(HLS_STD_DN[:n_bands])
    if len(mean) != n_bands or len(std) != n_bands:
        raise PrithviLoadError(f"{config_path}: mean/std do not have {n_bands} entries")

    module = _import_prithvi_mae(code_path)
    mae = module.PrithviMAE(**cfg)

    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    # The positional embeddings are fixed sin-cos buffers rebuilt for the requested
    # num_frames / img_size, which is why the official script discards the stored ones.
    state = {k: v for k, v in state.items() if "pos_embed" not in k}
    result = mae.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if "pos_embed" not in k]
    if missing:
        raise PrithviLoadError(
            f"{model_name}: checkpoint {Path(weights_path).name} does not fill {len(missing)} layer(s) of the "
            f"architecture built from {Path(config_path).name} (first: {missing[:3]}); they would stay "
            f"randomly initialised.",
            details="config.json, prithvi_mae.py and the checkpoint must come from the same repository revision.")
    if result.unexpected_keys:
        logger.warning("%s: %d checkpoint key(s) are not used by this architecture (first: %s)",
                       model_name, len(result.unexpected_keys), result.unexpected_keys[:3])

    backbone_cls, _ = _backbone_classes()
    model = backbone_cls(
        mae.encoder, embed_dim=int(cfg["embed_dim"]), patch_size=patch, img_size=int(cfg.get("img_size", 224)),
        num_frames=int(num_frames), n_bands=n_bands, mean=list(mean), std=list(std),
    )
    model._pygeovision_config = cfg
    return model.to(device).eval()


def _random_init_or_raise(model_name: str, spec: dict, device: str, allow_random_init: bool,
                          exc: BaseException) -> Any:
    if not allow_random_init:
        raise PrithviLoadError(
            f"Failed to load real, pretrained Prithvi weights for '{model_name}' ({spec.get('hf_id')}): {exc}",
            details="Nothing was substituted: a randomly initialised model would give plausible-looking but "
                    "meaningless output. Needs network access to the Hugging Face Hub (or use "
                    "load_prithvi_local with the three downloaded files) plus `pip install einops timm`. "
                    "Pass allow_random_init=True only for development/testing.",
        ) from exc
    logger.warning("Loading '%s' failed (%s); allow_random_init=True, so returning a random stand-in.",
                   model_name, exc)
    return _build_prithvi_surrogate(spec, device)


def _get_spec(model_name: str) -> dict:
    spec = PRITHVI_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown Prithvi model: '{model_name}'. Available: {list(PRITHVI_MODELS)}")
    return spec


def load_prithvi_hf(model_name: str = "prithvi_eo_2_0",
                    device: str = "cpu",
                    allow_random_init: bool = False,
                    num_frames: int = 1,
                    revision: str | None = None) -> Any:
    """Load a pretrained Prithvi backbone from the Hugging Face Hub.

    Args:
        model_name: key of :data:`PRITHVI_MODELS`.
        device: target device.
        allow_random_init: if the real weights cannot be loaded, return a flagged random stand-in
            (``_pygeovision_is_random_init = True``) instead of raising.  Development only.
        num_frames: number of time steps the model will see (the positional embedding is built for
            it).  1 for single-date features; the pre-training value is in ``spec["n_frames"]``.
        revision: Hub commit to download.  Defaults to the pinned ``spec["revision"]``.

    Returns:
        A :class:`PrithviBackbone` (encoder only; the MAE decoder is dropped).

    Raises:
        ValueError: unknown ``model_name``.
        NotImplementedError: ``model_name`` is a fine-tuned downstream checkpoint.
        PrithviLoadError: the weights could not be loaded (also a ``RuntimeError``).
    """
    spec = _get_spec(model_name)
    if not spec.get("loadable", True):
        raise NotImplementedError(f"'{model_name}': {spec.get('load_note', 'not a loadable backbone')}")
    rev = revision or spec.get("revision")
    logger.info("Loading Prithvi %s from %s@%s", model_name, spec["hf_id"], (rev or "main")[:8])
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(spec["hf_id"], "config.json", revision=rev)
        code_path = hf_hub_download(spec["hf_id"], "prithvi_mae.py", revision=rev)
        weights_path = hf_hub_download(spec["hf_id"], spec["checkpoint"], revision=rev)
        model = _assemble_backbone(model_name, spec, config_path, code_path, weights_path, num_frames, device)
        logger.info("Prithvi loaded: %s (%sM params, %d frame(s))", model_name, spec["params_m"], num_frames)
        return model
    except PrithviLoadError as exc:
        if not allow_random_init:
            raise
        return _random_init_or_raise(model_name, spec, device, allow_random_init, exc)
    except Exception as exc:  # network, auth, missing file, missing einops/timm ...
        return _random_init_or_raise(model_name, spec, device, allow_random_init, exc)


def load_prithvi_local(model_name: str, weights_path: str, device: str = "cpu",
                       num_frames: int = 1, config_path: str | None = None,
                       code_path: str | None = None) -> Any:
    """Load Prithvi from files already on disk (air-gapped use).

    Needs the three files of the repository revision: the checkpoint (``weights_path``),
    ``config.json`` and ``prithvi_mae.py``.  The latter two default to the directory of
    ``weights_path``.  A checkpoint that does not match the architecture raises
    :class:`PrithviLoadError` -- the earlier version loaded real weights into a different
    network with ``strict=False`` and returned a random model.
    """
    spec = _get_spec(model_name)
    if not spec.get("loadable", True):
        raise NotImplementedError(f"'{model_name}': {spec.get('load_note', 'not a loadable backbone')}")
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    config = Path(config_path) if config_path else weights.parent / "config.json"
    code = Path(code_path) if code_path else weights.parent / "prithvi_mae.py"
    for label, path in (("config.json", config), ("prithvi_mae.py", code)):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path}. Download it from the same revision of "
                f"{spec['hf_id']} and pass its path (config_path= / code_path=).")
    return _assemble_backbone(model_name, spec, config, code, weights, num_frames, device)



# ── Prithvi — main API ───────────────────────────────────────────────────────

class Prithvi:
    """Prithvi-EO foundation model: features, dense features and a trainable segmentation head.

    Example::

        model = Prithvi("prithvi_eo_2_0").load()
        features = model.extract_features("hls_scene.tif")          # (1, 1024)
        seg = model.build_segmentation_head(num_classes=11)         # train it, then use PrithviTasks

    The backbone is built for a fixed number of time steps (``num_frames``) and a fixed tile
    size (224).  Single-image methods use ``num_frames=1``; :class:`PrithviMultiTemporal`
    reloads the model for the number of dates it is given.
    """

    def __init__(self, variant: str = "prithvi_eo_2_0",
                 method: str = "hf",
                 device: str | None = None,
                 allow_random_init: bool = False,
                 num_frames: int = 1) -> None:
        self.variant = variant
        self.method  = method
        self.device  = device or self._auto_device()
        self.allow_random_init = allow_random_init
        self.num_frames = num_frames
        self._model  = None
        self._spec   = PRITHVI_MODELS.get(variant, {})
        self._local: tuple[str, str | None, str | None] | None = None

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

    def _build(self) -> Any:
        if self._local is not None:
            weights, config, code = self._local
            return load_prithvi_local(self.variant, weights, self.device, num_frames=self.num_frames,
                                      config_path=config, code_path=code)
        return load_prithvi_hf(self.variant, self.device, allow_random_init=self.allow_random_init,
                               num_frames=self.num_frames)

    def load(self, weights_path: str | None = None, config_path: str | None = None,
             code_path: str | None = None) -> Prithvi:
        """Load the backbone (Hub, or local files if ``weights_path`` is given). Returns self."""
        self._local = (weights_path, config_path, code_path) if weights_path else None
        self._model = self._build()
        return self

    def _ensure_loaded(self, num_frames: int | None = None) -> None:
        """Load if needed, and rebuild for ``num_frames`` if the loaded model is built for another."""
        if self._model is None:
            if num_frames is not None:
                self.num_frames = num_frames
            self._model = self._build()
            return
        have = getattr(self._model, "num_frames", None)
        if num_frames is not None and have is not None and have != num_frames:
            logger.info("Reloading %s for %d frame(s) (was built for %d)", self.variant, num_frames, have)
            self.num_frames = num_frames
            self._model = self._build()

    def _tile_size(self) -> int:
        return int(getattr(self._model, "img_size", None) or 224)

    def _load_geotiff(
        self,
        path: str,
        source: str = "hls",
        n_bands: int | None = None,
        source_bands: list[str] | None = None,
        resample_to_30m: bool = True,
    ) -> np.ndarray:
        """Load and prepare a GeoTIFF for Prithvi inference.

        Args:
            path: GeoTIFF path (can be a stacked multi-band file or a
                single-band file per channel — stacked recommended).
            source: ``"hls"`` | ``"sentinel2"`` | ``"landsat"``.
            n_bands: Override the number of Prithvi bands (default from spec).
            source_bands: Explicit list of band names for each channel in the
                raster (e.g. ``["B02","B03","B04","B08","B11","B12"]``).
                When ``None``, canonical ordering for *source* is assumed.
            resample_to_30m: Resample to Prithvi's native 30m resolution
                when the raster pixel size is finer than 20m. Prithvi was
                pre-trained on 30m HLS; submitting 10m Sentinel-2 directly
                degrades feature quality.

        Returns:
            ``(n_prithvi_bands, H, W)`` float32 array, normalised to ``[0, 1]``,
            in Prithvi HLS band order.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling as RSpl
        except ImportError as exc:
            raise ImportError("pip install rasterio") from exc

        n_prithvi = n_bands or self._spec.get("n_bands", 6)

        with rasterio.open(path) as src:
            avail      = min(src.count, n_prithvi)
            orig_h     = src.height
            orig_w     = src.width

            # Compute ground pixel size in metres — handles both geographic and
            # projected CRS so the 30m resampling threshold is always meaningful.
            raw_res = abs(src.transform.a)
            if src.crs and src.crs.is_geographic:
                # Degrees → metres: at the scene centre, 1° latitude ≈ 111,111 m
                import math
                centre_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                res_m = raw_res * 111111.0 * math.cos(math.radians(centre_lat))
            else:
                res_m = raw_res   # already in metres for projected CRS

            # Resample to 30m when finer than 20m (Prithvi native resolution)
            if resample_to_30m and res_m < 20.0:
                scale = res_m / 30.0   # e.g. 10m → scale=0.333
                new_H = max(1, int(orig_h * scale))
                new_W = max(1, int(orig_w * scale))
                data  = src.read(
                    list(range(1, avail + 1)),
                    out_shape=(avail, new_H, new_W),
                    resampling=RSpl.average,
                ).astype(np.float32)
                logger.debug(
                    "_load_geotiff: resampled %dx%d (%.1fm) → %dx%d (30m) for Prithvi",
                    orig_w, orig_h, res_m, new_W, new_H,
                )
            else:
                data = src.read(list(range(1, avail + 1))).astype(np.float32)

        # Pad to required band count (zero-fill missing bands)
        if data.shape[0] < n_prithvi:
            pad  = np.zeros((n_prithvi - data.shape[0], data.shape[1], data.shape[2]),
                            dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        # Reorder to Prithvi HLS band order using explicit band names when available
        data = map_bands(data, source=source, n_prithvi_bands=n_prithvi,
                         source_bands=source_bands)

        # Normalise — detect whether data is raw DN (max >> 1) or already [0,1]
        data_max = float(np.nanmax(data))
        if data_max > 10.0:
            data = normalise_hls(data)
            logger.debug("_load_geotiff: normalised DN→reflectance ÷%.0f", HLS_SCALE_FACTOR)
        elif data_max > 1.5:
            # Values in (1.5, 10] — unusual; warn and clip
            logger.warning(
                "_load_geotiff: max=%.3f looks denormalised but < 10 — clipping to [0,1]. "
                "Check your preprocessing scale factor.", data_max)
            data = np.clip(data, 0.0, 1.0)

        # Replace NaN with 0 (Prithvi cannot handle NaN)
        nan_mask = ~np.isfinite(data)
        if nan_mask.any():
            logger.debug("_load_geotiff: replacing %d NaN/Inf pixels with 0", nan_mask.sum())
            data[nan_mask] = 0.0

        return data


    def _tile_array(self, image_path: str, source: str = "hls") -> np.ndarray:
        """``(C, S, S)`` reflectance resized to the model's tile size ``S``."""
        import cv2
        data = self._load_geotiff(image_path, source)
        size = self._tile_size()
        return np.stack([cv2.resize(data[b], (size, size), interpolation=cv2.INTER_AREA)
                         for b in range(data.shape[0])]).astype(np.float32)

    def _tile_tensor(self, image_path: str, source: str = "hls") -> Any:
        import torch
        return torch.from_numpy(self._tile_array(image_path, source)).unsqueeze(0).to(self.device)

    def extract_features(self, image_path: str, source: str = "hls", pool: str = "cls") -> np.ndarray:
        """Scene-level feature vector from an HLS-style GeoTIFF.

        The whole scene is resized to the model's tile size, so this describes the scene, not a
        30 m pixel.

        Args:
            image_path: GeoTIFF with the six Prithvi bands (see ``source`` / band mapping).
            source: ``"hls"`` | ``"sentinel2"`` | ``"landsat"``.
            pool: ``"cls"`` -- the CLS token (an MAE encoder's CLS is not trained for
                classification); ``"mean"`` -- the mean of the patch tokens, usually the better
                global descriptor.

        Returns:
            ``(1, embed_dim)`` array.
        """
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        import torch
        self._ensure_loaded(1)
        with torch.no_grad():
            out = self._model(pixel_values=self._tile_tensor(image_path, source))
        feat = out.cls if pool == "cls" else out.frame_tokens.mean(dim=(1, 2))
        return feat.cpu().float().numpy()

    def extract_patch_features(self, image_path: str, source: str = "hls") -> np.ndarray:
        """Per-patch features for dense prediction: ``(h*w, embed_dim)``, row-major patch grid."""
        import torch
        self._ensure_loaded(1)
        with torch.no_grad():
            out = self._model(pixel_values=self._tile_tensor(image_path, source))
        return out.frame_tokens[0, 0].cpu().float().numpy()

    def build_segmentation_head(self, num_classes: int, freeze_backbone: bool = True) -> Any:
        """Semantic-segmentation model: Prithvi encoder + a **randomly initialised** conv head.

        The head must be trained before its output means anything.  The result maps
        ``(B, 6, S, S)`` reflectance to ``(B, num_classes, S, S)`` logits.

        Args:
            num_classes: number of output classes.
            freeze_backbone: freeze the encoder (recommended for small datasets).
        """
        self._ensure_loaded(1)
        import torch.nn as nn
        import torch.nn.functional as F

        embed_dim = int(self._model.embed_dim)

        class PrithviSegModel(nn.Module):
            def __init__(self, backbone, head_, freeze):
                super().__init__()
                self.backbone = backbone
                self.head = head_
                self._freeze = freeze

            def train(self, mode: bool = True):
                super().train(mode)
                if self._freeze:
                    self.backbone.eval()
                return self

            def forward(self, x):
                out = self.backbone(pixel_values=x)
                gh, gw = out.grid
                tokens = out.frame_tokens[:, 0]                         # (B, N, D); single frame
                feat = tokens.transpose(1, 2).reshape(tokens.shape[0], embed_dim, gh, gw)
                return F.interpolate(self.head(feat), size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)

        head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        if freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad = False
        return PrithviSegModel(self._model, head, freeze_backbone).to(self.device)

    def finetune_config(self) -> dict:
        """Recommended fine-tuning hyperparameters for Prithvi-EO-2.0."""
        return {
            "optimizer":      "AdamW",
            "learning_rate":  5e-5,
            "weight_decay":   0.01,
            "warmup_epochs":  5,
            "scheduler":      "cosine_annealing",
            "mixed_precision": "bf16",
            "batch_size":     8,
            "note": "Use freeze_backbone=True for datasets < 10k samples",
        }

    def __repr__(self) -> str:
        return (f"Prithvi(variant={self.variant!r}, "
                f"params={self._spec.get('params_m',0)}M, "
                f"coverage={self._spec.get('coverage','?')!r}, "
                f"device={self.device}, loaded={self._model is not None})")


# ── Saving / loading trained heads ───────────────────────────────────────────

_HEAD_FORMAT = "pygeovision-prithvi-head-v1"


def save_prithvi_head(model: Any, path: str, variant: str, task: str, num_classes: int) -> str:
    """Save the trained ``.head`` of a :meth:`Prithvi.build_segmentation_head` model.

    The file records the backbone variant, task and class count so :class:`PrithviTasks` can
    refuse a head that was trained for something else.
    """
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": _HEAD_FORMAT, "variant": variant, "task": task, "num_classes": num_classes,
                "state_dict": getattr(model, "head", model).state_dict()}, path)
    return str(path)


def _load_head_file(path: str, variant: str, task: str, num_classes: int | None) -> dict:
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or blob.get("format") != _HEAD_FORMAT:
        raise ValueError(f"{path} is not a Prithvi head file written by save_prithvi_head()")
    for key, want in (("variant", variant), ("task", task)):
        if blob.get(key) != want:
            raise ValueError(f"{path} was trained for {key}={blob.get(key)!r}, not {want!r}")
    if num_classes is not None and blob.get("num_classes") != num_classes:
        raise ValueError(f"{path} has {blob.get('num_classes')} classes, expected {num_classes}")
    return blob


# ── PrithviMultiTemporal ──────────────────────────────────────────────────────

class PrithviMultiTemporal:
    """Multi-date analysis in Prithvi's feature space.

    Prithvi's encoder attends jointly over all dates, so these tools describe how the *learned
    features* differ between dates.  They are screening indicators, not calibrated change or
    phenology models: validate against reference data before relying on them.

    Example::

        mt = PrithviMultiTemporal("prithvi_eo_2_0")
        out = mt.process_time_series(["jan.tif", "apr.tif", "jul.tif", "oct.tif"])
        change = mt.detect_change("before.tif", "after.tif")
    """

    def __init__(self, model_name: str = "prithvi_eo_2_0",
                 device: str | None = None, allow_random_init: bool = False) -> None:
        self.model_name = model_name
        self._prithvi   = Prithvi(model_name, device=device, allow_random_init=allow_random_init)

    def _encode(self, image_paths: list[str], source: str):
        import torch
        if not image_paths:
            raise ValueError("need at least one image")
        T = len(image_paths)
        self._prithvi._ensure_loaded(T)
        stack = np.stack([self._prithvi._tile_array(p, source) for p in image_paths])      # (T, C, S, S)
        tensor = torch.from_numpy(stack).unsqueeze(0).to(self._prithvi.device)             # (1, T, C, S, S)
        with torch.no_grad():
            return self._prithvi._model(pixel_values=tensor)

    def process_time_series(self, image_paths: list[str],
                             dates: list[str] | None = None,
                             source: str = "hls") -> dict[str, Any]:
        """Encode a chronological stack of HLS-style GeoTIFFs.

        Returns a dict with ``frame_embeddings`` ``(T, D)`` (mean patch token per date),
        ``frame_tokens`` ``(T, N, D)``, ``features`` ``(1 + T*N, D)`` (CLS first), ``grid``,
        ``n_frames``, ``dates`` and ``model``.
        """
        out = self._encode(image_paths, source)
        T = len(image_paths)
        frame_tokens = out.frame_tokens[0].cpu().float().numpy()
        return {
            "frame_embeddings": frame_tokens.mean(axis=1),
            "frame_tokens":     frame_tokens,
            "features":         out.last_hidden_state[0].cpu().float().numpy(),
            "grid":             tuple(out.grid),
            "n_frames":         T,
            "dates":            dates or [f"t{i}" for i in range(T)],
            "model":            self.model_name,
        }

    def detect_change(self, before_path: str, after_path: str,
                       source: str = "hls",
                       output_path: str | None = None) -> dict[str, Any]:
        """Feature-space change map between two dates.

        ``change_map`` is the per-patch L2 distance between the two dates' patch features,
        normalised by its maximum (patch grid, e.g. 14x14).  ``change_pct`` is the share of
        patches above half of that maximum -- a relative indicator, not an area of change.
        If ``output_path`` is given the map is resampled to the scene and written as a GeoTIFF.
        """
        result = self.process_time_series([before_path, after_path], dates=["before", "after"], source=source)
        tokens = result["frame_tokens"]                                  # (2, N, D)
        gh, gw = result["grid"]
        score = np.linalg.norm(tokens[1] - tokens[0], axis=-1).reshape(gh, gw)
        peak = float(score.max())
        change_map = score / peak if peak > 0 else np.zeros_like(score)
        out: dict[str, Any] = {
            "change_map":   change_map.astype(np.float32),
            "raw_distance": score.astype(np.float32),
            "change_pct":   float((change_map > 0.5).mean() * 100),
            "output_path":  None,
            "model":        self.model_name,
            "note":         "relative feature-distance indicator, not a calibrated change area",
        }
        if output_path:
            try:
                import cv2
                import rasterio
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(before_path) as src:
                    profile = src.profile.copy()
                    H, W = src.height, src.width
                full = cv2.resize(change_map.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                profile.update(count=1, dtype="float32", compress="lzw", nodata=None)
                for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                    profile.pop(_k, None)
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(full[np.newaxis].astype(np.float32))
                out["output_path"] = output_path
            except Exception as exc:
                logger.warning("Failed to save change map: %s", exc)
                out["output_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def monitor_trend(self, time_series_paths: list[str],
                       dates: list[str] | None = None,
                       source: str = "hls") -> dict[str, Any]:
        """How far each date's embedding has drifted from the first date, and whether it is growing.

        ``drift_from_first[t]`` is ``||e_t - e_0||``; ``trend_direction`` is the sign of its linear
        slope (``diverging`` / ``converging`` / ``stable``).  Embedding axes have no fixed sign or
        meaning, so a sign-based "increasing/decreasing" trend (used by earlier versions) was
        arbitrary.
        """
        result = self.process_time_series(time_series_paths, dates=dates, source=source)
        emb = result["frame_embeddings"]
        T = emb.shape[0]
        if T < 3:
            return {**result, "trend": "insufficient_data (need ≥ 3 time steps)"}
        drift = np.linalg.norm(emb - emb[0], axis=1)
        slope = float(np.polyfit(np.arange(T), drift, 1)[0])
        direction = "stable" if abs(slope) < 1e-8 else ("diverging" if slope > 0 else "converging")
        return {
            **result,
            "trend_direction":  direction,
            "trend_magnitude":  abs(slope),
            "drift_from_first": drift.tolist(),
        }

    def predict_seasonal(self, time_series_paths: list[str],
                          dates: list[str] | None = None) -> dict[str, Any]:
        """Dominant mode of variation across ≥ 4 dates (first principal component of the embeddings).

        ``seasonal_amplitude`` is the half-range of that component in embedding units;
        ``peak_step`` / ``trough_step`` are its extreme dates (sign fixed so the first date is not
        above the mean).  This is a statistic of the feature space, not a physical seasonal model.
        """
        result = self.process_time_series(time_series_paths, dates=dates)
        emb = result["frame_embeddings"]
        T = emb.shape[0]
        if T < 4:
            return {**result, "note": "Need ≥ 4 time steps for seasonal model"}
        centred = emb - emb.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        pc1 = centred @ vt[0]
        if pc1[0] > pc1.mean():
            pc1 = -pc1
        labels = result["dates"]
        peak, trough = int(np.argmax(pc1)), int(np.argmin(pc1))
        return {
            **result,
            "seasonal_amplitude": float((pc1.max() - pc1.min()) / 2),
            "peak_step":          peak,
            "trough_step":        trough,
            "peak_date":          labels[peak],
            "trough_date":        labels[trough],
            "note":               "first principal component of the date embeddings, not a physical seasonal model",
        }


# ── PrithviTasks — task-specific inference ────────────────────────────────────

class PrithviTasks:
    """Task-level inference on preprocessed HLS-style imagery.

    Inputs: a GeoTIFF with six bands in HLS order (see ``map_bands``), surface reflectance, passed
    through ``client.prepare_for_ai()`` first.

    Two routes, reported in ``result["method"]``:

    * ``"backbone"`` -- the Prithvi encoder plus a segmentation head **you trained** (pass the
      file written by :func:`save_prithvi_head` as ``head_weights={"land_cover": path}``).
    * ``"spectral"`` -- no trained head for the task: documented index rules (NDVI/MNDWI/NDBI/NBR).
      These are heuristics on reflectance, they do not use Prithvi, and they need no model download.

    ``crop_mapping`` and ``biomass_estimation`` have no honest spectral rule and therefore raise
    ``NotImplementedError`` until a trained head is supplied.

    Example::

        tasks = PrithviTasks("prithvi_eo_2_0", head_weights={"land_cover": "heads/lc.pt"})
        lc = tasks.land_cover("preprocessed.tif", source="sentinel2")     # method == "backbone"
        fl = tasks.flood_detection("preprocessed.tif", source="sentinel2") # method == "spectral"
    """

    LAND_COVER_CLASSES = LAND_COVER_CLASSES   # noqa: F821
    CROP_CLASSES       = CROP_CLASSES          # noqa: F821

    def __init__(self, model_name: str = "prithvi_eo_2_0",
                 device: str | None = None,
                 head_weights: dict[str, str] | None = None,
                 allow_random_init: bool = False) -> None:
        self.model_name = model_name
        self._prithvi   = Prithvi(model_name, device=device, allow_random_init=allow_random_init)
        self._head_weights = dict(head_weights or {})
        self._seg_heads: dict[str, Any] = {}

    def _seg_head(self, task: str, n_classes: int) -> Any:
        """Segmentation model with the trained head for ``task`` loaded (cached)."""
        if task not in self._seg_heads:
            blob = _load_head_file(self._head_weights[task], self.model_name, task, n_classes)
            self._prithvi._ensure_loaded(1)
            model = self._prithvi.build_segmentation_head(n_classes, freeze_backbone=True)
            model.head.load_state_dict(blob["state_dict"])        # strict: shape mismatch raises
            self._seg_heads[task] = model.eval()
        return self._seg_heads[task]

    _CANONICAL_BANDS = {
        "sentinel2": ["B02", "B03", "B04", "B08", "B11", "B12"],
        "landsat":   ["B2",  "B3",  "B4",  "B5",  "B6",  "B7"],
        "hls":       None,
    }

    def _infer_segmentation(
        self,
        image_path: str,
        task: str,
        n_classes: int,
        source: str = "hls",
        source_bands: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Segmentation inference: trained head if available, else spectral rules (or an error)."""
        use_backbone = task in self._head_weights
        if not use_backbone and task == "crop_mapping":
            raise NotImplementedError(
                "Crop-type mapping cannot be done honestly from spectral thresholds on one date. "
                "Train a head (Prithvi.build_segmentation_head + save_prithvi_head) and pass "
                "head_weights={'crop_mapping': path}.")

        bands = source_bands or self._CANONICAL_BANDS.get(source)
        data_hls = self._prithvi._load_geotiff(image_path, source=source, source_bands=bands)

        val = validate_prithvi_input(data_hls, source=source)
        for err in val["errors"]:
            raise ValueError(f"[Prithvi input] {err}")
        for warn in val["warnings"]:
            logger.warning("[Prithvi input] %s", warn)

        H, W = data_hls.shape[1], data_hls.shape[2]
        if use_backbone:
            pred_np = self._backbone_predict(data_hls, task, n_classes, H, W)
            logger.info("Prithvi [%s]: pretrained backbone + trained head", task)
        else:
            pred_np = self._spectral_predict(data_hls, task, n_classes)
            logger.info("Prithvi [%s]: spectral-index rules (no trained head supplied)", task)

        unique, counts = np.unique(pred_np, return_counts=True)
        class_pct = {int(u): round(float(c / pred_np.size * 100), 2) for u, c in zip(unique, counts)}
        result: dict[str, Any] = {
            "prediction":   pred_np,
            "class_pct":    class_pct,
            "n_classes":    n_classes,
            "task":         task,
            "model":        self.model_name,
            "method":       "backbone" if use_backbone else "spectral",
            "shape":        pred_np.shape,
            "source":       source,
        }
        if not use_backbone:
            result["note"] = ("spectral-index rules on reflectance; Prithvi was not used. "
                              "Pass head_weights for a learned prediction.")
        if output_path:
            self._save_prediction(pred_np, image_path, output_path)
            result["output_path"] = output_path
        return result

    def _spectral_predict(
        self, data_hls: np.ndarray, task: str, n_classes: int
    ) -> np.ndarray:
        """Route to the correct spectral rule-based predictor."""
        if task == "land_cover":
            return _spectral_land_cover(data_hls)
        elif task == "crop_mapping":
            return _spectral_crop_mapping(data_hls)
        elif task in ("flood", "flood_detection"):
            return _spectral_flood(data_hls)
        elif task in ("burn_scar", "burn_scar_detection"):
            return _spectral_burn_scar(data_hls)
        else:
            # Generic: use land cover with n_classes override
            pred = _spectral_land_cover(data_hls)
            pred = np.clip(pred, 0, n_classes - 1).astype(np.uint8)
            return pred


    def _backbone_predict(
        self,
        data_hls: np.ndarray,
        task: str,
        n_classes: int,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """Tiled inference with the Prithvi backbone + task segmentation head."""
        import math

        import torch
        import torch.nn.functional as F_

        model = self._seg_head(task, n_classes)
        model.eval()

        TILE = self._prithvi._tile_size()
        STEP = TILE * 7 // 8     # 1/8 overlap
        C, H, W = data_hls.shape

        pred_sum   = np.zeros((n_classes, H, W), dtype=np.float32)
        np.zeros((H, W), dtype=np.float32)

        # Build Gaussian weight window for blend
        def _gauss_window(size):
            ax  = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            g   = np.exp(-0.5 * (ax / (size * 0.3))**2)
            g2d = np.outer(g, g)
            return (g2d / g2d.max()).astype(np.float32)
        win = _gauss_window(TILE)

        # Pad to cover full image
        pad_H = math.ceil((H - TILE) / STEP) * STEP + TILE if H > TILE else TILE
        pad_W = math.ceil((W - TILE) / STEP) * STEP + TILE if W > TILE else TILE
        pad_data  = np.pad(data_hls,
                           ((0,0), (0, pad_H-H), (0, pad_W-W)),
                           mode="reflect")
        pad_sum   = np.zeros((n_classes, pad_H, pad_W), dtype=np.float32)
        pad_count = np.zeros((pad_H, pad_W), dtype=np.float32)

        rows = range(0, pad_H - TILE + 1, STEP) if pad_H >= TILE else [0]
        cols = range(0, pad_W - TILE + 1, STEP) if pad_W >= TILE else [0]

        with torch.no_grad():
            for r in rows:
                for c in cols:
                    chip = pad_data[:, r:r+TILE, c:c+TILE]
                    tensor = torch.tensor(chip).unsqueeze(0).to(self._prithvi.device)
                    logits = model(tensor)   # (1, n_classes, H', W')
                    probs  = F_.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    # Resize to tile size if head output ≠ TILE
                    if probs.shape[-1] != TILE:
                        import cv2
                        probs_r = np.stack([
                            cv2.resize(probs[k], (TILE, TILE),
                                       interpolation=cv2.INTER_LINEAR)
                            for k in range(n_classes)
                        ])
                    else:
                        probs_r = probs
                    pad_sum[:, r:r+TILE, c:c+TILE]   += probs_r * win
                    pad_count[r:r+TILE, c:c+TILE]    += win

        # Trim back to original size
        eps = 1e-8
        for k in range(n_classes):
            pred_sum[k] = pad_sum[k, :H, :W] / (pad_count[:H, :W] + eps)

        pred_np = pred_sum.argmax(axis=0).astype(np.uint8)

        # Upsample to original scene resolution if we resampled to 30m
        if (orig_h, orig_w) != (H, W):
            import cv2
            pred_np = cv2.resize(pred_np, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

        return pred_np

    # -----------------------------------------------------------------
    # Task-specific public API
    # -----------------------------------------------------------------


    # -----------------------------------------------------------------
    # Task-specific public API
    # -----------------------------------------------------------------

    def land_cover(
        self,
        image_path: str,
        source: str = "sentinel2",
        source_bands: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """9-class land cover classification (ESA WorldCover scheme).

        Classes: Tree cover, Shrubland, Grassland, Cropland, Built-up,
        Bare/sparse, Snow/ice, Water, Wetland.

        Args:
            image_path: Preprocessed 6-band GeoTIFF in HLS reflectance [0,1].
                        MUST have passed through ``prepare_for_ai()`` first.
            source: ``"sentinel2"`` | ``"landsat"`` | ``"hls"``.
            source_bands: Explicit band names (e.g. ``["B02","B03","B04","B08","B11","B12"]``).
            output_path: Save prediction raster here.

        Returns:
            Dict with ``prediction`` (H,W) uint8, ``class_names``, ``class_pct``.
        """
        result = self._infer_segmentation(
            image_path, "land_cover", len(LAND_COVER_CLASSES),
            source=source, source_bands=source_bands, output_path=output_path)
        result["class_names"] = LAND_COVER_CLASSES
        result["palette"]     = LAND_COVER_PALETTE
        return result

    def crop_mapping(
        self,
        image_path: str,
        source: str = "sentinel2",
        source_bands: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """10-class crop type mapping.

        Classes: corn, soybeans, cotton, winter_wheat, spring_wheat,
        rice, sorghum, other_grains, vegetables, other.
        """
        result = self._infer_segmentation(
            image_path, "crop_mapping", len(CROP_CLASSES),
            source=source, source_bands=source_bands, output_path=output_path)
        result["class_names"] = CROP_CLASSES
        return result

    def flood_detection(
        self,
        image_path: str,
        source: str = "sentinel2",
        source_bands: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Binary flood detection (0=dry, 1=flooded/water).

        Uses MNDWI threshold on HLS Green+SWIR1 bands. Spatially precise
        without requiring a separate cloud mask because MNDWI is less
        sensitive to atmospheric effects than NDWI.
        """
        result = self._infer_segmentation(
            image_path, "flood", 2,
            source=source, source_bands=source_bands, output_path=output_path)
        pred      = result["prediction"]
        flood_pct = float((pred == 1).mean() * 100)
        result.update({
            "flood_pct":   flood_pct,
            "class_names": ["dry / no flood", "flooded / water"],
        })
        return result

    def burn_scar_detection(
        self,
        image_path: str,
        source: str = "sentinel2",
        source_bands: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Binary burn scar detection (0=unburned, 1=burned).

        Uses NBR (Normalised Burn Ratio) on NIR and SWIR2 bands.
        Requires post-fire imagery; for change-based detection use
        ChangeFormer on pre/post pairs instead.
        """
        result = self._infer_segmentation(
            image_path, "burn_scar", 2,
            source=source, source_bands=source_bands, output_path=output_path)
        pred       = result["prediction"]
        burned_pct = float((pred == 1).mean() * 100)
        result.update({
            "burned_pct":  burned_pct,
            "class_names": ["unburned", "burned"],
        })
        return result


    def _save_prediction(
        self, pred: np.ndarray, reference_path: str, output_path: str
    ) -> None:
        """Save a prediction raster preserving the input scene's CRS and extent.

        Handles the case where ``pred`` was computed at 30m resolution
        but the reference scene is 10m — the prediction is upsampled to
        match before writing.
        """
        try:
            import pathlib

            import cv2
            import rasterio

            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(reference_path) as src:
                profile = src.profile.copy()
                ref_H   = src.height
                ref_W   = src.width

            # Upsample to reference resolution if needed (30m pred → 10m ref)
            if pred.shape != (ref_H, ref_W):
                pred = cv2.resize(pred.astype(np.float32), (ref_W, ref_H),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            profile.update(count=1, dtype="uint8", compress="lzw", nodata=255)
            for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                profile.pop(_k, None)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(pred.astype(np.uint8)[np.newaxis])

            logger.info("Prediction saved: %s  shape=(%d,%d)", output_path, ref_H, ref_W)

        except Exception as exc:
            logger.warning("_save_prediction failed (%s) — file not written", exc)


    def biomass_estimation(self, image_path: str, source: str = "hls") -> dict[str, Any]:
        """Above-ground biomass from Prithvi features through a regression head **you trained**.

        Needs ``head_weights={"biomass": path}`` (file from :func:`save_prithvi_head`-style
        ``torch.save`` with ``task="biomass"``; see :func:`build_regression_head`).  The value is in
        whatever units the head was trained on.  Without a head this raises -- the previous
        version pushed features through an untrained MLP and multiplied by 300.
        """
        if "biomass" not in self._head_weights:
            raise NotImplementedError(
                "biomass_estimation needs a regression head trained on biomass reference data; pass "
                "head_weights={'biomass': path}. No pretrained biomass head exists in this package.")
        import torch
        blob = _load_head_file(self._head_weights["biomass"], self.model_name, "biomass", 1)
        self._prithvi._ensure_loaded(1)
        head = build_regression_head(int(self._prithvi._model.embed_dim)).to(self._prithvi.device).eval()
        head.load_state_dict(blob["state_dict"])
        feats = torch.from_numpy(self._prithvi.extract_features(image_path, source, pool="mean")).to(self._prithvi.device)
        with torch.no_grad():
            value = float(head(feats).item())
        return {"estimated_biomass_t_ha": round(value, 3), "model": self.model_name, "source": source,
                "method": "backbone"}

    def deforestation_detection(self, before_path: str, after_path: str,
                                  output_path: str | None = None) -> dict[str, Any]:
        """Feature-space change between two dates (not specific to forest loss).

        See :meth:`PrithviMultiTemporal.detect_change` for what the map means.
        """
        mt = PrithviMultiTemporal(self.model_name, self._prithvi.device)
        mt._prithvi = self._prithvi
        return mt.detect_change(before_path, after_path, output_path=output_path)


def build_regression_head(embed_dim: int) -> Any:
    """MLP ``(B, embed_dim) -> (B, 1)`` used by :meth:`PrithviTasks.biomass_estimation`."""
    import torch.nn as nn
    return nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))


# ── Fine-tuning API ───────────────────────────────────────────────────────────

def finetune_prithvi(
    model_name: str = "prithvi_eo_2_0",
    dataset: Any = None,
    task: str = "land_cover",
    num_classes: int = 10,
    epochs: int = 50,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    mixed_precision: bool = True,
    distributed: bool = False,
    output_dir: str = "./checkpoints/prithvi/",
    **kwargs,
) -> dict[str, Any]:
    """Assemble the pieces of a Prithvi fine-tuning job.  **This does not train.**

    Returns the segmentation model (backbone unfrozen), optimiser, cosine scheduler, mixed-precision
    and checkpoint managers, ready to be driven by ``GeoTrainer`` or your own loop.  After training,
    store the head with :func:`save_prithvi_head` and hand it to :class:`PrithviTasks`.

    ``dataset`` and ``distributed`` are accepted for backward compatibility but are not used here;
    a warning is logged if they are set.

    Recommended hyperparameters (Prithvi-EO-2.0 paper): AdamW, lr 5e-5, weight decay 0.01, 5 warm-up
    epochs, bf16, batch size 8.
    """
    try:
        import torch

        from pygeovision.training.checkpoint import CheckpointManager
        from pygeovision.training.mixed_precision import MixedPrecisionManager
    except ImportError:
        return {"error": "torch + pygeovision.training required"}

    if dataset is not None or distributed:
        logger.warning("finetune_prithvi: `dataset` and `distributed` are not used; it only builds the "
                       "training components.")

    prithvi = Prithvi(model_name, allow_random_init=kwargs.get("allow_random_init", False))
    prithvi.load()
    model = prithvi.build_segmentation_head(num_classes, freeze_backbone=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mp_mgr    = MixedPrecisionManager(precision="bf16" if mixed_precision else "fp32")
    ckpt_mgr  = CheckpointManager(output_dir, monitor="val_iou", mode="max")

    logger.info("Prithvi fine-tuning components: %s | task=%s | classes=%d | epochs=%d",
                model_name, task, num_classes, epochs)
    return {
        "model":        model,
        "optimizer":    optimizer,
        "scheduler":    scheduler,
        "mp_manager":   mp_mgr,
        "ckpt_manager": ckpt_mgr,
        "config":       prithvi.finetune_config(),
        "task":         task,
        "num_classes":  num_classes,
        "status":       "ready",
        "note":         "components only -- no training has been run",
    }


# ── Convenience ──────────────────────────────────────────────────────────────

def list_prithvi_models() -> list[str]:
    return list(PRITHVI_MODELS.keys())


def get_prithvi_info(model_name: str) -> dict:
    spec = PRITHVI_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown Prithvi model: '{model_name}'")
    return {**spec, "name": model_name,
            "band_order": "HLS: Blue, Green, Red, NIR, SWIR1, SWIR2",
            "sentinel2_mapping": SENTINEL2_TO_PRITHVI,
            "landsat_mapping":   LANDSAT_TO_PRITHVI}
