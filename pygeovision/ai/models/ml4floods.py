"""The ml4floods ``WF2_unetv2_bgriswirs`` flood-segmentation network, run inside PyGeoVision.

``WF2_unetv2_bgriswirs`` is a U-Net trained on WorldFloods v2 (Sentinel-2 images with flood masks) by
Portales-Julia, Mateo-Garcia, Purcell and Gomez-Chova (2023, *Scientific Reports* 13, 20316,
https://doi.org/10.1038/s41598-023-47595-7) and released by the ml4floods project
(https://github.com/spaceml-org/ml4floods, https://huggingface.co/isp-uv-es/ml4floods).

What this module does
---------------------
* defines the network (a plain U-Net, written here so that PyGeoVision does not need the ``ml4floods`` package; the
  layer names match the released checkpoint so that it loads with ``strict=True``),
* loads the released ``model.pt`` (and checks ``config.json`` when it is there) and refuses a checkpoint that is not
  this model,
* normalises, pads and tiles an image, and returns the two probability maps the network predicts
  (``cloud`` and ``water``),
* turns them into the classes ``0 invalid, 1 land, 2 water, 3 cloud, 4 flood trace``.

What it does **not** do
-----------------------
* **It does not ship the weights.** They are released under CC BY-NC 4.0 (non-commercial use, attribution) while
  PyGeoVision is MIT, so the weights are not part of the package. :func:`download_weights` fetches them on request and
  only after you confirm that your use is non-commercial (``accept_noncommercial=True``); you can also pass a folder you
  downloaded yourself.
* **It does not validate the model.** No accuracy is claimed by PyGeoVision for this network. The reference for its
  accuracy is the paper above, measured on its own test set. On imagery that differs from its training data (see
  :data:`INPUT_NOTE`) the accuracy is unknown.

Input
-----
Six Sentinel-2 bands in this order: B2 (blue), B3 (green), B4 (red), B8 (NIR), B11 (SWIR1), B12 (SWIR2), as **reflectance
x 10000** (the scale of the Sentinel-2 digital numbers, with the 1000 offset of processing baseline 04.00 and later
removed), at 10 m. A pixel whose six values are all zero (or not finite) is *invalid*.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "WF2_unetv2_bgriswirs"
HF_REPO = "isp-uv-es/ml4floods"
HF_SUBFOLDER = f"models/{MODEL_ID}"
#: Commit of the Hugging Face repository that holds ``models/WF2_unetv2_bgriswirs`` (config.json, model.pt). Pinned so
#: that a run can be repeated; pass ``revision="main"`` to :func:`download_weights` to follow the repository.
HF_REVISION = "2a6cf9a02f887179c25dc24a81bb94a5504bbb12"

CITATION = (
    "Portales-Julia, E., Mateo-Garcia, G., Purcell, C. & Gomez-Chova, L. (2023). Global flood extent segmentation in "
    "optical satellite images. Scientific Reports 13, 20316. doi:10.1038/s41598-023-47595-7."
)
LICENSE_NOTE = (
    "The ml4floods model weights and the WorldFloods data are released under the Creative Commons "
    "Attribution-NonCommercial 4.0 licence (CC BY-NC 4.0): you may use them for non-commercial purposes if you credit "
    "the authors. PyGeoVision itself is MIT-licensed and does not include them."
)
INPUT_NOTE = (
    "The network was trained and is documented on Sentinel-2 Level-1C top-of-atmosphere reflectance (x 10000). "
    "Level-2A surface reflectance, which most open archives serve, is not the training product: the two differ most in "
    "the blue band and under haze, and the effect on this model has not been measured."
)

#: Input channels, in the order the network expects them (ml4floods channel configuration ``bgriswirs``).
BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
_IDX = {b: i for i, b in enumerate(BANDS)}
#: (mean, standard deviation) of each band on the training set, reflectance x 10000, from the ml4floods source.
NORMALIZATION = np.array(
    [
        [3758.07467509, 2794.09579088],  # B2
        [3238.08247208, 2549.4940614],  # B3
        [3418.90147615, 2811.78109878],  # B4
        [3981.96268494, 2500.47885249],  # B8
        [2391.66101119, 1500.02533014],  # B11
        [1790.32497137, 1241.9817628],  # B12
    ],
    dtype=np.float32,
)
#: 0-based positions of the six bands in a 13-band Sentinel-2 stack (B1, B2, ..., B8, B8A, B9, B10, B11, B12).
S2_13_BAND_INDEXES = (1, 2, 3, 7, 11, 12)
#: Below this brightness (sqrt of the summed squares of B4, B3, B2) a cloud prediction is treated as thin cloud/haze.
BRIGHTNESS_THRESHOLD = 3500.0
CLASS_NAMES = {0: "invalid", 1: "land", 2: "water", 3: "cloud", 4: "flood trace"}
_UNET_DIVISOR = 8


class ML4FloodsError(RuntimeError):
    """The checkpoint is not the supported network, or its files are missing."""


class LicenseNotAccepted(PermissionError):
    """The weights were requested without confirming the licence terms."""


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def _double_conv(nn: Any, cin: int, cout: int) -> Any:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


def build_network(in_channels: int = len(BANDS), num_outputs: int = 2) -> Any:
    """The U-Net of the released model: 4 levels (64-128-256-512), bilinear up-sampling, 2 sigmoid outputs.

    Output channel 0 is the logit of *cloud*, channel 1 the logit of *water* (the two problems are solved
    independently, so a pixel can be both).
    """
    import torch
    from torch import nn

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dconv_down1 = _double_conv(nn, in_channels, 64)
            self.dconv_down2 = _double_conv(nn, 64, 128)
            self.dconv_down3 = _double_conv(nn, 128, 256)
            self.dconv_down4 = _double_conv(nn, 256, 512)
            self.maxpool = nn.MaxPool2d(2)
            self.dconv_up3 = _double_conv(nn, 256 + 512, 256)
            self.dconv_up2 = _double_conv(nn, 128 + 256, 128)
            self.dconv_up1 = _double_conv(nn, 128 + 64, 64)
            self.conv_last = nn.Conv2d(64, num_outputs, 1)

        @staticmethod
        def _up(x: Any) -> Any:
            return nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        def forward(self, x: Any) -> Any:
            c1 = self.dconv_down1(x)
            c2 = self.dconv_down2(self.maxpool(c1))
            c3 = self.dconv_down3(self.maxpool(c2))
            x = self.dconv_down4(self.maxpool(c3))
            x = self.dconv_up3(torch.cat([self._up(x), c3], dim=1))
            x = self.dconv_up2(torch.cat([self._up(x), c2], dim=1))
            x = self.dconv_up1(torch.cat([self._up(x), c1], dim=1))
            return self.conv_last(x)

    return UNet()


# ---------------------------------------------------------------------------
# Weights: where they come from and how they are loaded
# ---------------------------------------------------------------------------


def download_weights(
    dest: str | Path | None = None,
    *,
    accept_noncommercial: bool = False,
    revision: str = HF_REVISION,
) -> Path:
    """Download ``config.json`` and ``model.pt`` of the model from Hugging Face; returns the folder.

    The weights are CC BY-NC 4.0 (see :data:`LICENSE_NOTE`). Nothing is downloaded unless you pass
    ``accept_noncommercial=True``, which says that your use is non-commercial and that you will credit the authors
    (:data:`CITATION`). Needs the ``huggingface_hub`` package and network access to huggingface.co.
    """
    if not accept_noncommercial:
        raise LicenseNotAccepted(
            f"{LICENSE_NOTE} Pass accept_noncommercial=True to download them for non-commercial use, and cite: "
            f"{CITATION}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ML4FloodsError(
            "downloading the weights needs the huggingface_hub package (pip install huggingface_hub); "
            f"or download {HF_REPO}/{HF_SUBFOLDER} yourself and pass that folder as checkpoint="
        ) from exc
    dest_dir = Path(dest).expanduser() if dest else DEFAULT_CACHE
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "model.pt"):
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{HF_SUBFOLDER}/{filename}",
            revision=revision,
            local_dir=str(dest_dir),
        )
    folder = dest_dir / HF_SUBFOLDER
    if not (folder / "model.pt").is_file():
        raise ML4FloodsError(f"the download finished but {folder / 'model.pt'} does not exist")
    logger.info("ml4floods weights in %s (CC BY-NC 4.0; cite %s)", folder, CITATION)
    return folder


DEFAULT_CACHE = Path.home() / ".cache" / "pygeovision" / "ml4floods"


def ensure_weights(dest: str | Path | None = None) -> Path:
    """The folder with the weights, downloading them from Hugging Face on first use.

    Used by the ``ml4floods_flood_mapping`` pipeline when no ``checkpoint`` is given. A copy already in ``dest`` (default
    ``~/.cache/pygeovision/ml4floods``) is reused without network access. Running this means you accept the licence of the
    weights (CC BY-NC 4.0: non-commercial use, credit the authors), which is logged each time.
    """
    folder = Path(dest).expanduser() if dest else DEFAULT_CACHE
    folder = folder / HF_SUBFOLDER
    weights = folder / "model.pt"
    if (
        weights.is_file()
        and weights.stat().st_size > 1_000_000
        and not _looks_like_lfs_pointer(weights)
    ):
        return folder
    logger.warning(
        "Downloading %s from Hugging Face (%s). %s Cite: %s",
        MODEL_ID,
        HF_REPO,
        LICENSE_NOTE,
        CITATION,
    )
    return download_weights(dest, accept_noncommercial=True)


def _looks_like_lfs_pointer(path: Path) -> bool:
    """A Git LFS pointer is a ~130-byte text file that stands in for the real weights after a clone without LFS."""
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(b"version https://git-lfs")
    except OSError:
        return False


def _resolve(path: str | Path) -> tuple[Path, Path | None]:
    """``(model.pt, config.json or None)`` of a checkpoint given as a folder or as the ``model.pt`` file.

    A folder is searched for ``model.pt``, then for the Hugging Face layout ``models/WF2_unetv2_bgriswirs/model.pt``
    below it, then for a single ``*.pt`` / ``*.pth`` / ``*.ckpt`` file."""
    p = Path(path).expanduser()
    if p.is_dir():
        weights = p / "model.pt"
        if not weights.is_file():
            nested = (
                p / HF_SUBFOLDER / "model.pt"
            )  # the folder that holds the models/ tree of the repository
            if nested.is_file():
                weights = nested
            else:
                found = sorted(
                    f for ext in ("pt", "pth", "ckpt") for f in p.glob(f"*.{ext}") if f.is_file()
                )
                if len(found) == 1:
                    weights = found[0]
        config = weights.parent / "config.json"
    else:
        weights, config = p, p.parent / "config.json"
    if not weights.is_file():
        here = p.resolve()
        listing = ""
        if p.is_dir():
            names = sorted(f.name for f in p.iterdir())[:12]
            listing = f" It contains: {', '.join(names) if names else 'nothing'}."
        elif not p.exists():
            listing = f" That path does not exist (relative paths are taken from the current directory, {Path.cwd()})."
        raise ML4FloodsError(
            f"no model.pt at {here}.{listing} Pass the folder that holds model.pt (and config.json) of {MODEL_ID}, or get "
            f"it with pygeovision.ai.models.ml4floods.download_weights(accept_noncommercial=True), which returns that folder."
        )
    if _looks_like_lfs_pointer(weights):
        raise ML4FloodsError(
            f"{weights} is a Git LFS pointer, not the weights (the repository was cloned without LFS). Run `git lfs pull` "
            f"in it, or fetch the file with pygeovision.ai.models.ml4floods.download_weights(accept_noncommercial=True)."
        )
    return weights, (config if config.is_file() else None)


def check_config(config_path: Path) -> dict[str, Any]:
    """Refuse a ``config.json`` that describes another network than the one this module implements."""
    cfg = json.loads(Path(config_path).read_text())
    hp = (cfg.get("model_params") or {}).get("hyperparameters") or {}
    mp = cfg.get("model_params") or {}
    data = cfg.get("data_params") or {}
    channel_cfg = data.get("channel_configuration") or hp.get("channel_configuration")
    problems = []
    if hp.get("model_type", "unet") != "unet":
        problems.append(f"model_type={hp.get('model_type')!r} (expected 'unet')")
    if mp.get("model_version", "v2") != "v2":
        problems.append(f"model_version={mp.get('model_version')!r} (expected 'v2')")
    if int(hp.get("num_classes", 2)) != 2:
        problems.append(f"num_classes={hp.get('num_classes')} (expected 2)")
    if channel_cfg not in (None, "bgriswirs"):
        problems.append(f"channel_configuration={channel_cfg!r} (expected 'bgriswirs')")
    if data.get("add_mndwi_input"):
        problems.append("add_mndwi_input is on (a 7th input channel is not supported)")
    if problems:
        raise ML4FloodsError(
            "this checkpoint is not the supported WF2_unetv2_bgriswirs network: "
            + "; ".join(problems)
        )
    return cfg


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _state_dict(raw: Any) -> dict[str, Any]:
    state = raw
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ML4FloodsError("model.pt does not contain a state dict of tensors")
    # the checkpoint is the state dict of a Lightning module whose network is the attribute ``network``
    return {(k[len("network.") :] if k.startswith("network.") else k): v for k, v in state.items()}


@dataclass
class LoadedModel:
    """The network, ready to predict, and what is needed to say which weights produced a result."""

    network: Any
    device: str
    weights_path: Path
    weights_sha256: str
    config: dict[str, Any] | None


def load_model(checkpoint: str | Path, device: str = "auto") -> LoadedModel:
    """Load the released weights strictly. ``checkpoint`` is the folder with ``model.pt`` (and ``config.json``), or the
    ``model.pt`` file. Raises :class:`ML4FloodsError` if it is not this network."""
    import torch

    weights, config_path = _resolve(checkpoint)
    cfg = check_config(config_path) if config_path else None
    raw = torch.load(weights, map_location="cpu", weights_only=True)
    state = _state_dict(raw)
    first = state.get("dconv_down1.0.weight")
    if first is None or tuple(first.shape[:2]) != (64, len(BANDS)):
        got = tuple(first.shape) if first is not None else "missing"
        raise ML4FloodsError(
            f"model.pt does not match the {len(BANDS)}-band U-Net (first layer {got}, expected (64, {len(BANDS)}, 3, 3)); "
            f"it is not {MODEL_ID}"
        )
    net = build_network()
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ML4FloodsError(
            f"model.pt does not fit the network of {MODEL_ID}: {str(exc)[:300]}"
        ) from exc
    dev = _device(device)
    net.to(dev).eval()
    return LoadedModel(net, dev, weights, sha256_of(weights), cfg)


def _device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def invalid_mask(image: np.ndarray) -> np.ndarray:
    """Pixels with no observation: all six bands zero, or any band not finite."""
    return np.all(np.nan_to_num(image, nan=0.0) == 0, axis=0) | ~np.all(np.isfinite(image), axis=0)


def select_bands(stack: np.ndarray) -> np.ndarray:
    """The six model bands from a 6-band (already selected) or a 13-band Sentinel-2 stack (B1..B12 with B8A, B9, B10)."""
    if stack.shape[0] == len(BANDS):
        return stack
    if stack.shape[0] == 13:
        return stack[list(S2_13_BAND_INDEXES)]
    raise ValueError(
        f"expected 6 bands ({', '.join(BANDS)}) or all 13 Sentinel-2 bands, got {stack.shape[0]}"
    )


def _normalize(image: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(image.astype(np.float32), nan=0.0)
    mean = NORMALIZATION[:, 0][:, None, None]
    std = NORMALIZATION[:, 1][:, None, None]
    return (x - mean) / (std + 1e-6)


def _predict_padded(net: Any, x: Any, device: str) -> Any:
    """Network output for ``x`` (1, C, H, W), padding to a multiple of 8 (the U-Net needs it) and cropping back."""
    import torch
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ph, pw = (-h) % _UNET_DIVISOR, (-w) % _UNET_DIVISOR
    if ph or pw:
        mode = "reflect" if (ph < h and pw < w) else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    with torch.no_grad():
        out = net(x.to(device))
    return out[..., :h, :w].float().cpu()


def predict_probabilities(
    model: LoadedModel,
    image: np.ndarray,
    *,
    tile_size: int = 512,
    context: int = 32,
) -> np.ndarray:
    """Probabilities ``(2, H, W)``: channel 0 *cloud*, channel 1 *water*; NaN on invalid pixels.

    ``image`` is ``(6, H, W)`` reflectance x 10000 (see the module docstring). Large images are cut into
    ``tile_size`` squares that are each read with ``context`` extra pixels on every side, which are dropped from the
    result, so tile borders do not show.
    """
    import torch

    if image.ndim != 3 or image.shape[0] != len(BANDS):
        raise ValueError(f"image must be ({len(BANDS)}, H, W), got {image.shape}")
    if tile_size < _UNET_DIVISOR or context < 0:
        raise ValueError("tile_size must be at least 8 and context cannot be negative")
    bad = invalid_mask(image)
    norm = torch.from_numpy(_normalize(image))
    _, height, width = norm.shape
    probs = np.zeros((2, height, width), dtype=np.float32)
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            i1, j1 = min(i + tile_size, height), min(j + tile_size, width)
            pi0, pj0 = max(i - context, 0), max(j - context, 0)
            pi1, pj1 = min(i1 + context, height), min(j1 + context, width)
            logits = _predict_padded(
                model.network, norm[:, pi0:pi1, pj0:pj1].unsqueeze(0), model.device
            )[0]
            sub = torch.sigmoid(logits)[
                :, i - pi0 : i - pi0 + (i1 - i), j - pj0 : j - pj0 + (j1 - j)
            ]
            probs[:, i:i1, j:j1] = sub.numpy()
    probs[:, bad] = np.nan
    return probs


def classify(
    image: np.ndarray,
    probabilities: np.ndarray,
    *,
    th_water: float = 0.5,
    th_cloud: float = 0.5,
    th_brightness: float = BRIGHTNESS_THRESHOLD,
    mask_clouds: bool = True,
    distinguish_flood_traces: bool = False,
) -> np.ndarray:
    """``(H, W)`` uint8 classes: 0 invalid, 1 land, 2 water, 3 cloud (and 4 flood trace when asked for).

    Water is ``P(water) > th_water``. Cloud is ``P(cloud) > th_cloud``; with ``mask_clouds`` only the *bright* cloud
    counts (sqrt(B4^2 + B3^2 + B2^2) above ``th_brightness``), as the thin cloud and haze that the network also
    flags still let the ground be seen. Cloud overrides water. With ``distinguish_flood_traces`` water with
    MNDWI < 0 becomes class 4 (traces of past flooding, e.g. wet soil) instead of water.
    """
    for name, v in (("th_water", th_water), ("th_cloud", th_cloud)):
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    bad = invalid_mask(image)
    p_cloud, p_water = probabilities[0], probabilities[1]
    out = np.ones(bad.shape, dtype=np.uint8)
    water = np.nan_to_num(p_water, nan=0.0) > float(th_water)
    out[water] = 2
    cloud = np.nan_to_num(p_cloud, nan=0.0) > float(th_cloud)
    if mask_clouds:
        rgb = image[[_IDX["B4"], _IDX["B3"], _IDX["B2"]]].astype(np.float32)
        bright = np.sqrt(np.sum(np.nan_to_num(rgb) ** 2, axis=0)) > float(th_brightness)
        cloud &= bright
    out[cloud] = 3
    if distinguish_flood_traces:
        green, swir = image[_IDX["B3"]].astype(np.float32), image[_IDX["B11"]].astype(np.float32)
        mndwi = (green - swir) / (green + swir + 1e-6)
        out[(out == 2) & (mndwi < 0)] = 4
    out[bad] = 0
    return out


def predict_scene(
    model: LoadedModel,
    image: np.ndarray,
    *,
    tile_size: int = 512,
    th_water: float = 0.5,
    th_cloud: float = 0.5,
    th_brightness: float = BRIGHTNESS_THRESHOLD,
    mask_clouds: bool = True,
    distinguish_flood_traces: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``(classes, probabilities)`` for a ``(6, H, W)`` or ``(13, H, W)`` image."""
    img = select_bands(image)
    probs = predict_probabilities(model, img, tile_size=tile_size)
    classes = classify(
        img,
        probs,
        th_water=th_water,
        th_cloud=th_cloud,
        th_brightness=th_brightness,
        mask_clouds=mask_clouds,
        distinguish_flood_traces=distinguish_flood_traces,
    )
    return classes, probs


def predict_geotiff(
    image_path: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    tile_size: int = 512,
    th_water: float = 0.5,
    th_cloud: float = 0.5,
    th_brightness: float = BRIGHTNESS_THRESHOLD,
    mask_clouds: bool = True,
    distinguish_flood_traces: bool = False,
) -> dict[str, Path]:
    """Run the model on a Sentinel-2 GeoTIFF of 6 bands (B2, B3, B4, B8, B11, B12) or 13 bands, reflectance x 10000
    (for the faithful input, Level-1C top-of-atmosphere, e.g. exported from Google Earth Engine's
    ``COPERNICUS/S2_HARMONIZED``). Writes ``ml4floods_classes.tif`` (uint8 classes) and ``ml4floods_probabilities.tif``
    (float32: cloud, water) next to the image's grid. Returns the two paths."""
    import rasterio

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, device)
    with rasterio.open(image_path) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        profile = src.profile
    classes, probs = predict_scene(
        model,
        data,
        tile_size=tile_size,
        th_water=th_water,
        th_cloud=th_cloud,
        th_brightness=th_brightness,
        mask_clouds=mask_clouds,
        distinguish_flood_traces=distinguish_flood_traces,
    )
    cprof = {**profile, "count": 1, "dtype": "uint8", "nodata": 0, "compress": "lzw"}
    pprof = {**profile, "count": 2, "dtype": "float32", "nodata": float("nan"), "compress": "lzw"}
    paths = {
        "classes": out / "ml4floods_classes.tif",
        "probabilities": out / "ml4floods_probabilities.tif",
    }
    with rasterio.open(paths["classes"], "w", **cprof) as dst:
        dst.write(classes[None])
        dst.update_tags(
            classes=", ".join(f"{k}={v}" for k, v in CLASS_NAMES.items()),
            model=MODEL_ID,
            license="CC BY-NC 4.0 (weights)",
        )
    with rasterio.open(paths["probabilities"], "w", **pprof) as dst:
        dst.write(probs)
        dst.set_band_description(1, "P(cloud)")
        dst.set_band_description(2, "P(water)")
    return paths


__all__ = [
    "BANDS",
    "CITATION",
    "CLASS_NAMES",
    "DEFAULT_CACHE",
    "HF_REVISION",
    "INPUT_NOTE",
    "LICENSE_NOTE",
    "MODEL_ID",
    "LicenseNotAccepted",
    "LoadedModel",
    "ML4FloodsError",
    "build_network",
    "classify",
    "download_weights",
    "ensure_weights",
    "load_model",
    "predict_geotiff",
    "predict_probabilities",
    "predict_scene",
]