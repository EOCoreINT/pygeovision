"""
pygeovision.training.data
==========================
Geospatial training dataset utilities.

This module provides PyTorch Dataset classes for common geospatial AI
training tasks.  All datasets expect:

  * Pre-processed imagery (float32, [0,1]) — not raw DN.
  * Paired label rasters (uint8 or int64) in the same CRS and resolution
    as the input imagery.
  * No NaN values — run validate_prithvi_input / validate_sar_ai_input first.

Typical usage::

    from pygeovision.training.data import GeoSegDataset
    from torch.utils.data import DataLoader

    train_ds = GeoSegDataset('./data/train/images', './data/train/labels',
                              chip_size=256, augment=True)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
"""
from __future__ import annotations

import logging
import pathlib
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger("pygeovision.training.data")


# ── Lazy torch import ─────────────────────────────────────────────────────────
def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for training datasets. "
            "Install it with: pip install torch torchvision"
        )


def _rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("rasterio is required: pip install rasterio")


# ── GeoSegDataset ─────────────────────────────────────────────────────────────

class GeoSegDataset:
    """Geospatial segmentation dataset backed by GeoTIFF image/label pairs.

    Tiles each scene into ``(chip_size × chip_size)`` chips with optional
    random flipping and rotation augmentation.  Compatible with
    ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    image_dir : str | Path
        Directory of preprocessed imagery GeoTIFFs (float32, [0,1]).
        All ``.tif`` files are included.
    label_dir : str | Path
        Directory of label GeoTIFFs (uint8 or int64, one-hot integer class IDs).
        File stems must match those in ``image_dir``.
    chip_size : int
        Spatial dimension of each chip (square).  Default 256.
    augment : bool
        If True, apply random horizontal/vertical flips and 90° rotations
        at load time.  Should be True for training splits only.
    transform : callable | None
        Optional callable applied to the (image, label) chip tuple AFTER
        augmentation.  Receives numpy arrays; must return torch Tensors if
        used with DataLoader.
    max_chips_per_scene : int | None
        If set, limits chips extracted from each scene (avoids very large
        datasets from high-resolution scenes).
    overlap : int
        Pixel overlap between adjacent chips.  0 = no overlap.
    skip_nodata : bool
        Skip chips where more than 50% of image pixels are zero (nodata).

    Example::

        from pygeovision.training.data import GeoSegDataset
        from torch.utils.data import DataLoader

        train_ds = GeoSegDataset(
            './data/train/images', './data/train/labels',
            chip_size=256, augment=True, max_chips_per_scene=50,
        )
        print(f"Training chips: {len(train_ds)}")
        img, lbl = train_ds[0]
        print(f"Image: {img.shape}  Labels: {lbl.shape}")
        loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    """

    def __init__(
        self,
        image_dir: str | pathlib.Path,
        label_dir: str | pathlib.Path,
        *,
        chip_size: int = 256,
        augment: bool = False,
        transform: Optional[Callable] = None,
        max_chips_per_scene: Optional[int] = None,
        overlap: int = 0,
        skip_nodata: bool = True,
    ) -> None:
        self.image_dir = pathlib.Path(image_dir)
        self.label_dir = pathlib.Path(label_dir)
        self.chip_size = chip_size
        self.augment   = augment
        self.transform = transform
        self.max_chips = max_chips_per_scene
        self.overlap   = overlap
        self.skip_nodata = skip_nodata

        self._chips: list[dict] = []   # deferred: populated on first access
        self._loaded = False

    def _load(self) -> None:
        """Extract all chips from all image/label pairs (lazy)."""
        if self._loaded:
            return
        rio = _rasterio()
        stride = self.chip_size - self.overlap

        for img_path in sorted(self.image_dir.glob("*.tif")):
            lbl_path = self.label_dir / img_path.name
            if not lbl_path.exists():
                # Try .tiff
                lbl_path = self.label_dir / (img_path.stem + ".tiff")
            if not lbl_path.exists():
                logger.warning("No label found for %s — skipping", img_path.name)
                continue

            try:
                with rio.open(str(img_path)) as src:
                    img  = src.read().astype("float32")   # (C, H, W)
                with rio.open(str(lbl_path)) as src:
                    lbl  = src.read(1).astype("int64")    # (H, W)
            except Exception as exc:
                logger.warning("Could not read %s: %s", img_path.name, exc)
                continue

            C, H, W = img.shape
            scene_chips = 0

            for y in range(0, H - self.chip_size + 1, stride):
                for x in range(0, W - self.chip_size + 1, stride):
                    img_chip = img[:, y:y + self.chip_size, x:x + self.chip_size]
                    lbl_chip = lbl[   y:y + self.chip_size, x:x + self.chip_size]

                    if self.skip_nodata:
                        zero_frac = (img_chip == 0).all(axis=0).mean()
                        if zero_frac > 0.50:
                            continue

                    self._chips.append({
                        "img": img_chip,
                        "lbl": lbl_chip,
                    })
                    scene_chips += 1

                    if self.max_chips and scene_chips >= self.max_chips:
                        break
                if self.max_chips and scene_chips >= self.max_chips:
                    break

            logger.info("Loaded %d chips from %s", scene_chips, img_path.name)

        self._loaded = True
        logger.info("GeoSegDataset: %d total chips from %s",
                    len(self._chips), self.image_dir)

    def _augment(
        self, img: np.ndarray, lbl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random flip + rotation augmentation."""
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
            lbl = lbl[::-1, :].copy()
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()
        k = np.random.randint(4)
        if k:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k, axes=(0, 1)).copy()
        # Random Gaussian noise (±0.5% amplitude)
        img = np.clip(img + np.random.normal(0, 0.005, img.shape).astype("float32"),
                      0.0, 1.0)
        return img, lbl

    def __len__(self) -> int:
        if not self._loaded:
            self._load()
        return len(self._chips)

    def __getitem__(self, idx: int):
        if not self._loaded:
            self._load()

        chip = self._chips[idx]
        img  = chip["img"].copy()
        lbl  = chip["lbl"].copy()

        if self.augment:
            img, lbl = self._augment(img, lbl)

        if self.transform is not None:
            return self.transform(img, lbl)

        # Default: convert to torch Tensors (lazy import)
        torch = _torch()
        return (
            torch.from_numpy(img),
            torch.from_numpy(lbl),
        )


# ── Convenience factory ───────────────────────────────────────────────────────

def make_geo_loaders(
    image_dir: str | pathlib.Path,
    label_dir: str | pathlib.Path,
    *,
    chip_size: int = 256,
    batch_size: int = 8,
    split: float = 0.80,
    max_chips_per_scene: Optional[int] = 100,
    num_workers: int = 0,
) -> tuple:
    """Build train + val DataLoaders from a directory of image/label pairs.

    Convenience wrapper around ``GeoSegDataset``.

    Returns
    -------
    (train_loader, val_loader)
    """
    torch = _torch()
    from torch.utils.data import DataLoader, random_split

    full_ds = GeoSegDataset(
        image_dir, label_dir,
        chip_size=chip_size, augment=True,
        max_chips_per_scene=max_chips_per_scene,
    )

    n      = len(full_ds)
    n_tr   = int(n * split)
    n_va   = n - n_tr

    tr_ds, va_ds = random_split(
        full_ds, [n_tr, n_va],
        generator=torch.Generator().manual_seed(42),
    )
    # Disable augmentation for val split
    va_ds.dataset = GeoSegDataset(
        image_dir, label_dir,
        chip_size=chip_size, augment=False,
        max_chips_per_scene=max_chips_per_scene,
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False)

    logger.info(
        "GeoSeg loaders: %d train chips / %d val chips  batch_size=%d",
        n_tr, n_va, batch_size,
    )
    return tr_loader, va_loader
