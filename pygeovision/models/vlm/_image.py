"""Shared image loading for the vision-language models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def list_images(image_dir: str | Path, suffixes: tuple[str, ...] = IMAGE_SUFFIXES) -> list[Path]:
    """Image files under ``image_dir`` (recursive, case-insensitive suffix match), sorted by path."""
    return sorted(p for p in Path(image_dir).rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def load_rgb_pil(image_path: str, stretch: str = "joint", bands: tuple[int, ...] | None = None) -> Any:
    """Open a raster file as an 8-bit RGB PIL image for a model that expects a picture.

    ``bands`` (1-based file band numbers, as R, G, B) says which bands to show; the default is the first three.
    Sentinel-2 files written B2, B3, B4, B8 need ``bands=(3, 2, 1)`` to look natural. One band is repeated to
    grey and two bands become R, G, R. A band number the file does not have raises ``ValueError``.
    Nodata and NaN pixels are ignored when choosing the stretch and shown as black. ``stretch="joint"``
    scales all bands between one shared pair of 2nd/98th percentiles, which keeps the colour balance
    a model relies on; ``"per_band"`` scales each band on its own (higher contrast, but colours shift).
    """
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(image_path) as src:
        if bands is None:
            chosen = list(range(1, min(src.count, 3) + 1))
        else:
            chosen = [int(b) for b in bands]
            if not 1 <= len(chosen) <= 3 or min(chosen) < 1 or max(chosen) > src.count:
                raise ValueError(f"bands={tuple(bands)} does not fit {image_path}, which has {src.count} band(s); "
                                 "give 1 to 3 band numbers between 1 and the band count")
        data = src.read(chosen, masked=True).astype("float64")
    mask = np.ma.getmaskarray(data) | ~np.isfinite(data.filled(0))
    vals = data.filled(0.0)
    out = np.zeros(vals.shape, dtype="float64")
    groups = [list(range(vals.shape[0]))] if stretch == "joint" else [[b] for b in range(vals.shape[0])]
    for g in groups:
        valid = vals[g][~mask[g]]
        if valid.size == 0:
            continue
        p2, p98 = np.percentile(valid, (2, 98))
        out[g] = np.clip((np.where(mask[g], p2, vals[g]) - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
    out[mask] = 0
    if out.shape[0] == 1:
        out = np.repeat(out, 3, axis=0)
    elif out.shape[0] == 2:
        out = np.stack([out[0], out[1], out[0]], axis=0)
    return Image.fromarray(out[:3].transpose(1, 2, 0).astype(np.uint8))
