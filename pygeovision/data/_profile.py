"""Small helpers for copying a rasterio profile into a new GeoTIFF."""
from __future__ import annotations

from typing import Any


def writable_profile(profile: dict[str, Any], **updates: Any) -> dict[str, Any]:
    """Copy ``profile``, apply ``updates`` and drop block sizes a striped file cannot take.

    ``src.profile`` always carries ``blockxsize``/``blockysize``. For a striped
    (untiled) source, writing those back makes GDAL emit
    ``CPLE_IllegalArg ... BLOCKXSIZE can only be used with TILED=YES`` on every
    write and ignore the options. Tiled sources keep their block layout.
    """
    out = dict(profile)
    out.update(updates)
    if not out.get("tiled"):
        out.pop("blockxsize", None)
        out.pop("blockysize", None)
    return out
