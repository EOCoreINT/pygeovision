"""GeoSegDataset must skip (and say so) a label that is not on the image's grid."""
import logging

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("torch")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.training.data import GeoSegDataset  # noqa: E402


def _write(path, arr, transform):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1], count=arr.shape[0],
                       dtype=str(arr.dtype), crs="EPSG:32630", transform=transform) as d:
        d.write(arr)


def _pair(root, name, label_transform, label_size=64):
    (root / "images").mkdir(exist_ok=True, parents=True)
    (root / "labels").mkdir(exist_ok=True, parents=True)
    _write(root / "images" / f"{name}.tif", np.random.rand(3, 64, 64).astype("float32") + 0.1,
           from_origin(500000, 700000, 10, 10))
    _write(root / "labels" / f"{name}.tif", np.zeros((1, label_size, label_size), "uint8"), label_transform)


def test_aligned_pair_is_used_and_shifted_or_resized_pair_is_skipped(tmp_path, caplog):
    same = from_origin(500000, 700000, 10, 10)
    _pair(tmp_path, "ok", same)
    _pair(tmp_path, "shifted", from_origin(500100, 700000, 10, 10))
    _pair(tmp_path, "smaller", same, label_size=48)
    ds = GeoSegDataset(tmp_path / "images", tmp_path / "labels", chip_size=32)
    with caplog.at_level(logging.WARNING):
        n = len(ds)
    assert n == 4                                   # only the aligned scene: 2 x 2 chips
    assert "shifted.tif" in caplog.text and "smaller.tif" in caplog.text
